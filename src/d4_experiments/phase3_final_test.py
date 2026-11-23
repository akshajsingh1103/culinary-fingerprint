"""Final Held-Out Test Set Evaluation (Run ONCE on the 3,456 held-out test recipes).

Pre-registered models ONLY (do not add or remove):
 - M0: Original Fingerprint Scorer (existing D4 result from outputs/d4_scorer/training_artifacts, do not retrain)
 - M1: Naive Bayes (trained on all 13,823 training recipes)
 - M4: Full Stack — NB + All Network Features (25 features)
 - M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)

Procedure:
1. Train each on all 13,823 training recipes, using the existing training-only D2 backbone
   (`outputs/d4_scorer/training_artifacts/backbone_<cuisine>.csv` and `nodes_train.csv`) and D3 definitions
   (`src/analysis.py` & `src/graph_utils.py`).
2. Build training meta-features for the 13,823 training recipes via 3-fold inner cross-fitting
   (`StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` on the 13,823 training recipes ONLY,
   rebuilding normalization, vocabulary, D2 backbone, D3 hubs/communities/bridges, NB, and pair log-odds
   on each 2/3 inner-training split without the held-out 1/3 inner fold).
3. Choose meta-model L2 regularization C by inner 3-fold CV on the 13,823 cross-fitted training features ONLY,
   using Macro-F1 as primary metric (ties broken by accuracy, then smaller C), with StandardScaler fitted
   strictly on training data.
4. Fit final M4 and M5 meta-models on all 13,823 standardized cross-fitted training rows.
5. Compute test features for the 3,456 held-out test recipes using the full-training extractor
   (backed by `outputs/d4_scorer/training_artifacts`) and evaluate M0, M1, M4, M5 exactly ONCE on the 3,456 test recipes.
6. Save results to `outputs/d4_experiments/final_test_results.csv`, detailed JSON to
   `outputs/d4_experiments/final_test_summary.json`, and full report to `outputs/d4_experiments/final_test_report.md`.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from src.analysis import compute_bridges
from src.build_networks import assert_training_only
from src.common import (
    ROOT,
    get_logger,
    load_config,
    load_raw_dataset,
    read_json,
    rpath,
    set_global_seed,
    sha256_obj,
    write_json,
)
from src.d4_experiments.phase1_diagnostics import (
    OUT_DIR,
    SEED,
    bootstrap_ci,
    mcnemar_test,
    normalize_partition_records,
)
from src.d4_experiments.phase2_experiments import (
    build_pair_csr,
    compute_log_odds_weights,
)
from src.d4_experiments.phase3_scorer_v2 import (
    FEATURE_NAMES,
    INNER_CURVEBALL_N_RAND,
    INNER_N_SPLITS,
    META_C_GRID,
    PAIR_LOG_ODDS_EPS,
    PHASE3_CACHE_DIR,
    PHASE3_DIR,
    PartitionNetworkFeatureExtractor,
    compute_full_metrics_row,
    get_feature_subsets,
    tune_and_fit_standardized_meta_logreg,
)
from src.evaluate import BinaryMultinomialNB
from src.fingerprint import DEFAULT_ARTIFACTS, FingerprintScorer
from src.graph_utils import backbone_graph, louvain, rank_desc, strengths
from src.normalize import normalizer_fingerprint

log = get_logger("d4_final_test")

FINAL_RESULTS_CSV = OUT_DIR / "final_test_results.csv"
FINAL_SUMMARY_JSON = OUT_DIR / "final_test_summary.json"
FINAL_REPORT_MD = OUT_DIR / "final_test_report.md"


def load_train_and_test_splits_for_final_eval(cfg: dict) -> tuple[list[dict], list[dict], set[int], set[int], str]:
    """Load raw records and partition into the 13,823 training records and 3,456 held-out test records
    preserving the exact recipe ID order of `data/processed/split.json`.
    """
    split_file = rpath(cfg["data"]["processed_dir"]) / "split.json"
    split = read_json(split_file)
    train_id_list = [int(i) for i in split["train_ids"]]
    test_id_list = [int(i) for i in split["test_ids"]]
    train_ids = set(train_id_list)
    test_ids = set(test_id_list)

    assert len(train_ids) == 13823
    assert len(test_ids) == 3456
    assert len(train_ids & test_ids) == 0

    records, ds_report = load_raw_dataset(cfg)
    cuisines = set(cfg["cuisines"])
    by_id = {
        int(r["id"]): {"id": int(r["id"]), "cuisine": r["cuisine"], "ingredients": list(r["ingredients"])}
        for r in records
        if r.get("cuisine") in cuisines
    }
    train_raw = [by_id[i] for i in train_id_list]
    test_raw = [by_id[i] for i in test_id_list]

    assert len(train_raw) == 13823
    assert len(test_raw) == 3456
    assert_training_only(train_raw, train_ids, test_ids)
    return train_raw, test_raw, train_ids, test_ids, ds_report["sha256"]


def build_full_training_extractor_from_existing_d4_artifacts(
    cfg: dict,
    train_recipes: list[dict],
    train_ids: set[int],
    test_ids: set[int],
    dataset_sha: str,
    artifacts_dir: Path = DEFAULT_ARTIFACTS,
) -> PartitionNetworkFeatureExtractor:
    """Instantiate PartitionNetworkFeatureExtractor on all 13,823 training recipes using the EXISTING
    training-only D2 backbone (`outputs/d4_scorer/training_artifacts/backbone_<c>.csv` and `nodes_train.csv`)
    and D3 definitions (`src/analysis.py`, `src/graph_utils.py`), without retraining or modifying D4 artifacts.
    """
    assert_training_only(train_recipes, train_ids, test_ids)
    cuisines = list(cfg["cuisines"])
    top_n_hubs = int(cfg["report"]["top_n_hubs"])
    top_n_bridges = int(cfg["report"]["top_n_bridges"])
    comm_seed = int(cfg["community"]["seed"])
    comm_res = float(cfg["community"]["resolution"])

    meta = read_json(artifacts_dir / "training_metadata.json")
    assert meta["n_train_recipes"] == 13823
    assert meta["n_test_recipes_excluded"] == 3456

    nodes_all = pd.read_csv(artifacts_dir / "nodes_train.csv")
    bbs = {c: pd.read_csv(artifacts_dir / f"backbone_{c}.csv") for c in cuisines}

    ext = PartitionNetworkFeatureExtractor(
        cfg_base=cfg,
        cuisines=cuisines,
        n_randomizations=1000,
        cache_dir=PHASE3_CACHE_DIR,
        dataset_sha=dataset_sha,
        eps=PAIR_LOG_ODDS_EPS,
    )
    ext.backbones_ = bbs
    ext.class_counts_ = dict(meta["train_class_counts"])

    strength_by_c: dict[str, dict[str, float]] = {}
    prev_by_c: dict[str, dict[str, float]] = {}

    for c in cuisines:
        bb_df = bbs[c]
        nodes_df = nodes_all[nodes_all["cuisine"] == c].reset_index(drop=True)
        G = backbone_graph(bb_df, nodes_df)
        ext.backbone_vocab_[c] = set(G.nodes)

        s = strengths(G)
        rs = rank_desc(s)
        ext.hubs_top10_[c] = {k for k, r in rs.items() if r <= top_n_hubs}
        strength_by_c[c] = s
        prev_by_c[c] = dict(zip(nodes_df["ingredient"], nodes_df["prevalence"]))

        part = louvain(G, seed=comm_seed, resolution=comm_res)
        ext.communities_part_[c] = part
        ext.n_communities_[c] = (max(part.values()) + 1) if part else 0

    bridges_df, bridge_cand_counts = compute_bridges(
        strength_by_c, prev_by_c, cuisines, top_n=top_n_bridges
    )
    ext.bridges_top5_union_ = {c: set() for c in cuisines}
    for _, row in bridges_df.iterrows():
        c_a, c_b = str(row["cuisine_pair"]).split("-")
        ing = str(row["ingredient"])
        ext.bridges_top5_union_[c_a].add(ing)
        ext.bridges_top5_union_[c_b].add(ing)

    ext.d3_summary_ = {
        "backbone_nodes": {c: len(ext.backbone_vocab_[c]) for c in cuisines},
        "backbone_edges": {c: len(ext.backbones_[c]) for c in cuisines},
        "n_communities": dict(ext.n_communities_),
        "hubs_top10": {c: sorted(ext.hubs_top10_[c]) for c in cuisines},
        "bridges_per_cuisine": {c: sorted(ext.bridges_top5_union_[c]) for c in cuisines},
        "bridge_table_rows": bridges_df.to_dict(orient="records"),
        "bridge_candidate_counts": bridge_cand_counts,
    }

    # Fit BinaryMultinomialNB on all 13,823 training recipes
    ext.nb_ = BinaryMultinomialNB(
        cuisines,
        alpha=float(cfg["naive_bayes"]["alpha"]),
        fit_prior=bool(cfg["naive_bayes"]["fit_prior"]),
    ).fit(train_recipes, train_ids, test_ids)

    # Fit Backbone Pair Log-Odds weights on all 13,823 training recipes using the union of D4 training backbone pairs
    bb_pair_set: set[tuple[str, str]] = set()
    for c in cuisines:
        df = ext.backbones_[c]
        for a, b in zip(df["ingredient_a"], df["ingredient_b"]):
            pair = (a, b) if a < b else (b, a)
            bb_pair_set.add(pair)
    ext.backbone_pairs_ = sorted(bb_pair_set)

    X_tr_pairs = build_pair_csr(train_recipes, ext.backbone_pairs_)
    y_tr_idx = np.asarray([ext.c_to_idx[r["cuisine"]] for r in train_recipes], dtype=np.int64)
    ext.W_pair_log_odds_ = compute_log_odds_weights(
        X_tr_pairs, y_tr_idx, len(cuisines), ext.eps
    )
    return ext


def run_final_held_out_test_evaluation() -> dict:
    """Execute the one-time final evaluation of M0, M1, M4, M5 on the 3,456 held-out test recipes."""
    set_global_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    PHASE3_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    cuisines = list(cfg["cuisines"])
    c_to_idx = {c: i for i, c in enumerate(cuisines)}
    t0_all = time.perf_counter()

    train_raw, test_raw, train_ids, test_ids, dataset_sha = load_train_and_test_splits_for_final_eval(cfg)

    # Normalize full training and test partitions independently
    train_recipes = normalize_partition_records(train_raw)
    test_recipes = normalize_partition_records(test_raw)

    y_train_str = [r["cuisine"] for r in train_recipes]
    y_test_str = [r["cuisine"] for r in test_recipes]
    y_train_idx = np.asarray([c_to_idx[c] for c in y_train_str], dtype=np.int64)

    # =========================================================================
    # STEP 1: 3-Fold Inner Cross-Fitting on the 13,823 Training Recipes ONLY
    # =========================================================================
    skf_inner = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=SEED)
    inner_splits = list(skf_inner.split(train_recipes, y_train_idx))

    X_train_crossfit = np.zeros((len(train_recipes), len(FEATURE_NAMES)), dtype=np.float64)
    covered_train = np.zeros(len(train_recipes), dtype=bool)
    inner_crossfit_meta = []

    for inner_m, (itr_idx, ival_idx) in enumerate(inner_splits):
        t_in0 = time.perf_counter()
        itr_raw = [train_raw[i] for i in itr_idx]
        ival_raw = [train_raw[i] for i in ival_idx]
        itr_ids = {r["id"] for r in itr_raw}
        ival_ids = {r["id"] for r in ival_raw}
        forbidden_inner = ival_ids | test_ids

        assert len(itr_ids & ival_ids) == 0
        assert len(itr_ids & test_ids) == 0
        assert len(ival_ids & test_ids) == 0
        assert_training_only(itr_raw, itr_ids, forbidden_inner)

        itr_recipes = normalize_partition_records(itr_raw)
        ival_recipes = normalize_partition_records(ival_raw)

        ext_inner = PartitionNetworkFeatureExtractor(
            cfg_base=cfg,
            cuisines=cuisines,
            n_randomizations=INNER_CURVEBALL_N_RAND,
            cache_dir=PHASE3_CACHE_DIR,
            dataset_sha=dataset_sha,
            eps=PAIR_LOG_ODDS_EPS,
        ).fit(itr_recipes, itr_ids, forbidden_inner)

        X_train_crossfit[ival_idx, :] = ext_inner.transform(ival_recipes)
        covered_train[ival_idx] = True

        inner_crossfit_meta.append({
            "inner_fold": inner_m,
            "n_inner_train": len(itr_recipes),
            "n_inner_val": len(ival_recipes),
            "n_backbone_pairs": len(ext_inner.backbone_pairs_),
            "n_communities": ext_inner.n_communities_,
            "runtime_seconds": round(time.perf_counter() - t_in0, 3),
        })

    assert np.all(covered_train), "Every training recipe must receive out-of-fold cross-fitted features"
    assert not np.any(np.isnan(X_train_crossfit)), "NaN detected in X_train_crossfit"

    # =========================================================================
    # STEP 2: Tune Meta-Model C on Cross-Fitted Training Matrix via Inner CV
    #         (Macro-F1 Primary, Accuracy Secondary) and Fit M4 & M5 on All 13,823 Training Rows
    # =========================================================================
    subsets = get_feature_subsets(cuisines)
    cols_m4 = subsets["M4_full_stack"]
    cols_m5 = subsets["M5_network_only_v2"]

    scaler_m4, clf_m4, best_c_m4, curve_m4 = tune_and_fit_standardized_meta_logreg(
        X_train_crossfit, y_train_idx, cols_m4, inner_splits, c_grid=META_C_GRID, seed=SEED
    )
    scaler_m5, clf_m5, best_c_m5, curve_m5 = tune_and_fit_standardized_meta_logreg(
        X_train_crossfit, y_train_idx, cols_m5, inner_splits, c_grid=META_C_GRID, seed=SEED
    )

    # =========================================================================
    # STEP 3: Load Existing D4 Training Artifacts & Fit Full-Training Extractor on 13,823 Recipes
    # =========================================================================
    ext_full_train = build_full_training_extractor_from_existing_d4_artifacts(
        cfg, train_recipes, train_ids, test_ids, dataset_sha, DEFAULT_ARTIFACTS
    )
    X_test_features = ext_full_train.transform(test_recipes)
    assert X_test_features.shape == (3456, 25)
    assert not np.any(np.isnan(X_test_features))

    # =========================================================================
    # STEP 4: Evaluate Pre-Registered Models (M0, M1, M4, M5) ONCE on the 3,456 Test Recipes
    # =========================================================================
    # M0: Original Fingerprint Scorer (existing D4 result from outputs/d4_scorer/training_artifacts, not retrained)
    scorer_m0 = FingerprintScorer.from_artifacts(DEFAULT_ARTIFACTS, normalized=False)
    preds_m0, fb_m0 = [], []
    for r in test_recipes:
        p0, f0 = scorer_m0.predict_with_fallback(r["ingredients"])
        preds_m0.append(p0)
        fb_m0.append(f0)
    m0_zfb = sum(x == "zero_score" for x in fb_m0)
    m0_tfb = sum(x == "tied_score" for x in fb_m0)

    # M1: Naive Bayes (trained on all 13,823 training recipes)
    preds_m1, m1_zfb, m1_tfb = ext_full_train.nb_.predict_with_fallback(test_recipes)

    # Verify M0 and M1 match existing outputs/d4_scorer/results.csv exactly
    d4_orig_df = pd.read_csv(ROOT / "outputs" / "d4_scorer" / "results.csv").set_index("model")
    row_m0, cm_m0 = compute_full_metrics_row(
        "M0: Original Fingerprint Scorer", y_test_str, preds_m0, cuisines, m0_zfb, m0_tfb
    )
    row_m1, cm_m1 = compute_full_metrics_row(
        "M1: Naive Bayes", y_test_str, preds_m1, cuisines, m1_zfb, m1_tfb
    )
    assert math.isclose(row_m0["accuracy"], float(d4_orig_df.loc["Fingerprint Scorer", "accuracy"]), rel_tol=1e-12)
    assert math.isclose(row_m0["macro_f1"], float(d4_orig_df.loc["Fingerprint Scorer", "macro_f1"]), rel_tol=1e-12)
    assert math.isclose(row_m1["accuracy"], float(d4_orig_df.loc["Naive Bayes", "accuracy"]), rel_tol=1e-12)
    assert math.isclose(row_m1["macro_f1"], float(d4_orig_df.loc["Naive Bayes", "macro_f1"]), rel_tol=1e-12)

    # M4: Full Stack (NB + All Network Features, 25 features)
    X_test_m4_std = scaler_m4.transform(X_test_features[:, cols_m4])
    preds_m4 = [cuisines[int(i)] for i in clf_m4.predict(X_test_m4_std)]
    row_m4, cm_m4 = compute_full_metrics_row(
        "M4: Full Stack (NB + network features)", y_test_str, preds_m4, cuisines, 0, 0
    )

    # M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)
    X_test_m5_std = scaler_m5.transform(X_test_features[:, cols_m5])
    preds_m5 = [cuisines[int(i)] for i in clf_m5.predict(X_test_m5_std)]
    row_m5, cm_m5 = compute_full_metrics_row(
        "M5: Network-Only Fingerprint Scorer v2", y_test_str, preds_m5, cuisines, 0, 0
    )

    # Reference CV metrics from Phase 3 (on the 13,823 training recipes)
    p3_summary = read_json(PHASE3_DIR / "phase3_summary.json")["model_summaries"]
    cv_lookup = {
        "M0: Original Fingerprint Scorer": p3_summary["M0: Original Fingerprint Scorer (reference)"],
        "M1: Naive Bayes": p3_summary["M1: Naive Bayes Alone (reference)"],
        "M4: Full Stack (NB + network features)": p3_summary["M4: Full Stack — NB + All Network Features (25 features)"],
        "M5: Network-Only Fingerprint Scorer v2": p3_summary["M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)"],
    }

    selected_c_lookup = {
        "M0: Original Fingerprint Scorer": np.nan,
        "M1: Naive Bayes": np.nan,
        "M4: Full Stack (NB + network features)": best_c_m4,
        "M5: Network-Only Fingerprint Scorer v2": best_c_m5,
    }
    n_feat_lookup = {
        "M0: Original Fingerprint Scorer": len(ext_full_train.backbone_pairs_),
        "M1: Naive Bayes": len(ext_full_train.nb_.vocab_),
        "M4: Full Stack (NB + network features)": len(cols_m4),
        "M5: Network-Only Fingerprint Scorer v2": len(cols_m5),
    }

    # Build final_test_results.csv (Macro-F1 primary, Accuracy secondary, per-cuisine P/R/F1, Weighted-F1 supplementary)
    csv_rows = []
    for row_m in [row_m0, row_m1, row_m4, row_m5]:
        m_name = row_m["model"]
        cv_info = cv_lookup[m_name]
        csv_rows.append({
            "model": m_name,
            "test_macro_f1": row_m["macro_f1"],
            "test_accuracy": row_m["accuracy"],
            "cv_macro_f1_mean": cv_info["cv_macro_f1_mean"],
            "cv_macro_f1_std": cv_info["cv_macro_f1_std"],
            "cv_accuracy_mean": cv_info["cv_accuracy_mean"],
            "cv_accuracy_std": cv_info["cv_accuracy_std"],
            "precision_indian": row_m["precision_indian"],
            "precision_italian": row_m["precision_italian"],
            "precision_mexican": row_m["precision_mexican"],
            "recall_indian": row_m["recall_indian"],
            "recall_italian": row_m["recall_italian"],
            "recall_mexican": row_m["recall_mexican"],
            "f1_indian": row_m["f1_indian"],
            "f1_italian": row_m["f1_italian"],
            "f1_mexican": row_m["f1_mexican"],
            "test_weighted_f1_appendix": row_m["weighted_f1"],
            "selected_C": selected_c_lookup[m_name],
            "n_features": n_feat_lookup[m_name],
            "test_samples": row_m["test_samples"],
            "zero_score_fallback_count": row_m["zero_score_fallback_count"],
            "tied_score_fallback_count": row_m["tied_score_fallback_count"],
        })

    final_df = pd.DataFrame(csv_rows)
    final_df.to_csv(FINAL_RESULTS_CSV, index=False, encoding="utf-8")

    # =========================================================================
    # STEP 5: Paired McNemar, Paired Bootstrap CIs (2,000 resamples, seed=42),
    #         and Recall Regression Guard on the Held-Out Test Set
    # =========================================================================
    preds_dict = {
        "M0: Original Fingerprint Scorer": preds_m0,
        "M1: Naive Bayes": preds_m1,
        "M4: Full Stack (NB + network features)": preds_m4,
        "M5: Network-Only Fingerprint Scorer v2": preds_m5,
    }
    rows_by_name = {r["model"]: r for r in [row_m0, row_m1, row_m4, row_m5]}
    boot_test = bootstrap_ci(y_test_str, preds_dict, cuisines, n_boot=2000, seed=SEED)

    def build_paired_comparison(label: str, m_a: str, m_b: str) -> dict:
        mc = mcnemar_test(y_test_str, preds_dict[m_a], preds_dict[m_b])
        k_fwd = f"{m_a} minus {m_b}"
        k_rev = f"{m_b} minus {m_a}"
        if k_fwd in boot_test["pairwise_differences"]:
            bd = boot_test["pairwise_differences"][k_fwd]
            d_f1_ci = bd["delta_macro_f1_ci95"]
            d_acc_ci = bd["delta_accuracy_ci95"]
        else:
            bd = boot_test["pairwise_differences"][k_rev]
            d_f1_ci = [-bd["delta_macro_f1_ci95"][1], -bd["delta_macro_f1_ci95"][0]]
            d_acc_ci = [-bd["delta_accuracy_ci95"][1], -bd["delta_accuracy_ci95"][0]]

        ra = rows_by_name[m_a]
        rb = rows_by_name[m_b]
        delta_f1 = float(ra["macro_f1"] - rb["macro_f1"])
        delta_acc = float(ra["accuracy"] - rb["accuracy"])
        recall_diffs = {
            c: float(ra[f"recall_{c}"] - rb[f"recall_{c}"])
            for c in cuisines
        }
        prec_diffs = {
            c: float(ra[f"precision_{c}"] - rb[f"precision_{c}"])
            for c in cuisines
        }
        f1_diffs = {
            c: float(ra[f"f1_{c}"] - rb[f"f1_{c}"])
            for c in cuisines
        }
        flagged_cuisines = [
            c for c in cuisines
            if (delta_f1 > 0.0) and (recall_diffs[c] < -0.01)
        ]
        ci_includes_zero = bool(d_f1_ci[0] <= 0.0 <= d_f1_ci[1])
        return {
            "comparison": label,
            "model_a": m_a,
            "model_b": m_b,
            "delta_macro_f1": delta_f1,
            "bootstrap_delta_macro_f1_ci95": d_f1_ci,
            "macro_f1_ci_includes_zero": ci_includes_zero,
            "delta_accuracy": delta_acc,
            "bootstrap_delta_accuracy_ci95": d_acc_ci,
            "recall_diffs": recall_diffs,
            "precision_diffs": prec_diffs,
            "f1_diffs": f1_diffs,
            "regression_guard_triggered": len(flagged_cuisines) > 0,
            "regression_guard_flagged_cuisines": flagged_cuisines,
            "mcnemar": mc,
        }

    paired_test_comparisons = {
        "M5 vs M0": build_paired_comparison(
            "M5 (Network-Only Fingerprint Scorer v2) vs. M0 (Original Fingerprint Scorer)",
            "M5: Network-Only Fingerprint Scorer v2",
            "M0: Original Fingerprint Scorer",
        ),
        "M4 vs M1": build_paired_comparison(
            "M4 (Full Stack: NB + network features) vs. M1 (Naive Bayes)",
            "M4: Full Stack (NB + network features)",
            "M1: Naive Bayes",
        ),
    }

    # Standardized coefficients on full 13,823 training set
    m4_coefs_full = []
    for col_j, fname in enumerate(FEATURE_NAMES):
        m4_coefs_full.append({
            "feature": fname,
            "indian": float(clf_m4.coef_[0, col_j]),
            "italian": float(clf_m4.coef_[1, col_j]),
            "mexican": float(clf_m4.coef_[2, col_j]),
        })
    net_names = [FEATURE_NAMES[j] for j in cols_m5]
    m5_coefs_full = []
    for local_j, fname in enumerate(net_names):
        m5_coefs_full.append({
            "feature": fname,
            "indian": float(clf_m5.coef_[0, local_j]),
            "italian": float(clf_m5.coef_[1, local_j]),
            "mexican": float(clf_m5.coef_[2, local_j]),
        })

    total_runtime = round(time.perf_counter() - t0_all, 3)

    summary_payload = {
        "evaluation_protocol": "ONE-TIME FINAL HELD-OUT TEST EVALUATION (3,456 test recipes; trained on 13,823 training recipes)",
        "seed": SEED,
        "n_train_recipes": len(train_recipes),
        "n_test_recipes": len(test_recipes),
        "dataset_sha256": dataset_sha,
        "normalizer_sha256": normalizer_fingerprint(),
        "inner_crossfit_n_splits": INNER_N_SPLITS,
        "inner_crossfit_curveball_n_randomizations": INNER_CURVEBALL_N_RAND,
        "full_training_backbone_source": str(DEFAULT_ARTIFACTS),
        "full_training_curveball_n_randomizations": 1000,
        "inner_crossfit_builds": inner_crossfit_meta,
        "full_training_d3_summary": ext_full_train.d3_summary_,
        "full_training_union_backbone_pairs_count": len(ext_full_train.backbone_pairs_),
        "full_training_nb_vocab_size": len(ext_full_train.nb_.vocab_),
        "meta_model_tuning": {
            "M4_selected_C": best_c_m4,
            "M4_inner_cv_curve": {str(k): v for k, v in curve_m4.items()},
            "M5_selected_C": best_c_m5,
            "M5_inner_cv_curve": {str(k): v for k, v in curve_m5.items()},
        },
        "test_metrics": csv_rows,
        "confusion_matrices": {
            "cuisines_order": cuisines,
            "M0: Original Fingerprint Scorer": cm_m0.tolist(),
            "M1: Naive Bayes": cm_m1.tolist(),
            "M4: Full Stack (NB + network features)": cm_m4.tolist(),
            "M5: Network-Only Fingerprint Scorer v2": cm_m5.tolist(),
        },
        "paired_comparisons": paired_test_comparisons,
        "bootstrap_ci": boot_test,
        "standardized_coefficients_full_train": {
            "M4_full_stack_25_features": m4_coefs_full,
            "M4_intercept": {c: float(clf_m4.intercept_[ci]) for ci, c in enumerate(cuisines)},
            "M5_network_only_22_features": m5_coefs_full,
            "M5_intercept": {c: float(clf_m5.intercept_[ci]) for ci, c in enumerate(cuisines)},
        },
        "predictions_sha256": {
            "M0": sha256_obj(preds_m0),
            "M1": sha256_obj(preds_m1),
            "M4": sha256_obj(preds_m4),
            "M5": sha256_obj(preds_m5),
        },
        "total_runtime_seconds": total_runtime,
    }
    write_json(summary_payload, FINAL_SUMMARY_JSON)
    format_final_test_report(summary_payload)
    log.info(
        "FINAL HELD-OUT TEST EVALUATION COMPLETE in %.2fs | Macro-F1: M0=%.4f M1=%.4f M4=%.4f M5=%.4f",
        total_runtime,
        row_m0["macro_f1"],
        row_m1["macro_f1"],
        row_m4["macro_f1"],
        row_m5["macro_f1"],
    )
    return summary_payload


def format_final_test_report(summary: dict) -> str:
    """Format outputs/d4_experiments/final_test_report.md from final_test_summary.json."""
    cuisines = ["indian", "italian", "mexican"]
    metrics_list = summary["test_metrics"]
    cms = summary["confusion_matrices"]
    paired = summary["paired_comparisons"]
    boot_models = summary["bootstrap_ci"]["models"]
    d3_info = summary["full_training_d3_summary"]
    tuning = summary["meta_model_tuning"]

    p_m5_m0 = paired["M5 vs M0"]
    p_m4_m1 = paired["M4 vs M1"]

    lines = [
        "# Deliverable 4 — Final Held-Out Test Set Evaluation Report (`N = 3,456` Test Recipes)",
        "",
        "> [!IMPORTANT]",
        "> **One-Time Pre-Registered Final Evaluation**:",
        "> - Evaluated **once** on the 3,456 held-out test recipes (`test_ids` in `data/processed/split.json`), with zero retuning or post-hoc changes.",
        "> - All four pre-registered models (`M0`, `M1`, `M4`, `M5`) were trained strictly on the **13,823 training recipes** using the existing training-only D2 Curveball backbones (`outputs/d4_scorer/training_artifacts`, 1,000 randomizations) and D3 definitions (`src/analysis.py`, `src/graph_utils.py`).",
        "> - Training meta-features for `M4` (25 features) and `M5` (22 features) were constructed via 3-fold inner cross-fitting on the 13,823 training recipes (`seed = 42`), and meta-model L2 regularization $C$ was chosen by inner CV on the training set using **Macro-F1** (primary metric).",
        "> - Original D1–D3 and D4 artifacts (`outputs/d4_scorer/results.csv`, etc.) remain completely untouched.",
        "",
        "---",
        "",
        "## 1. Primary Held-Out Test Results (`N = 3,456` Test Recipes; Macro-F1 Primary, Accuracy Secondary)",
        "",
        "Saved to `outputs/d4_experiments/final_test_results.csv`:",
        "",
        "| Pre-Registered Model | Test Macro-F1 [95% Bootstrap CI] (PRIMARY) | 5-Fold Train CV Macro-F1 (Mean ± SD) | Test Accuracy [95% Bootstrap CI] (SECONDARY) | 5-Fold Train CV Accuracy (Mean ± SD) | Test F1 `indian` | Test F1 `italian` | Test F1 `mexican` | Selected Meta $C$ | Zero-Score Fallbacks |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for m in metrics_list:
        m_name = m["model"]
        bm = boot_models[m_name]
        f1_ci = bm["macro_f1_ci95"]
        acc_ci = bm["accuracy_ci95"]
        c_str = f"`{m['selected_C']}`" if not (m["selected_C"] is None or math.isnan(float(m["selected_C"]))) else "—"
        lines.append(
            f"| **{m_name}** | **{m['test_macro_f1']*100:.2f}%** [{f1_ci[0]*100:.2f}%, {f1_ci[1]*100:.2f}%] | "
            f"{m['cv_macro_f1_mean']*100:.2f}% ± {m['cv_macro_f1_std']*100:.2f}% | "
            f"**{m['test_accuracy']*100:.2f}%** [{acc_ci[0]*100:.2f}%, {acc_ci[1]*100:.2f}%] | "
            f"{m['cv_accuracy_mean']*100:.2f}% ± {m['cv_accuracy_std']*100:.2f}% | "
            f"{m['f1_indian']*100:.2f}% | {m['f1_italian']*100:.2f}% | {m['f1_mexican']*100:.2f}% | "
            f"{c_str} | {m['zero_score_fallback_count']} |"
        )

    lines.extend([
        "",
        "### 1.1 Per-Cuisine Precision, Recall, and F1 on the Held-Out Test Set (`N = 3,456`)",
        "",
        "| Model | `indian` ($N=600$) P / R / F1 (%) | `italian` ($N=1,568$) P / R / F1 (%) | `mexican` ($N=1,288$) P / R / F1 (%) | Test Macro-F1 (%) | Test Accuracy (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for m in metrics_list:
        lines.append(
            f"| **{m['model']}** | "
            f"{m['precision_indian']*100:.2f} / {m['recall_indian']*100:.2f} / **{m['f1_indian']*100:.2f}** | "
            f"{m['precision_italian']*100:.2f} / {m['recall_italian']*100:.2f} / **{m['f1_italian']*100:.2f}** | "
            f"{m['precision_mexican']*100:.2f} / {m['recall_mexican']*100:.2f} / **{m['f1_mexican']*100:.2f}** | "
            f"**{m['test_macro_f1']*100:.2f}%** | **{m['test_accuracy']*100:.2f}%** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Paired Statistical Comparisons & Recall Regression Guard on the Held-Out Test Set",
        "",
        "| Paired Comparison (`Model A vs. Model B`) | Test $\\Delta$ Macro-F1 (`A - B`) [95% Bootstrap CI] | CI Excludes Zero? | Test $\\Delta$ Accuracy (`A - B`) [95% Bootstrap CI] | A Right, B Wrong ($b$) | A Wrong, B Right ($c$) | Discordant ($b+c$) | Net Test Recipes ($b-c$) | McNemar $\\chi^2$ (CC) | Exact Binomial $p$-value | $\\Delta$ Recall (`IN / IT / MX`, pp) | Recall Regression Guard ($>1\\text{ pp}$ drop)? |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for comp_key in ["M5 vs M0", "M4 vs M1"]:
        st = paired[comp_key]
        f1_ci = st["bootstrap_delta_macro_f1_ci95"]
        acc_ci = st["bootstrap_delta_accuracy_ci95"]
        mc = st["mcnemar"]
        rd = st["recall_diffs"]
        r_str = f"`{rd['indian']*100:+.2f} / {rd['italian']*100:+.2f} / {rd['mexican']*100:+.2f}`"
        excl_str = "**YES**" if not st["macro_f1_ci_includes_zero"] else "No (includes 0)"
        guard_str = (
            f"**FLAGGED ({', '.join(st['regression_guard_flagged_cuisines'])})**"
            if st["regression_guard_triggered"]
            else "Clear (No drop > 1 pp)"
        )
        lines.append(
            f"| **{st['comparison']}** | **{st['delta_macro_f1']*100:+.2f} pp** [{f1_ci[0]*100:+.2f}, {f1_ci[1]*100:+.2f}] | "
            f"{excl_str} | **{st['delta_accuracy']*100:+.2f} pp** [{acc_ci[0]*100:+.2f}, {acc_ci[1]*100:+.2f}] | "
            f"{mc['a_right_b_wrong']} | {mc['a_wrong_b_right']} | {mc['n_discordant']} | "
            f"**{mc['net_difference_a_minus_b']:+d}** | {mc['chi2_continuity_corrected']:.2f} | "
            f"`{mc['p_value_exact_binomial']:.4g}` | {r_str} | {guard_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Held-Out Test Confusion Matrices (`N = 3,456`; Rows = True Cuisine, Columns = Predicted Cuisine)",
        "",
        "| Model | True Cuisine | Pred `indian` | Pred `italian` | Pred `mexican` | Row Total | Per-Cuisine Precision (%) | Per-Cuisine Recall (%) | Per-Cuisine F1 (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for m in metrics_list:
        m_name = m["model"]
        cm = cms[m_name]
        for ci, c in enumerate(cuisines):
            row_tot = sum(cm[ci])
            lines.append(
                f"| **{m_name}** | `{c}` | {cm[ci][0]:,} | {cm[ci][1]:,} | {cm[ci][2]:,} | {row_tot:,} | "
                f"{m[f'precision_{c}']*100:.2f}% | {m[f'recall_{c}']*100:.2f}% | **{m[f'f1_{c}']*100:.2f}%** |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Full-Training D2/D3 Network Summary & Standardized Meta-Model Coefficients (`13,823` Training Recipes)",
        "",
        "- **Full-Training D2 Backbone (`1,000` Curveball randomizations from `outputs/d4_scorer/training_artifacts`)**:",
        f"  - Backbone nodes: `indian`: `{d3_info['backbone_nodes']['indian']}`, `italian`: `{d3_info['backbone_nodes']['italian']}`, `mexican`: `{d3_info['backbone_nodes']['mexican']}`.",
        f"  - Backbone edges: `indian`: `{d3_info['backbone_edges']['indian']:,}`, `italian`: `{d3_info['backbone_edges']['italian']:,}`, `mexican`: `{d3_info['backbone_edges']['mexican']:,}` (union validated backbone pairs = `{summary['full_training_union_backbone_pairs_count']:,}`).",
        f"  - Louvain communities (`seed=42, resolution=1.0`): `indian`: `{d3_info['n_communities']['indian']}`, `italian`: `{d3_info['n_communities']['italian']}`, `mexican`: `{d3_info['n_communities']['mexican']}`.",
        f"  - Top-10 Weighted-Degree Hubs: `indian`: `{', '.join(d3_info['hubs_top10']['indian'])}` | `italian`: `{', '.join(d3_info['hubs_top10']['italian'])}` | `mexican`: `{', '.join(d3_info['hubs_top10']['mexican'])}`.",
        f"  - D3 Top-5 Pairwise Bridges per Cuisine: `indian`: `{', '.join(d3_info['bridges_per_cuisine']['indian'])}` | `italian`: `{', '.join(d3_info['bridges_per_cuisine']['italian'])}` | `mexican`: `{', '.join(d3_info['bridges_per_cuisine']['mexican'])}`.",
        f"- **Inner 3-Fold CV Selected Regularization $C$ (on 13,823 Cross-Fitted Training Rows)**:",
        f"  - `M4` selected $C = {tuning['M4_selected_C']}$ (inner-CV Macro-F1 = `{tuning['M4_inner_cv_curve'][str(tuning['M4_selected_C'])]['macro_f1_mean']*100:.2f}%`).",
        f"  - `M5` selected $C = {tuning['M5_selected_C']}$ (inner-CV Macro-F1 = `{tuning['M5_inner_cv_curve'][str(tuning['M5_selected_C'])]['macro_f1_mean']*100:.2f}%`).",
        "",
        "---",
        "",
        "## Appendix: Supplementary Test Weighted-F1 (Metric Hierarchy Level 5)",
        "",
        "| Model | Held-Out Test Weighted-F1 (%) |",
        "| :--- | :---: |",
    ])

    for m in metrics_list:
        lines.append(f"| **{m['model']}** | {m['test_weighted_f1_appendix']*100:.2f}% |")

    md = "\n".join(lines) + "\n"
    FINAL_REPORT_MD.write_text(md, encoding="utf-8")
    return md


if __name__ == "__main__":
    run_final_held_out_test_evaluation()
