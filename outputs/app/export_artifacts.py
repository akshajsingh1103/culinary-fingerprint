"""One-time artifact exporter for the Culinary Fingerprint Streamlit application.

Reads the existing training-only pipeline artifacts (13,823 training recipes, seed 42,
same D2 Curveball backbone from `outputs/d4_scorer/training_artifacts`, same D3 hub/community/bridge
definitions, same 22-feature Network-Only Scorer v2 with 3-fold cross-fitted training features
and inner-CV selected L2 regularization C = 1.0) and writes self-contained runtime artifacts
to `outputs/app/artifacts/`.

Test set recipes are never used for fitting; a sample of real What's Cooking recipes
is exported solely for the interactive "Surprise me" display button.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.analysis import compute_bridges, compute_communities, compute_hubs
from src.backbone import build_backbones
from src.build_networks import assert_training_only, build_predictive_networks
from src.common import load_config, load_raw_dataset, read_json, rpath, set_global_seed, write_json
from src.fingerprint import DEFAULT_ARTIFACTS
from src.graph_utils import backbone_graph, louvain, rank_desc, strengths
from src.normalize import (
    PLURAL_WHITELIST,
    SINGULARIZE_EXCEPTIONS,
    SYNONYMS,
    normalize_recipe,
    normalizer_fingerprint,
)

APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts"
P3_CACHE_DIR = ROOT / "outputs" / "d4_experiments" / "phase3" / "cache"

SEED = 42
INNER_N_SPLITS = 3
INNER_CURVEBALL_N_RAND = 50
PAIR_LOG_ODDS_EPS = 1e-6
META_C_GRID = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
CUISINES = ["indian", "italian", "mexican"]

V2_FEATURE_NAMES = (
    [f"pair_log_odds_{c}" for c in CUISINES]
    + [f"comm_cov_frac_{c}" for c in CUISINES]
    + [f"comm_max_cov_{c}" for c in CUISINES]
    + [f"comm_entropy_{c}" for c in CUISINES]
    + [f"hub_count_{c}" for c in CUISINES]
    + [f"bridge_count_{c}" for c in CUISINES]
    + [f"backbone_ing_count_{c}" for c in CUISINES]
    + ["recipe_length"]
)
assert len(V2_FEATURE_NAMES) == 22


def _fast_macro_f1_3class(y_true_idx: np.ndarray, y_pred_idx: np.ndarray) -> float:
    cm = np.bincount(3 * y_true_idx + y_pred_idx, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    f1_c = np.divide(2.0 * tp, denom, out=np.zeros(3, dtype=np.float64), where=denom > 0)
    return float(f1_c.mean())


def _compute_pair_log_odds_map(
    recipes: list[dict],
    y_idx: np.ndarray,
    backbone_pairs: list[tuple[str, str]],
    n_classes: int = 3,
    eps: float = PAIR_LOG_ODDS_EPS,
) -> dict[tuple[str, str], list[float]]:
    """Compute log-odds weights w(c, p) = log((P(p|c)+eps)/(P(p|not c)+eps)) for each pair in backbone_pairs."""
    pair_set = set(backbone_pairs)
    active_ings = {x for p in backbone_pairs for x in p}
    n_total = len(recipes)
    class_totals = [int(np.sum(y_idx == ci)) for ci in range(n_classes)]

    pair_counts_per_class: dict[tuple[str, str], list[int]] = {
        p: [0] * n_classes for p in backbone_pairs
    }
    for r, ci in zip(recipes, y_idx):
        ings = [x for x in r["ingredients"] if x in active_ings]
        if len(ings) >= 2:
            for pair in itertools.combinations(ings, 2):
                if pair in pair_set:
                    pair_counts_per_class[pair][int(ci)] += 1

    weights_map: dict[tuple[str, str], list[float]] = {}
    for p in backbone_pairs:
        cnts = pair_counts_per_class[p]
        tot_p = sum(cnts)
        w_vec = []
        for ci in range(n_classes):
            n_c = float(class_totals[ci])
            n_not_c = float(n_total - class_totals[ci])
            p_c = cnts[ci] / n_c
            p_not_c = (tot_p - cnts[ci]) / n_not_c
            w_vec.append(float(math.log((p_c + eps) / (p_not_c + eps))))
        weights_map[p] = w_vec
    return weights_map


def _extract_22_features_for_partition(
    recipes: list[dict],
    cuisines: list[str],
    pair_weights_map: dict[tuple[str, str], list[float]],
    communities_part: dict[str, dict[str, int]],
    n_communities: dict[str, int],
    hubs_top10: dict[str, set[str]],
    bridges_union: dict[str, set[str]],
    backbone_vocab: dict[str, set[str]],
) -> np.ndarray:
    """Compute the 22-feature matrix of shape (len(recipes), 22) for normalized recipes."""
    X = np.zeros((len(recipes), 22), dtype=np.float64)
    active_pair_ings = {x for p in pair_weights_map for x in p}

    for i, r in enumerate(recipes):
        ings = r["ingredients"]
        L = len(ings)
        denom = float(max(L, 1))
        ing_set = set(ings)

        # Cols 0..2: pair_log_odds_{c}
        p_ings = [x for x in ings if x in active_pair_ings]
        if len(p_ings) >= 2:
            s0, s1, s2 = 0.0, 0.0, 0.0
            for pair in itertools.combinations(p_ings, 2):
                w = pair_weights_map.get(pair)
                if w is not None:
                    s0 += w[0]
                    s1 += w[1]
                    s2 += w[2]
            X[i, 0] = s0
            X[i, 1] = s1
            X[i, 2] = s2

        for ci, c in enumerate(cuisines):
            part_c = communities_part[c]
            n_comm_c = n_communities[c]
            if n_comm_c > 0 and L > 0:
                comm_counts = defaultdict(int)
                total_in_comm = 0
                for u in ing_set:
                    cid = part_c.get(u)
                    if cid is not None:
                        comm_counts[cid] += 1
                        total_in_comm += 1
                if total_in_comm > 0:
                    cov_frac = total_in_comm / denom
                    max_cov = max(comm_counts.values()) / denom
                    ent = 0.0
                    for cnt in comm_counts.values():
                        p = cnt / float(total_in_comm)
                        if p > 0.0:
                            ent -= p * math.log(p)
                else:
                    cov_frac = 0.0
                    max_cov = 0.0
                    ent = 0.0
            else:
                cov_frac = 0.0
                max_cov = 0.0
                ent = 0.0

            X[i, 3 + ci] = cov_frac
            X[i, 6 + ci] = max_cov
            X[i, 9 + ci] = ent
            X[i, 12 + ci] = float(len(ing_set & hubs_top10[c]))
            X[i, 15 + ci] = float(len(ing_set & bridges_union[c]))
            X[i, 18 + ci] = float(len(ing_set & backbone_vocab[c]))

        X[i, 21] = float(L)
    return X


def export_all_artifacts() -> dict:
    """Export all runtime artifacts for Scorer v1 and Scorer v2 to outputs/app/artifacts/."""
    set_global_seed(SEED)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cuisines = list(cfg["cuisines"])
    c_to_idx = {c: i for i, c in enumerate(cuisines)}

    split_file = rpath(cfg["data"]["processed_dir"]) / "split.json"
    split = read_json(split_file)
    train_id_list = [int(i) for i in split["train_ids"]]
    test_id_list = [int(i) for i in split["test_ids"]]
    train_ids = set(train_id_list)
    test_ids = set(test_id_list)

    records, ds_report = load_raw_dataset(cfg)
    by_id = {
        int(r["id"]): {"id": int(r["id"]), "cuisine": r["cuisine"], "ingredients": list(r["ingredients"])}
        for r in records
        if r.get("cuisine") in set(cuisines)
    }
    train_raw = [by_id[i] for i in train_id_list]
    test_raw = [by_id[i] for i in test_id_list]
    assert_training_only(train_raw, train_ids, test_ids)

    train_recipes = [
        {"id": r["id"], "cuisine": r["cuisine"], "ingredients": normalize_recipe(r["ingredients"])}
        for r in train_raw
    ]
    y_train_idx = np.asarray([c_to_idx[r["cuisine"]] for r in train_recipes], dtype=np.int64)

    # 1. 3-Fold Inner Cross-Fitting on the 13,823 Training Recipes ONLY (identical to final test run)
    skf_inner = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=SEED)
    inner_splits = list(skf_inner.split(train_recipes, y_train_idx))

    X_train_crossfit = np.zeros((len(train_recipes), 22), dtype=np.float64)
    min_count = int(cfg["network"]["min_ingredient_count"])
    top_n_hubs = int(cfg["report"]["top_n_hubs"])
    top_n_bridges = int(cfg["report"]["top_n_bridges"])
    comm_seed = int(cfg["community"]["seed"])
    comm_res = float(cfg["community"]["resolution"])

    cfg_inner = copy.deepcopy(cfg)
    cfg_inner["seed"] = SEED
    cfg_inner["null_model"]["n_randomizations"] = INNER_CURVEBALL_N_RAND
    cfg_inner["data"]["cache_dir"] = str(P3_CACHE_DIR)

    for itr_idx, ival_idx in inner_splits:
        itr_raw = [train_raw[i] for i in itr_idx]
        ival_raw = [train_raw[i] for i in ival_idx]
        itr_ids = {r["id"] for r in itr_raw}
        ival_ids = {r["id"] for r in ival_raw}
        forbidden_inner = ival_ids | test_ids
        assert_training_only(itr_raw, itr_ids, forbidden_inner)

        itr_recipes = [
            {"id": r["id"], "cuisine": r["cuisine"], "ingredients": normalize_recipe(r["ingredients"])}
            for r in itr_raw
        ]
        ival_recipes = [
            {"id": r["id"], "cuisine": r["cuisine"], "ingredients": normalize_recipe(r["ingredients"])}
            for r in ival_raw
        ]

        nets_in = build_predictive_networks(itr_recipes, cuisines, min_count, itr_ids, forbidden_inner)
        bb_res_in = build_backbones(
            itr_recipes, nets_in, cfg_inner, scope="predictive", use_cache=True, dataset_sha=ds_report["sha256"]
        )
        bbs_in = bb_res_in["backbone"]

        bb_vocab_in, hubs_in, comm_part_in, n_comm_in = {}, {}, {}, {}
        str_in, prev_in = {}, {}
        bb_pairs_set_in: set[tuple[str, str]] = set()

        for c in cuisines:
            df_c = bbs_in[c]
            nc = nets_in[c]["nodes"]
            G = backbone_graph(df_c, nc)
            bb_vocab_in[c] = set(G.nodes)
            s = strengths(G)
            rs = rank_desc(s)
            hubs_in[c] = {k for k, rk in rs.items() if rk <= top_n_hubs}
            str_in[c] = s
            prev_in[c] = dict(zip(nc["ingredient"], nc["prevalence"]))
            part = louvain(G, seed=comm_seed, resolution=comm_res)
            comm_part_in[c] = part
            n_comm_in[c] = (max(part.values()) + 1) if part else 0
            for a, b in zip(df_c["ingredient_a"], df_c["ingredient_b"]):
                bb_pairs_set_in.add((a, b) if a < b else (b, a))

        bridges_df_in, _ = compute_bridges(str_in, prev_in, cuisines, top_n=top_n_bridges)
        bridges_in = {c: set() for c in cuisines}
        for _, row in bridges_df_in.iterrows():
            ca, cb = str(row["cuisine_pair"]).split("-")
            ing = str(row["ingredient"])
            bridges_in[ca].add(ing)
            bridges_in[cb].add(ing)

        bb_pairs_in = sorted(bb_pairs_set_in)
        y_itr_idx = np.asarray([c_to_idx[r["cuisine"]] for r in itr_recipes], dtype=np.int64)
        pw_map_in = _compute_pair_log_odds_map(itr_recipes, y_itr_idx, bb_pairs_in, len(cuisines), PAIR_LOG_ODDS_EPS)

        X_train_crossfit[ival_idx, :] = _extract_22_features_for_partition(
            ival_recipes, cuisines, pw_map_in, comm_part_in, n_comm_in, hubs_in, bridges_in, bb_vocab_in
        )

    # 2. Tune L2 Regularization C on Cross-Fitted Training Matrix (Macro-F1 primary, Accuracy secondary)
    f1s_per_c: dict[float, list[float]] = {c_val: [] for c_val in META_C_GRID}
    accs_per_c: dict[float, list[float]] = {c_val: [] for c_val in META_C_GRID}
    for itr_idx, ival_idx in inner_splits:
        sc_in = StandardScaler()
        X_itr = sc_in.fit_transform(X_train_crossfit[itr_idx])
        X_ival = sc_in.transform(X_train_crossfit[ival_idx])
        y_itr, y_ival = y_train_idx[itr_idx], y_train_idx[ival_idx]
        clf = LogisticRegression(
            penalty="l2", C=float(META_C_GRID[0]), solver="lbfgs", max_iter=1000, warm_start=True, random_state=SEED
        )
        for c_val in META_C_GRID:
            clf.C = float(c_val)
            clf.fit(X_itr, y_itr)
            yp = clf.predict(X_ival)
            f1s_per_c[c_val].append(_fast_macro_f1_3class(y_ival, yp))
            accs_per_c[c_val].append(float(np.mean(yp == y_ival)))

    curve_v2 = {
        c_val: {
            "macro_f1_mean": float(np.mean(f1s_per_c[c_val])),
            "accuracy_mean": float(np.mean(accs_per_c[c_val])),
        }
        for c_val in META_C_GRID
    }
    best_c_v2 = sorted(
        META_C_GRID,
        key=lambda c_val: (-round(curve_v2[c_val]["macro_f1_mean"], 8), -round(curve_v2[c_val]["accuracy_mean"], 8), c_val),
    )[0]

    scaler_v2 = StandardScaler()
    X_tr_std = scaler_v2.fit_transform(X_train_crossfit)
    clf_v2 = LogisticRegression(
        penalty="l2", C=float(best_c_v2), solver="lbfgs", max_iter=1000, random_state=SEED
    )
    clf_v2.fit(X_tr_std, y_train_idx)

    # 3. Extract Full-Training D2 Backbones, D3 Hubs/Communities/Bridges, and Pair Log-Odds Weights
    nodes_train_df = pd.read_csv(DEFAULT_ARTIFACTS / "nodes_train.csv")
    train_meta = read_json(DEFAULT_ARTIFACTS / "training_metadata.json")
    nodes_train_df.to_csv(ARTIFACTS_DIR / "nodes_train.csv", index=False, encoding="utf-8")

    bbs_full = {}
    hubs_dfs, comms_dfs = {}, {}
    strength_full, prev_full = {}, {}
    bb_vocab_full, hubs_top10_full, comm_part_full, n_comm_full = {}, {}, {}, {}
    comm_labels_full: dict[str, dict[int, str]] = {}
    bb_pairs_set_full: set[tuple[str, str]] = set()

    for c in cuisines:
        bb_df = pd.read_csv(DEFAULT_ARTIFACTS / f"backbone_{c}.csv")
        bb_df.to_csv(ARTIFACTS_DIR / f"backbone_{c}.csv", index=False, encoding="utf-8")
        bbs_full[c] = bb_df

        nc = nodes_train_df[nodes_train_df["cuisine"] == c].reset_index(drop=True)
        G = backbone_graph(bb_df, nc)
        bb_vocab_full[c] = set(G.nodes)

        h_df, h_info = compute_hubs(bb_df, nc, c, top_n=top_n_hubs)
        h_df.to_csv(ARTIFACTS_DIR / f"hubs_{c}.csv", index=False, encoding="utf-8")
        hubs_dfs[c] = h_df
        hubs_top10_full[c] = {k for k, rk in h_info["rank_strength"].items() if rk <= top_n_hubs}
        strength_full[c] = h_info["strength"]
        prev_full[c] = dict(zip(nc["ingredient"], nc["prevalence"]))

        c_df, part = compute_communities(
            bb_df, nc, c, seed=comm_seed, resolution=comm_res, k=int(cfg["report"]["community_top_k"])
        )
        c_df.to_csv(ARTIFACTS_DIR / f"communities_{c}.csv", index=False, encoding="utf-8")
        comms_dfs[c] = c_df
        comm_part_full[c] = part
        n_comm_full[c] = (max(part.values()) + 1) if part else 0
        comm_labels_full[c] = (
            dict(zip(c_df["community_id"].astype(int), c_df["community_label"].astype(str)))
            if len(c_df)
            else {}
        )

        for a, b in zip(bb_df["ingredient_a"], bb_df["ingredient_b"]):
            bb_pairs_set_full.add((a, b) if a < b else (b, a))

    bridges_df_full, bridge_cand_counts = compute_bridges(
        strength_full, prev_full, cuisines, top_n=top_n_bridges
    )
    bridges_df_full.to_csv(ARTIFACTS_DIR / "bridges.csv", index=False, encoding="utf-8")
    bridges_union_full = {c: set() for c in cuisines}
    bridges_details_full: dict[str, list[dict]] = {c: [] for c in cuisines}
    for _, row in bridges_df_full.iterrows():
        ca, cb = str(row["cuisine_pair"]).split("-")
        ing = str(row["ingredient"])
        bridges_union_full[ca].add(ing)
        bridges_union_full[cb].add(ing)
        bridges_details_full[ca].append({
            "ingredient": ing,
            "pair": str(row["cuisine_pair"]),
            "other_cuisine": cb,
            "bridge_score": float(row["bridge_score"]),
            "rank": int(row["rank"]),
        })
        bridges_details_full[cb].append({
            "ingredient": ing,
            "pair": str(row["cuisine_pair"]),
            "other_cuisine": ca,
            "bridge_score": float(row["bridge_score"]),
            "rank": int(row["rank"]),
        })

    bb_pairs_full = sorted(bb_pairs_set_full)
    pw_map_full = _compute_pair_log_odds_map(
        train_recipes, y_train_idx, bb_pairs_full, len(cuisines), PAIR_LOG_ODDS_EPS
    )

    # Save pair log-odds weights CSV & JSON
    pw_rows = [
        {
            "ingredient_a": p[0],
            "ingredient_b": p[1],
            "log_odds_indian": w[0],
            "log_odds_italian": w[1],
            "log_odds_mexican": w[2],
        }
        for p, w in pw_map_full.items()
    ]
    pd.DataFrame(pw_rows).to_csv(ARTIFACTS_DIR / "pair_log_odds_v2.csv", index=False, encoding="utf-8")

    # 4. Save Normalization Rules
    norm_rules_payload = {
        "normalizer_sha256": normalizer_fingerprint(),
        "plural_whitelist": dict(sorted(PLURAL_WHITELIST.items())),
        "singularize_exceptions": sorted(SINGULARIZE_EXCEPTIONS),
        "synonyms": {k: {"target": v[0], "rationale": v[1]} for k, v in sorted(SYNONYMS.items())},
    }
    write_json(norm_rules_payload, ARTIFACTS_DIR / "normalization_rules.json")

    # 5. Save Ingredient Vocabulary Ordered by Training Recipe Frequency Descending
    global_counts = Counter(x for r in train_recipes for x in r["ingredients"])
    per_c_counts = {
        c: Counter(x for r in train_recipes if r["cuisine"] == c for x in r["ingredients"])
        for c in cuisines
    }
    union_backbone_vocab = set().union(*(bb_vocab_full[c] for c in cuisines))
    union_eligible_vocab = set(nodes_train_df["ingredient"].unique())

    vocab_entries = []
    for ing, cnt in sorted(global_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        in_bbs = [c for c in cuisines if ing in bb_vocab_full[c]]
        vocab_entries.append({
            "ingredient": ing,
            "train_recipe_count": int(cnt),
            "in_any_backbone": bool(ing in union_backbone_vocab),
            "in_any_eligible": bool(ing in union_eligible_vocab),
            "in_backbones": in_bbs,
            "counts_per_cuisine": {c: int(per_c_counts[c][ing]) for c in cuisines},
        })

    vocab_payload = {
        "n_total_training_vocab": len(vocab_entries),
        "n_union_backbone_vocab": len(union_backbone_vocab),
        "n_union_eligible_vocab": len(union_eligible_vocab),
        "ingredients_ordered_by_frequency": [e["ingredient"] for e in vocab_entries],
        "backbone_ingredients_ordered_by_frequency": [
            e["ingredient"] for e in vocab_entries if e["in_any_backbone"]
        ],
        "entries": vocab_entries,
    }
    write_json(vocab_payload, ARTIFACTS_DIR / "ingredient_vocabulary.json")

    # 6. Save Scorer v2 Model Parameters (Scaler, Logistic Weights, D3 Lookups)
    v2_model_payload = {
        "model_name": "Fingerprint Scorer v2 (Network-Only Stacked L2 Logistic Regression)",
        "cuisines": cuisines,
        "seed": SEED,
        "n_train_recipes": len(train_recipes),
        "train_class_counts": dict(train_meta["train_class_counts"]),
        "feature_names": V2_FEATURE_NAMES,
        "selected_C": float(best_c_v2),
        "pair_log_odds_eps": PAIR_LOG_ODDS_EPS,
        "scaler_mean": [float(x) for x in scaler_v2.mean_],
        "scaler_scale": [float(x) for x in scaler_v2.scale_],
        "coef": [[float(x) for x in row] for row in clf_v2.coef_],
        "intercept": [float(x) for x in clf_v2.intercept_],
        "d3_structures": {
            "backbone_vocab": {c: sorted(bb_vocab_full[c]) for c in cuisines},
            "hubs_top10": {c: sorted(hubs_top10_full[c]) for c in cuisines},
            "hub_strengths": {
                c: {k: float(strength_full[c][k]) for k in sorted(hubs_top10_full[c])}
                for c in cuisines
            },
            "bridges_union": {c: sorted(bridges_union_full[c]) for c in cuisines},
            "bridges_details": bridges_details_full,
            "communities_partition": {
                c: {k: int(v) for k, v in sorted(comm_part_full[c].items())}
                for c in cuisines
            },
            "n_communities": {c: int(n_comm_full[c]) for c in cuisines},
            "community_labels": {
                c: {str(cid): lbl for cid, lbl in sorted(comm_labels_full[c].items())}
                for c in cuisines
            },
        },
    }
    write_json(v2_model_payload, ARTIFACTS_DIR / "v2_model.json")

    # 7. Save Held-Out Test Results & Confusion Matrices for Scorer v1 and Scorer v2 ONLY
    final_test_df = pd.read_csv(ROOT / "outputs" / "d4_experiments" / "final_test_results.csv")
    final_test_json = read_json(ROOT / "outputs" / "d4_experiments" / "final_test_summary.json")

    v1_v2_map = {
        "M0: Original Fingerprint Scorer": "Fingerprint Scorer v1 (Original NPMI Sum)",
        "M5: Network-Only Fingerprint Scorer v2": "Fingerprint Scorer v2 (Network-Informed Stacked)",
    }
    sub_df = final_test_df[final_test_df["model"].isin(v1_v2_map.keys())].copy()
    sub_df["model"] = sub_df["model"].map(v1_v2_map)
    # Drop any columns or references to other models
    keep_cols = [
        "model",
        "test_macro_f1",
        "test_accuracy",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "precision_indian",
        "precision_italian",
        "precision_mexican",
        "recall_indian",
        "recall_italian",
        "recall_mexican",
        "f1_indian",
        "f1_italian",
        "f1_mexican",
        "test_weighted_f1_appendix",
        "selected_C",
        "n_features",
        "test_samples",
        "zero_score_fallback_count",
        "tied_score_fallback_count",
    ]
    sub_df = sub_df[keep_cols].reset_index(drop=True)
    sub_df.to_csv(ARTIFACTS_DIR / "final_test_results.csv", index=False, encoding="utf-8")

    p_v2_v1 = final_test_json["paired_comparisons"]["M5 vs M0"]
    test_summary_v1_v2 = {
        "n_train_recipes": 13823,
        "n_test_recipes": 3456,
        "cuisines": cuisines,
        "models": sub_df.to_dict(orient="records"),
        "confusion_matrices": {
            "Fingerprint Scorer v1 (Original NPMI Sum)": final_test_json["confusion_matrices"][
                "M0: Original Fingerprint Scorer"
            ],
            "Fingerprint Scorer v2 (Network-Informed Stacked)": final_test_json["confusion_matrices"][
                "M5: Network-Only Fingerprint Scorer v2"
            ],
        },
        "paired_comparison_v2_vs_v1": {
            "delta_macro_f1": p_v2_v1["delta_macro_f1"],
            "bootstrap_delta_macro_f1_ci95": p_v2_v1["bootstrap_delta_macro_f1_ci95"],
            "delta_accuracy": p_v2_v1["delta_accuracy"],
            "bootstrap_delta_accuracy_ci95": p_v2_v1["bootstrap_delta_accuracy_ci95"],
            "recall_diffs": p_v2_v1["recall_diffs"],
            "precision_diffs": p_v2_v1["precision_diffs"],
            "f1_diffs": p_v2_v1["f1_diffs"],
            "mcnemar": p_v2_v1["mcnemar"],
        },
        "pipeline_overview": {
            "d1": {
                "title": "D1 — Ingredient Co-Occurrence Networks",
                "training_recipes": 13823,
                "min_ingredient_count": 20,
                "eligible_nodes": {
                    c: int((nodes_train_df["cuisine"] == c).sum()) for c in cuisines
                },
            },
            "d2": {
                "title": "D2 — Curveball Null-Model Backbone Validation",
                "n_randomizations": 1000,
                "z_threshold": 3.0,
                "min_co_count": 5,
                "backbone_nodes": {c: len(bb_vocab_full[c]) for c in cuisines},
                "backbone_edges": {c: len(bbs_full[c]) for c in cuisines},
                "union_backbone_pairs": len(bb_pairs_full),
            },
            "d3": {
                "title": "D3 — Meso-Scale Hubs, Communities & Cross-Cuisine Bridges",
                "n_communities": {c: int(n_comm_full[c]) for c in cuisines},
                "top_n_hubs": top_n_hubs,
                "top_n_bridges_per_pair": top_n_bridges,
            },
        },
    }
    write_json(test_summary_v1_v2, ARTIFACTS_DIR / "final_test_summary_v1_v2.json")

    # 8. Export Sample Real What's Cooking Recipes for "Surprise me" (Stratified across Cuisines)
    rng = np.random.default_rng(SEED)
    sample_recipes = []
    # Include both held-out display examples and training examples with rich ingredient lists
    for split_label, pool in [("held-out display", test_raw), ("training", train_raw)]:
        for c in cuisines:
            c_pool = [r for r in pool if r["cuisine"] == c and 4 <= len(r["ingredients"]) <= 18]
            chosen_idx = rng.choice(len(c_pool), size=min(25, len(c_pool)), replace=False)
            for idx in sorted(chosen_idx):
                rec = c_pool[int(idx)]
                sample_recipes.append({
                    "id": int(rec["id"]),
                    "cuisine": rec["cuisine"],
                    "split": split_label,
                    "raw_ingredients": list(rec["ingredients"]),
                    "normalized_ingredients": normalize_recipe(rec["ingredients"]),
                })
    write_json({"recipes": sample_recipes}, ARTIFACTS_DIR / "sample_recipes.json")

    return {
        "artifacts_dir": str(ARTIFACTS_DIR),
        "selected_C_v2": float(best_c_v2),
        "n_vocab": len(vocab_entries),
        "n_backbone_pairs": len(bb_pairs_full),
        "n_sample_recipes": len(sample_recipes),
    }


if __name__ == "__main__":
    info = export_all_artifacts()
    print(json.dumps(info, indent=2))
