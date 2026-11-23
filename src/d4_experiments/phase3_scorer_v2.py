"""Phase 3 — Network-Informed Fingerprint Scorer v2 (Stacked) with Strict Inner Cross-Fitting.

Strictly enforces Test Set Isolation and Nested Cross-Fitting:
- Never loads, inspects, or evaluates the 3,456 held-out test recipes (`test_ids` in `split.json`).
- Uses the exact same 5 outer folds (`outputs/d4_experiments/cv_folds.json`, `seed = 42`) as Phases 1 & 2.
- Inside each outer fold `f`, builds training-row features via 3-fold inner cross-fitting:
  for each inner split `(itr, ival)`, normalizes recipes, builds D1 eligible networks, runs Curveball
  backbone validation (`n_randomizations = 50` documented inner shortcut), extracts D3 weighted-degree
  top-10 hubs, D3 Louvain communities, D3 top-5 pairwise bridges, fits BinaryMultinomialNB and
  backbone pair log-odds (`eps = 1e-6`) on `itr` ONLY, and computes the 25 features on `ival`.
- Fits the full pipeline on the outer training fold (`n_randomizations = 100`) to compute the 25 features
  on the outer validation fold.
- Standardizes features using training partition statistics ONLY (`StandardScaler` fitted inside each inner
  training split during hyperparameter tuning, and on the outer training fold for outer evaluation).
- Selects the L2 LogisticRegression meta-model regularization `C` via inner CV using **Macro-F1** as the
  PRIMARY metric (ties broken by accuracy, then smaller `C`).
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.analysis import compute_bridges, compute_communities, compute_hubs
from src.backbone import build_backbones
from src.build_networks import assert_training_only, build_predictive_networks
from src.common import (
    ROOT,
    get_logger,
    load_config,
    read_json,
    set_global_seed,
    sha256_obj,
    write_json,
)
from src.d4_experiments.phase1_diagnostics import (
    CACHE_DIR as P1_CACHE_DIR,
    CV_N_RANDOMIZATIONS,
    CV_N_SPLITS,
    OUT_DIR as P1_OUT_DIR,
    SEED,
    bootstrap_ci,
    load_training_records_only,
    mcnemar_test,
    normalize_partition_records,
)
from src.d4_experiments.phase2_experiments import (
    _fast_macro_f1_3class,
    build_pair_csr,
    compute_log_odds_weights,
)
from src.evaluate import BinaryMultinomialNB, compute_metrics_row
from src.fingerprint import FingerprintScorer, TIE_TOL
from src.graph_utils import backbone_graph, louvain, rank_desc, strengths
from src.normalize import normalizer_fingerprint

log = get_logger("d4_phase3")

PHASE3_DIR = P1_OUT_DIR / "phase3"
PHASE3_CACHE_DIR = PHASE3_DIR / "cache"
INNER_N_SPLITS = 3
INNER_CURVEBALL_N_RAND = 50
OUTER_CURVEBALL_N_RAND = 100
PAIR_LOG_ODDS_EPS = 1e-6
META_C_GRID = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

CUISINES_DEFAULT = ["indian", "italian", "mexican"]


def get_feature_names(cuisines: list[str] = CUISINES_DEFAULT) -> list[str]:
    """Return the canonical ordered list of 25 Phase 3 feature names."""
    names = []
    # Group 1: NB log-posterior (3 features)
    for c in cuisines:
        names.append(f"nb_log_prob_{c}")
    # Group 2: Backbone pair log-odds score (3 features)
    for c in cuisines:
        names.append(f"pair_log_odds_{c}")
    # Group 3: Louvain community features (9 features: cov_frac, max_cov, entropy per cuisine)
    for c in cuisines:
        names.append(f"comm_cov_frac_{c}")
    for c in cuisines:
        names.append(f"comm_max_cov_{c}")
    for c in cuisines:
        names.append(f"comm_entropy_{c}")
    # Group 4: D3 Hub ingredient count (3 features)
    for c in cuisines:
        names.append(f"hub_count_{c}")
    # Group 5: D3 Bridge ingredient count (3 features)
    for c in cuisines:
        names.append(f"bridge_count_{c}")
    # Group 6: Backbone ingredient coverage count (3 features)
    for c in cuisines:
        names.append(f"backbone_ing_count_{c}")
    # Group 7: Global recipe length (1 feature)
    names.append("recipe_length")
    assert len(names) == 25
    return names


FEATURE_NAMES = get_feature_names()
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def get_feature_subsets(cuisines: list[str] = CUISINES_DEFAULT) -> dict[str, list[int]]:
    """Return column index lists for M2-M5 and all leave-one-group-out ablations."""
    nb_cols = [FEATURE_INDEX[f"nb_log_prob_{c}"] for c in cuisines]
    pair_cols = [FEATURE_INDEX[f"pair_log_odds_{c}"] for c in cuisines]
    comm_cov_cols = [FEATURE_INDEX[f"comm_cov_frac_{c}"] for c in cuisines]
    comm_max_cols = [FEATURE_INDEX[f"comm_max_cov_{c}"] for c in cuisines]
    comm_ent_cols = [FEATURE_INDEX[f"comm_entropy_{c}"] for c in cuisines]
    comm_all_cols = comm_cov_cols + comm_max_cols + comm_ent_cols
    comm_struct_cols = comm_max_cols + comm_ent_cols
    hub_cols = [FEATURE_INDEX[f"hub_count_{c}"] for c in cuisines]
    bridge_cols = [FEATURE_INDEX[f"bridge_count_{c}"] for c in cuisines]
    bb_cnt_cols = [FEATURE_INDEX[f"backbone_ing_count_{c}"] for c in cuisines]
    len_cols = [FEATURE_INDEX["recipe_length"]]

    all_cols = list(range(len(FEATURE_NAMES)))
    net_only_cols = [i for i in all_cols if i not in nb_cols]

    return {
        "M2_calibrated_nb_only": nb_cols,
        "M3_nb_plus_pair_log_odds": sorted(nb_cols + pair_cols),
        "M4_full_stack": all_cols,
        "M5_network_only_v2": net_only_cols,
        # Leave-one-group-out ablations from M4
        "Abl1_M4_minus_NB": net_only_cols,
        "Abl2_M4_minus_PairLogOdds": [i for i in all_cols if i not in pair_cols],
        "Abl3_M4_minus_Communities": [i for i in all_cols if i not in comm_struct_cols],
        "Abl3b_M4_minus_AllCommunityCols": [i for i in all_cols if i not in comm_all_cols],
        "Abl4_M4_minus_Hubs": [i for i in all_cols if i not in hub_cols],
        "Abl5_M4_minus_Bridges": [i for i in all_cols if i not in bridge_cols],
        "Abl6_M4_minus_BackboneCovAndLength": [i for i in all_cols if i not in (bb_cnt_cols + comm_cov_cols + len_cols)],
        # Network-only sub-ablations (understanding M5)
        "M5_PairLogOdds_Only": pair_cols,
        "M5_D3_Struct_Only_NoPairLogOdds": [i for i in net_only_cols if i not in pair_cols],
    }


def assign_recipe_length_bin(n_ings: int) -> str:
    """Pre-registered fixed recipe-length bins defined in phase3_design.md prior to execution."""
    if n_ings <= 5:
        return "Short (1–5 ingredients)"
    elif n_ings <= 8:
        return "Medium-Short (6–8 ingredients)"
    elif n_ings <= 12:
        return "Medium-Long (9–12 ingredients)"
    else:
        return "Long (13+ ingredients)"


RECIPE_LENGTH_BINS_ORDER = [
    "Short (1–5 ingredients)",
    "Medium-Short (6–8 ingredients)",
    "Medium-Long (9–12 ingredients)",
    "Long (13+ ingredients)",
]


class PartitionNetworkFeatureExtractor:
    """Fits D1 eligible networks, D2 Curveball backbones, D3 hubs/communities/bridges,
    Naive Bayes, and backbone pair log-odds strictly on a given training partition `train_recipes`,
    and extracts the 25 Phase 3 features for any target recipe list.
    """

    def __init__(
        self,
        cfg_base: dict,
        cuisines: list[str],
        n_randomizations: int,
        cache_dir: Path,
        dataset_sha: str,
        eps: float = PAIR_LOG_ODDS_EPS,
    ):
        self.cfg = copy.deepcopy(cfg_base)
        self.cfg["seed"] = SEED
        self.cfg["null_model"]["n_randomizations"] = int(n_randomizations)
        self.cfg["data"]["cache_dir"] = str(cache_dir)
        self.cuisines = list(cuisines)
        self.c_to_idx = {c: i for i, c in enumerate(self.cuisines)}
        self.dataset_sha = dataset_sha
        self.eps = float(eps)

        self.nb_: BinaryMultinomialNB | None = None
        self.backbone_pairs_: list[tuple[str, str]] = []
        self.W_pair_log_odds_: np.ndarray | None = None
        self.backbone_vocab_: dict[str, set[str]] = {}
        self.hubs_top10_: dict[str, set[str]] = {}
        self.bridges_top5_union_: dict[str, set[str]] = {}
        self.communities_part_: dict[str, dict[str, int]] = {}
        self.n_communities_: dict[str, int] = {}
        self.backbones_: dict[str, pd.DataFrame] = {}
        self.class_counts_: dict[str, int] = {}
        self.d3_summary_: dict = {}

    def fit(self, train_recipes: list[dict], train_ids: set[int], forbidden_ids: set[int]) -> "PartitionNetworkFeatureExtractor":
        """Fit all D1-D3 and base scoring components strictly on `train_recipes`."""
        assert_training_only(train_recipes, train_ids, forbidden_ids)
        min_count = int(self.cfg["network"]["min_ingredient_count"])
        top_n_hubs = int(self.cfg["report"]["top_n_hubs"])
        top_n_bridges = int(self.cfg["report"]["top_n_bridges"])
        comm_seed = int(self.cfg["community"]["seed"])
        comm_res = float(self.cfg["community"]["resolution"])

        # 1. D1 eligible networks on train_recipes ONLY
        nets = build_predictive_networks(train_recipes, self.cuisines, min_count, train_ids, forbidden_ids)

        # 2. D2 Curveball backbones on train_recipes ONLY
        bb_res = build_backbones(
            train_recipes,
            nets,
            self.cfg,
            scope="predictive",
            use_cache=True,
            dataset_sha=self.dataset_sha,
        )
        self.backbones_ = bb_res["backbone"]
        self.class_counts_ = dict(sorted(Counter(r["cuisine"] for r in train_recipes).items()))

        # 3. D3 Graphs, Weighted-Degree Top-10 Hubs, Louvain Communities, and Top-5 Pairwise Bridges
        strength_by_c: dict[str, dict[str, float]] = {}
        prev_by_c: dict[str, dict[str, float]] = {}

        for c in self.cuisines:
            bb_df = self.backbones_[c]
            nodes_df = nets[c]["nodes"]
            G = backbone_graph(bb_df, nodes_df)
            self.backbone_vocab_[c] = set(G.nodes)

            # D3 weighted degree (strength) and top-10 weighted-degree hubs (src/analysis.py lines 55-61)
            s = strengths(G)
            rs = rank_desc(s)
            self.hubs_top10_[c] = {k for k, r in rs.items() if r <= top_n_hubs}
            strength_by_c[c] = s
            prev_by_c[c] = dict(zip(nodes_df["ingredient"], nodes_df["prevalence"]))

            # D3 Louvain communities (src/graph_utils.py:louvain)
            part = louvain(G, seed=comm_seed, resolution=comm_res)
            self.communities_part_[c] = part
            self.n_communities_[c] = (max(part.values()) + 1) if part else 0

        # D3 Bridges: compute_bridges from src/analysis.py (lines 119-131)
        bridges_df, bridge_cand_counts = compute_bridges(
            strength_by_c, prev_by_c, self.cuisines, top_n=top_n_bridges
        )
        self.bridges_top5_union_ = {c: set() for c in self.cuisines}
        for _, row in bridges_df.iterrows():
            c_a, c_b = str(row["cuisine_pair"]).split("-")
            ing = str(row["ingredient"])
            self.bridges_top5_union_[c_a].add(ing)
            self.bridges_top5_union_[c_b].add(ing)

        self.d3_summary_ = {
            "backbone_nodes": {c: len(self.backbone_vocab_[c]) for c in self.cuisines},
            "backbone_edges": {c: len(self.backbones_[c]) for c in self.cuisines},
            "n_communities": dict(self.n_communities_),
            "hubs_top10": {c: sorted(self.hubs_top10_[c]) for c in self.cuisines},
            "bridges_per_cuisine": {c: sorted(self.bridges_top5_union_[c]) for c in self.cuisines},
            "bridge_table_rows": bridges_df.to_dict(orient="records"),
            "bridge_candidate_counts": bridge_cand_counts,
        }

        # 4. Fit BinaryMultinomialNB on train_recipes ONLY
        self.nb_ = BinaryMultinomialNB(
            self.cuisines,
            alpha=float(self.cfg["naive_bayes"]["alpha"]),
            fit_prior=bool(self.cfg["naive_bayes"]["fit_prior"]),
        ).fit(train_recipes, train_ids, forbidden_ids)

        # 5. Fit Backbone Pair Log-Odds weights (ExpB1_bb style) on train_recipes ONLY
        bb_pair_set: set[tuple[str, str]] = set()
        for c in self.cuisines:
            df = self.backbones_[c]
            for a, b in zip(df["ingredient_a"], df["ingredient_b"]):
                pair = (a, b) if a < b else (b, a)
                bb_pair_set.add(pair)
        self.backbone_pairs_ = sorted(bb_pair_set)

        X_tr_pairs = build_pair_csr(train_recipes, self.backbone_pairs_)
        y_tr_idx = np.asarray([self.c_to_idx[r["cuisine"]] for r in train_recipes], dtype=np.int64)
        self.W_pair_log_odds_ = compute_log_odds_weights(
            X_tr_pairs, y_tr_idx, len(self.cuisines), self.eps
        )
        return self

    def transform(self, recipes: list[dict]) -> np.ndarray:
        """Compute the 25-feature matrix of shape (len(recipes), 25) for normalized `recipes`."""
        assert self.nb_ is not None and self.W_pair_log_odds_ is not None
        n = len(recipes)
        X_feat = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)

        # 1. NB log-posterior (columns 0, 1, 2)
        X_nb_csr = self.nb_._to_csr(recipes)
        nb_log_probs = self.nb_.clf_.predict_log_proba(X_nb_csr)
        X_feat[:, 0:3] = nb_log_probs

        # 2. Backbone pair log-odds score (columns 3, 4, 5)
        X_pair_csr = build_pair_csr(recipes, self.backbone_pairs_)
        pair_scores = X_pair_csr @ self.W_pair_log_odds_
        X_feat[:, 3:6] = pair_scores

        # 3-7. Per-recipe D3 community, hub, bridge, backbone coverage, and recipe length features
        for i, r in enumerate(recipes):
            ings = r["ingredients"]
            L = len(ings)
            denom = float(max(L, 1))
            ing_set = set(ings)

            for ci, c in enumerate(self.cuisines):
                part_c = self.communities_part_[c]
                n_comm_c = self.n_communities_[c]

                # Community counts for this recipe in cuisine c
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
                        # Shannon entropy of ingredient spread across communities in cuisine c
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

                # Cols 6..8: comm_cov_frac_{c}
                X_feat[i, 6 + ci] = cov_frac
                # Cols 9..11: comm_max_cov_{c}
                X_feat[i, 9 + ci] = max_cov
                # Cols 12..14: comm_entropy_{c}
                X_feat[i, 12 + ci] = ent
                # Cols 15..17: hub_count_{c}
                X_feat[i, 15 + ci] = float(len(ing_set & self.hubs_top10_[c]))
                # Cols 18..20: bridge_count_{c}
                X_feat[i, 18 + ci] = float(len(ing_set & self.bridges_top5_union_[c]))
                # Cols 21..23: backbone_ing_count_{c}
                X_feat[i, 21 + ci] = float(len(ing_set & self.backbone_vocab_[c]))

            # Col 24: recipe_length
            X_feat[i, 24] = float(L)

        return X_feat


def tune_and_fit_standardized_meta_logreg(
    X_train_raw: np.ndarray,
    y_train_idx: np.ndarray,
    col_indices: list[int],
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    c_grid: list[float] = META_C_GRID,
    seed: int = SEED,
) -> tuple[StandardScaler, LogisticRegression, float, dict[float, dict[str, float]]]:
    """Tune L2 LogisticRegression C on standardized subset of features via inner CV.

    PRIMARY selection metric (per Metric Hierarchy): Mean inner-CV **Macro-F1**.
    Tie-breaker: Mean inner-CV **Accuracy**, then smaller C (stronger L2 regularization).
    StandardScaler is fitted strictly inside each inner training split during tuning,
    and on the full outer training matrix for the final model.
    """
    X_sub = X_train_raw[:, col_indices]
    f1s_per_c: dict[float, list[float]] = {c_val: [] for c_val in c_grid}
    accs_per_c: dict[float, list[float]] = {c_val: [] for c_val in c_grid}

    for itr_idx, ival_idx in inner_splits:
        scaler_in = StandardScaler()
        X_itr = scaler_in.fit_transform(X_sub[itr_idx])
        X_ival = scaler_in.transform(X_sub[ival_idx])
        y_itr = y_train_idx[itr_idx]
        y_ival = y_train_idx[ival_idx]

        clf = LogisticRegression(
            penalty="l2",
            C=float(c_grid[0]),
            solver="lbfgs",
            max_iter=1000,
            warm_start=True,
            random_state=seed,
        )
        for c_val in c_grid:
            clf.C = float(c_val)
            clf.fit(X_itr, y_itr)
            yp = clf.predict(X_ival)
            f1s_per_c[c_val].append(_fast_macro_f1_3class(y_ival, yp))
            accs_per_c[c_val].append(float(np.mean(yp == y_ival)))

    curve = {
        c_val: {
            "macro_f1_mean": float(np.mean(f1s_per_c[c_val])),
            "accuracy_mean": float(np.mean(accs_per_c[c_val])),
        }
        for c_val in c_grid
    }

    # Primary: Macro-F1 desc; secondary: Accuracy desc; tertiary: smaller C
    best_c = sorted(
        c_grid,
        key=lambda c_val: (
            -round(curve[c_val]["macro_f1_mean"], 8),
            -round(curve[c_val]["accuracy_mean"], 8),
            c_val,
        ),
    )[0]

    final_scaler = StandardScaler()
    X_tr_std = final_scaler.fit_transform(X_sub)
    final_clf = LogisticRegression(
        penalty="l2",
        C=float(best_c),
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    final_clf.fit(X_tr_std, y_train_idx)
    return final_scaler, final_clf, float(best_c), curve


def compute_full_metrics_row(
    model_name: str,
    y_true: list[str],
    y_pred: list[str],
    cuisines: list[str],
    zero_fb: int = 0,
    tied_fb: int = 0,
) -> tuple[dict, np.ndarray]:
    """Compute Macro-F1 (primary), Accuracy (secondary), per-cuisine P/R/F1, and Weighted-F1 (appendix)."""
    row, cm = compute_metrics_row(model_name, y_true, y_pred, cuisines, zero_fb, tied_fb)
    weighted_f1 = float(f1_score(y_true, y_pred, labels=cuisines, average="weighted", zero_division=0))
    row["weighted_f1"] = weighted_f1
    return row, cm


def run_phase3_nested_cross_fitting() -> dict:
    """Execute Phase 3 nested 5-fold CV with 3-fold inner cross-fitting on the 13,823 training recipes ONLY."""
    set_global_seed(SEED)
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    PHASE3_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    cuisines = list(cfg["cuisines"])
    c_to_idx = {c: i for i, c in enumerate(cuisines)}
    t0_all = time.perf_counter()

    # Load ONLY the 13,823 training recipes; held-out test set (3,456) is never loaded or inspected
    train_raw, train_ids, test_ids, dataset_sha = load_training_records_only(cfg)

    # Verify outer splits against Phase 1 cv_folds.json
    cv_folds_path = P1_OUT_DIR / "cv_folds.json"
    assert cv_folds_path.exists(), f"Missing {cv_folds_path}"
    p1_folds_json = read_json(cv_folds_path)
    assert p1_folds_json["seed"] == SEED
    assert p1_folds_json["n_splits"] == CV_N_SPLITS
    assert p1_folds_json["n_train_recipes_total"] == 13823

    skf_outer = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=SEED)
    labels_all = [r["cuisine"] for r in train_raw]

    subsets_map = get_feature_subsets(cuisines)
    main_model_order = [
        "M0: Original Fingerprint Scorer (reference)",
        "M1: Naive Bayes Alone (reference)",
        "M2: Calibrated NB Stack Alone (3 features)",
        "M3: NB + Backbone Pair Log-Odds Stack (6 features)",
        "M4: Full Stack — NB + All Network Features (25 features)",
        "M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)",
    ]
    ablation_model_order = [
        "Abl1: M4 minus NB Log-Posteriors (= M5, 22 features)",
        "Abl2: M4 minus Backbone Pair Log-Odds (22 features)",
        "Abl3: M4 minus Louvain Community Structure (max_cov & entropy, 19 features)",
        "Abl3b: M4 minus All Louvain Community Features (cov_frac, max_cov & entropy, 16 features)",
        "Abl4: M4 minus D3 Hub Counts (22 features)",
        "Abl5: M4 minus D3 Bridge Counts (22 features)",
        "Abl6: M4 minus Backbone Node Coverage & Recipe Length (18 features)",
        "Abl_M5a: Network-Only — Backbone Pair Log-Odds Only (3 features)",
        "Abl_M5b: Network-Only — D3 Structural Summaries Only, No Pair Log-Odds (19 features)",
    ]
    all_models_order = main_model_order + ablation_model_order

    meta_spec_list = [
        ("M2: Calibrated NB Stack Alone (3 features)", subsets_map["M2_calibrated_nb_only"]),
        ("M3: NB + Backbone Pair Log-Odds Stack (6 features)", subsets_map["M3_nb_plus_pair_log_odds"]),
        ("M4: Full Stack — NB + All Network Features (25 features)", subsets_map["M4_full_stack"]),
        ("M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)", subsets_map["M5_network_only_v2"]),
        ("Abl1: M4 minus NB Log-Posteriors (= M5, 22 features)", subsets_map["Abl1_M4_minus_NB"]),
        ("Abl2: M4 minus Backbone Pair Log-Odds (22 features)", subsets_map["Abl2_M4_minus_PairLogOdds"]),
        ("Abl3: M4 minus Louvain Community Structure (max_cov & entropy, 19 features)", subsets_map["Abl3_M4_minus_Communities"]),
        ("Abl3b: M4 minus All Louvain Community Features (cov_frac, max_cov & entropy, 16 features)", subsets_map["Abl3b_M4_minus_AllCommunityCols"]),
        ("Abl4: M4 minus D3 Hub Counts (22 features)", subsets_map["Abl4_M4_minus_Hubs"]),
        ("Abl5: M4 minus D3 Bridge Counts (22 features)", subsets_map["Abl5_M4_minus_Bridges"]),
        ("Abl6: M4 minus Backbone Node Coverage & Recipe Length (18 features)", subsets_map["Abl6_M4_minus_BackboneCovAndLength"]),
        ("Abl_M5a: Network-Only — Backbone Pair Log-Odds Only (3 features)", subsets_map["M5_PairLogOdds_Only"]),
        ("Abl_M5b: Network-Only — D3 Structural Summaries Only, No Pair Log-Odds (19 features)", subsets_map["M5_D3_Struct_Only_NoPairLogOdds"]),
    ]

    fold_metrics_by_model: dict[str, list[dict]] = {m: [] for m in all_models_order}
    oof_preds_by_model: dict[str, dict[int, str]] = {m: {} for m in all_models_order}
    oof_probs_by_model: dict[str, dict[int, dict[str, float]]] = {m: {} for m in all_models_order}
    oof_recipe_meta: dict[int, dict] = {}
    fold_reports: list[dict] = []
    verification_log: list[dict] = []

    # Collect standardized logistic coefficients across folds for M4 and M5
    m4_std_coefs_per_fold: list[np.ndarray] = []
    m4_std_intercepts_per_fold: list[np.ndarray] = []
    m5_std_coefs_per_fold: list[np.ndarray] = []
    m5_std_intercepts_per_fold: list[np.ndarray] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf_outer.split(train_raw, labels_all)):
        t_fold0 = time.perf_counter()
        fold_train_raw = [train_raw[i] for i in tr_idx]
        fold_val_raw = [train_raw[i] for i in val_idx]

        fold_train_ids = {r["id"] for r in fold_train_raw}
        fold_val_ids = {r["id"] for r in fold_val_raw}
        outer_forbidden_ids = fold_val_ids | test_ids

        # Verify exact outer fold identity with Phase 1 & Phase 2
        p1_fold = p1_folds_json["folds"][fold_idx]
        assert sha256_obj(sorted(fold_train_ids)) == p1_fold["train_ids_sha256"]
        assert sha256_obj(sorted(fold_val_ids)) == p1_fold["val_ids_sha256"]
        assert sorted(fold_val_ids) == sorted(p1_fold["val_recipe_ids"])
        assert_training_only(fold_train_raw, fold_train_ids, outer_forbidden_ids)

        # Normalize outer training and outer validation partitions independently
        fold_train_recipes = normalize_partition_records(fold_train_raw)
        fold_val_recipes = normalize_partition_records(fold_val_raw)

        y_train_str = [r["cuisine"] for r in fold_train_recipes]
        y_val_str = [r["cuisine"] for r in fold_val_recipes]
        y_train_idx = np.asarray([c_to_idx[c] for c in y_train_str], dtype=np.int64)
        y_val_idx = np.asarray([c_to_idx[c] for c in y_val_str], dtype=np.int64)
        val_recipe_ids = [r["id"] for r in fold_val_recipes]

        # =====================================================================
        # 1. INNER 3-FOLD CROSS-FITTING ON OUTER TRAINING PARTITION
        # =====================================================================
        skf_inner = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=SEED + fold_idx)
        inner_splits = list(skf_inner.split(fold_train_recipes, y_train_idx))

        X_train_crossfit = np.zeros((len(fold_train_recipes), len(FEATURE_NAMES)), dtype=np.float64)
        covered_inner_rows = np.zeros(len(fold_train_recipes), dtype=bool)
        inner_build_meta = []

        for inner_m, (itr_idx, ival_idx) in enumerate(inner_splits):
            t_in0 = time.perf_counter()
            # Re-normalize from raw inner records to strictly rebuild normalization, vocab, D2 & D3
            itr_raw = [fold_train_raw[i] for i in itr_idx]
            ival_raw = [fold_train_raw[i] for i in ival_idx]
            itr_ids = {r["id"] for r in itr_raw}
            ival_ids = {r["id"] for r in ival_raw}
            inner_forbidden_ids = ival_ids | outer_forbidden_ids

            assert len(itr_ids & ival_ids) == 0
            assert len(itr_ids & inner_forbidden_ids) == 0
            assert_training_only(itr_raw, itr_ids, inner_forbidden_ids)

            itr_recipes = normalize_partition_records(itr_raw)
            ival_recipes = normalize_partition_records(ival_raw)

            extractor_inner = PartitionNetworkFeatureExtractor(
                cfg_base=cfg,
                cuisines=cuisines,
                n_randomizations=INNER_CURVEBALL_N_RAND,
                cache_dir=PHASE3_CACHE_DIR,
                dataset_sha=dataset_sha,
                eps=PAIR_LOG_ODDS_EPS,
            ).fit(itr_recipes, itr_ids, inner_forbidden_ids)

            X_ival_feats = extractor_inner.transform(ival_recipes)
            X_train_crossfit[ival_idx, :] = X_ival_feats
            covered_inner_rows[ival_idx] = True

            inner_build_meta.append({
                "inner_fold": inner_m,
                "n_inner_train": len(itr_recipes),
                "n_inner_val": len(ival_recipes),
                "n_backbone_pairs": len(extractor_inner.backbone_pairs_),
                "n_communities": extractor_inner.n_communities_,
                "runtime_seconds": round(time.perf_counter() - t_in0, 3),
            })

        assert np.all(covered_inner_rows), f"Fold {fold_idx}: Not all outer-training rows received cross-fitted features"
        assert not np.any(np.isnan(X_train_crossfit)), f"Fold {fold_idx}: NaN in X_train_crossfit"

        # =====================================================================
        # 2. FULL OUTER-TRAINING BUILD FOR OUTER-VALIDATION FEATURES + M0 & M1
        # =====================================================================
        extractor_outer = PartitionNetworkFeatureExtractor(
            cfg_base=cfg,
            cuisines=cuisines,
            n_randomizations=OUTER_CURVEBALL_N_RAND,
            cache_dir=P1_CACHE_DIR,
            dataset_sha=dataset_sha,
            eps=PAIR_LOG_ODDS_EPS,
        ).fit(fold_train_recipes, fold_train_ids, outer_forbidden_ids)

        X_val_outer = extractor_outer.transform(fold_val_recipes)
        assert not np.any(np.isnan(X_val_outer)), f"Fold {fold_idx}: NaN in X_val_outer"

        # Helper to record fold metrics and OOF predictions
        def record_model(
            m_name: str,
            preds: list[str],
            probs_mat: np.ndarray | None = None,
            zero_fb: int = 0,
            tied_fb: int = 0,
            extra: dict | None = None,
        ) -> dict:
            row, _ = compute_full_metrics_row(m_name, y_val_str, preds, cuisines, zero_fb, tied_fb)
            if extra:
                row.update(extra)
            fold_metrics_by_model[m_name].append(row)
            for local_i, (rid, p) in enumerate(zip(val_recipe_ids, preds)):
                oof_preds_by_model[m_name][rid] = p
                if probs_mat is not None:
                    oof_probs_by_model[m_name][rid] = {
                        c: float(probs_mat[local_i, ci]) for ci, c in enumerate(cuisines)
                    }
            return row

        # M0: Original Fingerprint Scorer
        scorer_m0 = FingerprintScorer(extractor_outer.backbones_, extractor_outer.class_counts_, normalized=False)
        preds_m0, fb_m0 = [], []
        for local_i, r in enumerate(fold_val_recipes):
            exp0 = scorer_m0.explain(r["ingredients"])
            preds_m0.append(exp0["prediction"])
            fb_m0.append(exp0["fallback"])
            oof_recipe_meta[r["id"]] = {
                "id": r["id"],
                "fold": fold_idx,
                "cuisine": r["cuisine"],
                "ingredients": exp0["normalized_input"],
                "n_ingredients": len(exp0["normalized_input"]),
                "length_bin": assign_recipe_length_bin(len(exp0["normalized_input"])),
                "m0_fallback": exp0["fallback"],
                "m0_scores": exp0["scores"],
                "features_25": {
                    FEATURE_NAMES[col_j]: float(X_val_outer[local_i, col_j])
                    for col_j in range(len(FEATURE_NAMES))
                },
            }

        row_m0 = record_model(
            "M0: Original Fingerprint Scorer (reference)",
            preds_m0,
            probs_mat=None,
            zero_fb=sum(x == "zero_score" for x in fb_m0),
            tied_fb=sum(x == "tied_score" for x in fb_m0),
        )
        assert math.isclose(row_m0["accuracy"], p1_fold["fold_metrics"]["Fingerprint Scorer"]["accuracy"], rel_tol=1e-12)

        # M1: Naive Bayes Alone
        preds_m1, m1_zfb, m1_tfb = extractor_outer.nb_.predict_with_fallback(fold_val_recipes)
        m1_probs = np.exp(X_val_outer[:, 0:3])
        row_m1 = record_model(
            "M1: Naive Bayes Alone (reference)",
            preds_m1,
            probs_mat=m1_probs,
            zero_fb=m1_zfb,
            tied_fb=m1_tfb,
        )
        assert math.isclose(row_m1["accuracy"], p1_fold["fold_metrics"]["Naive Bayes"]["accuracy"], rel_tol=1e-12)

        # =====================================================================
        # 3. FIT & EVALUATE STACKED META-MODELS (M2-M5 & ABLATIONS)
        # =====================================================================
        selected_c_by_model = {}
        for m_name, cols in meta_spec_list:
            scaler, clf_meta, best_c, c_curve = tune_and_fit_standardized_meta_logreg(
                X_train_crossfit, y_train_idx, cols, inner_splits, c_grid=META_C_GRID, seed=SEED
            )
            X_val_std = scaler.transform(X_val_outer[:, cols])
            preds_meta = [cuisines[int(i)] for i in clf_meta.predict(X_val_std)]
            probs_meta = clf_meta.predict_proba(X_val_std)
            record_model(
                m_name,
                preds_meta,
                probs_mat=probs_meta,
                zero_fb=0,
                tied_fb=0,
                extra={"best_C": best_c, "n_features": len(cols)},
            )
            selected_c_by_model[m_name] = best_c

            if m_name == "M4: Full Stack — NB + All Network Features (25 features)":
                m4_std_coefs_per_fold.append(clf_meta.coef_.copy())
                m4_std_intercepts_per_fold.append(clf_meta.intercept_.copy())
            elif m_name == "M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)":
                m5_std_coefs_per_fold.append(clf_meta.coef_.copy())
                m5_std_intercepts_per_fold.append(clf_meta.intercept_.copy())

        fold_dt = time.perf_counter() - t_fold0
        verification_log.append({
            "fold": fold_idx,
            "n_outer_train": len(fold_train_recipes),
            "n_outer_val": len(fold_val_recipes),
            "train_ids_sha256": sha256_obj(sorted(fold_train_ids)),
            "val_ids_sha256": sha256_obj(sorted(fold_val_ids)),
            "matches_phase1_cv_folds_json": True,
            "all_train_rows_cross_fitted_out_of_fold": bool(np.all(covered_inner_rows)),
            "inner_folds_count": INNER_N_SPLITS,
            "inner_curveball_n_randomizations": INNER_CURVEBALL_N_RAND,
            "outer_curveball_n_randomizations": OUTER_CURVEBALL_N_RAND,
            "train_val_overlap": len(fold_train_ids & fold_val_ids),
            "train_test_overlap": len(fold_train_ids & test_ids),
            "val_test_overlap": len(fold_val_ids & test_ids),
        })

        fold_reports.append({
            "fold": fold_idx,
            "n_train": len(fold_train_recipes),
            "n_val": len(fold_val_recipes),
            "outer_d3_summary": extractor_outer.d3_summary_,
            "outer_backbone_pairs_count": len(extractor_outer.backbone_pairs_),
            "inner_crossfit_builds": inner_build_meta,
            "selected_meta_C": selected_c_by_model,
            "runtime_seconds": round(fold_dt, 3),
        })

        log.info(
            "Fold %d done in %.2fs | Macro-F1: M0=%.4f M1(NB)=%.4f M2(CalNB)=%.4f M3(NB+Pair)=%.4f M4(Full)=%.4f M5(NetOnly)=%.4f",
            fold_idx,
            fold_dt,
            fold_metrics_by_model["M0: Original Fingerprint Scorer (reference)"][-1]["macro_f1"],
            fold_metrics_by_model["M1: Naive Bayes Alone (reference)"][-1]["macro_f1"],
            fold_metrics_by_model["M2: Calibrated NB Stack Alone (3 features)"][-1]["macro_f1"],
            fold_metrics_by_model["M3: NB + Backbone Pair Log-Odds Stack (6 features)"][-1]["macro_f1"],
            fold_metrics_by_model["M4: Full Stack — NB + All Network Features (25 features)"][-1]["macro_f1"],
            fold_metrics_by_model["M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)"][-1]["macro_f1"],
        )

    total_runtime = round(time.perf_counter() - t0_all, 3)
    return {
        "cuisines": cuisines,
        "main_model_order": main_model_order,
        "ablation_model_order": ablation_model_order,
        "all_models_order": all_models_order,
        "fold_metrics_by_model": fold_metrics_by_model,
        "oof_preds_by_model": oof_preds_by_model,
        "oof_probs_by_model": oof_probs_by_model,
        "oof_recipe_meta": oof_recipe_meta,
        "fold_reports": fold_reports,
        "verification_log": verification_log,
        "m4_std_coefs_per_fold": m4_std_coefs_per_fold,
        "m4_std_intercepts_per_fold": m4_std_intercepts_per_fold,
        "m5_std_coefs_per_fold": m5_std_coefs_per_fold,
        "m5_std_intercepts_per_fold": m5_std_intercepts_per_fold,
        "dataset_sha256": dataset_sha,
        "total_runtime_seconds": total_runtime,
    }


def summarize_phase3_results(p3_data: dict) -> dict:
    """Compute all Phase 3 tables, paired statistics (Macro-F1 primary), recall regression guard checks,
    standardized logistic coefficients, NB prediction flip analysis, and fixed recipe-length bin breakdown.
    Saves artifacts to outputs/d4_experiments/phase3/.
    """
    cuisines = p3_data["cuisines"]
    all_models = p3_data["all_models_order"]
    main_models = p3_data["main_model_order"]
    ablation_models = p3_data["ablation_model_order"]
    fold_metrics = p3_data["fold_metrics_by_model"]
    oof_preds = p3_data["oof_preds_by_model"]
    oof_probs = p3_data["oof_probs_by_model"]
    oof_meta = p3_data["oof_recipe_meta"]

    ordered_ids = sorted(oof_meta.keys())
    assert len(ordered_ids) == 13823
    y_true = [oof_meta[i]["cuisine"] for i in ordered_ids]

    # 1. Summarize every model across the 5 outer folds and pooled OOF (Macro-F1 PRIMARY)
    results_rows = []
    model_summaries = {}

    for m_name in all_models:
        f_rows = fold_metrics[m_name]
        assert len(f_rows) == CV_N_SPLITS
        f1s = [r["macro_f1"] for r in f_rows]
        accs = [r["accuracy"] for r in f_rows]
        wf1s = [r["weighted_f1"] for r in f_rows]

        p_in = [r["precision_indian"] for r in f_rows]
        p_it = [r["precision_italian"] for r in f_rows]
        p_mx = [r["precision_mexican"] for r in f_rows]
        r_in = [r["recall_indian"] for r in f_rows]
        r_it = [r["recall_italian"] for r in f_rows]
        r_mx = [r["recall_mexican"] for r in f_rows]
        f1_in = [r["f1_indian"] for r in f_rows]
        f1_it = [r["f1_italian"] for r in f_rows]
        f1_mx = [r["f1_mexican"] for r in f_rows]
        z_fb = sum(int(r["zero_score_fallback_count"]) for r in f_rows)
        t_fb = sum(int(r["tied_score_fallback_count"]) for r in f_rows)

        preds_m = [oof_preds[m_name][i] for i in ordered_ids]
        oof_row, cm_oof = compute_full_metrics_row(m_name, y_true, preds_m, cuisines, z_fb, t_fb)

        row_dict = {
            "model": m_name,
            "cv_macro_f1_mean": float(np.mean(f1s)),
            "cv_macro_f1_std": float(np.std(f1s, ddof=1)),
            "oof_pooled_macro_f1": float(oof_row["macro_f1"]),
            "cv_accuracy_mean": float(np.mean(accs)),
            "cv_accuracy_std": float(np.std(accs, ddof=1)),
            "oof_pooled_accuracy": float(oof_row["accuracy"]),
            "cv_f1_indian_mean": float(np.mean(f1_in)),
            "cv_f1_italian_mean": float(np.mean(f1_it)),
            "cv_f1_mexican_mean": float(np.mean(f1_mx)),
            "cv_precision_indian_mean": float(np.mean(p_in)),
            "cv_precision_italian_mean": float(np.mean(p_it)),
            "cv_precision_mexican_mean": float(np.mean(p_mx)),
            "cv_recall_indian_mean": float(np.mean(r_in)),
            "cv_recall_italian_mean": float(np.mean(r_it)),
            "cv_recall_mexican_mean": float(np.mean(r_mx)),
            "oof_precision_indian": float(oof_row["precision_indian"]),
            "oof_precision_italian": float(oof_row["precision_italian"]),
            "oof_precision_mexican": float(oof_row["precision_mexican"]),
            "oof_recall_indian": float(oof_row["recall_indian"]),
            "oof_recall_italian": float(oof_row["recall_italian"]),
            "oof_recall_mexican": float(oof_row["recall_mexican"]),
            "oof_f1_indian": float(oof_row["f1_indian"]),
            "oof_f1_italian": float(oof_row["f1_italian"]),
            "oof_f1_mexican": float(oof_row["f1_mexican"]),
            "cv_weighted_f1_mean_appendix": float(np.mean(wf1s)),
            "cv_weighted_f1_std_appendix": float(np.std(wf1s, ddof=1)),
            "oof_pooled_weighted_f1_appendix": float(oof_row["weighted_f1"]),
            "fold_0_macro_f1": float(f1s[0]),
            "fold_1_macro_f1": float(f1s[1]),
            "fold_2_macro_f1": float(f1s[2]),
            "fold_3_macro_f1": float(f1s[3]),
            "fold_4_macro_f1": float(f1s[4]),
            "fold_0_acc": float(accs[0]),
            "fold_1_acc": float(accs[1]),
            "fold_2_acc": float(accs[2]),
            "fold_3_acc": float(accs[3]),
            "fold_4_acc": float(accs[4]),
            "oof_samples": 13823,
            "zero_score_fallback_count": z_fb,
            "tied_score_fallback_count": t_fb,
        }
        results_rows.append(row_dict)
        model_summaries[m_name] = {
            **row_dict,
            "per_fold_rows": f_rows,
            "oof_confusion_matrix": cm_oof.tolist(),
        }

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(PHASE3_DIR / "results.csv", index=False, encoding="utf-8")

    # 2. Paired Bootstrap CIs, McNemar Tests, and Recall Regression Guard (>1.0 pp drop)
    boot_preds_map = {m: [oof_preds[m][i] for i in ordered_ids] for m in all_models}
    boot_ci = bootstrap_ci(y_true, boot_preds_map, cuisines, n_boot=2000, seed=SEED)

    primary_comparisons = [
        (
            "M5 (Network-Only v2) vs. M0 (Original Fingerprint Scorer)",
            "M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)",
            "M0: Original Fingerprint Scorer (reference)",
        ),
        (
            "M4 (Full Stack: NB + All Network Features) vs. M1 (Naive Bayes Alone)",
            "M4: Full Stack — NB + All Network Features (25 features)",
            "M1: Naive Bayes Alone (reference)",
        ),
        (
            "M4 (Full Stack: NB + All Network Features) vs. M2 (Calibrated NB Stack Alone)",
            "M4: Full Stack — NB + All Network Features (25 features)",
            "M2: Calibrated NB Stack Alone (3 features)",
        ),
        (
            "M3 (NB + Backbone Pair Log-Odds) vs. M1 (Naive Bayes Alone)",
            "M3: NB + Backbone Pair Log-Odds Stack (6 features)",
            "M1: Naive Bayes Alone (reference)",
        ),
        (
            "M3 (NB + Backbone Pair Log-Odds) vs. M2 (Calibrated NB Stack Alone)",
            "M3: NB + Backbone Pair Log-Odds Stack (6 features)",
            "M2: Calibrated NB Stack Alone (3 features)",
        ),
        (
            "M2 (Calibrated NB Stack Alone) vs. M1 (Naive Bayes Alone)",
            "M2: Calibrated NB Stack Alone (3 features)",
            "M1: Naive Bayes Alone (reference)",
        ),
        (
            "M5 (Network-Only v2, 22 features) vs. Abl_M5a (Pair Log-Odds Only, 3 features)",
            "M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)",
            "Abl_M5a: Network-Only — Backbone Pair Log-Odds Only (3 features)",
        ),
    ]

    paired_comparisons_summary = {}
    for comp_label, m_a, m_b in primary_comparisons:
        preds_a = boot_preds_map[m_a]
        preds_b = boot_preds_map[m_b]
        mc = mcnemar_test(y_true, preds_a, preds_b)

        k_fwd = f"{m_a} minus {m_b}"
        k_rev = f"{m_b} minus {m_a}"
        if k_fwd in boot_ci["pairwise_differences"]:
            bd = boot_ci["pairwise_differences"][k_fwd]
            d_f1_ci = bd["delta_macro_f1_ci95"]
            d_acc_ci = bd["delta_accuracy_ci95"]
        else:
            bd = boot_ci["pairwise_differences"][k_rev]
            d_f1_ci = [-bd["delta_macro_f1_ci95"][1], -bd["delta_macro_f1_ci95"][0]]
            d_acc_ci = [-bd["delta_accuracy_ci95"][1], -bd["delta_accuracy_ci95"][0]]

        sum_a = model_summaries[m_a]
        sum_b = model_summaries[m_b]
        delta_cv_f1 = float(sum_a["cv_macro_f1_mean"] - sum_b["cv_macro_f1_mean"])
        delta_oof_f1 = float(sum_a["oof_pooled_macro_f1"] - sum_b["oof_pooled_macro_f1"])
        delta_cv_acc = float(sum_a["cv_accuracy_mean"] - sum_b["cv_accuracy_mean"])
        delta_oof_acc = float(sum_a["oof_pooled_accuracy"] - sum_b["oof_pooled_accuracy"])

        # Per-cuisine recall differences and explicit Regression Guard (> 1.0 percentage point drop = -0.01)
        recall_diffs_cv = {
            c: float(sum_a[f"cv_recall_{c}_mean"] - sum_b[f"cv_recall_{c}_mean"])
            for c in cuisines
        }
        recall_diffs_oof = {
            c: float(sum_a[f"oof_recall_{c}"] - sum_b[f"oof_recall_{c}"])
            for c in cuisines
        }
        flagged_cuisines = [
            c for c in cuisines
            if (delta_cv_f1 > 0 or delta_oof_f1 > 0) and (recall_diffs_cv[c] < -0.01 or recall_diffs_oof[c] < -0.01)
        ]
        ci_includes_zero = bool(d_f1_ci[0] <= 0.0 <= d_f1_ci[1])

        fold_diffs = []
        for f_i in range(CV_N_SPLITS):
            ra = fold_metrics[m_a][f_i]
            rb = fold_metrics[m_b][f_i]
            fold_diffs.append({
                "fold": f_i,
                "a_macro_f1": float(ra["macro_f1"]),
                "b_macro_f1": float(rb["macro_f1"]),
                "delta_macro_f1": float(ra["macro_f1"] - rb["macro_f1"]),
                "a_accuracy": float(ra["accuracy"]),
                "b_accuracy": float(rb["accuracy"]),
                "delta_accuracy": float(ra["accuracy"] - rb["accuracy"]),
            })

        paired_comparisons_summary[comp_label] = {
            "model_a": m_a,
            "model_b": m_b,
            "delta_cv_macro_f1_mean": delta_cv_f1,
            "delta_oof_macro_f1": delta_oof_f1,
            "bootstrap_delta_macro_f1_ci95": d_f1_ci,
            "macro_f1_ci_includes_zero": ci_includes_zero,
            "delta_cv_accuracy_mean": delta_cv_acc,
            "delta_oof_accuracy": delta_oof_acc,
            "bootstrap_delta_accuracy_ci95": d_acc_ci,
            "recall_diffs_cv": recall_diffs_cv,
            "recall_diffs_oof": recall_diffs_oof,
            "regression_guard_triggered": len(flagged_cuisines) > 0,
            "regression_guard_flagged_cuisines": flagged_cuisines,
            "mcnemar": mc,
            "fold_wise_differences": fold_diffs,
        }

    # 3. Feature-Group Ablation Table from M4 (with paired Macro-F1 bootstrap CIs vs M4)
    m4_key = "M4: Full Stack — NB + All Network Features (25 features)"
    ablation_comparisons = []
    for abl_name in ablation_models:
        k_fwd = f"{abl_name} minus {m4_key}"
        k_rev = f"{m4_key} minus {abl_name}"
        if k_fwd in boot_ci["pairwise_differences"]:
            bd = boot_ci["pairwise_differences"][k_fwd]
            d_f1_ci = bd["delta_macro_f1_ci95"]
            d_acc_ci = bd["delta_accuracy_ci95"]
        else:
            bd = boot_ci["pairwise_differences"][k_rev]
            d_f1_ci = [-bd["delta_macro_f1_ci95"][1], -bd["delta_macro_f1_ci95"][0]]
            d_acc_ci = [-bd["delta_accuracy_ci95"][1], -bd["delta_accuracy_ci95"][0]]

        mc_abl = mcnemar_test(y_true, boot_preds_map[abl_name], boot_preds_map[m4_key])
        s_abl = model_summaries[abl_name]
        s_m4 = model_summaries[m4_key]
        ablation_comparisons.append({
            "ablation_model": abl_name,
            "cv_macro_f1_mean": s_abl["cv_macro_f1_mean"],
            "cv_macro_f1_std": s_abl["cv_macro_f1_std"],
            "oof_pooled_macro_f1": s_abl["oof_pooled_macro_f1"],
            "delta_macro_f1_vs_M4": float(s_abl["oof_pooled_macro_f1"] - s_m4["oof_pooled_macro_f1"]),
            "delta_macro_f1_ci95_vs_M4": d_f1_ci,
            "cv_accuracy_mean": s_abl["cv_accuracy_mean"],
            "cv_accuracy_std": s_abl["cv_accuracy_std"],
            "oof_pooled_accuracy": s_abl["oof_pooled_accuracy"],
            "delta_accuracy_vs_M4": float(s_abl["oof_pooled_accuracy"] - s_m4["oof_pooled_accuracy"]),
            "delta_accuracy_ci95_vs_M4": d_acc_ci,
            "mcnemar_p_vs_M4": mc_abl["p_value_exact_binomial"],
            "net_recipes_vs_M4": mc_abl["net_difference_a_minus_b"],
        })

    # 4. Standardized Logistic Coefficients for M4 (25 features) and M5 (22 features)
    m4_coefs_arr = np.stack(p3_data["m4_std_coefs_per_fold"], axis=0)  # (5, 3, 25)
    m4_int_arr = np.stack(p3_data["m4_std_intercepts_per_fold"], axis=0)  # (5, 3)
    m4_coef_mean = m4_coefs_arr.mean(axis=0)
    m4_coef_std = m4_coefs_arr.std(axis=0, ddof=1)

    m4_coef_table = []
    for col_j, fname in enumerate(FEATURE_NAMES):
        m4_coef_table.append({
            "feature": fname,
            "indian_mean": float(m4_coef_mean[0, col_j]),
            "indian_std": float(m4_coef_std[0, col_j]),
            "italian_mean": float(m4_coef_mean[1, col_j]),
            "italian_std": float(m4_coef_std[1, col_j]),
            "mexican_mean": float(m4_coef_mean[2, col_j]),
            "mexican_std": float(m4_coef_std[2, col_j]),
            "max_abs_mean": float(np.max(np.abs(m4_coef_mean[:, col_j]))),
        })

    net_only_cols = get_feature_subsets(cuisines)["M5_network_only_v2"]
    net_only_names = [FEATURE_NAMES[j] for j in net_only_cols]
    m5_coefs_arr = np.stack(p3_data["m5_std_coefs_per_fold"], axis=0)  # (5, 3, 22)
    m5_int_arr = np.stack(p3_data["m5_std_intercepts_per_fold"], axis=0)  # (5, 3)
    m5_coef_mean = m5_coefs_arr.mean(axis=0)
    m5_coef_std = m5_coefs_arr.std(axis=0, ddof=1)

    m5_coef_table = []
    for local_j, fname in enumerate(net_only_names):
        m5_coef_table.append({
            "feature": fname,
            "indian_mean": float(m5_coef_mean[0, local_j]),
            "indian_std": float(m5_coef_std[0, local_j]),
            "italian_mean": float(m5_coef_mean[1, local_j]),
            "italian_std": float(m5_coef_std[1, local_j]),
            "mexican_mean": float(m5_coef_mean[2, local_j]),
            "mexican_std": float(m5_coef_std[2, local_j]),
            "max_abs_mean": float(np.max(np.abs(m5_coef_mean[:, local_j]))),
        })

    # 5. Analysis of How Often Network Features Change NB's Prediction and Whether Those Changes Are Correct
    # Compare:
    #   (a) M4 (Full Stack) vs. M1 (Raw Naive Bayes)
    #   (b) M4 (Full Stack) vs. M2 (Calibrated NB Stack Alone — isolates pure effect of network features!)
    #   (c) M3 (NB + Pair Log-Odds) vs. M1 and vs. M2
    def analyze_prediction_changes(model_target: str, model_base: str) -> dict:
        p_t = boot_preds_map[model_target]
        p_b = boot_preds_map[model_base]
        changed_ids = [rid for rid, pt, pb in zip(ordered_ids, p_t, p_b) if pt != pb]
        fixed_ids = [
            rid for rid, yt, pt, pb in zip(ordered_ids, y_true, p_t, p_b)
            if pt != pb and pt == yt and pb != yt
        ]
        broken_ids = [
            rid for rid, yt, pt, pb in zip(ordered_ids, y_true, p_t, p_b)
            if pt != pb and pt != yt and pb == yt
        ]
        both_wrong_changed_ids = [
            rid for rid, yt, pt, pb in zip(ordered_ids, y_true, p_t, p_b)
            if pt != pb and pt != yt and pb != yt
        ]
        assert len(changed_ids) == len(fixed_ids) + len(broken_ids) + len(both_wrong_changed_ids)

        transition_counts = Counter(
            f"true={oof_meta[rid]['cuisine']}: {oof_preds[model_base][rid]} -> {oof_preds[model_target][rid]}"
            for rid in changed_ids
        )
        sample_fixed = [
            {
                "id": rid,
                "fold": oof_meta[rid]["fold"],
                "true_cuisine": oof_meta[rid]["cuisine"],
                "base_pred": oof_preds[model_base][rid],
                "target_pred": oof_preds[model_target][rid],
                "n_ingredients": oof_meta[rid]["n_ingredients"],
                "ingredients": oof_meta[rid]["ingredients"],
                "features": oof_meta[rid]["features_25"],
            }
            for rid in fixed_ids[:8]
        ]
        sample_broken = [
            {
                "id": rid,
                "fold": oof_meta[rid]["fold"],
                "true_cuisine": oof_meta[rid]["cuisine"],
                "base_pred": oof_preds[model_base][rid],
                "target_pred": oof_preds[model_target][rid],
                "n_ingredients": oof_meta[rid]["n_ingredients"],
                "ingredients": oof_meta[rid]["ingredients"],
                "features": oof_meta[rid]["features_25"],
            }
            for rid in broken_ids[:8]
        ]
        return {
            "target_model": model_target,
            "base_model": model_base,
            "n_total_recipes": len(ordered_ids),
            "n_predictions_changed": len(changed_ids),
            "pct_predictions_changed": 100.0 * len(changed_ids) / len(ordered_ids),
            "n_changed_to_correct_fixed": len(fixed_ids),
            "pct_of_changes_correct": (100.0 * len(fixed_ids) / len(changed_ids)) if changed_ids else 0.0,
            "n_changed_from_correct_to_wrong_broken": len(broken_ids),
            "pct_of_changes_broke_correct": (100.0 * len(broken_ids) / len(changed_ids)) if changed_ids else 0.0,
            "n_changed_both_wrong": len(both_wrong_changed_ids),
            "pct_of_changes_both_wrong": (100.0 * len(both_wrong_changed_ids) / len(changed_ids)) if changed_ids else 0.0,
            "net_correct_gain": len(fixed_ids) - len(broken_ids),
            "transition_breakdown": dict(sorted(transition_counts.items(), key=lambda kv: -kv[1])),
            "sample_fixed_recipes": sample_fixed,
            "sample_broken_recipes": sample_broken,
        }

    nb_change_analysis = {
        "M4_vs_M1_raw_NB": analyze_prediction_changes(
            "M4: Full Stack — NB + All Network Features (25 features)",
            "M1: Naive Bayes Alone (reference)",
        ),
        "M4_vs_M2_calibrated_NB": analyze_prediction_changes(
            "M4: Full Stack — NB + All Network Features (25 features)",
            "M2: Calibrated NB Stack Alone (3 features)",
        ),
        "M3_vs_M1_raw_NB": analyze_prediction_changes(
            "M3: NB + Backbone Pair Log-Odds Stack (6 features)",
            "M1: Naive Bayes Alone (reference)",
        ),
        "M3_vs_M2_calibrated_NB": analyze_prediction_changes(
            "M3: NB + Backbone Pair Log-Odds Stack (6 features)",
            "M2: Calibrated NB Stack Alone (3 features)",
        ),
    }

    # 6. Pre-Registered Recipe-Length Bins Table (Macro-F1 primary & Accuracy secondary across M0-M5)
    length_bin_rows = []
    for b_label in RECIPE_LENGTH_BINS_ORDER:
        sub_ids = [i for i in ordered_ids if oof_meta[i]["length_bin"] == b_label]
        n_sub = len(sub_ids)
        if n_sub == 0:
            continue
        yt_sub = [oof_meta[i]["cuisine"] for i in sub_ids]
        mean_ings = float(np.mean([oof_meta[i]["n_ingredients"] for i in sub_ids]))

        bin_entry = {
            "length_bin": b_label,
            "n_recipes": n_sub,
            "pct_recipes": 100.0 * n_sub / len(ordered_ids),
            "mean_ingredients": mean_ings,
        }
        for short_code, m_full in [
            ("M0", "M0: Original Fingerprint Scorer (reference)"),
            ("M1", "M1: Naive Bayes Alone (reference)"),
            ("M2", "M2: Calibrated NB Stack Alone (3 features)"),
            ("M3", "M3: NB + Backbone Pair Log-Odds Stack (6 features)"),
            ("M4", "M4: Full Stack — NB + All Network Features (25 features)"),
            ("M5", "M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)"),
        ]:
            yp_sub = [oof_preds[m_full][i] for i in sub_ids]
            bin_entry[f"{short_code}_macro_f1"] = float(
                f1_score(yt_sub, yp_sub, labels=cuisines, average="macro", zero_division=0)
            )
            bin_entry[f"{short_code}_accuracy"] = float(accuracy_score(yt_sub, yp_sub))

        # Paired differences inside this length bin
        mc_m5_m0 = mcnemar_test(
            yt_sub,
            [oof_preds["M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)"][i] for i in sub_ids],
            [oof_preds["M0: Original Fingerprint Scorer (reference)"][i] for i in sub_ids],
        )
        mc_m4_m1 = mcnemar_test(
            yt_sub,
            [oof_preds["M4: Full Stack — NB + All Network Features (25 features)"][i] for i in sub_ids],
            [oof_preds["M1: Naive Bayes Alone (reference)"][i] for i in sub_ids],
        )
        mc_m4_m2 = mcnemar_test(
            yt_sub,
            [oof_preds["M4: Full Stack — NB + All Network Features (25 features)"][i] for i in sub_ids],
            [oof_preds["M2: Calibrated NB Stack Alone (3 features)"][i] for i in sub_ids],
        )
        bin_entry["m5_vs_m0_net_recipes"] = mc_m5_m0["net_difference_a_minus_b"]
        bin_entry["m5_vs_m0_p"] = mc_m5_m0["p_value_exact_binomial"]
        bin_entry["m4_vs_m1_net_recipes"] = mc_m4_m1["net_difference_a_minus_b"]
        bin_entry["m4_vs_m1_p"] = mc_m4_m1["p_value_exact_binomial"]
        bin_entry["m4_vs_m2_net_recipes"] = mc_m4_m2["net_difference_a_minus_b"]
        bin_entry["m4_vs_m2_p"] = mc_m4_m2["p_value_exact_binomial"]
        length_bin_rows.append(bin_entry)

    summary_payload = {
        "seed": SEED,
        "n_outer_splits": CV_N_SPLITS,
        "n_inner_splits": INNER_N_SPLITS,
        "outer_curveball_n_randomizations": OUTER_CURVEBALL_N_RAND,
        "inner_curveball_n_randomizations": INNER_CURVEBALL_N_RAND,
        "pair_log_odds_eps": PAIR_LOG_ODDS_EPS,
        "n_train_recipes_evaluated_oof": 13823,
        "n_test_recipes_untouched": 3456,
        "dataset_sha256": p3_data["dataset_sha256"],
        "normalizer_sha256": normalizer_fingerprint(),
        "total_runtime_seconds": p3_data["total_runtime_seconds"],
        "feature_names_25": FEATURE_NAMES,
        "verification_log": p3_data["verification_log"],
        "fold_reports": p3_data["fold_reports"],
        "model_summaries": model_summaries,
        "paired_comparisons": paired_comparisons_summary,
        "ablation_comparisons_vs_M4": ablation_comparisons,
        "standardized_coefficients": {
            "M4_full_stack_25_features": m4_coef_table,
            "M4_intercept_mean": {c: float(m4_int_arr[:, ci].mean()) for ci, c in enumerate(cuisines)},
            "M4_intercept_std": {c: float(m4_int_arr[:, ci].std(ddof=1)) for ci, c in enumerate(cuisines)},
            "M5_network_only_22_features": m5_coef_table,
            "M5_intercept_mean": {c: float(m5_int_arr[:, ci].mean()) for ci, c in enumerate(cuisines)},
            "M5_intercept_std": {c: float(m5_int_arr[:, ci].std(ddof=1)) for ci, c in enumerate(cuisines)},
        },
        "nb_prediction_change_analysis": nb_change_analysis,
        "recipe_length_bins_table": length_bin_rows,
        "bootstrap_ci": boot_ci,
    }
    write_json(summary_payload, PHASE3_DIR / "phase3_summary.json")
    return summary_payload


def main() -> int:
    from src.d4_experiments.phase3_report_builder import format_phase3_report

    p3_data = run_phase3_nested_cross_fitting()
    summarize_phase3_results(p3_data)
    format_phase3_report()
    log.info(
        "Phase 3 nested cross-fitting complete in %.2fs. Outputs saved to %s",
        p3_data["total_runtime_seconds"],
        PHASE3_DIR,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

