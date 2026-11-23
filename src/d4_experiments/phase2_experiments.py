"""Phase 2 Controlled Experiments for Deliverable 4 (5-fold Nested CV on the 13,823 training recipes ONLY).

Strictly enforces Test Set Isolation and Nested Cross-Validation:
- Never loads, inspects, or evaluates the 3,456 held-out test recipes (`test_ids` in `split.json`).
- Uses the exact same 5 outer folds (`cv_folds.json`, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`)
  as Phase 1 so all model comparisons are strictly paired.
- Every tunable hyperparameter (epsilon, logistic regression C, pair support threshold,
  NPMI threshold, negative evidence weights) is selected via inner 3-fold StratifiedKFold
  inside each outer training partition (`seed = 42`). The outer validation fold is used
  solely to evaluate the configuration chosen by inner CV.
- For compute efficiency, each outer fold's Curveball backbone (`n_randomizations = 100`) is
  built once on the outer training partition and reused for inner tuning of parameters
  that do not alter the backbone (documented approximation per Addendum #1).
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
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

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
    CACHE_DIR,
    CV_N_RANDOMIZATIONS,
    CV_N_SPLITS,
    OUT_DIR,
    SEED,
    bootstrap_ci,
    load_training_records_only,
    mcnemar_test,
    normalize_partition_records,
)
from src.evaluate import BinaryMultinomialNB, compute_metrics_row
from src.fingerprint import FingerprintScorer, TIE_TOL
from src.normalize import normalizer_fingerprint

log = get_logger("d4_phase2")

INNER_N_SPLITS = 3
LR_C_GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
EPS_GRID = [1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
MIN_SUPPORT_GRID = [5, 10, 20]
NEG_LAMBDA_GRID = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
N_RANDOM_SEEDS = 10


def build_unigram_csr(recipes: list[dict], vocab: list[str]) -> sp.csr_matrix:
    """Build binary recipe x unigram CSR matrix from normalized recipes and sorted vocab."""
    v_idx = {w: i for i, w in enumerate(vocab)}
    indptr = [0]
    indices: list[int] = []
    for r in recipes:
        cols = sorted({v_idx[x] for x in r["ingredients"] if x in v_idx})
        indices.extend(cols)
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float64)
    return sp.csr_matrix(
        (data, np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
        shape=(len(recipes), len(vocab)),
    )


def build_pair_csr(recipes: list[dict], pair_vocab: list[tuple[str, str]]) -> sp.csr_matrix:
    """Build binary recipe x pair-presence CSR matrix from normalized recipes and sorted pair_vocab.

    Optimization: first filters recipe ingredients to those appearing in any pair of pair_vocab.
    """
    p_idx = {p: i for i, p in enumerate(pair_vocab)}
    active_ings = {x for p in pair_vocab for x in p}
    indptr = [0]
    indices: list[int] = []
    for r in recipes:
        ings = [x for x in r["ingredients"] if x in active_ings]
        if len(ings) >= 2:
            cols = sorted(
                p_idx[pair]
                for pair in itertools.combinations(ings, 2)
                if pair in p_idx
            )
            indices.extend(cols)
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float64)
    return sp.csr_matrix(
        (data, np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
        shape=(len(recipes), len(pair_vocab)),
    )


def fit_eval_nb_custom_vocab(
    X_train: sp.csr_matrix,
    y_train_idx: np.ndarray,
    X_val: sp.csr_matrix,
    cuisines: list[str],
    alpha: float = 1.0,
    fit_prior: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Fit MultinomialNB on binary CSR matrix and return predictions + log_probs on X_val."""
    clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
    clf.fit(X_train, y_train_idx)
    preds_idx = clf.predict(X_val)
    log_probs = clf.predict_log_proba(X_val)
    return [cuisines[int(i)] for i in preds_idx], log_probs


def _fast_macro_f1_3class(y_true_idx: np.ndarray, y_pred_idx: np.ndarray) -> float:
    """Fast 3-class macro-F1 via bincount."""
    cm = np.bincount(3 * y_true_idx + y_pred_idx, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    f1_c = np.divide(2.0 * tp, denom, out=np.zeros(3, dtype=np.float64), where=denom > 0)
    return float(f1_c.mean())


def tune_and_fit_l2_logreg(
    X_train: sp.csr_matrix,
    y_train_idx: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    c_grid: list[float] = LR_C_GRID,
    seed: int = SEED,
) -> tuple[LogisticRegression, float, dict[float, float]]:
    """Tune L2 LogisticRegression C via inner CV (maximizing accuracy, tie-broken by macro-F1 then smaller C)."""
    accs_per_c: dict[float, list[float]] = {c_val: [] for c_val in c_grid}
    f1s_per_c: dict[float, list[float]] = {c_val: [] for c_val in c_grid}

    for itr_idx, ival_idx in inner_splits:
        X_itr, y_itr = X_train[itr_idx], y_train_idx[itr_idx]
        X_ival, y_ival = X_train[ival_idx], y_train_idx[ival_idx]
        clf = LogisticRegression(
            penalty="l2",
            C=float(c_grid[0]),
            solver="lbfgs",
            max_iter=300,
            tol=1e-3,
            warm_start=True,
            random_state=seed,
        )
        for c_val in c_grid:
            clf.C = float(c_val)
            clf.fit(X_itr, y_itr)
            yp = clf.predict(X_ival)
            accs_per_c[c_val].append(float(np.mean(yp == y_ival)))
            f1s_per_c[c_val].append(_fast_macro_f1_3class(y_ival, yp))

    mean_acc_per_c = {c_val: float(np.mean(accs_per_c[c_val])) for c_val in c_grid}
    mean_f1_per_c = {c_val: float(np.mean(f1s_per_c[c_val])) for c_val in c_grid}

    # Best C by highest inner CV accuracy, tie-break by highest macro-F1, then smallest C (strongest regularization)
    best_c = sorted(
        c_grid,
        key=lambda c_val: (-round(mean_acc_per_c[c_val], 8), -round(mean_f1_per_c[c_val], 8), c_val),
    )[0]

    final_clf = LogisticRegression(
        penalty="l2",
        C=float(best_c),
        solver="lbfgs",
        max_iter=500,
        tol=1e-4,
        random_state=seed,
    )
    final_clf.fit(X_train, y_train_idx)
    return final_clf, float(best_c), mean_acc_per_c


def compute_log_odds_weights(
    X_train_pairs: sp.csr_matrix,
    y_train_idx: np.ndarray,
    n_classes: int,
    eps: float,
) -> np.ndarray:
    """Compute log-odds pair weights w(c, p) = log((P(p|c) + eps) / (P(p|not c) + eps)).

    Returns weight matrix W of shape (n_pairs, n_classes) so scores S = X @ W.
    Only active pairs present in a recipe contribute to the sum.
    """
    n_total = X_train_pairs.shape[0]
    total_counts = np.asarray(X_train_pairs.sum(axis=0)).ravel().astype(np.float64)
    W = np.zeros((X_train_pairs.shape[1], n_classes), dtype=np.float64)
    for c_i in range(n_classes):
        mask_c = (y_train_idx == c_i)
        n_c = float(np.sum(mask_c))
        n_not_c = float(n_total - n_c)
        counts_c = np.asarray(X_train_pairs[mask_c].sum(axis=0)).ravel().astype(np.float64)
        counts_not_c = total_counts - counts_c
        p_given_c = counts_c / n_c
        p_given_not_c = counts_not_c / n_not_c
        W[:, c_i] = np.log((p_given_c + eps) / (p_given_not_c + eps))
    return W


def predict_from_score_matrix(
    scores_mat: np.ndarray,
    cuisines: list[str],
    priority_order: list[str],
    fallback_preds: list[str] | None = None,
) -> tuple[list[str], int, int]:
    """Vectorized argmax predictions with deterministic tie/zero-score handling matching FingerprintScorer."""
    n_rows, n_c = scores_mat.shape
    c_to_idx = {c: i for i, c in enumerate(cuisines)}
    priority_idx = [c_to_idx[c] for c in priority_order]

    max_s = np.max(scores_mat, axis=1, keepdims=True)
    is_max = np.abs(scores_mat - max_s) <= TIE_TOL
    n_tied = np.sum(is_max, axis=1)
    is_zero = np.all(np.abs(scores_mat) <= TIE_TOL, axis=1)

    zero_fb = int(np.sum(is_zero))
    tied_fb = int(np.sum((n_tied > 1) & (~is_zero)))

    # Fast path for unique maxima
    pred_idx = np.argmax(scores_mat, axis=1)
    # Resolve ties using priority_idx
    tie_rows = np.where(n_tied > 1)[0]
    for r_i in tie_rows:
        for p_i in priority_idx:
            if is_max[r_i, p_i]:
                pred_idx[r_i] = p_i
                break

    preds = [cuisines[int(i)] for i in pred_idx]
    if fallback_preds is not None and zero_fb > 0:
        for r_i in np.where(is_zero)[0]:
            preds[int(r_i)] = fallback_preds[int(r_i)]
    return preds, zero_fb, tied_fb


def tune_and_predict_b1_log_odds(
    X_train_pairs: sp.csr_matrix,
    y_train_idx: np.ndarray,
    X_val_pairs: sp.csr_matrix,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    cuisines: list[str],
    priority_order: list[str],
    eps_grid: list[float] = EPS_GRID,
) -> tuple[list[str], int, int, float, dict[float, float], np.ndarray]:
    """Tune epsilon via inner CV on X_train_pairs, then compute log-odds weights on full outer train fold."""
    n_classes = len(cuisines)
    mean_acc_per_eps: dict[float, float] = {}
    mean_f1_per_eps: dict[float, float] = {}

    for eps in eps_grid:
        accs, f1s = [], []
        for itr_idx, ival_idx in inner_splits:
            X_itr, y_itr = X_train_pairs[itr_idx], y_train_idx[itr_idx]
            X_ival, y_ival = X_train_pairs[ival_idx], y_train_idx[ival_idx]
            W = compute_log_odds_weights(X_itr, y_itr, n_classes, eps)
            S_ival = X_ival @ W
            preds_ival, _, _ = predict_from_score_matrix(S_ival, cuisines, priority_order)
            y_ival_str = [cuisines[int(i)] for i in y_ival]
            accs.append(float(accuracy_score(y_ival_str, preds_ival)))
            f1s.append(float(f1_score(y_ival_str, preds_ival, labels=cuisines, average="macro", zero_division=0)))
        mean_acc_per_eps[eps] = float(np.mean(accs))
        mean_f1_per_eps[eps] = float(np.mean(f1s))

    best_eps = sorted(
        eps_grid,
        key=lambda e: (-round(mean_acc_per_eps[e], 8), -round(mean_f1_per_eps[e], 8), e),
    )[0]

    W_full = compute_log_odds_weights(X_train_pairs, y_train_idx, n_classes, best_eps)
    S_val = X_val_pairs @ W_full
    preds_val, zero_fb, tied_fb = predict_from_score_matrix(S_val, cuisines, priority_order)
    return preds_val, zero_fb, tied_fb, float(best_eps), mean_acc_per_eps, W_full


def extract_all_pair_counts(recipes: list[dict], allowed_vocab: set[str] | None = None) -> Counter:
    """Count recipe-level pair frequencies across `recipes` (optionally restricted to `allowed_vocab`)."""
    counts: Counter = Counter()
    for r in recipes:
        ings = [x for x in r["ingredients"] if (allowed_vocab is None or x in allowed_vocab)]
        if len(ings) >= 2:
            counts.update(itertools.combinations(ings, 2))
    return counts


def extract_eligible_candidate_pairs_from_fold_nets(
    fold_nets: dict[str, dict],
    fold_bbs: dict[str, pd.DataFrame],
    fold_train_recipes: list[dict],
    cuisines: list[str],
    min_co_count: int = 5,
) -> dict:
    """Construct the exact matched-size pair sets for Experiment E from the outer fold training partition ONLY.

    Universe of candidate pairs for Experiment E:
    All unordered ingredient pairs (a < b) among eligible nodes (`min_ingredient_count >= 20` in at least one
    cuisine on the fold's training partition) that satisfy the same minimum co-occurrence support constraint
    (`max_c co_count_c(a, b) >= 5`, matching the backbone filter `min_co_count = 5`).

    Returns:
      - `backbone_pairs`: sorted list of K_f union validated backbone pairs (z > 3.0, co_count >= 5)
      - `npmi_pairs`: sorted list of top K_f candidate pairs ranked by max per-cuisine NPMI (with co_count >= 5)
      - `freq_pairs`: sorted list of top K_f candidate pairs ranked by total training-fold recipe co-occurrence count
      - `eligible_universe`: sorted list of all candidate pairs in the eligible universe (with co_count >= 5)
      - `npmi_cutoff`: the effective NPMI threshold at rank K_f
      - `freq_cutoff`: the effective co-occurrence count threshold at rank K_f
    """
    # 1. D2-style fold backbone pairs (Curveball z > 3.0, co_count >= 5)
    bb_set: set[tuple[str, str]] = set()
    for c in cuisines:
        df = fold_bbs[c]
        for a, b in zip(df["ingredient_a"], df["ingredient_b"]):
            pair = (a, b) if a < b else (b, a)
            bb_set.add(pair)
    backbone_pairs = sorted(bb_set)
    k_target = len(backbone_pairs)

    # 2. Eligible candidate pair universe: pairs in fold_nets[c]["edges"] with co_count >= min_co_count (5)
    max_npmi_map: dict[tuple[str, str], float] = {}
    max_c_co_map: dict[tuple[str, str], int] = {}
    sum_c_co_map: dict[tuple[str, str], int] = defaultdict(int)

    for c in cuisines:
        edf = fold_nets[c]["edges"]
        sub = edf[edf["co_count"] >= min_co_count]
        for a, b, co, npmi in zip(sub["ingredient_a"], sub["ingredient_b"], sub["co_count"], sub["NPMI"]):
            pair = (a, b) if a < b else (b, a)
            co_i = int(co)
            npmi_f = float(npmi)
            if pair not in max_npmi_map or npmi_f > max_npmi_map[pair]:
                max_npmi_map[pair] = npmi_f
            if pair not in max_c_co_map or co_i > max_c_co_map[pair]:
                max_c_co_map[pair] = co_i
            sum_c_co_map[pair] += co_i

    eligible_universe = sorted(max_npmi_map.keys())
    assert k_target <= len(eligible_universe), (
        f"Backbone pair count {k_target} exceeds eligible universe {len(eligible_universe)}"
    )
    # Verify every backbone pair is inside the eligible universe
    assert bb_set <= set(eligible_universe), "Every backbone pair must belong to the eligible pair universe"

    # Also compute total training-fold co-occurrence across all 3 cuisines for pairs in eligible_universe
    union_elig_nodes = {x for p in eligible_universe for x in p}
    global_pair_counts = extract_all_pair_counts(fold_train_recipes, allowed_vocab=union_elig_nodes)

    # (ii) NPMI-thresholded pairs of matched size K_f: rank eligible_universe by max per-cuisine NPMI (co_count >= 5)
    ranked_by_npmi = sorted(
        eligible_universe,
        key=lambda p: (-round(max_npmi_map[p], 12), -global_pair_counts[p], p),
    )
    npmi_pairs = sorted(ranked_by_npmi[:k_target])
    npmi_cutoff = float(max_npmi_map[ranked_by_npmi[k_target - 1]])

    # (iii) Top-K most frequent pairs of matched size K_f: rank eligible_universe by training-fold frequency
    ranked_by_freq = sorted(
        eligible_universe,
        key=lambda p: (-global_pair_counts[p], -sum_c_co_map[p], -round(max_npmi_map[p], 12), p),
    )
    freq_pairs = sorted(ranked_by_freq[:k_target])
    freq_cutoff = int(global_pair_counts[ranked_by_freq[k_target - 1]])

    return {
        "k_target": k_target,
        "backbone_pairs": backbone_pairs,
        "npmi_pairs": npmi_pairs,
        "freq_pairs": freq_pairs,
        "eligible_universe": eligible_universe,
        "npmi_cutoff": npmi_cutoff,
        "freq_cutoff": freq_cutoff,
        "overlap_bb_npmi": len(bb_set & set(npmi_pairs)),
        "overlap_bb_freq": len(bb_set & set(freq_pairs)),
        "overlap_npmi_freq": len(set(npmi_pairs) & set(freq_pairs)),
    }


def compute_raw_scorer_matrix(scorer: FingerprintScorer, recipes: list[dict], cuisines: list[str]) -> np.ndarray:
    """Compute raw per-cuisine NPMI sum matrix of shape (n_recipes, n_cuisines)."""
    S = np.zeros((len(recipes), len(cuisines)), dtype=np.float64)
    for i, r in enumerate(recipes):
        ings = r["ingredients"]
        if len(ings) >= 2:
            for ci, c in enumerate(cuisines):
                edges_c = scorer.edges[c]
                total = 0.0
                for pair in itertools.combinations(ings, 2):
                    w = edges_c.get(pair)
                    if w is not None:
                        total += w
                S[i, ci] = total
    return S


def compute_Other_cuisine_exclusive_npmi_matrix(
    scorer: FingerprintScorer,
    recipes: list[dict],
    cuisines: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute negative-evidence matrices for Experiment F from training-fold backbones only.

    For each recipe and cuisine c:
    - `neg_all[i, c]`: sum of NPMI_{c'}(p) over competing cuisines c' != c for all backbone pairs in E_{c'}
    - `neg_excl[i, c]`: sum of NPMI_{c'}(p) over competing cuisines c' != c for pairs in E_{c'} that are NOT in E_c
    """
    n_c = len(cuisines)
    neg_all = np.zeros((len(recipes), n_c), dtype=np.float64)
    neg_excl = np.zeros((len(recipes), n_c), dtype=np.float64)
    for i, r in enumerate(recipes):
        ings = r["ingredients"]
        if len(ings) < 2:
            continue
        for pair in itertools.combinations(ings, 2):
            weights = [scorer.edges[c].get(pair) for c in cuisines]
            for ci in range(n_c):
                for cj in range(n_c):
                    if cj == ci:
                        continue
                    w_other = weights[cj]
                    if w_other is not None:
                        neg_all[i, ci] += w_other
                        if weights[ci] is None:
                            neg_excl[i, ci] += w_other
    return neg_all, neg_excl


def assign_coverage_bin_fixed(n_matched_sum: int) -> str:
    """Pre-registered fixed coverage bins based on total matched backbone pairs across cuisines (Addendum #4).

    Rule defined prior to inspecting Phase 2 results:
    - Low coverage (`0-4` matched pairs): zero-score fallbacks and very sparse pair matches (~20.8% of training set)
    - Mid coverage (`5-15` matched pairs): moderate pair density (~34.6% of training set)
    - High coverage (`16+` matched pairs): dense pair coverage (~44.6% of training set)
    """
    if n_matched_sum <= 4:
        return "Low (0–4 matched pairs)"
    elif n_matched_sum <= 15:
        return "Mid (5–15 matched pairs)"
    else:
        return "High (16+ matched pairs)"


def assign_coverage_bin_tertile(n_matched_sum: int, q1: float, q2: float) -> str:
    """Training-fold quantile (tertile) coverage bin computed strictly from outer training partition quantiles."""
    if n_matched_sum <= q1:
        return "Tertile 1 (Low)"
    elif n_matched_sum <= q2:
        return "Tertile 2 (Mid)"
    else:
        return "Tertile 3 (High)"


def run_phase2_nested_cv() -> dict:
    """Execute all Phase 2 experiments under 5-fold nested CV on the 13,823 training recipes ONLY."""
    set_global_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    cuisines = list(cfg["cuisines"])
    c_to_idx = {c: i for i, c in enumerate(cuisines)}
    t0_all = time.perf_counter()

    # Load ONLY the 13,823 training recipes; held-out test set (3,456) is never loaded or inspected
    train_raw, train_ids, test_ids, dataset_sha = load_training_records_only(cfg)

    # Verify against Phase 1 cv_folds.json so results are strictly paired
    cv_folds_path = OUT_DIR / "cv_folds.json"
    assert cv_folds_path.exists(), f"Missing {cv_folds_path} from Phase 1"
    phase1_folds_json = read_json(cv_folds_path)
    assert phase1_folds_json["seed"] == SEED
    assert phase1_folds_json["n_splits"] == CV_N_SPLITS
    assert phase1_folds_json["n_train_recipes_total"] == 13823

    cfg_cv = copy.deepcopy(cfg)
    cfg_cv["seed"] = SEED
    cfg_cv["null_model"]["n_randomizations"] = CV_N_RANDOMIZATIONS
    cfg_cv["data"]["cache_dir"] = str(CACHE_DIR)
    min_count = int(cfg_cv["network"]["min_ingredient_count"])
    min_co_count = int(cfg_cv["backbone"]["min_co_count"])

    skf_outer = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=SEED)
    labels_all = [r["cuisine"] for r in train_raw]

    # Define canonical list of models to track across all 5 outer folds
    model_order = [
        # Reference models
        "Ref1: Fingerprint Scorer (original)",
        "Ref2: Fingerprint (normalized ablation)",
        "Ref3: Naive Bayes (full vocab)",
        "Ref4: Naive Bayes (backbone vocab)",
        # Experiment A: Unigram-only
        "ExpA1: Unigram Naive Bayes (backbone vocab)",
        "ExpA2: Unigram L2 LogReg (backbone vocab)",
        "ExpA3: Unigram L2 LogReg (full vocab)",
        # Experiment B: Pair-only
        "ExpB1_bb: Pair Log-Odds (backbone pairs)",
        "ExpB1_all: Pair Log-Odds (all pairs >= min_support)",
        "ExpB1_elig5: Pair Log-Odds (all eligible pairs co>=5)",
        "ExpB2_bb: Pair L2 LogReg (backbone pairs)",
        "ExpB2_all: Pair L2 LogReg (all pairs >= min_support)",
        # Experiment C: Hybrid (unigram + pair-presence in one L2 LogReg)
        "ExpC_bb: Hybrid L2 LogReg (backbone unigrams + backbone pairs)",
        "ExpC_full_bb: Hybrid L2 LogReg (full unigrams + backbone pairs)",
        "ExpC_full_all: Hybrid L2 LogReg (full unigrams + all pairs >= min_support)",
        # Experiment D: Original scorer with improved unigram NB fallback
        "ExpD_nb_full: Fingerprint Scorer + Unigram NB Fallback (full vocab)",
        "ExpD_nb_bb: Fingerprint Scorer + Unigram NB Fallback (backbone vocab)",
        # Experiment E: Matched-size pair-set comparison inside B2 (pair-only) and C (hybrid)
        "ExpE_B2_i_curveball: Pair L2 LogReg (D2 Curveball z>3 pairs, matched K)",
        "ExpE_B2_ii_npmi: Pair L2 LogReg (NPMI-thresholded pairs, matched K)",
        "ExpE_B2_iii_freq: Pair L2 LogReg (Top-K frequent pairs, matched K)",
        "ExpE_B2_iv_random: Pair L2 LogReg (Random eligible pairs, 10-seed mean)",
        "ExpE_C_i_curveball: Hybrid L2 LogReg (BB unigrams + Curveball z>3 pairs, matched K)",
        "ExpE_C_ii_npmi: Hybrid L2 LogReg (BB unigrams + NPMI-thresholded pairs, matched K)",
        "ExpE_C_iii_freq: Hybrid L2 LogReg (BB unigrams + Top-K frequent pairs, matched K)",
        "ExpE_C_iv_random: Hybrid L2 LogReg (BB unigrams + Random eligible pairs, 10-seed mean)",
        # Experiment F: Calibration & negative evidence on the original scorer
        "ExpF1_zscore_all: Fingerprint Scorer (per-cuisine z-score on train fold)",
        "ExpF1_zscore_pos: Fingerprint Scorer (per-cuisine z-score on positive train scores)",
        "ExpF2_neg_raw: Fingerprint Scorer + Negative Evidence (raw score - lambda * competing NPMI)",
        "ExpF2_neg_zscore: Fingerprint Scorer (z-score + Negative Evidence)",
        "ExpF3_edge_div: Fingerprint Scorer (divide by backbone edge count |E_c| ablation)",
    ]

    fold_metrics_by_model: dict[str, list[dict]] = {m: [] for m in model_order}
    oof_preds_by_model: dict[str, dict[int, str]] = {m: {} for m in model_order}
    oof_meta_by_id: dict[int, dict] = {}
    fold_reports: list[dict] = []

    # Track 10 random seeds separately for Experiment E (iv)
    random_b2_fold_accs: list[list[float]] = [[] for _ in range(N_RANDOM_SEEDS)]
    random_b2_fold_f1s: list[list[float]] = [[] for _ in range(N_RANDOM_SEEDS)]
    random_b2_oof_preds: list[dict[int, str]] = [{} for _ in range(N_RANDOM_SEEDS)]

    random_c_fold_accs: list[list[float]] = [[] for _ in range(N_RANDOM_SEEDS)]
    random_c_fold_f1s: list[list[float]] = [[] for _ in range(N_RANDOM_SEEDS)]
    random_c_oof_preds: list[dict[int, str]] = [{} for _ in range(N_RANDOM_SEEDS)]

    # Store per-fold fitted artifacts for error analysis (ExpC vs ExpA)
    error_analysis_records: dict[int, dict] = {}
    verification_checks: list[dict] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf_outer.split(train_raw, labels_all)):
        t_fold0 = time.perf_counter()
        fold_train_raw = [train_raw[i] for i in tr_idx]
        fold_val_raw = [train_raw[i] for i in val_idx]

        fold_train_ids = {r["id"] for r in fold_train_raw}
        fold_val_ids = {r["id"] for r in fold_val_raw}
        forbidden_ids = fold_val_ids | test_ids

        # Verification 1: Exact match with Phase 1 cv_folds.json
        p1_fold = phase1_folds_json["folds"][fold_idx]
        assert sha256_obj(sorted(fold_train_ids)) == p1_fold["train_ids_sha256"], f"Fold {fold_idx} train SHA mismatch"
        assert sha256_obj(sorted(fold_val_ids)) == p1_fold["val_ids_sha256"], f"Fold {fold_idx} val SHA mismatch"
        assert sorted(fold_val_ids) == sorted(p1_fold["val_recipe_ids"]), f"Fold {fold_idx} val IDs mismatch"
        assert len(fold_train_ids & fold_val_ids) == 0, f"Fold {fold_idx} train/val overlap"
        assert len(fold_train_ids & test_ids) == 0, f"Fold {fold_idx} train/test overlap"
        assert len(fold_val_ids & test_ids) == 0, f"Fold {fold_idx} val/test overlap"
        assert_training_only(fold_train_raw, fold_train_ids, forbidden_ids)

        # Normalize outer training and outer validation partitions independently
        fold_train_recipes = normalize_partition_records(fold_train_raw)
        fold_val_recipes = normalize_partition_records(fold_val_raw)

        y_train_str = [r["cuisine"] for r in fold_train_recipes]
        y_val_str = [r["cuisine"] for r in fold_val_recipes]
        y_train_idx = np.asarray([c_to_idx[c] for c in y_train_str], dtype=np.int64)
        y_val_idx = np.asarray([c_to_idx[c] for c in y_val_str], dtype=np.int64)
        val_recipe_ids = [r["id"] for r in fold_val_recipes]

        # Inner 3-fold StratifiedKFold on fold_train_recipes ONLY (for nested tuning)
        skf_inner = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=SEED + fold_idx)
        inner_splits = list(skf_inner.split(fold_train_recipes, y_train_idx))
        for itr_i, ival_i in inner_splits:
            itr_ids = {fold_train_recipes[idx]["id"] for idx in itr_i}
            ival_ids = {fold_train_recipes[idx]["id"] for idx in ival_i}
            assert len(itr_ids & ival_ids) == 0
            assert len(itr_ids & forbidden_ids) == 0
            assert len(ival_ids & forbidden_ids) == 0

        # Build fold networks and Curveball backbones on fold_train_recipes ONLY
        fold_nets = build_predictive_networks(
            fold_train_recipes, cuisines, min_count, fold_train_ids, forbidden_ids
        )
        fold_bb_res = build_backbones(
            fold_train_recipes,
            fold_nets,
            cfg_cv,
            scope="predictive",
            use_cache=True,
            dataset_sha=dataset_sha,
        )
        fold_bbs = fold_bb_res["backbone"]
        fold_class_counts = dict(sorted(Counter(y_train_str).items()))

        # Reference scorers
        scorer_primary = FingerprintScorer(fold_bbs, fold_class_counts, normalized=False)
        scorer_norm = FingerprintScorer(fold_bbs, fold_class_counts, normalized=True)
        priority_order = list(scorer_primary.priority)

        # Vocabularies derived strictly from fold_train_recipes
        full_unigram_vocab = sorted({x for r in fold_train_recipes for x in r["ingredients"] if x})
        backbone_unigram_vocab = sorted(set().union(*(scorer_primary.vocab[c] for c in cuisines)))
        assert len(full_unigram_vocab) == p1_fold["full_unigram_vocab_size"]
        assert len(backbone_unigram_vocab) == p1_fold["union_backbone_vocab_size"]

        # Extract candidate pair sets for Experiment E & backbone pairs for B/C
        pair_sets_info = extract_eligible_candidate_pairs_from_fold_nets(
            fold_nets, fold_bbs, fold_train_recipes, cuisines, min_co_count=min_co_count
        )
        backbone_pairs = pair_sets_info["backbone_pairs"]
        npmi_pairs = pair_sets_info["npmi_pairs"]
        freq_pairs = pair_sets_info["freq_pairs"]
        eligible_universe = pair_sets_info["eligible_universe"]
        k_target = pair_sets_info["k_target"]
        assert k_target == p1_fold["union_backbone_pairs_size"]
        assert len(backbone_pairs) == len(npmi_pairs) == len(freq_pairs) == k_target

        # Compute training-partition coverage quantiles (tertiles) from fold_train_recipes ONLY (Addendum #4)
        S_train_raw = compute_raw_scorer_matrix(scorer_primary, fold_train_recipes, cuisines)
        train_matched_sums = []
        for r in fold_train_recipes:
            ings = r["ingredients"]
            m_sum = 0
            if len(ings) >= 2:
                for pair in itertools.combinations(ings, 2):
                    for c in cuisines:
                        if pair in scorer_primary.edges[c]:
                            m_sum += 1
            train_matched_sums.append(m_sum)
        q1_train, q2_train = np.percentile(train_matched_sums, [33.3333, 66.6667])

        # Helper to record a model's fold metrics and OOF predictions
        def record_model(
            m_name: str,
            preds: list[str],
            zero_fb: int = 0,
            tied_fb: int = 0,
            extra_info: dict | None = None,
        ) -> dict:
            row, _ = compute_metrics_row(m_name, y_val_str, preds, cuisines, zero_fb, tied_fb)
            if extra_info:
                row.update(extra_info)
            fold_metrics_by_model[m_name].append(row)
            for rid, p in zip(val_recipe_ids, preds):
                oof_preds_by_model[m_name][rid] = p
            return row

        # =====================================================================
        # REFERENCE MODELS & EXPERIMENT D (Scorer + Unigram NB Fallback)
        # =====================================================================
        nb_full = BinaryMultinomialNB(
            cuisines,
            alpha=float(cfg_cv["naive_bayes"]["alpha"]),
            fit_prior=bool(cfg_cv["naive_bayes"]["fit_prior"]),
        ).fit(fold_train_recipes, fold_train_ids, forbidden_ids)
        preds_nb_full, nb_full_zfb, nb_full_tfb = nb_full.predict_with_fallback(fold_val_recipes)

        X_tr_uni_bb = build_unigram_csr(fold_train_recipes, backbone_unigram_vocab)
        X_val_uni_bb = build_unigram_csr(fold_val_recipes, backbone_unigram_vocab)
        preds_nb_bb, _ = fit_eval_nb_custom_vocab(
            X_tr_uni_bb,
            y_train_idx,
            X_val_uni_bb,
            cuisines,
            alpha=float(cfg_cv["naive_bayes"]["alpha"]),
            fit_prior=bool(cfg_cv["naive_bayes"]["fit_prior"]),
        )
        nb_bb_zfb = int(np.sum(np.diff(X_val_uni_bb.indptr) == 0))

        X_tr_uni_full = build_unigram_csr(fold_train_recipes, full_unigram_vocab)
        X_val_uni_full = build_unigram_csr(fold_val_recipes, full_unigram_vocab)

        preds_fp, fb_fp = [], []
        preds_norm, fb_norm = [], []
        preds_exp_d_full, preds_exp_d_bb = [], []

        for local_i, r in enumerate(fold_val_recipes):
            exp_prim = scorer_primary.explain(r["ingredients"])
            exp_norm = scorer_norm.explain(r["ingredients"])

            p_prim, f_prim = exp_prim["prediction"], exp_prim["fallback"]
            p_norm, f_norm = exp_norm["prediction"], exp_norm["fallback"]
            preds_fp.append(p_prim)
            fb_fp.append(f_prim)
            preds_norm.append(p_norm)
            fb_norm.append(f_norm)

            if f_prim == "zero_score":
                preds_exp_d_full.append(preds_nb_full[local_i])
                preds_exp_d_bb.append(preds_nb_bb[local_i])
            else:
                preds_exp_d_full.append(p_prim)
                preds_exp_d_bb.append(p_prim)

            n_matched_per_c = {c: exp_prim["per_cuisine"][c]["n_matched"] for c in cuisines}
            m_sum = sum(n_matched_per_c.values())
            matched_distinct = {
                tuple(m["pair"])
                for c in cuisines
                for m in exp_prim["per_cuisine"][c]["matched_pairs"]
            }
            oof_meta_by_id[r["id"]] = {
                "id": r["id"],
                "fold": fold_idx,
                "cuisine": r["cuisine"],
                "ingredients": exp_prim["normalized_input"],
                "n_ingredients": len(exp_prim["normalized_input"]),
                "n_matched_sum_across_cuisines": m_sum,
                "n_matched_distinct_backbone_pairs": len(matched_distinct),
                "n_matched_per_cuisine": n_matched_per_c,
                "primary_fallback": f_prim,
                "primary_scores": exp_prim["scores"],
                "coverage_bin_fixed": assign_coverage_bin_fixed(m_sum),
                "coverage_bin_tertile": assign_coverage_bin_tertile(m_sum, float(q1_train), float(q2_train)),
            }

        row_ref1 = record_model(
            "Ref1: Fingerprint Scorer (original)",
            preds_fp,
            sum(x == "zero_score" for x in fb_fp),
            sum(x == "tied_score" for x in fb_fp),
        )
        row_ref2 = record_model(
            "Ref2: Fingerprint (normalized ablation)",
            preds_norm,
            sum(x == "zero_score" for x in fb_norm),
            sum(x == "tied_score" for x in fb_norm),
        )
        row_ref3 = record_model(
            "Ref3: Naive Bayes (full vocab)",
            preds_nb_full,
            nb_full_zfb,
            nb_full_tfb,
        )
        record_model(
            "Ref4: Naive Bayes (backbone vocab)",
            preds_nb_bb,
            nb_bb_zfb,
            0,
        )
        # Verify exact match with Phase 1 reference fold metrics
        assert math.isclose(row_ref1["accuracy"], p1_fold["fold_metrics"]["Fingerprint Scorer"]["accuracy"], rel_tol=1e-12)
        assert math.isclose(row_ref2["accuracy"], p1_fold["fold_metrics"]["Fingerprint (normalized ablation)"]["accuracy"], rel_tol=1e-12)
        assert math.isclose(row_ref3["accuracy"], p1_fold["fold_metrics"]["Naive Bayes"]["accuracy"], rel_tol=1e-12)

        # Record Experiment D
        record_model("ExpD_nb_full: Fingerprint Scorer + Unigram NB Fallback (full vocab)", preds_exp_d_full, 0, 0)
        record_model("ExpD_nb_bb: Fingerprint Scorer + Unigram NB Fallback (backbone vocab)", preds_exp_d_bb, 0, 0)

        # =====================================================================
        # EXPERIMENT A: Unigram-Only Models (Backbone Vocab & Full Vocab)
        # =====================================================================
        record_model("ExpA1: Unigram Naive Bayes (backbone vocab)", preds_nb_bb, nb_bb_zfb, 0)

        clf_a2, best_c_a2, c_curve_a2 = tune_and_fit_l2_logreg(X_tr_uni_bb, y_train_idx, inner_splits)
        preds_a2 = [cuisines[int(i)] for i in clf_a2.predict(X_val_uni_bb)]
        probs_a2 = clf_a2.predict_proba(X_val_uni_bb)
        record_model("ExpA2: Unigram L2 LogReg (backbone vocab)", preds_a2, extra_info={"best_C": best_c_a2, "n_features": X_tr_uni_bb.shape[1]})

        clf_a3, best_c_a3, c_curve_a3 = tune_and_fit_l2_logreg(X_tr_uni_full, y_train_idx, inner_splits)
        preds_a3 = [cuisines[int(i)] for i in clf_a3.predict(X_val_uni_full)]
        probs_a3 = clf_a3.predict_proba(X_val_uni_full)
        record_model("ExpA3: Unigram L2 LogReg (full vocab)", preds_a3, extra_info={"best_C": best_c_a3, "n_features": X_tr_uni_full.shape[1]})

        # =====================================================================
        # EXPERIMENT B: Pair-Only Variants (B1 Log-Odds, B2 L2 LogReg, B3 Backbone vs All Pairs >= min_support)
        # =====================================================================
        X_tr_pair_bb = build_pair_csr(fold_train_recipes, backbone_pairs)
        X_val_pair_bb = build_pair_csr(fold_val_recipes, backbone_pairs)

        # B1 on backbone pairs (tune eps via inner CV)
        preds_b1_bb, b1_bb_zfb, b1_bb_tfb, best_eps_b1_bb, eps_curve_b1_bb, _ = tune_and_predict_b1_log_odds(
            X_tr_pair_bb, y_train_idx, X_val_pair_bb, inner_splits, cuisines, priority_order
        )
        record_model(
            "ExpB1_bb: Pair Log-Odds (backbone pairs)",
            preds_b1_bb,
            b1_bb_zfb,
            b1_bb_tfb,
            extra_info={"best_eps": best_eps_b1_bb, "n_features": len(backbone_pairs)},
        )

        # B3 / B1 on all eligible pairs with co_count >= 5 (matched to backbone support constraint)
        X_tr_pair_elig5 = build_pair_csr(fold_train_recipes, eligible_universe)
        X_val_pair_elig5 = build_pair_csr(fold_val_recipes, eligible_universe)
        preds_b1_elig5, b1_e5_zfb, b1_e5_tfb, best_eps_b1_e5, _, _ = tune_and_predict_b1_log_odds(
            X_tr_pair_elig5, y_train_idx, X_val_pair_elig5, inner_splits, cuisines, priority_order
        )
        record_model(
            "ExpB1_elig5: Pair Log-Odds (all eligible pairs co>=5)",
            preds_b1_elig5,
            b1_e5_zfb,
            b1_e5_tfb,
            extra_info={"best_eps": best_eps_b1_e5, "n_features": len(eligible_universe)},
        )

        # B3 / B1 & B2 on ALL pairs across full training vocab with min_support tuned via inner CV
        all_train_pair_counts = extract_all_pair_counts(fold_train_recipes)
        pair_matrices_by_sup: dict[int, tuple[list[tuple[str, str]], sp.csr_matrix, sp.csr_matrix]] = {}
        for sup in MIN_SUPPORT_GRID:
            p_vocab_sup = sorted(p for p, cnt in all_train_pair_counts.items() if cnt >= sup)
            X_tr_sup = build_pair_csr(fold_train_recipes, p_vocab_sup)
            X_val_sup = build_pair_csr(fold_val_recipes, p_vocab_sup)
            pair_matrices_by_sup[sup] = (p_vocab_sup, X_tr_sup, X_val_sup)

        # Joint inner CV tuning of (min_support, eps) for B1_all
        best_b1_all_score = -1.0
        best_b1_all_sup = MIN_SUPPORT_GRID[0]
        best_b1_all_eps = EPS_GRID[0]
        for sup in MIN_SUPPORT_GRID:
            _, X_tr_sup, _ = pair_matrices_by_sup[sup]
            for eps in EPS_GRID:
                accs = []
                for itr_idx, ival_idx in inner_splits:
                    W = compute_log_odds_weights(X_tr_sup[itr_idx], y_train_idx[itr_idx], len(cuisines), eps)
                    S_ival = X_tr_sup[ival_idx] @ W
                    p_ival, _, _ = predict_from_score_matrix(S_ival, cuisines, priority_order)
                    y_ival_str = [cuisines[int(i)] for i in y_train_idx[ival_idx]]
                    accs.append(float(accuracy_score(y_ival_str, p_ival)))
                m_acc = float(np.mean(accs))
                if round(m_acc, 8) > round(best_b1_all_score, 8):
                    best_b1_all_score = m_acc
                    best_b1_all_sup = sup
                    best_b1_all_eps = eps

        p_vocab_b1_all, X_tr_b1_all, X_val_b1_all = pair_matrices_by_sup[best_b1_all_sup]
        W_b1_all = compute_log_odds_weights(X_tr_b1_all, y_train_idx, len(cuisines), best_b1_all_eps)
        preds_b1_all, b1_all_zfb, b1_all_tfb = predict_from_score_matrix(
            X_val_b1_all @ W_b1_all, cuisines, priority_order
        )
        record_model(
            "ExpB1_all: Pair Log-Odds (all pairs >= min_support)",
            preds_b1_all,
            b1_all_zfb,
            b1_all_tfb,
            extra_info={
                "best_min_support": best_b1_all_sup,
                "best_eps": best_b1_all_eps,
                "n_features": len(p_vocab_b1_all),
            },
        )

        # B2 on backbone pairs (L2 LogReg)
        clf_b2_bb, best_c_b2_bb, _ = tune_and_fit_l2_logreg(X_tr_pair_bb, y_train_idx, inner_splits)
        preds_b2_bb = [cuisines[int(i)] for i in clf_b2_bb.predict(X_val_pair_bb)]
        b2_bb_zfb = int(np.sum(np.diff(X_val_pair_bb.indptr) == 0))
        record_model(
            "ExpB2_bb: Pair L2 LogReg (backbone pairs)",
            preds_b2_bb,
            b2_bb_zfb,
            0,
            extra_info={"best_C": best_c_b2_bb, "n_features": len(backbone_pairs)},
        )

        # B2 on all pairs >= min_support (inner CV over min_support and C)
        best_b2_all_acc = -1.0
        best_b2_all_sup = MIN_SUPPORT_GRID[0]
        best_b2_all_c = LR_C_GRID[0]
        best_clf_b2_all = None
        for sup in MIN_SUPPORT_GRID:
            _, X_tr_sup, _ = pair_matrices_by_sup[sup]
            clf_sup, c_sup, acc_map_sup = tune_and_fit_l2_logreg(X_tr_sup, y_train_idx, inner_splits)
            if round(acc_map_sup[c_sup], 8) > round(best_b2_all_acc, 8):
                best_b2_all_acc = acc_map_sup[c_sup]
                best_b2_all_sup = sup
                best_b2_all_c = c_sup
                best_clf_b2_all = clf_sup

        p_vocab_b2_all, _, X_val_b2_all = pair_matrices_by_sup[best_b2_all_sup]
        preds_b2_all = [cuisines[int(i)] for i in best_clf_b2_all.predict(X_val_b2_all)]
        b2_all_zfb = int(np.sum(np.diff(X_val_b2_all.indptr) == 0))
        record_model(
            "ExpB2_all: Pair L2 LogReg (all pairs >= min_support)",
            preds_b2_all,
            b2_all_zfb,
            0,
            extra_info={
                "best_min_support": best_b2_all_sup,
                "best_C": best_b2_all_c,
                "n_features": len(p_vocab_b2_all),
            },
        )

        # =====================================================================
        # EXPERIMENT C: HYBRID (Unigram + Pair-Presence in One L2 LogReg)
        # =====================================================================
        # C1: Backbone unigrams + Backbone pairs (paired directly with ExpA2)
        X_tr_hyb_bb = sp.hstack([X_tr_uni_bb, X_tr_pair_bb], format="csr")
        X_val_hyb_bb = sp.hstack([X_val_uni_bb, X_val_pair_bb], format="csr")
        clf_c_bb, best_c_hyb_bb, _ = tune_and_fit_l2_logreg(X_tr_hyb_bb, y_train_idx, inner_splits)
        preds_c_bb = [cuisines[int(i)] for i in clf_c_bb.predict(X_val_hyb_bb)]
        probs_c_bb = clf_c_bb.predict_proba(X_val_hyb_bb)
        record_model(
            "ExpC_bb: Hybrid L2 LogReg (backbone unigrams + backbone pairs)",
            preds_c_bb,
            extra_info={
                "best_C": best_c_hyb_bb,
                "n_unigram_features": X_tr_uni_bb.shape[1],
                "n_pair_features": X_tr_pair_bb.shape[1],
                "n_features": X_tr_hyb_bb.shape[1],
            },
        )

        # C2: Full unigrams + Backbone pairs (paired directly with ExpA3)
        X_tr_hyb_full_bb = sp.hstack([X_tr_uni_full, X_tr_pair_bb], format="csr")
        X_val_hyb_full_bb = sp.hstack([X_val_uni_full, X_val_pair_bb], format="csr")
        clf_c_full_bb, best_c_hyb_full_bb, _ = tune_and_fit_l2_logreg(X_tr_hyb_full_bb, y_train_idx, inner_splits)
        preds_c_full_bb = [cuisines[int(i)] for i in clf_c_full_bb.predict(X_val_hyb_full_bb)]
        probs_c_full_bb = clf_c_full_bb.predict_proba(X_val_hyb_full_bb)
        record_model(
            "ExpC_full_bb: Hybrid L2 LogReg (full unigrams + backbone pairs)",
            preds_c_full_bb,
            extra_info={
                "best_C": best_c_hyb_full_bb,
                "n_unigram_features": X_tr_uni_full.shape[1],
                "n_pair_features": X_tr_pair_bb.shape[1],
                "n_features": X_tr_hyb_full_bb.shape[1],
            },
        )

        # C3: Full unigrams + All pairs >= min_support (using inner-CV tuned min_support from B2_all)
        _, X_tr_p_all, X_val_p_all = pair_matrices_by_sup[best_b2_all_sup]
        X_tr_hyb_full_all = sp.hstack([X_tr_uni_full, X_tr_p_all], format="csr")
        X_val_hyb_full_all = sp.hstack([X_val_uni_full, X_val_p_all], format="csr")
        clf_c_full_all, best_c_hyb_full_all, _ = tune_and_fit_l2_logreg(X_tr_hyb_full_all, y_train_idx, inner_splits)
        preds_c_full_all = [cuisines[int(i)] for i in clf_c_full_all.predict(X_val_hyb_full_all)]
        record_model(
            "ExpC_full_all: Hybrid L2 LogReg (full unigrams + all pairs >= min_support)",
            preds_c_full_all,
            extra_info={
                "best_C": best_c_hyb_full_all,
                "best_min_support": best_b2_all_sup,
                "n_unigram_features": X_tr_uni_full.shape[1],
                "n_pair_features": X_tr_p_all.shape[1],
                "n_features": X_tr_hyb_full_all.shape[1],
            },
        )

        # Save detailed feature contributions for discordant recipes between Hybrid and Unigram-only
        n_uni_bb = len(backbone_unigram_vocab)
        coef_hyb_bb = clf_c_bb.coef_  # shape (3, n_uni_bb + k_target)
        coef_uni_bb = clf_a2.coef_    # shape (3, n_uni_bb)
        coef_hyb_full = clf_c_full_bb.coef_  # shape (3, n_uni_full + k_target)
        coef_uni_full = clf_a3.coef_         # shape (3, n_uni_full)
        n_uni_full = len(full_unigram_vocab)

        for local_i, r in enumerate(fold_val_recipes):
            rid = r["id"]
            p_uni_bb = preds_a2[local_i]
            p_hyb_bb = preds_c_bb[local_i]
            p_uni_full = preds_a3[local_i]
            p_hyb_full = preds_c_full_bb[local_i]

            if (p_uni_bb != p_hyb_bb) or (p_uni_full != p_hyb_full):
                # Active backbone pairs in this recipe
                row_pair_indices = X_val_pair_bb[local_i].indices
                active_bb_pairs = [backbone_pairs[idx] for idx in row_pair_indices]

                # Contribution of active pairs in ExpC_bb (backbone unigrams + backbone pairs)
                pair_contribs_bb = []
                total_pair_logit_bb = np.zeros(len(cuisines), dtype=np.float64)
                for p_col in row_pair_indices:
                    w_vec = coef_hyb_bb[:, n_uni_bb + p_col]
                    total_pair_logit_bb += w_vec
                    pair_contribs_bb.append({
                        "pair": list(backbone_pairs[p_col]),
                        "weights": {c: float(w_vec[ci]) for ci, c in enumerate(cuisines)},
                        "max_abs_weight": float(np.max(np.abs(w_vec))),
                    })
                pair_contribs_bb.sort(key=lambda d: -d["max_abs_weight"])

                # Contribution of active pairs in ExpC_full_bb (full unigrams + backbone pairs)
                pair_contribs_full = []
                total_pair_logit_full = np.zeros(len(cuisines), dtype=np.float64)
                for p_col in row_pair_indices:
                    w_vec = coef_hyb_full[:, n_uni_full + p_col]
                    total_pair_logit_full += w_vec
                    pair_contribs_full.append({
                        "pair": list(backbone_pairs[p_col]),
                        "weights": {c: float(w_vec[ci]) for ci, c in enumerate(cuisines)},
                        "max_abs_weight": float(np.max(np.abs(w_vec))),
                    })
                pair_contribs_full.sort(key=lambda d: -d["max_abs_weight"])

                # Top unigram contributions in ExpA2
                uni_bb_indices = X_val_uni_bb[local_i].indices
                uni_contribs_bb = []
                for u_col in uni_bb_indices:
                    w_vec = coef_uni_bb[:, u_col]
                    uni_contribs_bb.append({
                        "ingredient": backbone_unigram_vocab[u_col],
                        "weights": {c: float(w_vec[ci]) for ci, c in enumerate(cuisines)},
                        "max_abs_weight": float(np.max(np.abs(w_vec))),
                    })
                uni_contribs_bb.sort(key=lambda d: -d["max_abs_weight"])

                error_analysis_records[rid] = {
                    "id": rid,
                    "fold": fold_idx,
                    "true_cuisine": r["cuisine"],
                    "n_ingredients": len(r["ingredients"]),
                    "ingredients": r["ingredients"],
                    "n_active_backbone_pairs": len(active_bb_pairs),
                    "coverage_bin_fixed": oof_meta_by_id[rid]["coverage_bin_fixed"],
                    "pred_unigram_bb": p_uni_bb,
                    "pred_hybrid_bb": p_hyb_bb,
                    "probs_unigram_bb": {c: float(probs_a2[local_i, ci]) for ci, c in enumerate(cuisines)},
                    "probs_hybrid_bb": {c: float(probs_c_bb[local_i, ci]) for ci, c in enumerate(cuisines)},
                    "total_pair_logit_shift_bb": {c: float(total_pair_logit_bb[ci]) for ci, c in enumerate(cuisines)},
                    "top_pairs_bb": pair_contribs_bb[:6],
                    "top_unigrams_bb": uni_contribs_bb[:6],
                    "pred_unigram_full": p_uni_full,
                    "pred_hybrid_full": p_hyb_full,
                    "probs_unigram_full": {c: float(probs_a3[local_i, ci]) for ci, c in enumerate(cuisines)},
                    "probs_hybrid_full": {c: float(probs_c_full_bb[local_i, ci]) for ci, c in enumerate(cuisines)},
                    "total_pair_logit_shift_full": {c: float(total_pair_logit_full[ci]) for ci, c in enumerate(cuisines)},
                    "top_pairs_full": pair_contribs_full[:6],
                }

        # =====================================================================
        # EXPERIMENT E: Matched-Size Pair-Set Comparison (Inside B2 and C)
        # =====================================================================
        # (i) D2 Curveball backbone pairs (already computed above as ExpB2_bb and ExpC_bb)
        record_model(
            "ExpE_B2_i_curveball: Pair L2 LogReg (D2 Curveball z>3 pairs, matched K)",
            preds_b2_bb,
            b2_bb_zfb,
            0,
            extra_info={"best_C": best_c_b2_bb, "n_pair_features": k_target},
        )
        record_model(
            "ExpE_C_i_curveball: Hybrid L2 LogReg (BB unigrams + Curveball z>3 pairs, matched K)",
            preds_c_bb,
            0,
            0,
            extra_info={"best_C": best_c_hyb_bb, "n_pair_features": k_target, "n_unigram_features": n_uni_bb},
        )

        # (ii) NPMI-thresholded pairs of exact matched size K_f
        X_tr_pair_npmi = build_pair_csr(fold_train_recipes, npmi_pairs)
        X_val_pair_npmi = build_pair_csr(fold_val_recipes, npmi_pairs)
        clf_e_b2_npmi, best_c_e_b2_npmi, _ = tune_and_fit_l2_logreg(X_tr_pair_npmi, y_train_idx, inner_splits)
        preds_e_b2_npmi = [cuisines[int(i)] for i in clf_e_b2_npmi.predict(X_val_pair_npmi)]
        record_model(
            "ExpE_B2_ii_npmi: Pair L2 LogReg (NPMI-thresholded pairs, matched K)",
            preds_e_b2_npmi,
            int(np.sum(np.diff(X_val_pair_npmi.indptr) == 0)),
            0,
            extra_info={
                "best_C": best_c_e_b2_npmi,
                "n_pair_features": k_target,
                "npmi_cutoff": pair_sets_info["npmi_cutoff"],
            },
        )

        X_tr_hyb_npmi = sp.hstack([X_tr_uni_bb, X_tr_pair_npmi], format="csr")
        X_val_hyb_npmi = sp.hstack([X_val_uni_bb, X_val_pair_npmi], format="csr")
        clf_e_c_npmi, best_c_e_c_npmi, _ = tune_and_fit_l2_logreg(X_tr_hyb_npmi, y_train_idx, inner_splits)
        preds_e_c_npmi = [cuisines[int(i)] for i in clf_e_c_npmi.predict(X_val_hyb_npmi)]
        record_model(
            "ExpE_C_ii_npmi: Hybrid L2 LogReg (BB unigrams + NPMI-thresholded pairs, matched K)",
            preds_e_c_npmi,
            0,
            0,
            extra_info={
                "best_C": best_c_e_c_npmi,
                "n_pair_features": k_target,
                "npmi_cutoff": pair_sets_info["npmi_cutoff"],
            },
        )

        # (iii) Top-K most frequent pairs of exact matched size K_f
        X_tr_pair_freq = build_pair_csr(fold_train_recipes, freq_pairs)
        X_val_pair_freq = build_pair_csr(fold_val_recipes, freq_pairs)
        clf_e_b2_freq, best_c_e_b2_freq, _ = tune_and_fit_l2_logreg(X_tr_pair_freq, y_train_idx, inner_splits)
        preds_e_b2_freq = [cuisines[int(i)] for i in clf_e_b2_freq.predict(X_val_pair_freq)]
        record_model(
            "ExpE_B2_iii_freq: Pair L2 LogReg (Top-K frequent pairs, matched K)",
            preds_e_b2_freq,
            int(np.sum(np.diff(X_val_pair_freq.indptr) == 0)),
            0,
            extra_info={
                "best_C": best_c_e_b2_freq,
                "n_pair_features": k_target,
                "freq_cutoff": pair_sets_info["freq_cutoff"],
            },
        )

        X_tr_hyb_freq = sp.hstack([X_tr_uni_bb, X_tr_pair_freq], format="csr")
        X_val_hyb_freq = sp.hstack([X_val_uni_bb, X_val_pair_freq], format="csr")
        clf_e_c_freq, best_c_e_c_freq, _ = tune_and_fit_l2_logreg(X_tr_hyb_freq, y_train_idx, inner_splits)
        preds_e_c_freq = [cuisines[int(i)] for i in clf_e_c_freq.predict(X_val_hyb_freq)]
        record_model(
            "ExpE_C_iii_freq: Hybrid L2 LogReg (BB unigrams + Top-K frequent pairs, matched K)",
            preds_e_c_freq,
            0,
            0,
            extra_info={
                "best_C": best_c_e_c_freq,
                "n_pair_features": k_target,
                "freq_cutoff": pair_sets_info["freq_cutoff"],
            },
        )

        # (iv) Random eligible pairs of exact matched size K_f across 10 seeds
        # Note: X_tr_pair_elig5 and X_val_pair_elig5 already have columns for all eligible_universe pairs!
        # Subsetting columns by sampled indices is instantaneous and exact.
        n_elig_u = len(eligible_universe)
        seed_b2_probs_sum = np.zeros((len(fold_val_recipes), len(cuisines)), dtype=np.float64)
        seed_c_probs_sum = np.zeros((len(fold_val_recipes), len(cuisines)), dtype=np.float64)
        seed_b2_accs_fold, seed_b2_f1s_fold = [], []
        seed_c_accs_fold, seed_c_f1s_fold = [], []

        for s_i in range(N_RANDOM_SEEDS):
            rng_e = np.random.default_rng(SEED + 1000 * (s_i + 1) + fold_idx)
            sampled_cols = np.sort(rng_e.choice(n_elig_u, size=k_target, replace=False))
            assert len(sampled_cols) == k_target

            X_tr_rand = X_tr_pair_elig5[:, sampled_cols].tocsr()
            X_val_rand = X_val_pair_elig5[:, sampled_cols].tocsr()

            # Pair-only B2 on random pair sample s_i (inner CV tuned C)
            clf_rb2, c_rb2, _ = tune_and_fit_l2_logreg(X_tr_rand, y_train_idx, inner_splits)
            p_rb2 = [cuisines[int(i)] for i in clf_rb2.predict(X_val_rand)]
            seed_b2_probs_sum += clf_rb2.predict_proba(X_val_rand)
            acc_rb2 = float(accuracy_score(y_val_str, p_rb2))
            f1_rb2 = float(f1_score(y_val_str, p_rb2, labels=cuisines, average="macro", zero_division=0))
            random_b2_fold_accs[s_i].append(acc_rb2)
            random_b2_fold_f1s[s_i].append(f1_rb2)
            seed_b2_accs_fold.append(acc_rb2)
            seed_b2_f1s_fold.append(f1_rb2)
            for rid, pred_s in zip(val_recipe_ids, p_rb2):
                random_b2_oof_preds[s_i][rid] = pred_s

            # Hybrid C on random pair sample s_i (inner CV tuned C)
            X_tr_hyb_rand = sp.hstack([X_tr_uni_bb, X_tr_rand], format="csr")
            X_val_hyb_rand = sp.hstack([X_val_uni_bb, X_val_rand], format="csr")
            clf_rc, c_rc, _ = tune_and_fit_l2_logreg(X_tr_hyb_rand, y_train_idx, inner_splits)
            p_rc = [cuisines[int(i)] for i in clf_rc.predict(X_val_hyb_rand)]
            seed_c_probs_sum += clf_rc.predict_proba(X_val_hyb_rand)
            acc_rc = float(accuracy_score(y_val_str, p_rc))
            f1_rc = float(f1_score(y_val_str, p_rc, labels=cuisines, average="macro", zero_division=0))
            random_c_fold_accs[s_i].append(acc_rc)
            random_c_fold_f1s[s_i].append(f1_rc)
            seed_c_accs_fold.append(acc_rc)
            seed_c_f1s_fold.append(f1_rc)
            for rid, pred_s in zip(val_recipe_ids, p_rc):
                random_c_oof_preds[s_i][rid] = pred_s

        # Record seed-0 representative OOF predictions & overwrite accuracy/macro_f1 with exact 10-seed mean for fold summary
        preds_b2_rand_rep = [random_b2_oof_preds[0][rid] for rid in val_recipe_ids]
        row_eb2_rand = record_model(
            "ExpE_B2_iv_random: Pair L2 LogReg (Random eligible pairs, 10-seed mean)",
            preds_b2_rand_rep,
            extra_info={
                "n_pair_features": k_target,
                "seed_accuracy_mean": float(np.mean(seed_b2_accs_fold)),
                "seed_accuracy_std": float(np.std(seed_b2_accs_fold, ddof=1)),
                "seed_macro_f1_mean": float(np.mean(seed_b2_f1s_fold)),
                "seed_macro_f1_std": float(np.std(seed_b2_f1s_fold, ddof=1)),
            },
        )
        row_eb2_rand["accuracy"] = float(np.mean(seed_b2_accs_fold))
        row_eb2_rand["macro_f1"] = float(np.mean(seed_b2_f1s_fold))

        preds_c_rand_rep = [random_c_oof_preds[0][rid] for rid in val_recipe_ids]
        row_ec_rand = record_model(
            "ExpE_C_iv_random: Hybrid L2 LogReg (BB unigrams + Random eligible pairs, 10-seed mean)",
            preds_c_rand_rep,
            extra_info={
                "n_pair_features": k_target,
                "seed_accuracy_mean": float(np.mean(seed_c_accs_fold)),
                "seed_accuracy_std": float(np.std(seed_c_accs_fold, ddof=1)),
                "seed_macro_f1_mean": float(np.mean(seed_c_f1s_fold)),
                "seed_macro_f1_std": float(np.std(seed_c_f1s_fold, ddof=1)),
            },
        )
        row_ec_rand["accuracy"] = float(np.mean(seed_c_accs_fold))
        row_ec_rand["macro_f1"] = float(np.mean(seed_c_f1s_fold))

        # =====================================================================
        # EXPERIMENT F: Calibration of Original Scorer & Negative Evidence
        # =====================================================================
        S_val_raw = compute_raw_scorer_matrix(scorer_primary, fold_val_recipes, cuisines)

        # F1a: Per-cuisine z-score standardization using ALL outer-training-partition recipes (non-zero recipes standardized; zero-score fallback preserved)
        mu_all = S_train_raw.mean(axis=0)
        sd_all = S_train_raw.std(axis=0, ddof=1)
        sd_all = np.where(sd_all > 1e-12, sd_all, 1.0)
        S_val_z_all = (S_val_raw - mu_all) / sd_all
        # Preserve zero_score detection when all 3 raw scores are zero
        zero_mask_val = np.all(np.abs(S_val_raw) <= TIE_TOL, axis=1)
        S_val_z_all_masked = S_val_z_all.copy()
        S_val_z_all_masked[zero_mask_val] = 0.0
        preds_f1_all, f1_all_zfb, f1_all_tfb = predict_from_score_matrix(
            S_val_z_all_masked, cuisines, priority_order
        )
        record_model(
            "ExpF1_zscore_all: Fingerprint Scorer (per-cuisine z-score on train fold)",
            preds_f1_all,
            f1_all_zfb,
            f1_all_tfb,
            extra_info={
                "train_mu": {c: float(mu_all[ci]) for ci, c in enumerate(cuisines)},
                "train_sd": {c: float(sd_all[ci]) for ci, c in enumerate(cuisines)},
            },
        )

        # F1b: Per-cuisine z-score standardization using positive-score outer-training-partition distribution per cuisine
        mu_pos = np.zeros(len(cuisines), dtype=np.float64)
        sd_pos = np.ones(len(cuisines), dtype=np.float64)
        for ci in range(len(cuisines)):
            pos_vals = S_train_raw[S_train_raw[:, ci] > 0, ci]
            if len(pos_vals) > 1:
                mu_pos[ci] = float(np.mean(pos_vals))
                sd_pos[ci] = float(np.std(pos_vals, ddof=1))
        S_val_z_pos = np.where(S_val_raw > 0, (S_val_raw - mu_pos) / sd_pos, -1e6)
        S_val_z_pos[zero_mask_val] = 0.0
        preds_f1_pos, f1_pos_zfb, f1_pos_tfb = predict_from_score_matrix(
            S_val_z_pos, cuisines, priority_order
        )
        record_model(
            "ExpF1_zscore_pos: Fingerprint Scorer (per-cuisine z-score on positive train scores)",
            preds_f1_pos,
            f1_pos_zfb,
            f1_pos_tfb,
            extra_info={
                "train_pos_mu": {c: float(mu_pos[ci]) for ci, c in enumerate(cuisines)},
                "train_pos_sd": {c: float(sd_pos[ci]) for ci, c in enumerate(cuisines)},
            },
        )

        # F2: Negative evidence added (penalize cuisine c when pairs present in recipe belong to competing backbones c' != c)
        neg_all_tr, neg_excl_tr = compute_Other_cuisine_exclusive_npmi_matrix(
            scorer_primary, fold_train_recipes, cuisines
        )
        neg_all_val, neg_excl_val = compute_Other_cuisine_exclusive_npmi_matrix(
            scorer_primary, fold_val_recipes, cuisines
        )

        # Tune (neg_mode in {'excl', 'all'}, lambda in NEG_LAMBDA_GRID) via inner CV on raw scores
        best_f2_raw_acc = -1.0
        best_f2_raw_cfg = ("excl", 0.5)
        for neg_mode, N_tr in [("excl", neg_excl_tr), ("all", neg_all_tr)]:
            for lam in NEG_LAMBDA_GRID:
                accs = []
                for itr_idx, ival_idx in inner_splits:
                    S_ival = S_train_raw[ival_idx] - lam * N_tr[ival_idx]
                    z_mask_ival = np.all(np.abs(S_train_raw[ival_idx]) <= TIE_TOL, axis=1)
                    S_ival[z_mask_ival] = 0.0
                    p_ival, _, _ = predict_from_score_matrix(S_ival, cuisines, priority_order)
                    y_ival_str = [cuisines[int(i)] for i in y_train_idx[ival_idx]]
                    accs.append(float(accuracy_score(y_ival_str, p_ival)))
                m_acc = float(np.mean(accs))
                if round(m_acc, 8) > round(best_f2_raw_acc, 8):
                    best_f2_raw_acc = m_acc
                    best_f2_raw_cfg = (neg_mode, float(lam))

        N_val_chosen = neg_excl_val if best_f2_raw_cfg[0] == "excl" else neg_all_val
        S_val_f2_raw = S_val_raw - best_f2_raw_cfg[1] * N_val_chosen
        S_val_f2_raw[zero_mask_val] = 0.0
        preds_f2_raw, f2_raw_zfb, f2_raw_tfb = predict_from_score_matrix(
            S_val_f2_raw, cuisines, priority_order
        )
        record_model(
            "ExpF2_neg_raw: Fingerprint Scorer + Negative Evidence (raw score - lambda * competing NPMI)",
            preds_f2_raw,
            f2_raw_zfb,
            f2_raw_tfb,
            extra_info={"neg_mode": best_f2_raw_cfg[0], "best_lambda": best_f2_raw_cfg[1]},
        )

        # F2b: Z-score standardization + Negative evidence (standardized strictly inside each inner training split)
        best_f2_z_acc = -1.0
        best_f2_z_cfg = ("excl", 0.5)
        for neg_mode, N_tr in [("excl", neg_excl_tr), ("all", neg_all_tr)]:
            for lam in NEG_LAMBDA_GRID:
                accs = []
                for itr_idx, ival_idx in inner_splits:
                    D_itr = S_train_raw[itr_idx] - lam * N_tr[itr_idx]
                    mu_itr = D_itr.mean(axis=0)
                    sd_itr = np.where(D_itr.std(axis=0, ddof=1) > 1e-12, D_itr.std(axis=0, ddof=1), 1.0)
                    D_ival = (S_train_raw[ival_idx] - lam * N_tr[ival_idx] - mu_itr) / sd_itr
                    z_mask_ival = np.all(np.abs(S_train_raw[ival_idx]) <= TIE_TOL, axis=1)
                    D_ival[z_mask_ival] = 0.0
                    p_ival, _, _ = predict_from_score_matrix(D_ival, cuisines, priority_order)
                    y_ival_str = [cuisines[int(i)] for i in y_train_idx[ival_idx]]
                    accs.append(float(accuracy_score(y_ival_str, p_ival)))
                m_acc = float(np.mean(accs))
                if round(m_acc, 8) > round(best_f2_z_acc, 8):
                    best_f2_z_acc = m_acc
                    best_f2_z_cfg = (neg_mode, float(lam))

        N_tr_z = neg_excl_tr if best_f2_z_cfg[0] == "excl" else neg_all_tr
        N_val_z = neg_excl_val if best_f2_z_cfg[0] == "excl" else neg_all_val
        D_tr_full = S_train_raw - best_f2_z_cfg[1] * N_tr_z
        mu_d = D_tr_full.mean(axis=0)
        sd_d = np.where(D_tr_full.std(axis=0, ddof=1) > 1e-12, D_tr_full.std(axis=0, ddof=1), 1.0)
        S_val_f2_z = (S_val_raw - best_f2_z_cfg[1] * N_val_z - mu_d) / sd_d
        S_val_f2_z[zero_mask_val] = 0.0
        preds_f2_z, f2_z_zfb, f2_z_tfb = predict_from_score_matrix(
            S_val_f2_z, cuisines, priority_order
        )
        record_model(
            "ExpF2_neg_zscore: Fingerprint Scorer (z-score + Negative Evidence)",
            preds_f2_z,
            f2_z_zfb,
            f2_z_tfb,
            extra_info={"neg_mode": best_f2_z_cfg[0], "best_lambda": best_f2_z_cfg[1]},
        )

        # F3: Separate simple ablation — divide raw score by backbone edge count |E_c| (Addendum #3)
        edge_counts_vec = np.asarray([len(scorer_primary.edges[c]) for c in cuisines], dtype=np.float64)
        S_val_edge_div = S_val_raw / edge_counts_vec
        preds_f3_ediv, f3_zfb, f3_tfb = predict_from_score_matrix(
            S_val_edge_div, cuisines, priority_order
        )
        record_model(
            "ExpF3_edge_div: Fingerprint Scorer (divide by backbone edge count |E_c| ablation)",
            preds_f3_ediv,
            f3_zfb,
            f3_tfb,
            extra_info={"backbone_edge_counts": {c: int(edge_counts_vec[ci]) for ci, c in enumerate(cuisines)}},
        )

        fold_dt = time.perf_counter() - t_fold0
        verification_checks.append({
            "fold": fold_idx,
            "n_train": len(fold_train_recipes),
            "n_val": len(fold_val_recipes),
            "train_ids_sha256": sha256_obj(sorted(fold_train_ids)),
            "val_ids_sha256": sha256_obj(sorted(fold_val_ids)),
            "matches_phase1_cv_folds_json": True,
            "train_val_overlap": len(fold_train_ids & fold_val_ids),
            "train_test_overlap": len(fold_train_ids & test_ids),
            "val_test_overlap": len(fold_val_ids & test_ids),
            "inner_cv_folds": INNER_N_SPLITS,
            "inner_splits_disjoint_from_val_and_test": True,
        })

        fold_reports.append({
            "fold": fold_idx,
            "n_train": len(fold_train_recipes),
            "n_val": len(fold_val_recipes),
            "full_unigram_vocab_size": len(full_unigram_vocab),
            "backbone_unigram_vocab_size": len(backbone_unigram_vocab),
            "backbone_pairs_size_K": k_target,
            "eligible_pair_universe_co5_size": len(eligible_universe),
            "all_pairs_min_support_selected": {
                "B1_all_min_support": best_b1_all_sup,
                "B1_all_pair_count": len(p_vocab_b1_all),
                "B2_all_min_support": best_b2_all_sup,
                "B2_all_pair_count": len(p_vocab_b2_all),
            },
            "selected_hyperparameters": {
                "ExpA2_C": best_c_a2,
                "ExpA3_C": best_c_a3,
                "ExpB1_bb_eps": best_eps_b1_bb,
                "ExpB1_elig5_eps": best_eps_b1_e5,
                "ExpB1_all_min_support": best_b1_all_sup,
                "ExpB1_all_eps": best_b1_all_eps,
                "ExpB2_bb_C": best_c_b2_bb,
                "ExpB2_all_min_support": best_b2_all_sup,
                "ExpB2_all_C": best_b2_all_c,
                "ExpC_bb_C": best_c_hyb_bb,
                "ExpC_full_bb_C": best_c_hyb_full_bb,
                "ExpC_full_all_C": best_c_hyb_full_all,
                "ExpE_B2_npmi_C": best_c_e_b2_npmi,
                "ExpE_B2_freq_C": best_c_e_b2_freq,
                "ExpE_C_npmi_C": best_c_e_c_npmi,
                "ExpE_C_freq_C": best_c_e_c_freq,
                "ExpE_npmi_cutoff": pair_sets_info["npmi_cutoff"],
                "ExpE_freq_cutoff": pair_sets_info["freq_cutoff"],
                "ExpF2_neg_raw_mode": best_f2_raw_cfg[0],
                "ExpF2_neg_raw_lambda": best_f2_raw_cfg[1],
                "ExpF2_neg_zscore_mode": best_f2_z_cfg[0],
                "ExpF2_neg_zscore_lambda": best_f2_z_cfg[1],
            },
            "pair_set_overlaps_with_backbone": {
                "overlap_bb_npmi": pair_sets_info["overlap_bb_npmi"],
                "overlap_bb_freq": pair_sets_info["overlap_bb_freq"],
                "overlap_npmi_freq": pair_sets_info["overlap_npmi_freq"],
            },
            "calibration_stats_train_only": {
                "mu_all": {c: float(mu_all[ci]) for ci, c in enumerate(cuisines)},
                "sd_all": {c: float(sd_all[ci]) for ci, c in enumerate(cuisines)},
                "mu_pos": {c: float(mu_pos[ci]) for ci, c in enumerate(cuisines)},
                "sd_pos": {c: float(sd_pos[ci]) for ci, c in enumerate(cuisines)},
                "backbone_edge_counts": {c: int(edge_counts_vec[ci]) for ci, c in enumerate(cuisines)},
                "coverage_tertile_cutoffs_train": [float(q1_train), float(q2_train)],
            },
            "runtime_seconds": round(fold_dt, 3),
        })
        log.info(
            "Fold %d finished in %.2fs | A2(BB-LR)=%.4f C_bb(Hyb-BB)=%.4f | A3(Full-LR)=%.4f C_full(Hyb-Full)=%.4f | B1_bb=%.4f B2_bb=%.4f",
            fold_idx,
            fold_dt,
            fold_metrics_by_model["ExpA2: Unigram L2 LogReg (backbone vocab)"][-1]["accuracy"],
            fold_metrics_by_model["ExpC_bb: Hybrid L2 LogReg (backbone unigrams + backbone pairs)"][-1]["accuracy"],
            fold_metrics_by_model["ExpA3: Unigram L2 LogReg (full vocab)"][-1]["accuracy"],
            fold_metrics_by_model["ExpC_full_bb: Hybrid L2 LogReg (full unigrams + backbone pairs)"][-1]["accuracy"],
            fold_metrics_by_model["ExpB1_bb: Pair Log-Odds (backbone pairs)"][-1]["accuracy"],
            fold_metrics_by_model["ExpB2_bb: Pair L2 LogReg (backbone pairs)"][-1]["accuracy"],
        )

    total_runtime = round(time.perf_counter() - t0_all, 3)
    return {
        "cuisines": cuisines,
        "model_order": model_order,
        "fold_metrics_by_model": fold_metrics_by_model,
        "oof_preds_by_model": oof_preds_by_model,
        "oof_meta_by_id": oof_meta_by_id,
        "fold_reports": fold_reports,
        "random_e_stats": {
            "b2_fold_accs_per_seed": random_b2_fold_accs,
            "b2_fold_f1s_per_seed": random_b2_fold_f1s,
            "c_fold_accs_per_seed": random_c_fold_accs,
            "c_fold_f1s_per_seed": random_c_fold_f1s,
        },
        "error_analysis_records": error_analysis_records,
        "verification_checks": verification_checks,
        "dataset_sha256": dataset_sha,
        "total_runtime_seconds": total_runtime,
    }


def summarize_all_experiments(p2_data: dict) -> dict:
    """Compute results table, Experiment C paired statistics, coverage-stratified table,
    Experiment D fallback accuracy, Experiment E random-seed stats, error analysis, and write
    outputs/d4_experiments/results.csv and outputs/d4_experiments/phase2_report.md.
    """
    cuisines = p2_data["cuisines"]
    model_order = p2_data["model_order"]
    fold_metrics = p2_data["fold_metrics_by_model"]
    oof_preds = p2_data["oof_preds_by_model"]
    oof_meta = p2_data["oof_meta_by_id"]
    fold_reports = p2_data["fold_reports"]
    random_e = p2_data["random_e_stats"]
    err_records = p2_data["error_analysis_records"]

    ordered_ids = sorted(oof_meta.keys())
    assert len(ordered_ids) == 13823
    y_true = [oof_meta[i]["cuisine"] for i in ordered_ids]

    # 1. Build results.csv rows across all models
    results_rows = []
    model_summaries = {}

    for m_name in model_order:
        f_rows = fold_metrics[m_name]
        assert len(f_rows) == CV_N_SPLITS
        accs = [r["accuracy"] for r in f_rows]
        f1s = [r["macro_f1"] for r in f_rows]
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
        oof_row, cm_oof = compute_metrics_row(m_name, y_true, preds_m, cuisines, z_fb, t_fb)

        # For the 10-seed random control in ExpE, also compute across-seed mean/std of the 5-fold CV score
        extra_seed_note = ""
        if m_name == "ExpE_B2_iv_random: Pair L2 LogReg (Random eligible pairs, 10-seed mean)":
            seed_cv_accs = [float(np.mean(s_accs)) for s_accs in random_e["b2_fold_accs_per_seed"]]
            seed_cv_f1s = [float(np.mean(s_f1s)) for s_f1s in random_e["b2_fold_f1s_per_seed"]]
            oof_acc_val = float(np.mean(seed_cv_accs))
            oof_f1_val = float(np.mean(seed_cv_f1s))
            extra_seed_note = f"10-seed CV mean={oof_acc_val:.4f} (seed SD={float(np.std(seed_cv_accs, ddof=1)):.4f})"
        elif m_name == "ExpE_C_iv_random: Hybrid L2 LogReg (BB unigrams + Random eligible pairs, 10-seed mean)":
            seed_cv_accs = [float(np.mean(s_accs)) for s_accs in random_e["c_fold_accs_per_seed"]]
            seed_cv_f1s = [float(np.mean(s_f1s)) for s_f1s in random_e["c_fold_f1s_per_seed"]]
            oof_acc_val = float(np.mean(seed_cv_accs))
            oof_f1_val = float(np.mean(seed_cv_f1s))
            extra_seed_note = f"10-seed CV mean={oof_acc_val:.4f} (seed SD={float(np.std(seed_cv_accs, ddof=1)):.4f})"
        else:
            oof_acc_val = float(oof_row["accuracy"])
            oof_f1_val = float(oof_row["macro_f1"])

        row_dict = {
            "model": m_name,
            "cv_accuracy_mean": float(np.mean(accs)),
            "cv_accuracy_std": float(np.std(accs, ddof=1)),
            "cv_macro_f1_mean": float(np.mean(f1s)),
            "cv_macro_f1_std": float(np.std(f1s, ddof=1)),
            "oof_pooled_accuracy": oof_acc_val,
            "oof_pooled_macro_f1": oof_f1_val,
            "cv_f1_indian_mean": float(np.mean(f1_in)),
            "cv_f1_italian_mean": float(np.mean(f1_it)),
            "cv_f1_mexican_mean": float(np.mean(f1_mx)),
            "cv_precision_indian_mean": float(np.mean(p_in)),
            "cv_precision_italian_mean": float(np.mean(p_it)),
            "cv_precision_mexican_mean": float(np.mean(p_mx)),
            "cv_recall_indian_mean": float(np.mean(r_in)),
            "cv_recall_italian_mean": float(np.mean(r_it)),
            "cv_recall_mexican_mean": float(np.mean(r_mx)),
            "oof_samples": 13823,
            "zero_score_fallback_count": z_fb,
            "tied_score_fallback_count": t_fb,
            "fold_0_acc": float(accs[0]),
            "fold_1_acc": float(accs[1]),
            "fold_2_acc": float(accs[2]),
            "fold_3_acc": float(accs[3]),
            "fold_4_acc": float(accs[4]),
            "notes": extra_seed_note,
        }
        results_rows.append(row_dict)
        model_summaries[m_name] = {
            **row_dict,
            "fold_accuracies": accs,
            "fold_macro_f1s": f1s,
            "oof_confusion_matrix": cm_oof.tolist(),
        }

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(OUT_DIR / "results.csv", index=False, encoding="utf-8")

    # 2. Experiment C Detailed Paired Statistics (Addendum #5)
    # Compare:
    #  (a) ExpC_bb (Backbone Unigrams + Backbone Pairs) vs ExpA2 (Backbone Unigrams Only)
    #  (b) ExpC_full_bb (Full Unigrams + Backbone Pairs) vs ExpA3 (Full Unigrams Only)
    #  (c) ExpC_full_all (Full Unigrams + All Pairs >= min_support) vs ExpA3 (Full Unigrams Only)
    exp_c_comparisons = [
        (
            "Backbone Vocab: Hybrid (ExpC_bb) vs. Unigram-Only (ExpA2)",
            "ExpC_bb: Hybrid L2 LogReg (backbone unigrams + backbone pairs)",
            "ExpA2: Unigram L2 LogReg (backbone vocab)",
        ),
        (
            "Full Vocab + Backbone Pairs: Hybrid (ExpC_full_bb) vs. Unigram-Only (ExpA3)",
            "ExpC_full_bb: Hybrid L2 LogReg (full unigrams + backbone pairs)",
            "ExpA3: Unigram L2 LogReg (full vocab)",
        ),
        (
            "Full Vocab + All Pairs >= min_support: Hybrid (ExpC_full_all) vs. Unigram-Only (ExpA3)",
            "ExpC_full_all: Hybrid L2 LogReg (full unigrams + all pairs >= min_support)",
            "ExpA3: Unigram L2 LogReg (full vocab)",
        ),
    ]

    boot_models = {
        m: [oof_preds[m][i] for i in ordered_ids]
        for m in [
            "ExpA2: Unigram L2 LogReg (backbone vocab)",
            "ExpC_bb: Hybrid L2 LogReg (backbone unigrams + backbone pairs)",
            "ExpA3: Unigram L2 LogReg (full vocab)",
            "ExpC_full_bb: Hybrid L2 LogReg (full unigrams + backbone pairs)",
            "ExpC_full_all: Hybrid L2 LogReg (full unigrams + all pairs >= min_support)",
            "Ref3: Naive Bayes (full vocab)",
            "Ref4: Naive Bayes (backbone vocab)",
            "ExpB1_bb: Pair Log-Odds (backbone pairs)",
            "ExpB2_bb: Pair L2 LogReg (backbone pairs)",
            "ExpB2_all: Pair L2 LogReg (all pairs >= min_support)",
            "ExpD_nb_full: Fingerprint Scorer + Unigram NB Fallback (full vocab)",
            "ExpF2_neg_zscore: Fingerprint Scorer (z-score + Negative Evidence)",
        ]
    }
    boot_ci_all = bootstrap_ci(y_true, boot_models, cuisines, n_boot=2000, seed=SEED)

    exp_c_stats = {}
    for label, m_hyb, m_uni in exp_c_comparisons:
        preds_hyb = [oof_preds[m_hyb][i] for i in ordered_ids]
        preds_uni = [oof_preds[m_uni][i] for i in ordered_ids]
        mc = mcnemar_test(y_true, preds_hyb, preds_uni)

        # Pairwise difference key in boot_ci_all
        k_forward = f"{m_hyb} minus {m_uni}"
        k_reverse = f"{m_uni} minus {m_hyb}"
        if k_forward in boot_ci_all["pairwise_differences"]:
            b_diff = boot_ci_all["pairwise_differences"][k_forward]
            d_acc_ci = b_diff["delta_accuracy_ci95"]
            d_f1_ci = b_diff["delta_macro_f1_ci95"]
        else:
            b_diff = boot_ci_all["pairwise_differences"][k_reverse]
            d_acc_ci = [-b_diff["delta_accuracy_ci95"][1], -b_diff["delta_accuracy_ci95"][0]]
            d_f1_ci = [-b_diff["delta_macro_f1_ci95"][1], -b_diff["delta_macro_f1_ci95"][0]]

        fold_diffs = []
        for f_i in range(CV_N_SPLITS):
            r_hyb = fold_metrics[m_hyb][f_i]
            r_uni = fold_metrics[m_uni][f_i]
            fold_diffs.append({
                "fold": f_i,
                "hybrid_acc": float(r_hyb["accuracy"]),
                "unigram_acc": float(r_uni["accuracy"]),
                "delta_acc": float(r_hyb["accuracy"] - r_uni["accuracy"]),
                "hybrid_f1": float(r_hyb["macro_f1"]),
                "unigram_f1": float(r_uni["macro_f1"]),
                "delta_f1": float(r_hyb["macro_f1"] - r_uni["macro_f1"]),
            })

        pooled_hyb_acc = model_summaries[m_hyb]["oof_pooled_accuracy"]
        pooled_uni_acc = model_summaries[m_uni]["oof_pooled_accuracy"]
        pooled_hyb_f1 = model_summaries[m_hyb]["oof_pooled_macro_f1"]
        pooled_uni_f1 = model_summaries[m_uni]["oof_pooled_macro_f1"]

        exp_c_stats[label] = {
            "hybrid_model": m_hyb,
            "unigram_model": m_uni,
            "hybrid_cv_acc_mean": model_summaries[m_hyb]["cv_accuracy_mean"],
            "hybrid_cv_acc_std": model_summaries[m_hyb]["cv_accuracy_std"],
            "unigram_cv_acc_mean": model_summaries[m_uni]["cv_accuracy_mean"],
            "unigram_cv_acc_std": model_summaries[m_uni]["cv_accuracy_std"],
            "hybrid_cv_f1_mean": model_summaries[m_hyb]["cv_macro_f1_mean"],
            "hybrid_cv_f1_std": model_summaries[m_hyb]["cv_macro_f1_std"],
            "unigram_cv_f1_mean": model_summaries[m_uni]["cv_macro_f1_mean"],
            "unigram_cv_f1_std": model_summaries[m_uni]["cv_macro_f1_std"],
            "pooled_delta_acc": float(pooled_hyb_acc - pooled_uni_acc),
            "pooled_delta_f1": float(pooled_hyb_f1 - pooled_uni_f1),
            "bootstrap_delta_acc_ci95": d_acc_ci,
            "bootstrap_delta_f1_ci95": d_f1_ci,
            "mcnemar": mc,
            "fold_wise_differences": fold_diffs,
        }

    # 3. Coverage-Stratified Accuracy & Macro-F1 Table (Addendum #4)
    # Compute under BOTH pre-registered schemes:
    # Scheme 1: Fixed matched-pair boundaries: Low (0-4), Mid (5-15), High (16+)
    # Scheme 2: Training-fold tertiles (Tertile 1 Low, Tertile 2 Mid, Tertile 3 High)
    def compute_coverage_stratified_table(bin_key: str, bin_order: list[str]) -> list[dict]:
        rows = []
        for b_label in bin_order:
            sub_ids = [i for i in ordered_ids if oof_meta[i][bin_key] == b_label]
            n_sub = len(sub_ids)
            if n_sub == 0:
                continue
            yt_sub = [oof_meta[i]["cuisine"] for i in sub_ids]
            mean_ings = float(np.mean([oof_meta[i]["n_ingredients"] for i in sub_ids]))
            mean_pairs = float(np.mean([oof_meta[i]["n_matched_sum_across_cuisines"] for i in sub_ids]))

            p_uni_bb = [oof_preds["ExpA2: Unigram L2 LogReg (backbone vocab)"][i] for i in sub_ids]
            p_hyb_bb = [oof_preds["ExpC_bb: Hybrid L2 LogReg (backbone unigrams + backbone pairs)"][i] for i in sub_ids]
            p_uni_full = [oof_preds["ExpA3: Unigram L2 LogReg (full vocab)"][i] for i in sub_ids]
            p_hyb_full = [oof_preds["ExpC_full_bb: Hybrid L2 LogReg (full unigrams + backbone pairs)"][i] for i in sub_ids]
            p_pair_bb = [oof_preds["ExpB2_bb: Pair L2 LogReg (backbone pairs)"][i] for i in sub_ids]
            p_fp_orig = [oof_preds["Ref1: Fingerprint Scorer (original)"][i] for i in sub_ids]

            mc_bb = mcnemar_test(yt_sub, p_hyb_bb, p_uni_bb)
            mc_full = mcnemar_test(yt_sub, p_hyb_full, p_uni_full)

            acc_uni_bb = float(accuracy_score(yt_sub, p_uni_bb))
            acc_hyb_bb = float(accuracy_score(yt_sub, p_hyb_bb))
            f1_uni_bb = float(f1_score(yt_sub, p_uni_bb, labels=cuisines, average="macro", zero_division=0))
            f1_hyb_bb = float(f1_score(yt_sub, p_hyb_bb, labels=cuisines, average="macro", zero_division=0))

            acc_uni_full = float(accuracy_score(yt_sub, p_uni_full))
            acc_hyb_full = float(accuracy_score(yt_sub, p_hyb_full))
            f1_uni_full = float(f1_score(yt_sub, p_uni_full, labels=cuisines, average="macro", zero_division=0))
            f1_hyb_full = float(f1_score(yt_sub, p_hyb_full, labels=cuisines, average="macro", zero_division=0))

            rows.append({
                "coverage_bin": b_label,
                "n_recipes": n_sub,
                "pct_recipes": 100.0 * n_sub / len(ordered_ids),
                "mean_ingredients": mean_ings,
                "mean_matched_pairs_sum": mean_pairs,
                "acc_fp_original": float(accuracy_score(yt_sub, p_fp_orig)),
                "acc_pair_b2_bb": float(accuracy_score(yt_sub, p_pair_bb)),
                "acc_unigram_bb": acc_uni_bb,
                "acc_hybrid_bb": acc_hyb_bb,
                "delta_acc_bb": acc_hyb_bb - acc_uni_bb,
                "f1_unigram_bb": f1_uni_bb,
                "f1_hybrid_bb": f1_hyb_bb,
                "delta_f1_bb": f1_hyb_bb - f1_uni_bb,
                "hyb_right_uni_wrong_bb": mc_bb["a_right_b_wrong"],
                "hyb_wrong_uni_right_bb": mc_bb["a_wrong_b_right"],
                "mcnemar_p_bb": mc_bb["p_value_exact_binomial"],
                "acc_unigram_full": acc_uni_full,
                "acc_hybrid_full": acc_hyb_full,
                "delta_acc_full": acc_hyb_full - acc_uni_full,
                "f1_unigram_full": f1_uni_full,
                "f1_hybrid_full": f1_hyb_full,
                "delta_f1_full": f1_hyb_full - f1_uni_full,
                "hyb_right_uni_wrong_full": mc_full["a_right_b_wrong"],
                "hyb_wrong_uni_right_full": mc_full["a_wrong_b_right"],
                "mcnemar_p_full": mc_full["p_value_exact_binomial"],
            })
        return rows

    cov_fixed_rows = compute_coverage_stratified_table(
        "coverage_bin_fixed",
        ["Low (0–4 matched pairs)", "Mid (5–15 matched pairs)", "High (16+ matched pairs)"],
    )
    cov_tertile_rows = compute_coverage_stratified_table(
        "coverage_bin_tertile",
        ["Tertile 1 (Low)", "Tertile 2 (Mid)", "Tertile 3 (High)"],
    )

    # 4. Experiment D Fallback Subset Breakdown
    zero_ids = [i for i in ordered_ids if oof_meta[i]["primary_fallback"] == "zero_score"]
    non_zero_ids = [i for i in ordered_ids if oof_meta[i]["primary_fallback"] != "zero_score"]
    yt_zero = [oof_meta[i]["cuisine"] for i in zero_ids]
    yt_nonzero = [oof_meta[i]["cuisine"] for i in non_zero_ids]

    exp_d_subset_stats = {
        "n_zero_score_fallback": len(zero_ids),
        "pct_zero_score_fallback": 100.0 * len(zero_ids) / len(ordered_ids),
        "zero_subset_acc_majority_fallback": float(
            accuracy_score(yt_zero, [oof_preds["Ref1: Fingerprint Scorer (original)"][i] for i in zero_ids])
        ),
        "zero_subset_acc_nb_full_fallback": float(
            accuracy_score(yt_zero, [oof_preds["ExpD_nb_full: Fingerprint Scorer + Unigram NB Fallback (full vocab)"][i] for i in zero_ids])
        ),
        "zero_subset_acc_nb_bb_fallback": float(
            accuracy_score(yt_zero, [oof_preds["ExpD_nb_bb: Fingerprint Scorer + Unigram NB Fallback (backbone vocab)"][i] for i in zero_ids])
        ),
        "zero_subset_f1_majority_fallback": float(
            f1_score(yt_zero, [oof_preds["Ref1: Fingerprint Scorer (original)"][i] for i in zero_ids], labels=cuisines, average="macro", zero_division=0)
        ),
        "zero_subset_f1_nb_full_fallback": float(
            f1_score(yt_zero, [oof_preds["ExpD_nb_full: Fingerprint Scorer + Unigram NB Fallback (full vocab)"][i] for i in zero_ids], labels=cuisines, average="macro", zero_division=0)
        ),
        "zero_subset_f1_nb_bb_fallback": float(
            f1_score(yt_zero, [oof_preds["ExpD_nb_bb: Fingerprint Scorer + Unigram NB Fallback (backbone vocab)"][i] for i in zero_ids], labels=cuisines, average="macro", zero_division=0)
        ),
        "non_zero_subset_acc_scorer": float(
            accuracy_score(yt_nonzero, [oof_preds["Ref1: Fingerprint Scorer (original)"][i] for i in non_zero_ids])
        ),
    }

    # 5. Experiment E 10-Seed Random Control Summary
    b2_seed_cv_means = [float(np.mean(s_accs)) for s_accs in random_e["b2_fold_accs_per_seed"]]
    b2_seed_f1_means = [float(np.mean(s_f1s)) for s_f1s in random_e["b2_fold_f1s_per_seed"]]
    c_seed_cv_means = [float(np.mean(s_accs)) for s_accs in random_e["c_fold_accs_per_seed"]]
    c_seed_f1_means = [float(np.mean(s_f1s)) for s_f1s in random_e["c_fold_f1s_per_seed"]]

    exp_e_random_summary = {
        "n_seeds": N_RANDOM_SEEDS,
        "b2_random_10seed_cv_acc_mean": float(np.mean(b2_seed_cv_means)),
        "b2_random_10seed_cv_acc_std_across_seeds": float(np.std(b2_seed_cv_means, ddof=1)),
        "b2_random_10seed_cv_f1_mean": float(np.mean(b2_seed_f1_means)),
        "b2_random_10seed_cv_f1_std_across_seeds": float(np.std(b2_seed_f1_means, ddof=1)),
        "b2_per_seed_cv_accs": b2_seed_cv_means,
        "c_random_10seed_cv_acc_mean": float(np.mean(c_seed_cv_means)),
        "c_random_10seed_cv_acc_std_across_seeds": float(np.std(c_seed_cv_means, ddof=1)),
        "c_random_10seed_cv_f1_mean": float(np.mean(c_seed_f1_means)),
        "c_random_10seed_cv_f1_std_across_seeds": float(np.std(c_seed_f1_means, ddof=1)),
        "c_per_seed_cv_accs": c_seed_cv_means,
    }

    # 6. Error Analysis Selection: Recipes where Hybrid beats Unigram-only and vice versa
    hyb_beats_uni_bb = [
        r for r in err_records.values()
        if r["pred_hybrid_bb"] == r["true_cuisine"] and r["pred_unigram_bb"] != r["true_cuisine"]
    ]
    uni_beats_hyb_bb = [
        r for r in err_records.values()
        if r["pred_unigram_bb"] == r["true_cuisine"] and r["pred_hybrid_bb"] != r["true_cuisine"]
    ]
    # Sort by magnitude of pair logit shift on the predicted class
    hyb_beats_uni_bb.sort(
        key=lambda r: (
            -(r["total_pair_logit_shift_bb"][r["true_cuisine"]] - r["total_pair_logit_shift_bb"][r["pred_unigram_bb"]]),
            r["id"],
        )
    )
    uni_beats_hyb_bb.sort(
        key=lambda r: (
            -(r["total_pair_logit_shift_bb"][r["pred_hybrid_bb"]] - r["total_pair_logit_shift_bb"][r["true_cuisine"]]),
            r["id"],
        )
    )

    hyb_beats_uni_full = [
        r for r in err_records.values()
        if r["pred_hybrid_full"] == r["true_cuisine"] and r["pred_unigram_full"] != r["true_cuisine"]
    ]
    uni_beats_hyb_full = [
        r for r in err_records.values()
        if r["pred_unigram_full"] == r["true_cuisine"] and r["pred_hybrid_full"] != r["true_cuisine"]
    ]
    hyb_beats_uni_full.sort(
        key=lambda r: (
            -(r["total_pair_logit_shift_full"][r["true_cuisine"]] - r["total_pair_logit_shift_full"][r["pred_unigram_full"]]),
            r["id"],
        )
    )
    uni_beats_hyb_full.sort(
        key=lambda r: (
            -(r["total_pair_logit_shift_full"][r["pred_hybrid_full"]] - r["total_pair_logit_shift_full"][r["true_cuisine"]]),
            r["id"],
        )
    )

    summary_payload = {
        "seed": SEED,
        "n_outer_splits": CV_N_SPLITS,
        "n_inner_splits": INNER_N_SPLITS,
        "curveball_n_randomizations_cv": CV_N_RANDOMIZATIONS,
        "n_train_recipes_evaluated_oof": 13823,
        "n_test_recipes_untouched": 3456,
        "dataset_sha256": p2_data["dataset_sha256"],
        "normalizer_sha256": normalizer_fingerprint(),
        "total_runtime_seconds": p2_data["total_runtime_seconds"],
        "verification_checks": p2_data["verification_checks"],
        "fold_reports": fold_reports,
        "model_summaries": model_summaries,
        "experiment_c_paired_statistics": exp_c_stats,
        "coverage_stratified_fixed_bins": cov_fixed_rows,
        "coverage_stratified_tertile_bins": cov_tertile_rows,
        "experiment_d_fallback_subset": exp_d_subset_stats,
        "experiment_e_random_10seed_summary": exp_e_random_summary,
        "bootstrap_ci": boot_ci_all,
        "error_analysis_examples": {
            "backbone_vocab_hybrid_beats_unigram_count": len(hyb_beats_uni_bb),
            "backbone_vocab_unigram_beats_hybrid_count": len(uni_beats_hyb_bb),
            "full_vocab_hybrid_beats_unigram_count": len(hyb_beats_uni_full),
            "full_vocab_unigram_beats_hybrid_count": len(uni_beats_hyb_full),
            "top_hybrid_beats_unigram_bb": hyb_beats_uni_bb[:6],
            "top_unigram_beats_hybrid_bb": uni_beats_hyb_bb[:6],
            "top_hybrid_beats_unigram_full": hyb_beats_uni_full[:5],
            "top_unigram_beats_hybrid_full": uni_beats_hyb_full[:5],
        },
    }
    write_json(summary_payload, OUT_DIR / "phase2_summary.json")
    return summary_payload


def main() -> int:
    from src.d4_experiments.phase2_report_builder import format_phase2_report

    p2_data = run_phase2_nested_cv()
    summarize_all_experiments(p2_data)
    format_phase2_report()
    log.info(
        "Phase 2 nested CV complete in %.2fs. Results and report written to %s",
        p2_data["total_runtime_seconds"],
        OUT_DIR,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

