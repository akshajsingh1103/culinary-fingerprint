"""Tests for Deliverable 4 Phase 1 & Phase 2 CV experiments (src/d4_experiments/ and outputs/d4_experiments/).

Verifies:
1. Test set isolation: all CV folds and Phase 2 experiments strictly cover only the 13,823 training recipes
   and have zero overlap with the 3,456 held-out test recipes.
2. Nested CV partition integrity: all 5 outer folds in Phase 2 match Phase 1 `cv_folds.json` SHA-256 hashes,
   and inner 3-fold splits are strictly disjoint from outer validation and held-out test partitions.
3. Unit tests for Phase 2 mathematical components:
   - Unigram and pair-presence CSR matrix construction (`build_unigram_csr`, `build_pair_csr`)
   - Log-odds pair scoring weights (`compute_log_odds_weights`)
   - Vectorized score-matrix argmax with tie and zero-score fallback handling (`predict_from_score_matrix`)
   - Coverage bin rules (`assign_coverage_bin_fixed`, `assign_coverage_bin_tertile`)
   - Paired McNemar test and paired bootstrap CI (`mcnemar_test`, `bootstrap_ci`)
4. Verification of saved Phase 2 outputs (`outputs/d4_experiments/results.csv`, `phase2_report.md`, `phase2_summary.json`):
   - Exact pairing of Reference models (`Ref1`, `Ref2`, `Ref3`) with Phase 1 fold metrics
   - Experiment E matched feature counts across all 4 pair sets and 10 random seeds
   - Experiment C paired statistical tests and coverage-stratified tables
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common import ROOT, read_json
from src.d4_experiments.phase1_diagnostics import bootstrap_ci, mcnemar_test
from src.d4_experiments.phase2_experiments import (
    assign_coverage_bin_fixed,
    assign_coverage_bin_tertile,
    build_pair_csr,
    build_unigram_csr,
    compute_log_odds_weights,
    predict_from_score_matrix,
    tune_and_fit_l2_logreg,
)
from src.split_data import load_split

EXP_DIR = ROOT / "outputs" / "d4_experiments"


def test_phase2_artifacts_exist_and_non_empty():
    required = [
        "cv_folds.json",
        "phase1_summary.json",
        "diagnostics.md",
        "results.csv",
        "phase2_summary.json",
        "phase2_report.md",
    ]
    for fname in required:
        p = EXP_DIR / fname
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty Phase 1/2 artifact: {p}"


def test_phase2_zero_test_leakage_and_outer_fold_pairing(cfg):
    split = load_split(cfg)
    train_ids = {int(x) for x in split["train_ids"]}
    test_ids = {int(x) for x in split["test_ids"]}
    assert len(train_ids) == 13823
    assert len(test_ids) == 3456
    assert len(train_ids & test_ids) == 0

    cv_folds = read_json(EXP_DIR / "cv_folds.json")
    p2_summary = read_json(EXP_DIR / "phase2_summary.json")

    assert cv_folds["n_train_recipes_total"] == 13823
    assert p2_summary["n_train_recipes_evaluated_oof"] == 13823
    assert p2_summary["n_test_recipes_untouched"] == 3456

    all_val_ids = []
    for fold_idx in range(5):
        f1 = cv_folds["folds"][fold_idx]
        v2 = p2_summary["verification_checks"][fold_idx]
        assert f1["fold"] == v2["fold"] == fold_idx
        assert f1["train_ids_sha256"] == v2["train_ids_sha256"]
        assert f1["val_ids_sha256"] == v2["val_ids_sha256"]
        assert v2["train_val_overlap"] == 0
        assert v2["train_test_overlap"] == 0
        assert v2["val_test_overlap"] == 0
        assert v2["inner_splits_disjoint_from_val_and_test"] is True

        fold_val_set = {int(i) for i in f1["val_recipe_ids"]}
        assert len(fold_val_set & test_ids) == 0, f"Test ID leaked into fold {fold_idx}"
        assert fold_val_set <= train_ids
        all_val_ids.extend(f1["val_recipe_ids"])

    # Verify the 5 outer validation folds form an exact partition of the 13,823 training recipes
    assert len(all_val_ids) == 13823
    assert set(all_val_ids) == train_ids


def test_phase2_results_csv_and_reference_model_reproduction():
    df = pd.read_csv(EXP_DIR / "results.csv")
    assert len(df) == 30
    assert (df["oof_samples"] == 13823).all()
    assert ((df["cv_accuracy_mean"] > 0.4) & (df["cv_accuracy_mean"] <= 1.0)).all()
    assert ((df["cv_macro_f1_mean"] > 0.4) & (df["cv_macro_f1_mean"] <= 1.0)).all()

    p1_summary = read_json(EXP_DIR / "phase1_summary.json")["cv_summary"]
    by_model = df.set_index("model")

    # Verify exact reproduction of Phase 1 reference baselines on the shared 5 folds
    ref_map = [
        ("Ref1: Fingerprint Scorer (original)", "Fingerprint Scorer"),
        ("Ref2: Fingerprint (normalized ablation)", "Fingerprint (normalized ablation)"),
        ("Ref3: Naive Bayes (full vocab)", "Naive Bayes"),
    ]
    for p2_name, p1_name in ref_map:
        row = by_model.loc[p2_name]
        p1 = p1_summary[p1_name]
        assert math.isclose(float(row["cv_accuracy_mean"]), float(p1["accuracy_mean"]), rel_tol=1e-12)
        assert math.isclose(float(row["cv_accuracy_std"]), float(p1["accuracy_std"]), rel_tol=1e-12)
        assert math.isclose(float(row["cv_macro_f1_mean"]), float(p1["macro_f1_mean"]), rel_tol=1e-12)
        assert math.isclose(float(row["oof_pooled_accuracy"]), float(p1["oof_pooled"]["accuracy"]), rel_tol=1e-12)


def test_phase2_experiment_e_matched_feature_counts_and_random_seeds():
    p2_summary = read_json(EXP_DIR / "phase2_summary.json")
    expected_k_per_fold = [4865, 4859, 4909, 4823, 4867]
    for fold_idx, fr in enumerate(p2_summary["fold_reports"]):
        assert fr["backbone_pairs_size_K"] == expected_k_per_fold[fold_idx]
        assert fr["eligible_pair_universe_co5_size"] > fr["backbone_pairs_size_K"]

    rand_e = p2_summary["experiment_e_random_10seed_summary"]
    assert rand_e["n_seeds"] == 10
    assert len(rand_e["b2_per_seed_cv_accs"]) == 10
    assert len(rand_e["c_per_seed_cv_accs"]) == 10
    assert math.isclose(
        rand_e["b2_random_10seed_cv_acc_mean"],
        float(np.mean(rand_e["b2_per_seed_cv_accs"])),
        rel_tol=1e-12,
    )
    assert math.isclose(
        rand_e["c_random_10seed_cv_acc_mean"],
        float(np.mean(rand_e["c_per_seed_cv_accs"])),
        rel_tol=1e-12,
    )


def test_phase2_experiment_c_statistics_and_coverage_bins():
    p2_summary = read_json(EXP_DIR / "phase2_summary.json")
    exp_c = p2_summary["experiment_c_paired_statistics"]
    assert len(exp_c) == 3
    for comp_name, st in exp_c.items():
        mc = st["mcnemar"]
        assert mc["n_total"] == 13823
        assert mc["both_correct"] + mc["a_right_b_wrong"] + mc["a_wrong_b_right"] + mc["both_wrong"] == 13823
        assert mc["n_discordant"] == mc["a_right_b_wrong"] + mc["a_wrong_b_right"]
        assert mc["net_difference_a_minus_b"] == mc["a_right_b_wrong"] - mc["a_wrong_b_right"]
        assert len(st["fold_wise_differences"]) == 5
        assert st["bootstrap_delta_acc_ci95"][0] <= st["pooled_delta_acc"] <= st["bootstrap_delta_acc_ci95"][1]

    # Verify coverage bins partition all 13,823 OOF training recipes under both fixed and tertile rules
    cov_fixed = p2_summary["coverage_stratified_fixed_bins"]
    cov_tertile = p2_summary["coverage_stratified_tertile_bins"]
    assert sum(r["n_recipes"] for r in cov_fixed) == 13823
    assert sum(r["n_recipes"] for r in cov_tertile) == 13823


def test_unit_unigram_and_pair_csr_and_log_odds():
    recipes = [
        {"id": 1, "cuisine": "indian", "ingredients": ["cumin", "garam masala", "turmeric"]},
        {"id": 2, "cuisine": "indian", "ingredients": ["cumin", "garam masala"]},
        {"id": 3, "cuisine": "italian", "ingredients": ["basil", "garlic", "olive oil"]},
        {"id": 4, "cuisine": "mexican", "ingredients": ["cumin", "jalapeno", "lime"]},
    ]
    u_vocab = ["basil", "cumin", "garam masala", "garlic", "jalapeno", "lime", "olive oil", "turmeric"]
    X_u = build_unigram_csr(recipes, u_vocab)
    assert X_u.shape == (4, 8)
    assert X_u.nnz == 11

    p_vocab = [("basil", "garlic"), ("cumin", "garam masala"), ("cumin", "jalapeno")]
    X_p = build_pair_csr(recipes, p_vocab)
    assert X_p.shape == (4, 3)
    # Recipe 1 & 2 match ("cumin", "garam masala"); Recipe 3 matches ("basil", "garlic"); Recipe 4 matches ("cumin", "jalapeno")
    expected_dense = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.array_equal(X_p.toarray(), expected_dense)

    y_idx = np.array([0, 0, 1, 2], dtype=np.int64)
    eps = 1e-3
    W = compute_log_odds_weights(X_p, y_idx, n_classes=3, eps=eps)
    assert W.shape == (3, 3)
    # Pair 1 ("cumin", "garam masala") occurs in 2/2 Indian recipes (P=1.0) and 0/2 non-Indian recipes (P=0.0)
    expected_w_in = math.log((1.0 + eps) / (0.0 + eps))
    expected_w_it = math.log((0.0 + eps) / ((2.0 / 3.0) + eps))
    assert math.isclose(W[1, 0], expected_w_in, rel_tol=1e-12)
    assert math.isclose(W[1, 1], expected_w_it, rel_tol=1e-12)


def test_unit_predict_from_score_matrix_and_coverage_bins():
    cuisines = ["indian", "italian", "mexican"]
    priority = ["italian", "mexican", "indian"]
    scores = np.array([
        [0.8, 0.2, 0.1],   # unique winner: indian
        [0.0, 0.0, 0.0],   # zero_score fallback -> italian (or custom fallback)
        [0.5, 0.5, 0.1],   # tied_score between indian & italian -> italian by priority
    ])
    preds, z_fb, t_fb = predict_from_score_matrix(scores, cuisines, priority)
    assert preds == ["indian", "italian", "italian"]
    assert z_fb == 1
    assert t_fb == 1

    # Custom fallback on zero_score
    preds_custom, z_fb2, t_fb2 = predict_from_score_matrix(
        scores, cuisines, priority, fallback_preds=["mexican", "mexican", "mexican"]
    )
    assert preds_custom == ["indian", "mexican", "italian"]
    assert z_fb2 == 1 and t_fb2 == 1

    assert assign_coverage_bin_fixed(0) == "Low (0–4 matched pairs)"
    assert assign_coverage_bin_fixed(4) == "Low (0–4 matched pairs)"
    assert assign_coverage_bin_fixed(5) == "Mid (5–15 matched pairs)"
    assert assign_coverage_bin_fixed(15) == "Mid (5–15 matched pairs)"
    assert assign_coverage_bin_fixed(16) == "High (16+ matched pairs)"

    assert assign_coverage_bin_tertile(8, 9.0, 22.0) == "Tertile 1 (Low)"
    assert assign_coverage_bin_tertile(15, 9.0, 22.0) == "Tertile 2 (Mid)"
    assert assign_coverage_bin_tertile(25, 9.0, 22.0) == "Tertile 3 (High)"


def test_unit_mcnemar_and_inner_logreg_tuner():
    y_true = ["indian", "italian", "mexican", "indian", "italian", "mexican"]
    pred_a = ["indian", "italian", "mexican", "italian", "italian", "mexican"]  # 5/6 correct
    pred_b = ["indian", "mexican", "mexican", "indian", "indian", "mexican"]    # 4/6 correct
    mc = mcnemar_test(y_true, pred_a, pred_b)
    assert mc["n_total"] == 6
    assert mc["both_correct"] == 3
    assert mc["a_right_b_wrong"] == 2
    assert mc["a_wrong_b_right"] == 1
    assert mc["both_wrong"] == 0
    assert mc["net_difference_a_minus_b"] == 1

    boot = bootstrap_ci(y_true, {"A": pred_a, "B": pred_b}, ["indian", "italian", "mexican"], n_boot=50, seed=42)
    assert "A minus B" in boot["pairwise_differences"]

    rng = np.random.default_rng(42)
    X_syn = sp.csr_matrix(rng.integers(0, 2, size=(30, 10)).astype(np.float64))
    y_syn = np.array([0, 1, 2] * 10, dtype=np.int64)
    inner_splits = [
        (np.arange(0, 20), np.arange(20, 30)),
        (np.concatenate([np.arange(0, 10), np.arange(20, 30)]), np.arange(10, 20)),
    ]
    clf, best_c, c_map = tune_and_fit_l2_logreg(X_syn, y_syn, inner_splits, c_grid=[0.1, 1.0])
    assert best_c in (0.1, 1.0)
    assert set(c_map.keys()) == {0.1, 1.0}
    assert clf.predict(X_syn).shape == (30,)
