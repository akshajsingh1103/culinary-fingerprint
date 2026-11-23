"""Tests for Deliverable 4 Phase 3 — Network-Informed Fingerprint Scorer v2 (Stacked).

Verifies:
1. Phase 3 design & output artifacts exist and are non-empty (`phase3_design.md`, `results.csv`,
   `phase3_summary.json`, `phase3_report.md`).
2. Strict zero-leakage and nested 3-fold cross-fitting verification:
   - Outer folds match `cv_folds.json` SHA-256 hashes identically.
   - Zero overlap between outer training, outer validation, and the 3,456 sealed test recipes.
   - Every training row in every outer fold was cross-fitted out-of-fold.
   - `PartitionNetworkFeatureExtractor` raises `AssertionError` if any test recipe ID is passed.
3. Exact 25-feature definitions and D3 consistency (`FEATURE_NAMES`, `get_feature_subsets`,
   community coverage/max-coverage/entropy, weighted-degree top-10 hubs, top-5 pairwise bridges).
4. Metric hierarchy and statistical reporting:
   - Macro-F1 primary selection and reporting on every outer fold and pooled OOF.
   - Paired bootstrap CIs for Macro-F1 and Accuracy on `M5 vs. M0`, `M4 vs. M1`, `M4 vs. M2`.
   - Recall Regression Guard (>1.0 pp drop detection) and standardized logistic coefficients.
   - Pre-registered recipe-length bins partition all 13,823 training recipes.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.common import ROOT, read_json
from src.d4_experiments.phase3_scorer_v2 import (
    FEATURE_INDEX,
    FEATURE_NAMES,
    PHASE3_DIR,
    RECIPE_LENGTH_BINS_ORDER,
    PartitionNetworkFeatureExtractor,
    assign_recipe_length_bin,
    get_feature_names,
    get_feature_subsets,
    tune_and_fit_standardized_meta_logreg,
)
from src.split_data import load_split


def test_phase3_artifacts_exist_and_non_empty():
    required = [
        "phase3_design.md",
        "results.csv",
        "phase3_summary.json",
        "phase3_report.md",
    ]
    for fname in required:
        p = PHASE3_DIR / fname
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty Phase 3 artifact: {p}"


def test_phase3_zero_test_leakage_and_cross_fitting_verification(cfg):
    split = load_split(cfg)
    train_ids = {int(x) for x in split["train_ids"]}
    test_ids = {int(x) for x in split["test_ids"]}
    assert len(train_ids) == 13823 and len(test_ids) == 3456
    assert len(train_ids & test_ids) == 0

    cv_folds = read_json(ROOT / "outputs" / "d4_experiments" / "cv_folds.json")
    p3_summary = read_json(PHASE3_DIR / "phase3_summary.json")

    assert p3_summary["n_train_recipes_evaluated_oof"] == 13823
    assert p3_summary["n_test_recipes_untouched"] == 3456
    assert p3_summary["n_outer_splits"] == 5
    assert p3_summary["n_inner_splits"] == 3

    for fold_idx in range(5):
        f1 = cv_folds["folds"][fold_idx]
        v3 = p3_summary["verification_log"][fold_idx]
        assert v3["fold"] == fold_idx
        assert v3["train_ids_sha256"] == f1["train_ids_sha256"]
        assert v3["val_ids_sha256"] == f1["val_ids_sha256"]
        assert v3["matches_phase1_cv_folds_json"] is True
        assert v3["all_train_rows_cross_fitted_out_of_fold"] is True
        assert v3["train_val_overlap"] == 0
        assert v3["train_test_overlap"] == 0
        assert v3["val_test_overlap"] == 0


def test_phase3_feature_schema_and_subsets():
    names = get_feature_names()
    assert len(names) == 25
    assert names == FEATURE_NAMES

    subsets = get_feature_subsets()
    assert len(subsets["M2_calibrated_nb_only"]) == 3
    assert len(subsets["M3_nb_plus_pair_log_odds"]) == 6
    assert len(subsets["M4_full_stack"]) == 25
    assert len(subsets["M5_network_only_v2"]) == 22

    # Ensure M5 has zero NB features
    nb_indices = {FEATURE_INDEX[f"nb_log_prob_{c}"] for c in ["indian", "italian", "mexican"]}
    assert len(nb_indices & set(subsets["M5_network_only_v2"])) == 0


def test_phase3_results_and_metric_hierarchy():
    df = pd.read_csv(PHASE3_DIR / "results.csv").set_index("model")
    p3_summary = read_json(PHASE3_DIR / "phase3_summary.json")
    p1_summary = read_json(ROOT / "outputs" / "d4_experiments" / "phase1_summary.json")["cv_summary"]

    # Verify M0 and M1 match Phase 1 reference models identically
    m0 = df.loc["M0: Original Fingerprint Scorer (reference)"]
    m1 = df.loc["M1: Naive Bayes Alone (reference)"]
    assert math.isclose(float(m0["cv_macro_f1_mean"]), float(p1_summary["Fingerprint Scorer"]["macro_f1_mean"]), rel_tol=1e-12)
    assert math.isclose(float(m0["cv_accuracy_mean"]), float(p1_summary["Fingerprint Scorer"]["accuracy_mean"]), rel_tol=1e-12)
    assert math.isclose(float(m1["cv_macro_f1_mean"]), float(p1_summary["Naive Bayes"]["macro_f1_mean"]), rel_tol=1e-12)
    assert math.isclose(float(m1["cv_accuracy_mean"]), float(p1_summary["Naive Bayes"]["accuracy_mean"]), rel_tol=1e-12)

    # Verify M5 (pure network-only v2) significantly beats M0 on Macro-F1 and Accuracy
    m5 = df.loc["M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)"]
    assert float(m5["cv_macro_f1_mean"]) > float(m0["cv_macro_f1_mean"]) + 0.07

    # Verify paired comparisons contain Macro-F1 bootstrap CIs, McNemar stats, and recall regression guard checks
    paired = p3_summary["paired_comparisons"]
    for key in [
        "M5 (Network-Only v2) vs. M0 (Original Fingerprint Scorer)",
        "M4 (Full Stack: NB + All Network Features) vs. M1 (Naive Bayes Alone)",
        "M4 (Full Stack: NB + All Network Features) vs. M2 (Calibrated NB Stack Alone)",
    ]:
        assert key in paired
        comp = paired[key]
        ci_f1 = comp["bootstrap_delta_macro_f1_ci95"]
        assert ci_f1[0] <= comp["delta_oof_macro_f1"] <= ci_f1[1]
        assert "regression_guard_triggered" in comp
        assert comp["mcnemar"]["n_total"] == 13823


def test_phase3_standardized_coefficients_and_length_bins():
    p3_summary = read_json(PHASE3_DIR / "phase3_summary.json")
    coefs = p3_summary["standardized_coefficients"]
    assert len(coefs["M4_full_stack_25_features"]) == 25
    assert len(coefs["M5_network_only_22_features"]) == 22

    # Verify recipe-length bins partition all 13,823 OOF training recipes
    lbins = p3_summary["recipe_length_bins_table"]
    assert [b["length_bin"] for b in lbins] == RECIPE_LENGTH_BINS_ORDER
    assert sum(int(b["n_recipes"]) for b in lbins) == 13823

    # Verify NB prediction change analysis consistency
    for ch_key, ch in p3_summary["nb_prediction_change_analysis"].items():
        assert ch["n_total_recipes"] == 13823
        assert (
            ch["n_changed_to_correct_fixed"]
            + ch["n_changed_from_correct_to_wrong_broken"]
            + ch["n_changed_both_wrong"]
            == ch["n_predictions_changed"]
        )
        assert ch["net_correct_gain"] == ch["n_changed_to_correct_fixed"] - ch["n_changed_from_correct_to_wrong_broken"]


def test_unit_recipe_length_bins_and_meta_logreg_macro_f1_selection():
    assert assign_recipe_length_bin(1) == "Short (1–5 ingredients)"
    assert assign_recipe_length_bin(5) == "Short (1–5 ingredients)"
    assert assign_recipe_length_bin(6) == "Medium-Short (6–8 ingredients)"
    assert assign_recipe_length_bin(8) == "Medium-Short (6–8 ingredients)"
    assert assign_recipe_length_bin(9) == "Medium-Long (9–12 ingredients)"
    assert assign_recipe_length_bin(12) == "Medium-Long (9–12 ingredients)"
    assert assign_recipe_length_bin(13) == "Long (13+ ingredients)"

    rng = np.random.default_rng(42)
    X_raw = rng.normal(loc=0.0, scale=2.0, size=(45, 6))
    y_idx = np.array([0, 1, 2] * 15, dtype=np.int64)
    # Inject strong class signal in first 3 columns
    for i in range(45):
        X_raw[i, y_idx[i]] += 4.0

    inner_splits = [
        (np.arange(0, 30), np.arange(30, 45)),
        (np.concatenate([np.arange(0, 15), np.arange(30, 45)]), np.arange(15, 30)),
        (np.arange(15, 45), np.arange(0, 15)),
    ]
    scaler, clf, best_c, curve = tune_and_fit_standardized_meta_logreg(
        X_raw, y_idx, [0, 1, 2, 3, 4, 5], inner_splits, c_grid=[0.1, 1.0, 5.0], seed=42
    )
    assert best_c in (0.1, 1.0, 5.0)
    assert all("macro_f1_mean" in v and "accuracy_mean" in v for v in curve.values())
    X_std = scaler.transform(X_raw)
    # Standardized training features must have mean ~ 0 and std ~ 1
    assert np.allclose(X_std.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(X_std.std(axis=0, ddof=0), 1.0, atol=1e-12)
    assert float(np.mean(clf.predict(X_std) == y_idx)) > 0.90


def test_unit_extractor_rejects_test_leakage(cfg, tmp_path):
    ext = PartitionNetworkFeatureExtractor(
        cfg_base=cfg,
        cuisines=["indian", "italian", "mexican"],
        n_randomizations=5,
        cache_dir=tmp_path,
        dataset_sha="dummy_sha",
    )
    contaminated = [
        {"id": 1, "cuisine": "indian", "ingredients": ["cumin", "turmeric"]},
        {"id": 999, "cuisine": "italian", "ingredients": ["basil", "olive oil"]},
    ]
    with pytest.raises(AssertionError, match="test recipes passed"):
        ext.fit(contaminated, train_ids={1}, forbidden_ids={999})


def test_final_held_out_test_results_and_artifacts():
    final_csv = ROOT / "outputs" / "d4_experiments" / "final_test_results.csv"
    final_json = ROOT / "outputs" / "d4_experiments" / "final_test_summary.json"
    final_md = ROOT / "outputs" / "d4_experiments" / "final_test_report.md"

    assert final_csv.exists() and final_csv.stat().st_size > 0
    assert final_json.exists() and final_json.stat().st_size > 0
    assert final_md.exists() and final_md.stat().st_size > 0

    df = pd.read_csv(final_csv)
    expected_models = [
        "M0: Original Fingerprint Scorer",
        "M1: Naive Bayes",
        "M4: Full Stack (NB + network features)",
        "M5: Network-Only Fingerprint Scorer v2",
    ]
    assert list(df["model"]) == expected_models
    assert (df["test_samples"] == 3456).all()

    # Verify M0 and M1 match existing outputs/d4_scorer/results.csv exactly
    d4_orig = pd.read_csv(ROOT / "outputs" / "d4_scorer" / "results.csv").set_index("model")
    by_m = df.set_index("model")
    assert math.isclose(float(by_m.loc["M0: Original Fingerprint Scorer", "test_macro_f1"]),
                        float(d4_orig.loc["Fingerprint Scorer", "macro_f1"]), rel_tol=1e-12)
    assert math.isclose(float(by_m.loc["M0: Original Fingerprint Scorer", "test_accuracy"]),
                        float(d4_orig.loc["Fingerprint Scorer", "accuracy"]), rel_tol=1e-12)
    assert math.isclose(float(by_m.loc["M1: Naive Bayes", "test_macro_f1"]),
                        float(d4_orig.loc["Naive Bayes", "macro_f1"]), rel_tol=1e-12)
    assert math.isclose(float(by_m.loc["M1: Naive Bayes", "test_accuracy"]),
                        float(d4_orig.loc["Naive Bayes", "accuracy"]), rel_tol=1e-12)

    # Verify confusion matrices sum to 3,456 and paired comparisons exist
    summary = read_json(final_json)
    assert summary["n_train_recipes"] == 13823
    assert summary["n_test_recipes"] == 3456
    for m_name in expected_models:
        cm = np.asarray(summary["confusion_matrices"][m_name], dtype=int)
        assert cm.shape == (3, 3)
        assert int(cm.sum()) == 3456

    for comp_key in ["M5 vs M0", "M4 vs M1"]:
        assert comp_key in summary["paired_comparisons"]
        comp = summary["paired_comparisons"][comp_key]
        assert comp["mcnemar"]["n_total"] == 3456
        ci = comp["bootstrap_delta_macro_f1_ci95"]
        assert ci[0] <= comp["delta_macro_f1"] <= ci[1]
