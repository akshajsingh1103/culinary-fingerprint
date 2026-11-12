"""Deliverable 4 evaluation & confusion-matrix acceptance tests."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.common import read_json
from src.evaluate import BinaryMultinomialNB
from src.fingerprint import FingerprintScorer
from src.preprocess import load_processed
from src.split_data import load_split


def test_d4_files_and_results_table(cfg, d4_dir):
    req = [
        "results.csv",
        "confusion_matrix.png",
        "confusion_matrix_ablation.png",
        "confusion_matrix_nb.png",
        "confusion_matrix_normalized.png",
        "confusion_matrix_normalized_ablation.png",
        "confusion_matrix_normalized_nb.png",
        "evaluation_metadata.json",
        "garam_masala_diagnostic.json",
    ]
    for f in req:
        p = d4_dir / f
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty {f}"

    df = pd.read_csv(d4_dir / "results.csv")
    expected_models = ["Fingerprint Scorer", "Fingerprint (normalized ablation)", "Naive Bayes"]
    assert list(df["model"]) == expected_models
    for col in [
        "model", "accuracy", "macro_f1",
        "precision_indian", "precision_italian", "precision_mexican",
        "recall_indian", "recall_italian", "recall_mexican",
        "f1_indian", "f1_italian", "f1_mexican",
        "test_samples", "zero_score_fallback_count", "tied_score_fallback_count",
    ]:
        assert col in df.columns

    assert (df["test_samples"] == 3456).all()
    assert ((df["accuracy"] > 0.0) & (df["accuracy"] <= 1.0)).all()
    assert ((df["macro_f1"] > 0.0) & (df["macro_f1"] <= 1.0)).all()


def test_d4_metadata_and_confusion_matrices(cfg, d4_dir):
    meta = read_json(d4_dir / "evaluation_metadata.json")
    assert meta["run_mode"] == "FULL"
    assert meta["n_randomizations"] == 1000
    assert len(meta["dataset_sha256"]) == 64
    assert "phase_runtimes_seconds" in meta and len(meta["phase_runtimes_seconds"]) >= 8

    cms = meta["confusion_matrices"]
    assert cms["cuisines_order"] == ["indian", "italian", "mexican"]
    for key in ["fingerprint_scorer", "fingerprint_normalized_ablation", "naive_bayes"]:
        mat = np.asarray(cms[key], dtype=int)
        assert mat.shape == (3, 3)
        assert int(mat.sum()) == meta["split"]["n_test"] == 3456


def _recompute_metrics_from_predictions(y_true: list[str], y_pred: list[str], cuisines: list[str]):
    idx = {c: i for i, c in enumerate(cuisines)}
    cm = np.zeros((len(cuisines), len(cuisines)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[idx[yt], idx[yp]] += 1
    total = int(cm.sum())
    acc = float(np.trace(cm) / total)
    prec, rec, f1 = {}, {}, {}
    for i, c in enumerate(cuisines):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        prec[c], rec[c], f1[c] = p, r, f
    macro_f1 = sum(f1.values()) / len(cuisines)
    return cm, acc, macro_f1, prec, rec, f1


def test_d4_deterministic_reproducibility_from_saved_training_artifacts(cfg, d4_dir):
    meta = read_json(d4_dir / "evaluation_metadata.json")
    results = pd.read_csv(d4_dir / "results.csv").set_index("model")
    recipes, _ = load_processed(cfg)
    split = load_split(cfg)
    train_ids = set(split["train_ids"])
    test_ids = set(split["test_ids"])
    train_recipes = [r for r in recipes if r["id"] in train_ids]
    test_recipes = [r for r in recipes if r["id"] in test_ids]
    assert [r["id"] for r in test_recipes] == split["test_ids"]
    y_true = [r["cuisine"] for r in test_recipes]
    cuisines = list(cfg["cuisines"])

    sc_primary = FingerprintScorer.from_artifacts(d4_dir / "training_artifacts", normalized=False)
    sc_norm = FingerprintScorer.from_artifacts(d4_dir / "training_artifacts", normalized=True)
    nb = BinaryMultinomialNB(
        cuisines,
        alpha=float(cfg["naive_bayes"]["alpha"]),
        fit_prior=bool(cfg["naive_bayes"]["fit_prior"]),
    ).fit(train_recipes, train_ids, test_ids)

    exp_primary = [sc_primary.explain(r["ingredients"]) for r in test_recipes]
    exp_norm = [sc_norm.explain(r["ingredients"]) for r in test_recipes]
    preds_primary = [e["prediction"] for e in exp_primary]
    preds_norm = [e["prediction"] for e in exp_norm]
    preds_nb, nb_zero_fb, nb_tied_fb = nb.predict_with_fallback(test_recipes)

    model_specs = [
        ("Fingerprint Scorer", "fingerprint_scorer", preds_primary, exp_primary),
        ("Fingerprint (normalized ablation)", "fingerprint_normalized_ablation", preds_norm, exp_norm),
        ("Naive Bayes", "naive_bayes", preds_nb, None),
    ]
    for model_name, cm_key, preds, explanations in model_specs:
        cm, acc, macro_f1, prec, rec, f1 = _recompute_metrics_from_predictions(y_true, preds, cuisines)
        expected_cm = np.asarray(meta["confusion_matrices"][cm_key], dtype=int)
        assert np.array_equal(cm, expected_cm), f"{model_name}: confusion matrix mismatch"

        row = results.loc[model_name]
        assert math.isclose(acc, float(row["accuracy"]), rel_tol=1e-12)
        assert math.isclose(macro_f1, float(row["macro_f1"]), rel_tol=1e-12)
        for c in cuisines:
            assert math.isclose(prec[c], float(row[f"precision_{c}"]), rel_tol=1e-12)
            assert math.isclose(rec[c], float(row[f"recall_{c}"]), rel_tol=1e-12)
            assert math.isclose(f1[c], float(row[f"f1_{c}"]), rel_tol=1e-12)
        assert int(row["test_samples"]) == len(y_true) == 3456
        if explanations is not None:
            zero_fb = sum(1 for e in explanations if e["fallback"] == "zero_score")
            tie_fb = sum(1 for e in explanations if e["fallback"] == "tied_score")
            assert int(row["zero_score_fallback_count"]) == zero_fb
            assert int(row["tied_score_fallback_count"]) == tie_fb
        else:
            assert int(row["zero_score_fallback_count"]) == nb_zero_fb == 0
            assert int(row["tied_score_fallback_count"]) == nb_tied_fb == 0

    assert meta["split"]["train_test_overlap"] == 0


def test_nb_standalone_normalization_and_fallback_reporting():
    cuisines = ["indian", "italian", "mexican"]
    train = [
        {"id": 1, "cuisine": "indian", "ingredients": ["Garam Masala", "Tomatoes"]},
        {"id": 2, "cuisine": "italian", "ingredients": ["Basil", "Olive Oil"]},
        {"id": 3, "cuisine": "italian", "ingredients": ["Garlic", "Olive Oil"]},
        {"id": 4, "cuisine": "mexican", "ingredients": ["Tortillas", "Scallions"]},
    ]
    nb = BinaryMultinomialNB(cuisines, alpha=1.0, fit_prior=True).fit(train, {1, 2, 3, 4}, {5, 6})
    # Raw unnormalized test inputs should normalize identically ("Tomatoes" -> "tomato", "Scallions" -> "green onion")
    assert "tomato" in nb.vocab_ and "green onion" in nb.vocab_
    preds, zero_fb, tied_fb = nb.predict_with_fallback([
        {"id": 5, "cuisine": "mexican", "ingredients": ["  TORTILLAS  ", "Green  Onions"]},
        {"id": 6, "cuisine": "indian", "ingredients": ["completely_unseen_ingredient_xyz"]},
    ])
    assert preds[0] == "mexican"
    assert preds[1] == "italian"  # prior fallback to majority training class
    assert zero_fb == 1
    assert tied_fb == 0
