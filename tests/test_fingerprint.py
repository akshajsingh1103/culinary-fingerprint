"""Deliverable 4 Fingerprint Scorer tests (hand-constructed fixtures + real artifacts)."""
from __future__ import annotations

import inspect
import math

import pandas as pd

import src.fingerprint as fp_mod
from src.common import read_json
from src.fingerprint import FingerprintScorer, predict, predict_scores


def _hand_fixture_scorer(normalized: bool = False) -> FingerprintScorer:
    bb_indian = pd.DataFrame([
        {"ingredient_a": "cumin", "ingredient_b": "garam masala", "NPMI": 0.45},
        {"ingredient_a": "cumin seed", "ingredient_b": "ground turmeric", "NPMI": 0.35},
    ])
    bb_italian = pd.DataFrame([
        {"ingredient_a": "basil", "ingredient_b": "olive oil", "NPMI": 0.40},
        {"ingredient_a": "garlic", "ingredient_b": "olive oil", "NPMI": 0.30},
    ])
    bb_mexican = pd.DataFrame([
        {"ingredient_a": "cilantro", "ingredient_b": "lime", "NPMI": 0.50},
        {"ingredient_a": "cumin", "ingredient_b": "garlic", "NPMI": 0.25},
    ])
    counts = {"indian": 2403, "italian": 6270, "mexican": 5150}
    return FingerprintScorer(
        {"indian": bb_indian, "italian": bb_italian, "mexican": bb_mexican},
        counts,
        normalized=normalized,
    )


def test_hand_constructed_fixture_scoring_and_ablation():
    sc = _hand_fixture_scorer(normalized=False)
    sc_norm = _hand_fixture_scorer(normalized=True)

    # 2 matched pairs in italian (0.40 + 0.30 = 0.70), 1 matched pair in mexican (0.50), 0 in indian
    ing = ["basil", "olive oil", "garlic", "cilantro", "lime"]
    scores = sc.predict_scores(ing)
    assert math.isclose(scores["indian"], 0.0, abs_tol=1e-12)
    assert math.isclose(scores["italian"], 0.70, rel_tol=1e-12)
    assert math.isclose(scores["mexican"], 0.50, rel_tol=1e-12)
    assert sc.predict(ing) == "italian"

    # Under normalized ablation: italian = 0.70 / 2 = 0.35, mexican = 0.50 / 1 = 0.50 -> mexican wins!
    scores_n = sc_norm.predict_scores(ing)
    assert math.isclose(scores_n["italian"], 0.35, rel_tol=1e-12)
    assert math.isclose(scores_n["mexican"], 0.50, rel_tol=1e-12)
    assert sc_norm.predict(ing) == "mexican"


def test_input_ordering_duplicates_and_unknown_ingredients():
    sc = _hand_fixture_scorer(normalized=False)
    s1 = sc.predict_scores(["olive oil", "basil", "garlic"])
    s2 = sc.predict_scores(["GARLIC", "Basil", "olive oil", "basil", "unknown_dragonfruit_xyz"])
    assert s1 == s2
    assert sc.predict(["unknown_a", "unknown_b", "unknown_c"]) == "italian"  # most frequent training class


def test_fallback_policy_zero_and_nonzero_ties():
    bb_a = pd.DataFrame([{"ingredient_a": "x", "ingredient_b": "y", "NPMI": 0.50}])
    bb_b = pd.DataFrame([{"ingredient_a": "x", "ingredient_b": "y", "NPMI": 0.50}])
    bb_c = pd.DataFrame([{"ingredient_a": "x", "ingredient_b": "y", "NPMI": 0.20}])
    sc = FingerprintScorer(
        {"indian": bb_a, "italian": bb_b, "mexican": bb_c},
        {"indian": 100, "italian": 300, "mexican": 200},
    )
    # Zero-score tie -> fallback to italian (highest training count = 300)
    pred0, fb0 = sc.predict_with_fallback(["unseen1", "unseen2"])
    assert pred0 == "italian" and fb0 == "zero_score"

    # Nonzero tie between indian (0.50) and italian (0.50) -> italian wins by training count
    pred_tie, fb_tie = sc.predict_with_fallback(["x", "y"])
    assert pred_tie == "italian" and fb_tie == "tied_score"

    # Equal training counts -> alphabetical tie-break
    sc_eq = FingerprintScorer(
        {"indian": bb_a, "italian": bb_b, "mexican": bb_c},
        {"indian": 200, "italian": 200, "mexican": 100},
    )
    pred_eq, fb_eq = sc_eq.predict_with_fallback(["x", "y"])
    assert pred_eq == "indian" and fb_eq == "tied_score"


def test_real_scorer_api_and_garam_masala_investigation(cfg, d4_dir):
    # Verify no hardcoded special-casing in src/fingerprint.py
    src_code = inspect.getsource(fp_mod)
    assert "garam masala" not in src_code.lower()
    assert "turmeric" not in src_code.lower()

    # Check public API
    scores = predict_scores(["garam masala", "cumin", "turmeric"])
    pred = predict(["garam masala", "cumin", "turmeric"])
    assert set(scores.keys()) == set(cfg["cuisines"])
    assert pred in cfg["cuisines"]

    # Check that diagnostic artifact exists and matches live scorer.explain()
    diag_path = d4_dir / "garam_masala_diagnostic.json"
    assert diag_path.exists()
    diag = read_json(diag_path)

    scorer = FingerprintScorer.from_artifacts(d4_dir / "training_artifacts", normalized=False)
    live_literal = scorer.explain(["garam masala", "cumin", "turmeric"])
    assert live_literal["prediction"] == pred
    assert live_literal["scores"] == diag["literal_test"]["diagnostic"]["scores"]
    assert pred == "indian", (
        f"predict(['garam masala', 'cumin', 'turmeric']) returned {pred!r} instead of 'indian'. "
        f"Diagnostic: {live_literal}"
    )

    # Also verify exact-dataset-string variants are recorded and match live scorer
    for key in ["dataset_string_variant_ground", "dataset_string_variant_seed"]:
        inp = diag[key]["input"]
        assert scorer.predict(inp) == diag[key]["actual_prediction"] == "indian"
