"""Fingerprint Scorer (Deliverable 4): backbone-NPMI pair scoring with deterministic fallback.

Score (primary, exactly as specified)::

    Score(c) = Σ_{a<b, (a,b) ∈ E_c} NPMI_c(a, b)

over all unordered pairs of distinct, normalised, deduplicated input ingredients,
where E_c is cuisine c's *training-split* backbone. Absent pairs contribute 0.

Normalised ablation (separate experiment)::

    Score_norm(c) = Score(c) / (#matched pairs in E_c), or 0 if no pair matches.

Tie / fallback policy: among cuisines sharing the maximum score (|Δ| <= 1e-12),
pick the one most frequent in the training split (ties in frequency broken
alphabetically). If all three scores are 0 the fallback is a *zero-score*
fallback; any other tie is a *tied-score* fallback. Both are counted in evaluation.

Public API (uses ``outputs/d4_scorer/training_artifacts``)::

    predict(ingredients: list[str]) -> str
    predict_scores(ingredients: list[str]) -> dict
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common import ROOT, read_json
from src.normalize import normalize_recipe

TIE_TOL = 1e-12
DEFAULT_ARTIFACTS = ROOT / "outputs" / "d4_scorer" / "training_artifacts"


class FingerprintScorer:
    """Score recipes against per-cuisine backbones. ``normalized=True`` gives the ablation."""

    def __init__(self, backbones: dict[str, pd.DataFrame], class_counts: dict[str, int], normalized: bool = False):
        self.cuisines = list(backbones)
        missing = [c for c in self.cuisines if c not in class_counts]
        if missing:
            raise ValueError(f"class_counts missing for {missing}")
        self.normalized = normalized
        self.edges: dict[str, dict[tuple[str, str], float]] = {}
        for c, df in backbones.items():
            d = {}
            for a, b, w in zip(df["ingredient_a"], df["ingredient_b"], df["NPMI"]):
                a, b = (a, b) if a < b else (b, a)
                d[(a, b)] = float(w)
            self.edges[c] = d
        self.class_counts = {c: int(class_counts[c]) for c in self.cuisines}
        self.priority = sorted(self.cuisines, key=lambda c: (-self.class_counts[c], c))
        self.vocab = {c: {x for p in self.edges[c] for x in p} for c in self.cuisines}

    # ------------------------------------------------------------------ core
    def explain(self, ingredients: list[str]) -> dict:
        """Full diagnostic: normalised input, matched pairs with NPMI, scores, prediction and fallback."""
        norm = normalize_recipe(list(ingredients))
        per = {}
        for c in self.cuisines:
            matched = []
            for a, b in itertools.combinations(norm, 2):  # norm is sorted -> a < b
                w = self.edges[c].get((a, b))
                if w is not None:
                    matched.append({"pair": [a, b], "NPMI": w})
            total = float(sum(m["NPMI"] for m in matched))
            score = (total / len(matched) if matched else 0.0) if self.normalized else total
            per[c] = {"matched_pairs": matched, "n_matched": len(matched), "npmi_sum": total, "score": score,
                      "inputs_in_backbone_vocab": [x for x in norm if x in self.vocab[c]],
                      "training_class_count": self.class_counts[c]}
        scores = {c: per[c]["score"] for c in self.cuisines}
        label, fb = self._decide(scores)
        return {"normalized_input": norm, "per_cuisine": per, "scores": scores, "prediction": label,
                "fallback": fb, "priority_order": self.priority, "variant": "normalized" if self.normalized else "primary"}

    def _decide(self, scores: dict) -> tuple[str, str | None]:
        m = max(scores.values())
        tied = [c for c in self.cuisines if abs(scores[c] - m) <= TIE_TOL]
        if len(tied) == 1:
            return tied[0], None
        choice = next(c for c in self.priority if c in tied)
        if all(abs(s) <= TIE_TOL for s in scores.values()):
            return choice, "zero_score"
        return choice, "tied_score"

    def predict_scores(self, ingredients: list[str]) -> dict:
        return dict(self.explain(ingredients)["scores"])

    def predict_with_fallback(self, ingredients: list[str]) -> tuple[str, str | None]:
        e = self.explain(ingredients)
        return e["prediction"], e["fallback"]

    def predict(self, ingredients: list[str]) -> str:
        return self.explain(ingredients)["prediction"]

    # ------------------------------------------------------------------ io
    @classmethod
    def from_artifacts(cls, directory: str | Path = DEFAULT_ARTIFACTS, normalized: bool = False) -> "FingerprintScorer":
        directory = Path(directory)
        meta_path = directory / "training_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} missing; run `python evaluate.py` to build training artifacts.")
        meta = read_json(meta_path)
        bbs = {c: pd.read_csv(directory / f"backbone_{c}.csv") for c in meta["cuisines"]}
        return cls(bbs, meta["train_class_counts"], normalized=normalized)


_DEFAULT: dict[bool, FingerprintScorer] = {}


def _default(normalized: bool = False) -> FingerprintScorer:
    if normalized not in _DEFAULT:
        _DEFAULT[normalized] = FingerprintScorer.from_artifacts(DEFAULT_ARTIFACTS, normalized)
    return _DEFAULT[normalized]


def predict(ingredients: list[str]) -> str:
    """Predict 'indian', 'italian' or 'mexican' with the primary training-split Fingerprint Scorer."""
    return _default(False).predict(ingredients)


def predict_scores(ingredients: list[str]) -> dict:
    """Return the three primary Fingerprint scores {cuisine: score}."""
    return _default(False).predict_scores(ingredients)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score an ingredient list with the training-split Fingerprint Scorer.")
    ap.add_argument("ingredients", nargs="+", help="Ingredient names (quote multi-word names)")
    ap.add_argument("--normalized", action="store_true", help="Use the normalised-score ablation")
    ap.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    args = ap.parse_args(argv)
    sc = FingerprintScorer.from_artifacts(args.artifacts, args.normalized)
    print(json.dumps(sc.explain(args.ingredients), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
