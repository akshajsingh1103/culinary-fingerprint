"""Self-contained runtime inference engine for the Culinary Fingerprint web application.

Loads pre-computed artifacts from `outputs/app/artifacts/` and performs:
- Deterministic ingredient normalization (using `normalization_rules.json`)
- Free-text recipe list parsing and vocabulary matching
- Fingerprint Scorer v1 (original backbone NPMI sum with tie/zero-score fallback)
- Fingerprint Scorer v2 (22-feature network-informed stacked L2 logistic regression)
- Detailed "Why this prediction" structural explanation (matched backbone pairs,
  D3 hubs, Louvain communities, cross-cuisine bridges, and unrecognized items).

Self-contained: makes NO imports from `src/` or `src/d4_experiments/`.
"""
from __future__ import annotations

import itertools
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
DEFAULT_APP_ARTIFACTS = APP_DIR / "artifacts"
TIE_TOL = 1e-12

_QUOTES = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u00b4": "'", "\u0060": "'"}
_SYMBOLS = ("\u00ae", "\u2122", "\u00a9")  # ®, ™, ©
_WS = re.compile(r"\s+")


class CulinaryFingerprintRuntime:
    """Self-contained runtime for Scorer v1 and Scorer v2 inference and explanation."""

    def __init__(self, artifacts_dir: str | Path = DEFAULT_APP_ARTIFACTS):
        self.artifacts_dir = Path(artifacts_dir)
        if not self.artifacts_dir.exists():
            raise FileNotFoundError(
                f"App artifacts directory {self.artifacts_dir} not found. "
                "Run `python outputs/app/export_artifacts.py` first."
            )

        # 1. Load normalization rules
        norm_data = json.loads((self.artifacts_dir / "normalization_rules.json").read_text(encoding="utf-8"))
        self.plural_whitelist: dict[str, str] = dict(norm_data["plural_whitelist"])
        self.singularize_exceptions: frozenset[str] = frozenset(norm_data["singularize_exceptions"])
        self.synonyms: dict[str, str] = {k: v["target"] for k, v in norm_data["synonyms"].items()}

        # 2. Load v2 model parameters and D3 structures
        v2_data = json.loads((self.artifacts_dir / "v2_model.json").read_text(encoding="utf-8"))
        self.cuisines: list[str] = list(v2_data["cuisines"])
        self.c_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.cuisines)}
        self.class_counts: dict[str, int] = {c: int(v2_data["train_class_counts"][c]) for c in self.cuisines}
        self.priority_order: list[str] = sorted(self.cuisines, key=lambda c: (-self.class_counts[c], c))

        self.feature_names: list[str] = list(v2_data["feature_names"])
        self.selected_c_v2: float = float(v2_data["selected_C"])
        self.scaler_mean: np.ndarray = np.asarray(v2_data["scaler_mean"], dtype=np.float64)
        self.scaler_scale: np.ndarray = np.asarray(v2_data["scaler_scale"], dtype=np.float64)
        self.coef_v2: np.ndarray = np.asarray(v2_data["coef"], dtype=np.float64)  # (3, 22)
        self.intercept_v2: np.ndarray = np.asarray(v2_data["intercept"], dtype=np.float64)  # (3,)

        d3 = v2_data["d3_structures"]
        self.backbone_vocab: dict[str, set[str]] = {c: set(d3["backbone_vocab"][c]) for c in self.cuisines}
        self.union_backbone_vocab: set[str] = set().union(*self.backbone_vocab.values())
        self.hubs_top10: dict[str, set[str]] = {c: set(d3["hubs_top10"][c]) for c in self.cuisines}
        self.hub_strengths: dict[str, dict[str, float]] = {
            c: {k: float(v) for k, v in d3["hub_strengths"][c].items()} for c in self.cuisines
        }
        self.bridges_union: dict[str, set[str]] = {c: set(d3["bridges_union"][c]) for c in self.cuisines}
        self.bridges_details: dict[str, list[dict]] = d3["bridges_details"]
        self.communities_part: dict[str, dict[str, int]] = {
            c: {k: int(v) for k, v in d3["communities_partition"][c].items()} for c in self.cuisines
        }
        self.n_communities: dict[str, int] = {c: int(d3["n_communities"][c]) for c in self.cuisines}
        self.community_labels: dict[str, dict[int, str]] = {
            c: {int(k): str(v) for k, v in d3["community_labels"][c].items()} for c in self.cuisines
        }

        # 3. Load per-cuisine backbone edge tables (for Scorer v1 and network graph visualization)
        self.v1_edges: dict[str, dict[tuple[str, str], float]] = {}
        self.v1_edge_details: dict[str, dict[tuple[str, str], dict]] = {}
        for c in self.cuisines:
            bb_df = pd.read_csv(self.artifacts_dir / f"backbone_{c}.csv")
            edges_c: dict[tuple[str, str], float] = {}
            details_c: dict[tuple[str, str], dict] = {}
            for a, b, npmi, z_sc, co in zip(
                bb_df["ingredient_a"],
                bb_df["ingredient_b"],
                bb_df["NPMI"],
                bb_df["z_score"],
                bb_df["co_count"],
            ):
                pair = (str(a), str(b)) if str(a) < str(b) else (str(b), str(a))
                edges_c[pair] = float(npmi)
                details_c[pair] = {
                    "pair": [pair[0], pair[1]],
                    "NPMI": float(npmi),
                    "z_score": float(z_sc),
                    "co_count": int(co),
                }
            self.v1_edges[c] = edges_c
            self.v1_edge_details[c] = details_c

        # 4. Load v2 pair log-odds lookup table
        pw_df = pd.read_csv(self.artifacts_dir / "pair_log_odds_v2.csv")
        self.v2_pair_log_odds: dict[tuple[str, str], tuple[float, float, float]] = {}
        for a, b, w_in, w_it, w_mx in zip(
            pw_df["ingredient_a"],
            pw_df["ingredient_b"],
            pw_df["log_odds_indian"],
            pw_df["log_odds_italian"],
            pw_df["log_odds_mexican"],
        ):
            pair = (str(a), str(b)) if str(a) < str(b) else (str(b), str(a))
            self.v2_pair_log_odds[pair] = (float(w_in), float(w_it), float(w_mx))
        self.active_pair_ings: set[str] = {x for p in self.v2_pair_log_odds for x in p}

        # 5. Load ingredient vocabulary ordered by frequency
        vocab_data = json.loads((self.artifacts_dir / "ingredient_vocabulary.json").read_text(encoding="utf-8"))
        self.ingredients_ordered: list[str] = list(vocab_data["ingredients_ordered_by_frequency"])
        self.backbone_ingredients_ordered: list[str] = list(vocab_data["backbone_ingredients_ordered_by_frequency"])
        self.known_vocab_set: set[str] = set(self.ingredients_ordered)
        self.vocab_entry_by_name: dict[str, dict] = {
            e["ingredient"]: e for e in vocab_data["entries"]
        }

        # 6. Load sample recipes and final test results (v1 & v2 only)
        samples_data = json.loads((self.artifacts_dir / "sample_recipes.json").read_text(encoding="utf-8"))
        self.sample_recipes: list[dict] = list(samples_data["recipes"])
        self.test_results_df: pd.DataFrame = pd.read_csv(self.artifacts_dir / "final_test_results.csv")
        self.test_summary_v1_v2: dict = json.loads(
            (self.artifacts_dir / "final_test_summary_v1_v2.json").read_text(encoding="utf-8")
        )

    # ------------------------------------------------------------------ Normalization
    def normalize_ingredient(self, name: str) -> str:
        """Deterministically normalize a single ingredient string (exact parity with src/normalize.py)."""
        if not isinstance(name, str):
            raise TypeError(f"Ingredient must be a string, got {type(name).__name__}")
        s = name
        # Stage 1: unicode
        for k, v in _QUOTES.items():
            s = s.replace(k, v)
        for sym in _SYMBOLS:
            s = s.replace(sym, " ")
        s = unicodedata.normalize("NFKC", s)
        for k, v in _QUOTES.items():
            s = s.replace(k, v)
        for sym in _SYMBOLS:
            s = s.replace(sym, " ")
        s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
        s = unicodedata.normalize("NFC", s)
        # Stage 2: lowercase
        s = s.lower()
        # Stage 3: whitespace
        s = _WS.sub(" ", s.strip())
        # Stage 4: singularize
        if s and s not in self.singularize_exceptions:
            head, sep, last = s.rpartition(" ")
            if last in self.plural_whitelist:
                s = f"{head}{sep}{self.plural_whitelist[last]}"
        # Stage 5: synonym
        if s in self.synonyms:
            s = self.synonyms[s]
        return s

    def normalize_recipe(self, ingredients: list[str]) -> list[str]:
        """Normalize, drop empties, deduplicate, and sort an ingredient list."""
        out = {self.normalize_ingredient(x) for x in ingredients if isinstance(x, str)}
        out.discard("")
        return sorted(out)

    def parse_pasted_text(self, raw_text: str) -> dict:
        """Parse comma-, semicolon-, or newline-separated ingredient text and match against vocabulary."""
        if not raw_text or not raw_text.strip():
            return {"raw_tokens": [], "recognized": [], "recognized_backbone": [], "unrecognized": []}
        tokens = [t.strip() for t in re.split(r"[,;\n\r]+", raw_text) if t.strip()]
        recognized = []
        recognized_backbone = []
        unrecognized = []
        seen_rec = set()
        for tok in tokens:
            norm = self.normalize_ingredient(tok)
            if not norm:
                continue
            if norm in self.known_vocab_set:
                if norm not in seen_rec:
                    seen_rec.add(norm)
                    recognized.append(norm)
                    if norm in self.union_backbone_vocab:
                        recognized_backbone.append(norm)
            else:
                unrecognized.append({"raw": tok, "normalized": norm})
        return {
            "raw_tokens": tokens,
            "recognized": recognized,
            "recognized_backbone": recognized_backbone,
            "unrecognized": unrecognized,
        }

    # ------------------------------------------------------------------ Scorer v1
    def _decide_v1(self, scores: dict[str, float]) -> tuple[str, str | None]:
        m = max(scores.values())
        tied = [c for c in self.cuisines if abs(scores[c] - m) <= TIE_TOL]
        if len(tied) == 1:
            return tied[0], None
        choice = next(c for c in self.priority_order if c in tied)
        if all(abs(s) <= TIE_TOL for s in scores.values()):
            return choice, "zero_score"
        return choice, "tied_score"

    def predict_v1(self, ingredients: list[str]) -> dict:
        """Run Fingerprint Scorer v1 (original backbone NPMI sum) and return full diagnostic."""
        norm = self.normalize_recipe(ingredients)
        per_c = {}
        for c in self.cuisines:
            matched = []
            edges_c = self.v1_edge_details[c]
            for a, b in itertools.combinations(norm, 2):
                d = edges_c.get((a, b))
                if d is not None:
                    matched.append(d)
            matched_sorted = sorted(matched, key=lambda x: (-x["NPMI"], x["pair"][0], x["pair"][1]))
            total_npmi = float(sum(m["NPMI"] for m in matched))
            per_c[c] = {
                "matched_pairs": matched_sorted,
                "n_matched": len(matched_sorted),
                "score": total_npmi,
                "inputs_in_backbone_vocab": [x for x in norm if x in self.backbone_vocab[c]],
            }
        scores = {c: per_c[c]["score"] for c in self.cuisines}
        pred, fallback = self._decide_v1(scores)

        # Pseudo-share of positive NPMI for visual bar display
        total_pos = sum(max(s, 0.0) for s in scores.values())
        score_shares = (
            {c: max(scores[c], 0.0) / total_pos for c in self.cuisines}
            if total_pos > TIE_TOL
            else {c: 1.0 / len(self.cuisines) for c in self.cuisines}
        )
        return {
            "prediction": pred,
            "fallback": fallback,
            "scores": scores,
            "score_shares": score_shares,
            "per_cuisine": per_c,
            "normalized_input": norm,
        }

    # ------------------------------------------------------------------ Scorer v2
    def extract_v2_features(self, norm_ingredients: list[str]) -> tuple[np.ndarray, dict]:
        """Compute the 22-feature vector for Scorer v2 from an already normalized, sorted ingredient list,
        along with human-readable structural explanations per cuisine.
        """
        feat = np.zeros(22, dtype=np.float64)
        L = len(norm_ingredients)
        denom = float(max(L, 1))
        ing_set = set(norm_ingredients)

        # 1. Pair log-odds features (indices 0..2) and matched pair contributions
        p_ings = [x for x in norm_ingredients if x in self.active_pair_ings]
        s0, s1, s2 = 0.0, 0.0, 0.0
        matched_v2_pairs = []
        if len(p_ings) >= 2:
            for pair in itertools.combinations(p_ings, 2):
                w = self.v2_pair_log_odds.get(pair)
                if w is not None:
                    s0 += w[0]
                    s1 += w[1]
                    s2 += w[2]
                    matched_v2_pairs.append({
                        "pair": [pair[0], pair[1]],
                        "log_odds": {
                            "indian": w[0],
                            "italian": w[1],
                            "mexican": w[2],
                        },
                    })
        feat[0], feat[1], feat[2] = s0, s1, s2

        # 2. Per-cuisine D3 structural features (indices 3..20)
        structural_details: dict[str, dict] = {}
        for ci, c in enumerate(self.cuisines):
            part_c = self.communities_part[c]
            n_comm_c = self.n_communities[c]
            comm_members: dict[int, list[str]] = defaultdict(list)
            total_in_comm = 0

            if n_comm_c > 0 and L > 0:
                for u in norm_ingredients:
                    cid = part_c.get(u)
                    if cid is not None:
                        comm_members[cid].append(u)
                        total_in_comm += 1
                if total_in_comm > 0:
                    cov_frac = total_in_comm / denom
                    max_cov = max(len(m) for m in comm_members.values()) / denom
                    ent = 0.0
                    for m in comm_members.values():
                        p = len(m) / float(total_in_comm)
                        if p > 0.0:
                            ent -= p * math.log(p)
                else:
                    cov_frac, max_cov, ent = 0.0, 0.0, 0.0
            else:
                cov_frac, max_cov, ent = 0.0, 0.0, 0.0

            hubs_found = sorted(
                ing_set & self.hubs_top10[c],
                key=lambda k: (-self.hub_strengths[c].get(k, 0.0), k),
            )
            bridges_found = sorted(ing_set & self.bridges_union[c])
            bb_ings_found = sorted(ing_set & self.backbone_vocab[c])

            feat[3 + ci] = cov_frac
            feat[6 + ci] = max_cov
            feat[9 + ci] = ent
            feat[12 + ci] = float(len(hubs_found))
            feat[15 + ci] = float(len(bridges_found))
            feat[18 + ci] = float(len(bb_ings_found))

            comm_breakdown = []
            for cid, members in sorted(comm_members.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                comm_breakdown.append({
                    "community_id": int(cid),
                    "community_label": self.community_labels[c].get(int(cid), f"Community {cid}"),
                    "members_found": members,
                    "count": len(members),
                    "share_of_recipe": len(members) / denom,
                })

            bridge_items_detail = [
                bd for bd in self.bridges_details[c] if bd["ingredient"] in ing_set
            ]

            structural_details[c] = {
                "pair_log_odds_sum": float(feat[ci]),
                "comm_cov_frac": float(cov_frac),
                "comm_max_cov": float(max_cov),
                "comm_entropy": float(ent),
                "hub_count": len(hubs_found),
                "hubs_found": [
                    {"ingredient": h, "weighted_degree": round(self.hub_strengths[c].get(h, 0.0), 3)}
                    for h in hubs_found
                ],
                "bridge_count": len(bridges_found),
                "bridges_found": bridges_found,
                "bridges_detail": bridge_items_detail,
                "backbone_ing_count": len(bb_ings_found),
                "backbone_ings_found": bb_ings_found,
                "communities_breakdown": comm_breakdown,
            }

        # Index 21: recipe_length
        feat[21] = float(L)
        return feat, {
            "per_cuisine": structural_details,
            "matched_union_backbone_pairs": matched_v2_pairs,
        }

    def predict_v2(self, ingredients: list[str]) -> dict:
        """Run Fingerprint Scorer v2 (22-feature network-informed stacked L2 logistic regression)."""
        norm = self.normalize_recipe(ingredients)
        feat_raw, struct_info = self.extract_v2_features(norm)
        feat_std = (feat_raw - self.scaler_mean) / self.scaler_scale
        logits = self.coef_v2 @ feat_std + self.intercept_v2

        # Numerically stable softmax
        shift_logits = logits - float(np.max(logits))
        exp_logits = np.exp(shift_logits)
        probs = exp_logits / float(np.sum(exp_logits))

        pred_idx = int(np.argmax(probs))
        pred_cuisine = self.cuisines[pred_idx]

        return {
            "prediction": pred_cuisine,
            "confidence": float(probs[pred_idx]),
            "probabilities": {c: float(probs[i]) for i, c in enumerate(self.cuisines)},
            "logits": {c: float(logits[i]) for i, c in enumerate(self.cuisines)},
            "raw_features": {
                self.feature_names[j]: float(feat_raw[j]) for j in range(len(self.feature_names))
            },
            "standardized_features": {
                self.feature_names[j]: float(feat_std[j]) for j in range(len(self.feature_names))
            },
            "structural_explanation": struct_info,
            "normalized_input": norm,
        }

    def predict_both(self, raw_ingredients: list[str]) -> dict:
        """Run both Scorer v1 and Scorer v2 on a list of raw or normalized ingredients and return a unified explanation."""
        norm_all = self.normalize_recipe(raw_ingredients)
        recognized_vocab = [x for x in norm_all if x in self.known_vocab_set]
        recognized_backbone = [x for x in norm_all if x in self.union_backbone_vocab]
        unrecognized = [x for x in norm_all if x not in self.known_vocab_set]
        outside_backbone = [x for x in recognized_vocab if x not in self.union_backbone_vocab]

        v1_res = self.predict_v1(raw_ingredients)
        v2_res = self.predict_v2(raw_ingredients)

        return {
            "normalized_input": norm_all,
            "recognized_vocab": recognized_vocab,
            "recognized_backbone": recognized_backbone,
            "outside_backbone_vocab": outside_backbone,
            "unrecognized": unrecognized,
            "n_total_normalized": len(norm_all),
            "n_recognized_vocab": len(recognized_vocab),
            "n_recognized_backbone": len(recognized_backbone),
            "low_confidence_warning": len(recognized_backbone) < 2,
            "empty_input": len(norm_all) == 0,
            "v1": v1_res,
            "v2": v2_res,
            "models_agree": v1_res["prediction"] == v2_res["prediction"],
        }
