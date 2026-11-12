"""Deliverable 4 evaluation pipeline: training-only backbones, Fingerprint Scorer,
normalized ablation, Multinomial Naive Bayes baseline, diagnostics, and confusion matrices.

Running ``python evaluate.py`` (or ``python src/evaluate.py``) reproduces the
entire Deliverable 4 evaluation from scratch:
1. Load and validate the real dataset ``data/raw/train.json``.
2. Deterministically normalise ingredients.
3. Perform the stratified 80/20 train/test split (seed 42) on recipe IDs.
4. Build eligible nodes, PMI/NPMI edges, 1,000-sample Curveball null distributions
   and validated backbones using **training recipes only** (enforced by
   :func:`src.build_networks.assert_training_only`).
5. Fit the binary-incidence ``MultinomialNB`` baseline on the **same training recipes**.
6. Predict on the held-out 20% test set with:
   * ``Fingerprint Scorer``
   * ``Fingerprint (normalized ablation)``
   * ``Naive Bayes``
7. Execute and save the ``["garam masala", "cumin", "turmeric"]`` vocabulary and
   pair-matching diagnostic alongside exact-dataset-string variants.
8. Save ``outputs/d4_scorer/results.csv``, four confusion matrix PNGs, training
   artifacts and ``outputs/d4_scorer/evaluation_metadata.json`` (with dataset hash
   and measured runtime per phase).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB

from src.backbone import BACKBONE_COLUMNS, SUMMARY_COLUMNS, build_backbones
from src.build_networks import NODE_COLUMNS, assert_training_only, build_predictive_networks
from src.common import (DataError, get_logger, is_debug_run, load_config, load_raw_dataset, print_dataset_report,
                        rpath, set_global_seed, sha256_obj, timed, write_json)
from src.fingerprint import FingerprintScorer, TIE_TOL, _DEFAULT
from src.normalize import normalize_ingredient, normalizer_fingerprint
from src.preprocess import select_and_normalize
from src.split_data import make_split

log = get_logger()

RESULTS_COLUMNS = [
    "model",
    "accuracy",
    "macro_f1",
    "precision_indian",
    "precision_italian",
    "precision_mexican",
    "recall_indian",
    "recall_italian",
    "recall_mexican",
    "f1_indian",
    "f1_italian",
    "f1_mexican",
    "test_samples",
    "zero_score_fallback_count",
    "tied_score_fallback_count",
]

PROBE_TERMS = ["cumin", "ground cumin", "cumin seed", "turmeric", "ground turmeric", "garam masala"]


class BinaryMultinomialNB:
    """Scikit-learn MultinomialNB over binary recipe-level ingredient presence.

    Vocabulary is learned exclusively from training recipes; unseen test
    ingredients are safely ignored.
    """

    def __init__(self, cuisines: list[str], alpha: float = 1.0, fit_prior: bool = True):
        self.cuisines = list(cuisines)
        self.alpha = float(alpha)
        self.fit_prior = bool(fit_prior)
        self.vocab_: list[str] = []
        self.vocab_index_: dict[str, int] = {}
        self.clf_: MultinomialNB | None = None
        self.train_recipe_ids_: set[int] = set()

    @staticmethod
    def _norm_map(recipes: list[dict]) -> dict[str, str]:
        uniq = {x for r in recipes for x in r["ingredients"]}
        return {x: normalize_ingredient(x) for x in uniq}

    def _to_csr(self, recipes: list[dict], norm_map: dict[str, str] | None = None) -> sp.csr_matrix:
        if norm_map is None:
            norm_map = self._norm_map(recipes)
        indptr = [0]
        indices: list[int] = []
        for r in recipes:
            norm_ings = {norm_map[x] for x in r["ingredients"]}
            norm_ings.discard("")
            cols = sorted({self.vocab_index_[x] for x in norm_ings if x in self.vocab_index_})
            indices.extend(cols)
            indptr.append(len(indices))
        data = np.ones(len(indices), dtype=np.float64)
        return sp.csr_matrix(
            (data, np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
            shape=(len(recipes), len(self.vocab_)),
        )

    def fit(self, train_recipes: list[dict], train_ids: set[int], test_ids: set[int]) -> "BinaryMultinomialNB":
        assert_training_only(train_recipes, train_ids, test_ids)
        self.train_recipe_ids_ = {int(r["id"]) for r in train_recipes}
        nmap = self._norm_map(train_recipes)
        vocab_set = {nmap[x] for r in train_recipes for x in r["ingredients"]}
        vocab_set.discard("")
        self.vocab_ = sorted(vocab_set)
        self.vocab_index_ = {w: i for i, w in enumerate(self.vocab_)}
        X = self._to_csr(train_recipes, norm_map=nmap)
        c_to_idx = {c: i for i, c in enumerate(self.cuisines)}
        y = np.asarray([c_to_idx[r["cuisine"]] for r in train_recipes], dtype=np.int64)
        self.clf_ = MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior)
        self.clf_.fit(X, y)
        return self

    def predict(self, recipes: list[dict]) -> list[str]:
        preds, _, _ = self.predict_with_fallback(recipes)
        return preds

    def predict_with_fallback(self, recipes: list[dict]) -> tuple[list[str], int, int]:
        """Predict cuisines and return (predictions, zero_feature_fallback_count, tied_score_fallback_count)."""
        if self.clf_ is None:
            raise RuntimeError("Model not fitted")
        X = self._to_csr(recipes)
        preds = self.clf_.predict(X)
        row_nnz = np.diff(X.indptr)
        zero_fb = int(np.sum(row_nnz == 0))
        log_probs = self.clf_.predict_log_proba(X)
        if log_probs.shape[1] >= 2:
            sorted_lp = np.sort(log_probs, axis=1)
            tied_mask = (row_nnz > 0) & (np.abs(sorted_lp[:, -1] - sorted_lp[:, -2]) <= TIE_TOL)
            tied_fb = int(np.sum(tied_mask))
        else:
            tied_fb = 0
        return [self.cuisines[int(i)] for i in preds], zero_fb, tied_fb


def compute_metrics_row(model_name: str, y_true: list[str], y_pred: list[str], cuisines: list[str],
                        zero_fb: int, tied_fb: int) -> tuple[dict, np.ndarray]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=cuisines, average="macro", zero_division=0))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=cuisines, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=cuisines)
    if int(cm.sum()) != len(y_true):
        raise AssertionError("Confusion matrix sum does not match test sample count")
    row = {
        "model": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "precision_indian": float(prec[0]),
        "precision_italian": float(prec[1]),
        "precision_mexican": float(prec[2]),
        "recall_indian": float(rec[0]),
        "recall_italian": float(rec[1]),
        "recall_mexican": float(rec[2]),
        "f1_indian": float(f1[0]),
        "f1_italian": float(f1[1]),
        "f1_mexican": float(f1[2]),
        "test_samples": int(len(y_true)),
        "zero_score_fallback_count": int(zero_fb),
        "tied_score_fallback_count": int(tied_fb),
    }
    return row, cm


def plot_confusion_matrix(cm: np.ndarray, cuisines: list[str], title: str, path: Path,
                          normalized: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    labels = [c.title() for c in cuisines]
    if normalized:
        row_sums = cm.sum(axis=1, keepdims=True)
        mat = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
        annot = np.empty(cm.shape, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{mat[i, j]:.3f}\n({cm[i, j]})"
        sns.heatmap(mat, annot=annot, fmt="", cmap="Blues", xticklabels=labels, yticklabels=labels,
                    vmin=0.0, vmax=1.0, cbar_kws={"label": "Row proportion"}, ax=ax)
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
                    cbar_kws={"label": "Test recipes"}, ax=ax)
    ax.set_xlabel("Predicted cuisine")
    ax.set_ylabel("True cuisine")
    ax.set_title(f"{title}\n(Total test recipes = {int(cm.sum()):,})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_garam_masala_diagnostic(scorer: FingerprintScorer, train_nets: dict, cuisines: list[str]) -> dict:
    """Inspect backbone vocabulary membership and run literal + dataset-string probes."""
    vocab_membership = {}
    for term in PROBE_TERMS:
        vocab_membership[term] = {
            c: {
                "in_training_backbone_vocab": bool(term in scorer.vocab[c]),
                "in_training_eligible_nodes": bool(term in set(train_nets[c]["nodes"]["ingredient"])),
                "training_recipe_count": int(
                    train_nets[c]["all_prevalence"]
                    .loc[train_nets[c]["all_prevalence"]["ingredient"] == term, "recipe_count"]
                    .sum()
                ),
            }
            for c in cuisines
        }

    literal_input = ["garam masala", "cumin", "turmeric"]
    literal_diag = scorer.explain(literal_input)

    variant_ground = ["garam masala", "ground cumin", "ground turmeric"]
    variant_ground_diag = scorer.explain(variant_ground)

    variant_seed = ["garam masala", "cumin seed", "ground turmeric"]
    variant_seed_diag = scorer.explain(variant_seed)

    return {
        "probe_terms_vocabulary_check": vocab_membership,
        "literal_test": {
            "input": literal_input,
            "expected_by_prompt": "indian",
            "actual_prediction": literal_diag["prediction"],
            "matches_expected": bool(literal_diag["prediction"] == "indian"),
            "diagnostic": literal_diag,
        },
        "dataset_string_variant_ground": {
            "input": variant_ground,
            "actual_prediction": variant_ground_diag["prediction"],
            "diagnostic": variant_ground_diag,
        },
        "dataset_string_variant_seed": {
            "input": variant_seed,
            "actual_prediction": variant_seed_diag["prediction"],
            "diagnostic": variant_seed_diag,
        },
    }


def run(cfg: dict, use_cache: bool = True) -> dict:
    set_global_seed(int(cfg["seed"]))
    cuisines = list(cfg["cuisines"])
    timings: dict[str, float] = {}
    t_total0 = time.perf_counter()

    with timed("1_load_and_validate_dataset", timings, log):
        records, ds_report = load_raw_dataset(cfg)
        print_dataset_report(ds_report, log)

    with timed("2_normalize_ingredients", timings, log):
        recipes, _, _ = select_and_normalize(records, cuisines)

    with timed("3_stratified_split", timings, log):
        split = make_split(recipes, cfg)
        train_ids = set(split["train_ids"])
        test_ids = set(split["test_ids"])
        assert len(train_ids & test_ids) == 0
        train_recipes = [r for r in recipes if r["id"] in train_ids]
        test_recipes = [r for r in recipes if r["id"] in test_ids]
        assert_training_only(train_recipes, train_ids, test_ids)

    with timed("4_build_training_networks_and_backbones", timings, log):
        mc = int(cfg["network"]["min_ingredient_count"])
        train_nets = build_predictive_networks(train_recipes, cuisines, mc, train_ids, test_ids)
        train_bb_res = build_backbones(
            train_recipes, train_nets, cfg, scope="predictive",
            use_cache=use_cache, dataset_sha=ds_report["sha256"]
        )

    d4 = rpath(cfg["outputs"]["d4"])
    art = d4 / "training_artifacts"
    art.mkdir(parents=True, exist_ok=True)

    # Save training-only artifacts for FingerprintScorer
    pd.concat([train_nets[c]["nodes"] for c in cuisines], ignore_index=True)[NODE_COLUMNS].to_csv(
        art / "nodes_train.csv", index=False, encoding="utf-8"
    )
    for c in cuisines:
        train_bb_res["backbone"][c][BACKBONE_COLUMNS].to_csv(
            art / f"backbone_{c}.csv", index=False, encoding="utf-8"
        )
    train_bb_res["summary_df"][SUMMARY_COLUMNS + ["eligible_nodes"]].to_csv(
        art / "backbone_summary_train.csv", index=False, encoding="utf-8"
    )
    train_meta = {
        "scope": "predictive (80% stratified training split only)",
        "run_mode": "DEBUG" if is_debug_run(cfg) else "FULL",
        "seed": int(cfg["seed"]),
        "cuisines": cuisines,
        "n_train_recipes": len(train_recipes),
        "n_test_recipes_excluded": len(test_recipes),
        "train_class_counts": split["train_class_counts"],
        "train_ids_sha256": split["train_ids_sha256"],
        "test_ids_sha256": split["test_ids_sha256"],
        "dataset_sha256": ds_report["sha256"],
        "normalizer_sha256": normalizer_fingerprint(),
        "null_model_diagnostics": train_bb_res["diagnostics"],
    }
    write_json(train_meta, art / "training_metadata.json")
    _DEFAULT.clear()  # reset cached default scorers so predict() reloads fresh artifacts

    with timed("5_fit_naive_bayes_baseline", timings, log):
        nb = BinaryMultinomialNB(
            cuisines,
            alpha=float(cfg["naive_bayes"]["alpha"]),
            fit_prior=bool(cfg["naive_bayes"]["fit_prior"]),
        ).fit(train_recipes, train_ids, test_ids)

    with timed("6_evaluate_models_on_test_split", timings, log):
        scorer_primary = FingerprintScorer.from_artifacts(art, normalized=False)
        scorer_norm = FingerprintScorer.from_artifacts(art, normalized=True)

        y_true = [r["cuisine"] for r in test_recipes]

        preds_primary, fb_primary = [], []
        preds_norm, fb_norm = [], []
        for r in test_recipes:
            p1, f1 = scorer_primary.predict_with_fallback(r["ingredients"])
            preds_primary.append(p1)
            fb_primary.append(f1)
            p2, f2 = scorer_norm.predict_with_fallback(r["ingredients"])
            preds_norm.append(p2)
            fb_norm.append(f2)

        preds_nb, nb_zero_fb, nb_tied_fb = nb.predict_with_fallback(test_recipes)

        row_fp, cm_fp = compute_metrics_row(
            "Fingerprint Scorer", y_true, preds_primary, cuisines,
            sum(x == "zero_score" for x in fb_primary),
            sum(x == "tied_score" for x in fb_primary),
        )
        row_norm, cm_norm = compute_metrics_row(
            "Fingerprint (normalized ablation)", y_true, preds_norm, cuisines,
            sum(x == "zero_score" for x in fb_norm),
            sum(x == "tied_score" for x in fb_norm),
        )
        row_nb, cm_nb = compute_metrics_row(
            "Naive Bayes", y_true, preds_nb, cuisines, nb_zero_fb, nb_tied_fb
        )

        results_df = pd.DataFrame([row_fp, row_norm, row_nb], columns=RESULTS_COLUMNS)
        results_csv_path = d4 / "results.csv"
        try:
            results_df.to_csv(results_csv_path, index=False, encoding="utf-8")
        except PermissionError:
            if results_csv_path.exists():
                existing_df = pd.read_csv(results_csv_path)
                pd.testing.assert_frame_equal(existing_df, results_df, check_exact=False, rtol=1e-12, atol=1e-12)
                log.warning("results.csv is locked by another process; verified existing file matches computed results.")
            else:
                raise

    with timed("7_plot_confusion_matrices", timings, log):
        plot_confusion_matrix(cm_fp, cuisines, "Fingerprint Scorer — Confusion Matrix (Counts)",
                              d4 / "confusion_matrix.png", normalized=False)
        plot_confusion_matrix(cm_norm, cuisines, "Fingerprint (Normalized Ablation) — Confusion Matrix (Counts)",
                              d4 / "confusion_matrix_ablation.png", normalized=False)
        plot_confusion_matrix(cm_nb, cuisines, "Naive Bayes Baseline — Confusion Matrix (Counts)",
                              d4 / "confusion_matrix_nb.png", normalized=False)
        plot_confusion_matrix(cm_fp, cuisines, "Fingerprint Scorer — Row-Normalized Confusion Matrix",
                              d4 / "confusion_matrix_normalized.png", normalized=True)
        plot_confusion_matrix(cm_norm, cuisines, "Fingerprint (Normalized Ablation) — Row-Normalized Confusion Matrix",
                              d4 / "confusion_matrix_normalized_ablation.png", normalized=True)
        plot_confusion_matrix(cm_nb, cuisines, "Naive Bayes Baseline — Row-Normalized Confusion Matrix",
                              d4 / "confusion_matrix_normalized_nb.png", normalized=True)

    with timed("8_garam_masala_diagnostic", timings, log):
        gm_diag = run_garam_masala_diagnostic(scorer_primary, train_nets, cuisines)
        write_json(gm_diag, d4 / "garam_masala_diagnostic.json")
        log.info("Garam masala vocabulary check (in training backbone vocab):")
        for term, per_c in gm_diag["probe_terms_vocabulary_check"].items():
            log.info("  %-16s -> %s", term, {c: (per_c[c]["in_training_backbone_vocab"],
                                                 f"n={per_c[c]['training_recipe_count']}") for c in cuisines})
        log.info("Literal test predict(['garam masala', 'cumin', 'turmeric']) -> %s (scores=%s, fallback=%s)",
                 gm_diag["literal_test"]["actual_prediction"],
                 gm_diag["literal_test"]["diagnostic"]["scores"],
                 gm_diag["literal_test"]["diagnostic"]["fallback"])
        log.info("Dataset-string variant predict(['garam masala', 'ground cumin', 'ground turmeric']) -> %s (scores=%s)",
                 gm_diag["dataset_string_variant_ground"]["actual_prediction"],
                 gm_diag["dataset_string_variant_ground"]["diagnostic"]["scores"])
        log.info("Dataset-string variant predict(['garam masala', 'cumin seed', 'ground turmeric']) -> %s (scores=%s)",
                 gm_diag["dataset_string_variant_seed"]["actual_prediction"],
                 gm_diag["dataset_string_variant_seed"]["diagnostic"]["scores"])

    timings["total_evaluation_seconds"] = round(time.perf_counter() - t_total0, 3)

    eval_meta = {
        "run_mode": "DEBUG" if is_debug_run(cfg) else "FULL",
        "n_randomizations": int(cfg["null_model"]["n_randomizations"]),
        "dataset_sha256": ds_report["sha256"],
        "dataset_path": ds_report["path"],
        "dataset_total_records": ds_report["total_records"],
        "selected_records": len(recipes),
        "seed": int(cfg["seed"]),
        "effective_config": cfg,
        "normalizer_sha256": normalizer_fingerprint(),
        "split": {
            "n_train": split["n_train"],
            "n_test": split["n_test"],
            "train_class_counts": split["train_class_counts"],
            "test_class_counts": split["test_class_counts"],
            "train_ids_sha256": split["train_ids_sha256"],
            "test_ids_sha256": split["test_ids_sha256"],
            "train_test_overlap": 0,
        },
        "training_backbone_summary": train_bb_res["summary"],
        "naive_bayes": {
            "vocabulary_size": len(nb.vocab_),
            "alpha": nb.alpha,
            "fit_prior": nb.fit_prior,
        },
        "confusion_matrices": {
            "cuisines_order": cuisines,
            "fingerprint_scorer": cm_fp.tolist(),
            "fingerprint_normalized_ablation": cm_norm.tolist(),
            "naive_bayes": cm_nb.tolist(),
        },
        "metrics": [row_fp, row_norm, row_nb],
        "garam_masala_acceptance_diagnostic": gm_diag,
        "phase_runtimes_seconds": timings,
        "predictions_sha256": {
            "fingerprint_scorer": sha256_obj(preds_primary),
            "fingerprint_normalized_ablation": sha256_obj(preds_norm),
            "naive_bayes": sha256_obj(preds_nb),
        },
    }
    write_json(eval_meta, d4 / "evaluation_metadata.json")

    for row in [row_fp, row_norm, row_nb]:
        log.info("MODEL %-34s | acc=%.4f | macro_f1=%.4f | F1(IN/IT/MX)=%.3f/%.3f/%.3f | zero_fb=%d tied_fb=%d",
                 row["model"], row["accuracy"], row["macro_f1"],
                 row["f1_indian"], row["f1_italian"], row["f1_mexican"],
                 row["zero_score_fallback_count"], row["tied_score_fallback_count"])
    return eval_meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reproduce Deliverable 4 training-only evaluation from scratch.")
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    ap.add_argument("--seed", type=int, default=None, help="Override random seed (default 42)")
    ap.add_argument("--n-null", type=int, default=None, help="DEBUG ONLY: number of null samples (default 1000)")
    ap.add_argument("--no-cache", action="store_true", help="Recompute null-model chains without cache")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, {"seed": args.seed, "null_model.n_randomizations": args.n_null})
    try:
        run(cfg, use_cache=not args.no_cache)
    except DataError as exc:
        log.error(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
