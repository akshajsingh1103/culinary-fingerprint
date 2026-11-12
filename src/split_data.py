"""Stratified 80/20 train/test split (seed 42), performed before any learned step.

The split operates on recipe IDs only; it does not look at ingredients, so no
frequency filtering, graph construction, null model or model fitting can
influence it. IDs are saved to ``data/processed/split.json`` (not a public output).

Usage::

    python src/split_data.py [--config config.yaml] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import train_test_split

from src.common import get_logger, load_config, read_json, rpath, sha256_obj, write_json

log = get_logger()


def make_split(recipes: list[dict], cfg: dict) -> dict:
    """Return a dict with sorted ``train_ids`` and ``test_ids`` (stratified by cuisine)."""
    recipes = sorted(recipes, key=lambda r: r["id"])
    ids = [r["id"] for r in recipes]
    labels = [r["cuisine"] for r in recipes]
    test_frac = float(cfg["split"]["test_fraction"])
    if abs(test_frac + float(cfg["split"]["train_fraction"]) - 1.0) > 1e-9:
        raise ValueError("train_fraction + test_fraction must equal 1")
    tr, te = train_test_split(ids, test_size=test_frac, random_state=int(cfg["seed"]),
                              stratify=labels if cfg["split"].get("stratify", True) else None)
    tr, te = sorted(tr), sorted(te)
    if set(tr) & set(te):
        raise AssertionError("Train/test overlap detected")
    lab = dict(zip(ids, labels))
    return {
        "seed": int(cfg["seed"]),
        "test_fraction": test_frac,
        "train_ids": tr,
        "test_ids": te,
        "n_train": len(tr),
        "n_test": len(te),
        "train_class_counts": dict(sorted(Counter(lab[i] for i in tr).items())),
        "test_class_counts": dict(sorted(Counter(lab[i] for i in te).items())),
        "train_ids_sha256": sha256_obj(tr),
        "test_ids_sha256": sha256_obj(te),
    }


def split_path(cfg: dict) -> Path:
    return rpath(cfg["data"]["processed_dir"]) / "split.json"


def run(cfg: dict, recipes: list[dict] | None = None) -> dict:
    if recipes is None:
        from src.preprocess import load_processed
        recipes, _ = load_processed(cfg)
    split = make_split(recipes, cfg)
    write_json(split, split_path(cfg))
    log.info("Split: train=%d test=%d | train classes=%s | test classes=%s", split["n_train"], split["n_test"],
             split["train_class_counts"], split["test_class_counts"])
    return split


def load_split(cfg: dict) -> dict:
    p = split_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `python src/split_data.py` first.")
    return read_json(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Create the stratified train/test split of recipe IDs.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None, help="Override random seed (default from config: 42)")
    args = ap.parse_args(argv)
    run(load_config(args.config, {"seed": args.seed}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
