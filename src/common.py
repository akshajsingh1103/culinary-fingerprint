"""Shared utilities: configuration, logging, hashing, JSON I/O and dataset loading.

Every pipeline module imports from here so that paths, seeds and provenance
are handled identically across deliverables.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"
REQUIRED_FIELDS = ("id", "cuisine", "ingredients")


class DataError(RuntimeError):
    """Raised when the input dataset is missing or invalid."""


# --------------------------------------------------------------------------- config
def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load ``config.yaml`` and apply a flat dict of dotted-key overrides.

    Example override: ``{"seed": 7, "null_model.n_randomizations": 50}``.
    """
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg = copy.deepcopy(cfg)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    # Keep the community seed tied to the global seed unless explicitly overridden.
    if overrides and overrides.get("seed") is not None and "community.seed" not in overrides:
        cfg["community"]["seed"] = cfg["seed"]
    return cfg


def rpath(rel: str | Path) -> Path:
    """Resolve a repository-relative path."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def is_debug_run(cfg: dict) -> bool:
    """A run is DEBUG if the null model uses anything other than 1,000 samples."""
    return int(cfg["null_model"]["n_randomizations"]) != 1000


# --------------------------------------------------------------------------- logging
def get_logger(name: str = "culinary") -> logging.Logger:
    """Return a configured logger writing concise messages to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


@contextmanager
def timed(label: str, store: dict | None = None, logger: logging.Logger | None = None):
    """Context manager measuring wall-clock seconds for a phase."""
    log = logger or get_logger()
    t0 = time.perf_counter()
    log.info("START %s", label)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if store is not None:
            store[label] = round(dt, 3)
        log.info("END   %s (%.2f s)", label, dt)


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy global RNGs (numba chains are seeded separately)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


# --------------------------------------------------------------------------- hashing / io
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    """Stable SHA-256 of a JSON-serialisable object."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def write_json(obj: Any, path: str | Path) -> None:
    """Write pretty, sorted JSON (UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_json_default, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def ensure_dirs(paths: Iterable[str | Path]) -> None:
    for p in paths:
        rpath(p).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- dataset
def missing_dataset_message(path: Path) -> str:
    return (
        f"Dataset not found at: {path}\n"
        "Download 'train.json' from the Kaggle competition \"What's Cooking?\" "
        "(https://www.kaggle.com/c/whats-cooking/data), unzip it, and place it at:\n"
        f"    {ROOT / 'data' / 'raw' / 'train.json'}\n"
        "The pipeline will not run on synthetic or substitute data."
    )


def load_raw_dataset(cfg: dict) -> tuple[list[dict], dict]:
    """Load and validate the raw Kaggle JSON.

    Returns the full list of records plus an inspection report. Raises
    :class:`DataError` with actionable instructions if the file is missing,
    unparsable or structurally invalid.
    """
    path = rpath(cfg["data"]["raw_path"])
    if not path.exists():
        raise DataError(missing_dataset_message(path))
    report: dict[str, Any] = {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                              "exists": True, "size_bytes": path.stat().st_size}
    try:
        with path.open("r", encoding="utf-8") as fh:
            records = json.load(fh)
    except json.JSONDecodeError as exc:  # pragma: no cover - depends on file
        raise DataError(f"JSON parsing failed for {path}: {exc}") from exc
    if not isinstance(records, list):
        raise DataError(f"Expected a JSON list of recipes in {path}, got {type(records).__name__}")
    report["json_parse_ok"] = True
    report["sha256"] = sha256_file(path)
    report["total_records"] = len(records)

    malformed: list[dict] = []
    for i, r in enumerate(records):
        problems = []
        if not isinstance(r, dict):
            problems.append("not an object")
        else:
            for f in REQUIRED_FIELDS:
                if f not in r:
                    problems.append(f"missing '{f}'")
            ing = r.get("ingredients")
            if "ingredients" in r and (not isinstance(ing, list) or not all(isinstance(x, str) for x in ing)):
                problems.append("ingredients not a list of strings")
            elif isinstance(ing, list) and len(ing) == 0:
                problems.append("empty ingredient list")
            if "cuisine" in r and not isinstance(r.get("cuisine"), str):
                problems.append("cuisine not a string")
        if problems:
            malformed.append({"index": i, "problems": problems})
    report["malformed_records"] = len(malformed)
    report["malformed_examples"] = malformed[:20]
    ids = [r.get("id") for r in records if isinstance(r, dict)]
    report["duplicate_ids"] = len(ids) - len(set(ids))

    counts: dict[str, int] = {}
    for r in records:
        if isinstance(r, dict) and isinstance(r.get("cuisine"), str):
            counts[r["cuisine"]] = counts.get(r["cuisine"], 0) + 1
    target = cfg["cuisines"]
    report["target_counts"] = {c: counts.get(c, 0) for c in target}
    report["other_cuisine_records"] = sum(v for k, v in counts.items() if k not in target)
    report["all_cuisine_counts"] = dict(sorted(counts.items()))
    for c in target:
        if counts.get(c, 0) == 0:
            raise DataError(f"Target cuisine '{c}' has no records in {path}")
    return records, report


def print_dataset_report(report: dict, log: logging.Logger | None = None) -> None:
    """Print the mandatory first-action dataset inspection."""
    log = log or get_logger()
    log.info("Dataset inspection: %s", report["path"])
    log.info("  exists=%s size=%s bytes json_parse_ok=%s", report["exists"], f"{report['size_bytes']:,}",
             report["json_parse_ok"])
    log.info("  total records=%s  sha256=%s", f"{report['total_records']:,}", report["sha256"][:16] + "...")
    for c, n in report["target_counts"].items():
        log.info("  %-8s %s", c, f"{n:,}")
    log.info("  other cuisines=%s  malformed=%d  duplicate ids=%d", f"{report['other_cuisine_records']:,}",
             report["malformed_records"], report["duplicate_ids"])
