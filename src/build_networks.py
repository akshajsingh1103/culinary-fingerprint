"""Ingredient node tables and PMI/NPMI co-occurrence edge lists (Deliverable 1).

Descriptive networks (D1-D3) use **all** selected recipes. The same functions are
reused by :mod:`src.evaluate` on the **training split only** to build the
predictive networks (D4); :func:`assert_training_only` guards that boundary.

Definitions (per cuisine, binary recipe-level presence):
    N = #recipes, n(a) = #recipes containing a, n(a,b) = #recipes containing both.
    PMI  = log2( P(a,b) / (P(a) P(b)) ) = log2( n(a,b) N / (n(a) n(b)) )
    NPMI = PMI / (-log2 P(a,b))
Degenerate case P(a,b) = 1 (both ingredients in every recipe): PMI = 0 and the
ratio is 0/0; we set NPMI = 1, its limiting value for perfect co-occurrence
(Bouma, 2009). Zero co-occurrence pairs are not edges.

Usage::

    python src/build_networks.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common import get_logger, load_config, rpath, write_json

log = get_logger()
NODE_COLUMNS = ["ingredient", "cuisine", "recipe_count", "prevalence"]
EDGE_COLUMNS = ["ingredient_a", "ingredient_b", "co_count", "PMI", "NPMI"]


def recipes_for(recipes: list[dict], cuisine: str) -> list[dict]:
    """Recipes of one cuisine, sorted by id."""
    return sorted((r for r in recipes if r["cuisine"] == cuisine), key=lambda r: r["id"])


def incidence_matrix(recipes: list[dict], vocab: list[str] | None = None) -> tuple[sp.csr_matrix, list[str]]:
    """Binary recipe x ingredient CSR matrix. Ingredients outside ``vocab`` are dropped.

    ``vocab`` defaults to all ingredients present, sorted alphabetically.
    """
    if vocab is None:
        vocab = sorted({x for r in recipes for x in r["ingredients"]})
    index = {v: i for i, v in enumerate(vocab)}
    indptr = [0]
    indices: list[int] = []
    for r in recipes:
        cols = sorted({index[x] for x in r["ingredients"] if x in index})
        indices.extend(cols)
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.int32)
    B = sp.csr_matrix((data, np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
                      shape=(len(recipes), len(vocab)))
    return B, list(vocab)


def ingredient_counts(recipes: list[dict]) -> Counter:
    return Counter(x for r in recipes for x in set(r["ingredients"]))


def eligible_vocab(recipes: list[dict], min_count: int) -> list[str]:
    """Ingredients occurring in at least ``min_count`` recipes (``>=``), sorted."""
    return sorted(k for k, v in ingredient_counts(recipes).items() if v >= min_count)


def node_table(recipes: list[dict], cuisine: str, min_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (eligible nodes table, all-ingredient prevalence table) for one cuisine."""
    N = len(recipes)
    cnt = ingredient_counts(recipes)
    rows = [{"ingredient": k, "cuisine": cuisine, "recipe_count": int(v), "prevalence": 100.0 * v / N,
             "eligible": bool(v >= min_count)} for k, v in sorted(cnt.items())]
    allp = pd.DataFrame(rows, columns=NODE_COLUMNS + ["eligible"])
    nodes = allp[allp["eligible"]][NODE_COLUMNS].reset_index(drop=True)
    return nodes, allp


def pmi_npmi(co: np.ndarray, na: np.ndarray, nb: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised PMI and NPMI (log base 2) for co-occurrence counts > 0."""
    co = np.asarray(co, dtype=np.float64)
    na = np.asarray(na, dtype=np.float64)
    nb = np.asarray(nb, dtype=np.float64)
    if np.any(co <= 0):
        raise ValueError("pmi_npmi requires strictly positive co-occurrence counts")
    pmi = np.log2(co) + np.log2(N) - np.log2(na) - np.log2(nb)
    h = -(np.log2(co) - np.log2(N))  # -log2 P(a,b) >= 0
    npmi = np.empty_like(pmi)
    full = co >= N  # P(a,b) == 1 -> limiting convention NPMI = 1
    npmi[full] = 1.0
    npmi[~full] = pmi[~full] / h[~full]
    npmi = np.clip(npmi, -1.0, 1.0)  # guard tiny floating-point overshoot only
    return pmi, npmi


def cooccurrence(B: sp.csr_matrix) -> sp.coo_matrix:
    """Upper-triangular (k=1) co-occurrence counts from B^T B."""
    C = (B.T @ B).tocoo()
    mask = C.row < C.col
    return sp.coo_matrix((C.data[mask], (C.row[mask], C.col[mask])), shape=C.shape)


def edge_table(recipes: list[dict], vocab: list[str]) -> pd.DataFrame:
    """All co-occurring pairs among ``vocab`` with co_count, PMI, NPMI.

    Pairs are unordered with ``ingredient_a < ingredient_b`` (vocab is sorted, so row<col suffices).
    """
    if sorted(vocab) != list(vocab):
        raise ValueError("vocab must be sorted")
    N = len(recipes)
    B, vocab = incidence_matrix(recipes, vocab)
    if B.shape[1] < 2 or N == 0:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    n = np.asarray(B.sum(axis=0)).ravel()
    U = cooccurrence(B)
    keep = U.data > 0
    r, c, co = U.row[keep], U.col[keep], U.data[keep].astype(np.int64)
    pmi, npmi = pmi_npmi(co, n[r], n[c], N)
    v = np.asarray(vocab, dtype=object)
    df = pd.DataFrame({"ingredient_a": v[r], "ingredient_b": v[c], "co_count": co, "PMI": pmi, "NPMI": npmi})
    return df.sort_values(["ingredient_a", "ingredient_b"]).reset_index(drop=True)


def build_cuisine_network(recipes: list[dict], cuisine: str, min_count: int) -> dict:
    """Nodes, all-prevalence table and edges for one cuisine's recipes."""
    rc = recipes_for(recipes, cuisine)
    nodes, allp = node_table(rc, cuisine, min_count)
    edges = edge_table(rc, nodes["ingredient"].tolist())
    return {"cuisine": cuisine, "n_recipes": len(rc), "nodes": nodes, "all_prevalence": allp, "edges": edges,
            "recipe_ids": [r["id"] for r in rc]}


def assert_training_only(recipes: list[dict], train_ids, test_ids) -> None:
    """Raise if any recipe passed to the predictive builder is not a training recipe."""
    ids = {r["id"] for r in recipes}
    train_ids, test_ids = set(train_ids), set(test_ids)
    if train_ids & test_ids:
        raise AssertionError("train/test ID sets overlap")
    leaked = ids & test_ids
    if leaked:
        raise AssertionError(f"{len(leaked)} test recipes passed to the predictive network builder")
    if not ids <= train_ids:
        raise AssertionError("recipes outside the training split passed to the predictive builder")


def build_predictive_networks(train_recipes: list[dict], cuisines: list[str], min_count: int,
                              train_ids, test_ids) -> dict:
    """Training-only networks for Deliverable 4, with an explicit leakage guard."""
    assert_training_only(train_recipes, train_ids, test_ids)
    return {c: build_cuisine_network(train_recipes, c, min_count) for c in cuisines}


def write_d1(nets: dict, cfg: dict) -> None:
    """Write nodes.csv, edges_<cuisine>.csv and the all-ingredient prevalence table."""
    d1 = rpath(cfg["outputs"]["d1"])
    d1.mkdir(parents=True, exist_ok=True)
    pd.concat([nets[c]["nodes"] for c in cfg["cuisines"]], ignore_index=True)[NODE_COLUMNS].to_csv(
        d1 / "nodes.csv", index=False, encoding="utf-8")
    pd.concat([nets[c]["all_prevalence"] for c in cfg["cuisines"]], ignore_index=True).to_csv(
        d1 / "ingredient_prevalence_all.csv", index=False, encoding="utf-8")
    summary = {}
    for c in cfg["cuisines"]:
        nets[c]["edges"][EDGE_COLUMNS].to_csv(d1 / f"edges_{c}.csv", index=False, encoding="utf-8")
        e = nets[c]["edges"]
        summary[c] = {"recipes": nets[c]["n_recipes"], "eligible_nodes": len(nets[c]["nodes"]),
                      "distinct_ingredients": len(nets[c]["all_prevalence"]), "edges": len(e),
                      "edges_negative_npmi": int((e["NPMI"] < 0).sum()) if len(e) else 0}
    write_json({"scope": "descriptive (all selected recipes)", "min_ingredient_count": cfg["network"]["min_ingredient_count"],
                "per_cuisine": summary}, d1 / "d1_summary.json")


def run(cfg: dict, recipes: list[dict] | None = None) -> dict:
    """Build descriptive (full-data) networks and write Deliverable 1."""
    if recipes is None:
        from src.preprocess import load_processed
        recipes, _ = load_processed(cfg)
    mc = int(cfg["network"]["min_ingredient_count"])
    nets = {c: build_cuisine_network(recipes, c, mc) for c in cfg["cuisines"]}
    write_d1(nets, cfg)
    for c in cfg["cuisines"]:
        n = nets[c]
        log.info("D1 %-8s recipes=%5d distinct=%5d eligible nodes=%4d edges=%6d", c, n["n_recipes"],
                 len(n["all_prevalence"]), len(n["nodes"]), len(n["edges"]))
    return nets


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build descriptive ingredient nodes and PMI/NPMI edge lists (D1).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--min-count", type=int, default=None, help="Override minimum recipe count (default 20)")
    args = ap.parse_args(argv)
    run(load_config(args.config, {"network.min_ingredient_count": args.min_count}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
