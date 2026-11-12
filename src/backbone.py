"""Null-model statistics and statistically validated backbones (Deliverable 2).

For every observed edge among eligible ingredients:
    expected_mean = mean of the null co-counts across the chain's samples
    expected_std  = sample standard deviation (ddof = 1)
    z_score       = (co_count - expected_mean) / expected_std      (only if expected_std > 0)

Zero-variance policy (conservative): if ``expected_std == 0`` the z-score is left
undefined (blank in ``null_stats_<cuisine>.csv``), the pair is flagged
``zero_variance = True`` and it is **excluded** from the backbone. Counts are logged.

Backbone rule: ``z_score > 3`` (strict) **and** ``co_count >= 5``.
A z-score threshold is not a multiple-testing-corrected significance level.

Summary columns (``backbone_summary.csv``):
    nodes          – ingredients with >= 1 backbone edge (backbone graph nodes)
    edges_before   – raw co-occurrence edges among eligible ingredients (D1 edge list)
    edges_after    – backbone edges
    pct_retained   – 100 * edges_after / edges_before (0 if edges_before == 0)
    n_communities  – Louvain communities of the backbone graph (max(NPMI,0) weights)
    modularity     – weighted modularity of that partition

Usage::

    python src/backbone.py [--config config.yaml] [--n-null N] [--no-cache]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.build_networks import build_cuisine_network, incidence_matrix, recipes_for
from src.common import get_logger, is_debug_run, load_config, rpath, sha256_obj, write_json
from src.graph_utils import backbone_graph, louvain, modularity
from src.normalize import normalizer_fingerprint
from src.null_model import chain_seed, run_null_cached

log = get_logger()
BACKBONE_COLUMNS = ["ingredient_a", "ingredient_b", "co_count", "expected_mean", "expected_std", "z_score", "NPMI"]
SUMMARY_COLUMNS = ["cuisine", "nodes", "edges_before", "edges_after", "pct_retained", "n_communities", "modularity"]


def incidence_inputs(recipes: list[dict], cfg: dict, nets: dict | None = None) -> dict:
    """Per cuisine: (full binary incidence matrix, eligible column indices, #observed eligible pairs)."""
    out = {}
    mc = int(cfg["network"]["min_ingredient_count"])
    for c in cfg["cuisines"]:
        rc = recipes_for(recipes, c)
        net = nets[c] if nets else build_cuisine_network(recipes, c, mc)
        B, vocab = incidence_matrix(rc)
        pos = {v: i for i, v in enumerate(vocab)}
        elig = np.array([pos[v] for v in net["nodes"]["ingredient"]], dtype=np.int64)
        out[c] = (B, elig, len(net["edges"]))
    return out


def null_stats_table(edges: pd.DataFrame, null_res: dict, elig_vocab: list[str]) -> pd.DataFrame:
    """Attach expected_mean, expected_std, z_score and zero_variance flag to each observed edge."""
    idx = {v: i for i, v in enumerate(elig_vocab)}
    df = edges.copy()
    if not len(df):
        for col in ("expected_mean", "expected_std", "z_score"):
            df[col] = pd.Series(dtype=float)
        df["zero_variance"] = pd.Series(dtype=bool)
        return df
    ia = df["ingredient_a"].map(idx).to_numpy()
    ib = df["ingredient_b"].map(idx).to_numpy()
    if np.any(ia >= ib):
        raise AssertionError("edge endpoints not in sorted eligible order")
    obs = null_res["observed"][ia, ib]
    if not np.array_equal(obs.astype(np.int64), df["co_count"].to_numpy(np.int64)):
        raise AssertionError("null-model observed co-counts disagree with the edge list")
    mean = null_res["mean"][ia, ib].astype(float)
    std = null_res["std"][ia, ib].astype(float)
    zero = ~(std > 0)
    z = np.full(len(df), np.nan)
    z[~zero] = (df["co_count"].to_numpy(float)[~zero] - mean[~zero]) / std[~zero]
    df["expected_mean"] = mean
    df["expected_std"] = std
    df["z_score"] = z
    df["zero_variance"] = zero
    return df


def filter_backbone(stats: pd.DataFrame, z_threshold: float, min_co: int) -> pd.DataFrame:
    """Keep edges with finite z_score > z_threshold (strict), co_count >= min_co, std > 0."""
    if not len(stats):
        return pd.DataFrame(columns=BACKBONE_COLUMNS)
    m = (~stats["zero_variance"]) & np.isfinite(stats["z_score"]) & (stats["z_score"] > z_threshold) & \
        (stats["co_count"] >= min_co) & np.isfinite(stats["expected_mean"]) & np.isfinite(stats["NPMI"])
    return stats.loc[m, BACKBONE_COLUMNS].sort_values(["ingredient_a", "ingredient_b"]).reset_index(drop=True)


def summary_row(cuisine: str, nodes_df: pd.DataFrame, edges: pd.DataFrame, bb: pd.DataFrame, cfg: dict) -> dict:
    G = backbone_graph(bb, nodes_df)
    part = louvain(G, int(cfg["community"]["seed"]), float(cfg["community"]["resolution"]))
    before, after = len(edges), len(bb)
    return {"cuisine": cuisine, "nodes": G.number_of_nodes(), "edges_before": before, "edges_after": after,
            "pct_retained": (100.0 * after / before) if before else 0.0,
            "n_communities": len(set(part.values())),
            "modularity": modularity(G, part, float(cfg["community"]["resolution"])),
            "eligible_nodes": len(nodes_df)}


def build_backbones(recipes: list[dict], nets: dict, cfg: dict, scope: str, use_cache: bool,
                    dataset_sha: str) -> dict:
    """Run one Curveball chain per cuisine and filter backbones. ``scope``: descriptive|predictive."""
    nm = cfg["null_model"]
    cache_dir = rpath(cfg["data"]["cache_dir"]) if use_cache else None
    inputs = incidence_inputs(recipes, cfg, nets)
    out = {"stats": {}, "backbone": {}, "diagnostics": {}, "summary": []}
    for idx, c in enumerate(cfg["cuisines"]):
        B, elig, _ = inputs[c]
        nrec = B.shape[0]
        burn = int(nm["burn_in_multiplier"]) * nrec
        thin = int(nm["thinning_multiplier"]) * nrec
        if int(nm["burn_in_multiplier"]) < 5:
            raise ValueError("burn_in_multiplier must be >= 5 (requirement: burn-in >= 5 x n_recipes trades)")
        seed = chain_seed(cfg["seed"], idx, scope)
        payload = {"dataset_sha256": dataset_sha, "scope": scope, "cuisine": c,
                   "recipe_ids": sha256_obj(nets[c]["recipe_ids"]), "normalizer": normalizer_fingerprint(),
                   "min_count": cfg["network"]["min_ingredient_count"], "global_seed": cfg["seed"]}
        t0 = time.perf_counter()
        res, hit = run_null_cached(B, elig, int(nm["n_randomizations"]), burn, thin, seed, cache_dir, payload,
                                   int(nm.get("max_acf_lag", 20)))
        dt = time.perf_counter() - t0
        elig_vocab = nets[c]["nodes"]["ingredient"].tolist()
        stats = null_stats_table(nets[c]["edges"], res, elig_vocab)
        bb = filter_backbone(stats, float(cfg["backbone"]["z_threshold"]), int(cfg["backbone"]["min_co_count"]))
        diag = dict(res["diagnostics"])
        diag["wall_seconds_including_cache"] = round(dt, 3)
        diag["zero_variance_pairs_observed_edges"] = int(stats["zero_variance"].sum()) if len(stats) else 0
        diag["zero_variance_pairs_with_co_count_ge_min"] = int(
            (stats["zero_variance"] & (stats["co_count"] >= int(cfg["backbone"]["min_co_count"]))).sum()) if len(stats) else 0
        diag["edges_z_eq_threshold"] = int((stats["z_score"] == float(cfg["backbone"]["z_threshold"])).sum()) if len(stats) else 0
        out["stats"][c], out["backbone"][c], out["diagnostics"][c] = stats, bb, diag
        out["summary"].append(summary_row(c, nets[c]["nodes"], nets[c]["edges"], bb, cfg))
        log.info("[%s] %-8s chain %s in %.1fs | trades attempted=%d changed=%d | ESS(S)=%.0f lag1-ACF(S)=%.3f | "
                 "edges %d -> %d | zero-var excluded=%d", scope, c, "CACHED" if hit else "ran", dt,
                 diag["attempted_trades"], diag["changed_trades"], diag.get("ess_S", float("nan")),
                 (diag.get("acf_S") or [float("nan")])[0], len(nets[c]["edges"]), len(bb),
                 diag["zero_variance_pairs_observed_edges"])
    out["summary_df"] = pd.DataFrame(out["summary"])
    return out


def null_metadata(cfg: dict, res: dict, scope: str) -> dict:
    return {"scope": scope, "run_mode": "DEBUG" if is_debug_run(cfg) else "FULL",
            "n_randomizations": int(cfg["null_model"]["n_randomizations"]), "null_model_config": cfg["null_model"],
            "backbone_config": cfg["backbone"], "seed": cfg["seed"], "per_cuisine": res["diagnostics"]}


def run(cfg: dict, recipes=None, nets=None, use_cache: bool = True, dataset_sha: str | None = None) -> dict:
    """Descriptive backbones on all selected recipes; writes Deliverable 2 CSVs."""
    if recipes is None or dataset_sha is None:
        from src.preprocess import load_processed
        recipes, dataset_sha = load_processed(cfg)
    if nets is None:
        mc = int(cfg["network"]["min_ingredient_count"])
        nets = {c: build_cuisine_network(recipes, c, mc) for c in cfg["cuisines"]}
    res = build_backbones(recipes, nets, cfg, "descriptive", use_cache, dataset_sha)
    d2 = rpath(cfg["outputs"]["d2"])
    d2.mkdir(parents=True, exist_ok=True)
    for c in cfg["cuisines"]:
        res["backbone"][c][BACKBONE_COLUMNS].to_csv(d2 / f"backbone_{c}.csv", index=False, encoding="utf-8")
        res["stats"][c].to_csv(d2 / f"null_stats_{c}.csv", index=False, encoding="utf-8")
    summ = res["summary_df"]
    summ[SUMMARY_COLUMNS + ["eligible_nodes"]].to_csv(d2 / "backbone_summary.csv", index=False, encoding="utf-8")
    meta = null_metadata(cfg, res, "descriptive")
    meta["dataset_sha256"] = dataset_sha
    write_json(meta, d2 / "null_model_metadata.json")
    write_json(cfg, d2 / "effective_config.json")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the Curveball null model and extract backbones (D2 CSVs).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-null", type=int, default=None, help="DEBUG ONLY: number of null samples (default 1000)")
    ap.add_argument("--no-cache", action="store_true", help="Ignore and do not write the null-model cache")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, {"seed": args.seed, "null_model.n_randomizations": args.n_null})
    if is_debug_run(cfg):
        log.warning("DEBUG run: n_randomizations=%d != 1000; outputs are labelled DEBUG",
                    cfg["null_model"]["n_randomizations"])
    run(cfg, use_cache=not args.no_cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
