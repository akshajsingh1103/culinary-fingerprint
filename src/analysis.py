"""Hubs, universal vs cuisine-specific hubs, Louvain communities and bridges (Deliverable 3).

Definitions
-----------
* Weighted degree (strength): Σ max(NPMI, 0) over incident backbone edges.
* Betweenness: normalised weighted betweenness on the undirected backbone, distance = 1 / max(NPMI, 0)
  (edges with zero strength are excluded from shortest paths).
* Hub: top-10 by weighted degree in a cuisine backbone. Ranks: value desc, ties alphabetical.
* Universal hub: top-10 in all three cuisines; two-cuisine: exactly two; cuisine-specific: exactly one.
* Bridge(a; A, B) = min(Strength_A(a), Strength_B(a)) for ingredients present as nodes in both backbones.
  This measures shared structural importance, not cultural exchange or causality.

Usage::

    python src/analysis.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from src.common import get_logger, load_config, rpath, write_json
from src.graph_utils import backbone_graph, louvain, rank_desc, strengths, weighted_betweenness

log = get_logger()
HUB_COLUMNS = ["cuisine", "ingredient", "weighted_degree", "weighted_degree_rank", "betweenness", "betweenness_rank",
               "degree", "prevalence", "top10_weighted_degree", "top10_betweenness"]
COMM_COLUMNS = ["cuisine", "community_id", "ingredient", "community_size", "within_community_rank",
                "within_community_weighted_degree", "in_top5", "top5_members", "community_label", "label_type"]
BRIDGE_COLUMNS = ["cuisine_pair", "ingredient", "weighted_degree_a", "weighted_degree_b", "bridge_score",
                  "prevalence_a", "prevalence_b", "rank"]
LABEL_TYPE = "data-derived descriptive label (not a validated semantic classification)"


# --------------------------------------------------------------------------- computations
def compute_hubs(bb: pd.DataFrame, nodes: pd.DataFrame, cuisine: str, top_n: int = 10) -> tuple[pd.DataFrame, dict]:
    """Top-n by weighted degree ∪ top-n by betweenness, with both metrics and full ranks."""
    G = backbone_graph(bb, nodes)
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=HUB_COLUMNS), {"strength": {}, "betweenness": {}, "G": G}
    s = strengths(G)
    btw = weighted_betweenness(G)
    rs, rb = rank_desc(s), rank_desc(btw)
    sel = sorted(set(k for k, r in rs.items() if r <= top_n) | set(k for k, r in rb.items() if r <= top_n))
    rows = [{"cuisine": cuisine, "ingredient": k, "weighted_degree": s[k], "weighted_degree_rank": rs[k],
             "betweenness": btw.get(k, 0.0), "betweenness_rank": rb[k], "degree": G.degree(k),
             "prevalence": G.nodes[k]["prevalence"], "top10_weighted_degree": rs[k] <= top_n,
             "top10_betweenness": rb[k] <= top_n} for k in sel]
    df = pd.DataFrame(rows, columns=HUB_COLUMNS)
    df["_o"] = np.where(df["top10_weighted_degree"], df["weighted_degree_rank"], top_n + df["betweenness_rank"])
    df = df.sort_values(["_o", "ingredient"]).drop(columns="_o").reset_index(drop=True)
    return df, {"strength": s, "betweenness": btw, "G": G, "rank_strength": rs, "rank_betweenness": rb}


def classify_hubs(hub_info: dict, prevalence_all: pd.DataFrame, cuisines: list[str], top_n: int = 10) -> pd.DataFrame:
    """Universal / two-cuisine / cuisine-specific classification of top-n weighted-degree hubs."""
    tops = {c: {k for k, r in hub_info[c].get("rank_strength", {}).items() if r <= top_n} for c in cuisines}
    union = sorted(set().union(*tops.values()))
    rows = []
    for ing in union:
        inc = [c for c in cuisines if ing in tops[c]]
        cls = {3: "universal", 2: "two-cuisine", 1: "cuisine-specific"}[len(inc)] if len(cuisines) == 3 else str(len(inc))
        row = {"ingredient": ing, "classification": cls, "n_cuisines_top10": len(inc), "top10_in": ";".join(inc)}
        for c in cuisines:
            rs = hub_info[c].get("rank_strength", {})
            pa = prevalence_all[(prevalence_all["cuisine"] == c) & (prevalence_all["ingredient"] == ing)]
            row[f"rank_{c}"] = rs.get(ing, np.nan)
            row[f"weighted_degree_{c}"] = hub_info[c].get("strength", {}).get(ing, np.nan)
            row[f"prevalence_{c}"] = float(pa["prevalence"].iloc[0]) if len(pa) else 0.0
            row[f"in_backbone_{c}"] = ing in rs
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        order = {"universal": 0, "two-cuisine": 1, "cuisine-specific": 2}
        df["_o"] = df["classification"].map(order)
        df = df.sort_values(["_o", "top10_in", "ingredient"]).drop(columns="_o").reset_index(drop=True)
    return df


def compute_communities(bb: pd.DataFrame, nodes: pd.DataFrame, cuisine: str, seed: int, resolution: float,
                        k: int = 5) -> tuple[pd.DataFrame, dict]:
    """Louvain communities with within-community weighted-degree ranking and top-k members."""
    G = backbone_graph(bb, nodes)
    part = louvain(G, seed, resolution)
    if not part:
        return pd.DataFrame(columns=COMM_COLUMNS), part
    groups: dict[int, list] = {}
    for n, c in part.items():
        groups.setdefault(c, []).append(n)
    rows = []
    for cid in sorted(groups):
        mem = groups[cid]
        mset = set(mem)
        wd = {n: float(sum(d["weight"] for _, nb, d in G.edges(n, data=True) if nb in mset)) for n in mem}
        rk = rank_desc(wd)
        top = sorted(mem, key=lambda n: rk[n])[:k]
        label = " / ".join(top[:3])
        for n in sorted(mem, key=lambda n: rk[n]):
            rows.append({"cuisine": cuisine, "community_id": cid, "ingredient": n, "community_size": len(mem),
                         "within_community_rank": rk[n], "within_community_weighted_degree": wd[n],
                         "in_top5": rk[n] <= k, "top5_members": "; ".join(top), "community_label": label,
                         "label_type": LABEL_TYPE})
    return pd.DataFrame(rows, columns=COMM_COLUMNS), part


def compute_bridges(strength: dict, prevalence: dict, cuisines: list[str], top_n: int = 5) -> tuple[pd.DataFrame, dict]:
    """Bridge candidates for every cuisine pair; returns top-n per pair and candidate counts."""
    rows, counts = [], {}
    for A, B in itertools.combinations(cuisines, 2):
        cand = sorted(set(strength[A]) & set(strength[B]))
        scored = sorted(((min(strength[A][x], strength[B][x]), x) for x in cand), key=lambda t: (-t[0], t[1]))
        counts[f"{A}-{B}"] = len(cand)
        for r, (sc, x) in enumerate(scored[:top_n], start=1):
            rows.append({"cuisine_pair": f"{A}-{B}", "ingredient": x, "weighted_degree_a": strength[A][x],
                         "weighted_degree_b": strength[B][x], "bridge_score": sc,
                         "prevalence_a": prevalence[A].get(x, 0.0), "prevalence_b": prevalence[B].get(x, 0.0),
                         "rank": r})
    return pd.DataFrame(rows, columns=BRIDGE_COLUMNS), counts


# --------------------------------------------------------------------------- figures
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_edges(summary: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary["edges_before"], 0.4, label="raw edges (eligible nodes)", color="#9ecae1")
    ax.bar(x + 0.2, summary["edges_after"], 0.4, label="backbone edges (z>3, co>=5)", color="#08519c")
    for i, (b, a, p) in enumerate(zip(summary["edges_before"], summary["edges_after"], summary["pct_retained"])):
        ax.text(i + 0.2, a, f"{p:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [c.title() for c in summary["cuisine"]])
    ax.set_yscale("log")
    ax.set_ylabel("edges (log scale)")
    ax.set_title("Edges before and after backbone filtering")
    ax.legend(fontsize=7)
    _save(fig, path)


def fig_hubs(hubs: dict, cuisines: list[str], path: Path, top_n: int = 10):
    fig, axes = plt.subplots(1, len(cuisines), figsize=(10, 3.6))
    for ax, c in zip(np.atleast_1d(axes), cuisines):
        h = hubs[c]
        h = h[h["top10_weighted_degree"]].sort_values("weighted_degree_rank").head(top_n) if len(h) else h
        ax.barh(h["ingredient"][::-1], h["weighted_degree"][::-1], color="#3182bd")
        ax.set_title(f"{c.title()}: top-{top_n} weighted degree", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Σ max(NPMI,0)", fontsize=8)
    _save(fig, path)


def fig_venn(cls: pd.DataFrame, cuisines: list[str], path: Path):
    sets = {c: set(cls.loc[cls[f"rank_{c}"] <= 10, "ingredient"]) if len(cls) else set() for c in cuisines}
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    centers = [(-0.6, 0.35), (0.6, 0.35), (0.0, -0.6)]
    colors = ["#e6550d", "#31a354", "#3182bd"]
    for (x, y), col, c in zip(centers, colors, cuisines):
        ax.add_patch(Circle((x, y), 1.0, alpha=0.25, color=col))
        ax.text(x * 1.9, y * 1.9 + (0.1 if y > 0 else -0.25), c.title(), ha="center", fontsize=10, weight="bold")
    A, B, C = (sets[c] for c in cuisines)
    regions = {(-0.95, 0.6): A - B - C, (0.95, 0.6): B - A - C, (0.0, -1.05): C - A - B, (0.0, 0.75): (A & B) - C,
               (-0.6, -0.3): (A & C) - B, (0.6, -0.3): (B & C) - A, (0.0, 0.0): A & B & C}
    for (x, y), s in regions.items():
        ax.text(x, y, str(len(s)), ha="center", va="center", fontsize=12)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Overlap of top-10 weighted-degree hubs", fontsize=10)
    _save(fig, path)


def fig_communities(comms: dict, cuisines: list[str], path: Path):
    fig, ax = plt.subplots(figsize=(6, 3.0))
    width = 0.8 / len(cuisines)
    maxk = 0
    for i, c in enumerate(cuisines):
        df = comms[c]
        sizes = df.drop_duplicates("community_id").sort_values("community_id")["community_size"].to_numpy() if len(df) else np.array([])
        maxk = max(maxk, len(sizes))
        ax.bar(np.arange(len(sizes)) + i * width, sizes, width, label=c.title())
    ax.set_xlabel("community (ordered by size)")
    ax.set_ylabel("ingredients")
    ax.set_title("Louvain community sizes")
    ax.set_xticks(np.arange(maxk) + width, [str(k) for k in range(maxk)], fontsize=7)
    ax.legend(fontsize=7)
    _save(fig, path)


def fig_bridges(bridges: pd.DataFrame, path: Path):
    pairs = list(dict.fromkeys(bridges["cuisine_pair"])) if len(bridges) else []
    fig, axes = plt.subplots(1, max(len(pairs), 1), figsize=(10, 2.8))
    for ax, p in zip(np.atleast_1d(axes), pairs):
        b = bridges[bridges["cuisine_pair"] == p].sort_values("rank")
        ax.barh(b["ingredient"][::-1], b["bridge_score"][::-1], color="#756bb1")
        ax.set_title(p.replace("-", " – ").title(), fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("bridge score", fontsize=8)
    _save(fig, path)


# --------------------------------------------------------------------------- driver
def run(cfg: dict) -> dict:
    cuisines = cfg["cuisines"]
    d1, d2, d3 = rpath(cfg["outputs"]["d1"]), rpath(cfg["outputs"]["d2"]), rpath(cfg["outputs"]["d3"])
    figs = d3 / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(d1 / "nodes.csv")
    prev_all = pd.read_csv(d1 / "ingredient_prevalence_all.csv")
    summary = pd.read_csv(d2 / "backbone_summary.csv")
    top_n = int(cfg["report"]["top_n_hubs"])
    hubs, info, comms, prev = {}, {}, {}, {}
    for c in cuisines:
        bb = pd.read_csv(d2 / f"backbone_{c}.csv")
        nc = nodes[nodes["cuisine"] == c]
        hubs[c], info[c] = compute_hubs(bb, nc, c, top_n)
        hubs[c].to_csv(d3 / f"hubs_{c}.csv", index=False, encoding="utf-8")
        comms[c], _ = compute_communities(bb, nc, c, int(cfg["community"]["seed"]),
                                          float(cfg["community"]["resolution"]), int(cfg["report"]["community_top_k"]))
        comms[c].to_csv(d3 / f"communities_{c}.csv", index=False, encoding="utf-8")
        prev[c] = dict(zip(nc["ingredient"], nc["prevalence"]))
    cls = classify_hubs(info, prev_all, cuisines, top_n)
    cls.to_csv(d3 / "universal_vs_specific_hubs.csv", index=False, encoding="utf-8")
    bridges, counts = compute_bridges({c: info[c]["strength"] for c in cuisines}, prev, cuisines,
                                      int(cfg["report"]["top_n_bridges"]))
    bridges.to_csv(d3 / "bridges.csv", index=False, encoding="utf-8")

    fig_edges(summary, figs / "fig1_backbone_edges.png")
    fig_hubs(hubs, cuisines, figs / "fig2_top_hubs.png", top_n)
    fig_venn(cls, cuisines, figs / "fig3_hub_overlap.png")
    fig_communities(comms, cuisines, figs / "fig4_community_sizes.png")
    fig_bridges(bridges, figs / "fig5_bridges.png")

    meta = {"bridge_candidate_counts": counts,
            "hub_classification_counts": cls["classification"].value_counts().to_dict() if len(cls) else {},
            "n_communities": {c: int(comms[c]["community_id"].nunique()) if len(comms[c]) else 0 for c in cuisines},
            "singleton_communities": {c: int((comms[c].drop_duplicates("community_id")["community_size"] == 1).sum())
                                      if len(comms[c]) else 0 for c in cuisines},
            "betweenness": "weighted, normalised, distance = 1/max(NPMI,0); zero-strength edges excluded"}
    write_json(meta, d3 / "analysis_summary.json")
    for c in cuisines:
        top = hubs[c][hubs[c]["top10_weighted_degree"]]["ingredient"].tolist()[:5] if len(hubs[c]) else []
        log.info("Hubs %-8s top-5 by weighted degree: %s", c, top)
    log.info("Hub classes: %s | bridge candidates: %s", meta["hub_classification_counts"], counts)
    return {"hubs": hubs, "classification": cls, "communities": comms, "bridges": bridges, "meta": meta}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compute hubs, communities and bridges from the backbones (D3 tables).")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    run(load_config(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
