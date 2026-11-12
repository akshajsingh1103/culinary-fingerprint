"""Graph helpers shared by backbone summary, visualisation and analysis.

Conventions
-----------
* Association strength ``w = max(NPMI, 0)``; negative NPMI is preserved in CSVs
  but contributes zero strength (used for weighted degree and Louvain).
* Weighted betweenness uses distance ``d = 1 / w`` on edges with ``w > 0``;
  edges with ``w == 0`` are omitted from the distance graph (infinite distance).
* Louvain: ``networkx.community.louvain_communities`` with fixed seed on ``w``.
  Communities are relabelled deterministically: by size (desc), then by the
  alphabetically first member, so IDs are stable across runs.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def backbone_graph(backbone: pd.DataFrame, nodes: pd.DataFrame | None = None) -> nx.Graph:
    """Undirected graph from a backbone table, inserted in sorted order for determinism."""
    G = nx.Graph()
    if nodes is not None and len(backbone):
        prev = dict(zip(nodes["ingredient"], nodes["prevalence"]))
    else:
        prev = {}
    df = backbone.sort_values(["ingredient_a", "ingredient_b"]) if len(backbone) else backbone
    for a, b, npmi, co in zip(df.get("ingredient_a", []), df.get("ingredient_b", []), df.get("NPMI", []),
                              df.get("co_count", [])):
        w = max(float(npmi), 0.0)
        G.add_edge(a, b, npmi=float(npmi), weight=w, co_count=int(co), distance=(1.0 / w) if w > 0 else np.inf)
    for n in G.nodes:
        G.nodes[n]["prevalence"] = float(prev.get(n, np.nan))
    return G


def strengths(G: nx.Graph) -> dict:
    """Weighted degree = sum of max(NPMI,0) over incident edges."""
    return {n: float(sum(d["weight"] for _, _, d in G.edges(n, data=True))) for n in G.nodes}


def weighted_betweenness(G: nx.Graph) -> dict:
    """Normalised weighted betweenness with distance 1/w on positive-strength edges."""
    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes))
    H.add_edges_from((a, b, {"distance": d["distance"]}) for a, b, d in G.edges(data=True) if d["weight"] > 0)
    if H.number_of_nodes() == 0:
        return {}
    return nx.betweenness_centrality(H, weight="distance", normalized=True)


def louvain(G: nx.Graph, seed: int, resolution: float = 1.0) -> dict:
    """Return {node: community_id} with deterministic relabelling. Empty graph -> {}."""
    if G.number_of_nodes() == 0:
        return {}
    if G.size(weight="weight") <= 0:
        comms = [{n} for n in G.nodes]
    else:
        comms = nx.community.louvain_communities(G, weight="weight", seed=seed, resolution=resolution)
    comms = sorted((sorted(c) for c in comms), key=lambda c: (-len(c), c[0]))
    return {n: i for i, c in enumerate(comms) for n in c}


def modularity(G: nx.Graph, part: dict, resolution: float = 1.0) -> float:
    """Weighted modularity of a partition; 0.0 for graphs with no positive weight."""
    if G.number_of_nodes() == 0 or G.size(weight="weight") <= 0:
        return 0.0
    groups: dict[int, set] = {}
    for n, c in part.items():
        groups.setdefault(c, set()).add(n)
    return float(nx.community.modularity(G, list(groups.values()), weight="weight", resolution=resolution))


def rank_desc(values: dict) -> dict:
    """Rank 1..n by value descending, ties broken alphabetically by key."""
    order = sorted(values, key=lambda k: (-values[k], k))
    return {k: i + 1 for i, k in enumerate(order)}
