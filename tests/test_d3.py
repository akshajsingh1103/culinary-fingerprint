"""Deliverable 3 acceptance tests (hubs, communities, bridges, figures, MD & PDF report)."""
from __future__ import annotations

import math
import re

import networkx as nx
import numpy as np
import pandas as pd
from pypdf import PdfReader

from src.analysis import classify_hubs, compute_bridges, compute_communities, compute_hubs
from src.graph_utils import backbone_graph, louvain, strengths, weighted_betweenness


# --------------------------------------------------------------------------- unit tests (empty / disconnected graphs)
def test_analysis_handles_empty_and_disconnected_graphs():
    empty_bb = pd.DataFrame(columns=["ingredient_a", "ingredient_b", "co_count", "expected_mean",
                                     "expected_std", "z_score", "NPMI"])
    empty_nodes = pd.DataFrame(columns=["ingredient", "cuisine", "recipe_count", "prevalence"])
    hubs, info = compute_hubs(empty_bb, empty_nodes, "indian", top_n=10)
    assert len(hubs) == 0 and info["G"].number_of_nodes() == 0
    comms, part = compute_communities(empty_bb, empty_nodes, "indian", seed=42, resolution=1.0)
    assert len(comms) == 0 and part == {}

    # Disconnected 2-component graph with < 10 nodes and < 5 bridge candidates
    disc_bb = pd.DataFrame([
        {"ingredient_a": "a", "ingredient_b": "b", "co_count": 10, "expected_mean": 2.0,
         "expected_std": 1.0, "z_score": 8.0, "NPMI": 0.5},
        {"ingredient_a": "c", "ingredient_b": "d", "co_count": 8, "expected_mean": 2.0,
         "expected_std": 1.0, "z_score": 6.0, "NPMI": 0.4},
    ])
    disc_nodes = pd.DataFrame([
        {"ingredient": "a", "cuisine": "indian", "recipe_count": 30, "prevalence": 10.0},
        {"ingredient": "b", "cuisine": "indian", "recipe_count": 30, "prevalence": 10.0},
        {"ingredient": "c", "cuisine": "indian", "recipe_count": 25, "prevalence": 8.0},
        {"ingredient": "d", "cuisine": "indian", "recipe_count": 25, "prevalence": 8.0},
    ])
    hubs_d, info_d = compute_hubs(disc_bb, disc_nodes, "indian", top_n=10)
    assert len(hubs_d) == 4  # never invents missing nodes when < 10 exist
    comms_d, _ = compute_communities(disc_bb, disc_nodes, "indian", seed=42, resolution=1.0)
    assert comms_d["community_id"].nunique() == 2

    # Bridges when fewer than 5 shared candidates exist
    str_map = {"indian": {"a": 1.5, "b": 0.8}, "italian": {"a": 2.0}, "mexican": {"b": 0.5}}
    prev_map = {"indian": {"a": 10.0, "b": 5.0}, "italian": {"a": 12.0}, "mexican": {"b": 4.0}}
    br, counts = compute_bridges(str_map, prev_map, ["indian", "italian", "mexican"], top_n=5)
    assert counts["indian-italian"] == 1
    assert counts["indian-mexican"] == 1
    assert counts["italian-mexican"] == 0
    assert len(br) == 2


# --------------------------------------------------------------------------- real Deliverable 3 artifacts
def test_d3_report_files_and_sections(d3_dir):
    md_path = d3_dir / "report.md"
    pdf_path = d3_dir / "report.pdf"
    assert md_path.exists()
    assert pdf_path.exists()

    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)
    assert 3 <= n_pages <= 4, f"Expected 3-4 pages in report.pdf, got {n_pages}"

    md_text = md_path.read_text(encoding="utf-8")
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    required_sections = [
        "(a) Top-10 hubs",
        "(b) Universal and cuisine-specific hubs",
        "(c) Louvain communities",
        "(d) Strongest bridge ingredients",
    ]
    for sec in required_sections:
        assert sec in md_text, f"Missing section {sec!r} in report.md"
        assert sec in pdf_text, f"Missing section {sec!r} in report.pdf"

    # Referenced figures exist
    for m in re.finditer(r"\(figures/([^)]+\.png)\)", md_text):
        fig_file = d3_dir / "figures" / m.group(1)
        assert fig_file.exists(), f"Referenced figure does not exist: {fig_file}"


def test_d3_hub_rankings_agree_with_graph(cfg, d1_dir, d2_dir, d3_dir):
    nodes_all = pd.read_csv(d1_dir / "nodes.csv")
    for c in cfg["cuisines"]:
        bb = pd.read_csv(d2_dir / f"backbone_{c}.csv")
        nc = nodes_all[nodes_all["cuisine"] == c]
        G = backbone_graph(bb, nc)
        s = strengths(G)
        btw = weighted_betweenness(G)
        hubs = pd.read_csv(d3_dir / f"hubs_{c}.csv")
        assert (hubs["top10_weighted_degree"].sum() == 10)
        assert (hubs["top10_betweenness"].sum() == 10)
        for _, row in hubs.iterrows():
            ing = row["ingredient"]
            assert math.isclose(row["weighted_degree"], s[ing], rel_tol=1e-9)
            assert math.isclose(row["betweenness"], btw[ing], rel_tol=1e-9)


def test_d3_communities_and_bridges_agree_with_definitions(cfg, d1_dir, d2_dir, d3_dir):
    nodes_all = pd.read_csv(d1_dir / "nodes.csv")
    str_by_cuisine = {}
    for c in cfg["cuisines"]:
        bb = pd.read_csv(d2_dir / f"backbone_{c}.csv")
        nc = nodes_all[nodes_all["cuisine"] == c]
        G = backbone_graph(bb, nc)
        str_by_cuisine[c] = strengths(G)
        part = louvain(G, int(cfg["community"]["seed"]), float(cfg["community"]["resolution"]))
        cdf = pd.read_csv(d3_dir / f"communities_{c}.csv")
        assert set(cdf["ingredient"]) == set(G.nodes)
        for _, row in cdf.iterrows():
            assert int(row["community_id"]) == part[row["ingredient"]]
        assert (cdf["label_type"].str.contains("data-derived descriptive label")).all()

    bridges = pd.read_csv(d3_dir / "bridges.csv")
    for pair, grp in bridges.groupby("cuisine_pair"):
        a, b = pair.split("-")
        shared = set(str_by_cuisine[a]) & set(str_by_cuisine[b])
        expected = sorted(((min(str_by_cuisine[a][x], str_by_cuisine[b][x]), x) for x in shared),
                          key=lambda t: (-t[0], t[1]))[:5]
        assert list(grp.sort_values("rank")["ingredient"]) == [x for _, x in expected]
        for (_, row), (exp_score, _) in zip(grp.sort_values("rank").iterrows(), expected):
            assert math.isclose(row["bridge_score"], exp_score, rel_tol=1e-9)
            assert math.isclose(row["bridge_score"], min(row["weighted_degree_a"], row["weighted_degree_b"]), rel_tol=1e-9)
