"""Interactive PyVis visualisations of the validated backbones (Deliverable 2).

Encoding
--------
* node size  = ``node_size_base + node_size_scale * prevalence`` (prevalence in %, linear => proportional
  above a constant minimum so tiny nodes stay visible)
* node colour = Louvain community (fixed seed, deterministic relabelling)
* edge width = ``edge_width_min + edge_width_scale * max(NPMI, 0)`` (negative NPMI drawn at the minimum
  width; CSV values are never altered)
* tooltip: ingredient, prevalence, community, degree, weighted degree
The vis-network JavaScript is embedded (``cdn_resources='in_line'``) so the HTML works offline.

Usage::

    python src/visualize.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from pyvis.network import Network

from src.common import get_logger, load_config, rpath
from src.graph_utils import backbone_graph, louvain, strengths

log = get_logger()
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
           "#17becf", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d", "#31a354",
           "#756bb1", "#636363"]


def render(cuisine: str, backbone: pd.DataFrame, nodes: pd.DataFrame, cfg: dict, path: Path) -> dict:
    """Render one backbone to ``path``; returns basic counts. Safe for empty/disconnected graphs."""
    vc = cfg["visualization"]
    G = backbone_graph(backbone, nodes)
    part = louvain(G, int(cfg["community"]["seed"]), float(cfg["community"]["resolution"]))
    strength = strengths(G)
    net = Network(height=vc["height"], width=vc["width"], directed=False, notebook=False,
                  cdn_resources=vc.get("cdn_resources", "in_line"),
                  heading=f"{cuisine.title()} cuisine — validated ingredient backbone "
                          f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    for n in sorted(G.nodes):
        prev = G.nodes[n]["prevalence"]
        cid = part.get(n, -1)
        title = (f"Ingredient: {n}\nPrevalence: {prev:.2f}%\nCommunity: {cid}\nDegree: {G.degree(n)}\n"
                 f"Weighted degree: {strength[n]:.3f}")
        net.add_node(n, label=n, title=title, size=float(vc["node_size_base"]) + float(vc["node_size_scale"]) * prev,
                     color=PALETTE[cid % len(PALETTE)] if cid >= 0 else "#999999", font={"size": 14})
    for a, b, d in sorted(G.edges(data=True)):
        a, b = sorted((a, b))
        net.add_edge(a, b, width=float(vc["edge_width_min"]) + float(vc["edge_width_scale"]) * d["weight"],
                     title=f"{a} — {b}\nNPMI: {d['npmi']:.3f}\nco_count: {d['co_count']}")
    net.toggle_physics(True)
    net.show_buttons(filter_=["physics"])
    html_text = net.generate_html(notebook=False)
    if G.number_of_nodes() == 0:
        html_text = html_text.replace("<body>", f"<body><h3>{html.escape(cuisine)}: backbone is empty "
                                                f"(no edge passed the filter).</h3>", 1)
    path.write_text(html_text, encoding="utf-8")
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "communities": len(set(part.values()))}


def run(cfg: dict) -> dict:
    d1, d2 = rpath(cfg["outputs"]["d1"]), rpath(cfg["outputs"]["d2"])
    nodes = pd.read_csv(d1 / "nodes.csv")
    out = {}
    for c in cfg["cuisines"]:
        bb = pd.read_csv(d2 / f"backbone_{c}.csv")
        out[c] = render(c, bb, nodes[nodes["cuisine"] == c], cfg, d2 / f"graph_{c}.html")
        log.info("Graph %-8s nodes=%d edges=%d communities=%d -> graph_%s.html", c, out[c]["nodes"], out[c]["edges"],
                 out[c]["communities"], c)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render interactive PyVis HTML graphs of the backbones.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    run(load_config(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
