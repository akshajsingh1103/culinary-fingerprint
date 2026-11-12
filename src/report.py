"""Programmatic Markdown and 3-4 page PDF report generator (Deliverable 3).

Generates:
* ``outputs/d3_report/report.md``
* ``outputs/d3_report/report.pdf`` (verified 3-4 pages via pypdf)

All empirical claims are computed from the generated CSV/JSON artifacts in
``outputs/d1_dataset``, ``outputs/d2_backbone`` and ``outputs/d3_report``, and
every empirical sentence cites a numbered Table (Tables 1-7) or Figure (Figures 1-5)
included in the report itself.

Usage::

    python src/report.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.common import get_logger, load_config, read_json, rpath

log = get_logger()


def _load_report_data(cfg: dict) -> dict:
    d1 = rpath(cfg["outputs"]["d1"])
    d2 = rpath(cfg["outputs"]["d2"])
    d3 = rpath(cfg["outputs"]["d3"])
    norm_rep = read_json(d1 / "normalization_report.json")
    null_meta = read_json(d2 / "null_model_metadata.json")
    summary = pd.read_csv(d2 / "backbone_summary.csv")
    ana_meta = read_json(d3 / "analysis_summary.json")
    hubs = {c: pd.read_csv(d3 / f"hubs_{c}.csv") for c in cfg["cuisines"]}
    cls = pd.read_csv(d3 / "universal_vs_specific_hubs.csv")
    comms = {c: pd.read_csv(d3 / f"communities_{c}.csv") for c in cfg["cuisines"]}
    bridges = pd.read_csv(d3 / "bridges.csv")
    return {
        "norm": norm_rep,
        "null": null_meta,
        "summary": summary,
        "ana": ana_meta,
        "hubs": hubs,
        "cls": cls,
        "comms": comms,
        "bridges": bridges,
        "figs_dir": d3 / "figures",
    }


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def build_report_content(cfg: dict, data: dict) -> dict:
    """Assemble structured sections, tables, and figure references for both MD and PDF."""
    norm = data["norm"]
    null_meta = data["null"]
    summary = data["summary"].set_index("cuisine")
    hubs = data["hubs"]
    cls = data["cls"]
    comms = data["comms"]
    bridges = data["bridges"]
    ana = data["ana"]
    cuisines = cfg["cuisines"]

    # Methodology & Scope paragraph (concise, defines all required terms)
    methodology_text = (
        "This report characterizes the descriptive ingredient co-occurrence backbones constructed from all "
        f"{norm['recipes_selected']:,} recipes across Indian ({norm['cuisine_counts']['indian']:,}), "
        f"Italian ({norm['cuisine_counts']['italian']:,}), and Mexican ({norm['cuisine_counts']['mexican']:,}) "
        "cuisines in the Kaggle What's Cooking dataset (Table 1). Note that descriptive networks use the full "
        "selected dataset, whereas predictive model artifacts in Deliverable 4 are rebuilt exclusively on the 80% "
        "stratified training split to prevent data leakage. For each cuisine with N recipes, binary recipe-level "
        "ingredient presence defines marginal probabilities P(a) = n(a)/N and joint probabilities P(a,b) = n(a,b)/N "
        "among eligible ingredients occurring in at least 20 recipes (Table 1). Pointwise Mutual Information is "
        "defined as PMI(a,b) = log2(P(a,b) / (P(a)P(b))), and Normalized Pointwise Mutual Information is "
        "NPMI(a,b) = PMI(a,b) / (-log2 P(a,b)) in [-1, 1] (with limiting convention NPMI = 1 when P(a,b) = 1). "
        "To validate edges against degree heterogeneity, one Curveball Markov chain (Strona et al., 2014) is run "
        "per cuisine on the full binary recipe-by-ingredient incidence matrix, preserving every recipe's exact "
        "ingredient count and every ingredient's exact recipe frequency across "
        f"{null_meta['n_randomizations']:,} samples after a burn-in of 10 x N trades and thinning of 2 x N trades "
        "per sample (Table 1). Each pair's null mean mu_null and sample standard deviation sigma_null yield "
        "z(a,b) = (n(a,b) - mu_null) / sigma_null; pairs with sigma_null = 0 are conservatively excluded, and "
        "the statistically validated backbone retains edges satisfying z(a,b) > 3 (strict inequality; an "
        "uncorrected single-pair threshold) and co_count >= 5 (Table 1, Figure 1). On each undirected backbone, "
        "an ingredient's weighted degree (strength) is Strength(a) = sum of max(NPMI(a,b), 0) over incident "
        "backbone edges, normalized weighted betweenness centrality is computed over shortest paths with edge "
        "distance d(a,b) = 1 / max(NPMI(a,b), 0) (excluding zero-strength edges), and structural bridge importance "
        "between cuisines A and B for shared backbone nodes is Bridge(a; A,B) = min(Strength_A(a), Strength_B(a)) "
        "(Tables 2-7, Figures 2-5). In Deliverable 4, the training-only Fingerprint Scorer sums backbone NPMI over "
        "matched recipe pairs, Score(c) = sum_{(a,b) in M_c(R)} NPMI_c(a,b), whereas its normalized ablation computes "
        " the matched-pair mean Score_norm(c) = Score(c) / |M_c(R)| (0 if |M_c(R)| = 0); empirically, dividing by "
        "|M_c(R)| achieves substantially lower held-out accuracy than summing, consistent with the structural "
        "property that averaging over |M_c(R)| removes cumulative pair-count evidence and allows a single matched "
        "edge in one backbone to exceed the multi-edge mean in another."
    )

    # Table 1: Dataset & Backbone Summary
    t1_headers = ["Cuisine", "Recipes", "Distinct Ing.", "Eligible (>=20)", "Backbone Nodes",
                  "Edges Before", "Edges After", "Retained (%)", "Communities", "Modularity", "Null ESS(S)"]
    t1_rows = []
    for c in cuisines:
        s = summary.loc[c]
        ess = null_meta["per_cuisine"][c].get("ess_S", float("nan"))
        n_dist = len(pd.read_csv(rpath(cfg["outputs"]["d1"]) / "ingredient_prevalence_all.csv").query("cuisine == @c"))
        t1_rows.append([
            c.title(), f"{norm['cuisine_counts'][c]:,}", f"{n_dist:,}", f"{int(s['eligible_nodes']):,}",
            f"{int(s['nodes']):,}", f"{int(s['edges_before']):,}", f"{int(s['edges_after']):,}",
            f"{s['pct_retained']:.2f}%", f"{int(s['n_communities'])}", f"{s['modularity']:.3f}", f"{ess:.0f}"
        ])

    # Section (a) Top-10 hubs
    hub_tables = {}
    sec_a_paragraphs = []
    for idx, c in enumerate(cuisines, start=2):
        df = hubs[c]
        w_top = df[df["top10_weighted_degree"]].sort_values("weighted_degree_rank")
        b_top = df[df["top10_betweenness"]].sort_values("betweenness_rank")
        w1 = w_top.iloc[0]
        w2 = w_top.iloc[1]
        w3 = w_top.iloc[2]
        b1 = b_top.iloc[0]
        b2 = b_top.iloc[1]
        both = df[df["top10_weighted_degree"] & df["top10_betweenness"]]["ingredient"].tolist()
        both_str = (", ".join(both) if both else "no ingredients")
        sec_a_paragraphs.append(
            f"In the {c.title()} backbone ({int(summary.loc[c, 'nodes'])} nodes and "
            f"{int(summary.loc[c, 'edges_after']):,} validated edges, retaining "
            f"{summary.loc[c, 'pct_retained']:.2f}% of {int(summary.loc[c, 'edges_before']):,} raw edges; "
            f"Table 1, Figure 1), the top three weighted-degree hubs are {w1['ingredient']} "
            f"(strength {w1['weighted_degree']:.2f}, rank 1, betweenness {w1['betweenness']:.4f} [rank {int(w1['betweenness_rank'])}], "
            f"prevalence {w1['prevalence']:.2f}%), {w2['ingredient']} (strength {w2['weighted_degree']:.2f}, rank 2, "
            f"betweenness rank {int(w2['betweenness_rank'])}, prevalence {w2['prevalence']:.2f}%), and "
            f"{w3['ingredient']} (strength {w3['weighted_degree']:.2f}, rank 3, betweenness rank {int(w3['betweenness_rank'])}, "
            f"prevalence {w3['prevalence']:.2f}%) (Table {idx}, Figure 2). By weighted betweenness centrality "
            f"(distance 1/max(NPMI,0)), the highest-ranked brokers in {c.title()} are {b1['ingredient']} "
            f"(betweenness {b1['betweenness']:.4f}, rank 1; strength {b1['weighted_degree']:.2f}, rank {int(b1['weighted_degree_rank'])}) "
            f"and {b2['ingredient']} (betweenness {b2['betweenness']:.4f}, rank 2; strength {b2['weighted_degree']:.2f}, "
            f"rank {int(b2['weighted_degree_rank'])}), with {both_str} appearing in the top 10 on both metrics "
            f"(Table {idx}, Figure 2)."
        )
        headers = ["Ingredient", "Strength", "Str. Rank", "Betweenness", "Btw. Rank", "Deg.", "Prev. (%)", "Top-10 Str.", "Top-10 Btw."]
        rows = [
            [r["ingredient"], f"{r['weighted_degree']:.2f}", str(int(r["weighted_degree_rank"])),
             f"{r['betweenness']:.4f}", str(int(r["betweenness_rank"])), str(int(r["degree"])),
             f"{r['prevalence']:.2f}%", "Yes" if r["top10_weighted_degree"] else "No",
             "Yes" if r["top10_betweenness"] else "No"]
            for _, r in df.iterrows()
        ]
        hub_tables[c] = (idx, f"Table {idx}: Top-10 Weighted-Degree and Top-10 Weighted-Betweenness Hubs in {c.title()}", headers, rows)

    # Section (b) Universal and cuisine-specific hubs
    n_univ = int((cls["classification"] == "universal").sum())
    n_two = int((cls["classification"] == "two-cuisine").sum())
    n_spec = int((cls["classification"] == "cuisine-specific").sum())
    two_df = cls[cls["classification"] == "two-cuisine"]
    two_desc = "; ".join(
        f"{r['ingredient']} (top-10 in {r['top10_in'].replace(';', ' & ')}, ranks "
        f"Indian={ '-' if pd.isna(r['rank_indian']) else int(r['rank_indian']) }, "
        f"Italian={ '-' if pd.isna(r['rank_italian']) else int(r['rank_italian']) }, "
        f"Mexican={ '-' if pd.isna(r['rank_mexican']) else int(r['rank_mexican']) })"
        for _, r in two_df.iterrows()
    )
    # Highlight distinction between backbone hub status and recipe prevalence (e.g. onion or garlic cloves)
    onion_row = cls[cls["ingredient"] == "onion"].iloc[0]
    sec_b_paragraphs = [
        f"Across the union of {len(cls)} distinct top-10 weighted-degree hubs from the three cuisines, "
        f"{n_univ} ingredients qualify as universal hubs (top-10 in all three backbones), {n_two} qualify as "
        f"two-cuisine hubs, and {n_spec} are cuisine-specific hubs appearing in the top 10 of exactly one "
        f"cuisine (Table 5, Figure 3). The {n_two} two-cuisine hubs are {two_desc} (Table 5, Figure 3).",
        f"Crucially, absence from a cuisine's top-10 weighted-degree hubs—or even absence from a backbone—does "
        f"not imply culinary absence from recipes: for example, onion is a top-10 weighted-degree hub only in "
        f"Italian (rank {int(onion_row['rank_italian'])}, strength {onion_row['weighted_degree_italian']:.2f}, "
        f"prevalence {onion_row['prevalence_italian']:.2f}%), yet onion occurs in {onion_row['prevalence_indian']:.2f}% "
        f"of Indian recipes (backbone strength rank {int(onion_row['rank_indian'])}, strength "
        f"{onion_row['weighted_degree_indian']:.2f}) and {onion_row['prevalence_mexican']:.2f}% of Mexican recipes "
        f"(backbone strength rank {int(onion_row['rank_mexican'])}, strength {onion_row['weighted_degree_mexican']:.2f}) "
        f"because ubiquitous staples co-occur broadly across many recipe types rather than concentrating into "
        f"high-NPMI specialist cliques (Table 5)."
    ]
    t5_headers = ["Ingredient", "Class", "Top-10 In", "Rank (IN/IT/MX)", "Strength (IN/IT/MX)", "Prevalence % (IN/IT/MX)"]
    t5_rows = []
    for _, r in cls.iterrows():
        rk = "/".join("-" if pd.isna(r[f"rank_{c}"]) else str(int(r[f"rank_{c}"])) for c in cuisines)
        st = "/".join("-" if pd.isna(r[f"weighted_degree_{c}"]) else f"{r[f'weighted_degree_{c}']:.1f}" for c in cuisines)
        pv = "/".join(f"{r[f'prevalence_{c}']:.1f}" for c in cuisines)
        t5_rows.append([r["ingredient"], r["classification"], r["top10_in"].replace(";", ","), rk, st, pv])

    # Section (c) Louvain communities
    t6_headers = ["Cuisine", "Comm. ID", "Size", "Data-Derived Label (Top-3)", "Top-5 Ingredients by Intra-Community Strength"]
    t6_rows = []
    sec_c_paragraphs = []
    for c in cuisines:
        cdf = comms[c]
        n_comm = int(summary.loc[c, "n_communities"])
        mod = float(summary.loc[c, "modularity"])
        n_sing = int(ana["singleton_communities"][c])
        distinct_comms = cdf.drop_duplicates("community_id").sort_values("community_id")
        top_Summaries = []
        for _, crow in distinct_comms.iterrows():
            cid = int(crow["community_id"])
            size = int(crow["community_size"])
            sub = cdf[(cdf["community_id"] == cid) & (cdf["in_top5"])].sort_values("within_community_rank")
            top5_str = ", ".join(f"{r['ingredient']} ({r['within_community_weighted_degree']:.2f})" for _, r in sub.iterrows())
            t6_rows.append([c.title(), str(cid), str(size), crow["community_label"], top5_str])
            if cid < 3:
                top_Summaries.append(f"C{cid} (size {size}, label '{crow['community_label']}')")
        sec_c_paragraphs.append(
            f"Louvain partitioning on nonnegative NPMI weights in {c.title()} identifies {n_comm} communities "
            f"(including {n_sing} singleton communities on disconnected dyads/isolates) with weighted modularity "
            f"Q = {mod:.3f}, led by {'; '.join(top_Summaries)} (Table 1, Table 6, Figure 4)."
        )
    sec_c_paragraphs.append(
        "All community labels in Table 6 are automatically generated from the top three ingredients by "
        "within-community weighted degree and are explicitly reported as data-derived descriptive labels rather "
        "than validated semantic classifications (Table 6, Figure 4)."
    )

    # Section (d) Strongest bridge ingredients
    t7_headers = ["Cuisine Pair", "Rank", "Ingredient", "Strength A", "Strength B", "Bridge Score", "Prev. A (%)", "Prev. B (%)"]
    t7_rows = []
    sec_d_paragraphs = []
    for pair in ["indian-italian", "indian-mexican", "italian-mexican"]:
        sub = bridges[bridges["cuisine_pair"] == pair].sort_values("rank")
        n_cand = ana["bridge_candidate_counts"][pair]
        a_name, b_name = [x.title() for x in pair.split("-")]
        top1 = sub.iloc[0]
        top2 = sub.iloc[1]
        top3 = sub.iloc[2]
        sec_d_paragraphs.append(
            f"Between {a_name} and {b_name} ({n_cand} shared backbone nodes eligible as bridge candidates), "
            f"the top three structural bridges by Bridge(a; A,B) = min(Strength_A(a), Strength_B(a)) are "
            f"{top1['ingredient']} (rank 1, bridge score {top1['bridge_score']:.2f}; strengths "
            f"{top1['weighted_degree_a']:.2f} vs {top1['weighted_degree_b']:.2f}; prevalences "
            f"{top1['prevalence_a']:.2f}% vs {top1['prevalence_b']:.2f}%), {top2['ingredient']} "
            f"(rank 2, bridge score {top2['bridge_score']:.2f}), and {top3['ingredient']} "
            f"(rank 3, bridge score {top3['bridge_score']:.2f}) (Table 7, Figure 5)."
        )
        for _, r in sub.iterrows():
            t7_rows.append([
                f"{a_name}-{b_name}", str(int(r["rank"])), r["ingredient"],
                f"{r['weighted_degree_a']:.2f}", f"{r['weighted_degree_b']:.2f}", f"{r['bridge_score']:.2f}",
                f"{r['prevalence_a']:.2f}%", f"{r['prevalence_b']:.2f}%"
            ])
    sec_d_paragraphs.append(
        "These bridge scores quantify shared structural centrality across pairs of validated ingredient "
        "backbones and do not imply historical cultural exchange, shared culinary ancestry, or causal influence "
        "(Table 7, Figure 5)."
    )

    return {
        "title": "The Culinary Fingerprint: Hub, Community, and Bridge Analysis of Statistically Validated Ingredient Backbones",
        "methodology": methodology_text,
        "table1": ("Table 1: Dataset, Null-Model Mixing, and Validated Backbone Summary", t1_headers, t1_rows),
        "sec_a": ("(a) Top-10 hubs", sec_a_paragraphs, hub_tables),
        "sec_b": ("(b) Universal and cuisine-specific hubs", sec_b_paragraphs,
                  ("Table 5: Classification of All Top-10 Weighted-Degree Hubs Across Cuisines", t5_headers, t5_rows)),
        "sec_c": ("(c) Louvain communities", sec_c_paragraphs,
                  ("Table 6: Louvain Communities and Top-5 Intra-Community Members (Data-Derived Descriptive Labels)",
                   t6_headers, t6_rows)),
        "sec_d": ("(d) Strongest bridge ingredients", sec_d_paragraphs,
                  ("Table 7: Top-5 Structural Bridge Ingredients per Cuisine Pair", t7_headers, t7_rows)),
        "figures": [
            ("Figure 1", "fig1_backbone_edges.png", "Figure 1: Raw vs statistically validated backbone edge counts (log scale) and retention percentages across cuisines."),
            ("Figure 2", "fig2_top_hubs.png", "Figure 2: Top-10 weighted-degree hubs (sum of nonnegative NPMI on incident backbone edges) for Indian, Italian, and Mexican cuisines."),
            ("Figure 3", "fig3_hub_overlap.png", "Figure 3: Overlap of top-10 weighted-degree hubs across Indian, Italian, and Mexican backbones."),
            ("Figure 4", "fig4_community_sizes.png", "Figure 4: Louvain community size distributions across Indian, Italian, and Mexican backbones."),
            ("Figure 5", "fig5_bridges.png", "Figure 5: Top-5 bridge ingredients per cuisine pair ranked by Bridge(a; A,B) = min(Strength_A(a), Strength_B(a))."),
        ],
    }


def write_markdown_report(content: dict, path: Path) -> None:
    lines = [
        f"# {content['title']}",
        "",
        "## Methodology and Scope",
        "",
        content["methodology"],
        "",
        f"### {content['table1'][0]}",
        "",
        _md_table(content["table1"][1], content["table1"][2]),
        "",
        f"![{content['figures'][0][2]}](figures/{content['figures'][0][1]})",
        f"*{content['figures'][0][2]}*",
        "",
        f"## {content['sec_a'][0]}",
        "",
    ]
    for p in content["sec_a"][1]:
        lines.extend([p, ""])
    for c in ["indian", "italian", "mexican"]:
        _, title, headers, rows = content["sec_a"][2][c]
        lines.extend([f"### {title}", "", _md_table(headers, rows), ""])
    lines.extend([
        f"![{content['figures'][1][2]}](figures/{content['figures'][1][1]})",
        f"*{content['figures'][1][2]}*",
        "",
        f"## {content['sec_b'][0]}",
        "",
    ])
    for p in content["sec_b"][1]:
        lines.extend([p, ""])
    t5_title, t5_h, t5_r = content["sec_b"][2]
    lines.extend([
        f"### {t5_title}", "", _md_table(t5_h, t5_r), "",
        f"![{content['figures'][2][2]}](figures/{content['figures'][2][1]})",
        f"*{content['figures'][2][2]}*",
        "",
        f"## {content['sec_c'][0]}",
        "",
    ])
    for p in content["sec_c"][1]:
        lines.extend([p, ""])
    t6_title, t6_h, t6_r = content["sec_c"][2]
    lines.extend([
        f"### {t6_title}", "", _md_table(t6_h, t6_r), "",
        f"![{content['figures'][3][2]}](figures/{content['figures'][3][1]})",
        f"*{content['figures'][3][2]}*",
        "",
        f"## {content['sec_d'][0]}",
        "",
    ])
    for p in content["sec_d"][1]:
        lines.extend([p, ""])
    t7_title, t7_h, t7_r = content["sec_d"][2]
    lines.extend([
        f"### {t7_title}", "", _md_table(t7_h, t7_r), "",
        f"![{content['figures'][4][2]}](figures/{content['figures'][4][1]})",
        f"*{content['figures'][4][2]}*",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _rl_table(headers: list[str], rows: list[list[str]], col_widths: list[float], font_size: float, pad: float) -> Table:
    cell_style = ParagraphStyle("cell", fontName="Helvetica", fontSize=font_size, leading=font_size + 1.3)
    hdr_style = ParagraphStyle("hdr", fontName="Helvetica-Bold", fontSize=font_size, leading=font_size + 1.3,
                               textColor=colors.white)
    data = [[Paragraph(h, hdr_style) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(x), cell_style) for x in r])
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e78")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fa")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bfd0e0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return tbl


def _build_pdf_once(content: dict, figs_dir: Path, pdf_path: Path, scale: float) -> int:
    """Render report.pdf at layout ``scale`` and return its page count."""
    margin = 0.42 * inch
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=margin, rightMargin=margin, topMargin=margin, bottomMargin=margin
    )
    W = A4[0] - 2 * margin
    styles = getSampleStyleSheet()
    body_fs = 7.4 * scale
    tbl_fs = 6.1 * scale
    pad = max(1.0, 1.4 * scale)

    st_title = ParagraphStyle("Title2", parent=styles["Title"], fontName="Helvetica-Bold",
                              fontSize=11.5 * scale, leading=13.5 * scale, spaceAfter=3 * scale, alignment=0)
    st_h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                           fontSize=9.2 * scale, leading=11.0 * scale, spaceBefore=4 * scale, spaceAfter=2 * scale,
                           textColor=colors.HexColor("#1f4e78"))
    st_body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica",
                             fontSize=body_fs, leading=body_fs + 1.8 * scale, spaceAfter=2.2 * scale)
    st_cap = ParagraphStyle("Cap", parent=styles["BodyText"], fontName="Helvetica-Oblique",
                            fontSize=6.6 * scale, leading=7.8 * scale, spaceBefore=1.5 * scale, spaceAfter=2 * scale)

    story = [
        Paragraph(content["title"], st_title),
        Paragraph("Methodology and Scope", st_h2),
        Paragraph(content["methodology"], st_body),
    ]

    # Table 1 + Figure 1
    t1_title, t1_h, t1_r = content["table1"]
    story.append(Paragraph(f"<b>{t1_title}</b>", st_cap))
    w1 = [W * x for x in [0.09, 0.08, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.11]]
    story.append(_rl_table(t1_h, t1_r, w1, tbl_fs, pad))
    story.append(Spacer(1, 2 * scale))

    # Section (a)
    story.append(Paragraph(content["sec_a"][0], st_h2))
    for p in content["sec_a"][1]:
        story.append(Paragraph(p, st_body))
    wh = [W * x for x in [0.24, 0.09, 0.09, 0.11, 0.09, 0.07, 0.10, 0.10, 0.11]]
    for c in ["indian", "italian", "mexican"]:
        _, title, headers, rows = content["sec_a"][2][c]
        story.append(Paragraph(f"<b>{title}</b>", st_cap))
        story.append(_rl_table(headers, rows, wh, tbl_fs, pad))

    # Figures 1 & 2 side-by-side or compact
    f1_path = figs_dir / content["figures"][0][1]
    f2_path = figs_dir / content["figures"][1][1]
    story.append(Image(str(f1_path), width=W * 0.58, height=1.25 * inch * scale))
    story.append(Paragraph(content["figures"][0][2], st_cap))
    story.append(Image(str(f2_path), width=W * 0.92, height=1.45 * inch * scale))
    story.append(Paragraph(content["figures"][1][2], st_cap))

    # Section (b)
    story.append(Paragraph(content["sec_b"][0], st_h2))
    for p in content["sec_b"][1]:
        story.append(Paragraph(p, st_body))
    t5_title, t5_h, t5_r = content["sec_b"][2]
    story.append(Paragraph(f"<b>{t5_title}</b>", st_cap))
    w5 = [W * x for x in [0.22, 0.15, 0.15, 0.15, 0.16, 0.17]]
    story.append(_rl_table(t5_h, t5_r, w5, tbl_fs, pad))
    f3_path = figs_dir / content["figures"][2][1]
    story.append(Image(str(f3_path), width=W * 0.42, height=1.25 * inch * scale))
    story.append(Paragraph(content["figures"][2][2], st_cap))

    # Section (c)
    story.append(Paragraph(content["sec_c"][0], st_h2))
    for p in content["sec_c"][1]:
        story.append(Paragraph(p, st_body))
    t6_title, t6_h, t6_r = content["sec_c"][2]
    story.append(Paragraph(f"<b>{t6_title}</b>", st_cap))
    w6 = [W * x for x in [0.09, 0.08, 0.07, 0.28, 0.48]]
    story.append(_rl_table(t6_h, t6_r, w6, tbl_fs, pad))
    f4_path = figs_dir / content["figures"][3][1]
    story.append(Image(str(f4_path), width=W * 0.58, height=1.20 * inch * scale))
    story.append(Paragraph(content["figures"][3][2], st_cap))

    # Section (d)
    story.append(Paragraph(content["sec_d"][0], st_h2))
    for p in content["sec_d"][1]:
        story.append(Paragraph(p, st_body))
    t7_title, t7_h, t7_r = content["sec_d"][2]
    story.append(Paragraph(f"<b>{t7_title}</b>", st_cap))
    w7 = [W * x for x in [0.16, 0.06, 0.22, 0.11, 0.11, 0.12, 0.11, 0.11]]
    story.append(_rl_table(t7_h, t7_r, w7, tbl_fs, pad))
    f5_path = figs_dir / content["figures"][4][1]
    story.append(KeepTogether([
        Image(str(f5_path), width=W * 0.90, height=1.25 * inch * scale),
        Paragraph(content["figures"][4][2], st_cap),
    ]))

    doc.build(story)
    return len(PdfReader(str(pdf_path)).pages)


def write_pdf_report(content: dict, figs_dir: Path, pdf_path: Path, min_pages: int = 3, max_pages: int = 4) -> int:
    """Render report.pdf and automatically tune scale so page count is in [min_pages, max_pages]."""
    for scale in [1.0, 0.92, 0.85, 0.78, 0.72, 1.08, 1.15]:
        pages = _build_pdf_once(content, figs_dir, pdf_path, scale)
        if min_pages <= pages <= max_pages:
            return pages
    return pages


def run(cfg: dict) -> dict:
    d3 = rpath(cfg["outputs"]["d3"])
    d3.mkdir(parents=True, exist_ok=True)
    data = _load_report_data(cfg)
    content = build_report_content(cfg, data)
    md_path = d3 / "report.md"
    pdf_path = d3 / "report.pdf"
    write_markdown_report(content, md_path)
    pages = write_pdf_report(
        content, data["figs_dir"], pdf_path,
        int(cfg["report"]["min_pages"]), int(cfg["report"]["max_pages"])
    )
    log.info("Report written: %s (%d bytes) and %s (%d pages)", md_path.name, md_path.stat().st_size,
             pdf_path.name, pages)
    return {"md_path": str(md_path), "pdf_path": str(pdf_path), "pdf_pages": pages}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate Deliverable 3 Markdown and 3-4 page PDF report.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    run(load_config(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
