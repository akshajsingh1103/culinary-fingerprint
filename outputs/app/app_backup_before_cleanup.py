"""Culinary Fingerprint — Interactive Network-Based Cuisine Classifier (Streamlit App).

Predicts a recipe's cuisine (Indian / Italian / Mexican) using Fingerprint Scorer v2
(22-feature network-informed stacked L2 logistic regression) and Fingerprint Scorer v1
(original backbone NPMI sum) side by side.

Run from the repository root:
    streamlit run outputs/app/app.py
"""
from __future__ import annotations

import base64
import random
import re
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from scorer_runtime import CulinaryFingerprintRuntime

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Culinary Fingerprint — Network Cuisine Classifier",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUISINE_META = {
    "indian": {
        "title": "Indian",
        "icon": "🟠",
        "color": "#F59E0B",
        "glow": "rgba(245, 158, 11, 0.24)",
        "border": "rgba(245, 158, 11, 0.65)",
        "tagline": "Aromatic spice backbones, temperings & masala communities",
    },
    "italian": {
        "title": "Italian",
        "icon": "🟢",
        "color": "#10B981",
        "glow": "rgba(16, 185, 129, 0.24)",
        "border": "rgba(16, 185, 129, 0.65)",
        "tagline": "Herb-allium-olive oil foundations, pasta & cheese modules",
    },
    "mexican": {
        "title": "Mexican",
        "icon": "🔴",
        "color": "#EF4444",
        "glow": "rgba(239, 68, 68, 0.24)",
        "border": "rgba(239, 68, 68, 0.65)",
        "tagline": "Chile-citrus-cilantro-corn backbones & salsa modules",
    },
}

COMMUNITY_PALETTE = [
    "#F59E0B",
    "#10B981",
    "#EF4444",
    "#38BDF8",
    "#A855F7",
    "#EC4899",
    "#84CC16",
    "#F97316",
]

PATTERN_SVG_PATH = APP_DIR / "assets" / "ingredient_pattern.svg"

FALLBACK_INLINE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="360" height="360" viewBox="0 0 360 360">
  <g fill="none" stroke="#E2E8F0" stroke-width="1.35" stroke-linecap="round" stroke-linejoin="round" stroke-opacity="0.055">
    <circle cx="60" cy="60" r="22"/><path d="M60 38 L60 30"/>
    <path d="M168 40 C178 38, 184 54, 166 80 C158 70, 160 44, 168 40 Z"/>
    <circle cx="290" cy="62" r="20"/><circle cx="290" cy="62" r="3"/>
    <path d="M60 165 C75 172, 80 190, 60 212 C40 190, 45 172, 60 165 Z"/>
    <circle cx="180" cy="185" r="23"/>
    <path d="M285 162 L302 162 L293 215 Z"/>
  </g>
</svg>"""


@st.cache_data
def _get_pattern_data_uri() -> str:
    """Load the hand-drawn ingredient line-art SVG from assets/ and return a base64 CSS data URI."""
    if PATTERN_SVG_PATH.exists():
        svg_bytes = PATTERN_SVG_PATH.read_bytes()
    else:
        svg_bytes = FALLBACK_INLINE_SVG.encode("utf-8")
    b64 = base64.b64encode(svg_bytes).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def apply_theme() -> None:
    """Inject all application CSS through one helper function (via st.html) with zero raw HTML leaking as text.

    Features:
    - Tiled hand-drawn ingredient line-art SVG background (tomato, chili, garlic, basil leaf,
      onion, carrot, lime, mushroom, cumin/spice seeds) at ~5.5% stroke opacity over #0F172A.
    - Soft radial vignette overlay that gently fades the pattern toward the central content area
      so it never competes with text.
    - Solid/translucent high-contrast card surfaces (#1E293B with subtle backdrop blur), consistent
      14px border radius, soft layered shadows, cuisine-specific accent glows (saffron Indian,
      emerald Italian, crimson Mexican), and a gentle fade-in animation on the prediction cards.
    """
    data_uri = _get_pattern_data_uri()
    css = f"""
    <style>
    @keyframes cfFadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(7px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    /* Main app background: radial vignette layered over tiled hand-drawn ingredient SVG */
    .stApp {{
        background-color: #0F172A !important;
        background-image:
            radial-gradient(
                ellipse 78% 72% at 50% 38%,
                rgba(15, 23, 42, 0.90) 0%,
                rgba(15, 23, 42, 0.72) 55%,
                rgba(15, 23, 42, 0.48) 100%
            ),
            url("{data_uri}") !important;
        background-repeat: no-repeat, repeat !important;
        background-size: 100% 100%, 360px 360px !important;
        background-attachment: fixed, fixed !important;
        color: #F8FAFC !important;
    }}

    .block-container {{
        padding-top: 1.35rem !important;
        padding-bottom: 2.4rem !important;
        max-width: 1280px !important;
    }}

    /* All bordered containers act as high-contrast, softly shadowed rounded cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(30, 41, 59, 0.94) !important;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(148, 163, 184, 0.20) !important;
        border-radius: 14px !important;
        padding: 1.15rem 1.3rem !important;
        box-shadow: 0 10px 28px -6px rgba(2, 6, 23, 0.45), 0 4px 12px -2px rgba(2, 6, 23, 0.28) !important;
        transition: box-shadow 0.22s ease, border-color 0.22s ease;
    }}

    /* Header banner card */
    .st-key-header_card {{
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.97) 0%, rgba(15, 23, 42, 0.96) 100%) !important;
        border: 1px solid rgba(148, 163, 184, 0.26) !important;
        border-radius: 14px !important;
        box-shadow: 0 12px 30px -8px rgba(2, 6, 23, 0.55) !important;
    }}

    /* Input panel card */
    .st-key-input_card {{
        background: rgba(30, 41, 59, 0.95) !important;
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
    }}

    /* Predicted cuisine headline cards with subtle cuisine-coloured accent glow & gentle fade-in */
    .st-key-v2_card_indian, .st-key-v1_card_indian {{
        border-top: 4px solid #F59E0B !important;
        border-color: rgba(245, 158, 11, 0.50) !important;
        box-shadow: 0 10px 30px -6px rgba(245, 158, 11, 0.22), 0 4px 14px -2px rgba(2, 6, 23, 0.45) !important;
        animation: cfFadeInUp 0.36s cubic-bezier(0.22, 1, 0.36, 1) both;
    }}
    .st-key-v2_card_italian, .st-key-v1_card_italian {{
        border-top: 4px solid #10B981 !important;
        border-color: rgba(16, 185, 129, 0.50) !important;
        box-shadow: 0 10px 30px -6px rgba(16, 185, 129, 0.22), 0 4px 14px -2px rgba(2, 6, 23, 0.45) !important;
        animation: cfFadeInUp 0.36s cubic-bezier(0.22, 1, 0.36, 1) both;
    }}
    .st-key-v2_card_mexican, .st-key-v1_card_mexican {{
        border-top: 4px solid #EF4444 !important;
        border-color: rgba(239, 68, 68, 0.50) !important;
        box-shadow: 0 10px 30px -6px rgba(239, 68, 68, 0.22), 0 4px 14px -2px rgba(2, 6, 23, 0.45) !important;
        animation: cfFadeInUp 0.36s cubic-bezier(0.22, 1, 0.36, 1) both;
    }}

    /* Progress bar polish inside cuisine cards */
    .st-key-v2_card_indian div[data-testid="stProgressBar"] > div > div > div > div,
    .st-key-v1_card_indian div[data-testid="stProgressBar"] > div > div > div > div {{
        background-color: #F59E0B !important;
    }}
    .st-key-v2_card_italian div[data-testid="stProgressBar"] > div > div > div > div,
    .st-key-v1_card_italian div[data-testid="stProgressBar"] > div > div > div > div {{
        background-color: #10B981 !important;
    }}
    .st-key-v2_card_mexican div[data-testid="stProgressBar"] > div > div > div > div,
    .st-key-v1_card_mexican div[data-testid="stProgressBar"] > div > div > div > div {{
        background-color: #EF4444 !important;
    }}

    /* Solid readable backgrounds for inputs, multiselect, textareas, expanders, and tables */
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] > textarea,
    div[data-baseweb="input"] > div {{
        background-color: rgba(15, 23, 42, 0.92) !important;
        border-radius: 10px !important;
        border-color: rgba(148, 163, 184, 0.28) !important;
    }}
    div[data-testid="stExpander"] {{
        background-color: rgba(15, 23, 42, 0.78) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.20) !important;
    }}
    div[data-testid="stDataFrame"],
    div[data-testid="stMetric"] {{
        background-color: rgba(30, 41, 59, 0.94) !important;
        border-radius: 12px !important;
    }}
    div[data-testid="stMetric"] {{
        padding: 0.85rem 1.05rem !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        box-shadow: 0 4px 14px -2px rgba(2, 6, 23, 0.30) !important;
    }}
    </style>
    """
    st.html(css)


@st.cache_resource
def load_runtime() -> CulinaryFingerprintRuntime:
    return CulinaryFingerprintRuntime()


def build_matched_backbone_pyvis_html(
    runtime: CulinaryFingerprintRuntime,
    cuisine: str,
    norm_ingredients: list[str],
    max_edges: int = 25,
) -> tuple[str, int, int]:
    """Build an offline PyVis interactive network HTML for the top matched backbone edges in `cuisine`."""
    v1_info = runtime.predict_v1(norm_ingredients)["per_cuisine"][cuisine]
    matched_edges = v1_info["matched_pairs"][:max_edges]
    bb_ings = set(v1_info["inputs_in_backbone_vocab"])
    hubs_c = runtime.hubs_top10[cuisine]
    bridges_c = runtime.bridges_union[cuisine]
    part_c = runtime.communities_part[cuisine]
    comm_labels_c = runtime.community_labels[cuisine]

    net = Network(
        height="360px",
        width="100%",
        bgcolor="#0F172A",
        font_color="#F8FAFC",
        directed=False,
        notebook=False,
        cdn_resources="in_line",
    )

    nodes_to_show = set(bb_ings)
    for e in matched_edges:
        nodes_to_show.add(e["pair"][0])
        nodes_to_show.add(e["pair"][1])

    for node in sorted(nodes_to_show):
        cid = part_c.get(node, 0)
        comm_lbl = comm_labels_c.get(cid, f"Community {cid}")
        is_hub = node in hubs_c
        is_bridge = node in bridges_c
        role_tags = []
        if is_hub:
            role_tags.append("Top-10 Hub")
        if is_bridge:
            role_tags.append("Cross-Cuisine Bridge")
        role_str = " | ".join(role_tags) if role_tags else "Backbone Node"

        node_color = COMMUNITY_PALETTE[cid % len(COMMUNITY_PALETTE)]
        border_color = "#F8FAFC" if (is_hub or is_bridge) else node_color
        size = 26 if is_hub else (22 if is_bridge else 17)
        tooltip = f"{node}\nRole: {role_str}\nLouvain Community {cid}: {comm_lbl}"

        net.add_node(
            node,
            label=node,
            title=tooltip,
            size=size,
            color={"background": node_color, "border": border_color},
            borderWidth=3 if (is_hub or is_bridge) else 1,
            font={"size": 14, "color": "#F8FAFC"},
        )

    accent_color = CUISINE_META[cuisine]["color"]
    for e in matched_edges:
        a, b = e["pair"]
        npmi = float(e["NPMI"])
        z_sc = float(e["z_score"])
        co = int(e["co_count"])
        width = 1.5 + 8.0 * max(npmi, 0.0)
        net.add_edge(
            a,
            b,
            width=width,
            color=accent_color,
            title=f"{a} + {b}\nNPMI: {npmi:.3f} | Curveball z: {z_sc:.2f} | Train co-count: {co}",
        )

    net.set_options(
        """
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -42,
              "centralGravity": 0.015,
              "springLength": 95,
              "springConstant": 0.08
            },
            "maxVelocity": 40,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 120}
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 80,
            "navigationButtons": false
          }
        }
        """
    )
    raw_html = net.generate_html(notebook=False)
    # Strip external Bootstrap CDN tags so the iframe renders 100% offline
    clean_html = re.sub(r"<link[^>]+cdn\.jsdelivr\.net[^>]*>", "", raw_html, flags=re.IGNORECASE | re.DOTALL)
    clean_html = re.sub(
        r"<script[^>]+cdn\.jsdelivr\.net[^>]*>\s*</script>", "", clean_html, flags=re.IGNORECASE | re.DOTALL
    )
    clean_html = clean_html.replace("background-color: white;", "background-color: #0F172A;")
    return clean_html, len(nodes_to_show), len(matched_edges)


def init_session_state() -> None:
    if "selected_ingredients" not in st.session_state:
        st.session_state["selected_ingredients"] = [
            "garlic",
            "onion",
            "tomato",
            "cumin",
            "garam masala",
            "ground turmeric",
            "ginger",
            "cilantro leaves",
        ]
    if "pasted_text" not in st.session_state:
        st.session_state["pasted_text"] = ""
    if "surprise_recipe" not in st.session_state:
        st.session_state["surprise_recipe"] = None
    if "has_predicted" not in st.session_state:
        st.session_state["has_predicted"] = True


def render_header() -> None:
    with st.container(border=True, key="header_card"):
        h_left, h_right = st.columns([1.45, 1.0], vertical_alignment="center")
        with h_left:
            st.markdown("## 🍲 Culinary Fingerprint")
            st.caption(
                "Network-Informed Cuisine Classification & Backbone Structure Explorer "
                "(**Fingerprint Scorer v2** vs. **Fingerprint Scorer v1**)"
            )
        with h_right:
            st.info(
                "**Supported cuisines**: Only **Indian**, **Italian** and **Mexican** are supported; "
                "every recipe is forced into one of these three."
            )


def render_predict_tab(runtime: CulinaryFingerprintRuntime) -> None:
    col_in, col_out = st.columns([1.0, 1.35], gap="large")

    with col_in:
        with st.container(border=True, key="input_card"):
            st.caption("1. RECIPE INGREDIENT INPUT")

            # Action buttons row: Surprise me, Predict Cuisine & Clear
            btn_c1, btn_c2, btn_c3 = st.columns([1.15, 1.0, 0.85])
            with btn_c1:
                if st.button("🎲 Surprise me", width="stretch", help="Load a random real What's Cooking recipe"):
                    rec = random.choice(runtime.sample_recipes)
                    st.session_state["surprise_recipe"] = rec
                    st.session_state["selected_ingredients"] = [
                        x for x in rec["normalized_ingredients"] if x in runtime.known_vocab_set
                    ]
                    st.session_state["pasted_text"] = ""
                    st.session_state["has_predicted"] = False
                    st.rerun()
            with btn_c3:
                if st.button("🗑️ Clear", width="stretch"):
                    st.session_state["selected_ingredients"] = []
                    st.session_state["pasted_text"] = ""
                    st.session_state["surprise_recipe"] = None
                    st.session_state["has_predicted"] = False
                    st.rerun()

            # Show mystery recipe banner if "Surprise me" was clicked
            surprise = st.session_state.get("surprise_recipe")
            if surprise is not None:
                if st.session_state.get("has_predicted", False):
                    act_c = surprise["cuisine"]
                    meta_act = CUISINE_META[act_c]
                    st.success(
                        f"🎲 **Sample Recipe #{surprise['id']}** ({surprise['split']}): "
                        f"Actual cuisine revealed → **{meta_act['icon']} Actual cuisine: {meta_act['title']}**"
                    )
                else:
                    st.info(
                        f"🎲 Loaded **Sample Recipe #{surprise['id']}** "
                        f"({len(surprise['normalized_ingredients'])} ingredients). "
                        "*True cuisine is hidden — click **Predict Cuisine** to reveal!*"
                    )

            # Search-as-you-type multiselect over known vocabulary ordered by frequency
            selected_chips = st.multiselect(
                "Search & select What's Cooking ingredients (ordered by recipe frequency):",
                options=runtime.ingredients_ordered,
                key="selected_ingredients",
                placeholder="Type to search 4,105 normalized ingredients (e.g. garlic, cumin, mozzarella, jalapeno)...",
            )

            # Optional paste a list box
            with st.expander(
                "📋 Optional: Paste a comma- or line-separated ingredient list",
                expanded=bool(st.session_state.get("pasted_text")),
            ):
                pasted_raw = st.text_area(
                    "Paste ingredients (normalized with the pipeline's 5-stage deterministic rules):",
                    key="pasted_text",
                    height=95,
                    placeholder="e.g. Roma Tomatoes, Scallions, Garbanzo Beans, Fresh Cilantro, Unseen Exotic Spice",
                )
                parsed_paste = runtime.parse_pasted_text(pasted_raw)
                if pasted_raw and pasted_raw.strip():
                    rec_list = parsed_paste["recognized"]
                    unrec_list = parsed_paste["unrecognized"]
                    st.markdown(
                        f"**Recognized in vocabulary ({len(rec_list)})**: "
                        + (", ".join(f"`{x}`" for x in rec_list) if rec_list else "*none*")
                    )
                    if unrec_list:
                        st.warning(
                            "**Unrecognized items (not in What's Cooking training vocabulary)**: "
                            + ", ".join(f"`{u['raw']}` (normalized: `{u['normalized']}`)" for u in unrec_list)
                        )
            if not (st.session_state.get("pasted_text") or "").strip():
                parsed_paste = {"raw_tokens": [], "recognized": [], "recognized_backbone": [], "unrecognized": []}

            # Primary Predict button
            with btn_c2:
                if st.button("🔍 Predict Cuisine", type="primary", width="stretch"):
                    st.session_state["has_predicted"] = True
                    st.rerun()

    # Combine multiselect ingredients + raw pasted tokens for inference
    combined_input_tokens = list(selected_chips) + list(parsed_paste["raw_tokens"])
    result = runtime.predict_both(combined_input_tokens)

    with col_out:
        if result["empty_input"]:
            st.info(
                "👆 **No ingredients entered yet.** Select ingredients from the search box, paste an ingredient list, "
                "or click **🎲 Surprise me** to load a real What's Cooking recipe."
            )
            return

        if not st.session_state.get("has_predicted", True):
            st.warning(
                "🎲 **Mystery recipe loaded!** Click **🔍 Predict Cuisine** on the left to run Scorer v2 & Scorer v1 "
                "and reveal the recipe's actual cuisine."
            )
            return

        # Edge-case warnings
        if parsed_paste["unrecognized"]:
            unrec_names = ", ".join(f"`{u['raw']}`" for u in parsed_paste["unrecognized"])
            st.warning(
                f"⚠️ **Unrecognized ingredients ignored**: {unrec_names} do not appear in the What's Cooking "
                "training vocabulary and do not contribute to network scoring."
            )

        if result["low_confidence_warning"]:
            st.warning(
                f"⚠️ **Low-confidence warning**: Only **{result['n_recognized_backbone']}** ingredient(s) belong to the "
                "validated cuisine backbones (fewer than 2 backbone ingredients, so zero ingredient pairs can form)."
            )

        # -------------------------------------------------------------------
        # Side-by-Side Headline Results: Scorer v2 (Primary) & Scorer v1
        # -------------------------------------------------------------------
        v2 = result["v2"]
        v1 = result["v1"]
        c_v2 = v2["prediction"]
        c_v1 = v1["prediction"]
        m_v2 = CUISINE_META[c_v2]
        m_v1 = CUISINE_META[c_v1]

        status_parts = []
        if result["models_agree"]:
            status_parts.append(f"✅ **Scorer v2 & Scorer v1 Agree** on **{m_v2['title']}**")
        else:
            status_parts.append(
                f"⚡ **Models Disagree** — Scorer v2 predicts **{m_v2['title']}** vs. Scorer v1 predicts **{m_v1['title']}**"
            )

        if surprise is not None and st.session_state.get("has_predicted", False):
            act_c = surprise["cuisine"]
            m_act = CUISINE_META[act_c]
            v2_ok = "✓" if c_v2 == act_c else "✗"
            v1_ok = "✓" if c_v1 == act_c else "✗"
            status_parts.append(
                f"**Actual Cuisine: {m_act['icon']} {m_act['title']}** (v2 {v2_ok} | v1 {v1_ok})"
            )

        st.markdown(" &nbsp;•&nbsp; ".join(status_parts))

        h_col2, h_col1 = st.columns([1.25, 1.0], gap="medium")

        with h_col2:
            with st.container(border=True, key=f"v2_card_{c_v2}"):
                st.caption("HEADLINE PREDICTION — FINGERPRINT SCORER V2 (NETWORK-INFORMED STACKED)")
                st.markdown(
                    f"### {m_v2['icon']} {m_v2['title']} &nbsp; `{v2['confidence']*100:.1f}% confidence`"
                )
                st.caption(m_v2["tagline"])
                for c_key in runtime.cuisines:
                    cm = CUISINE_META[c_key]
                    prob_val = float(v2["probabilities"][c_key])
                    st.progress(
                        min(max(prob_val, 0.0), 1.0),
                        text=f"{cm['icon']} **{cm['title']}** — {prob_val*100:.1f}%",
                    )

        with h_col1:
            with st.container(border=True, key=f"v1_card_{c_v1}"):
                st.caption("BASELINE COMPARISON — FINGERPRINT SCORER V1 (NPMI SUM)")
                st.markdown(
                    f"### {m_v1['icon']} {m_v1['title']} &nbsp; `Sum NPMI: {v1['scores'][c_v1]:.3f}`"
                )
                if v1["fallback"] == "zero_score":
                    st.caption("⚠️ Zero backbone pairs matched — v1 defaulted to majority class (Italian).")
                elif v1["fallback"] == "tied_score":
                    st.caption("⚠️ Tied NPMI sum — broken by training class frequency.")
                else:
                    st.caption("Raw positive NPMI sum across matched backbone edges")

                for c_key in runtime.cuisines:
                    cm = CUISINE_META[c_key]
                    raw_s = float(v1["scores"][c_key])
                    n_m = int(v1["per_cuisine"][c_key]["n_matched"])
                    share_val = float(v1["score_shares"][c_key])
                    st.progress(
                        min(max(share_val, 0.0), 1.0),
                        text=f"{cm['icon']} **{cm['title']}** ({n_m} pairs) — NPMI sum: `{raw_s:.3f}`",
                    )

    # -----------------------------------------------------------------------
    # Bottom Full-Width Section: "Why this prediction" (v2) + Interactive Graph
    # -----------------------------------------------------------------------
    st.divider()
    why_col, graph_col = st.columns([1.25, 1.0], gap="large")

    with why_col:
        with st.container(border=True, key="why_prediction_card"):
            st.markdown("### 🔎 Why This Prediction (Fingerprint Scorer v2 Breakdown)")
            struct_per_c = v2["structural_explanation"]["per_cuisine"]

            # Summary metrics table across the 3 cuisines
            summary_rows = []
            for c_key in runtime.cuisines:
                sc_c = struct_per_c[c_key]
                v1_c = v1["per_cuisine"][c_key]
                summary_rows.append({
                    "Cuisine": CUISINE_META[c_key]["title"],
                    "v2 Probability": f"{v2['probabilities'][c_key]*100:.1f}%",
                    "Pair Log-Odds Sum (v2)": f"{sc_c['pair_log_odds_sum']:+.2f}",
                    "Matched Pairs (v1 NPMI)": f"{v1_c['n_matched']} (sum={v1_c['score']:.2f})",
                    "Backbone Nodes": f"{sc_c['backbone_ing_count']} / {result['n_total_normalized']}",
                    "Community Coverage": f"{sc_c['comm_cov_frac']*100:.0f}% (max={sc_c['comm_max_cov']*100:.0f}%)",
                    "Top-10 Hubs": sc_c["hub_count"],
                    "D3 Bridges": sc_c["bridge_count"],
                })
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

            # Per-cuisine detailed structural tabs
            c_tabs = st.tabs([
                f"🟠 Indian ({v1['per_cuisine']['indian']['n_matched']} pairs)",
                f"🟢 Italian ({v1['per_cuisine']['italian']['n_matched']} pairs)",
                f"🔴 Mexican ({v1['per_cuisine']['mexican']['n_matched']} pairs)",
            ])
            for c_idx, c_key in enumerate(runtime.cuisines):
                with c_tabs[c_idx]:
                    sc_c = struct_per_c[c_key]
                    v1_c = v1["per_cuisine"][c_key]
                    cm = CUISINE_META[c_key]

                    # 1. Matched Backbone Pairs
                    st.markdown(f"**Matched `{cm['title']}` Backbone Pairs ({v1_c['n_matched']}):**")
                    if v1_c["matched_pairs"]:
                        pair_rows = []
                        for mp in v1_c["matched_pairs"][:15]:
                            p_tup = (mp["pair"][0], mp["pair"][1])
                            lo = runtime.v2_pair_log_odds.get(p_tup, (0.0, 0.0, 0.0))
                            pair_rows.append({
                                "Ingredient A": mp["pair"][0],
                                "Ingredient B": mp["pair"][1],
                                f"{cm['title']} NPMI": round(mp["NPMI"], 3),
                                "Curveball z": round(mp["z_score"], 2),
                                f"v2 Log-Odds ({cm['title']})": round(lo[c_idx], 3),
                            })
                        st.dataframe(pd.DataFrame(pair_rows), width="stretch", hide_index=True)
                    else:
                        st.caption(f"No validated `{cm['title']}` backbone pairs matched in this recipe.")

                    # 2. Hubs & Bridges Found
                    hb_c1, hb_c2 = st.columns(2)
                    with hb_c1:
                        st.markdown(f"**D3 Top-10 Hubs Found ({sc_c['hub_count']}):**")
                        if sc_c["hubs_found"]:
                            for h in sc_c["hubs_found"]:
                                st.markdown(f"- `{h['ingredient']}` (strength: `{h['weighted_degree']:.2f}`)")
                        else:
                            st.caption("None of this cuisine's top-10 weighted-degree hubs present.")
                    with hb_c2:
                        st.markdown(f"**D3 Cross-Cuisine Bridges Found ({sc_c['bridge_count']}):**")
                        if sc_c["bridges_detail"]:
                            for bd in sc_c["bridges_detail"]:
                                st.markdown(
                                    f"- `{bd['ingredient']}` (`{bd['pair']}` rank #{bd['rank']}, score `{bd['bridge_score']:.2f}`)"
                                )
                        else:
                            st.caption("No top-5 cross-cuisine bridge ingredients present.")

                    # 3. Louvain Community Coverage
                    st.markdown(
                        f"**Louvain Community Coverage**: `{sc_c['comm_cov_frac']*100:.1f}%` of recipe "
                        f"| **Max Single-Community Share**: `{sc_c['comm_max_cov']*100:.1f}%` "
                        f"| **Spread Entropy**: `{sc_c['comm_entropy']:.3f}`"
                    )
                    if sc_c["communities_breakdown"]:
                        for cb in sc_c["communities_breakdown"]:
                            members_str = ", ".join(f"`{m}`" for m in cb["members_found"])
                            st.markdown(
                                f"- **Community {cb['community_id']}** (*{cb['community_label']}*): "
                                f"{cb['count']} ingredient(s) ({cb['share_of_recipe']*100:.0f}% of recipe) → {members_str}"
                            )

            # Unrecognized or non-backbone ingredients footer
            if result["unrecognized"] or result["outside_backbone_vocab"]:
                with st.expander("ℹ️ Unrecognized & Long-Tail Ingredients Outside the Validated Backbones"):
                    if result["unrecognized"]:
                        st.markdown(
                            "**Unrecognized (not in What's Cooking training vocabulary)**: "
                            + ", ".join(f"`{x}`" for x in result["unrecognized"])
                        )
                    if result["outside_backbone_vocab"]:
                        st.markdown(
                            "**In training vocabulary, but outside the 642-node validated backbones (rare/long-tail)**: "
                            + ", ".join(f"`{x}`" for x in result["outside_backbone_vocab"])
                        )

    with graph_col:
        with st.container(border=True, key="graph_panel_card"):
            st.markdown(f"### 🕸️ Interactive Backbone Graph ({m_v2['title']})")
            graph_c = st.selectbox(
                "Inspect matched backbone subgraph for cuisine:",
                options=runtime.cuisines,
                index=runtime.cuisines.index(c_v2),
                format_func=lambda c: f"{CUISINE_META[c]['title']} Backbone Subgraph ({v1['per_cuisine'][c]['n_matched']} matched edges)",
            )
            html_graph, n_nodes_g, n_edges_g = build_matched_backbone_pyvis_html(
                runtime, graph_c, result["normalized_input"], max_edges=25
            )
            if n_nodes_g == 0:
                st.info(f"No ingredients from this recipe appear in the `{CUISINE_META[graph_c]['title']}` backbone.")
            else:
                st.caption(
                    f"Showing **{n_nodes_g}** matched backbone node(s) and **{n_edges_g}** validated edge(s) "
                    f"in **{CUISINE_META[graph_c]['title']}**. Node color = Louvain community; "
                    "white border = D3 Top-10 Hub or Top-5 Bridge; edge width = NPMI association strength."
                )
                components.html(html_graph, height=375, scrolling=False)


def render_results_tab(runtime: CulinaryFingerprintRuntime) -> None:
    st.markdown("### 📊 Held-Out Test Set Evaluation: Fingerprint Scorer v1 vs. Scorer v2 (`N = 3,456` Test Recipes)")
    st.caption(
        "Evaluated once on the 3,456 held-out test recipes (`data/processed/split.json`) after training strictly "
        "on the 13,823 training recipes (`seed = 42`). Stored in `outputs/app/artifacts/final_test_results.csv`."
    )

    df = runtime.test_results_df.copy()
    paired = runtime.test_summary_v1_v2["paired_comparison_v2_vs_v1"]
    cms = runtime.test_summary_v1_v2["confusion_matrices"]

    # Headline KPI cards
    k1, k2, k3, k4 = st.columns(4)
    v1_row = df.iloc[0]
    v2_row = df.iloc[1]
    with k1:
        st.metric(
            "Scorer v2 Test Macro-F1 (Primary)",
            f"{v2_row['test_macro_f1']*100:.2f}%",
            f"+{paired['delta_macro_f1']*100:.2f} pp vs. Scorer v1",
        )
    with k2:
        st.metric(
            "Scorer v2 Test Accuracy",
            f"{v2_row['test_accuracy']*100:.2f}%",
            f"+{paired['delta_accuracy']*100:.2f} pp vs. Scorer v1",
        )
    with k3:
        st.metric(
            "Scorer v1 Test Macro-F1",
            f"{v1_row['test_macro_f1']*100:.2f}%",
            f"Accuracy: {v1_row['test_accuracy']*100:.2f}%",
            delta_color="off",
        )
    with k4:
        mc = paired["mcnemar"]
        st.metric(
            "Net Test Recipes Gained (v2 - v1)",
            f"+{mc['net_difference_a_minus_b']} recipes",
            f"{mc['a_right_b_wrong']} fixed vs. {mc['a_wrong_b_right']} broken (p = 7.7e-35)",
        )

    # Formatted comparison table
    display_rows = []
    for _, r in df.iterrows():
        display_rows.append({
            "Model": r["model"],
            "Test Macro-F1 (Primary)": f"{r['test_macro_f1']*100:.2f}%",
            "Test Accuracy": f"{r['test_accuracy']*100:.2f}%",
            "5-Fold Train CV Macro-F1": f"{r['cv_macro_f1_mean']*100:.2f}% ± {r['cv_macro_f1_std']*100:.2f}%",
            "5-Fold Train CV Accuracy": f"{r['cv_accuracy_mean']*100:.2f}% ± {r['cv_accuracy_std']*100:.2f}%",
            "Indian Recall / F1": f"{r['recall_indian']*100:.2f}% / {r['f1_indian']*100:.2f}%",
            "Italian Recall / F1": f"{r['recall_italian']*100:.2f}% / {r['f1_italian']*100:.2f}%",
            "Mexican Recall / F1": f"{r['recall_mexican']*100:.2f}% / {r['f1_mexican']*100:.2f}%",
            "Zero-Score Fallbacks": int(r["zero_score_fallback_count"]),
        })
    st.dataframe(pd.DataFrame(display_rows), width="stretch", hide_index=True)

    # Paired statistical significance summary
    f1_ci = paired["bootstrap_delta_macro_f1_ci95"]
    acc_ci = paired["bootstrap_delta_accuracy_ci95"]
    with st.container(border=True, key="paired_stats_card"):
        st.caption("PAIRED STATISTICAL COMPARISON ON HELD-OUT TEST SET (N = 3,456)")
        st.markdown(
            f"- **Δ Macro-F1 (Scorer v2 minus Scorer v1)**: **+{paired['delta_macro_f1']*100:.2f} pp** "
            f"(95% paired bootstrap CI: `[+{f1_ci[0]*100:.2f} pp, +{f1_ci[1]*100:.2f} pp]`).\n"
            f"- **Δ Accuracy (Scorer v2 minus Scorer v1)**: **+{paired['delta_accuracy']*100:.2f} pp** "
            f"(95% paired bootstrap CI: `[+{acc_ci[0]*100:.2f} pp, +{acc_ci[1]*100:.2f} pp]`).\n"
            f"- **Paired McNemar Test**: Scorer v2 is correct on **{mc['a_right_b_wrong']}** test recipes where Scorer v1 errs, "
            f"while Scorer v1 is correct on only **{mc['a_wrong_b_right']}** test recipes where Scorer v2 errs "
            f"(χ²_CC = `{mc['chi2_continuity_corrected']:.2f}`, exact binomial *p* = `{mc['p_value_exact_binomial']:.3e}`).\n"
            f"- **Per-Cuisine Recall Gains (v2 minus v1)**: "
            f"Indian: **+{paired['recall_diffs']['indian']*100:.2f} pp** (80.00% → 91.33%) | "
            f"Italian: **+{paired['recall_diffs']['italian']*100:.2f} pp** (91.71% → 96.56%) | "
            f"Mexican: **+{paired['recall_diffs']['mexican']*100:.2f} pp** (86.34% → 92.47%)."
        )

    # Confusion matrices side by side
    cm_col1, cm_col2 = st.columns(2, gap="large")
    c_titles = [CUISINE_META[c]["title"] for c in runtime.cuisines]

    with cm_col1:
        with st.container(border=True, key="cm_v1_card"):
            st.markdown("#### Confusion Matrix — Fingerprint Scorer v1")
            mat1 = np.asarray(cms["Fingerprint Scorer v1 (Original NPMI Sum)"], dtype=int)
            df_cm1 = pd.DataFrame(
                mat1,
                index=[f"True {t}" for t in c_titles],
                columns=[f"Pred {t}" for t in c_titles],
            )
            df_cm1["Row Total"] = df_cm1.sum(axis=1)
            st.dataframe(df_cm1, width="stretch")

    with cm_col2:
        with st.container(border=True, key="cm_v2_card"):
            st.markdown("#### Confusion Matrix — Fingerprint Scorer v2")
            mat2 = np.asarray(cms["Fingerprint Scorer v2 (Network-Informed Stacked)"], dtype=int)
            df_cm2 = pd.DataFrame(
                mat2,
                index=[f"True {t}" for t in c_titles],
                columns=[f"Pred {t}" for t in c_titles],
            )
            df_cm2["Row Total"] = df_cm2.sum(axis=1)
            st.dataframe(df_cm2, width="stretch")


def render_how_it_works_tab(runtime: CulinaryFingerprintRuntime) -> None:
    st.markdown("### 🧭 How Culinary Fingerprint Works (Deliverables 1–4 in Plain Language)")
    ov = runtime.test_summary_v1_v2["pipeline_overview"]

    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True, key="how_d1_card"):
            st.caption("DELIVERABLE 1 — CO-OCCURRENCE NETWORKS")
            st.markdown("#### 1. Normalization & PMI / NPMI Edge Construction")
            st.markdown(
                "Every ingredient string passes through a deterministic 5-stage normalizer (Unicode & trademark cleaning, "
                "lowercasing, whitespace collapsing, conservative count-noun singularization, and a 5-entry botanical synonym table). "
                f"On the **13,823 training recipes**, we build a separate co-occurrence network for each cuisine over "
                f"eligible ingredients occurring in at least **20 training recipes** "
                f"(`Indian: {ov['d1']['eligible_nodes']['indian']}`, "
                f"`Italian: {ov['d1']['eligible_nodes']['italian']}`, "
                f"`Mexican: {ov['d1']['eligible_nodes']['mexican']}` nodes) "
                "and weight each co-occurring ingredient pair by Normalized Pointwise Mutual Information (**NPMI** ∈ [-1, 1])."
            )

        with st.container(border=True, key="how_d3_card"):
            st.caption("DELIVERABLE 3 — MESO-SCALE NETWORK STRUCTURE")
            st.markdown("#### 3. Hubs, Louvain Communities & Cross-Cuisine Bridges")
            st.markdown(
                "From each cuisine's validated backbone graph, we extract three structural roles:\n"
                "- **Top-10 Weighted-Degree Hubs**: Core ingredients with the highest sum of positive NPMI incident edges.\n"
                f"- **Louvain Communities**: Cohesive flavor modules (`{ov['d3']['n_communities']['indian']}` per cuisine, "
                "such as baking/pastry modules, masala bases, or pasta-cheese modules).\n"
                "- **Cross-Cuisine Bridges**: Shared backbone nodes scored by "
                "`Bridge(u; A, B) = min(Strength_A(u), Strength_B(u))` (top-5 per cuisine pair)."
            )

    with c2:
        with st.container(border=True, key="how_d2_card"):
            st.caption("DELIVERABLE 2 — STATISTICAL BACKBONE EXTRACTION")
            st.markdown("#### 2. Curveball Null-Model Validation")
            st.markdown(
                "Raw co-occurrence counts and NPMI can be distorted by very common staples or rare coincidences. "
                "We run **1,000 Curveball bipartite Markov-chain randomizations** per cuisine — preserving the exact number of "
                "ingredients in every recipe and the exact frequency of every ingredient — and retain only edges with "
                "**z > 3.0** and co-occurrence **≥ 5**. This prunes over 92% of raw edges, leaving "
                f"**{ov['d2']['backbone_edges']['indian']:,} Indian**, "
                f"**{ov['d2']['backbone_edges']['italian']:,} Italian**, and "
                f"**{ov['d2']['backbone_edges']['mexican']:,} Mexican** validated backbone edges "
                f"(**{ov['d2']['union_backbone_pairs']:,}** unique pairs across **642** backbone ingredients)."
            )

        with st.container(border=True, key="how_d4_card"):
            st.caption("DELIVERABLE 4 — FINGERPRINT SCORER V1 VS. SCORER V2")
            st.markdown("#### 4. From Raw NPMI Sums (v1) to Stacked Network Summaries (v2)")
            st.markdown(
                "- **Scorer v1 (87.12% Test Macro-F1)**: Sums raw within-cuisine backbone NPMI over matched pairs. Because "
                "Italian has twice as many backbone edges as Indian and raw NPMI has no negative evidence against competing cuisines, "
                "shared staples over-predict Italian and short recipes with zero matched pairs default to Italian.\n"
                "- **Scorer v2 (93.99% Test Macro-F1, +6.87 pp)**: Combines **22 purely network-derived features** "
                "(contrastive backbone pair log-odds, Louvain community coverage, max single-community concentration, community entropy, "
                "D3 hub counts, D3 bridge counts, backbone node coverage, and recipe length) in an L2-regularized logistic regression "
                "trained via 3-fold cross-fitting on the training split."
            )


def main() -> None:
    apply_theme()
    init_session_state()
    runtime = load_runtime()
    render_header()

    tab_pred, tab_res, tab_how = st.tabs([
        "🔍 Predict",
        "📊 Results (Held-Out Test)",
        "🧭 How it works",
    ])
    with tab_pred:
        render_predict_tab(runtime)
    with tab_res:
        render_results_tab(runtime)
    with tab_how:
        render_how_it_works_tab(runtime)


if __name__ == "__main__":
    main()
