"""Parity and self-containment acceptance tests for the Culinary Fingerprint Streamlit app (`outputs/app/`).

Verifies:
1. Required app files and exported artifacts exist in `outputs/app/` and `outputs/app/artifacts/`.
2. Strict absence of any Naive Bayes reference anywhere in the app code, documentation, or exported artifacts.
3. Strict absence of any `src/d4_experiments` import in `outputs/app/scorer_runtime.py` or `outputs/app/app.py`.
4. Exact parity on 250 TRAINING recipes (>= 200 required):
   - App Scorer v1 predictions and per-cuisine NPMI scores match the pipeline's `FingerprintScorer` identically.
   - App Scorer v2 predictions match the pipeline's `M5: Network-Only Fingerprint Scorer v2` identically,
     and v2 predicted probabilities match to within `1e-6` (`max_abs_diff < 1e-6`).
5. Edge cases and UI helpers: empty input, single ingredient (low-confidence warning), pasted text with
   unrecognized items, and offline PyVis network graph generation (zero external CDN script/link tags).
"""
from __future__ import annotations

import ast
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "outputs" / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app import build_matched_backbone_pyvis_html
from scorer_runtime import CulinaryFingerprintRuntime
from src.common import load_config, read_json
from src.d4_experiments.phase1_diagnostics import normalize_partition_records
from src.d4_experiments.phase3_final_test import (
    load_train_and_test_splits_for_final_eval,
)
from src.d4_experiments.phase3_scorer_v2 import (
    FEATURE_NAMES,
    INNER_CURVEBALL_N_RAND,
    INNER_N_SPLITS,
    META_C_GRID,
    PAIR_LOG_ODDS_EPS,
    PHASE3_CACHE_DIR,
    SEED,
    PartitionNetworkFeatureExtractor,
    get_feature_subsets,
    tune_and_fit_standardized_meta_logreg,
)
from src.fingerprint import DEFAULT_ARTIFACTS, FingerprintScorer
from src.normalize import normalize_ingredient, normalize_recipe
from sklearn.model_selection import StratifiedKFold


def test_app_directory_structure_and_artifacts_exist():
    required_files = [
        APP_DIR / "app.py",
        APP_DIR / "export_artifacts.py",
        APP_DIR / "scorer_runtime.py",
        APP_DIR / ".streamlit" / "config.toml",
        APP_DIR / "README.md",
        APP_DIR / "requirements_app.txt",
        APP_DIR / "artifacts" / "backbone_indian.csv",
        APP_DIR / "artifacts" / "backbone_italian.csv",
        APP_DIR / "artifacts" / "backbone_mexican.csv",
        APP_DIR / "artifacts" / "hubs_indian.csv",
        APP_DIR / "artifacts" / "hubs_italian.csv",
        APP_DIR / "artifacts" / "hubs_mexican.csv",
        APP_DIR / "artifacts" / "communities_indian.csv",
        APP_DIR / "artifacts" / "communities_italian.csv",
        APP_DIR / "artifacts" / "communities_mexican.csv",
        APP_DIR / "artifacts" / "bridges.csv",
        APP_DIR / "artifacts" / "v2_model.json",
        APP_DIR / "artifacts" / "pair_log_odds_v2.csv",
        APP_DIR / "artifacts" / "normalization_rules.json",
        APP_DIR / "artifacts" / "ingredient_vocabulary.json",
        APP_DIR / "artifacts" / "sample_recipes.json",
        APP_DIR / "artifacts" / "final_test_results.csv",
        APP_DIR / "artifacts" / "final_test_summary_v1_v2.json",
    ]
    for p in required_files:
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty app file: {p}"


def test_app_contains_no_naive_bayes_references_and_no_d4_experiments_imports():
    forbidden_patterns = [
        r"naive\s*[_-]?\s*bayes",
        r"multinomial\s*nb",
        r"binarymultinomialnb",
        r"\bnb_log_prob\b",
        r"\bnb_pred\b",
    ]
    # Check all code, config, markdown, and exported artifact files inside outputs/app/
    for p in APP_DIR.rglob("*"):
        if p.is_dir() or "__pycache__" in p.parts:
            continue
        if p.suffix.lower() in {".py", ".md", ".toml", ".txt", ".json", ".csv"}:
            text = p.read_text(encoding="utf-8")
            for pat in forbidden_patterns:
                m = re.search(pat, text, flags=re.IGNORECASE)
                assert m is None, f"Forbidden reference {m.group(0)!r} found in {p}"

    # Verify scorer_runtime.py and app.py do not import from src/d4_experiments
    for py_file in [APP_DIR / "scorer_runtime.py", APP_DIR / "app.py"]:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "d4_experiments" not in alias.name, f"Forbidden import {alias.name} in {py_file}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "d4_experiments" not in mod, f"Forbidden import from {mod} in {py_file}"

    # Verify final_test_results.csv in outputs/app/artifacts/ has only Scorer v1 and Scorer v2
    df_res = pd.read_csv(APP_DIR / "artifacts" / "final_test_results.csv")
    assert len(df_res) == 2
    assert list(df_res["model"]) == [
        "Fingerprint Scorer v1 (Original NPMI Sum)",
        "Fingerprint Scorer v2 (Network-Informed Stacked)",
    ]


def test_app_parity_on_250_training_recipes_v1_and_v2():
    """Verify on 250 TRAINING recipes (>= 200 required) that app Scorer v1 and Scorer v2 predictions
    equal the pipeline's, and v2 probabilities match to within 1e-6.
    """
    cfg = load_config()
    cuisines = list(cfg["cuisines"])
    c_to_idx = {c: i for i, c in enumerate(cuisines)}
    train_raw, _, train_ids, test_ids, dataset_sha = load_train_and_test_splits_for_final_eval(cfg)
    train_recipes = normalize_partition_records(train_raw)

    # Select 250 training recipes deterministically across the training set (including short, medium, and long recipes)
    rng = np.random.default_rng(SEED)
    sample_indices = sorted(rng.choice(len(train_raw), size=250, replace=False))
    sample_train_raw = [train_raw[i] for i in sample_indices]
    sample_train_norm = [train_recipes[i] for i in sample_indices]
    assert len(sample_train_raw) == 250

    # Instantiate app runtime
    rt = CulinaryFingerprintRuntime()

    # 1. Normalization parity across all 250 training recipes
    for raw_r in sample_train_raw:
        assert rt.normalize_recipe(raw_r["ingredients"]) == normalize_recipe(raw_r["ingredients"])
        for ing in raw_r["ingredients"]:
            assert rt.normalize_ingredient(ing) == normalize_ingredient(ing)

    # 2. Scorer v1 parity with pipeline's FingerprintScorer
    pipeline_v1 = FingerprintScorer.from_artifacts(DEFAULT_ARTIFACTS, normalized=False)
    v1_matches = 0
    for raw_r in sample_train_raw:
        app_v1 = rt.predict_v1(raw_r["ingredients"])
        pipe_v1 = pipeline_v1.explain(raw_r["ingredients"])
        assert app_v1["prediction"] == pipe_v1["prediction"], f"v1 prediction mismatch on recipe {raw_r['id']}"
        assert app_v1["fallback"] == pipe_v1["fallback"], f"v1 fallback mismatch on recipe {raw_r['id']}"
        for c in cuisines:
            assert math.isclose(app_v1["scores"][c], pipe_v1["scores"][c], rel_tol=1e-12, abs_tol=1e-12)
        v1_matches += 1
    assert v1_matches == 250

    # 3. Scorer v2 parity with the pipeline's M5 model trained on all 13,823 training recipes
    from src.d4_experiments.phase3_final_test import build_full_training_extractor_from_existing_d4_artifacts

    ext_full = build_full_training_extractor_from_existing_d4_artifacts(
        cfg, train_recipes, train_ids, test_ids, dataset_sha, DEFAULT_ARTIFACTS
    )
    X_pipe_25 = ext_full.transform(sample_train_norm)
    cols_m5 = get_feature_subsets(cuisines)["M5_network_only_v2"]
    X_pipe_22 = X_pipe_25[:, cols_m5]

    # Reconstruct pipeline M5 probabilities using the saved full-training M5 weights from final_test_summary.json
    final_summary = read_json(ROOT / "outputs" / "d4_experiments" / "final_test_summary.json")
    v2_art = read_json(APP_DIR / "artifacts" / "v2_model.json")

    # Verify the exported v2_model.json coefficients match final_test_summary.json M5 coefficients to 1e-8
    pipe_m5_coefs = final_summary["standardized_coefficients_full_train"]["M5_network_only_22_features"]
    pipe_m5_int = final_summary["standardized_coefficients_full_train"]["M5_intercept"]
    pipe_coef_mat = np.zeros((len(cuisines), len(v2_art["feature_names"])), dtype=np.float64)
    pipe_int_vec = np.zeros(len(cuisines), dtype=np.float64)
    for j, row_c in enumerate(pipe_m5_coefs):
        assert row_c["feature"] == v2_art["feature_names"][j]
        for ci, c in enumerate(cuisines):
            pipe_coef_mat[ci, j] = float(row_c[c])
            assert math.isclose(float(row_c[c]), float(v2_art["coef"][ci][j]), rel_tol=1e-8, abs_tol=1e-8)
    for ci, c in enumerate(cuisines):
        pipe_int_vec[ci] = float(pipe_m5_int[c])
        assert math.isclose(float(pipe_m5_int[c]), float(v2_art["intercept"][ci]), rel_tol=1e-8, abs_tol=1e-8)

    max_prob_diff = 0.0
    v2_matches = 0
    for idx_local, raw_r in enumerate(sample_train_raw):
        app_v2 = rt.predict_v2(raw_r["ingredients"])
        # Pipeline feature vector and pipeline M5 weights for this recipe
        x_22 = X_pipe_22[idx_local]
        x_std = (x_22 - rt.scaler_mean) / rt.scaler_scale
        logits = pipe_coef_mat @ x_std + pipe_int_vec
        e_l = np.exp(logits - np.max(logits))
        pipe_probs = e_l / np.sum(e_l)
        pipe_pred = cuisines[int(np.argmax(pipe_probs))]

        assert app_v2["prediction"] == pipe_pred, f"v2 prediction mismatch on recipe {raw_r['id']}"
        for ci, c in enumerate(cuisines):
            diff = abs(app_v2["probabilities"][c] - float(pipe_probs[ci]))
            if diff > max_prob_diff:
                max_prob_diff = diff
            assert diff < 1e-6, f"v2 probability diff {diff} >= 1e-6 on recipe {raw_r['id']} ({c})"
        v2_matches += 1

    assert v2_matches == 250
    assert max_prob_diff < 1e-6


def test_app_edge_cases_and_offline_pyvis_graph():
    rt = CulinaryFingerprintRuntime()

    # 1. Empty input
    empty_res = rt.predict_both([])
    assert empty_res["empty_input"] is True
    assert empty_res["low_confidence_warning"] is True

    # 2. Fewer than 2 recognized backbone ingredients -> low_confidence_warning = True
    single_res = rt.predict_both(["Garam Masala", "completely_unknown_ingredient_xyz"])
    assert single_res["empty_input"] is False
    assert single_res["low_confidence_warning"] is True
    assert single_res["recognized_vocab"] == ["garam masala"]
    assert single_res["unrecognized"] == ["completely_unknown_ingredient_xyz"]

    # 3. Free-text paste parser with synonyms, plurals, and unrecognized tokens
    parsed = rt.parse_pasted_text("Roma Tomatoes, Scallions, Garbanzo Beans\nUnseen Alien Spice 99")
    assert "roma tomato" in parsed["recognized"]
    assert "green onion" in parsed["recognized"]
    assert "chickpea" in parsed["recognized"]
    assert len(parsed["unrecognized"]) == 1
    assert parsed["unrecognized"][0]["raw"] == "Unseen Alien Spice 99"

    # 4. Offline PyVis HTML graph generation (>=480px height, network.fit(), visible labels, no external CDN tags)
    html_str, n_nodes, n_edges = build_matched_backbone_pyvis_html(
        rt, "indian", ["garlic", "ginger", "onion", "tomato", "garam masala", "ground turmeric"]
    )
    assert n_nodes >= 4 and n_edges >= 3
    assert "vis-network" in html_str or "vis.Network" in html_str
    assert "500px" in html_str
    assert "network.fit(" in html_str
    for expected_node in ["garlic", "ginger", "onion", "tomato", "garam masala", "ground turmeric"]:
        assert f'"label": "{expected_node}"' in html_str
    ext_links = re.findall(r"<(?:link|script)[^>]+(?:href|src)=[\"']https?://[^\"']+[\"']", html_str, flags=re.I)
    assert len(ext_links) == 0, f"Offline graph HTML contains external CDN tags: {ext_links}"


def test_panel_data_per_cuisine_comes_from_that_cuisines_hub_and_community_tables():
    """Verify for every cuisine (indian, italian, mexican) that get_cuisine_panel_data reads strictly
    from that cuisine's hub and community tables and matches the exact v2 feature vector values.
    """
    from app import PRESET_RECIPES, get_cuisine_panel_data

    rt = CulinaryFingerprintRuntime()

    # Load ground-truth per-cuisine hub and community CSV tables from outputs/app/artifacts/
    hubs_tables: dict[str, dict[str, float]] = {}
    comms_tables: dict[str, dict[str, tuple[int, str]]] = {}
    for c in rt.cuisines:
        h_df = pd.read_csv(APP_DIR / "artifacts" / f"hubs_{c}.csv")
        top10_h = h_df[h_df["top10_weighted_degree"] == True]
        hubs_tables[c] = {
            str(r["ingredient"]): round(float(r["weighted_degree"]), 3)
            for _, r in top10_h.iterrows()
        }

        c_df = pd.read_csv(APP_DIR / "artifacts" / f"communities_{c}.csv")
        comms_tables[c] = {
            str(r["ingredient"]): (int(r["community_id"]), str(r["community_label"]))
            for _, r in c_df.iterrows()
        }

    # Test across recipes from all three cuisines, including Mexican recipes that contain 'green chilies'
    test_ingredient_lists = [
        PRESET_RECIPES["indian"],
        PRESET_RECIPES["italian"],
        PRESET_RECIPES["mexican"],
        ["american cheese", "chili con carne", "green chilies", "tortilla chips"],
    ]

    for ings in test_ingredient_lists:
        res = rt.predict_both(ings)
        raw_feats = res["v2"]["raw_features"]
        L = len(res["normalized_input"])

        for c in rt.cuisines:
            panel = get_cuisine_panel_data(rt, res, c)
            assert panel["cuisine"] == c

            # 1. Hubs in panel must come strictly from hubs_{c}.csv top-10 weighted-degree rows
            expected_hubs_c = {u for u in res["normalized_input"] if u in hubs_tables[c]}
            panel_hub_names = {h["ingredient"] for h in panel["hubs_found"]}
            assert panel_hub_names == expected_hubs_c, (
                f"Hub mismatch for cuisine {c}: got {panel_hub_names}, expected {expected_hubs_c}"
            )
            for h in panel["hubs_found"]:
                assert math.isclose(
                    h["weighted_degree"], hubs_tables[c][h["ingredient"]], rel_tol=1e-9, abs_tol=1e-9
                ), f"Hub weighted degree for {h['ingredient']} in {c} does not match hubs_{c}.csv"

            # 2. Communities in panel must come strictly from communities_{c}.csv
            for cb in panel["communities_breakdown"]:
                cid = cb["community_id"]
                clabel = cb["community_label"]
                for m_ing in cb["members_found"]:
                    assert m_ing in comms_tables[c], f"{m_ing} not found in communities_{c}.csv"
                    exp_cid, exp_label = comms_tables[c][m_ing]
                    assert cid == exp_cid, f"Community ID mismatch for {m_ing} in {c}: {cid} != {exp_cid}"
                    assert clabel == exp_label, (
                        f"Community label mismatch for {m_ing} in {c}: {clabel!r} != {exp_label!r}"
                    )

            # 3. Community coverage numbers must equal the exact features v2 uses for that cuisine
            assert math.isclose(panel["comm_cov_frac"], raw_feats[f"comm_cov_frac_{c}"], rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(panel["comm_max_cov"], raw_feats[f"comm_max_cov_{c}"], rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(panel["comm_entropy"], raw_feats[f"comm_entropy_{c}"], rel_tol=1e-12, abs_tol=1e-12)
            assert panel["hub_count"] == int(round(raw_feats[f"hub_count_{c}"]))
            assert panel["bridge_count"] == int(round(raw_feats[f"bridge_count_{c}"]))
            assert panel["backbone_ing_count"] == int(round(raw_feats[f"backbone_ing_count_{c}"]))

            if L > 0:
                total_members = sum(cb["count"] for cb in panel["communities_breakdown"])
                assert math.isclose(panel["comm_cov_frac"], total_members / float(L), rel_tol=1e-12, abs_tol=1e-12)


def _assert_no_raw_html_in_rendered_elements(at) -> None:
    """Assert that no rendered markdown, caption, info, warning, or success element contains '<div', 'style=', or inline code backticks."""
    text_elements = (
        [m.value for m in at.markdown]
        + [c.value for c in at.caption]
        + [i.value for i in at.info]
        + [w.value for w in at.warning]
        + [s.value for s in at.success]
    )
    assert len(text_elements) > 0
    for txt in text_elements:
        assert "<div" not in txt, f"Raw '<div' found in rendered output: {txt!r}"
        assert "style=" not in txt, f"Raw 'style=' found in rendered output: {txt!r}"
        assert "`" not in txt, f"Code-styled backtick found in rendered output: {txt!r}"


def _collect_tab_texts_and_columns(tab) -> list[str]:
    """Collect all rendered text strings and dataframe column/cell strings inside a Streamlit AppTest tab."""
    items: list[str] = []
    for m in tab.markdown:
        items.append(str(m.value))
    for c in tab.caption:
        items.append(str(c.value))
    for i in tab.info:
        items.append(str(i.value))
    for w in tab.warning:
        items.append(str(w.value))
    for s in tab.success:
        items.append(str(s.value))
    for met in tab.metric:
        items.append(str(met.label))
        items.append(str(met.value))
    for df_el in tab.dataframe:
        df_val = df_el.value
        if isinstance(df_val, pd.DataFrame):
            items.extend(str(col) for col in df_val.columns)
            for row_vals in df_val.astype(str).values:
                items.extend(str(v) for v in row_vals)
    return items


def test_streamlit_apptest_renders_and_interacts_cleanly():
    from streamlit.testing.v1 import AppTest

    # Verify offline SVG pattern asset exists and is embedded as a base64 data URI via apply_theme()
    svg_asset = APP_DIR / "assets" / "ingredient_pattern.svg"
    assert svg_asset.exists() and svg_asset.stat().st_size > 0
    svg_text = svg_asset.read_text(encoding="utf-8")
    svg_no_xmlns = svg_text.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "<svg" in svg_text and "https://" not in svg_no_xmlns and "http://" not in svg_no_xmlns

    from app import PRESET_RECIPES, _get_pattern_data_uri, apply_theme, format_confidence

    assert callable(apply_theme)
    uri = _get_pattern_data_uri()
    assert uri.startswith("data:image/svg+xml;base64,")

    # Verify confidence formatting rules (>99.9% displays as '>99%', <0.1% displays as '<0.1%')
    assert format_confidence(0.9995) == ">99%"
    assert format_confidence(1.0) == ">99%"
    assert format_confidence(0.999) == "99.9%"
    assert format_confidence(0.942) == "94.2%"
    assert format_confidence(0.0005) == "<0.1%"
    assert format_confidence(0.0) == "<0.1%"

    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Streamlit AppTest raised exception on initial render: {at.exception}"
    assert len(at.tabs) == 6
    tab_labels = [t.label for t in at.tabs]
    assert "🔍 Predict" in tab_labels
    assert "📊 Results (Held-Out Test)" in tab_labels
    assert "🧭 How it works" in tab_labels
    _assert_no_raw_html_in_rendered_elements(at)

    # Verify summary table last column is 'Nodes found'
    summary_df = at.tabs[0].dataframe[0].value
    assert list(summary_df.columns)[-1] == "Nodes found"

    # Switch to a Mexican recipe containing 'green chilies' and verify the Mexican tab is first/default
    at.multiselect[0].set_value(PRESET_RECIPES["mexican"]).run()
    assert not at.exception
    cuisine_subtabs = [
        t for t in at.tabs if t.label not in {"🔍 Predict", "📊 Results (Held-Out Test)", "🧭 How it works"}
    ]
    assert cuisine_subtabs[0].label.startswith("🔴 Mexican"), (
        f"Expected Mexican tab first for a Mexican prediction, got {cuisine_subtabs[0].label!r}"
    )
    mex_tab_text = "\n".join(_collect_tab_texts_and_columns(cuisine_subtabs[0]))
    assert "Mexican Top-10 Hubs Found" in mex_tab_text
    assert "Mexican Louvain Community Coverage" in mex_tab_text
    assert "sour cream / shredded cheddar cheese / taco seasoning mix" in mex_tab_text

    # Click 'Surprise me' button -> hides actual cuisine until 'Predict' is clicked
    surprise_btn = [b for b in at.button if "Surprise me" in b.label][0]
    surprise_btn.click().run()
    assert not at.exception
    assert at.session_state["has_predicted"] is False
    assert at.session_state["surprise_recipe"] is not None
    _assert_no_raw_html_in_rendered_elements(at)

    # Click 'Predict' button -> reveals single-card prediction and actual cuisine
    predict_btn = [b for b in at.button if "Predict" in b.label][0]
    assert predict_btn.label == "🔍 Predict"
    predict_btn.click().run()
    assert not at.exception
    assert at.session_state["has_predicted"] is True
    _assert_no_raw_html_in_rendered_elements(at)

    # Click 'Clear' button -> empties ingredients
    clear_btn = [b for b in at.button if "Clear" in b.label][0]
    clear_btn.click().run()
    assert not at.exception
    assert at.session_state["selected_ingredients"] == []
    _assert_no_raw_html_in_rendered_elements(at)


def test_predict_tab_contains_no_v1_or_npmi_sum_and_results_tab_labels():
    """Verify that the Predict tab (and app header) contains no 'v1' or 'NPMI sum' mentions,
    shows a single result card, and the Results tab compares 'Baseline (v1)' vs. 'Network-informed (v2)'.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception

    # Identify tabs by label: Predict tab + its 3 cuisine sub-tabs vs Results tab vs How it works tab
    predict_tabs = [
        t for t in at.tabs if t.label not in {"📊 Results (Held-Out Test)", "🧭 How it works"}
    ]
    results_tab = [t for t in at.tabs if t.label == "📊 Results (Held-Out Test)"][0]

    predict_texts: list[str] = []
    for pt in predict_tabs:
        predict_texts.extend(_collect_tab_texts_and_columns(pt))

    # Also include the top header markdown/captions outside tabs (first 2 markdown/caption elements)
    header_texts = [at.markdown[0].value, at.caption[0].value, at.info[0].value]
    all_predict_view_texts = header_texts + predict_texts
    assert len(all_predict_view_texts) > 10

    for txt in all_predict_view_texts:
        low = txt.lower()
        assert "v1" not in low, f"Forbidden 'v1' mention found on Predict tab/header: {txt!r}"
        assert "npmi sum" not in low, f"Forbidden 'NPMI sum' mention found on Predict tab/header: {txt!r}"
        assert "agree" not in low and "disagree" not in low, (
            f"Forbidden Agree/Disagree badge text found on Predict tab: {txt!r}"
        )

    # Verify single result card caption appears once on the Predict tab
    pred_captions = [c.value for c in predict_tabs[0].caption]
    result_card_captions = [
        c for c in pred_captions if "PREDICTED CUISINE — NETWORK-INFORMED FINGERPRINT SCORER" in c
    ]
    assert len(result_card_captions) == 1

    # Verify Results tab contains "Baseline (v1)" and "Network-informed (v2)"
    results_texts = "\n".join(_collect_tab_texts_and_columns(results_tab))
    assert "Baseline (v1)" in results_texts
    assert "Network-informed (v2)" in results_texts



