"""Deliverable 1 & normalisation acceptance tests."""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd

from src.build_networks import edge_table, node_table, pmi_npmi
from src.normalize import normalize_ingredient, normalize_recipe
from src.preprocess import load_processed


# --------------------------------------------------------------------------- normalisation unit tests
def test_normalization_deterministic_and_idempotent():
    samples = [
        "  Tomatoes ",
        "Green  Onions",
        "Scallions",
        "Bertolli\u00ae Alfredo Sauce",
        "cr\u00e8me fra\u00eeche",
        "1% low-fat milk",
        "(14.5 oz.) diced tomatoes",
        "garbanzo beans",
        "asafetida",
        "chat masala",
        "ground cumin",
        "cumin seed",
    ]
    for s in samples:
        first = normalize_ingredient(s)
        second = normalize_ingredient(first)
        assert first == second, f"Not idempotent on {s!r}: {first!r} != {second!r}"


def test_normalization_preserves_meaningful_distinctions_and_deduplicates():
    # Fresh vs dried, ground vs seed, oils, cheeses, peppers, quantities/digits preserved
    assert normalize_ingredient("fresh cilantro") != normalize_ingredient("dried cilantro")
    assert normalize_ingredient("ground cumin") != normalize_ingredient("cumin seed")
    assert normalize_ingredient("ground turmeric") != normalize_ingredient("fresh turmeric")
    assert normalize_ingredient("extra-virgin olive oil") != normalize_ingredient("sesame oil")
    assert normalize_ingredient("parmesan cheese") != normalize_ingredient("mozzarella cheese")
    assert normalize_ingredient("1% low-fat milk") == "1% low-fat milk"
    # Intended merges
    assert normalize_ingredient("tomatoes") == "tomato"
    assert normalize_ingredient("onions") == "onion"
    assert normalize_ingredient("scallions") == "green onion"
    assert normalize_ingredient("green onions") == "green onion"
    # Recipe-level deduplication and order independence
    r1 = normalize_recipe(["Scallions", "green onions", "Tomatoes", "tomato"])
    r2 = normalize_recipe(["tomato", "Green Onions"])
    assert r1 == r2 == ["green onion", "tomato"]


def test_trademark_and_symbol_normalization_regression():
    # Regression test for trademark/copyright symbols (™, ®, ©) and narrow 'veal scallops' exception
    cases = {
        "Green Giant\u2122 sliced mushrooms": "green giant sliced mushroom",
        "Old El Paso\u2122 chopped green chiles": "old el paso chopped green chiles",
        "Old El Paso\u2122 taco seasoning mix": "old el paso taco seasoning mix",
        "Pillsbury\u2122 classic pizza crust": "pillsbury classic pizza crust",
        "Pillsbury\u2122 Crescent Recipe Creations\u00ae refrigerated seamless dough sheet": (
            "pillsbury crescent recipe creations refrigerated seamless dough sheet"
        ),
        "Knorr\u00ae Fiesta Sides\u2122 - Mexican Rice": "knorr fiesta sides - mexican rice",
        "Bertolli\u00ae Alfredo Sauce": "bertolli alfredo sauce",
        "TestBrand\u00a9 Extra Virgin Olive Oil": "testbrand extra virgin olive oil",
        "scallops": "scallop",
        "sea scallops": "sea scallop",
        "bay scallops": "bay scallop",
        "veal scallops": "veal scallops",
    }
    for raw, expected in cases.items():
        actual = normalize_ingredient(raw)
        assert actual == expected, f"Expected {raw!r} -> {expected!r}, got {actual!r}"
        assert "tm" not in actual.split(), f"Corrupted trademark token in {actual!r}"
        assert normalize_ingredient(actual) == actual


# --------------------------------------------------------------------------- hand-computed PMI/NPMI tests
def test_pmi_npmi_hand_computed():
    # N = 100 recipes, n(a) = 50, n(b) = 25, n(a,b) = 20
    # P(a)=0.5, P(b)=0.25, P(a,b)=0.20 => PMI = log2(0.20 / 0.125) = log2(1.6)
    # NPMI = log2(1.6) / (-log2(0.20))
    pmi, npmi = pmi_npmi(np.array([20]), np.array([50]), np.array([25]), 100)
    expected_pmi = math.log2(0.20 / (0.5 * 0.25))
    expected_npmi = expected_pmi / (-math.log2(0.20))
    assert math.isclose(pmi[0], expected_pmi, rel_tol=1e-12)
    assert math.isclose(npmi[0], expected_npmi, rel_tol=1e-12)

    # Degenerate case P(a,b) = 1: N = 10, n(a)=10, n(b)=10, co=10 -> PMI=0, NPMI=1.0 by convention
    pmi_deg, npmi_deg = pmi_npmi(np.array([10]), np.array([10]), np.array([10]), 10)
    assert math.isclose(pmi_deg[0], 0.0, abs_tol=1e-12)
    assert math.isclose(npmi_deg[0], 1.0, rel_tol=1e-12)

    # Negative PMI case: N = 100, n(a)=80, n(b)=80, co=50 -> P(a,b)=0.50 < 0.64
    pmi_neg, npmi_neg = pmi_npmi(np.array([50]), np.array([80]), np.array([80]), 100)
    assert pmi_neg[0] < 0 and npmi_neg[0] < 0


def test_hand_constructed_fixture_network():
    recipes = [
        {"id": 1, "cuisine": "indian", "ingredients": ["a", "b", "c"]},
        {"id": 2, "cuisine": "indian", "ingredients": ["a", "b"]},
        {"id": 3, "cuisine": "indian", "ingredients": ["a", "c"]},
        {"id": 4, "cuisine": "indian", "ingredients": ["b", "c"]},
    ]
    nodes, _ = node_table(recipes, "indian", min_count=2)
    assert set(nodes["ingredient"]) == {"a", "b", "c"}
    edges = edge_table(recipes, ["a", "b", "c"])
    assert len(edges) == 3
    assert list(edges["co_count"]) == [2, 2, 2]
    # N=4, na=3, nb=3, co=2 -> P(a,b)=0.5, P(a)=0.75, P(b)=0.75 -> PMI = log2(0.5 / 0.5625) < 0
    assert (edges["PMI"] < 0).all()


# --------------------------------------------------------------------------- real Deliverable 1 artifacts
def test_d1_files_and_schemas(cfg, d1_dir):
    req_files = ["nodes.csv", "edges_indian.csv", "edges_italian.csv", "edges_mexican.csv", "merge_log.csv"]
    for fname in req_files:
        assert (d1_dir / fname).exists(), f"Missing {fname}"

    nodes = pd.read_csv(d1_dir / "nodes.csv")
    for col in ["ingredient", "cuisine", "recipe_count", "prevalence"]:
        assert col in nodes.columns
    assert set(nodes["cuisine"]) == set(cfg["cuisines"])
    assert (nodes["recipe_count"] >= cfg["network"]["min_ingredient_count"]).all()
    assert ((nodes["prevalence"] > 0.0) & (nodes["prevalence"] <= 100.0)).all()

    recipes, _ = load_processed(cfg)
    counts_by_cuisine = {c: sum(1 for r in recipes if r["cuisine"] == c) for c in cfg["cuisines"]}
    for _, row in nodes.iloc[::25].iterrows():
        n_c = counts_by_cuisine[row["cuisine"]]
        assert math.isclose(row["prevalence"], 100.0 * row["recipe_count"] / n_c, rel_tol=1e-9)


def test_d1_edges_valid_and_match_nodes(cfg, d1_dir):
    from collections import Counter
    from itertools import combinations

    nodes = pd.read_csv(d1_dir / "nodes.csv")
    recipes, _ = load_processed(cfg)
    for c in cfg["cuisines"]:
        df = pd.read_csv(d1_dir / f"edges_{c}.csv")
        for col in ["ingredient_a", "ingredient_b", "co_count", "PMI", "NPMI"]:
            assert col in df.columns
        assert len(df) > 0
        # Alphabetical order and no self-loops
        assert (df["ingredient_a"] < df["ingredient_b"]).all()
        # No duplicate unordered pairs
        assert df.duplicated(subset=["ingredient_a", "ingredient_b"]).sum() == 0
        # Endpoints exist in nodes.csv for the same cuisine
        c_nodes_df = nodes.loc[nodes["cuisine"] == c]
        c_nodes = set(c_nodes_df["ingredient"])
        assert set(df["ingredient_a"]) <= c_nodes
        assert set(df["ingredient_b"]) <= c_nodes
        # Finite PMI/NPMI and NPMI in [-1, 1] within numerical tolerance
        assert np.isfinite(df["PMI"]).all()
        assert np.isfinite(df["NPMI"]).all()
        assert ((df["NPMI"] >= -1.0 - 1e-9) & (df["NPMI"] <= 1.0 + 1e-9)).all()
        # Negative PMI/NPMI preserved (not filtered out)
        assert (df["NPMI"] < 0).any(), f"Expected negative NPMI edges to be preserved in {c}"
        # Loads cleanly into NetworkX without self-loops or duplicate edges
        G = nx.from_pandas_edgelist(df, "ingredient_a", "ingredient_b", ["co_count", "PMI", "NPMI"])
        assert nx.number_of_selfloops(G) == 0
        assert G.number_of_edges() == len(df)

        # Independently recompute all marginal counts, pairwise co-counts, PMI, and NPMI from recipes
        c_recipes = [r for r in recipes if r["cuisine"] == c]
        n_total = len(c_recipes)
        ing_counts = Counter()
        pair_counts = Counter()
        for r in c_recipes:
            el = sorted(x for x in set(r["ingredients"]) if x in c_nodes)
            ing_counts.update(el)
            pair_counts.update(combinations(el, 2))

        # Verify every node count matches nodes.csv
        for _, nrow in c_nodes_df.iterrows():
            assert int(nrow["recipe_count"]) == ing_counts[nrow["ingredient"]]

        # Verify exact edge completeness and exact co_count, PMI, and NPMI values across all rows
        assert len(df) == len(pair_counts), f"{c}: expected {len(pair_counts)} edges, got {len(df)}"
        a_col = df["ingredient_a"].tolist()
        b_col = df["ingredient_b"].tolist()
        co_expected = np.array([pair_counts[(a, b)] for a, b in zip(a_col, b_col)], dtype=np.int64)
        assert np.array_equal(df["co_count"].to_numpy(dtype=np.int64), co_expected)

        na_expected = np.array([ing_counts[a] for a in a_col], dtype=np.float64)
        nb_expected = np.array([ing_counts[b] for b in b_col], dtype=np.float64)
        pa = na_expected / n_total
        pb = nb_expected / n_total
        pab = co_expected.astype(np.float64) / n_total
        pmi_expected = np.log2(pab / (pa * pb))
        npmi_expected = np.where(pab >= 1.0, 1.0, pmi_expected / (-np.log2(pab)))
        assert np.allclose(df["PMI"].to_numpy(dtype=np.float64), pmi_expected, rtol=1e-12, atol=1e-12)
        assert np.allclose(df["NPMI"].to_numpy(dtype=np.float64), npmi_expected, rtol=1e-12, atol=1e-12)


def test_d1_merge_log_authentic(d1_dir):
    ml = pd.read_csv(d1_dir / "merge_log.csv")
    for col in ["original_name", "normalized_name", "rule"]:
        assert col in ml.columns
    assert len(ml) > 0
    # Every logged entry must actually change original_name and match normalize_ingredient(original_name)
    for _, row in ml.iterrows():
        assert row["original_name"] != row["normalized_name"]
        assert normalize_ingredient(row["original_name"]) == row["normalized_name"]
        assert isinstance(row["rule"], str) and len(row["rule"]) > 0
