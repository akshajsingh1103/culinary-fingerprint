"""Deliverable 2 acceptance tests (null model, backbone CSVs, summary, PyVis HTML).

Reads saved artifacts and runs fast unit tests on tiny matrices; never reruns
the 1,000-randomisation null model.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.backbone import filter_backbone, null_stats_table
from src.common import read_json
from src.graph_utils import backbone_graph, louvain, modularity
from src.null_model import check_randomization, curveball_randomize, run_null


# --------------------------------------------------------------------------- unit tests (tiny matrices)
def test_curveball_preserves_degrees_binarity_and_seed_reproducibility():
    rng = np.random.default_rng(42)
    mat = (rng.random((40, 25)) < 0.30).astype(np.int32)
    B = sp.csr_matrix(mat)
    chk = check_randomization(B, n_trades=400, seed=42)
    assert chk["row_degrees_preserved"]
    assert chk["col_degrees_preserved"]
    assert chk["binary"]
    assert chk["reproducible_same_seed"]
    assert chk["different_seed_differs"]

    # Repeated calls with same seed are bit-for-bit identical
    R1 = curveball_randomize(B, 250, 42)
    R2 = curveball_randomize(B, 250, 42)
    assert (R1 != R2).nnz == 0


def test_zero_variance_excluded_and_strict_z_threshold():
    # Construct a synthetic stats table containing:
    # - edge with z_score == 3.0 exactly (must NOT survive strict z > 3)
    # - edge with zero_variance == True and co_count == 10 (must NOT survive)
    # - edge with z_score == 3.01 and co_count == 4 (must NOT survive co_count >= 5)
    # - edge with z_score == 3.01 and co_count == 5 and negative NPMI (MUST survive)
    stats = pd.DataFrame([
        {"ingredient_a": "a", "ingredient_b": "b", "co_count": 10, "expected_mean": 4.0,
         "expected_std": 2.0, "z_score": 3.0, "NPMI": 0.25, "zero_variance": False},
        {"ingredient_a": "a", "ingredient_b": "c", "co_count": 10, "expected_mean": 10.0,
         "expected_std": 0.0, "z_score": np.nan, "NPMI": 0.40, "zero_variance": True},
        {"ingredient_a": "b", "ingredient_b": "c", "co_count": 4, "expected_mean": 1.0,
         "expected_std": 0.5, "z_score": 6.0, "NPMI": 0.30, "zero_variance": False},
        {"ingredient_a": "c", "ingredient_b": "d", "co_count": 5, "expected_mean": 1.0,
         "expected_std": 1.0, "z_score": 4.0, "NPMI": -0.05, "zero_variance": False},
    ])
    bb = filter_backbone(stats, z_threshold=3.0, min_co=5)
    assert len(bb) == 1
    assert bb.iloc[0]["ingredient_a"] == "c" and bb.iloc[0]["ingredient_b"] == "d"
    assert bb.iloc[0]["NPMI"] == -0.05


def test_zero_variance_detected_from_degenerate_matrix():
    # If two ingredients appear in every recipe, their co-occurrence is invariant across all trades
    B = sp.csr_matrix(np.array([
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
    ], dtype=np.int32))
    res = run_null(B, np.array([0, 1, 2, 3]), n_samples=15, burn_in=20, thin=5, seed=42)
    edges = pd.DataFrame([{"ingredient_a": "i0", "ingredient_b": "i1", "co_count": 4, "PMI": 0.0, "NPMI": 1.0}])
    st = null_stats_table(edges, res, ["i0", "i1", "i2", "i3"])
    assert bool(st.iloc[0]["zero_variance"]) is True
    assert np.isnan(st.iloc[0]["z_score"])
    bb = filter_backbone(st, z_threshold=3.0, min_co=1)
    assert len(bb) == 0


# --------------------------------------------------------------------------- real Deliverable 2 artifacts
def test_d2_files_and_backbone_constraints(cfg, d1_dir, d2_dir):
    for c in cfg["cuisines"]:
        assert (d2_dir / f"backbone_{c}.csv").exists()
        assert (d2_dir / f"null_stats_{c}.csv").exists()
        assert (d2_dir / f"graph_{c}.html").exists()
    assert (d2_dir / "backbone_summary.csv").exists()
    assert (d2_dir / "null_model_metadata.json").exists()

    meta = read_json(d2_dir / "null_model_metadata.json")
    assert meta["n_randomizations"] == 1000
    assert meta["run_mode"] == "FULL"

    req_cols = ["ingredient_a", "ingredient_b", "co_count", "expected_mean", "expected_std", "z_score", "NPMI"]
    z_thresh = float(cfg["backbone"]["z_threshold"])
    min_co = int(cfg["backbone"]["min_co_count"])

    for c in cfg["cuisines"]:
        raw_edges = pd.read_csv(d1_dir / f"edges_{c}.csv")
        null_st = pd.read_csv(d2_dir / f"null_stats_{c}.csv")
        bb = pd.read_csv(d2_dir / f"backbone_{c}.csv")
        for col in req_cols:
            assert col in bb.columns
            assert col in null_st.columns
        assert "zero_variance" in null_st.columns

        # Verify null_stats matches raw edges 1-to-1
        assert len(null_st) == len(raw_edges)
        assert list(null_st["ingredient_a"]) == list(raw_edges["ingredient_a"])
        assert list(null_st["ingredient_b"]) == list(raw_edges["ingredient_b"])
        assert np.array_equal(null_st["co_count"].to_numpy(np.int64), raw_edges["co_count"].to_numpy(np.int64))
        assert np.allclose(null_st["NPMI"].to_numpy(float), raw_edges["NPMI"].to_numpy(float), rtol=1e-12, atol=1e-12)

        # Verify zero_variance flag and z-score formula across all observed pairs
        std = null_st["expected_std"].to_numpy(float)
        zv = null_st["zero_variance"].to_numpy(bool)
        assert np.array_equal(zv, ~(std > 0.0))
        if np.any(zv):
            assert np.isnan(null_st.loc[zv, "z_score"]).all()
        non_zv = ~zv
        expected_z = (
            null_st.loc[non_zv, "co_count"].to_numpy(float) - null_st.loc[non_zv, "expected_mean"].to_numpy(float)
        ) / null_st.loc[non_zv, "expected_std"].to_numpy(float)
        assert np.allclose(null_st.loc[non_zv, "z_score"].to_numpy(float), expected_z, rtol=1e-12, atol=1e-12)

        # Verify exact completeness of backbone_<cuisine>.csv against null_stats_<cuisine>.csv
        qualifying_mask = (
            (~zv)
            & np.isfinite(null_st["z_score"].to_numpy(float))
            & (null_st["z_score"].to_numpy(float) > z_thresh)
            & (null_st["co_count"].to_numpy(np.int64) >= min_co)
        )
        expected_bb = (
            null_st.loc[qualifying_mask, req_cols]
            .sort_values(["ingredient_a", "ingredient_b"])
            .reset_index(drop=True)
        )
        assert len(bb) == len(expected_bb) and len(bb) > 0
        assert list(bb["ingredient_a"]) == list(expected_bb["ingredient_a"])
        assert list(bb["ingredient_b"]) == list(expected_bb["ingredient_b"])
        for col in ["co_count", "expected_mean", "expected_std", "z_score", "NPMI"]:
            assert np.isfinite(bb[col]).all()
            assert np.allclose(bb[col].to_numpy(float), expected_bb[col].to_numpy(float), rtol=1e-12, atol=1e-12)
        assert (bb["ingredient_a"] < bb["ingredient_b"]).all()
        assert bb.duplicated(subset=["ingredient_a", "ingredient_b"]).sum() == 0
        assert (bb["z_score"] > z_thresh).all(), f"{c}: found z_score <= {z_thresh}"
        assert (bb["co_count"] >= min_co).all(), f"{c}: found co_count < {min_co}"
        assert (bb["expected_std"] > 0.0).all(), f"{c}: zero-variance edge leaked into backbone"


def test_d2_summary_matches_files_and_reduces_edges(cfg, d1_dir, d2_dir):
    summary = pd.read_csv(d2_dir / "backbone_summary.csv")
    for col in ["cuisine", "nodes", "edges_before", "edges_after", "pct_retained", "n_communities", "modularity"]:
        assert col in summary.columns
    nodes_all = pd.read_csv(d1_dir / "nodes.csv")
    for _, row in summary.iterrows():
        c = row["cuisine"]
        raw = pd.read_csv(d1_dir / f"edges_{c}.csv")
        bb = pd.read_csv(d2_dir / f"backbone_{c}.csv")
        assert int(row["edges_before"]) == len(raw)
        assert int(row["edges_after"]) == len(bb)
        assert int(row["edges_after"]) < int(row["edges_before"])
        assert math.isclose(row["pct_retained"], 100.0 * len(bb) / len(raw), rel_tol=1e-9)

        G = backbone_graph(bb, nodes_all[nodes_all["cuisine"] == c])
        assert int(row["nodes"]) == G.number_of_nodes()
        part = louvain(G, int(cfg["community"]["seed"]), float(cfg["community"]["resolution"]))
        assert int(row["n_communities"]) == len(set(part.values()))
        assert math.isclose(row["modularity"], modularity(G, part, float(cfg["community"]["resolution"])), rel_tol=1e-6)


def test_d2_html_visualizations_nonempty_and_self_contained(cfg, d2_dir):
    for c in cfg["cuisines"]:
        html_path = d2_dir / f"graph_{c}.html"
        content = html_path.read_text(encoding="utf-8")
        assert len(content) > 10_000
        assert "vis.Network" in content or "vis-network" in content
        assert "nodes = new vis.DataSet" in content
        assert "edges = new vis.DataSet" in content
        assert c.title() in content
