"""End-to-end orchestrator and acceptance validator for The Culinary Fingerprint.

Supported usage::

    python run_all.py --help
    python run_all.py
    python run_all.py --seed 42
    python run_all.py --skip-report
    python run_all.py --validate-only
    python run_all.py --n-null 1000 --no-cache
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from src import analysis, backbone, build_networks, evaluate, null_model, poster, preprocess, report, split_data, visualize
from src.common import (DataError, get_logger, is_debug_run, load_config, load_raw_dataset, print_dataset_report,
                        read_json, rpath, set_global_seed, timed, write_json)
from src.fingerprint import predict, predict_scores

log = get_logger()


def validate_deliverables(cfg: dict, skip_report: bool = False, run_pytest_suite: bool = True) -> dict:
    """Run automated PASS/FAIL acceptance checks across Deliverables 1-4 and pytest."""
    d1 = rpath(cfg["outputs"]["d1"])
    d2 = rpath(cfg["outputs"]["d2"])
    d3 = rpath(cfg["outputs"]["d3"])
    d4 = rpath(cfg["outputs"]["d4"])
    poster_path = rpath(cfg["outputs"]["poster"])
    cuisines = cfg["cuisines"]

    checks: dict[str, dict] = {}

    # --- Deliverable 1 ---
    d1_files = ["nodes.csv", "edges_indian.csv", "edges_italian.csv", "edges_mexican.csv", "merge_log.csv"]
    d1_missing = [f for f in d1_files if not (d1 / f).exists()]
    if d1_missing:
        checks["D1_dataset"] = {
            "deliverable": "D1: Ingredient Co-Occurrence Network Dataset",
            "required_files": [f"outputs/d1_dataset/{f}" for f in d1_files],
            "status": "FAIL",
            "evidence": f"Missing files: {d1_missing}",
        }
    else:
        nodes = pd.read_csv(d1 / "nodes.csv")
        ml = pd.read_csv(d1 / "merge_log.csv")
        edge_counts = {c: len(pd.read_csv(d1 / f"edges_{c}.csv")) for c in cuisines}
        ok = (
            set(["ingredient", "cuisine", "recipe_count", "prevalence"]) <= set(nodes.columns)
            and (nodes["recipe_count"] >= cfg["network"]["min_ingredient_count"]).all()
            and len(ml) > 0
            and all(v > 0 for v in edge_counts.values())
        )
        checks["D1_dataset"] = {
            "deliverable": "D1: Ingredient Co-Occurrence Network Dataset",
            "required_files": [f"outputs/d1_dataset/{f}" for f in d1_files],
            "status": "PASS" if ok else "FAIL",
            "evidence": f"nodes={len(nodes)} rows (min_count>=20); edges={edge_counts}; merge_log={len(ml)} rows.",
        }

    # --- Deliverable 2 ---
    d2_files = [
        "backbone_indian.csv", "backbone_italian.csv", "backbone_mexican.csv",
        "backbone_summary.csv", "graph_indian.html", "graph_italian.html", "graph_mexican.html",
    ]
    d2_missing = [f for f in d2_files if not (d2 / f).exists()]
    if d2_missing:
        checks["D2_backbone"] = {
            "deliverable": "D2: Statistically Validated Backbones & PyVis Graphs",
            "required_files": [f"outputs/d2_backbone/{f}" for f in d2_files],
            "status": "FAIL",
            "evidence": f"Missing files: {d2_missing}",
        }
    else:
        summ = pd.read_csv(d2 / "backbone_summary.csv")
        meta = read_json(d2 / "null_model_metadata.json") if (d2 / "null_model_metadata.json").exists() else {}
        bbs = {c: pd.read_csv(d2 / f"backbone_{c}.csv") for c in cuisines}
        all_z_ok = all((df["z_score"] > 3.0).all() and (df["co_count"] >= 5).all() for df in bbs.values())
        reduced_ok = bool((summ["edges_after"] < summ["edges_before"]).all())
        full_mode_ok = meta.get("run_mode") == "FULL" and int(meta.get("n_randomizations", 0)) == 1000
        ok = all_z_ok and reduced_ok and full_mode_ok
        checks["D2_backbone"] = {
            "deliverable": "D2: Statistically Validated Backbones & PyVis Graphs",
            "required_files": [f"outputs/d2_backbone/{f}" for f in d2_files],
            "status": "PASS" if ok else "FAIL",
            "evidence": (
                f"mode={meta.get('run_mode')} (n_null={meta.get('n_randomizations')}); "
                f"edges_after={ {c: len(bbs[c]) for c in cuisines} }; all z>3 & co>=5={all_z_ok}; "
                f"edges_after<edges_before={reduced_ok}."
            ),
        }

    # --- Deliverable 3 ---
    d3_files = [
        "hubs_indian.csv", "hubs_italian.csv", "hubs_mexican.csv",
        "universal_vs_specific_hubs.csv", "communities_indian.csv",
        "communities_italian.csv", "communities_mexican.csv", "bridges.csv",
    ]
    if not skip_report:
        d3_files = ["report.pdf", "report.md"] + d3_files
    d3_missing = [f for f in d3_files if not (d3 / f).exists()]
    if d3_missing:
        checks["D3_report"] = {
            "deliverable": "D3: Hub, Community, Bridge Analysis & Report",
            "required_files": [f"outputs/d3_report/{f}" for f in d3_files],
            "status": "FAIL",
            "evidence": f"Missing files: {d3_missing}",
        }
    else:
        pages = len(PdfReader(str(d3 / "report.pdf")).pages) if (d3 / "report.pdf").exists() else 0
        bridges = pd.read_csv(d3 / "bridges.csv")
        cls = pd.read_csv(d3 / "universal_vs_specific_hubs.csv")
        pages_ok = True if skip_report else (3 <= pages <= 4)
        ok = pages_ok and len(bridges) == 15 and len(cls) > 0
        checks["D3_report"] = {
            "deliverable": "D3: Hub, Community, Bridge Analysis & Report",
            "required_files": [f"outputs/d3_report/{f}" for f in d3_files],
            "status": "PASS" if ok else "FAIL",
            "evidence": f"report.pdf pages={pages} (req 3-4); distinct top-10 hubs={len(cls)}; bridges rows={len(bridges)}.",
        }

    # --- Deliverable 4 ---
    d4_files = [
        "results.csv", "confusion_matrix.png", "confusion_matrix_nb.png",
        "confusion_matrix_normalized.png", "confusion_matrix_normalized_nb.png",
        "evaluation_metadata.json",
    ]
    d4_missing = [f for f in d4_files if not (d4 / f).exists()]
    if d4_missing:
        checks["D4_scorer"] = {
            "deliverable": "D4: Fingerprint Scorer, Naive Bayes Baseline & Evaluation",
            "required_files": [f"outputs/d4_scorer/{f}" for f in d4_files],
            "status": "FAIL",
            "evidence": f"Missing files: {d4_missing}",
        }
    else:
        res_df = pd.read_csv(d4 / "results.csv").set_index("model")
        em = read_json(d4 / "evaluation_metadata.json")
        gm_pred = predict(["garam masala", "cumin", "turmeric"])
        gm_scores = predict_scores(["garam masala", "cumin", "turmeric"])
        full_mode_ok = em.get("run_mode") == "FULL" and int(em.get("n_randomizations", 0)) == 1000
        ok = (
            list(res_df.index) == ["Fingerprint Scorer", "Fingerprint (normalized ablation)", "Naive Bayes"]
            and (res_df["test_samples"] == 3456).all()
            and gm_pred == "indian"
            and full_mode_ok
        )
        checks["D4_scorer"] = {
            "deliverable": "D4: Fingerprint Scorer, Naive Bayes Baseline & Evaluation",
            "required_files": [f"outputs/d4_scorer/{f}" for f in d4_files],
            "status": "PASS" if ok else "FAIL",
            "evidence": (
                f"mode={em.get('run_mode')}; Fingerprint acc={res_df.loc['Fingerprint Scorer', 'accuracy']:.4f} "
                f"(macro_f1={res_df.loc['Fingerprint Scorer', 'macro_f1']:.4f}); "
                f"Ablation acc={res_df.loc['Fingerprint (normalized ablation)', 'accuracy']:.4f}; "
                f"Naive Bayes acc={res_df.loc['Naive Bayes', 'accuracy']:.4f}; "
                f"predict(['garam masala','cumin','turmeric'])={gm_pred!r} ({gm_scores})."
            ),
        }

    # --- Poster outline ---
    poster_ok = poster_path.exists() and poster_path.stat().st_size > 500
    checks["Poster_outline"] = {
        "deliverable": "Poster Outline",
        "required_files": ["outputs/poster_outline.md"],
        "status": "PASS" if poster_ok else "FAIL",
        "evidence": f"exists={poster_path.exists()}, size={poster_path.stat().st_size if poster_path.exists() else 0} bytes.",
    }

    # --- Pytest suite ---
    if run_pytest_suite:
        ret = int(pytest.main(["-q", "tests"]))
        checks["Pytest_suite"] = {
            "deliverable": "Automated Pytest Suite (tests/)",
            "required_files": [
                "tests/test_d1.py", "tests/test_d2.py", "tests/test_d3.py",
                "tests/test_fingerprint.py", "tests/test_evaluation.py", "tests/test_no_leakage.py",
            ],
            "status": "PASS" if ret == 0 else "FAIL",
            "evidence": f"pytest exit code={ret}",
        }

    overall_pass = all(c["status"] == "PASS" for c in checks.values())
    return {
        "overall_status": "PASS" if overall_pass else "FAIL",
        "run_mode": "DEBUG" if is_debug_run(cfg) else "FULL",
        "checks": checks,
    }


def run_pipeline(cfg: dict, skip_report: bool = False, validate_only: bool = False, use_cache: bool = True) -> dict:
    set_global_seed(int(cfg["seed"]))
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    if not validate_only:
        with timed("phase1_validate_and_preprocess", timings, log):
            records, ds_report = load_raw_dataset(cfg)
            print_dataset_report(ds_report, log)
            preprocess.run(cfg)
            split_data.run(cfg)

        with timed("phase2_deliverable1_networks", timings, log):
            recipes, dataset_sha = preprocess.load_processed(cfg)
            nets = build_networks.run(cfg, recipes)
            log.info("CHECKPOINT 1 (Deliverable 1): PASS — nodes and raw PMI/NPMI edge lists written.")

        with timed("phase3_null_model_checks_and_benchmark", timings, log):
            inputs = backbone.incidence_inputs(recipes, cfg, nets)
            chk = {c: null_model.check_randomization(B, 5 * B.shape[0], cfg["seed"]) for c, (B, _, _) in inputs.items()}
            if not all(r["passed"] for r in chk.values()):
                raise RuntimeError(f"Null-model degree-preservation check failed: {chk}")
            bench = null_model.benchmark(inputs, cfg)
            d2 = rpath(cfg["outputs"]["d2"])
            d2.mkdir(parents=True, exist_ok=True)
            write_json(chk, d2 / "null_model_checks.json")
            write_json(bench, d2 / "null_benchmark.json")
            log.info("CHECKPOINT 2 (Null-Model Benchmark): PASS — degree preservation verified; "
                     "estimated 1,000-sample runtime=%.2fs across all cuisines.",
                     bench["estimated_total_seconds_all_cuisines"])

        with timed("phase4_deliverable2_backbones_and_graphs", timings, log):
            backbone.run(cfg, recipes=recipes, nets=nets, use_cache=use_cache, dataset_sha=dataset_sha)
            visualize.run(cfg)

        with timed("phase5_deliverable3_analysis_and_report", timings, log):
            analysis.run(cfg)
            if not skip_report:
                report.run(cfg)

        with timed("phase6_deliverable4_evaluation", timings, log):
            evaluate.run(cfg, use_cache=use_cache)

        with timed("phase7_poster_outline", timings, log):
            poster.generate_poster_outline(cfg)

    with timed("phase8_acceptance_validation", timings, log):
        summary = validate_deliverables(cfg, skip_report=skip_report, run_pytest_suite=True)

    timings["total_wall_seconds"] = round(time.perf_counter() - t0, 3)
    val_path = rpath(cfg["outputs"]["validation"])
    prev = read_json(val_path) if val_path.exists() else {}

    artifact_runtimes: dict[str, object] = {}
    d2_meta_path = rpath(cfg["outputs"]["d2"]) / "null_model_metadata.json"
    if d2_meta_path.exists():
        d2_meta = read_json(d2_meta_path)
        chain_sec = {
            c: d2_meta.get("per_cuisine", {}).get(c, {}).get("chain_seconds")
            for c in cfg["cuisines"]
        }
        if all(isinstance(v, (int, float)) for v in chain_sec.values()):
            chain_sec["total_chain_seconds"] = round(sum(chain_sec.values()), 3)
        artifact_runtimes["d2_descriptive_null_chains_seconds"] = chain_sec

    d4_train_meta_path = rpath(cfg["outputs"]["d4"]) / "training_artifacts" / "training_metadata.json"
    if d4_train_meta_path.exists():
        d4_train_meta = read_json(d4_train_meta_path)
        pred_chain_sec = {
            c: d4_train_meta.get("null_model_diagnostics", {}).get(c, {}).get("chain_seconds")
            for c in cfg["cuisines"]
        }
        if all(isinstance(v, (int, float)) for v in pred_chain_sec.values()):
            pred_chain_sec["total_chain_seconds"] = round(sum(pred_chain_sec.values()), 3)
        artifact_runtimes["d4_predictive_null_chains_seconds"] = pred_chain_sec

    d4_meta_path = rpath(cfg["outputs"]["d4"]) / "evaluation_metadata.json"
    if d4_meta_path.exists():
        d4_meta = read_json(d4_meta_path)
        if "phase_runtimes_seconds" in d4_meta:
            artifact_runtimes["d4_evaluation_phases_seconds"] = d4_meta["phase_runtimes_seconds"]

    summary["invocation_mode"] = "validate_only" if validate_only else "full_pipeline"
    if not validate_only:
        summary["full_pipeline_runtimes_seconds"] = timings
        summary["validate_only_runtimes_seconds"] = prev.get("validate_only_runtimes_seconds")
        summary["phase_runtimes_seconds"] = timings
    else:
        prev_phases = prev.get("phase_runtimes_seconds", {})
        prev_full = prev.get("full_pipeline_runtimes_seconds") or (
            prev_phases if "phase1_validate_and_preprocess" in prev_phases else None
        )
        summary["full_pipeline_runtimes_seconds"] = prev_full
        summary["validate_only_runtimes_seconds"] = timings
        summary["phase_runtimes_seconds"] = prev_full if prev_full is not None else timings

    if artifact_runtimes:
        summary["persisted_artifact_runtimes_seconds"] = artifact_runtimes

    write_json(summary, val_path)
    log.info("Validation summary (%s, %s): %s -> %s", summary["run_mode"], summary["invocation_mode"], summary["overall_status"], val_path)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="End-to-end pipeline and acceptance validator for The Culinary Fingerprint."
    )
    ap.add_argument("--config", default=None, help="Path to config.yaml (default: repo config.yaml)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed override (default: 42)")
    ap.add_argument("--skip-report", action="store_true", help="Skip generating report.md and report.pdf")
    ap.add_argument("--validate-only", action="store_true", help="Run acceptance checks on existing outputs only")
    ap.add_argument("--n-null", type=int, default=None, help="DEBUG ONLY: override number of null samples (default 1000)")
    ap.add_argument("--no-cache", action="store_true", help="Recompute null-model chains without using cache")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, {"seed": args.seed, "null_model.n_randomizations": args.n_null})
    try:
        summary = run_pipeline(
            cfg,
            skip_report=args.skip_report,
            validate_only=args.validate_only,
            use_cache=not args.no_cache,
        )
    except DataError as exc:
        log.error(str(exc))
        return 2
    return 0 if summary["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
