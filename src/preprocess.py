"""Load, validate and normalise the three target cuisines; write the merge log.

Outputs
-------
* ``data/processed/recipes_normalized.json`` – id, cuisine, normalised & deduplicated ingredients.
* ``outputs/d1_dataset/merge_log.csv``       – every raw name changed by normalisation.
* ``outputs/d1_dataset/normalization_report.json`` – stage counts, affected recipes, examples.

Usage::

    python src/preprocess.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common import (DataError, get_logger, load_config, load_raw_dataset, print_dataset_report, rpath,
                        write_json)
from src.normalize import FORMAT_STAGES, PLURAL_WHITELIST, STAGES, SYNONYMS, normalize_with_trace

log = get_logger()


def select_and_normalize(records: list[dict], cuisines: list[str]) -> tuple[list[dict], pd.DataFrame, dict]:
    """Keep target cuisines, normalise ingredient names and compute an audit report.

    Returns ``(recipes, merge_log_df, report)``. ``recipes`` are sorted by id for
    determinism; each has keys ``id``, ``cuisine``, ``ingredients`` (sorted, unique).
    """
    selected = sorted((r for r in records if r.get("cuisine") in cuisines), key=lambda r: r["id"])
    raw_counter: Counter = Counter(x for r in selected for x in r["ingredients"])
    trace = {raw: normalize_with_trace(raw) for raw in sorted(raw_counter)}

    # distinct names after each cumulative stage
    stage_names = [s for s, _ in STAGES]
    distinct = {"raw": len(raw_counter)}
    for s in stage_names:
        distinct[f"after_{s}"] = len({t[2][s] for t in trace.values()})
    distinct["after_format_cleanup(lowercase/whitespace/unicode)"] = distinct[f"after_{FORMAT_STAGES[-1]}"]
    distinct["final"] = distinct[f"after_{stage_names[-1]}"]

    # recipes affected per rule and duplicate removal
    recipes_affected = Counter()
    dedup_recipes = 0
    dedup_removed = 0
    out: list[dict] = []
    for r in selected:
        rules_here = set()
        finals = []
        for x in r["ingredients"]:
            final, rules, _ = trace[x]
            rules_here.update(rules)
            finals.append(final)
        for rule in rules_here:
            recipes_affected[rule] += 1
        uniq = sorted(set(f for f in finals if f))
        if len(uniq) < len(finals):
            dedup_recipes += 1
            dedup_removed += len(finals) - len(uniq)
        out.append({"id": int(r["id"]), "cuisine": r["cuisine"], "ingredients": uniq})

    # merge log: every raw name whose final form differs
    final_groups: dict[str, list[str]] = defaultdict(list)
    for raw, (final, _, _) in trace.items():
        final_groups[final].append(raw)
    rows = []
    for raw, (final, rules, _) in trace.items():
        if final != raw:
            rows.append({"original_name": raw, "normalized_name": final, "rule": "+".join(rules),
                         "occurrences": raw_counter[raw], "group_size": len(final_groups[final])})
    merge_log = pd.DataFrame(rows, columns=["original_name", "normalized_name", "rule", "occurrences", "group_size"])
    merge_log = merge_log.sort_values(["normalized_name", "original_name"]).reset_index(drop=True)

    names_changed = Counter()
    for raw, (_, rules, _) in trace.items():
        for rule in rules:
            names_changed[rule] += 1
    merged_groups = {k: sorted(v) for k, v in final_groups.items() if len(v) > 1}
    examples = defaultdict(list)
    for raw, (final, rules, _) in trace.items():
        for rule in rules:
            if len(examples[rule]) < 6:
                examples[rule].append(f"{raw} -> {final}")
    used_syn = {k: {"target": v[0], "rationale": v[1],
                    "raw_names_mapped": sorted(raw for raw, t in trace.items() if t[2]["singularize"] == k)}
                for k, v in SYNONYMS.items()}
    report = {
        "recipes_selected": len(out),
        "cuisine_counts": dict(Counter(r["cuisine"] for r in out)),
        "ingredient_occurrences_raw": int(sum(raw_counter.values())),
        "distinct_names": distinct,
        "names_changed_per_rule": dict(names_changed),
        "recipes_affected_per_rule": dict(recipes_affected),
        "recipes_with_duplicates_after_normalization": dedup_recipes,
        "duplicate_occurrences_removed": dedup_removed,
        "merged_groups_count": len(merged_groups),
        "raw_names_in_merged_groups": int(sum(len(v) for v in merged_groups.values())),
        "merged_groups_examples": dict(sorted(merged_groups.items(), key=lambda kv: -len(kv[1]))[:25]),
        "rule_examples": dict(examples),
        "synonym_dictionary": used_syn,
        "plural_whitelist_size": len(PLURAL_WHITELIST),
        "merge_log_rows": len(merge_log),
    }
    return out, merge_log, report


def audit_full_dataset(records: list[dict]) -> dict:
    """Rerun the preliminary audit across all recipes in train.json."""
    import re
    raw_names = [x for r in records for x in r.get("ingredients", []) if isinstance(x, str)]
    unique_raw = set(raw_names)
    basic = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    groups: dict[str, set[str]] = defaultdict(set)
    for x in unique_raw:
        groups[basic(x)].add(x)
    collisions = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    dup_recipes = 0
    for r in records:
        cleaned = [basic(x) for x in r.get("ingredients", []) if isinstance(x, str)]
        if len(cleaned) != len(set(cleaned)):
            dup_recipes += 1
    return {
        "total_recipes": len(records),
        "total_ingredient_occurrences": len(raw_names),
        "distinct_raw_strings": len(unique_raw),
        "distinct_after_basic_lowercase_whitespace": len(groups),
        "uppercase_or_mixed_case_names": sum(x != x.lower() for x in unique_raw),
        "leading_or_trailing_whitespace_names": sum(x != x.strip() for x in unique_raw),
        "repeated_internal_whitespace_names": sum(bool(re.search(r"\s{2,}", x)) for x in unique_raw),
        "formatting_equivalent_groups_count": len(collisions),
        "formatting_equivalent_groups": collisions,
        "recipes_with_duplicates_after_basic_cleanup": dup_recipes,
    }


def run(cfg: dict) -> dict:
    """Execute preprocessing and write outputs; return the report."""
    records, ds_report = load_raw_dataset(cfg)
    print_dataset_report(ds_report, log)
    full_audit = audit_full_dataset(records)
    log.info("Full-dataset audit (all 20 cuisines): recipes=%d occurrences=%d raw_distinct=%d "
             "basic_cleanup_distinct=%d uppercase=%d internal_ws=%d eq_groups=%d dup_recipes=%d",
             full_audit["total_recipes"], full_audit["total_ingredient_occurrences"],
             full_audit["distinct_raw_strings"], full_audit["distinct_after_basic_lowercase_whitespace"],
             full_audit["uppercase_or_mixed_case_names"], full_audit["repeated_internal_whitespace_names"],
             full_audit["formatting_equivalent_groups_count"],
             full_audit["recipes_with_duplicates_after_basic_cleanup"])
    recipes, merge_log, report = select_and_normalize(records, cfg["cuisines"])
    report["dataset"] = ds_report
    report["full_dataset_audit"] = full_audit
    proc = rpath(cfg["data"]["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    write_json({"dataset_sha256": ds_report["sha256"], "recipes": recipes}, proc / "recipes_normalized.json")
    d1 = rpath(cfg["outputs"]["d1"])
    d1.mkdir(parents=True, exist_ok=True)
    merge_log.to_csv(d1 / "merge_log.csv", index=False, encoding="utf-8")
    write_json(report, d1 / "normalization_report.json")

    d = report["distinct_names"]
    log.info("Target cuisines (Indian/Italian/Mexican) distinct names: raw=%d | after unicode=%d | lowercase=%d | whitespace=%d | singularize=%d | final=%d",
             d["raw"], d["after_unicode"], d["after_lowercase"], d["after_whitespace"], d["after_singularize"],
             d["final"])
    for rule in [s for s, _ in STAGES]:
        log.info("  rule %-11s names changed=%4d recipes affected=%5d e.g. %s", rule,
                 report["names_changed_per_rule"].get(rule, 0), report["recipes_affected_per_rule"].get(rule, 0),
                 report["rule_examples"].get(rule, [])[:2])
    log.info("Merged groups (>=2 raw names -> one final): %d; recipes with duplicates after normalisation: %d",
             report["merged_groups_count"], report["recipes_with_duplicates_after_normalization"])
    log.info("merge_log.csv rows: %d", len(merge_log))
    return report


def load_processed(cfg: dict) -> tuple[list[dict], str]:
    """Load normalised recipes produced by :func:`run`; returns (recipes, dataset_sha256)."""
    p = rpath(cfg["data"]["processed_dir"]) / "recipes_normalized.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `python src/preprocess.py` first.")
    from src.common import read_json
    obj = read_json(p)
    return obj["recipes"], obj["dataset_sha256"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate the Kaggle dataset, keep Indian/Italian/Mexican, "
                                             "normalise ingredient names and write merge_log.csv.")
    ap.add_argument("--config", default=None, help="Path to config.yaml (default: repo config.yaml)")
    args = ap.parse_args(argv)
    try:
        run(load_config(args.config))
    except DataError as exc:
        log.error(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
