"""Dedicated data-leakage prevention tests for Deliverable 4."""
from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from src.build_networks import assert_training_only, build_predictive_networks
from src.common import read_json
from src.evaluate import BinaryMultinomialNB
from src.preprocess import load_processed
from src.split_data import load_split


def test_split_disjointness_and_coverage(cfg):
    recipes, _ = load_processed(cfg)
    split = load_split(cfg)
    train_ids = set(split["train_ids"])
    test_ids = set(split["test_ids"])
    all_ids = {r["id"] for r in recipes}

    assert len(train_ids & test_ids) == 0, "Train and test recipe IDs must be strictly disjoint"
    assert train_ids | test_ids == all_ids, "Train and test IDs must partition all selected recipes"
    assert len(train_ids) == 13823 and len(test_ids) == 3456


def test_leakage_guards_reject_test_recipes(cfg):
    recipes, _ = load_processed(cfg)
    split = load_split(cfg)
    train_ids = set(split["train_ids"])
    test_ids = set(split["test_ids"])
    train_recipes = [r for r in recipes if r["id"] in train_ids]
    test_recipes = [r for r in recipes if r["id"] in test_ids]

    # Passing even ONE test recipe to assert_training_only / build_predictive_networks / BinaryMultinomialNB.fit must fail
    contaminated = train_recipes[:50] + [test_recipes[0]]
    with pytest.raises(AssertionError, match="test recipes passed"):
        assert_training_only(contaminated, train_ids, test_ids)

    with pytest.raises(AssertionError, match="test recipes passed"):
        build_predictive_networks(contaminated, cfg["cuisines"], 20, train_ids, test_ids)

    with pytest.raises(AssertionError, match="test recipes passed"):
        BinaryMultinomialNB(cfg["cuisines"]).fit(contaminated, train_ids, test_ids)


def test_predictive_artifacts_derived_strictly_from_training_split(cfg, d1_dir, d2_dir, d4_dir):
    recipes, _ = load_processed(cfg)
    split = load_split(cfg)
    train_ids = set(split["train_ids"])
    train_recipes = [r for r in recipes if r["id"] in train_ids]

    art_dir = d4_dir / "training_artifacts"
    train_meta = read_json(art_dir / "training_metadata.json")
    assert train_meta["train_ids_sha256"] == split["train_ids_sha256"]
    assert train_meta["n_train_recipes"] == len(train_recipes)

    # Verify nodes_train.csv counts match exact counts on train_recipes only, and differ from full-dataset nodes.csv
    nodes_train = pd.read_csv(art_dir / "nodes_train.csv")
    nodes_full = pd.read_csv(d1_dir / "nodes.csv")
    assert len(nodes_train) < len(nodes_full), "Training eligible node table must be smaller than full-dataset table"

    for c in cfg["cuisines"]:
        c_train = [r for r in train_recipes if r["cuisine"] == c]
        cnt = Counter(x for r in c_train for x in set(r["ingredients"]))
        sub = nodes_train[nodes_train["cuisine"] == c]
        for _, row in sub.iterrows():
            assert int(row["recipe_count"]) == cnt[row["ingredient"]]

        # Verify predictive backbone is distinct from descriptive full-dataset backbone
        bb_train = pd.read_csv(art_dir / f"backbone_{c}.csv")
        bb_full = pd.read_csv(d2_dir / f"backbone_{c}.csv")
        assert len(bb_train) != len(bb_full)
