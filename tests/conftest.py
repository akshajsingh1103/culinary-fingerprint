"""Shared pytest fixtures for Culinary Fingerprint tests.

Tests inspect the saved real artifacts when present and use small deterministic
fixtures for unit-level mathematical/algorithmic checks. Tests never rerun the
1,000-randomisation null model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_config, rpath


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def d1_dir(cfg) -> Path:
    return rpath(cfg["outputs"]["d1"])


@pytest.fixture(scope="session")
def d2_dir(cfg) -> Path:
    return rpath(cfg["outputs"]["d2"])


@pytest.fixture(scope="session")
def d3_dir(cfg) -> Path:
    return rpath(cfg["outputs"]["d3"])


@pytest.fixture(scope="session")
def d4_dir(cfg) -> Path:
    return rpath(cfg["outputs"]["d4"])
