"""Nicho por categoría y alertas de máscara de BG60k."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bg60k_filters", ROOT / "training" / "bg60k_filters.py")
bf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bf)

EXACT, KEYWORDS = bf.load_rules()


@pytest.mark.parametrize("category, tier", [
    ("Ceramic or Mug", "core"),
    ("scented candle", "core"),
    ("TV accessories", "core"),
    ("hair dryer", "core"),
    ("adult milk powder", "core"),
    ("desk lamp", "core"),
    ("running shoes", "near"),
    ("Liquor", "near"),
    ("gold ring", "near"),
    ("flat screen tv", "out"),
    ("Side table or coffee table", "out"),
    ("Men T-shirt", "out"),
    ("candied dried fruits", "out"),
    ("live cat", "out"),
    ("Tremella earrings", "out"),
    ("shower room", "out"),
    ("Outdoor sun protection clothing", "out"),
    ("instant food", "out"),
    ("Green potted plants", "out"),
    ("flowers", "out"),
    ("notebook", "unclassified"),
    ("something unknown", "unclassified"),
])
def test_classify(category, tier):
    assert bf.classify(category, EXACT, KEYWORDS)[0] == tier


def test_keywords_respect_word_boundaries():
    assert bf.classify("earring", EXACT, KEYWORDS) == ("near", "earring")
    assert bf.classify("children tableware", EXACT, KEYWORDS)[0] == "core"


def row(**overrides):
    base = {"status": "ok", "fg_frac": 0.3, "n_components": 1, "largest_frac": 1.0,
            "edges_touched": 0, "mask_rgb_err": 0.0}
    return base | overrides


@pytest.mark.parametrize("overrides, expected", [
    ({}, []),
    ({"fg_frac": 0.0, "n_components": 0, "largest_frac": 0.0}, ["empty_mask", "fragmented"]),
    ({"n_components": 2, "largest_frac": 0.6}, ["multi_object", "fragmented"]),
    ({"fg_frac": 0.03}, ["tiny_product"]),
    ({"fg_frac": 0.7}, ["large_product"]),
    ({"edges_touched": 1}, ["truncated"]),
    ({"mask_rgb_err": 4.0}, ["mask_mismatch"]),
    ({"status": "error: x"}, ["unreadable"]),
])
def test_alerts(overrides, expected):
    assert bf.alerts_for(row(**overrides)) == expected
