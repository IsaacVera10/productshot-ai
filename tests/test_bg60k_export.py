"""Captions del piloto de BG60k: preset del estilo más el producto."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "training"), str(root / "backend")]

import bg60k_export as be  # noqa: E402
from productshot.services.presets import PRESETS  # noqa: E402


def test_caption_starts_with_the_inference_preset():
    for style, preset in ((s.value, p) for s, p in PRESETS.items()):
        assert be.caption(style, "mug") == f"{preset.prompt}, mug"


def test_nouns_are_short_and_ids_override():
    categories, ids = be.noun_lookup()
    assert all(0 < len(n.split()) <= 4 for n in categories.values())
    assert ids[20] == "coffee mug"
