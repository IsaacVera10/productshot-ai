"""Split del piloto de BG60k: reparto por estilo, componentes y movimiento a train."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import bg60k_split as bs  # noqa: E402


def items(n: int, style: str = "studio", start: int = 0) -> list[dict]:
    return [{"id": (start + k) * 100, "style": style, "row": start + k, "phash_cutout": f"{(start + k) * 977:016x}"}
            for k in range(n)]


def test_assign_keeps_fractions_per_style_and_is_reproducible():
    rows = items(20) + items(10, "warm", start=20)
    split = bs.assign(rows, seed=0)
    assert split == bs.assign(rows, seed=0)
    studio = [split[r["id"]] for r in rows if r["style"] == "studio"]
    assert studio.count("val") == 2 and studio.count("test") == 2


def test_components_join_chains_of_confirmed_pairs():
    groups = bs.components([(1, 2), (2, 3), (7, 8)])
    assert sorted(map(sorted, groups)) == [[1, 2, 3], [7, 8]]


def test_resolve_moves_split_components_to_train_and_counts_unsure():
    split = {1: "val", 2: "train", 3: "test", 4: "test", 5: "val"}
    decided = {(1, 2): "same", (3, 4): "variant", (4, 5): "different", (2, 5): "unsure"}
    moved, _ = bs.resolve(split, decided)
    assert moved == {1: "train", 2: "train", 3: "test", 4: "test", 5: "train"}


def test_audit_pairs_only_cross_partitions():
    rows = items(4)
    emb = np.eye(4, dtype=np.float32)
    split = {0: "train", 100: "train", 200: "val", 300: "test"}
    pairs = bs.audit_pairs(rows, emb, split)
    assert (0, 100) not in pairs
    assert {(0, 200), (100, 200), (200, 300), (0, 300), (100, 300)} <= set(pairs)
