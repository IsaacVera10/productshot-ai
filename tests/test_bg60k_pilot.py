"""Selección en línea del piloto de BG60k con candidatos y embeddings sintéticos."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import bg60k_pilot as bp  # noqa: E402

CONFIG = {"target": 2, "near_max": 1, "per_family": 1, "radius": 0.9, "phash_radius": 7, "id_radius": 10,
          "order": ["studio", "warm"]}


def candidate(i: int, style: str, z: float, **extra) -> dict:
    """Ids separados por 100 y pHash aleatorios: sin conflictos salvo los que pida cada prueba."""
    phash = f"{np.random.default_rng(i).integers(0, 2**63):016x}"
    base = {"id": i * 100, "style_z": style, f"z_{style}": z, "niche": "core", "family": f"f{i}",
            "img_sha256": f"s{i}", "phash_cutout": phash, "category": f"c{i}"}
    return base | extra


def pilot(rows: list[dict], emb: np.ndarray) -> bp.Pilot:
    return bp.Pilot([r | {"row": k} for k, r in enumerate(rows)], emb, dict(CONFIG))


def orthogonal(n: int) -> np.ndarray:
    return np.eye(n, dtype=np.float32)


def run_all(p: bp.Pilot, reject: set[int] = frozenset()) -> list[tuple[str, int]]:
    seen = []
    while presented := p.present():
        style, r = presented
        seen.append((style, r["id"]))
        p.decide(r["id"], "reject" if r["id"] in reject else "accept")
    return seen


def test_turns_alternate_and_stop_at_target():
    rows = [candidate(i, s, 10 - i) for i, s in enumerate(["studio", "studio", "studio", "warm", "warm"])]
    p = pilot(rows, orthogonal(5))
    assert run_all(p) == [("studio", 0), ("warm", 300), ("studio", 100), ("warm", 400)]
    assert p.summary()["accepted"] == {"studio": 2, "warm": 2}


def test_rejection_is_replaced_from_the_same_queue():
    rows = [candidate(i, "studio", 10 - i) for i in range(3)]
    p = pilot(rows, orthogonal(3))
    assert [i for _, i in run_all(p, reject={0})] == [0, 100, 200]
    assert p.summary()["accepted"]["studio"] == 2


def test_identity_conflicts_and_caps_skip_candidates():
    emb = orthogonal(6)
    emb[1] = emb[0]
    rows = [
        candidate(0, "studio", 9),
        candidate(1, "studio", 8),
        candidate(2, "studio", 7, family="f0"),
        candidate(3, "studio", 6, niche="near"),
        candidate(4, "studio", 5, niche="near"),
        candidate(5, "warm", 9, category="c0", id=5),
    ]
    p = pilot(rows, emb)
    seen = run_all(p)
    assert seen == [("studio", 0), ("studio", 300)]
    reasons = {s["id"]: s["reason"] for s in p.skips}
    assert reasons == {5: "id_near", 100: "dino", 200: "family_cap"}


def test_near_cap_counts_accepted_and_pending():
    rows = [candidate(i, "studio", 9 - i, niche="near") for i in range(3)]
    p = pilot(rows, orthogonal(3))
    assert [i for _, i in run_all(p)] == [0]
    assert [s["reason"] for s in p.skips] == ["near_cap", "near_cap"]
    assert p.summary()["near"]["studio"] == 1


def test_rejected_candidates_do_not_block_others():
    emb = orthogonal(2)
    emb[1] = emb[0]
    rows = [candidate(0, "studio", 9), candidate(1, "studio", 8)]
    p = pilot(rows, emb)
    assert [i for _, i in run_all(p, reject={0})] == [0, 100]


def test_replay_reproduces_state_and_undo_reopens_the_last_decision(monkeypatch):
    monkeypatch.setattr(bp, "CONFIG", CONFIG)
    rows = [candidate(i, "studio", 10 - i) | {"row": i} for i in range(4)]
    emb = orthogonal(4)
    log = [{"id": "0", "label": "accept"}, {"id": "100", "label": "reject_style"}]
    p = bp.replay(rows, emb, log)
    assert p.status == {0: ("studio", "accepted"), 100: ("studio", "rejected")}
    assert p.present()[1]["id"] == 200
    undone = bp.replay(rows, emb, log + [{"id": "0", "label": "undo"}])
    assert undone.present()[1]["id"] == 100


def test_replay_detects_changed_configuration(monkeypatch):
    monkeypatch.setattr(bp, "CONFIG", CONFIG)
    rows = [candidate(i, "studio", 10 - i) | {"row": i} for i in range(2)]
    with pytest.raises(SystemExit):
        bp.replay(rows, orthogonal(2), [{"id": "100", "label": "accept"}])


def test_family_lookup_merges_categories_and_keywords():
    lookup = bp.family_lookup()
    assert bp.family({"category": "Game notebook", "niche_rule": "exact: laptop"}, lookup) == "laptop"
    assert bp.family({"category": "second hand notebook", "niche_rule": "exact: laptop"}, lookup) == "laptop"
    assert bp.family({"category": "cell phone", "niche_rule": "phone"}, lookup) == "phone"
    assert bp.family({"category": "Ceramic or Mug", "niche_rule": "exact: taza"}, lookup) == "cup"
    assert bp.family({"category": "keyboard", "niche_rule": "keyboard"}, lookup) == "keyboard"
