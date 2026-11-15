"""Puntajes de estilo y vecinos de identidad de BG60k con embeddings sintéticos."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import bg60k_identity as identity  # noqa: E402
import bg60k_style as style  # noqa: E402


def test_prototype_is_renormalized_mean_of_its_group():
    text = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    names, protos = style.prototypes(text, ["style:a", "style:a", "other:b"])
    assert names == ["style:a", "other:b"]
    assert np.allclose(protos[0], [2 ** -0.5, 2 ** -0.5, 0])
    assert np.allclose(np.linalg.norm(protos, axis=1), 1)


def test_score_picks_best_style_margin_and_standardized_style(tmp_path):
    groups = [f"style:{s}" for s in style.STYLES] + ["other:black"]
    text = np.eye(6, dtype=np.float32)
    images = np.array([[0.9, 0.1, 0, 0, 0, 0], [0.5, 0, 0, 0, 0.6, 0], [0.8, 0, 0.7, 0, 0, 0]], dtype=np.float32)
    np.savez(tmp_path / "clip.npz", ids=np.array([7, 8, 9]), clip_image=images, clip_text=text,
             prompt_group=np.array(groups))
    rows = {r["id"]: r for r in style.score(tmp_path / "clip.npz")}
    assert rows[7]["style"] == "studio" and rows[7]["second"] == "natural"
    assert rows[8]["style"] == "minimal" and rows[8]["margin"] > 0
    assert rows[9]["style_z"] == "warm"  # studio alto en todas: el z-score lo descuenta


def test_nearest_skips_itself_and_orders_by_cosine():
    emb = np.array([[1, 0], [0.99, 0.14], [0, 1], [0.1, 0.99]], dtype=np.float32)
    idx, sim = identity.nearest(emb, k=2)
    assert idx[0, 0] == 1 and idx[2, 0] == 3
    assert all(idx[i, 0] != i for i in range(4))
    assert (sim[:, 0] >= sim[:, 1]).all()


def test_neighbor_rows_include_phash_distance():
    rows = identity.neighbor_rows(np.array([10, 20]), np.array([[1], [0]]), np.array([[0.9], [0.9]]),
                                  {10: "ff00", 20: "ff01"})
    assert rows[0] == {"id": 10, "neighbor": 20, "rank": 0, "cosine": pytest.approx(0.9), "phash_distance": 1}
