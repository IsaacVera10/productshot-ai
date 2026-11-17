"""Comparación entre evaluaciones y revisión ciega."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import lora_compare as lc  # noqa: E402


def write_eval(folder: Path, config: str, margin: float) -> None:
    (folder / "images").mkdir(parents=True)
    rows = []
    for item in (1, 2, 3):
        for style in ("studio", "warm"):
            name = f"{item}_{style}_{config}.jpg"
            Image.new("RGB", (8, 8)).save(folder / "images" / name)
            rows.append({"id": item, "style": style, "config": config, "clip_margin": margin + item / 100,
                         "dino_product": 0.95, "image": name})
    (folder / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_compare_pairs_by_product_and_style(tmp_path):
    write_eval(tmp_path / "a", "ckpt400@1", 0.0)
    write_eval(tmp_path / "b", "ckpt500@1", 0.01)
    _, _, a = lc.load(f"{tmp_path / 'a'}:ckpt400@1")
    _, _, b = lc.load(f"{tmp_path / 'b'}:ckpt500@1")
    result = lc.compare(a, b)
    assert result["products"] == 3 and result["delta"] == pytest.approx(0.01)
    assert set(result["by_style"]) == {"studio", "warm"}


def test_blind_hides_which_is_which(tmp_path):
    write_eval(tmp_path / "a", "x", 0.0)
    write_eval(tmp_path / "b", "y", 0.0)
    a_dir, _, a = lc.load(f"{tmp_path / 'a'}:x")
    b_dir, _, b = lc.load(f"{tmp_path / 'b'}:y")
    lc.blind(a_dir, a, b_dir, b, tmp_path / "ciega")
    key = json.loads((tmp_path / "ciega" / "clave.json").read_text())
    assert len(key) == 3 and all({k["1"], k["2"]} == {"A", "B"} for k in key)
    assert len(list((tmp_path / "ciega").glob("*.jpg"))) == 6
