"""Selección de candidatas para el LoRA por autodestilación."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "training"), str(root / "backend")]

from PIL import Image  # noqa: E402

import distill_select as ds  # noqa: E402


def row(image, style="warm", top1="warm", margin=0.01, dino=0.95, item=1):
    return {"id": item, "style": style, "text_top1": top1, "text_margin": margin, "dino_product": dino,
            "image": image}


def test_best_candidate_must_win_its_style_and_keep_the_product():
    rows = [row("a", margin=0.02), row("b", margin=0.05), row("c", margin=0.09, top1="studio"),
            row("d", margin=0.08, dino=0.5), row("e", style="studio", top1="studio", margin=0.01)]
    chosen = ds.choose(rows, excluded=set())
    assert {k: v["image"] for k, v in chosen.items()} == {(1, "warm"): "b", (1, "studio"): "e"}
    assert ds.choose(rows, excluded={"b"})[(1, "warm")]["image"] == "a"


def test_framed_mask_matches_the_generation_canvas():
    photo = Image.new("RGB", (800, 800), "white")
    mask = Image.new("L", (800, 800), 0)
    mask.paste(255, (200, 100, 600, 700))

    def png(im):
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    framed = ds.framed_mask(png(photo), png(mask), {"height": 0.45, "bottom": 0.85}, 1024)
    assert framed.size == (1024, 1024)
    box = framed.point(lambda v: 255 if v >= 128 else 0).getbbox()
    assert box[3] == round(0.85 * 1024) and box[3] - box[1] == round(0.45 * 1024)
