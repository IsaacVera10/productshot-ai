"""Captions del piloto de BG60k: preset del estilo más el producto."""

from __future__ import annotations

import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
from PIL import Image  # noqa: E402
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "training"), str(root / "backend")]

import bg60k_export as be  # noqa: E402
from productshot.domain import Style  # noqa: E402
from productshot.services.presets import PRESETS  # noqa: E402


def test_caption_starts_with_the_inference_preset():
    for style, preset in ((s.value, p) for s, p in PRESETS.items()):
        assert be.caption(style, "mug") == f"{preset.prompt}, mug"


def test_nouns_are_short_and_ids_override():
    categories, ids = be.noun_lookup()
    assert all(0 < len(n.split()) <= 4 for n in categories.values())
    assert ids[20] == "coffee mug"


def png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_product_mask_comes_from_the_cutout_alpha():
    cutout = Image.new("RGBA", (4, 2), (0, 0, 0, 0))
    cutout.putpixel((1, 0), (9, 9, 9, 200))
    cutout.putpixel((2, 0), (9, 9, 9, 100))
    mask = Image.open(io.BytesIO(be.product_mask(png(cutout), (4, 2))))
    assert mask.mode == "L" and mask.tobytes() == bytes([0, 255, 0, 0, 0, 0, 0, 0])
    with pytest.raises(SystemExit):
        be.product_mask(png(cutout), (8, 8))


def test_preset_export_with_masks(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    photo = png(Image.new("RGB", (4, 2), "white"))
    cutout = png(Image.new("RGBA", (4, 2), (0, 0, 0, 255)))
    with zipfile.ZipFile(raw / "z.zip", "w") as z:
        z.writestr("1.jpg", photo)
        z.writestr("1.png", cutout)
    row = {"id": 1, "split": "train", "style": "studio", "noun": "mug", "caption": be.caption("studio", "mug"),
           "img_zip": "z.zip", "img_member": "1.jpg", "mask_zip": "z.zip", "mask_member": "1.png"}
    out = tmp_path / "export"
    (out / "eval").mkdir(parents=True)
    be.export([row], raw, out, "preset_masked", "preset", masks=True)
    with tarfile.open(out / "preset_masked.tar") as tar:
        meta = json.loads(tar.extractfile("metadata.jsonl").read())
        assert meta == {"file_name": "1.png", "text": PRESETS[Style.STUDIO].prompt, "style": "studio",
                        "mask": "1.mask"}
        assert Image.open(tar.extractfile("1.mask")).size == (4, 2)
