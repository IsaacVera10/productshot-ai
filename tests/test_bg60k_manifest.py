"""Manifiesto de BG60k con zip sintéticos."""

from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bg60k_manifest", ROOT / "training" / "bg60k_manifest.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)


def pair(size=64, box=(16, 16, 48, 40), second=None, color=(200, 40, 40), bg=(30, 120, 220)):
    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[:] = bg
    alpha = np.zeros((size, size), np.uint8)
    for b in [box] + ([second] if second else []):
        x0, y0, x1, y1 = b
        rgb[y0:y1, x0:x1] = color
        alpha[y0:y1, x0:x1] = 255
    mask = np.dstack([rgb * (alpha[..., None] > 0), alpha]).astype(np.uint8)
    return rgb, mask


def png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_mask_stats_single_product():
    _, mask = pair()
    s = bm.mask_stats(mask[..., 3])
    assert s["fg_frac"] == pytest.approx(32 * 24 / 64**2)
    assert s["n_components"] == 1 and s["largest_frac"] == 1.0
    assert (s["bbox_x0"], s["bbox_y0"], s["bbox_x1"], s["bbox_y1"]) == (16, 16, 48, 40)
    assert s["edges_touched"] == 0


def test_mask_stats_two_components_and_edges():
    _, mask = pair(box=(0, 0, 20, 64), second=(40, 10, 60, 30))
    s = bm.mask_stats(mask[..., 3])
    assert s["n_components"] == 2
    assert s["largest_frac"] == pytest.approx(20 * 64 / (20 * 64 + 20 * 20))
    assert s["edges_touched"] == 3


def test_mask_stats_empty():
    s = bm.mask_stats(np.zeros((32, 32), np.uint8))
    assert s["fg_frac"] == 0 and s["n_components"] == 0 and s["bbox_x0"] is None


def test_analyze_pair_reads_alpha_and_checks_rgb():
    rgb, mask = pair()
    row = bm.analyze_pair(png(rgb), png(mask))
    assert row["status"] == "ok" and row["mask_mode"] == "RGBA"
    assert row["mask_rgb_err"] == 0 and row["mask_rgb_outside"] == 0
    assert len(row["phash_img"]) == 16 and len(row["phash_cutout"]) == 16


def test_cutout_phash_ignores_background():
    a = bm.analyze_pair(*map(png, pair(bg=(30, 120, 220))))
    b = bm.analyze_pair(*map(png, pair(bg=(240, 230, 20))))
    assert a["phash_img"] != b["phash_img"]
    assert bm.hamming(a["phash_cutout"], b["phash_cutout"]) == 0


def test_build_on_synthetic_zips(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    ids = {0: "mug", 10: "candle", 3: "soap"}
    (raw / "bg60k_info.txt").write_text("".join(f"{i}.png\t{c}\n" for i, c in ids.items()))
    for shard, members in {0: [0, 10], 3: [3]}.items():
        with zipfile.ZipFile(raw / f"bg60k_imgs_{shard}.zip", "w") as z:
            for i in members:
                z.writestr(f"bg60k_imgs_{shard}/{i}.png", png(pair()[0]))
    with zipfile.ZipFile(raw / "bg60k_masks_0.zip", "w") as z:
        z.writestr("bg60k_masks_0/0_mask.png", png(pair()[1]))
    out = tmp_path / "out"
    s = bm.build(raw, out, workers=1)
    assert s["shards_complete"] == [0] and s["shards_partial"] == [3]
    assert s["rows"] == 2 and s["ok"] == 1 and s["status"] == {"ok": 1, "missing_member": 1}
    import pyarrow.parquet as pq
    t = pq.read_table(out / "manifest.parquet").to_pylist()
    assert [r["id"] for r in t] == [0, 10] and t[0]["category"] == "mug" and t[0]["shard"] == 0
    assert json.loads((out / "manifest_summary.json").read_text())["pairs_expected"] == 2


def test_build_resumes_saved_shards(tmp_path: Path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "bg60k_info.txt").write_text("0.png\tmug\n3.png\tsoap\n")
    for s in (0, 3):
        with zipfile.ZipFile(raw / f"bg60k_imgs_{s}.zip", "w") as z:
            z.writestr(f"bg60k_imgs_{s}/{s}.png", png(pair()[0]))
        with zipfile.ZipFile(raw / f"bg60k_masks_{s}.zip", "w") as z:
            z.writestr(f"bg60k_masks_{s}/{s}_mask.png", png(pair()[1]))
    out = tmp_path / "out"
    first = bm.build(raw, out, workers=1)
    assert first["ok"] == 2 and first["shards_reused"] == []
    (out / "shards" / "shard_3.parquet").unlink()
    calls = []
    real = bm.process_chunk
    monkeypatch.setattr(bm, "process_chunk", lambda *a: calls.append(a[2]) or real(*a))
    second = bm.build(raw, out, workers=1)
    assert second["shards_reused"] == [0] and calls == [[3]] and second["ok"] == 2
