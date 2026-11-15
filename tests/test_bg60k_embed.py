"""Etapa embed en Colab y paquete de imágenes de BG60k, sin Colab ni GPU."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embed_config_has_models_styles_and_other_subclasses():
    cfg = load("run_colab", "training/run_colab.py").build_config("embed", "r1", deadline_unix=9.0,
                                                                  pack_sha256="abc")
    assert cfg["models"]["clip"]["id"] and cfg["models"]["dino"]["id"]
    assert set(cfg["prompts"]["styles"]) == {"studio", "natural", "warm", "lifestyle", "minimal"}
    assert all(len(texts) >= 3 for texts in cfg["prompts"]["styles"].values())
    assert len(cfg["prompts"]["other"]) >= 5
    assert cfg["pack_sha256"] == "abc" and cfg["deadline_unix"] == 9.0
    json.dumps(cfg)


def unit_rows(n: int, dim: int, seed: int):
    np = pytest.importorskip("numpy")
    x = np.random.default_rng(seed).normal(size=(n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def write_result(out: Path, run_id: str, *, dino_ids=(1,), broken: bool = False, bad_sha: bool = False) -> None:
    import numpy as np
    out.mkdir(parents=True, exist_ok=True)
    index = {"items": [{"id": 1, "cutout": True}, {"id": 2, "cutout": False}]}
    (out / "index.json").write_text(json.dumps(index))
    clip_image = unit_rows(2, 8, 0)
    if broken:
        clip_image[0, 0] = np.nan
    np.savez(out / "clip.npz", ids=np.asarray([1, 2]), clip_image=clip_image,
             clip_text=unit_rows(3, 8, 1).astype(np.float32), prompt_group=np.asarray(["a", "b", "c"]))
    np.savez(out / "dino.npz", ids=np.asarray(list(dino_ids)), dino=unit_rows(len(dino_ids), 4, 2))
    files = {n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in ("clip.npz", "dino.npz", "index.json")}
    if bad_sha:
        files["clip.npz"] = "0" * 64
    (out / "manifest.json").write_text(json.dumps({"run_id": run_id, "stage": "embed", "status": "ok",
                                                   "files": files}))


def test_embed_result_is_checked(tmp_path):
    pytest.importorskip("numpy")
    drv = load("run_colab", "training/run_colab.py")
    write_result(tmp_path / "ok", "r1")
    assert drv.check_result(tmp_path / "ok", "r1")["stage"] == "embed"
    for name, kwargs in {"sha": {"bad_sha": True}, "nan": {"broken": True}, "ids": {"dino_ids": (2,)}}.items():
        write_result(tmp_path / name, "r1", **kwargs)
        with pytest.raises(drv.Failure):
            drv.check_result(tmp_path / name, "r1")


def test_colab_embed_is_self_contained():
    tree = ast.parse((ROOT / "training" / "colab_embed.py").read_text(encoding="utf-8"))
    imported = {(node.module if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
                for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    assert "productshot" not in imported


def test_embed_dry_run_uploads_pack_and_stops(tmp_path):
    pack = tmp_path / "pack.tar"
    pack.write_bytes(b"")
    out = subprocess.run(
        [sys.executable, str(ROOT / "training" / "run_colab.py"), "--stage", "embed", "--pack", str(pack),
         "--max-minutes", "30", "--dry-run"],
        capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "backend"), "TMPDIR": str(tmp_path)},
    ).stdout
    cmds = [line[2:] for line in out.splitlines() if line.startswith("+ ")]
    assert not any("requirements-colab" in c for c in cmds)
    upload = next(i for i, c in enumerate(cmds) if "/content/pack.tar" in c)
    run = next(i for i, c in enumerate(cmds) if " exec " in f" {c} " and "colab_embed.py" in c)
    stop = next(i for i, c in enumerate(cmds) if " stop" in c)
    assert upload < run < stop


def test_pack_keeps_one_copy_of_repeated_images_and_alerted_candidates(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa
    pack = load("bg60k_pack", "training/bg60k_pack.py")
    rows = [
        {"id": 3, "niche": "core", "alerts": [], "img_sha256": "a"},
        {"id": 1, "niche": "core", "alerts": [], "img_sha256": "a"},
        {"id": 2, "niche": "near", "alerts": [], "img_sha256": "b"},
        {"id": 4, "niche": "out", "alerts": [], "img_sha256": "c"},
        {"id": 5, "niche": "core", "alerts": ["truncated"], "img_sha256": "d"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "c.parquet")
    unique, duplicates = pack.candidates(tmp_path / "c.parquet")
    assert [r["id"] for r in unique] == [1, 2, 5] and duplicates == {3: 1}


def test_product_cutout_is_square_on_neutral_gray():
    from PIL import Image
    pack = load("bg60k_pack", "training/bg60k_pack.py")
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    mask = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    mask.paste((255, 0, 0, 255), (10, 30, 50, 50))
    cut = pack.product_cutout(img, mask, (10, 30, 50, 50))
    assert cut.size == (pack.SIDE, pack.SIDE)
    assert cut.getpixel((pack.SIDE // 2, pack.SIDE // 2)) == (255, 0, 0)
    assert cut.getpixel((pack.SIDE // 2, 2)) == (pack.NEUTRAL,) * 3


def test_split_parts_rebuild_the_file(tmp_path):
    drv = load("run_colab", "training/run_colab.py")
    original = tmp_path / "pack.tar"
    original.write_bytes(os.urandom(2500))
    (tmp_path / "parts").mkdir()
    parts = drv.split_parts(original, tmp_path / "parts", size=1000)
    assert [p.name for p in parts] == ["pack.tar.part000", "pack.tar.part001", "pack.tar.part002"]
    assert b"".join(p.read_bytes() for p in parts) == original.read_bytes()
