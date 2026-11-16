"""Etapa eval del LoRA: configuración, paquete subido, resultado y una pasada con modelos falsos."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import tarfile
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from productshot.domain import Style
from productshot.services.presets import NEGATIVE_PROMPT, PRESETS
from productshot.services.sdxl_generation import SdxlParams

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


drv = load("run_colab_eval", "training/run_colab.py")
job = load("colab_eval", "training/colab_eval.py")


def adapters_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    for step in (900, 300, 1200, 600):
        (run / f"checkpoint-{step}_pytorch_lora_weights.safetensors").write_bytes(f"w{step}".encode())
    (run / "pytorch_lora_weights.safetensors").write_bytes(b"final")
    return run


def test_eval_config_mirrors_production():
    cfg = drv.build_config("eval", "r1", deadline_unix=5.0, adapters={"ckpt300": {"file": "a", "sha256": "x"}})
    p = SdxlParams()
    assert cfg["base_model"] == p.model_id and len(cfg["base_revision"]) == 40
    assert cfg["style_prompts"] == {s.value: PRESETS[s].prompt for s in Style}
    assert cfg["negative_prompt"] == NEGATIVE_PROMPT
    assert cfg["generation"] == {"steps": p.steps, "guidance_scale": p.guidance_scale, "strength": p.strength,
                                 "target_side": p.target_side, "mask_blur": p.mask_blur, "lora_scale": p.lora_scale}
    assert all(len(m["revision"]) == 40 for m in cfg["models"].values())
    json.dumps(cfg)
    with pytest.raises(ValueError):
        drv.build_config("eval", "r1")


def test_adapters_sorted_by_step_and_final_excluded(tmp_path):
    assert list(drv.collect_adapters(adapters_dir(tmp_path))) == ["ckpt300", "ckpt600", "ckpt900", "ckpt1200"]


def test_eval_tar_layout_and_hashes(tmp_path):
    split = tmp_path / "val"
    split.mkdir()
    (split / "eval.jsonl").write_text('{"id": 1}\n')
    (split / "1.png").write_bytes(b"png")
    train = tmp_path / "dataset.tar"
    with tarfile.open(train, "w") as tar:
        tar.add(split / "1.png", arcname="20.png")
    adapters = drv.collect_adapters(adapters_dir(tmp_path))
    spec = drv.build_eval_tar(tmp_path / "eval.tar", split, train, adapters)
    with tarfile.open(tmp_path / "eval.tar") as tar:
        names = set(tar.getnames())
        assert tar.extractfile("train/20.png").read() == b"png"
    assert {"val/eval.jsonl", "val/1.png", "train/20.png", "adapters/ckpt300.safetensors"} <= names
    assert spec["ckpt1200"]["sha256"] == hashlib.sha256(b"w1200").hexdigest()


def write_eval_result(out: Path, run_id: str, completed: list[int], *, drop: int = 0, dup: bool = False) -> None:
    out.mkdir(exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    cfg = drv.build_config("eval", run_id, adapters={"ckpt300": {}, "ckpt600": {}})
    styles = list(cfg["style_prompts"])
    rows = []
    for i in completed:
        for style in styles:
            for config in ("off", "ckpt300", "ckpt600"):
                name = f"{i}_{style}_{config}.jpg"
                Image.new("RGB", (8, 8)).save(out / "images" / name)
                rows.append({"id": i, "style": style, "config": config, "image": name, "clip_target": 0.6,
                             "clip_margin": 0.01, "clip_top1": style, "dino_product": 0.9,
                             "protected_mean_abs_diff": 2.0, "seconds": 9.0})
    rows = rows[: len(rows) - drop] + (rows[:1] if dup else [])
    (out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    centroids = np.eye(len(styles))
    np.savez(out / "style_refs.npz", styles=np.asarray(styles), centroids=centroids)
    files = {n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in ("metrics.jsonl", "style_refs.npz")}
    (out / "manifest.json").write_text(json.dumps({"run_id": run_id, "stage": "eval", "status": "ok", "config": cfg,
                                                   "files": files, "completed": completed}))


def test_check_result_eval(tmp_path):
    write_eval_result(tmp_path / "ok", "r1", [5, 6])
    assert drv.check_result(tmp_path / "ok", "r1")["completed"] == [5, 6]
    (tmp_path / "ok" / "images" / "5_warm_off.jpg").unlink()
    with pytest.raises(drv.Failure):
        drv.check_result(tmp_path / "ok", "r1")
    for case, kwargs in {"vacío": {"completed": []}, "falta": {"completed": [5], "drop": 1},
                         "duplicado": {"completed": [5], "dup": True}}.items():
        write_eval_result(tmp_path / case, "r1", **kwargs)
        with pytest.raises(drv.Failure):
            drv.check_result(tmp_path / case, "r1")


def test_product_crop_matches_pack_bbox():
    alpha = Image.new("L", (40, 40), 0)
    alpha.paste(255, (10, 10, 30, 30))
    alpha.paste(60, (0, 0, 40, 4))  # borde tenue: no cuenta para el bbox, como en el manifiesto
    image = Image.new("RGB", (40, 40), (0, 255, 0))
    crop = np.asarray(job.product_crop(image, alpha, 16, 128, 92), dtype=int)
    assert crop.shape == (16, 16, 3) and crop[8, 8, 1] > 200 and crop[0, 0, 1] > 200


def test_eval_job_is_self_contained():
    tree = ast.parse((ROOT / "training" / "colab_eval.py").read_text(encoding="utf-8"))
    imported = {
        (node.module if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "productshot" not in imported


def test_seed_is_stable():
    assert job.seed_for(479, "minimal") != job.seed_for(479, "studio")
    assert job.seed_for(479, "minimal") == int(hashlib.sha256(b"479:minimal").hexdigest()[:8], 16)


class FakePipe:
    """Pinta el fondo de un color por adaptador y ensucia el producto, para comprobar G8."""

    def __init__(self):
        self.active = None
        self.calls = []
        self.mask_processor = types.SimpleNamespace(blur=lambda mask, blur_factor: mask)

    def set_adapters(self, names, adapter_weights):
        self.active = (tuple(names), tuple(adapter_weights))

    def __call__(self, *, prompt, image, mask_image, height, width, generator, **kw):
        self.calls.append({"prompt": prompt, "size": (width, height), "active": self.active,
                           "seed": generator.seed, **kw})
        shade = 40 if self.active[1] == (0.0,) else 200
        return types.SimpleNamespace(images=[Image.new("RGB", (width, height), (shade, 0, 0))])


class FakeEncoder:
    def __init__(self, dim: int):
        self.dim = dim

    def images(self, images):
        x = np.stack([np.asarray(im.convert("RGB").resize((4, 4)), dtype=np.float32).ravel()[: self.dim] + 1
                      for im in images])
        return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_evaluate_product_pairs_configs_and_recomposes(tmp_path, monkeypatch):
    val = tmp_path / "eval" / "val"
    val.mkdir(parents=True)
    alpha = Image.new("L", (80, 80), 0)
    alpha.paste(255, (20, 20, 60, 60))
    cutout = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    cutout.paste(Image.new("RGBA", (40, 40), (10, 200, 30, 255)), (20, 20))
    cutout.save(val / "7_cutout.png")
    monkeypatch.setattr(job, "EVAL", tmp_path / "eval")
    monkeypatch.setattr(job, "IMAGES", tmp_path / "images")
    job.IMAGES.mkdir()
    fake_torch = types.SimpleNamespace(Generator=lambda device: types.SimpleNamespace(
        manual_seed=lambda s: types.SimpleNamespace(seed=s)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    cfg = drv.build_config("eval", "r1", adapters={"ckpt300": {}, "ckpt600": {}})
    cfg["generation"]["target_side"] = 64
    styles = list(cfg["style_prompts"])
    rng = np.random.default_rng(0)
    centroids = {s: v / np.linalg.norm(v) for s, v in zip(styles, rng.random((5, 48)))}
    pipe = FakePipe()
    rows = job.evaluate_product(pipe, FakeEncoder(48), FakeEncoder(48), centroids, cfg, {"id": 7, "style": "warm"})

    assert len(rows) == len(pipe.calls) == 5 * 3
    assert {c["size"] for c in pipe.calls} == {(64, 64)}
    assert pipe.calls[0]["active"] == (("ckpt300",), (0.0,))  # apagado = peso 0, como producción
    assert pipe.calls[1]["active"] == (("ckpt300",), (1.0,))
    assert pipe.calls[2]["active"] == (("ckpt600",), (1.0,))
    for style in styles:
        seeds = {c["seed"] for c, r in zip(pipe.calls, rows) if r["style"] == style}
        assert seeds == {job.seed_for(7, style)}
    assert all(c["prompt"] == cfg["style_prompts"][r["style"]] for c, r in zip(pipe.calls, rows))
    final = np.asarray(Image.open(job.IMAGES / rows[1]["image"]).convert("RGB"), dtype=int)
    assert np.abs(final[30, 30] - (10, 200, 30)).max() <= 3  # G8: el producto vuelve de la entrada
    assert final[5, 5, 0] > 150  # el fondo es el generado
    assert all(r["protected_mean_abs_diff"] > 0 for r in rows)
    assert all(-1 <= r["clip_margin"] <= 2 and r["clip_top1"] in styles for r in rows)


rep = load("lora_eval_report", "training/lora_eval_report.py")
CONFIGS = ["off", "ckpt300", "ckpt600", "ckpt900"]


def synthetic_rows(effect: dict[str, float], n: int = 20) -> list[dict]:
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        base = rng.normal(0, 0.02)
        for style in ("warm", "studio"):
            for config in CONFIGS:
                rows.append({"id": i, "style": style, "config": config, "clip_target": 0.6,
                             "clip_margin": base + effect.get(config, 0) + rng.normal(0, 0.002), "clip_top1": style,
                             "dino_product": 0.9, "protected_mean_abs_diff": 3.0})
    return rows


def test_report_prefers_earlier_equivalent_checkpoint():
    rows = synthetic_rows({"ckpt300": 0.001, "ckpt600": 0.030, "ckpt900": 0.031})
    _, decision = rep.report(rows, CONFIGS, ["warm", "studio"])
    assert decision["proposal"] == "ckpt600" and decision["best_mean"] == "ckpt900"


def test_report_keeps_baseline_without_improvement():
    _, decision = rep.report(synthetic_rows({"ckpt300": -0.01, "ckpt600": -0.01, "ckpt900": -0.01}), CONFIGS, ["warm", "studio"])
    assert decision["proposal"] == "off"


def test_report_is_provisional_with_few_products():
    _, decision = rep.report(synthetic_rows({"ckpt300": 0.05}, n=3), CONFIGS, ["warm", "studio"])
    assert decision["proposal"] is None


def test_eval_variants_with_scales():
    cfg = drv.build_config("eval", "r1", adapters={"ckpt300": {}}, scales=[0.3, 0.6], max_products=17)
    assert cfg["variants"] == {"ckpt300@0.3": {"adapter": "ckpt300", "scale": 0.3},
                               "ckpt300@0.6": {"adapter": "ckpt300", "scale": 0.6}}
    assert cfg["max_products"] == 17
    default = drv.build_config("eval", "r1", adapters={"ckpt300": {}})
    assert default["variants"] == {"ckpt300": {"adapter": "ckpt300", "scale": SdxlParams().lora_scale}}


@pytest.mark.parametrize("kwargs", [{"max_products": 0}, {"scales": [float("nan")]}, {"scales": [0.3, 0.30000001]}])
def test_eval_rejects_invalid_variants(kwargs):
    with pytest.raises(ValueError):
        drv.build_config("eval", "r1", adapters={"ckpt300": {}}, **kwargs)


def test_eval_variants_order_for_tie_break():
    cfg = drv.build_config("eval", "r1", adapters={"ckpt1200": {}, "ckpt300": {}}, scales=[0.6, 0.3])
    assert list(cfg["variants"]) == ["ckpt300@0.3", "ckpt300@0.6", "ckpt1200@0.3", "ckpt1200@0.6"]
