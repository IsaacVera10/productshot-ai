"""Worker de generación en Modal (ADR-009, alternativa A).

Sirve el contrato HTTP de `productshot.worker` con SDXL en una GPU L4. Es el mismo
worker que corre en la laptop; aquí solo cambia dónde se aloja.

Comandos (desde la raíz del repo, con `modal setup` hecho):

    modal run deploy/modal/worker.py::main           # prueba de humo: un lote de cinco
    modal run deploy/modal/worker.py::experiment     # experimento de expansión de silueta
    modal run deploy/modal/worker.py::main --image foto.png --mask mascara.png
    modal deploy deploy/modal/worker.py              # endpoint permanente, escala a cero

Costos: la GPU solo se cobra mientras hay un contenedor vivo. `scaledown_window` lo
apaga a los 2 minutos sin solicitudes y `max_containers=1` impide que se multiplique.
Los pesos (~7 GB) quedan en un volumen, así que se descargan una sola vez.

Secreto requerido: `productshot-worker` con `GENERATION_WORKER_TOKEN` (el mismo valor
va en el `.env` de la API).
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import modal

# Solo en la máquina local existe el repo; dentro del contenedor el archivo está en /root.
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
CACHE = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Generación (versiones fijas: la revisión del modelo se registra en cada variación)
        "torch==2.5.1",
        "diffusers==0.31.0",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "safetensors==0.4.5",
        "huggingface_hub==0.26.2",
        # Contrato HTTP (mismas versiones que uv.lock)
        "fastapi==0.162.0",
        "pydantic==2.14.2",
        "pydantic-settings==2.16.2",
        "pillow==12.3.0",
        "httpx==0.28.1",
        # Segmentación del producto (en CPU dentro del mismo contenedor)
        "rembg==2.0.59",
        "onnxruntime==1.19.2",
    )
    .env({"HF_HOME": f"{CACHE}/hf", "U2NET_HOME": f"{CACHE}/u2net", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_dir(ROOT / "backend" / "productshot", "/root/productshot")
)

app = modal.App("productshot-worker", image=image)
cache = modal.Volume.from_name("productshot-hf-cache", create_if_missing=True)
secret = modal.Secret.from_name("productshot-worker")


@app.cls(
    gpu="L4",
    volumes={CACHE: cache},
    secrets=[secret],
    timeout=900,
    scaledown_window=120,
    max_containers=1,
)
class Worker:
    @modal.enter()
    def load(self) -> None:
        from productshot.services.rembg_segmentation import RembgSegmentationService
        from productshot.services.sdxl_generation import SdxlInpaintBackend

        self.backend = SdxlInpaintBackend()
        self.backend.load()
        self.segmenter = RembgSegmentationService()
        self.segmenter.load()
        cache.commit()  # guarda los pesos descargados para el próximo arranque

    @modal.asgi_app()
    def web(self):
        from productshot.worker.app import create_worker_app

        return create_worker_app(
            self.backend, token=os.environ.get("GENERATION_WORKER_TOKEN"), segmenter=self.segmenter
        )

    @modal.method()
    def smoke(self, image_png: bytes, mask_png: bytes, mode: str = "intermediate") -> dict:
        """Un lote de cinco con las mismas piezas que usa la API; devuelve métricas e imágenes."""
        import torch
        from PIL import Image

        from productshot.domain import CANONICAL_STYLES, Mode
        from productshot.services.generation import GenerationRequest, VariationSpec

        img = Image.open(io.BytesIO(image_png)).convert("RGB")
        mask = Image.open(io.BytesIO(mask_png)).convert("L")
        specs = tuple(VariationSpec(position=i + 1, style=s, seed=1000 + i) for i, s in enumerate(CANONICAL_STYLES))
        t0 = time.perf_counter()
        arts = self.backend.generate(
            GenerationRequest(session_id="smoke", mode=Mode(mode), image=img, mask=mask, specs=specs)
        )
        total = time.perf_counter() - t0
        images = {}
        for a in arts:
            buf = io.BytesIO()
            a.image.save(buf, format="PNG")
            images[f"{a.position}_{a.style.value}.png"] = buf.getvalue()
        return {
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
            "model_revision": self.backend.model_revision,
            "load_seconds": self.backend.load_seconds,
            "batch_seconds": round(total, 1),
            "per_image": [{k: a.params[k] for k in ("seconds", "peak_vram_gb", "generation_size")} | {"style": a.style.value} for a in arts],
            "params": {k: arts[0].params[k] for k in ("steps", "guidance_scale", "strength")},
            "images": images,
        }


    @modal.method()
    def experiment(self, cases: list[dict], configs: dict[str, dict], styles: list[str]) -> dict:
        """Expansión de silueta: fotos generadas por SDXL, máscara de rembg y varias mitigaciones.

        Las fotos de producto las genera el propio SDXL (texto a imagen), así no se usan
        imágenes de terceros. Para cada combinación devuelve la imagen recompuesta y la
        métrica de expansión: fracción del área del producto que rembg detecta como primer
        plano NUEVO en una franja alrededor del producto (0 = no se prolongó la silueta).
        """
        import dataclasses

        import torch
        from diffusers import StableDiffusionXLPipeline
        from PIL import Image, ImageChops, ImageFilter

        from productshot.config import Settings
        from productshot.domain import Mode, Style
        from productshot.services.generation import GenerationRequest, VariationSpec
        from productshot.services.guardrails import enforce_fidelity

        def png(img, side=512) -> bytes:
            img = img.copy()
            img.thumbnail((side, side))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        def binary(mask_l, keep_black=True):
            return mask_l.point(lambda v: 255 if (v < 128) == keep_black else 0)

        t2i = StableDiffusionXLPipeline.from_pipe(self.backend._pipe)
        base = self.backend.params
        settings = Settings(_env_file=None)
        out = {"cases": []}
        for case in cases:
            photo = t2i(
                prompt=case["prompt"], negative_prompt="people, hands, text overlay, watermark, blurry",
                width=1024, height=768, num_inference_steps=30,
                generator=torch.Generator("cuda").manual_seed(case["seed"]),
            ).images[0]
            mask = self.segmenter.segment(photo).mask
            product = binary(mask)  # 255 = producto
            area = max(1, sum(1 for v in product.getdata() if v))
            near = product.filter(ImageFilter.MaxFilter(81))
            edge = product.filter(ImageFilter.MaxFilter(9))
            band = ImageChops.subtract(near, edge)  # franja de ~4 a 40 px alrededor del producto
            entry = {"name": case["name"], "photo": png(photo), "mask": png(mask), "results": []}
            for cfg_name, cfg in configs.items():
                cfg = dict(cfg)
                if cfg.pop("use_hint", False):
                    cfg["prompt_prefix"] = case["hint"]
                self.backend.params = dataclasses.replace(base, **cfg)
                specs = tuple(VariationSpec(position=i + 1, style=Style(st), seed=500 + i) for i, st in enumerate(styles))
                arts = self.backend.generate(
                    GenerationRequest(session_id="exp", mode=Mode.INTERMEDIATE, image=photo, mask=mask, specs=specs)
                )
                for art in arts:
                    final, fid = enforce_fidelity(photo, mask, art.image, settings)
                    fg = binary(self.segmenter.segment(final).mask)
                    grown = ImageChops.multiply(fg, band)
                    expansion = round(sum(1 for v in grown.getdata() if v) / area, 3)
                    entry["results"].append({
                        "config": cfg_name, "style": art.style.value, "expansion": expansion,
                        "drift": fid["protected_mean_abs_diff"], "seconds": art.params["seconds"],
                        "image": png(final),
                    })
            self.backend.params = base
            out["cases"].append(entry)
        return out


EXPERIMENT_CASES = [
    {"name": "taza_logo", "seed": 11, "hint": "a white ceramic mug",
     "prompt": "smartphone photo of a white ceramic coffee mug with a small printed logo on a kitchen counter, natural light"},
    {"name": "jarron_mate", "seed": 12, "hint": "a matte black ceramic vase",
     "prompt": "amateur photo of a matte black ceramic vase on a wooden desk at home"},
    {"name": "blanco_sobre_claro", "seed": 13, "hint": "a white plastic lotion bottle",
     "prompt": "amateur photo of a white plastic lotion bottle on a light beige table"},
    {"name": "caja_con_texto", "seed": 14, "hint": "a small cardboard tea box",
     "prompt": "smartphone photo of a small cardboard tea box with small printed text on a table"},
]
EXPERIMENT_CONFIGS = {
    "A_base": {},
    "B_prefijo": {"use_hint": True},
    "C_margen": {"mask_dilate_px": 12, "mask_blur": 0},
    "D_strength_085": {"strength": 0.85},
    "E_prefijo_margen": {"use_hint": True, "mask_dilate_px": 12, "mask_blur": 0},
}


@app.local_entrypoint()
def experiment(styles: str = "studio,lifestyle", out: str = "") -> None:
    """`PYTHONPATH=backend modal run deploy/modal/worker.py::experiment`"""
    from PIL import Image, ImageDraw

    style_list = [s.strip() for s in styles.split(",") if s.strip()]
    result = Worker().experiment.remote(EXPERIMENT_CASES, EXPERIMENT_CONFIGS, style_list)
    out_dir = Path(out or "data/experiments/expansion") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for case in result["cases"]:
        tiles = [("foto", case["photo"]), ("máscara rembg", case["mask"])]
        tiles += [(f"{r['config']} · {r['style']} · exp {r['expansion']}", r["image"]) for r in case["results"]]
        ims = [Image.open(io.BytesIO(b)).convert("RGB") for _, b in tiles]
        w, h = ims[0].size
        cols = 2 + len(style_list)
        rows = len(EXPERIMENT_CONFIGS)
        sheet = Image.new("RGB", (cols * w, rows * (h + 18)), "white")
        draw = ImageDraw.Draw(sheet)
        positions = [(0, 0), (1, 0)] + [(2 + i % len(style_list), i // len(style_list)) for i in range(len(case["results"]))]
        for (label, _), im, (c, r) in zip(tiles, ims, positions):
            sheet.paste(im, (c * w, r * (h + 18) + 18))
            draw.text((c * w + 4, r * (h + 18) + 3), label, fill="black")
        sheet.save(out_dir / f"{case['name']}.png")
        for r in case["results"]:
            table.append({"case": case["name"], **{k: v for k, v in r.items() if k != "image"}})
    (out_dir / "metrics.json").write_text(json.dumps(table, indent=2, ensure_ascii=False))
    by_cfg: dict[str, list[float]] = {}
    for row in table:
        by_cfg.setdefault(row["config"], []).append(row["expansion"])
    for cfg, vals in by_cfg.items():
        print(f"{cfg:18s} expansión media {sum(vals) / len(vals):.3f}  máx {max(vals):.3f}")
    print(f"Resultados en {out_dir}")


def _demo_product(size=(1024, 768)):
    """Producto sintético (una taza simple) con su máscara exacta, si no se pasa una foto."""
    from PIL import Image, ImageDraw

    w, h = size
    img = Image.new("RGB", size, (214, 210, 202))
    mask = Image.new("L", size, 255)
    for target, fill_body, fill_handle in ((img, (176, 38, 44), (150, 30, 36)), (mask, 0, 0)):
        d = ImageDraw.Draw(target)
        d.rounded_rectangle([w * 0.38, h * 0.30, w * 0.60, h * 0.78], radius=18, fill=fill_body)
        d.ellipse([w * 0.56, h * 0.40, w * 0.70, h * 0.62], outline=fill_handle, width=int(w * 0.022))
    return img, mask


@app.local_entrypoint()
def main(image: str = "", mask: str = "", mode: str = "intermediate", out: str = "") -> None:
    from PIL import Image

    if image:
        img = Image.open(image).convert("RGB")
        msk = Image.open(mask).convert("L") if mask else None
        if msk is None:
            raise SystemExit("Con --image también hace falta --mask (blanco = fondo, negro = producto).")
    else:
        img, msk = _demo_product()

    def png(i) -> bytes:
        buf = io.BytesIO()
        i.save(buf, format="PNG")
        return buf.getvalue()

    result = Worker().smoke.remote(png(img), png(msk), mode)
    out_dir = Path(out or "data/smoke") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    # G8 local: el producto se recompone desde la foto, igual que hace la API con cada lote.
    from productshot.config import Settings
    from productshot.services.guardrails import enforce_fidelity

    fidelity = {}
    for name, data in result.pop("images").items():
        (out_dir / f"raw_{name}").write_bytes(data)
        final, metrics = enforce_fidelity(img, msk, Image.open(io.BytesIO(data)), Settings(_env_file=None))
        final.save(out_dir / name)
        fidelity[name] = metrics["protected_mean_abs_diff"]
    result["product_drift_before_recompose"] = fidelity
    img.save(out_dir / "0_input.png")
    msk.save(out_dir / "0_mask.png")
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Imágenes en {out_dir}")
