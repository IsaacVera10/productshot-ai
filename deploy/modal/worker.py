"""Worker de generación en Modal (ADR-009, alternativa A).

Sirve el contrato HTTP de `productshot.worker` con SDXL en una GPU L4. Es el mismo
worker que corre en la laptop; aquí solo cambia dónde se aloja.

Comandos (desde la raíz del repo, con `modal setup` hecho):

    modal run deploy/modal/worker.py::main           # prueba de humo: un lote de cinco
    modal run --detach deploy/modal/worker.py::experiment --config-set nueve   # expansión de silueta
    modal run deploy/modal/worker.py::collect --run-id <RUN_ID>       # bajar resultados
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
    # Tope por llamada. El lote de la API dura ~2 min; los experimentos, hasta ~40 min.
    timeout=3600,
    scaledown_window=120,
    max_containers=1,
)
class Worker:
    @modal.enter()
    def load(self) -> None:
        from productshot.services.rembg_segmentation import RembgSegmentationService
        from productshot.services.sdxl_generation import SdxlInpaintBackend

        from productshot.services.sdxl_generation import DEFAULT_MODEL_ID, SdxlParams

        # SDXL_MODEL_ID permite cambiar de checkpoint sin tocar código (P3, P4).
        self.backend = SdxlInpaintBackend(SdxlParams(model_id=os.environ.get("SDXL_MODEL_ID", DEFAULT_MODEL_ID)))
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
    def experiment(self, cases: list[dict], configs: dict[str, dict], styles: list[str],
                   seeds: list[int], run_id: str) -> dict:
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
        # Cada resultado se guarda en el volumen apenas sale: si el cliente local se cae, lo
        # hecho no se pierde (`modal run --detach` y luego `worker.py::collect`).
        run_dir = Path(CACHE) / "experiments" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        def save(name: str, data: bytes) -> None:
            (run_dir / name).write_bytes(data)

        # 1) Fotos y máscaras, una vez por caso (mismas semillas: mismas fotos en cada corrida).
        prepared = []
        for case in cases:
            photo = t2i(
                prompt=case["prompt"], negative_prompt="people, hands, text overlay, watermark, blurry",
                width=1024, height=768, num_inference_steps=30,
                generator=torch.Generator("cuda").manual_seed(case["seed"]),
            ).images[0]
            mask = self.segmenter.segment(photo).mask
            product = binary(mask)  # 255 = producto
            near = product.filter(ImageFilter.MaxFilter(81))
            edge = product.filter(ImageFilter.MaxFilter(9))
            prepared.append({
                "case": case, "photo": photo, "mask": mask,
                "area": max(1, sum(1 for v in product.getdata() if v)),
                "band": ImageChops.subtract(near, edge),  # franja de ~4 a 40 px alrededor del producto
                "entry": {"name": case["name"], "photo": png(photo), "mask": png(mask), "results": []},
            })
            save(f"{case['name']}__0_foto.png", prepared[-1]["entry"]["photo"])
            save(f"{case['name']}__1_mascara.png", prepared[-1]["entry"]["mask"])
        cache.commit()

        def record(p, cfg_name, style, seed, generated, seconds):
            final, fid = enforce_fidelity(p["photo"], p["mask"], generated, settings)
            fg = binary(self.segmenter.segment(final).mask)
            grown = ImageChops.multiply(fg, p["band"])
            row = {
                "case": p["case"]["name"], "config": cfg_name, "style": style, "seed": seed,
                "expansion": round(sum(1 for v in grown.getdata() if v) / p["area"], 3),
                "drift": fid["protected_mean_abs_diff"], "seconds": seconds,
            }
            image_name = f"{row['case']}__{cfg_name}__{style}__s{seed}.png"
            save(image_name, png(final))
            with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({**row, "image": image_name}, ensure_ascii=False) + "\n")
            p["entry"]["results"].append(row)

        # 2) Configuraciones. engine: "four" (SDXL base, 4 canales), "nine" (UNet de inpainting
        #    de 9 canales) o "composite" (fondo generado SIN producto y producto pegado encima).
        nine = None
        for cfg_name, cfg in configs.items():
            cfg = dict(cfg)
            engine = cfg.pop("engine", "four")
            use_hint = cfg.pop("use_hint", False)
            if engine == "nine" and nine is None:
                from productshot.services.sdxl_generation import NINE_CHANNEL_MODEL_ID, SdxlInpaintBackend, SdxlParams

                self.backend._pipe.to("cpu")  # libera la GPU para el segundo modelo
                torch.cuda.empty_cache()
                nine = SdxlInpaintBackend(SdxlParams(model_id=NINE_CHANNEL_MODEL_ID))
                nine.load()
                cache.commit()
            if engine != "nine" and nine is not None:
                raise ValueError("Las configuraciones de nueve canales deben ir al final.")
            for p in prepared:
                for seed in seeds:
                    specs = tuple(
                        VariationSpec(position=i + 1, style=Style(st), seed=seed + i) for i, st in enumerate(styles)
                    )
                    if engine == "composite":
                        from productshot.services.generation import build_prompt

                        for spec in specs:
                            t0 = time.perf_counter()
                            bg = t2i(
                                prompt=build_prompt(spec.style, ()) + ", empty surface in the center, nothing on it",
                                negative_prompt="product, object, mug, bottle, vase, box, people, text, watermark",
                                width=1024, height=768, num_inference_steps=30,
                                generator=torch.Generator("cuda").manual_seed(spec.seed),
                            ).images[0].resize(p["photo"].size)
                            record(p, cfg_name, spec.style.value, seed, bg, round(time.perf_counter() - t0, 2))
                        continue
                    backend = nine if engine == "nine" else self.backend
                    params = dataclasses.replace(backend.params if engine == "nine" else base, **cfg)
                    if use_hint:
                        params = dataclasses.replace(params, prompt_prefix=p["case"]["hint"])
                    previous, backend.params = backend.params, params
                    try:
                        arts = backend.generate(GenerationRequest(
                            session_id="exp", mode=Mode.INTERMEDIATE, image=p["photo"], mask=p["mask"], specs=specs,
                        ))
                    finally:
                        backend.params = previous
                    for art in arts:
                        record(p, cfg_name, art.style.value, seed, art.image, art.params["seconds"])
            cache.commit()  # una configuración completa queda a salvo
        info = {
            "base_revision": self.backend.model_revision,
            "nine_revision": nine.model_revision if nine is not None else None,
            "configs": list(configs), "styles": styles, "seeds": seeds,
        }
        (run_dir / "done.json").write_text(json.dumps(info, indent=2))
        cache.commit()
        return info


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
EXPERIMENT_SETS = {
    # 2026-11-13: mitigaciones sobre el UNet de cuatro canales (docs/05).
    "mitigaciones": {
        "A_base": {},
        "B_prefijo": {"use_hint": True},
        "C_margen": {"mask_dilate_px": 12, "mask_blur": 0},
        "D_strength_085": {"strength": 0.85},
        "E_prefijo_margen": {"use_hint": True, "mask_dilate_px": 12, "mask_blur": 0},
    },
    # UNet de nueve canales frente a la base y a componer sobre un fondo vacío (P4).
    "nueve": {
        "A_base": {},
        "D_strength_085": {"strength": 0.85},
        "H_fondo_compuesto": {"engine": "composite"},
        "F_nueve_canales": {"engine": "nine"},
        "G_nueve_strength_085": {"engine": "nine", "strength": 0.85},
    },
}


@app.local_entrypoint()
def experiment(config_set: str = "nueve", styles: str = "studio,lifestyle", seeds: str = "500,900") -> None:
    """Lanzar con `--detach` para que siga aunque se cierre la terminal:

        PYTHONPATH=backend modal run --detach deploy/modal/worker.py::experiment --config-set nueve

    Al terminar (o si la terminal se cayó), bajar los resultados con `worker.py::collect`.
    """
    configs = EXPERIMENT_SETS[config_set]
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{config_set}"
    print(f"RUN_ID={run_id}", flush=True)
    style_list = [s.strip() for s in styles.split(",") if s.strip()]
    seed_list = [int(x) for x in seeds.split(",") if x.strip()]
    Worker().experiment.remote(EXPERIMENT_CASES, configs, style_list, seed_list, run_id)
    collect(run_id)


@app.local_entrypoint()
def collect(run_id: str, out: str = "data/experiments/expansion") -> None:
    """Baja del volumen los resultados de un experimento y arma las grillas y el resumen."""
    import subprocess
    import sys

    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "modal", "volume", "get", "--force", "productshot-hf-cache",
         f"experiments/{run_id}", str(dest)],
        check=True,
    )
    summarize_run(dest / run_id)


def summarize_run(run_dir: Path) -> None:
    """Grilla por caso (filas: configuraciones; columnas: estilo × semilla) y medias por configuración."""
    from PIL import Image, ImageDraw

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    done = run_dir / "done.json"
    print("Corrida completa." if done.exists() else "Corrida INCOMPLETA: solo lo guardado hasta el corte.")
    configs = list(dict.fromkeys(r["config"] for r in rows))
    combos = list(dict.fromkeys((r["style"], r["seed"]) for r in rows))
    for case in dict.fromkeys(r["case"] for r in rows):
        photo = Image.open(run_dir / f"{case}__0_foto.png").convert("RGB")
        w, h = photo.size
        sheet = Image.new("RGB", ((2 + len(combos)) * w, len(configs) * (h + 18)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, name in enumerate(("0_foto", "1_mascara")):
            sheet.paste(Image.open(run_dir / f"{case}__{name}.png").convert("RGB"), (i * w, 18))
        for r in (r for r in rows if r["case"] == case):
            c = 2 + combos.index((r["style"], r["seed"]))
            y = configs.index(r["config"]) * (h + 18)
            sheet.paste(Image.open(run_dir / r["image"]).convert("RGB"), (c * w, y + 18))
            draw.text((c * w + 4, y + 3), f"{r['config']} · {r['style']} · s{r['seed']} · exp {r['expansion']}", fill="black")
        sheet.save(run_dir / f"grilla_{case}.png")
    for cfg in configs:
        vals = [r["expansion"] for r in rows if r["config"] == cfg]
        print(f"{cfg:22s} n={len(vals):2d}  expansión media {sum(vals) / len(vals):.3f}  máx {max(vals):.3f}")
    print(f"Resultados en {run_dir}")


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
