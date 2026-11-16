"""Experimento de calidad del modo intermedio (2026-11-15, noche).

Ataca los cuatro defectos vistos con fotos reales, con las mismas fotos y semillas:
1. segmentación: `birefnet-general` frente a `isnet-general-use` (rembg);
2. encuadre de catálogo: `framing.reframe` (4:5, producto centrado, recortado del ORIGINAL);
3. luz: `framing.harmonize` aplicado sobre la imagen recompuesta (se guardan ambas);
4. modelo: RealVisXL V4.0 inpainting (receta W) frente a FLUX.1 Fill [dev].

Configuraciones:
- W_bi: RealVisXL, foto completa, máscara BiRefNet (aísla el efecto de la segmentación)
- W_rf: RealVisXL, encuadre 4:5
- X_rf: FLUX.1 Fill [dev], encuadre 4:5

    PYTHONPATH=backend uv run --with modal modal deploy deploy/modal/quality_lab.py
    # y encolar con spawn (ver data/experiments/quality/spawned.json)

Requiere el secreto `huggingface` (HF_TOKEN; FLUX.1 Fill es gated, licencia no comercial).
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
CACHE = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "diffusers==0.32.2",  # FluxFillPipeline llegó en 0.32
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "safetensors==0.4.5",
        "huggingface_hub==0.26.2",
        "peft==0.13.2",
        "sentencepiece==0.2.0",
        "protobuf==5.28.3",
        "fastapi==0.162.0",
        "pydantic==2.14.2",
        "pydantic-settings==2.16.2",
        "pillow==12.3.0",
        "httpx==0.28.1",
        "rembg==2.0.59",
        # BiRefNet en GPU (en CPU tardaba 35 s por foto); rembg usa CUDA si está disponible.
        "onnxruntime-gpu==1.19.2",
    )
    .env({"HF_HOME": f"{CACHE}/hf", "U2NET_HOME": f"{CACHE}/u2net", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_dir(ROOT / "backend" / "productshot", "/root/productshot")
)

app = modal.App("productshot-quality", image=image)
cache = modal.Volume.from_name("productshot-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"


@app.cls(gpu="L40S", volumes={CACHE: cache}, secrets=[hf_secret], timeout=3600, scaledown_window=60,
         max_containers=1)
class QualityLab:
    @modal.method()
    def run(self, photos: dict[str, bytes], categories: dict[str, str], seed: int, run_id: str,
            configs: list[str], segmenters: tuple[str, ...] = ("isnet-general-use", "birefnet-general")) -> dict:
        import dataclasses
        import gc

        import torch
        from PIL import Image, ImageOps

        from productshot.config import Settings
        from productshot.domain import Category, Mode, Style
        from productshot.services import guardrails
        from productshot.services.framing import clean_mask, harmonize, reframe
        from productshot.services.generation import GenerationRequest, VariationSpec
        from productshot.services.images import prepare_upload
        from productshot.services.rembg_segmentation import RembgSegmentationService
        from productshot.services.scenes import build_scene_prompt
        from productshot.services.sdxl_generation import REALVIS_INPAINT_PARAMS, SdxlInpaintBackend

        settings = Settings(_env_file=None)
        run_dir = Path(CACHE) / "experiments_quality" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        previews: dict[str, bytes] = {}
        rows: list[dict] = []

        def keep(name: str, img: Image.Image) -> None:
            img.save(run_dir / f"{name}.png")
            small = img.convert("RGB").copy()
            small.thumbnail((768, 768))
            buf = io.BytesIO()
            small.save(buf, format="JPEG", quality=90)
            previews[f"{name}.jpg"] = buf.getvalue()

        # 1) Segmentación: isnet (actual) y BiRefNet, a la resolución de trabajo de la API.
        seg = {m: RembgSegmentationService(m) for m in segmenters}
        prepared = {}
        for name, data in photos.items():
            prep = prepare_upload(data, max_bytes=settings.max_upload_bytes,
                                  max_pixels=settings.max_image_pixels, work_max_side=settings.work_max_side)
            original = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            masks = {}
            for model, s in seg.items():
                t0 = time.perf_counter()
                masks[model] = s.segment(prep.working).mask
                rows.append({"photo": name, "segmenter": model, "seconds": round(time.perf_counter() - t0, 2),
                             "mask": guardrails.mask_metrics(masks[model], settings)})
                keep(f"{name}__mask_{model}", masks[model])
            mask, cleaned = clean_mask(masks["birefnet-general"])
            rows.append({"photo": name, "clean_mask": cleaned})
            keep(f"{name}__mask_final", mask)
            full_mask = mask.resize(original.size, Image.Resampling.BILINEAR)
            rf_img, rf_mask = reframe(original, full_mask)
            prepared[name] = {"working": prep.working, "mask": mask, "rf_img": rf_img, "rf_mask": rf_mask,
                              "category": Category(categories.get(name, "otro"))}
            keep(f"{name}__0_foto", prep.working)
            keep(f"{name}__0_encuadre", rf_img)
        # BiRefNet en GPU (onnxruntime) retiene ~15 GB: se libera antes de cargar los generadores.
        seg = s = None
        gc.collect()
        torch.cuda.empty_cache()
        cache.commit()

        specs = tuple(VariationSpec(position=i + 1, style=st, seed=seed + i) for i, st in enumerate(Style))

        def record(name, cfg, style, inp, msk, generated, seconds, prompt):
            final, fid = guardrails.enforce_fidelity(inp, msk, generated, settings)
            harmonized, hinfo = harmonize(final, msk)
            base = f"{name}__{cfg}__{style.value}"
            keep(base, final)
            keep(base + "__H", harmonized)
            rows.append({"photo": name, "config": cfg, "style": style.value, "seconds": seconds,
                         "drift": fid["protected_mean_abs_diff"], "harmonize": hinfo, "prompt": prompt})

        # 2) RealVisXL (receta W): foto completa con BiRefNet y encuadre 4:5.
        rv_cfgs = [c for c in configs if c.startswith(("W", "V"))]
        if rv_cfgs:
            rv = SdxlInpaintBackend(REALVIS_INPAINT_PARAMS)
            rv.load()
            cache.commit()
            for cfg in rv_cfgs:
                key_img, key_mask = ("rf_img", "rf_mask") if cfg.endswith("_rf") else ("working", "mask")
                for name, p in prepared.items():
                    arts = rv.generate(GenerationRequest(session_id="q", mode=Mode.INTERMEDIATE, image=p[key_img],
                                                         mask=p[key_mask], specs=specs, category=p["category"]))
                    for a in arts:
                        record(name, cfg, a.style, p[key_img], p[key_mask], a.image, a.params["seconds"],
                               a.params["prompt"])
                    cache.commit()
            rv = None
            gc.collect()
            torch.cuda.empty_cache()

        # 3) FLUX.1 Fill [dev] sobre el encuadre 4:5. Sin negative prompt (FLUX no lo usa).
        flux_cfgs = [c for c in configs if c.startswith(("X", "F"))]
        if flux_cfgs:
            from diffusers import FluxFillPipeline

            pipe = FluxFillPipeline.from_pretrained(FLUX_MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
            pipe.set_progress_bar_config(disable=True)
            cache.commit()
            for cfg, (name, p) in [(c, item) for c in flux_cfgs for item in prepared.items()]:
                img, msk = (p["rf_img"], p["rf_mask"]) if cfg.endswith("_rf") else (p["working"], p["mask"])
                w, h = (img.width // 16) * 16, (img.height // 16) * 16
                for spec in specs:
                    prompt = build_scene_prompt(spec.style, p["category"])
                    t0 = time.perf_counter()
                    out = pipe(prompt=prompt, image=img.resize((w, h)), mask_image=msk.resize((w, h)),
                               height=h, width=w, guidance_scale=30.0, num_inference_steps=28,
                               max_sequence_length=512,
                               generator=torch.Generator("cuda").manual_seed(spec.seed)).images[0]
                    out = out.resize(img.size, Image.Resampling.LANCZOS)
                    record(name, cfg, spec.style, img, msk, out, round(time.perf_counter() - t0, 2), prompt)
                cache.commit()
            manifest_extra = {"flux_peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}
        else:
            manifest_extra = {}

        manifest = {"run_id": run_id, "configs": configs, "seed": seed, "categories": categories,
                    "gpu": torch.cuda.get_device_name(0), "flux_model": FLUX_MODEL_ID,
                    "intermediate_params": dataclasses.asdict(REALVIS_INPAINT_PARAMS), **manifest_extra}
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        (run_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))
        cache.commit()
        return {"manifest": manifest, "rows": rows, "previews": previews}


@app.local_entrypoint()
def main(run_id: str = "", photos: str = "data/fotos", out: str = "data/experiments/quality"):
    """Encola la corrida en la app desplegada (no depende de la laptop) o baja una ya hecha."""
    if run_id:  # bajar resultados de una llamada encolada
        spawned = json.loads((Path(out) / "spawned.json").read_text())
        call = modal.FunctionCall.from_id(spawned[run_id])
        res = call.get(timeout=30)
        d = Path(out) / run_id
        d.mkdir(parents=True, exist_ok=True)
        for n, b in res["previews"].items():
            (d / n).write_bytes(b)
        (d / "rows.json").write_text(json.dumps(res["rows"], indent=2, default=str))
        (d / "manifest.json").write_text(json.dumps(res["manifest"], indent=2, default=str))
        print(f"Resultados en {d}")
        return
    print("Usa spawn_quality.py para encolar.")
