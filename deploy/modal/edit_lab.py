"""Experimento: modelos de EDICIÓN por instrucciones para el modo intermedio (2026-11-16).

El inpainting con el producto fijo (SDXL, RealVisXL, FLUX.1 Fill) se ve "pegado": la luz del
producto no coincide con la escena. Aquí se prueban editores que re-dibujan la foto completa
conservando la identidad del producto, con instrucciones al estilo de las que usó Aaron en
ChatGPT/Gemini:

- Q21: Qwen/Qwen-Image-2.1 (Qwen Research License, 7B + Qwen3-VL 8B)
- K2:  black-forest-labs/FLUX.2-klein-9B (FLUX Non-Commercial, destilado a 4 pasos)
- KX:  black-forest-labs/FLUX.1-Kontext-dev (FLUX.1-dev NC; mejor CER en ProductConsistency)
- KXR: KX + LoRA chestnutlzj/Edit-R1-FLUX.1-Kontext-dev (Apache-2.0)

Cada foto entra de dos formas: "full" (la foto casera) y "cut" (el producto recortado con
BiRefNet sobre gris claro), para ver si la segmentación ayuda como guía.

    PYTHONPATH=backend uv run --with modal modal deploy deploy/modal/edit_lab.py
    uv run --with modal python <lanzador>   # spawn por modelo (ver data/experiments/edit/spawned.json)
    PYTHONPATH=backend uv run --with modal modal run deploy/modal/edit_lab.py::collect --run-id <id>
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
CACHE = "/cache"

gen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "diffusers==0.43.0",
        "transformers==5.23.0",
        "accelerate==1.15.0",
        "peft==0.21.3",
        "sentencepiece==0.2.3",
        "protobuf",
        "safetensors",
        "pillow==12.3.0",
    )
    .env({"HF_HOME": f"{CACHE}/hf", "HF_HUB_DISABLE_TELEMETRY": "1"})
)
seg_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("rembg==2.0.59", "onnxruntime==1.19.2", "pillow==12.3.0", "pydantic-settings==2.16.2",
                 "pydantic==2.14.2", "fastapi==0.162.0", "httpx==0.28.1")
    .env({"U2NET_HOME": f"{CACHE}/u2net"})
    .add_local_dir(ROOT / "backend" / "productshot", "/root/productshot")
)

app = modal.App("productshot-edit")
cache = modal.Volume.from_name("productshot-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

STYLES = {
    "studio": "Studio shot on a seamless light grey backdrop, soft diffused softbox lighting, soft contact shadow",
    "natural": "Outdoor shot on a natural stone surface in soft daylight, green foliage softly blurred in the background",
    "warm": "Warm golden hour sunlight on a wooden table, cozy amber tones, soft bokeh in the background",
    "lifestyle": "Lifestyle shot on a {setting} in a bright modern home, {props} softly out of focus in the background",
    "minimal": "Minimalist composition on a plain off-white surface against a plain wall, generous empty space, "
               "single soft shadow",
}
SETTINGS = {
    "tecnologia": ("clean home office desk", "a laptop and a notebook"),
    "hogar": ("living room side table", "a sofa and a small plant"),
    "cocina": ("kitchen breakfast table", "a coffee pot and a small plant"),
    "juguetes": ("bedroom desk", "a bookshelf and a small plant"),
}
PRODUCTS = {  # foto -> (sustantivo en inglés, categoría)
    "000_hub": ("USB-C hub", "tecnologia"),
    "001_candle": ("scented candle in a black glass jar", "hogar"),
    "002_cup": ("printed ceramic mug", "cocina"),
    "003_printer": ("pink mini thermal printer next to its retail box", "tecnologia"),
    "015_caja_ssd_ugreen": ("NVMe SSD enclosure in its retail box", "tecnologia"),
    "016_caja_ssd_wd": ("NVMe SSD in its retail box", "tecnologia"),
}


STYLE_WORDS = {"studio": "studio", "natural": "natural, outdoor", "warm": "warm, cozy",
               "lifestyle": "lifestyle", "minimal": "minimalist"}


def short_instruction(photo: str, style: str) -> str:
    """Pedido corto, como lo escribiría una persona del nicho (el que Aaron usó en ChatGPT)."""
    noun, _ = PRODUCTS[photo]
    return (f"Make a professional {STYLE_WORDS[style]} photo to sell this {noun}, keeping the product and all the "
            "information printed on it exactly as it is, so that it looks more attractive without misrepresenting it.")


def instruction(photo: str, style: str) -> str:
    noun, cat = PRODUCTS[photo]
    setting, props = SETTINGS[cat]
    scene = STYLES[style].format(setting=setting, props=props)
    return (f"Turn this into a professional e-commerce photo to sell this exact {noun}. {scene}. "
            "Keep the product identical to the original: same shape, proportions, colors, logo and every "
            "printed word and number exactly as shown. Do not add or remove text. Photorealistic, sharp focus "
            "on the product, high-end commercial product photography.")


@app.function(image=seg_image, volumes={CACHE: cache}, cpu=8, memory=16384, timeout=1800)
def cutouts(photos: dict[str, bytes]) -> dict[str, dict[str, bytes]]:
    """Máscara BiRefNet (CPU) y recorte del producto sobre gris claro, a la resolución de trabajo."""
    from PIL import Image, ImageOps

    from productshot.services.framing import clean_mask
    from productshot.services.rembg_segmentation import RembgSegmentationService

    seg = RembgSegmentationService("birefnet-general")
    out = {}
    for name, data in photos.items():
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        mask, _ = clean_mask(seg.segment(img).mask)  # blanco = fondo
        cut = Image.composite(Image.new("RGB", img.size, (236, 236, 236)), img, mask)
        enc = {}
        for k, im in (("full", img), ("cut", cut), ("mask", mask)):
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            enc[k] = buf.getvalue()
        out[name] = enc
    cache.commit()
    return out


@app.cls(image=gen_image, gpu="H100", volumes={CACHE: cache}, secrets=[hf_secret], timeout=3600,
         scaledown_window=60, max_containers=4)
class EditLab:
    @modal.method()
    def run(self, config: str, inputs: dict[str, dict[str, bytes]], run_id: str, seed: int = 500,
            prompt_mode: str = "long", label: str | None = None) -> dict:
        label = label or config
        import torch
        from PIL import Image

        run_dir = Path(CACHE) / "experiments_edit" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        if config == "Q21":
            from diffusers import QwenImage21Pipeline

            pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16).to("cuda")
            call = dict(num_inference_steps=40)
        elif config == "Q21T":
            from diffusers import QwenImage21Pipeline

            pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1-Turbo",
                                                       torch_dtype=torch.bfloat16).to("cuda")
            call = dict(num_inference_steps=8)
        elif config == "K2":
            from diffusers import Flux2KleinPipeline

            pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-9B",
                                                      torch_dtype=torch.bfloat16).to("cuda")
            call = dict(num_inference_steps=4, guidance_scale=1.0)
        elif config in ("KX", "KXR"):
            from diffusers import FluxKontextPipeline

            pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev",
                                                       torch_dtype=torch.bfloat16).to("cuda")
            if config == "KXR":
                pipe.load_lora_weights("chestnutlzj/Edit-R1-FLUX.1-Kontext-dev")
            call = dict(num_inference_steps=28, guidance_scale=2.5)
        else:
            raise ValueError(config)
        pipe.set_progress_bar_config(disable=True)
        load_s = round(time.perf_counter() - t0, 1)
        cache.commit()

        previews, rows = {}, []
        for name, enc in inputs.items():
            for kind in ("full", "cut"):
                src = Image.open(io.BytesIO(enc[kind])).convert("RGB")
                # ~1 MP con la proporción de la foto, múltiplos de 16 (lo que esperan los tres).
                scale = (1024 * 1024 / (src.width * src.height)) ** 0.5
                w, h = max(16, round(src.width * scale / 16) * 16), max(16, round(src.height * scale / 16) * 16)
                for i, style in enumerate(STYLES):
                    prompt = short_instruction(name, style) if prompt_mode == "short" else instruction(name, style)
                    torch.cuda.reset_peak_memory_stats()
                    t1 = time.perf_counter()
                    kw = dict(prompt=prompt, image=src.resize((w, h), Image.Resampling.LANCZOS),
                              generator=torch.Generator("cuda").manual_seed(seed + i), **call)
                    if not config.startswith("Q21"):
                        kw.update(height=h, width=w)
                    try:
                        out = pipe(**kw).images[0]
                        err = None
                    except Exception as exc:  # noqa: BLE001 - se registra y se sigue con el resto
                        out, err = None, f"{type(exc).__name__}: {exc}"[:400]
                    secs = round(time.perf_counter() - t1, 2)
                    base = f"{name}__{label}__{kind}__{style}"
                    row = {"photo": name, "config": label, "prompt_mode": prompt_mode, "input": kind, "style": style, "seed": seed + i,
                           "seconds": secs, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 1),
                           "prompt": prompt, "error": err}
                    if out is not None:
                        out.save(run_dir / f"{base}.png")
                        row["size"] = list(out.size)
                        small = out.convert("RGB").copy()
                        small.thumbnail((768, 768))
                        buf = io.BytesIO()
                        small.save(buf, format="JPEG", quality=90)
                        previews[f"{base}.jpg"] = buf.getvalue()
                    rows.append(row)
                cache.commit()
        meta = {"config": label, "model_config": config, "prompt_mode": prompt_mode, "load_seconds": load_s, "gpu": torch.cuda.get_device_name(0), "call": call}
        (run_dir / f"rows_{label}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
        cache.commit()
        return {"meta": meta, "rows": rows, "previews": previews}


@app.local_entrypoint()
def collect(run_id: str, out: str = "data/experiments/edit"):
    """Baja los resultados de las llamadas encoladas para `run_id` y arma una grilla por foto."""
    from PIL import Image, ImageDraw

    spawned = json.loads((Path(out) / "spawned.json").read_text())[run_id]
    d = Path(out) / run_id
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg, call_id in spawned["calls"].items():
        try:
            res = modal.FunctionCall.from_id(call_id).get(timeout=5)
        except TimeoutError:
            print(f"{cfg}: todavía corriendo")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"{cfg}: ERROR {type(exc).__name__}: {str(exc)[:300]}")
            continue
        for n, b in res["previews"].items():
            (d / n).write_bytes(b)
        rows += res["rows"]
        ok = [r for r in res["rows"] if not r["error"]]
        print(f"{cfg}: {len(ok)}/{len(res['rows'])} imágenes, carga {res['meta']['load_seconds']} s, "
              f"{sum(r['seconds'] for r in ok) / max(1, len(ok)):.1f} s/imagen, "
              f"VRAM pico {max((r['peak_vram_gb'] for r in ok), default=0)} GB")
        errs = {r["error"] for r in res["rows"] if r["error"]}
        for e in list(errs)[:3]:
            print("   error:", e)
    (d / "rows.json").write_text(json.dumps(rows, indent=1))
    inputs = Path(out) / "inputs"
    configs = list(dict.fromkeys(r["config"] for r in rows))
    cell = (230, 290)
    for photo in PRODUCTS:
        lines = [(c, k) for c in configs for k in ("full", "cut")]
        if not lines:
            break
        g = Image.new("RGB", (cell[0] * 6, cell[1] * len(lines) + 18), "white")
        dr = ImageDraw.Draw(g)
        for j, s in enumerate(["entrada", *STYLES]):
            dr.text((j * cell[0] + 4, 3), s, fill="black")
        for i, (c, k) in enumerate(lines):
            files = [inputs / f"{photo}__{k}.png"] + [d / f"{photo}__{c}__{k}__{s}.jpg" for s in STYLES]
            for j, f in enumerate(files):
                if f.exists():
                    im = Image.open(f).convert("RGB")
                    im.thumbnail(cell)
                    g.paste(im, (j * cell[0] + (cell[0] - im.width) // 2, 18 + i * cell[1] + (cell[1] - im.height) // 2))
            dr.rectangle((0, 18 + i * cell[1], 70, 18 + i * cell[1] + 12), fill="white")
            dr.text((3, 18 + i * cell[1]), f"{c} {k}", fill="red")
        g.save(d / f"grilla__{photo}.jpg", quality=88)
    print(f"Resultados en {d}")
