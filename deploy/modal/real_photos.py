"""Experimento del modo intermedio con fotos caseras reales (Hito 2, P3).

App aparte de `worker.py` (no toca `Worker` ni `Lab`). Reproduce el flujo de la API sobre
cada foto: `prepare_upload` (lado mayor 1024), `rembg` (G13), inpainting y recomposición
(G8). Para cada configuración y estilo mide:

- expansión de silueta: la misma métrica de `Lab` (primer plano nuevo en la franja de 4 a
  40 px alrededor del producto, dividido por el área del producto);
- cumplimiento del estilo con CLIP (ViT-L/14): similitud con cinco descripciones paralelas
  del ENTORNO de cada preset; `clip_margin` = similitud con su estilo menos la mayor de los
  otros cuatro (> 0: CLIP reconoce el estilo pedido). Las fotos originales van como control;
- deriva del producto antes de recomponer (G8).

    PYTHONPATH=backend uv run --with modal modal run deploy/modal/real_photos.py \
        --photos data/fotos --configs A,G,I,F --seed 500

Las imágenes completas quedan en el volumen (`experiments_real/<RUN_ID>`); al local
bajan en JPEG a `data/experiments/real/<RUN_ID>/` junto con `metrics.jsonl` y una grilla
por foto. `data/` no se versiona.
"""

from __future__ import annotations

import io
import json
import subprocess
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
CACHE = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Mismas versiones que deploy/modal/worker.py
        "torch==2.5.1",
        "diffusers==0.31.0",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "safetensors==0.4.5",
        "huggingface_hub==0.26.2",
        "peft==0.13.2",
        "fastapi==0.162.0",
        "pydantic==2.14.2",
        "pydantic-settings==2.16.2",
        "pillow==12.3.0",
        "httpx==0.28.1",
        "rembg==2.0.59",
        "onnxruntime==1.19.2",
    )
    .env({"HF_HOME": f"{CACHE}/hf", "U2NET_HOME": f"{CACHE}/u2net", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_dir(ROOT / "backend" / "productshot", "/root/productshot")
)

app = modal.App("productshot-real", image=image)
cache = modal.Volume.from_name("productshot-hf-cache", create_if_missing=True)

# engine: "four" = SDXL base (4 canales); "nine" = UNet de inpainting de 9 canales.
CONFIGS = {
    "A": {"engine": "four", "strength": 0.99},  # actual
    "G": {"engine": "nine", "strength": 0.85},  # mejor expansión en fotos sintéticas
    "I": {"engine": "nine", "strength": 0.90},
    "J": {"engine": "nine", "strength": 0.95},
    "F": {"engine": "nine", "strength": 0.99},
    "K": {"engine": "nine", "strength": 1.0},
    # Nueve canales con el FONDO de la entrada reemplazado antes de generar: con strength < 1
    # los latentes iniciales parten de la foto, y el fondo original (pared, mantel) ancla los
    # colores. "palette" lo cambia por el degradado del preset; "blend" mezcla 50 % del fondo
    # original muy difuminado (conserva el plano de la mesa) con ese degradado.
    "P": {"engine": "nine", "strength": 0.95, "prefill": "palette"},
    "Q": {"engine": "nine", "strength": 0.90, "prefill": "blend"},
    "R": {"engine": "nine", "strength": 0.95, "prefill": "blend"},
    # R y Q con prompts de escena por categoría (SCENE_STYLE / CATEGORIES).
    "S": {"engine": "nine", "strength": 0.95, "prefill": "blend", "scene": True},
    "T": {"engine": "nine", "strength": 0.90, "prefill": "blend", "scene": True},
    # Checkpoint fotorrealista de inpainting (RealVisXL V4.0, 9 canales, openrail++) con la receta R.
    "U": {"engine": "realvis", "strength": 0.95, "prefill": "blend"},
    "V": {"engine": "realvis", "strength": 0.95, "prefill": "blend", "scene": True},
    "W": {"engine": "realvis", "strength": 0.90, "prefill": "blend", "scene": True},
}
REALVIS_MODEL_ID = "OzzyGT/RealVisXL_V4.0_inpainting"
ENGINE_ORDER = ["four", "nine", "realvis"]


# Prompts con escena según la CATEGORÍA del producto (experimento; `presets.py` no cambia).
# El estilo fija luz, superficie y ánimo sin utilería; la categoría elige el lugar y la
# utilería de `lifestyle`, para que un hub no termine en una cocina con pan.
SCENE_STYLE = {
    "studio": "professional product photography, seamless light grey paper backdrop, even softbox lighting, "
              "soft shadow under the product, clean catalog shot",
    "natural": "product photo outdoors on a natural stone surface, soft daylight, green foliage softly blurred "
               "in the background",
    "warm": "product photo on a warm wooden surface, golden hour sunlight, cozy amber tones, soft bokeh background",
    "lifestyle": "lifestyle product photo on a {setting} in a bright modern home, {props} softly blurred in the "
                 "background, shallow depth of field",
    "minimal": "minimalist product photo, plain off-white background, large negative space, no props, "
               "single soft shadow",
}
CATEGORIES = {
    "tecnologia": {"setting": "clean home office desk", "props": "a laptop and a notebook"},
    "hogar": {"setting": "living room side table", "props": "a book and a small plant"},
    "cocina": {"setting": "kitchen breakfast table", "props": "a coffee pot and a small plant"},
    "juguetes": {"setting": "bedroom desk", "props": "colorful books and a small plant"},
}
SCENE_NEGATIVE = (
    "text, watermark, logo, letters, people, hands, faces, duplicate product, extra objects on product, "
    "light stands, softbox, tripod, studio equipment, camera, food, fruit, bread, "
    "blurry, low quality, distorted, nsfw"
)


def scene_prompt(style: str, category: str) -> str:
    return SCENE_STYLE[style].format(**CATEGORIES[category])


def prefill_background(working, mask, style, how: str):
    """Reemplaza el fondo (blanco en la máscara) por el degradado del preset o una mezcla."""
    from PIL import Image, ImageFilter

    from productshot.services.presets import PRESETS

    top, bottom = PRESETS[style].mock_colors
    w, h = working.size
    grad = Image.new("RGB", (1, h))
    for y in range(h):
        t = y / max(1, h - 1)
        grad.putpixel((0, y), tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)))
    grad = grad.resize((w, h))
    if how == "blend":
        grad = Image.blend(working.filter(ImageFilter.GaussianBlur(24)), grad, 0.5)
    return Image.composite(grad, working, mask.convert("L"))

# Descripciones paralelas del ENTORNO de cada preset (sin producto ni adjetivos de calidad).
CLIP_TEXTS = {
    "studio": "a photo of an object in front of a seamless light grey studio backdrop",
    "natural": "a photo of an object on a stone surface outdoors with green leaves in the background",
    "warm": "a photo of an object on a wooden surface in warm golden amber light",
    "lifestyle": "a photo of an object on a kitchen table in a bright modern home with props in the background",
    "minimal": "a photo of an object on a plain off-white background with empty space and no props",
}
CLIP_MODEL = "openai/clip-vit-large-patch14"


@app.cls(gpu="L4", volumes={CACHE: cache}, timeout=3600, scaledown_window=60, max_containers=1)
class RealLab:
    @modal.method()
    def run(self, photos: dict[str, bytes], configs: list[str], seed: int, run_id: str,
            code_version: dict, categories: dict[str, str] | None = None) -> dict:
        import gc
        import importlib.metadata as md

        import torch
        from PIL import Image, ImageChops, ImageFilter

        from productshot.config import Settings
        from productshot.domain import Mode, Style
        from productshot.services import guardrails
        from productshot.services.generation import GenerationRequest, VariationSpec
        from productshot.services.images import prepare_upload
        from productshot.services.rembg_segmentation import RembgSegmentationService
        from productshot.services.sdxl_generation import (
            DEFAULT_MODEL_ID,
            NINE_CHANNEL_MODEL_ID,
            SdxlInpaintBackend,
            SdxlParams,
        )

        from productshot.services import sdxl_generation as sdxl_mod

        original_prompt_fns = (sdxl_mod.build_prompt, sdxl_mod.NEGATIVE_PROMPT)
        settings = Settings(_env_file=None)
        run_dir = Path(CACHE) / "experiments_real" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        styles = [s.value for s in Style]
        previews: dict[str, bytes] = {}

        def save_png(name: str, img) -> None:
            img.save(run_dir / name, format="PNG")

        def preview(name: str, img) -> None:
            small = img.convert("RGB").copy()
            small.thumbnail((768, 768))
            buf = io.BytesIO()
            small.save(buf, format="JPEG", quality=90)
            previews[name] = buf.getvalue()

        def binary(mask_l):
            # 255 = producto (la máscara del proyecto tiene el producto en negro).
            return mask_l.point(lambda v: 255 if v < 128 else 0)

        segmenter = RembgSegmentationService()
        segmenter.load()

        # 1) Preparación igual que la API, G12 y G13.
        prepared = {}
        for name, data in photos.items():
            prep = prepare_upload(data, max_bytes=settings.max_upload_bytes,
                                  max_pixels=settings.max_image_pixels, work_max_side=settings.work_max_side)
            working = prep.working
            mask = segmenter.segment(working).mask
            product = binary(mask)
            mm = guardrails.mask_metrics(mask, settings)
            pm = guardrails.photo_metrics(working)
            prepared[name] = {
                "working": working, "mask": mask,
                "area": max(1, sum(1 for v in product.getdata() if v)),
                "band": ImageChops.subtract(product.filter(ImageFilter.MaxFilter(81)),
                                            product.filter(ImageFilter.MaxFilter(9))),
                "info": {
                    "size": list(working.size), "mask": mm, "photo": pm,
                    "g13": [w.code for w in guardrails.mask_warnings(mm, settings)],
                    "g12": [w.code for w in guardrails.photo_warnings(pm, settings)],
                },
            }
            save_png(f"{name}__0_foto.png", working)
            save_png(f"{name}__1_mascara.png", mask)
            preview(f"{name}__0_foto.jpg", working)
            preview(f"{name}__1_mascara.jpg", mask)

        manifest = {
            "run_id": run_id, "configs": {c: CONFIGS[c] for c in configs}, "seed": seed,
            "seed_rule": "el estilo i (0..4) usa seed + i; mismas semillas en todas las configuraciones",
            "styles": styles, "categories": categories, "scene_style": SCENE_STYLE,
            "scene_categories": CATEGORIES, "scene_negative": SCENE_NEGATIVE, "clip_model": CLIP_MODEL, "clip_texts": CLIP_TEXTS,
            "photos": {n: p["info"] for n, p in prepared.items()},
            "packages": {p: md.version(p) for p in ("torch", "diffusers", "transformers", "rembg")},
            "gpu": torch.cuda.get_device_name(0), "code_version": code_version, "revisions": {},
        }

        def write_manifest():
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
            cache.commit()

        write_manifest()

        # 2) Generación: primero cuatro canales, después nueve (un modelo en GPU a la vez).
        rows = []
        finals = []  # (row index, imagen final) para CLIP
        specs = tuple(VariationSpec(position=i + 1, style=Style(st), seed=seed + i) for i, st in enumerate(styles))
        order = sorted(configs, key=lambda c: ENGINE_ORDER.index(CONFIGS[c]["engine"]))
        backend = None
        current_engine = None
        for cfg_name in order:
            cfg = CONFIGS[cfg_name]
            if cfg["engine"] != current_engine:
                backend = None
                gc.collect()
                torch.cuda.empty_cache()
                model_id = {"four": DEFAULT_MODEL_ID, "nine": NINE_CHANNEL_MODEL_ID,
                            "realvis": REALVIS_MODEL_ID}[cfg["engine"]]
                backend = SdxlInpaintBackend(SdxlParams(model_id=model_id))
                backend.load()
                cache.commit()
                current_engine = cfg["engine"]
                manifest["revisions"][current_engine] = backend.model_revision
                write_manifest()
            backend.params = SdxlParams(model_id=backend.params.model_id, strength=cfg["strength"])
            categories = categories or {}
            for name, p in prepared.items():
                # Escena por categoría: el backend arma el prompt con `build_prompt` y usa
                # NEGATIVE_PROMPT de su módulo; solo en este contenedor se sustituyen.
                if cfg.get("scene"):
                    category = categories.get(name, "hogar")
                    sdxl_mod.build_prompt = lambda style, adj, _c=category: scene_prompt(style.value, _c)
                    sdxl_mod.NEGATIVE_PROMPT = SCENE_NEGATIVE
                else:
                    sdxl_mod.build_prompt, sdxl_mod.NEGATIVE_PROMPT = original_prompt_fns
                if cfg.get("prefill"):
                    arts = []
                    for spec in specs:
                        filled = prefill_background(p["working"], p["mask"], spec.style, cfg["prefill"])
                        arts += backend.generate(GenerationRequest(
                            session_id="real", mode=Mode.INTERMEDIATE, image=filled, mask=p["mask"],
                            specs=(spec,)))
                else:
                    arts = backend.generate(GenerationRequest(
                        session_id="real", mode=Mode.INTERMEDIATE, image=p["working"], mask=p["mask"], specs=specs))
                for art in arts:
                    final, fid = guardrails.enforce_fidelity(p["working"], p["mask"], art.image, settings)
                    fg = binary(segmenter.segment(final).mask)
                    grown = sum(1 for v in ImageChops.multiply(fg, p["band"]).getdata() if v)
                    img_name = f"{name}__{cfg_name}__{art.style.value}__s{art.seed}"
                    save_png(img_name + ".png", final)
                    preview(img_name + ".jpg", final)
                    row = {
                        "photo": name, "config": cfg_name, "engine": cfg["engine"], "strength": cfg["strength"],
                        "style": art.style.value, "seed": art.seed,
                        "expansion": round(grown / p["area"], 3), "expansion_px": grown, "area_px": p["area"],
                        "drift": fid["protected_mean_abs_diff"], "seconds": art.params["seconds"],
                        "peak_vram_gb": art.params["peak_vram_gb"], "prompt": art.params["prompt"],
                        "image": img_name + ".png",
                    }
                    rows.append(row)
                    finals.append((len(rows) - 1, final))
                    with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                cache.commit()
        backend = None
        sdxl_mod.build_prompt, sdxl_mod.NEGATIVE_PROMPT = original_prompt_fns
        gc.collect()
        torch.cuda.empty_cache()

        # 3) CLIP: cumplimiento del estilo (y las fotos originales como control).
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(CLIP_MODEL, torch_dtype=torch.float16).to("cuda").eval()
        proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
        cache.commit()
        names = list(CLIP_TEXTS)
        with torch.no_grad():
            t = proc(text=[CLIP_TEXTS[n] for n in names], return_tensors="pt", padding=True).to("cuda")
            text_emb = model.get_text_features(**t)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            def sims(img):
                px = proc(images=img.convert("RGB"), return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
                e = model.get_image_features(pixel_values=px)
                e = e / e.norm(dim=-1, keepdim=True)
                return {n: round(float(v), 4) for n, v in zip(names, (e @ text_emb.T)[0])}

            for idx, img in finals:
                s = sims(img)
                target = rows[idx]["style"]
                rows[idx]["clip"] = s
                rows[idx]["clip_margin"] = round(s[target] - max(v for k, v in s.items() if k != target), 4)
                rows[idx]["clip_top"] = max(s, key=s.get)
            manifest["clip_control"] = {n: sims(p["working"]) for n, p in prepared.items()}

        (run_dir / "metrics_final.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
        write_manifest()
        return {"manifest": manifest, "rows": rows, "previews": previews}


def _code_version() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT, check=True).stdout

    try:
        dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
        return {"commit": git("rev-parse", "HEAD").strip(), "dirty": dirty,
                "diff": git("diff", "HEAD") if dirty else None}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "error": str(exc)}


def _summary(rows: list[dict]) -> str:
    from collections import defaultdict
    from statistics import mean, median

    by = defaultdict(list)
    for r in rows:
        by[r["config"]].append(r)
    out = ["config | expansión media (mediana) | margen CLIP medio | estilo reconocido | deriva G8 media | s/imagen"]
    for c, rs in by.items():
        ex = [r["expansion"] for r in rs]
        out.append(
            f"{c} | {mean(ex):.3f} ({median(ex):.3f}) | {mean(r['clip_margin'] for r in rs):+.4f} | "
            f"{sum(r['clip_top'] == r['style'] for r in rs)}/{len(rs)} | {mean(r['drift'] for r in rs):.1f} | "
            f"{mean(r['seconds'] for r in rs):.1f}"
        )
    return "\n".join(out)


def _grids(out_dir: Path, rows: list[dict], configs: list[str]) -> None:
    """Una grilla por foto: filas = configuraciones, columnas = foto, máscara y cinco estilos."""
    from PIL import Image, ImageDraw

    cell = 256
    styles = ["studio", "natural", "warm", "lifestyle", "minimal"]
    for photo in sorted({r["photo"] for r in rows}):
        grid = Image.new("RGB", (cell * 7, cell * len(configs) + 24), "white")
        d = ImageDraw.Draw(grid)
        for j, label in enumerate(["foto", "máscara", *styles]):
            d.text((j * cell + 6, 6), label, fill="black")
        for i, c in enumerate(configs):
            files = [f"{photo}__0_foto.jpg", f"{photo}__1_mascara.jpg"]
            for st in styles:
                match = [r for r in rows if r["photo"] == photo and r["config"] == c and r["style"] == st]
                files.append(match[0]["image"].replace(".png", ".jpg") if match else None)
            for j, f in enumerate(files):
                if f and (out_dir / f).exists():
                    im = Image.open(out_dir / f).convert("RGB")
                    im.thumbnail((cell, cell))
                    grid.paste(im, (j * cell + (cell - im.width) // 2, 24 + i * cell + (cell - im.height) // 2))
            d.text((6, 24 + i * cell + 6), c, fill="red")
        grid.save(out_dir / f"grilla__{photo}.jpg", quality=88)


@app.local_entrypoint()
def main(photos: str = "data/fotos", configs: str = "A,G,I,F", seed: int = 500, out: str = "data/experiments/real",
         categories: str = ""):
    # p. ej. --categories "000_hub=tecnologia,001_candle=hogar"; sin categoría: "hogar"
    cats = dict(kv.split("=", 1) for kv in categories.split(",") if "=" in kv)
    bad = {k: v for k, v in cats.items() if v not in CATEGORIES}
    if bad:
        raise SystemExit(f"Categorías desconocidas: {bad}. Opciones: {list(CATEGORIES)}")
    cfgs = [c.strip() for c in configs.split(",") if c.strip()]
    unknown = [c for c in cfgs if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"Configuraciones desconocidas: {unknown}. Opciones: {list(CONFIGS)}")
    files = sorted(p for p in Path(photos).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise SystemExit(f"No hay fotos en {photos}")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    print(f"Corrida {run_id}: {len(files)} fotos × {len(cfgs)} configuraciones × 5 estilos")
    result = RealLab().run.remote({p.stem: p.read_bytes() for p in files}, cfgs, seed, run_id, _code_version(), cats)
    out_dir = Path(out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in result["previews"].items():
        (out_dir / name).write_bytes(data)
    (out_dir / "manifest.json").write_text(json.dumps(result["manifest"], indent=2, ensure_ascii=False, default=str))
    (out_dir / "metrics.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in result["rows"]))
    _grids(out_dir, result["rows"], sorted(cfgs, key=lambda c: ENGINE_ORDER.index(CONFIGS[c]["engine"])))
    print(_summary(result["rows"]))
    print(json.dumps({n: i for n, i in result["manifest"]["photos"].items()}, indent=1, ensure_ascii=False))
    print(f"Resultados en {out_dir}")
