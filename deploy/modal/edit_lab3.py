"""Ronda 3 del editor (2026-11-16): escenas con sentido para cada producto.

En las rondas 1 y 2, Qwen-Image-2.1 (y su Turbo) fue el que mejor conservó el producto: no
gira la taza ni cambia la vela. Pero las escenas eran plantillas genéricas, iguales para
todos los productos. Aquí se comparan dos formas de elegir la escena, con Qwen-Image-2.1-Turbo
(8 pasos):

- A ("curated"): una biblioteca de escenas escrita a mano por categoría y estilo.
- B ("director"): Qwen3-VL-8B-Instruct (Apache-2.0) mira la foto y escribe las cinco escenas
  para ESE producto, con reglas (props relevantes, sin personas ni texto, el producto no cambia).
  La foto no sale del worker.

Estudio y minimalista usan la misma descripción en A y B (no tienen contexto que inventar).
La instrucción agrega "misma orientación y ángulo; no rotar, no espejar, no agregar partes".

    PYTHONPATH=backend uv run --with modal modal deploy deploy/modal/edit_lab3.py
"""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

import modal

CACHE = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "diffusers==0.43.0",
        "transformers==5.23.0",
        "accelerate==1.15.0",
        "sentencepiece==0.2.3",
        "protobuf",
        "safetensors",
        "pillow==12.3.0",
        "qwen-vl-utils==0.0.14",
    )
    .env({"HF_HOME": f"{CACHE}/hf", "HF_HUB_DISABLE_TELEMETRY": "1"})
)
app = modal.App("productshot-edit3", image=image)
cache = modal.Volume.from_name("productshot-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

GENERATOR_ID = "Qwen/Qwen-Image-2.1-Turbo"
DIRECTOR_ID = "Qwen/Qwen3-VL-8B-Instruct"
STYLES = ("studio", "natural", "warm", "lifestyle", "minimal")

FIXED = {
    "studio": "Studio shot on a seamless light grey paper backdrop, soft diffused softbox light from the left, "
              "gentle contact shadow, nothing else in the frame",
    "minimal": "Minimalist shot on a plain matte off-white surface against a plain off-white wall, generous empty "
               "space around the product, one soft natural shadow",
}
CURATED = {
    "tecnologia": {
        "natural": "On a light oak desk beside a large window, soft morning daylight, a small succulent far in the "
                   "background, clean and airy",
        "warm": "On a dark walnut desk in a cozy evening workspace, warm desk-lamp glow, soft amber bokeh from "
                "string lights far behind",
        "lifestyle": "On a tidy minimalist desk setup, a closed laptop and a keyboard softly out of focus behind it, "
                     "bright natural light",
    },
    "hogar": {
        "natural": "On a natural linen cloth beside a vase of dried eucalyptus and pampas grass, soft diffused window "
                   "daylight, calm neutral tones",
        "warm": "On a rustic wooden tray next to a chunky wool blanket, a few lit tea lights softly blurred behind, "
                "warm cozy evening glow",
        "lifestyle": "On a shelf in a modern spa-like bathroom, rolled white towels and a small plant softly out of "
                     "focus behind",
    },
    "cocina": {
        "natural": "On a wooden breakfast table by a sunny window, a small plate with a croissant and a few coffee "
                   "beans softly out of focus behind",
        "warm": "On a rustic wooden kitchen counter in warm early-morning sunlight, a folded linen napkin and coffee "
                "beans nearby, amber tones",
        "lifestyle": "On a bright Scandinavian kitchen table, a French press and a small plant softly out of focus in "
                     "the background",
    },
    "juguetes": {
        "natural": "On a white shelf next to a window in soft daylight, a few colorful books softly blurred behind",
        "warm": "On a wooden bedroom shelf, warm fairy lights softly glowing far in the background",
        "lifestyle": "On a tidy collector's display shelf with soft LED accent lighting, other shelves softly out of "
                     "focus behind",
    },
}
PRODUCTS = {  # foto -> (sustantivo en inglés, categoría)
    "000_hub": ("USB-C hub", "tecnologia"),
    "001_candle": ("scented candle in a black glass jar", "hogar"),
    "002_cup": ("printed ceramic mug", "cocina"),
    "003_printer": ("pink mini thermal printer next to its retail box", "tecnologia"),
    "007_hub_en_empaque": ("USB hub in its open retail packaging", "tecnologia"),
    "008_nendoroid": ("anime collectible figure on its stand", "juguetes"),
    "014_quemador_incienso": ("ceramic backflow incense burner", "hogar"),
    "015_caja_ssd_ugreen": ("NVMe SSD enclosure in its retail box", "tecnologia"),
    "016_caja_ssd_wd": ("NVMe SSD in its retail box", "tecnologia"),
    "017_esfera_marmol": ("polished marble sphere", "hogar"),
    "019_capibara": ("capybara-shaped desk fan", "juguetes"),
    "021_lata_en_mano": ("canned milk soda drink", "cocina"),
}

KEEP = ("Keep the product identical to the original photo: same shape, proportions, colors, logo and every "
        "printed word and number exactly as shown. Keep the same orientation and viewing angle of the product; do "
        "not rotate or mirror it and do not add or remove parts. The product is the only hero object; no people, "
        "no hands. Photorealistic, sharp focus on the product, high-end commercial product photography.")

DIRECTOR_PROMPT = """You are the art director of an e-commerce product photo studio.
Look only at the main product in this photo and ignore its messy background.
Write one scene for a professional photo that would help sell THIS product on Instagram, for each style:
- studio: seamless studio backdrop and studio light
- natural: soft daylight and natural materials
- warm: cozy warm light
- lifestyle: a realistic place where this exact product is used, with one or two relevant props far in the background
- minimal: plain surface and plain wall, lots of empty space
Rules: the product itself does not change and stays the only hero object; props must suit the product and never
touch or hide it; no people, hands, animals, text, signs or other products of the same kind. Each scene is ONE
sentence of at most 35 words describing surface, background, light and mood.
Also give the product name in English (2 to 8 words).
Answer ONLY with JSON: {"product": "...", "scenes": {"studio": "...", "natural": "...", "warm": "...",
"lifestyle": "...", "minimal": "..."}}"""


def instruction(noun: str, scene: str) -> str:
    return f"Turn this into a professional e-commerce photo to sell this exact {noun}. {scene}. {KEEP}"


@app.cls(gpu="H100", volumes={CACHE: cache}, secrets=[hf_secret], timeout=3600, scaledown_window=60,
         max_containers=1)
class Round3:
    @modal.method()
    def run(self, photos: dict[str, bytes], run_id: str, seeds: list[int]) -> dict:
        import torch
        from PIL import Image

        run_dir = Path(CACHE) / "experiments_edit" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        imgs = {n: Image.open(io.BytesIO(b)).convert("RGB") for n, b in photos.items()}

        # 1) Director de arte (B): una pasada por foto.
        from transformers import AutoModelForImageTextToText, AutoProcessor

        t0 = time.perf_counter()
        proc = AutoProcessor.from_pretrained(DIRECTOR_ID)
        vlm = AutoModelForImageTextToText.from_pretrained(DIRECTOR_ID, torch_dtype=torch.bfloat16).to("cuda").eval()
        director, raw = {}, {}
        for n, img in imgs.items():
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": DIRECTOR_PROMPT}]}]
            inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                              return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = vlm.generate(**inputs, max_new_tokens=700, do_sample=False)
            text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            raw[n] = text
            try:
                data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
                assert all(isinstance(data["scenes"][s], str) and data["scenes"][s].strip() for s in STYLES)
                director[n] = {"product": str(data.get("product") or PRODUCTS[n][0]), "scenes": data["scenes"]}
            except Exception as exc:  # noqa: BLE001 - se registra y se usa la biblioteca curada
                director[n] = None
                raw[n] += f"\n[parse error: {exc}]"
        director_s = round(time.perf_counter() - t0, 1)
        del vlm
        torch.cuda.empty_cache()
        (run_dir / "director.json").write_text(json.dumps({"parsed": director, "raw": raw}, indent=1))
        cache.commit()

        # 2) Generador.
        from diffusers import QwenImage21Pipeline

        pipe = QwenImage21Pipeline.from_pretrained(GENERATOR_ID, torch_dtype=torch.bfloat16).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        previews, rows = {}, []
        for n, src in imgs.items():
            noun, cat = PRODUCTS[n]
            scale = (1024 * 1024 / (src.width * src.height)) ** 0.5
            w, h = round(src.width * scale / 16) * 16, round(src.height * scale / 16) * 16
            src = src.resize((w, h), Image.Resampling.LANCZOS)
            for mode in ("A", "B"):
                d = director[n] if mode == "B" else None
                for i, style in enumerate(STYLES):
                    if mode == "B" and d:
                        scene, prod = d["scenes"][style].rstrip(". "), d["product"]
                    else:
                        scene = FIXED.get(style) or CURATED[cat][style]
                        prod = noun
                    if mode == "B" and not d:
                        mode_used = "A(fallback)"
                    else:
                        mode_used = mode
                    prompt = instruction(prod, scene)
                    for seed in seeds:
                        t1 = time.perf_counter()
                        out = pipe(prompt=prompt, image=src, num_inference_steps=8,
                                   generator=torch.Generator("cuda").manual_seed(seed + i)).images[0]
                        base = f"{n}__{mode}__{style}__s{seed}"
                        out.save(run_dir / f"{base}.png")
                        small = out.convert("RGB").copy()
                        small.thumbnail((768, 768))
                        buf = io.BytesIO()
                        small.save(buf, format="JPEG", quality=90)
                        previews[f"{base}.jpg"] = buf.getvalue()
                        rows.append({"photo": n, "mode": mode_used, "style": style, "seed": seed + i,
                                     "seconds": round(time.perf_counter() - t1, 2), "prompt": prompt})
                cache.commit()
        meta = {"generator": GENERATOR_ID, "director": DIRECTOR_ID, "director_seconds": director_s,
                "gpu": torch.cuda.get_device_name(0), "seeds": seeds}
        (run_dir / "rows.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
        cache.commit()
        return {"meta": meta, "rows": rows, "director": director, "raw": raw, "previews": previews}


@app.local_entrypoint()
def collect(call_id: str, run_id: str, out: str = "data/experiments/edit"):
    from PIL import Image, ImageDraw

    res = modal.FunctionCall.from_id(call_id).get(timeout=10)
    d = Path(out) / run_id
    d.mkdir(parents=True, exist_ok=True)
    for n, b in res["previews"].items():
        (d / n).write_bytes(b)
    (d / "director.json").write_text(json.dumps({"parsed": res["director"], "raw": res["raw"]}, indent=1))
    (d / "rows.json").write_text(json.dumps(res["rows"], indent=1))
    seeds = res["meta"]["seeds"]
    cell = (220, 280)
    for photo in PRODUCTS:
        lines = [(m, s) for m in ("A", "B") for s in seeds]
        g = Image.new("RGB", (cell[0] * 6, cell[1] * len(lines) + 18), "white")
        dr = ImageDraw.Draw(g)
        for j, s in enumerate(["entrada", *STYLES]):
            dr.text((j * cell[0] + 4, 3), s, fill="black")
        for i, (m, seed) in enumerate(lines):
            files = [Path(out) / "inputs" / f"{photo}__full.png"] + [d / f"{photo}__{m}__{s}__s{seed}.jpg" for s in STYLES]
            for j, f in enumerate(files):
                if f.exists():
                    im = Image.open(f).convert("RGB")
                    im.thumbnail(cell)
                    g.paste(im, (j * cell[0] + (cell[0] - im.width) // 2, 18 + i * cell[1] + (cell[1] - im.height) // 2))
            dr.rectangle((0, 18 + i * cell[1], 70, 30 + i * cell[1]), fill="white")
            dr.text((3, 18 + i * cell[1]), f"{m} s{seed}", fill="red")
        g.save(d / f"grilla__{photo}.jpg", quality=86)
    print(json.dumps(res["meta"], indent=1))
    print("director:", {k: (v["product"] if v else None) for k, v in res["director"].items()})
    print(f"Resultados en {d}")
