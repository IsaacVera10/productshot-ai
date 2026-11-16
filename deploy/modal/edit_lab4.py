"""Ronda 4 del editor (2026-11-16): varias candidatas por estilo, verificación y selección.

Sobre la ronda 3 (Qwen-Image-2.1-Turbo, escenas A curadas y B del director de arte):

- Cuatro semillas por estilo y por modo de escena.
- Verificación automática de cada candidata, a la resolución de trabajo:
    * fidelidad del producto: similitud coseno DINOv2 (facebook/dinov2-base, Apache-2.0) entre el
      recorte del producto en la foto original (caja de la máscara BiRefNet) y la misma caja en la
      candidata. Qwen conserva el encuadre, así que la caja queda alineada;
    * cambio de escena: diferencia media (0 a 255) del FONDO entre la candidata y la foto original,
      a 128 px. Atrapa la falla de la ronda 3 en que el modelo devolvía casi la foto original.
- Se elige por estilo la candidata con mayor fidelidad entre las que sí cambiaron la escena.
- El director v2 recibe la categoría que elige el vendedor y reglas nuevas: reemplazar todo el
  fondo, no encender ni modificar el producto, y variar la utilería.

    PYTHONPATH=backend uv run --with modal modal deploy deploy/modal/edit_lab4.py
"""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent if modal.is_local() else Path("/root")
CACHE = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0", "torchvision==0.23.0", "diffusers==0.43.0", "transformers==5.23.0",
        "accelerate==1.15.0", "sentencepiece==0.2.3", "protobuf", "safetensors", "pillow==12.3.0",
        "qwen-vl-utils==0.0.14",
    )
    .env({"HF_HOME": f"{CACHE}/hf", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_file(HERE / "edit_lab3.py", "/root/edit_lab3.py")
)
app = modal.App("productshot-edit4", image=image)
cache = modal.Volume.from_name("productshot-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

DINO_ID = "facebook/dinov2-base"
MIN_SCENE_CHANGE = 25.0  # diferencia media del fondo (0-255); por debajo = "no cambió la escena"

DIRECTOR_V2 = """You are the art director of an e-commerce product photo studio.
Look only at the main product in this photo and ignore its messy background.
The seller says the product category is: {category}.
Write one scene for a professional photo that would help sell THIS product on Instagram, for each style:
- studio: seamless studio backdrop and studio light
- natural: soft daylight and natural materials
- warm: cozy warm light
- lifestyle: a realistic place where this exact product is used, with one or two props that fit it, far in the background
- minimal: plain surface and plain wall, lots of empty space
Rules:
- The original background and surface are completely replaced; nothing of the original room remains.
- The product itself does not change: do not light it, open it, switch it on or add parts. It stays the only hero object.
- Props must suit THIS product, never touch or hide it, and must vary between scenes. Avoid mugs, books and
  bookshelves unless they truly fit the product.
- No people, hands, animals, text, signs or other products of the same kind.
- Each scene is ONE sentence of at most 35 words describing surface, background, light and mood.
Also give the product name in English (2 to 8 words).
Answer ONLY with JSON: {{"product": "...", "scenes": {{"studio": "...", "natural": "...", "warm": "...",
"lifestyle": "...", "minimal": "..."}}}}"""

REPLACE = "Replace the entire original background and surface; nothing from the original surroundings remains."


@app.cls(gpu="H100", volumes={CACHE: cache}, secrets=[hf_secret], timeout=3600, scaledown_window=60,
         max_containers=1)
class Round4:
    @modal.method()
    def run(self, photos: dict[str, dict[str, bytes]], run_id: str, seeds: list[int]) -> dict:
        import sys

        import torch
        import torch.nn.functional as F
        from PIL import Image, ImageChops, ImageStat

        sys.path.insert(0, "/root")
        from edit_lab3 import CURATED, FIXED, KEEP, PRODUCTS, STYLES

        run_dir = Path(CACHE) / "experiments_edit" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        def work_size(img):
            s = (1024 * 1024 / (img.width * img.height)) ** 0.5
            return round(img.width * s / 16) * 16, round(img.height * s / 16) * 16

        data = {}
        for n, enc in photos.items():
            src = Image.open(io.BytesIO(enc["full"])).convert("RGB")
            size = work_size(src)
            src = src.resize(size, Image.Resampling.LANCZOS)
            mask = Image.open(io.BytesIO(enc["mask"])).convert("L").resize(size, Image.Resampling.BILINEAR)
            prod = mask.point(lambda v: 255 if v < 128 else 0)  # 255 = producto
            x0, y0, x1, y1 = prod.getbbox()
            px, py = int((x1 - x0) * 0.05), int((y1 - y0) * 0.05)
            box = (max(0, x0 - px), max(0, y0 - py), min(size[0], x1 + px), min(size[1], y1 + py))
            data[n] = {"src": src, "bg": mask.point(lambda v: 255 if v >= 128 else 0), "box": box}

        # 1) Director v2.
        from transformers import AutoImageProcessor, AutoModel, AutoModelForImageTextToText, AutoProcessor

        t0 = time.perf_counter()
        proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        vlm = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-8B-Instruct",
                                                          torch_dtype=torch.bfloat16).to("cuda").eval()
        director, raw = {}, {}
        for n, d in data.items():
            cat = PRODUCTS[n][1]
            msgs = [{"role": "user", "content": [{"type": "image", "image": d["src"]},
                                                 {"type": "text", "text": DIRECTOR_V2.format(category=cat)}]}]
            inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                              return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = vlm.generate(**inputs, max_new_tokens=700, do_sample=False)
            text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            raw[n] = text
            try:
                j = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
                assert all(isinstance(j["scenes"][s], str) and j["scenes"][s].strip() for s in STYLES)
                director[n] = {"product": str(j.get("product") or PRODUCTS[n][0]), "scenes": j["scenes"]}
            except Exception as exc:  # noqa: BLE001
                director[n] = None
                raw[n] += f"\n[parse error: {exc}]"
        director_s = round(time.perf_counter() - t0, 1)
        del vlm
        torch.cuda.empty_cache()
        (run_dir / "director.json").write_text(json.dumps({"parsed": director, "raw": raw}, indent=1))
        cache.commit()

        # 2) Verificador DINOv2.
        dproc = AutoImageProcessor.from_pretrained(DINO_ID)
        dino = AutoModel.from_pretrained(DINO_ID).to("cuda").eval()

        def emb(img):
            with torch.no_grad():
                x = dproc(images=img, return_tensors="pt").to("cuda")
                return F.normalize(dino(**x).pooler_output, dim=-1)

        for d in data.values():
            d["ref"] = emb(d["src"].crop(d["box"]))

        def score(d, out):
            fid = float((emb(out.crop(d["box"])) @ d["ref"].T).item())
            a = out.convert("L").resize((128, 128))
            b = d["src"].convert("L").resize((128, 128))
            m = d["bg"].resize((128, 128))
            change = ImageStat.Stat(ImageChops.difference(a, b), mask=m).mean[0]
            return round(fid, 4), round(change, 1)

        # 3) Generación.
        from diffusers import QwenImage21Pipeline

        pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", torch_dtype=torch.bfloat16).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        rows, previews = [], {}
        for n, d in data.items():
            noun, cat = PRODUCTS[n]
            for mode in ("A", "B"):
                dd = director[n] if mode == "B" else None
                for i, style in enumerate(STYLES):
                    if dd:
                        scene, prod = dd["scenes"][style].rstrip(". "), dd["product"]
                    else:
                        scene, prod = FIXED.get(style) or CURATED[cat][style], noun
                    prompt = (f"Turn this into a professional e-commerce photo to sell this exact {prod}. {scene}. "
                              f"{REPLACE} {KEEP}")
                    for seed in seeds:
                        t1 = time.perf_counter()
                        out = pipe(prompt=prompt, image=d["src"], num_inference_steps=8,
                                   generator=torch.Generator("cuda").manual_seed(seed + i)).images[0].convert("RGB")
                        if out.size != d["src"].size:
                            out = out.resize(d["src"].size, Image.Resampling.LANCZOS)
                        fid, change = score(d, out)
                        base = f"{n}__{mode}__{style}__s{seed}"
                        out.save(run_dir / f"{base}.png")
                        small = out.copy()
                        small.thumbnail((768, 768))
                        buf = io.BytesIO()
                        small.save(buf, format="JPEG", quality=90)
                        previews[f"{base}.jpg"] = buf.getvalue()
                        rows.append({"photo": n, "mode": mode if (mode == "A" or dd) else "A(fallback)",
                                     "style": style, "seed": seed, "fidelity": fid, "scene_change": change,
                                     "seconds": round(time.perf_counter() - t1, 2), "prompt": prompt,
                                     "image": base})
                cache.commit()

        # 4) Selección: mayor fidelidad entre las que cambiaron la escena.
        best = {}
        for r in rows:
            key = (r["photo"], r["mode"][0], r["style"])
            ok = r["scene_change"] >= MIN_SCENE_CHANGE
            cand = (ok, r["fidelity"])
            if key not in best or cand > best[key][0]:
                best[key] = (cand, r)
        selected = [{"photo": k[0], "mode": k[1], "style": k[2], "passed_scene_check": v[0][0], **v[1]}
                    for k, v in best.items()]
        meta = {"generator": "Qwen/Qwen-Image-2.1-Turbo", "director": "Qwen/Qwen3-VL-8B-Instruct", "verifier": DINO_ID,
                "director_seconds": director_s, "seeds": seeds, "min_scene_change": MIN_SCENE_CHANGE,
                "gpu": torch.cuda.get_device_name(0)}
        (run_dir / "rows.json").write_text(json.dumps({"meta": meta, "rows": rows, "selected": selected}, indent=1))
        cache.commit()
        return {"meta": meta, "rows": rows, "selected": selected, "director": director, "raw": raw,
                "previews": previews}


@app.local_entrypoint()
def collect(call_id: str, run_id: str, out: str = "data/experiments/edit"):
    """Baja la ronda 4: imágenes, puntajes y dos grillas por producto (elegidas y todas)."""
    import sys
    from statistics import mean

    from PIL import Image, ImageDraw

    sys.path.insert(0, str(HERE))
    from edit_lab3 import STYLES

    res = modal.FunctionCall.from_id(call_id).get(timeout=10)
    d = Path(out) / run_id
    d.mkdir(parents=True, exist_ok=True)
    for n, b in res["previews"].items():
        (d / n).write_bytes(b)
    for k in ("rows", "selected", "director", "raw", "meta"):
        (d / f"{k}.json").write_text(json.dumps(res[k], indent=1))
    rows, sel = res["rows"], res["selected"]
    photos = list(dict.fromkeys(r["photo"] for r in rows))
    cell = (220, 280)
    for photo in photos:
        # Elegidas: fila A y fila B, con fidelidad y cambio de escena.
        g = Image.new("RGB", (cell[0] * 6, cell[1] * 2 + 18), "white")
        dr = ImageDraw.Draw(g)
        for j, s in enumerate(["entrada", *STYLES]):
            dr.text((j * cell[0] + 4, 3), s, fill="black")
        for i, mode in enumerate(("A", "B")):
            src = Path(out) / "inputs" / f"{photo}__full.png"
            if src.exists():
                im = Image.open(src).convert("RGB")
                im.thumbnail(cell)
                g.paste(im, ((cell[0] - im.width) // 2, 18 + i * cell[1]))
            for j, s in enumerate(STYLES):
                r = next((x for x in sel if x["photo"] == photo and x["mode"] == mode and x["style"] == s), None)
                if not r:
                    continue
                im = Image.open(d / f"{r['image']}.jpg").convert("RGB")
                im.thumbnail(cell)
                x, y = (j + 1) * cell[0] + (cell[0] - im.width) // 2, 18 + i * cell[1]
                g.paste(im, (x, y))
                tag = f"{mode} f={r['fidelity']:.2f} c={r['scene_change']:.0f}" + ("" if r["passed_scene_check"] else " !")
                dr.rectangle((x, y, x + 130, y + 12), fill="white")
                dr.text((x + 2, y), tag, fill="red")
        g.save(d / f"elegidas__{photo}.jpg", quality=86)
    print(json.dumps(res["meta"], indent=1))
    for mode in ("A", "B"):
        s = [x for x in sel if x["mode"] == mode]
        allm = [x for x in rows if x["mode"].startswith(mode)]
        print(f"{mode}: fidelidad media elegidas {mean(x['fidelity'] for x in s):.3f} | todas {mean(x['fidelity'] for x in allm):.3f}"
              f" | candidatas que no cambiaron la escena {sum(x['scene_change'] < res['meta']['min_scene_change'] for x in allm)}/{len(allm)}")
    print(f"Resultados en {d}")
