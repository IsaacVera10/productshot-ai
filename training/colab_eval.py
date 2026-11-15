"""Evaluación de los checkpoints del LoRA con val, dentro del VM de Colab (docs/10).

Lo lanza `training/run_colab.py --stage eval`, que sube `/content/job.json`, `/content/colab_job.py`
(se reutiliza su instalación), `requirements-colab.txt` y `/content/eval.tar` en partes con:
`val/` (foto, recorte RGBA y `eval.jsonl`), `train/` (las fotos de train con su estilo) y
`adapters/` (un `.safetensors` por checkpoint).

Para cada producto de val y cada estilo genera con el LoRA apagado y con cada checkpoint, con la
misma semilla, el mismo pipeline y los mismos parámetros que producción. La entrada es el recorte
sobre gris neutro y la máscara sale del alpha de BG60k, así que no incluye errores del
segmentador real. Cada imagen se recompone como en G8 y se mide:

- estilo: CLIP de la imagen recompuesta contra el centroide de train de cada estilo, calculado
  aquí mismo con el mismo modelo y preprocesado (coseno al pedido, margen frente al mejor de
  los otros y si gana el pedido);
- producto, antes de recomponer (diagnóstico): DINOv2 del recorte generado frente al de entrada
  y diferencia media dentro de la máscara.

Las métricas se calculan sobre la imagen recompuesta sin comprimir; el JPEG que se guarda es solo
para la revisión visual (la fidelidad exacta la garantiza G8 en producción).

Los productos van en orden aleatorio fijo y cada uno se guarda completo antes de pasar al
siguiente. El plazo se comprueba antes de cada generación: si vence, el producto en curso se
descarta y se empaqueta lo anterior, así los resultados parciales siguen pareados.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import sys
import tarfile
import time
from pathlib import Path

CONTENT = Path("/content")
EVAL = CONTENT / "eval"
OUT = CONTENT / "out"
IMAGES = OUT / "images"
OFF = "off"
BATCH = 64
DEADLINE = float("inf")


class OutOfTime(Exception):
    pass


def check_deadline(stage: str) -> None:
    if time.time() > DEADLINE:
        raise OutOfTime(stage)


def log(msg: str) -> None:
    print(f"[eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_for(item_id: int, style: str) -> int:
    return int(hashlib.sha256(f"{item_id}:{style}".encode()).hexdigest()[:8], 16)


def pack_view(image, side: int, quality: int):
    """Mismo preprocesado que `bg60k_pack.py`: reducción LANCZOS y JPEG."""
    from PIL import Image

    buf = io.BytesIO()
    image.convert("RGB").resize((side, side), Image.Resampling.LANCZOS).save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def product_crop(image, alpha, side: int, neutral: int, quality: int):
    """Mismo recorte que `bg60k_pack.product_cutout` (bbox con alpha > 127, como el manifiesto) y JPEG."""
    from PIL import Image

    gray = Image.new("RGB", image.size, (neutral,) * 3)
    bbox = alpha.point(lambda v: 255 if v > 127 else 0).getbbox()
    crop = Image.composite(image.convert("RGB"), gray, alpha).crop(bbox)
    square = Image.new("RGB", (max(crop.size),) * 2, (neutral,) * 3)
    square.paste(crop, ((square.width - crop.width) // 2, (square.height - crop.height) // 2))
    buf = io.BytesIO()
    square.resize((side, side), Image.Resampling.LANCZOS).save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


class Clip:
    def __init__(self, model_id: str, revision: str) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.model = CLIPModel.from_pretrained(model_id, revision=revision).cuda().eval()
        self.processor = CLIPProcessor.from_pretrained(model_id, revision=revision)

    def images(self, images: list):
        import numpy as np

        torch = self.torch
        chunks = []
        for k in range(0, len(images), BATCH):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                pixels = self.processor(images=images[k:k + BATCH], return_tensors="pt")["pixel_values"].to("cuda")
                embeds = self.model.visual_projection(self.model.vision_model(pixel_values=pixels).pooler_output)
            chunks.append(torch.nn.functional.normalize(embeds.float(), dim=-1).cpu().numpy())
        return np.concatenate(chunks)


class Dino:
    def __init__(self, model_id: str, revision: str) -> None:
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.model = AutoModel.from_pretrained(model_id, revision=revision).cuda().eval()
        self.mean = torch.tensor((0.485, 0.456, 0.406), device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device="cuda").view(1, 3, 1, 1)

    def images(self, images: list):
        import numpy as np

        torch = self.torch
        pixels = torch.from_numpy(np.stack([np.asarray(im) for im in images])).cuda()
        pixels = (pixels.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            pooled = self.model(pixel_values=pixels).pooler_output
        return torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy()


def unpack(cfg: dict) -> dict:
    tar_path = CONTENT / "eval.tar"
    with tar_path.open("wb") as dst:
        for k in range(cfg["eval_tar"]["parts"]):
            part = CONTENT / f"eval.tar.part{k:03d}"
            dst.write(part.read_bytes())
            part.unlink()
    if sha256(tar_path) != cfg["eval_tar"]["sha256"]:
        raise RuntimeError("eval.tar no coincide con el sha256 de la laptop.")
    with tarfile.open(tar_path) as tar:
        tar.extractall(EVAL, filter="data")
    for name, spec in cfg["adapters"].items():
        if sha256(EVAL / spec["file"]) != spec["sha256"]:
            raise RuntimeError(f"El adaptador {name} no coincide con su sha256.")
    return {"sha256": cfg["eval_tar"]["sha256"]}


def style_references(clip: Clip, cfg: dict) -> tuple[dict, dict]:
    """Centroides CLIP de train por estilo y cuánto aciertan con las fotos reales de val."""
    import numpy as np
    from PIL import Image

    pp = cfg["preprocess"]
    styles = list(cfg["style_prompts"])
    train = [json.loads(line) for line in (EVAL / "train" / "metadata.jsonl").open(encoding="utf-8")]
    embeds = clip.images([pack_view(Image.open(EVAL / "train" / r["file_name"]), pp["side"], pp["jpeg_quality"])
                          for r in train])
    centroids = np.stack([embeds[[r["style"] == s for r in train]].mean(axis=0) for s in styles])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    val = [json.loads(line) for line in (EVAL / "val" / "eval.jsonl").open(encoding="utf-8")]
    real = clip.images([pack_view(Image.open(EVAL / "val" / f"{r['id']}.png"), pp["side"], pp["jpeg_quality"])
                        for r in val])
    hits = [styles[int(np.argmax(c))] == r["style"] for c, r in zip(real @ centroids.T, val)]
    np.savez(OUT / "style_refs.npz", styles=np.asarray(styles), centroids=centroids,
             train_ids=np.asarray([int(Path(r["file_name"]).stem) for r in train]), train_clip=embeds.astype(np.float16),
             val_ids=np.asarray([r["id"] for r in val]), val_clip=real.astype(np.float16))
    summary = {"train_images": len(train), "val_real_top1": sum(hits), "val_real_n": len(hits),
               "centroid_cosine": np.round(centroids @ centroids.T, 4).tolist()}
    log(f"Referencias de estilo: {summary['val_real_top1']}/{len(hits)} fotos reales de val aciertan")
    return {s: c for s, c in zip(styles, centroids)}, summary


def load_pipeline(cfg: dict):
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        cfg["base_model"], revision=cfg["base_revision"], torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    for name, spec in cfg["adapters"].items():
        path = EVAL / spec["file"]
        pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=name)
    return pipe


def evaluate_product(pipe, clip: Clip, dino: Dino, centroids: dict, cfg: dict, row: dict) -> list[dict]:
    import numpy as np
    import torch
    from PIL import Image, ImageChops, ImageStat

    g, pp = cfg["generation"], cfg["preprocess"]
    cutout = Image.open(EVAL / "val" / f"{row['id']}_cutout.png").convert("RGBA")
    alpha = cutout.getchannel("A")
    working = Image.composite(cutout.convert("RGB"), Image.new("RGB", cutout.size, (pp["neutral"],) * 3), alpha)
    mask = alpha.point(lambda v: 255 - v)  # blanco = fondo editable, negro = producto
    protected = mask.point(lambda v: 255 if v < 128 else 0)
    scale = g["target_side"] / max(working.size)
    gen_size = tuple(max(8, round(v * scale / 8) * 8) for v in working.size)
    gen_image = working.resize(gen_size, Image.Resampling.LANCZOS)
    gen_mask = pipe.mask_processor.blur(mask.resize(gen_size, Image.Resampling.BILINEAR), blur_factor=g["mask_blur"])
    ref_crop = product_crop(working, alpha, pp["side"], pp["neutral"], pp["jpeg_quality"])
    styles = list(centroids)
    matrix = np.stack([centroids[s] for s in styles])

    rows, finals, crops = [], [], []
    for style in styles:
        seed = seed_for(row["id"], style)
        for config in [OFF, *cfg["adapters"]]:
            check_deadline(f"producto {row['id']}")
            if config == OFF:
                pipe.set_adapters([next(iter(cfg["adapters"]))], adapter_weights=[0.0])
            else:
                pipe.set_adapters([config], adapter_weights=[g["lora_scale"]])
            t0 = time.perf_counter()
            out = pipe(
                prompt=cfg["style_prompts"][style], negative_prompt=cfg["negative_prompt"],
                image=gen_image, mask_image=gen_mask, height=gen_size[1], width=gen_size[0],
                num_inference_steps=g["steps"], guidance_scale=g["guidance_scale"], strength=g["strength"],
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            seconds = round(time.perf_counter() - t0, 2)
            generated = out.convert("RGB").resize(working.size, Image.Resampling.LANCZOS)
            drift = sum(ImageStat.Stat(ImageChops.difference(generated, working), mask=protected).mean) / 3
            final = Image.composite(generated, working, mask)  # G8
            name = f"{row['id']}_{style}_{config}.jpg"
            final.save(IMAGES / name, quality=95, subsampling=0)
            finals.append(pack_view(final, pp["side"], pp["jpeg_quality"]))
            crops.append(product_crop(generated, alpha, pp["side"], pp["neutral"], pp["jpeg_quality"]))
            rows.append({"id": row["id"], "item_style": row["style"], "style": style, "config": config,
                         "seed": seed, "seconds": seconds, "image": name,
                         "protected_mean_abs_diff": round(drift, 3)})
    clip_scores = clip.images(finals) @ matrix.T
    dino_embeds = dino.images([ref_crop, *crops])
    dino_cos = dino_embeds[1:] @ dino_embeds[0]
    for r, scores, d in zip(rows, clip_scores, dino_cos):
        target = styles.index(r["style"])
        others = np.delete(scores, target)
        r.update({"clip_target": round(float(scores[target]), 5),
                  "clip_margin": round(float(scores[target] - others.max()), 5),
                  "clip_top1": styles[int(np.argmax(scores))],
                  "dino_product": round(float(d), 5)})
    return rows


def main() -> None:
    import numpy as np  # noqa: F401 - falla pronto si el VM no lo trae
    import torch

    sys.path.insert(0, str(CONTENT))
    import colab_job

    global DEADLINE
    cfg = json.loads((CONTENT / "job.json").read_text(encoding="utf-8"))
    DEADLINE = colab_job.DEADLINE = float(cfg.get("deadline_unix") or float("inf"))
    IMAGES.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"run_id": cfg["run_id"], "stage": cfg["stage"], "status": "running",
                      "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": cfg}
    manifest_path = OUT / "manifest.json"
    metrics_path = OUT / "metrics.jsonl"

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    try:
        save()
        torch.cuda.reset_peak_memory_stats()
        manifest["install"] = colab_job.install()
        manifest["environment"] = {"gpu": torch.cuda.get_device_name(0), "packages": manifest["install"]["after"]}
        manifest["eval_tar"] = unpack(cfg)
        save()

        models = cfg["models"]
        clip = Clip(models["clip"]["id"], models["clip"]["revision"])
        dino = Dino(models["dino"]["id"], models["dino"]["revision"])
        check_deadline("referencias de estilo")
        centroids, manifest["style_references"] = style_references(clip, cfg)
        save()

        check_deadline("carga del pipeline")
        t0 = time.perf_counter()
        pipe = load_pipeline(cfg)
        manifest["pipeline_load_seconds"] = round(time.perf_counter() - t0, 1)
        val = [json.loads(line) for line in (EVAL / "val" / "eval.jsonl").open(encoding="utf-8")]
        random.Random(cfg["order_seed"]).shuffle(val)
        manifest["order"] = [r["id"] for r in val]
        manifest["completed"] = []
        product_seconds: list[float] = []
        for row in val:
            t1 = time.perf_counter()
            try:
                if time.time() + max(product_seconds, default=0) * 1.2 > DEADLINE:
                    raise OutOfTime("estimación")
                rows = evaluate_product(pipe, clip, dino, centroids, cfg, row)
            except OutOfTime:
                for leftover in IMAGES.glob(f"{row['id']}_*.jpg"):
                    leftover.unlink()
                log(f"Sin tiempo: quedan {len(val) - len(manifest['completed'])} productos sin evaluar")
                break
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
            product_seconds.append(time.perf_counter() - t1)
            manifest["completed"].append(row["id"])
            manifest["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            save()
            done = len(manifest["completed"])
            log(f"Producto {done}/{len(val)} ({row['id']}, {row['style']}) en {product_seconds[-1]:.0f} s")
        manifest["partial"] = len(manifest["completed"]) < len(val)
        manifest["files"] = {"metrics.jsonl": sha256(metrics_path), "style_refs.npz": sha256(OUT / "style_refs.npz")}
        manifest["status"] = "ok"
    except BaseException as exc:
        manifest["status"] = f"error: {exc!r}"
        raise
    finally:
        manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()
        with tarfile.open(OUT / "result.tar.gz", "w:gz") as tar:
            for name in ("manifest.json", "metrics.jsonl", "style_refs.npz"):
                if (OUT / name).exists():
                    tar.add(OUT / name, arcname=name)
            if IMAGES.exists():
                tar.add(IMAGES, arcname="images")
        log(f"Resultado en {OUT / 'result.tar.gz'} (estado: {manifest['status']})")


if __name__ == "__main__":
    main()
