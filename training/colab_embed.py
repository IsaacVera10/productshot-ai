"""Embeddings de BG60k dentro del VM de Colab (etapas 4 y 5 de docs/10).

Lo lanza `training/run_colab.py --stage embed` después de subir `/content/job.json` y
`/content/pack.tar` en partes (de `training/bg60k_pack.py`). Calcula CLIP de cada foto y de las
descripciones de estilo, y DINOv2 de los recortes de producto. Antes del lote completo prueba
un lote pequeño, y compara el preprocesado del paquete con el oficial de CLIP en las imágenes
de referencia. Deja en `/content/out/result.tar.gz` los embeddings, el índice y el manifiesto.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tarfile
import time
from pathlib import Path

CONTENT = Path("/content")
PACK = CONTENT / "pack"
OUT = CONTENT / "out"
BATCH = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_revision(model_id: str, revision: str) -> str:
    from huggingface_hub import model_info
    return model_info(model_id, revision=revision).sha


class Clip:
    """Usa las proyecciones del modelo directamente: no depende de qué devuelva `get_*_features`."""

    def __init__(self, model_id: str, revision: str) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.model = CLIPModel.from_pretrained(model_id, revision=revision).cuda().eval()
        self.processor = CLIPProcessor.from_pretrained(model_id, revision=revision)

    def text(self, prompts: list[str]):
        torch = self.torch
        with torch.inference_mode():
            tokens = self.processor(text=prompts, return_tensors="pt", padding=True).to("cuda")
            pooled = self.model.text_model(**tokens).pooler_output
            return torch.nn.functional.normalize(self.model.text_projection(pooled), dim=-1).cpu().numpy()

    def images(self, images: list):
        torch = self.torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to("cuda")
            pooled = self.model.vision_model(pixel_values=pixels).pooler_output
            embeds = self.model.visual_projection(pooled)
        return torch.nn.functional.normalize(embeds.float(), dim=-1).cpu().numpy()


class Dino:
    def __init__(self, model_id: str, revision: str) -> None:
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.model = AutoModel.from_pretrained(model_id, revision=revision).cuda().eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)

    def images(self, images: list):
        import numpy as np
        torch = self.torch
        # Sin el resize y el recorte central del procesador: el recorte ya viene cuadrado y completo.
        pixels = torch.from_numpy(np.stack([np.asarray(im) for im in images])).cuda()
        pixels = (pixels.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            pooled = self.model(pixel_values=pixels).pooler_output
        return torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy()


def embed_all(encoder, paths: list[Path], deadline: float, label: str):
    import numpy as np
    from PIL import Image
    chunks = []
    for k in range(0, len(paths), BATCH):
        if time.time() > deadline:
            raise TimeoutError(f"se acabó el plazo durante {label}")
        chunks.append(encoder.images([Image.open(p).convert("RGB") for p in paths[k:k + BATCH]]))
    return np.concatenate(chunks)


def check_block(name: str, x, dim: int | None = None) -> dict:
    import numpy as np
    norms = np.linalg.norm(x, axis=1)
    if not np.isfinite(x).all() or (dim and x.shape[1] != dim) or np.abs(norms - 1).max() > 1e-3:
        raise RuntimeError(f"{name}: embeddings inválidos (forma {x.shape}, normas {norms.min()}..{norms.max()})")
    return {"shape": list(x.shape), "norm_min": float(norms.min()), "norm_max": float(norms.max())}


def reference_check(clip: Clip, ids: list[int]) -> dict:
    """Compara la foto reducida del paquete con el original pasado por el procesador oficial."""
    import numpy as np
    from PIL import Image
    packed = clip.images([Image.open(PACK / "images" / f"{i}.jpg").convert("RGB") for i in ids])
    original = clip.images([Image.open(PACK / "originals" / f"{i}.png").convert("RGB") for i in ids])
    cosine = (packed * original).sum(axis=1)
    return {"n": len(ids), "cosine_min": float(cosine.min()), "cosine_mean": float(cosine.mean()),
            "cosine_p5": float(np.quantile(cosine, 0.05))}


def main() -> None:
    import numpy as np
    import torch

    cfg = json.loads((CONTENT / "job.json").read_text(encoding="utf-8"))
    deadline = float(cfg.get("deadline_unix") or float("inf"))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"run_id": cfg["run_id"], "stage": cfg["stage"], "status": "running",
                      "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": cfg}
    manifest_path = OUT / "manifest.json"

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    try:
        save()
        import PIL
        import transformers
        manifest["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                                   "transformers": transformers.__version__, "numpy": np.__version__,
                                   "pillow": PIL.__version__, "gpu": torch.cuda.get_device_name(0)}
        log(f"Entorno: {manifest['environment']}")

        with (CONTENT / "pack.tar").open("wb") as dst:
            for k in range(cfg["pack_parts"]):
                part = CONTENT / f"pack.tar.part{k:03d}"
                dst.write(part.read_bytes())
                part.unlink()
        manifest["pack_sha256"] = sha256(CONTENT / "pack.tar")
        if manifest["pack_sha256"] != cfg["pack_sha256"]:
            raise RuntimeError("El paquete subido no coincide con el sha256 de la laptop.")
        with tarfile.open(CONTENT / "pack.tar") as tar:
            tar.extractall(CONTENT, filter="data")
        index = json.loads((PACK / "index.json").read_text(encoding="utf-8"))
        shutil.copy(PACK / "index.json", OUT / "index.json")
        ids = [item["id"] for item in index["items"]]
        cut_ids = [item["id"] for item in index["items"] if item["cutout"]]
        manifest["pack"] = {"items": len(ids), "with_cutout": len(cut_ids), "preprocess": index["preprocess"],
                            "candidates_sha256": index["candidates_sha256"]}
        log(f"Paquete: {len(ids)} fotos, {len(cut_ids)} recortes")

        clip_cfg, dino_cfg = cfg["models"]["clip"], cfg["models"]["dino"]
        clip_rev = resolve_revision(clip_cfg["id"], clip_cfg["revision"])
        dino_rev = resolve_revision(dino_cfg["id"], dino_cfg["revision"])
        manifest["models"] = {"clip": {"id": clip_cfg["id"], "revision": clip_rev},
                              "dino": {"id": dino_cfg["id"], "revision": dino_rev}}
        save()

        groups = [(f"style:{s}", t) for s, texts in cfg["prompts"]["styles"].items() for t in texts]
        groups += [(f"other:{s}", t) for s, texts in cfg["prompts"]["other"].items() for t in texts]

        clip = Clip(clip_cfg["id"], clip_rev)
        text = clip.text([t for _, t in groups])
        probe = embed_all(clip, [PACK / "images" / f"{i}.jpg" for i in ids[:64]], deadline, "la prueba de CLIP")
        manifest["probe"] = {"clip_text": check_block("clip_text", text), "clip_image": check_block("clip_image", probe)}
        manifest["reference_check"] = reference_check(clip, index["references"])
        log(f"Prueba OK; referencia: {manifest['reference_check']}")
        save()

        t0 = time.time()
        clip_img = embed_all(clip, [PACK / "images" / f"{i}.jpg" for i in ids], deadline, "CLIP")
        manifest["clip_image"] = check_block("clip_image", clip_img, probe.shape[1]) | {"seconds": round(time.time() - t0, 1)}
        np.savez(OUT / "clip.npz", ids=np.asarray(ids), clip_image=clip_img.astype(np.float16), clip_text=text,
                 prompt_group=np.asarray([g for g, _ in groups]), prompt_text=np.asarray([t for _, t in groups]))
        save()
        del clip
        torch.cuda.empty_cache()

        t1 = time.time()
        dino = Dino(dino_cfg["id"], dino_rev)
        dino_emb = embed_all(dino, [PACK / "cutouts" / f"{i}.jpg" for i in cut_ids], deadline, "DINOv2")
        manifest["dino"] = check_block("dino", dino_emb) | {"seconds": round(time.time() - t1, 1)}
        np.savez(OUT / "dino.npz", ids=np.asarray(cut_ids), dino=dino_emb.astype(np.float16))

        manifest["files"] = {name: sha256(OUT / name) for name in ("clip.npz", "dino.npz", "index.json")}
        manifest["status"] = "ok"
    except BaseException as exc:
        manifest["status"] = f"error: {exc!r}"
        raise
    finally:
        manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()
        with tarfile.open(OUT / "result.tar.gz", "w:gz") as tar:
            for name in ("manifest.json", "index.json", "clip.npz", "dino.npz"):
                if (OUT / name).exists():
                    tar.add(OUT / name, arcname=name)
        log(f"Resultado en {OUT / 'result.tar.gz'} (estado: {manifest['status']})")


if __name__ == "__main__":
    main()
