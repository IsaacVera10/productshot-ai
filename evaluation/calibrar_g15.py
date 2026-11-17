"""Calibración de G15: probabilidad por grupo de `CATEGORY_LABELS` para una carpeta de fotos.

    uv run --no-project --with torch --with transformers==4.46.3 --with pillow \
        python evaluation/calibrar_g15.py <carpeta> [<carpeta> ...] [--tar archivo.tar]

Usa las etiquetas y el CLIP de `services/screening.py` (mismo softmax sumado por grupo, misma
vista: la foto reducida a 1024 px de lado mayor, como la sube la API). Imprime por foto los
grupos bloqueados que pasan el umbral y un resumen; corre en GPU si hay, si no en CPU.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from productshot.services.screening import BLOCKED_CATEGORIES, CATEGORY_LABELS, CLIP_MODEL_ID  # noqa: E402

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def photos(folders: list[Path], tars: list[Path]):
    from PIL import Image

    for folder in folders:
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in SUFFIXES and "cutout" not in path.stem:
                yield str(path), Image.open(path).convert("RGB")
    for tar in tars:
        with tarfile.open(tar) as t:
            for m in t.getmembers():
                if m.isfile() and Path(m.name).suffix.lower() in SUFFIXES:
                    yield f"{tar.name}:{m.name}", Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")


def main(argv: list[str] | None = None) -> int:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", type=Path, nargs="*")
    parser.add_argument("--tar", type=Path, action="append", default=[])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, help="JSONL con todos los puntajes")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=dtype).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    groups = [(g, t) for g, texts in CATEGORY_LABELS.items() for t in texts]
    with torch.no_grad():
        tokens = proc(text=[t for _, t in groups], return_tensors="pt", padding=True).to(device)
        text = model.get_text_features(**tokens)
        text = text / text.norm(dim=-1, keepdim=True)

    rows = []
    for name, image in photos(args.folders, args.tar):
        image.thumbnail((1024, 1024))
        with torch.no_grad():
            px = proc(images=image, return_tensors="pt")["pixel_values"].to(device, dtype)
            emb = model.get_image_features(pixel_values=px)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            probs = (model.logit_scale.exp() * emb @ text.T).float().softmax(-1)[0].tolist()
        scores: dict[str, float] = {}
        for (g, _), p in zip(groups, probs):
            scores[g] = scores.get(g, 0.0) + p
        blocked = {g: round(scores[g], 4) for g in BLOCKED_CATEGORIES if scores[g] >= args.threshold}
        top = max(BLOCKED_CATEGORIES, key=scores.get)
        rows.append({"photo": name, "blocked": blocked, "top_blocked": top, "top_score": round(scores[top], 4),
                     "scores": {g: round(v, 4) for g, v in scores.items()}})
        if blocked:
            print(f"BLOQUEA {name}: {blocked}")
    if args.out:
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    worst = sorted(rows, key=lambda r: -r["top_score"])[:5]
    print(json.dumps({"device": device, "photos": len(rows), "bloqueadas": sum(bool(r["blocked"]) for r in rows),
                      "mayores": [(r["photo"], r["top_blocked"], r["top_score"]) for r in worst]},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
