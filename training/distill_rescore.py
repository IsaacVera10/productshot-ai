"""Recalifica las candidatas de autodestilación con descripciones neutras de cada estilo.

    uv run --no-project --with torch --with transformers==4.46.3 --with pillow \
        python training/distill_rescore.py data/lora/distill2/<RUN_ID>

Con los prompts v2 la selección por texto queda sesgada: todos empiezan con «professional product
photograph» y el de studio, el más genérico, gana casi siempre. Aquí cada estilo se describe solo
por lo que lo distingue y se escribe `metrics_neutral.jsonl` con `neutral_margin` y `neutral_top1`,
calculados con el mismo CLIP de la evaluación sobre la misma vista (224 px, JPEG 92).
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

CLIP_MODEL = "openai/clip-vit-large-patch14"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
DESCRIPTIONS = {
    "studio": "a product photo in a photo studio with a plain light grey seamless backdrop",
    "natural": "a product photo outdoors on natural stone with green leaves",
    "warm": "a product photo on a wooden table with warm golden light and bokeh",
    "lifestyle": "a product photo on a kitchen counter in a home",
    "minimal": "a minimalist product photo on a plain off-white background with empty space",
}


def view(path: Path, side: int = 224, quality: int = 92):
    from PIL import Image

    buf = io.BytesIO()
    Image.open(path).convert("RGB").resize((side, side), Image.Resampling.LANCZOS).save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def main(argv: list[str] | None = None) -> int:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION)
    styles = list(DESCRIPTIONS)
    with torch.inference_mode():
        tokens = processor(text=[DESCRIPTIONS[s] for s in styles], return_tensors="pt", padding=True).to(device)
        texts = torch.nn.functional.normalize(model.get_text_features(**tokens), dim=-1)

    rows = [json.loads(line) for line in (args.run / "metrics.jsonl").open(encoding="utf-8")]
    out = []
    for k in range(0, len(rows), 32):
        batch = rows[k:k + 32]
        with torch.inference_mode():
            pixels = processor(images=[view(args.run / "images" / r["image"]) for r in batch],
                               return_tensors="pt")["pixel_values"].to(device)
            scores = torch.nn.functional.normalize(model.get_image_features(pixel_values=pixels), dim=-1) @ texts.T
        for r, s in zip(batch, scores.cpu().tolist()):
            target = styles.index(r["style"])
            others = max(v for j, v in enumerate(s) if j != target)
            out.append(r | {"neutral_margin": round(s[target] - others, 5), "neutral_top1": styles[s.index(max(s))]})
    (args.run / "metrics_neutral.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    hits = {s: sum(r["neutral_top1"] == r["style"] for r in out if r["style"] == s) for s in styles}
    print(json.dumps({"device": device, "candidates": len(out), "neutral_wins": hits}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
