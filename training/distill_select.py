"""Elige las candidatas de entrenamiento y arma el dataset del LoRA por autodestilación (docs/10).

    PYTHONPATH=backend:training uv run --group dataset python training/distill_select.py \
        data/lora/distill/<RUN_ID> --out data/bg60k/export/distill.tar

Lee la corrida `run_colab.py --stage eval --distill` (metrics.jsonl e images/). De cada producto y
estilo se queda con la candidata de mayor margen CLIP contra el texto del preset, siempre que el
estilo pedido gane, que el producto se conserve (DINOv2) y que no esté en `--exclude` (revisión
visual). La máscara se rehace con el mismo reencuadre que usó la generación, así que cada foto
lleva el producto exactamente donde quedó. El caption es el prompt con el que se generó (config.style_prompts), igual que en inferencia.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

from PIL import Image

from bg60k_export import add_bytes
from productshot.services.layout import Layout, compose

MIN_DINO = 0.9


def choose(rows: list[dict], excluded: set[str], score: str = "text") -> dict[tuple[int, str], dict]:
    """`score`: "text" (prompt de cada estilo) o "neutral" (descripciones de distill_rescore.py)."""
    margin, top1 = f"{score}_margin", f"{score}_top1"
    best: dict[tuple[int, str], dict] = {}
    for r in rows:
        if r[top1] != r["style"] or r["dino_product"] < MIN_DINO or r["image"] in excluded:
            continue
        key = (r["id"], r["style"])
        if key not in best or r[margin] > best[key][margin]:
            best[key] = r
    return best


def framed_mask(photo: bytes, mask: bytes, layout: dict, target_side: int) -> Image.Image:
    """El mismo lienzo que `colab_eval.distill_product`: lado mayor a target_side y reencuadre."""
    image, alpha = Image.open(io.BytesIO(photo)).convert("RGB"), Image.open(io.BytesIO(mask)).convert("L")
    scale = target_side / max(image.size)
    size = tuple(max(8, round(v * scale / 8) * 8) for v in image.size)
    image, alpha = image.resize(size, Image.Resampling.LANCZOS), alpha.resize(size, Image.Resampling.BILINEAR)
    _, editable = compose(image, alpha.point(lambda v: 255 - v), Layout(**layout))
    return editable.point(lambda v: 255 - v)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--source", type=Path, default=Path("data/bg60k/export/preset_masked.tar"))
    parser.add_argument("--out", type=Path, default=Path("data/bg60k/export/distill.tar"))
    parser.add_argument("--neutral", action="store_true",
                        help="elegir con metrics_neutral.jsonl (distill_rescore.py) en vez del texto del prompt")
    parser.add_argument("--exclude", type=Path, help="CSV con la columna image de las candidatas descartadas")
    args = parser.parse_args(argv)

    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    metrics = "metrics_neutral.jsonl" if args.neutral else "metrics.jsonl"
    rows = [json.loads(line) for line in (args.run / metrics).open(encoding="utf-8")]
    excluded = set()
    if args.exclude:
        with args.exclude.open(encoding="utf-8") as f:
            excluded = {r["image"] for r in csv.DictReader(f)}
    chosen = choose(rows, excluded, "neutral" if args.neutral else "text")

    with tarfile.open(args.source) as src:
        files = {m.name: src.extractfile(m).read() for m in src.getmembers() if m.isfile()}
    meta, counts = [], {}
    tmp = args.out.with_suffix(".tar.tmp")
    with tarfile.open(tmp, "w") as tar:
        for (item, style), r in sorted(chosen.items()):
            image = (args.run / "images" / r["image"]).read_bytes()
            mask = framed_mask(files[f"{item}.png"], files[f"{item}.mask"], cfg["layout"],
                               cfg["generation"]["target_side"])
            if Image.open(io.BytesIO(image)).size != mask.size:
                raise SystemExit(f"{r['image']}: la máscara rehecha no tiene el tamaño de la imagen.")
            buf = io.BytesIO()
            mask.point(lambda v: 255 if v >= 128 else 0).save(buf, format="PNG")
            stem = f"{item}_{style}"
            add_bytes(tar, f"{stem}.jpg", image)
            add_bytes(tar, f"{stem}.mask", buf.getvalue())
            meta.append({"file_name": f"{stem}.jpg", "text": cfg["style_prompts"][style], "style": style,
                         "mask": f"{stem}.mask", "source": r["image"], "margin": r["neutral_margin" if args.neutral else "text_margin"]})
            counts[style] = counts.get(style, 0) + 1
        add_bytes(tar, "metadata.jsonl", "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta).encode())
    tmp.replace(args.out)
    candidates = {(r["id"], r["style"]) for r in rows}
    print(json.dumps({"candidates": len(rows), "pairs": len(candidates), "chosen": len(meta),
                      "excluded": len(excluded), "by_style": counts}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
