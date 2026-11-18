"""Paquete de imágenes reducidas para calcular embeddings en Colab (etapas 4 y 5 de docs/10).

    uv run --group dataset python training/bg60k_pack.py

Toma los candidatos `core` y `near`, deja una copia por imagen repetida y guarda la foto
completa a 224 px para CLIP. Los que no tienen alertas llevan además el recorte del producto
sobre gris para DINOv2. Si se corta, la siguiente corrida valida y reutiliza lo ya escrito.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import PIL
import pyarrow.parquet as pq
from PIL import Image

SIDE = 224
NEUTRAL = 128
QUALITY = 92
REFERENCES = 128
PREPROCESS = {"side": SIDE, "neutral": NEUTRAL, "resample": "lanczos", "jpeg_quality": QUALITY,
              "pillow": PIL.__version__}


def candidates(path: Path, tiers: tuple[str, ...] = ("core", "near")) -> tuple[list[dict], dict[int, int]]:
    rows = sorted((r for r in pq.read_table(path).to_pylist() if r["niche"] in tiers), key=lambda r: r["id"])
    first: dict[str, int] = {}
    duplicates: dict[int, int] = {}
    unique = []
    for r in rows:
        if r["img_sha256"] in first:
            duplicates[r["id"]] = first[r["img_sha256"]]
        else:
            first[r["img_sha256"]] = r["id"]
            unique.append(r)
    return unique, duplicates


def product_cutout(img: Image.Image, mask: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    alpha = mask.getchannel("A") if mask.mode == "RGBA" else mask.convert("L")
    gray = Image.new("RGB", img.size, (NEUTRAL,) * 3)
    crop = Image.composite(img, gray, alpha).crop(bbox)
    side = max(crop.size)
    square = Image.new("RGB", (side, side), (NEUTRAL,) * 3)
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return square.resize((SIDE, SIDE), Image.Resampling.LANCZOS)


def is_valid(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return im.size == (SIDE, SIDE)
    except Exception:
        return False


def save_atomic(image: Image.Image, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    image.save(tmp, format="JPEG", quality=QUALITY)
    tmp.replace(path)


class Zips:
    def __init__(self, raw: Path) -> None:
        self.raw = raw
        self.open: dict[str, zipfile.ZipFile] = {}

    def read(self, name: str, member: str) -> bytes:
        if name not in self.open:
            self.open[name] = zipfile.ZipFile(self.raw / name)
        return self.open[name].read(member)

    def close(self) -> None:
        for z in self.open.values():
            z.close()


def check_signature(out: Path) -> None:
    path = out / "preprocess.json"
    if path.exists() and json.loads(path.read_text()) != PREPROCESS:
        raise SystemExit(f"{out} se armó con otro preprocesado; borrar la carpeta para rehacerlo.")
    path.write_text(json.dumps(PREPROCESS))


def build(data: Path, raw: Path) -> dict:
    rows, duplicates = candidates(data / "candidates.parquet")
    out = data / "pack"
    for sub in ("images", "cutouts", "originals"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    check_signature(out)
    references = set(random.Random(0).sample([r["id"] for r in rows if not r["alerts"]], REFERENCES))

    zips = Zips(raw)
    start = time.time()
    try:
        for k, r in enumerate(rows, 1):
            img_path = out / "images" / f'{r["id"]}.jpg'
            cut_path = out / "cutouts" / f'{r["id"]}.jpg' if not r["alerts"] else None
            ref_path = out / "originals" / f'{r["id"]}.png' if r["id"] in references else None
            if is_valid(img_path) and (cut_path is None or is_valid(cut_path)) and (ref_path is None or ref_path.exists()):
                continue
            img_bytes = zips.read(r["img_zip"], r["img_member"])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            save_atomic(img.resize((SIDE, SIDE), Image.Resampling.LANCZOS), img_path)
            if cut_path:
                mask = Image.open(io.BytesIO(zips.read(r["mask_zip"], r["mask_member"])))
                bbox = (r["bbox_x0"], r["bbox_y0"], r["bbox_x1"], r["bbox_y1"])
                save_atomic(product_cutout(img, mask, bbox), cut_path)
            if ref_path:
                ref_path.write_bytes(img_bytes)
            if k % 200 == 0:
                print(f"\r{k}/{len(rows)}, {time.time() - start:.0f} s", end="", file=sys.stderr, flush=True)
    finally:
        zips.close()
    print(file=sys.stderr)

    index = {
        "preprocess": PREPROCESS,
        "candidates_sha256": hashlib.sha256((data / "candidates.parquet").read_bytes()).hexdigest(),
        "items": [{"id": r["id"], "niche": r["niche"], "category": r["category"], "alerts": r["alerts"],
                   "cutout": not r["alerts"], "img_sha256": r["img_sha256"]} for r in rows],
        "duplicates": {str(k): v for k, v in duplicates.items()},
        "references": sorted(references),
    }
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False))
    with tarfile.open(data / "pack.tar", "w") as tar:
        tar.add(out, arcname="pack", filter=lambda t: None if t.name.endswith(".tmp") else t)
    return {"items": len(rows), "with_cutout": sum(1 for r in rows if not r["alerts"]),
            "duplicates": len(duplicates), "tar_mb": round((data / "pack.tar").stat().st_size / 1e6, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paquete de imágenes para embeddings de BG60k.")
    parser.add_argument("--data", type=Path, default=Path("data/bg60k"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/bg60k"))
    args = parser.parse_args(argv)
    print(json.dumps(build(args.data, args.raw)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
