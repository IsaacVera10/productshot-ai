"""Manifiesto de BG60k: una fila por par imagen y máscara (etapa 1 de docs/10).

    uv run --group dataset python training/bg60k_manifest.py --raw data/raw/bg60k --out data/bg60k

Lee los zip sin extraerlos y guarda cada shard al terminarlo, así una corrida cortada
continúa donde quedó.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import re
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from PIL import Image
from scipy import fft, ndimage

ALPHA_ON = 127
MIN_COMPONENT = 0.01
EDGE_PX = 2
NEUTRAL = 128
SHARP_SIDE = 512
CHUNK = 200

SHARD_RE = re.compile(r"^bg60k_(imgs|masks)_(\d+)\.zip$")
DIST_KEYS = ["fg_frac", "soft_frac", "largest_frac", "bg_brightness", "bg_std", "fg_brightness",
             "fg_sharpness", "mask_rgb_err", "mask_rgb_outside"]


def phash(gray: np.ndarray, hash_size: int = 8) -> str:
    side = hash_size * 4
    pixels = np.asarray(Image.fromarray(gray.astype(np.uint8)).resize((side, side), Image.Resampling.LANCZOS),
                        dtype=np.float64)
    low = fft.dct(fft.dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")[:hash_size, :hash_size]
    bits = "".join("1" if b else "0" for b in (low > np.median(low)).flatten())
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def cutout_gray(rgb: np.ndarray, alpha: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    a = alpha.astype(np.float64)[..., None] / 255.0
    x0, y0, x1, y1 = bbox
    crop = luma(rgb.astype(np.float64) * a + NEUTRAL * (1 - a))[y0:y1, x0:x1]
    h, w = crop.shape
    side = max(h, w)
    square = np.full((side, side), NEUTRAL, dtype=np.float64)
    top, left = (side - h) // 2, (side - w) // 2
    square[top:top + h, left:left + w] = crop
    return square


def mask_stats(alpha: np.ndarray) -> dict:
    h, w = alpha.shape
    total = h * w
    fg = alpha > ALPHA_ON
    area = int(fg.sum())
    stats = {"fg_frac": area / total, "soft_frac": float(((alpha > 0) & (alpha < 255)).sum() / total)}
    if area == 0:
        return stats | {"n_components": 0, "largest_frac": 0.0, "bbox_x0": None, "bbox_y0": None,
                        "bbox_x1": None, "bbox_y1": None, "edges_touched": 0, "cx": None, "cy": None}

    labels, _ = ndimage.label(fg)
    sizes = np.bincount(labels.ravel())[1:]
    ys, xs = np.nonzero(fg)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    edges = [x0 <= EDGE_PX, y0 <= EDGE_PX, x1 >= w - EDGE_PX, y1 >= h - EDGE_PX]
    return stats | {
        "n_components": int((sizes >= MIN_COMPONENT * total).sum()),
        "largest_frac": float(sizes.max() / area),
        "bbox_x0": x0, "bbox_y0": y0, "bbox_x1": x1, "bbox_y1": y1,
        "edges_touched": sum(edges),
        "cx": float(xs.mean() / w), "cy": float(ys.mean() / h),
    }


def product_sharpness(rgb: np.ndarray, fg: np.ndarray) -> float | None:
    h, w = fg.shape
    scale = SHARP_SIDE / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    gray = np.asarray(Image.fromarray(rgb).convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float64)
    inside = np.asarray(Image.fromarray(fg.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)) > 0
    # Sin erosionar, el borde del recorte contaría como detalle del producto.
    inside = ndimage.binary_erosion(inside, iterations=3)
    if inside.sum() < 64:
        return None
    return float(ndimage.laplace(gray)[inside].var())


def analyze_pair(img_bytes: bytes, mask_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(img_bytes))
    mask = Image.open(io.BytesIO(mask_bytes))
    row = {"img_w": img.width, "img_h": img.height, "img_mode": img.mode,
           "mask_w": mask.width, "mask_h": mask.height, "mask_mode": mask.mode}
    rgb = np.asarray(img.convert("RGB"))
    if mask.mode == "RGBA":
        arr = np.asarray(mask)
        alpha, mask_rgb = arr[..., 3], arr[..., :3]
    else:
        alpha, mask_rgb = np.asarray(mask.convert("L")), None
    if alpha.shape != rgb.shape[:2]:
        return row | {"status": "size_mismatch"}

    row |= mask_stats(alpha)
    fg = alpha > ALPHA_ON
    bg = ~fg
    lum = luma(rgb.astype(np.float64))
    row["bg_brightness"] = float(lum[bg].mean()) if bg.any() else None
    row["bg_std"] = float(lum[bg].std()) if bg.any() else None
    row["fg_brightness"] = float(lum[fg].mean()) if fg.any() else None

    row["mask_rgb_err"] = row["mask_rgb_outside"] = None
    if mask_rgb is not None:
        opaque, empty = alpha >= 250, alpha == 0
        if opaque.any():
            row["mask_rgb_err"] = float(np.abs(mask_rgb[opaque].astype(np.int16) - rgb[opaque]).mean())
        if empty.any():
            row["mask_rgb_outside"] = float(mask_rgb[empty].mean())

    row["fg_sharpness"] = product_sharpness(rgb, fg) if fg.any() else None
    row["phash_img"] = phash(lum)
    row["phash_cutout"] = None
    if fg.any():
        bbox = (row["bbox_x0"], row["bbox_y0"], row["bbox_x1"], row["bbox_y1"])
        row["phash_cutout"] = phash(cutout_gray(rgb, alpha, bbox))
    return row | {"status": "ok"}


def read_info(path: Path) -> dict[int, str]:
    info = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            name, category = line.split("\t", 1)
            info[int(Path(name).stem)] = category.strip()
    return info


def find_shards(raw: Path) -> tuple[dict[int, dict[str, Path]], list[int], list[int]]:
    shards: dict[int, dict[str, Path]] = {}
    for path in sorted(raw.glob("bg60k_*.zip")):
        if m := SHARD_RE.match(path.name):
            shards.setdefault(int(m.group(2)), {})[m.group(1)] = path
    complete = sorted(i for i, kinds in shards.items() if {"imgs", "masks"} <= kinds.keys())
    partial = sorted(i for i in shards if i not in complete)
    return shards, complete, partial


def member_index(zf: zipfile.ZipFile, suffix: str) -> dict[int, str]:
    index = {}
    for name in zf.namelist():
        base = name.rsplit("/", 1)[-1]
        stem = base.removesuffix(suffix)
        if base.endswith(suffix) and stem.isdigit():
            index[int(stem)] = name
    return index


def process_chunk(img_zip: str, mask_zip: str, ids: list[int]) -> list[dict]:
    rows = []
    with zipfile.ZipFile(img_zip) as zi, zipfile.ZipFile(mask_zip) as zm:
        images, masks = member_index(zi, ".png"), member_index(zm, "_mask.png")
        for i in ids:
            row = {"id": i, "img_zip": Path(img_zip).name, "mask_zip": Path(mask_zip).name,
                   "img_member": images.get(i), "mask_member": masks.get(i)}
            if row["img_member"] is None or row["mask_member"] is None:
                rows.append(row | {"status": "missing_member"})
                continue
            try:
                img_bytes, mask_bytes = zi.read(row["img_member"]), zm.read(row["mask_member"])
                row["img_sha256"] = hashlib.sha256(img_bytes).hexdigest()
                row["mask_sha256"] = hashlib.sha256(mask_bytes).hexdigest()
                row |= analyze_pair(img_bytes, mask_bytes)
            except Exception as exc:
                row["status"] = f"error: {type(exc).__name__}: {exc}"[:300]
            rows.append(row)
    return rows


def quantiles(values: list) -> dict | None:
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return None
    qs = {f"p{q}": float(np.quantile(v, q / 100)) for q in (1, 5, 25, 50, 75, 95, 99)}
    return {"n": int(v.size), "min": float(v.min()), "max": float(v.max())} | qs


def repeated(values) -> int:
    return sum(c - 1 for c in Counter(values).values() if c > 1)


def summarize(rows: list[dict], info: dict[int, str], complete: list[int], partial: list[int]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tools": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                  "pillow": Image.__version__, "pyarrow": pa.__version__},
        "params": {"alpha_on": ALPHA_ON, "min_component": MIN_COMPONENT, "edge_px": EDGE_PX,
                   "neutral": NEUTRAL, "sharp_side": SHARP_SIDE},
        "shards_complete": complete,
        "shards_partial": partial,
        "info_rows": len(info),
        "pairs_expected": sum(1 for i in info if i % 10 in complete),
        "rows": len(rows),
        "ok": len(ok),
        "status": dict(Counter(r.get("status", "?").split(":")[0] for r in rows)),
        "categories_total": len(set(info.values())),
        "categories_with_pairs": len({r["category"] for r in ok}),
        "sizes": dict(Counter(f'{r["img_w"]}x{r["img_h"]}' for r in ok)),
        "mask_modes": dict(Counter(r["mask_mode"] for r in ok)),
        "n_components": dict(sorted(Counter(r["n_components"] for r in ok).items())),
        "edges_touched": dict(sorted(Counter(r["edges_touched"] for r in ok).items())),
        "dist": {k: quantiles([r.get(k) for r in ok]) for k in DIST_KEYS},
        "exact_dup_images": repeated(r["img_sha256"] for r in ok),
        "same_phash_cutout": repeated(r["phash_cutout"] for r in ok if r["phash_cutout"]),
    }


def write_parquet(rows: list[dict], path: Path) -> None:
    columns = list(dict.fromkeys(k for r in rows for k in r))
    pq.write_table(pa.Table.from_pylist([{k: r.get(k) for k in columns} for r in rows]), path)


def run_tasks(tasks: list[tuple], workers: int, on_progress) -> list[dict]:
    rows: list[dict] = []
    if workers <= 1:
        for task in tasks:
            rows.extend(process_chunk(*task))
            on_progress(len(rows))
        return rows
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(process_chunk, *t) for t in tasks]):
            rows.extend(future.result())
            on_progress(len(rows))
    return rows


def load_saved_shard(path: Path, ids: list[int]) -> list[dict] | None:
    if not path.exists():
        return None
    rows = pq.read_table(path).to_pylist()
    return rows if sorted(r["id"] for r in rows) == ids else None


def build(raw: Path, out: Path, workers: int, limit: int | None = None) -> dict:
    info = read_info(raw / "bg60k_info.txt")
    shards, complete, partial = find_shards(raw)
    ids_by_shard = {s: sorted(i for i in info if i % 10 == s)[:limit] for s in complete}
    total = sum(len(ids) for ids in ids_by_shard.values())
    parts = out / "shards"
    parts.mkdir(parents=True, exist_ok=True)

    start = time.time()
    rows: list[dict] = []
    reused: list[int] = []
    for s, ids in ids_by_shard.items():
        part = parts / f"shard_{s}.parquet"
        if (saved := load_saved_shard(part, ids)) is not None:
            rows.extend(saved)
            reused.append(s)
            continue

        done = len(rows)
        tasks = [(str(shards[s]["imgs"]), str(shards[s]["masks"]), ids[k:k + CHUNK])
                 for k in range(0, len(ids), CHUNK)]
        shard_rows = run_tasks(tasks, workers, lambda n: print(
            f"\rshard {s}: {done + n}/{total} pares, {time.time() - start:.0f} s",
            end="", file=sys.stderr, flush=True))
        for r in shard_rows:
            r["category"], r["shard"] = info[r["id"]], s
        shard_rows.sort(key=lambda r: r["id"])

        tmp = part.with_suffix(".tmp")
        write_parquet(shard_rows, tmp)
        tmp.replace(part)
        rows.extend(shard_rows)
    print(file=sys.stderr)

    rows.sort(key=lambda r: r["id"])
    write_parquet(rows, out / "manifest.parquet")
    summary = summarize(rows, info, complete, partial) | {
        "seconds": round(time.time() - start, 1),
        "shards_reused": reused,
        "source": {"dataset": "BG60k", "url": "https://github.com/Whileherham/BG60k",
                   "terms": "solo uso académico (README de BG60k); sin licencia con nombre"},
    }
    (out / "manifest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manifiesto de BG60k.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/bg60k"))
    parser.add_argument("--out", type=Path, default=Path("data/bg60k"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="pares por shard, para probar")
    args = parser.parse_args(argv)

    summary = build(args.raw, args.out, args.workers, args.limit)
    print(json.dumps({k: summary[k] for k in ["rows", "ok", "status", "pairs_expected", "seconds"]}))
    return 0 if summary["ok"] == summary["rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
