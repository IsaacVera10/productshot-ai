"""Vecinos de identidad entre recortes de producto de BG60k con DINOv2 (etapa 4 de docs/10).

    uv run --group dataset python training/bg60k_identity.py --run data/bg60k/embeddings/<RUN_ID>
    uv run --group dataset python training/bg60k_identity.py --run ... --band 0.85 0.90 --out-sheet pares.jpg

Calcula los k vecinos más cercanos de cada recorte (coseno) y la distancia de Hamming entre sus
pHash. No agrupa todavía: las hojas de pares por banda de similitud sirven para calibrar el
umbral con revisión humana antes de unir grupos.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from bg60k_manifest import hamming

K = 5
BLOCK = 2048


def nearest(emb: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    emb = emb.astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    idx = np.empty((len(emb), k), dtype=np.int64)
    sim = np.empty((len(emb), k), dtype=np.float32)
    for start in range(0, len(emb), BLOCK):
        block = emb[start:start + BLOCK] @ emb.T
        rows = np.arange(len(block))
        block[rows, start + rows] = -1
        top = np.argpartition(-block, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(block, top, axis=1), axis=1)
        idx[start:start + BLOCK] = np.take_along_axis(top, order, axis=1)
        sim[start:start + BLOCK] = np.take_along_axis(block, idx[start:start + BLOCK], axis=1)
    return idx, sim


def neighbor_rows(ids: np.ndarray, idx: np.ndarray, sim: np.ndarray, phash: dict[int, str]) -> list[dict]:
    rows = []
    for a, (neighbors, sims) in enumerate(zip(idx, sim)):
        for rank, (b, s) in enumerate(zip(neighbors, sims)):
            ia, ib = int(ids[a]), int(ids[b])
            rows.append({"id": ia, "neighbor": ib, "rank": rank, "cosine": float(s),
                         "phash_distance": hamming(phash[ia], phash[ib])})
    return rows


def pair_sheet(pairs: list[dict], cutouts: Path, out: Path, cols: int = 4, thumb: int = 160) -> None:
    sheet = Image.new("RGB", (cols * thumb * 2 + cols * 8, -(-len(pairs) // cols) * (thumb + 16)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, p in enumerate(pairs):
        x, y = (k % cols) * (thumb * 2 + 8), (k // cols) * (thumb + 16)
        for offset, i in enumerate((p["id"], p["neighbor"])):
            sheet.paste(Image.open(cutouts / f"{i}.jpg").resize((thumb, thumb)), (x + offset * thumb, y))
        draw.text((x + 2, y + thumb + 2), f'{p["id"]}-{p["neighbor"]} cos {p["cosine"]:.3f} h{p["phash_distance"]}',
                  fill="black")
    sheet.save(out, quality=88)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vecinos de identidad de BG60k con DINOv2.")
    parser.add_argument("--run", type=Path, required=True, help="carpeta de la corrida embed")
    parser.add_argument("--data", type=Path, default=Path("data/bg60k"))
    parser.add_argument("--band", type=float, nargs=2, help="coseno mínimo y máximo de los pares a mostrar")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-sheet", type=Path, default=Path("data/bg60k/identity_sheet.jpg"))
    args = parser.parse_args(argv)
    out = args.data / "identity_neighbors.parquet"

    if args.band:
        low, high = args.band
        pairs = [r for r in pq.read_table(out).to_pylist() if r["rank"] == 0 and low <= r["cosine"] < high]
        sample = random.Random(args.seed).sample(pairs, min(args.n, len(pairs)))
        pair_sheet(sample, args.data / "pack" / "cutouts", args.out_sheet)
        print(f"{len(pairs)} pares en [{low}, {high}); hoja en {args.out_sheet}; "
              f"semilla {args.seed}: {[(p['id'], p['neighbor']) for p in sample]}")
        return 0

    dino = np.load(args.run / "dino.npz")
    phash = {r["id"]: r["phash_cutout"] for r in pq.read_table(args.data / "candidates.parquet").to_pylist()}
    idx, sim = nearest(dino["dino"])
    rows = neighbor_rows(dino["ids"], idx, sim, phash)
    pq.write_table(pa.Table.from_pylist(rows), out)

    top1 = sim[:, 0]
    bands = [0.0, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.01]
    summary = {
        "run": args.run.name,
        "items": len(top1),
        "k": K,
        "nearest_cosine": {f"p{q}": round(float(np.quantile(top1, q / 100)), 4) for q in (5, 25, 50, 75, 95)},
        "nearest_by_band": {f"[{a}, {b})": int(((top1 >= a) & (top1 < b)).sum()) for a, b in zip(bands, bands[1:])},
        "nearest_phash_distance_le_6": sum(1 for r in rows if r["rank"] == 0 and r["phash_distance"] <= 6),
    }
    (args.data / "identity_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
