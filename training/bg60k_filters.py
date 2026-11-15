"""Nicho por categoría y alertas de máscara sobre el manifiesto de BG60k (etapas 2 y 3 de docs/10).

    uv run --group dataset python training/bg60k_filters.py
    uv run --group dataset python training/bg60k_filters.py --sheet multi_object --out-sheet hoja.jpg

No descarta nada: agrega `niche`, `niche_rule` y `alerts` a cada fila y escribe
`bg60k_categories_review.csv` con la asignación de cada categoría. Las hojas de contacto
muestran la imagen con el fondo atenuado según el alfa, para calibrar umbrales.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import tomllib
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

RULES = Path(__file__).with_name("bg60k_categories.toml")
REVIEW = Path(__file__).with_name("bg60k_categories_review.csv")
PRIORITY = {"out": 0, "near": 1, "core": 2}

ALERTS = {
    "empty_mask": lambda r: r["fg_frac"] == 0,
    "multi_object": lambda r: r["n_components"] >= 2,
    "fragmented": lambda r: r["largest_frac"] < 0.85,
    "tiny_product": lambda r: 0 < r["fg_frac"] < 0.05,
    "large_product": lambda r: r["fg_frac"] > 0.60,
    "truncated": lambda r: r["edges_touched"] >= 1,
    "mask_mismatch": lambda r: (r["mask_rgb_err"] or 0) > 1.0,
}


def load_rules(path: Path = RULES) -> tuple[dict, list[tuple[str, str, re.Pattern]]]:
    rules = tomllib.loads(path.read_text(encoding="utf-8"))
    keywords = [(tier, kw, re.compile(rf"\b{re.escape(kw)}(s|es)?\b"))
                for tier in ("out", "near", "core") for kw in rules["keywords"][tier]]
    return rules["exact"], keywords


def classify(category: str, exact: dict, keywords: list) -> tuple[str, str]:
    if category in exact:
        tier, reason = exact[category]
        return tier, f"exact: {reason}"
    name = category.lower()
    hits = [(tier, kw) for tier, kw, pattern in keywords if pattern.search(name)]
    if not hits:
        return "unclassified", ""
    # Una frase más específica gana a la palabra que contiene ("tv accessories" frente a "tv").
    specific = [h for h in hits if not any(h[1] != o[1] and h[1] in o[1] for o in hits)]
    tier, kw = min(specific, key=lambda h: PRIORITY[h[0]])
    return tier, kw


def alerts_for(row: dict) -> list[str]:
    if row.get("status") != "ok":
        return ["unreadable"]
    return [name for name, check in ALERTS.items() if check(row)]


def apply(manifest: Path, rules: Path = RULES) -> list[dict]:
    exact, keywords = load_rules(rules)
    rows = pq.read_table(manifest).to_pylist()
    by_category = {c: classify(c, exact, keywords) for c in {r["category"] for r in rows}}
    for r in rows:
        r["niche"], r["niche_rule"] = by_category[r["category"]]
        r["alerts"] = alerts_for(r)
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_category_review(rows: list[dict], path: Path = REVIEW) -> None:
    counts = Counter((r["category"], r["niche"], r["niche_rule"]) for r in rows)
    clean = Counter(r["category"] for r in rows if not r["alerts"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["category", "pairs", "pairs_without_alerts", "niche", "rule"])
        for (category, niche, rule), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0])):
            writer.writerow([category, n, clean[category], niche, rule])


def summarize(rows: list[dict]) -> dict:
    clean = [r for r in rows if not r["alerts"]]
    tiers = Counter(r["niche"] for r in rows)
    unclassified = Counter(r["category"] for r in rows if r["niche"] == "unclassified")
    return {
        "rows": len(rows),
        "niche": dict(tiers),
        "niche_categories": dict(Counter(t for t, _ in {(r["niche"], r["category"]) for r in rows})),
        "alerts": dict(Counter(a for r in rows for a in r["alerts"])),
        "without_alerts": len(clean),
        "without_alerts_by_niche": dict(Counter(r["niche"] for r in clean)),
        "unclassified_top": unclassified.most_common(40),
        "thresholds": {"multi_object": ">= 2 componentes", "fragmented": "largest_frac < 0.85",
                       "tiny_product": "fg_frac < 0.05", "large_product": "fg_frac > 0.60",
                       "truncated": "edges_touched >= 1", "mask_mismatch": "mask_rgb_err > 1.0"},
    }


def overlay(img: Image.Image, mask: Image.Image) -> Image.Image:
    alpha = mask.getchannel("A") if mask.mode == "RGBA" else mask.convert("L")
    dim = Image.blend(img, Image.new("RGB", img.size, (255, 0, 160)), 0.55)
    return Image.composite(img, dim, alpha)


def contact_sheet(rows: list[dict], raw: Path, out: Path, n: int = 24, seed: int = 0, cols: int = 6,
                  thumb: int = 200) -> list[int]:
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    sheet = Image.new("RGB", (cols * thumb * 2, -(-len(sample) // cols) * (thumb + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    zips: dict[str, zipfile.ZipFile] = {}
    for k, r in enumerate(sample):
        zi = zips.setdefault(r["img_zip"], zipfile.ZipFile(raw / r["img_zip"]))
        zm = zips.setdefault(r["mask_zip"], zipfile.ZipFile(raw / r["mask_zip"]))
        img = Image.open(io.BytesIO(zi.read(r["img_member"]))).convert("RGB")
        mask = Image.open(io.BytesIO(zm.read(r["mask_member"])))
        x, y = (k % cols) * thumb * 2, (k // cols) * (thumb + 28)
        for offset, tile in enumerate([img, overlay(img, mask)]):
            tile.thumbnail((thumb, thumb))
            sheet.paste(tile, (x + offset * thumb, y))
        draw.text((x + 2, y + thumb + 2), f'{r["id"]} {r["category"][:24]}', fill="black")
        draw.text((x + 2, y + thumb + 14), f'fg {r["fg_frac"]:.2f} c{r["n_components"]} e{r["edges_touched"]}',
                  fill="black")
    for z in zips.values():
        z.close()
    sheet.save(out, quality=88)
    return [r["id"] for r in sample]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nicho y alertas de máscara de BG60k.")
    parser.add_argument("--data", type=Path, default=Path("data/bg60k"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/bg60k"))
    parser.add_argument("--sheet", help="alerta, nivel de nicho o categoría a mostrar en una hoja de contacto")
    parser.add_argument("--out-sheet", type=Path, default=Path("data/bg60k/sheet.jpg"))
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rows = apply(args.data / "manifest.parquet")
    if args.sheet:
        chosen = [r for r in rows if args.sheet in r["alerts"] or args.sheet in (r["niche"], r["category"])]
        ids = contact_sheet(chosen, args.raw, args.out_sheet, args.n, args.seed)
        print(f"{len(chosen)} filas; hoja en {args.out_sheet}; ids (semilla {args.seed}): {ids}")
        return 0

    pq.write_table(pa.Table.from_pylist(rows), args.data / "candidates.parquet")
    write_category_review(rows)
    summary = summarize(rows) | {"manifest_sha256": sha256(args.data / "manifest.parquet"),
                                 "rules_sha256": sha256(RULES)}
    (args.data / "filters_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "unclassified_top"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
