"""Puntajes de estilo con CLIP para los candidatos de BG60k (etapa 5 de docs/10).

    uv run --group dataset python training/bg60k_style.py --run data/bg60k/embeddings/<RUN_ID>
    uv run --group dataset python training/bg60k_style.py --run ... --sheet studio --order low_margin

No decide el estilo: calcula el puntaje de cada estilo y de cada subclase de `other`, el margen
entre los dos mejores estilos y la cobertura provisional. El puntaje crudo favorece a `studio`
(casi toda foto de producto le parece de estudio a CLIP), así que el ranking usa también el
puntaje estandarizado por estilo (`z_<estilo>`). Las hojas sirven para calibrar a mano.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bg60k_filters import contact_sheet

STYLES = ["studio", "natural", "warm", "lifestyle", "minimal"]


def unit(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def prototypes(text: np.ndarray, groups: list[str]) -> tuple[list[str], np.ndarray]:
    names = list(dict.fromkeys(groups))
    protos = [unit(text[[g == name for g in groups]].mean(axis=0)) for name in names]
    return names, np.stack(protos)


def score(clip_path: Path) -> list[dict]:
    clip = np.load(clip_path)
    names, protos = prototypes(unit(clip["clip_text"]), clip["prompt_group"].tolist())
    sims = unit(clip["clip_image"]) @ protos.T
    style_cols = [names.index(f"style:{s}") for s in STYLES]
    other_names = [n for n in names if n.startswith("other:")]
    other_cols = [names.index(n) for n in other_names]

    std = sims[:, style_cols].std(axis=0)
    z = (sims[:, style_cols] - sims[:, style_cols].mean(axis=0)) / np.where(std > 0, std, 1)

    rows = []
    for i, s, zs in zip(clip["ids"].tolist(), sims, z):
        styles = s[style_cols]
        order = np.argsort(styles)[::-1]
        others = s[other_cols]
        best_other = int(np.argmax(others))
        rows.append({
            "id": i,
            "style": STYLES[order[0]],
            "style_score": float(styles[order[0]]),
            "margin": float(styles[order[0]] - styles[order[1]]),
            "second": STYLES[order[1]],
            "other": other_names[best_other].removeprefix("other:"),
            "other_score": float(others[best_other]),
            "other_minus_style": float(others[best_other] - styles[order[0]]),
            "style_z": STYLES[int(np.argmax(zs))],
            **{f"s_{name}": float(v) for name, v in zip(STYLES, styles)},
            **{f"z_{name}": float(v) for name, v in zip(STYLES, zs)},
        })
    return rows


def quantiles(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {f"p{q}": round(float(np.quantile(v, q / 100)), 4) for q in (5, 25, 50, 75, 95)}


def summarize(rows: list[dict]) -> dict:
    raw, standardized = defaultdict(Counter), defaultdict(Counter)
    for r in rows:
        group = f'{r["niche"]}/{"limpio" if not r["alerts"] else "alertas"}'
        raw[group][r["style"]] += 1
        standardized[group][r["style_z"]] += 1
    return {
        "items": len(rows),
        "best_style_by_group": {k: dict(sorted(v.items())) for k, v in sorted(raw.items())},
        "best_z_style_by_group": {k: dict(sorted(v.items())) for k, v in sorted(standardized.items())},
        "margin": quantiles([r["margin"] for r in rows]),
        "style_score": quantiles([r["style_score"] for r in rows]),
        "other_beats_style": sum(r["other_minus_style"] > 0 for r in rows),
        "other_beats_style_by_subclass": dict(Counter(r["other"] for r in rows if r["other_minus_style"] > 0)),
        "confusions": dict(Counter(f'{r["style"]}>{r["second"]}' for r in rows if r["margin"] < 0.01).most_common(10)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Puntajes de estilo de BG60k con CLIP.")
    parser.add_argument("--run", type=Path, required=True, help="carpeta de la corrida embed")
    parser.add_argument("--data", type=Path, default=Path("data/bg60k"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/bg60k"))
    parser.add_argument("--sheet", help="estilo o subclase de other a mostrar")
    parser.add_argument("--order", choices=("top", "low_margin", "random"), default="random")
    parser.add_argument("--clean", action="store_true", help="solo candidatos sin alertas")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-sheet", type=Path, default=Path("data/bg60k/style_sheet.jpg"))
    args = parser.parse_args(argv)

    candidates = {r["id"]: r for r in pq.read_table(args.data / "candidates.parquet").to_pylist()}
    rows = [candidates[r["id"]] | r for r in score(args.run / "clip.npz")]

    if args.sheet:
        if args.sheet in STYLES:
            chosen = [r for r in rows if r["style_z"] == args.sheet]
        else:
            chosen = [r for r in rows if r["other"] == args.sheet and r["other_minus_style"] > 0]
        if args.clean:
            chosen = [r for r in chosen if not r["alerts"]]
        if args.order == "top" and args.sheet in STYLES:
            chosen = sorted(chosen, key=lambda r: -r[f"z_{args.sheet}"])[:args.n]
        elif args.order == "top":
            chosen = sorted(chosen, key=lambda r: -r["other_score"])[:args.n]
        elif args.order == "low_margin":
            chosen = sorted(chosen, key=lambda r: r["margin"])[:args.n]
        ids = contact_sheet(chosen, args.raw, args.out_sheet, args.n, args.seed)
        print(f"{len(chosen)} filas; hoja en {args.out_sheet}; ids: {ids}")
        return 0

    keep = ["id", "style", "style_score", "margin", "second", "style_z", "other", "other_score",
            "other_minus_style", *[f"s_{s}" for s in STYLES], *[f"z_{s}" for s in STYLES]]
    pq.write_table(pa.Table.from_pylist([{k: r[k] for k in keep} for r in rows]), args.data / "style_scores.parquet")
    summary = summarize(rows) | {"run": args.run.name}
    (args.data / "style_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
