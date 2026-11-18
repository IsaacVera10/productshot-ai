"""Compara dos configuraciones de evaluaciones distintas y arma la revisión ciega (docs/10).

    python training/lora_compare.py --a <EVAL_A>:ckpt400@1 --b <EVAL_B>:ckpt500@1 [--blind <carpeta>]

Las dos evaluaciones deben cubrir los mismos productos con la misma regla de semillas; el Δ del
margen CLIP (B − A) se promedia primero por producto y el IC bootstrap remuestrea productos, como
en `lora_eval_report.py`. `--blind` escribe 15 parejas (un estilo sorteado por producto, semilla 0)
con el orden A/B al azar y la clave aparte, para que Isaac elija sin saber cuál es cuál.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from lora_eval_report import bootstrap_ci


def load(spec: str) -> tuple[Path, str, dict[tuple[int, str], dict]]:
    folder, _, config = spec.rpartition(":")
    rows = [json.loads(line) for line in (Path(folder) / "metrics.jsonl").open(encoding="utf-8")]
    return Path(folder), config, {(r["id"], r["style"]): r for r in rows if r["config"] == config}


def compare(a: dict, b: dict) -> dict:
    if set(a) != set(b):
        raise SystemExit(f"Las evaluaciones no cubren lo mismo: {len(set(a) ^ set(b))} pares sin pareja.")
    by_product, by_style = defaultdict(list), defaultdict(list)
    for key in a:
        d = b[key]["clip_margin"] - a[key]["clip_margin"]
        by_product[key[0]].append(d)
        by_style[key[1]].append(d)
    deltas = np.array([np.mean(v) for v in by_product.values()])
    lo, hi = bootstrap_ci(deltas)
    return {"products": len(deltas), "delta": round(float(deltas.mean()), 5), "ci95": [round(lo, 5), round(hi, 5)],
            "by_style": {s: round(float(np.mean(v)), 5) for s, v in sorted(by_style.items())},
            "dino": [round(float(np.mean([r["dino_product"] for r in x.values()])), 4) for x in (a, b)]}


def blind(a_dir: Path, a: dict, b_dir: Path, b: dict, out: Path, seed: int = 0) -> None:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    styles = sorted({s for _, s in a})
    key = []
    for n, item in enumerate(sorted({i for i, _ in a})):
        style = rng.choice(styles)
        pair = [("A", a_dir / "images" / a[(item, style)]["image"]), ("B", b_dir / "images" / b[(item, style)]["image"])]
        rng.shuffle(pair)
        for side, (label, path) in zip(("1", "2"), pair):
            shutil.copy(path, out / f"{n:02d}_{side}.jpg")
        key.append({"pair": n, "id": item, "style": style, "1": pair[0][0], "2": pair[1][0]})
    (out / "clave.json").write_text(json.dumps(key, indent=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", required=True, help="evaluación:config de referencia (producción)")
    parser.add_argument("--b", required=True, help="evaluación:config candidata")
    parser.add_argument("--blind", type=Path, help="carpeta para la revisión ciega")
    args = parser.parse_args(argv)
    a_dir, a_cfg, a = load(args.a)
    b_dir, b_cfg, b = load(args.b)
    result = {"a": f"{a_dir.name}:{a_cfg}", "b": f"{b_dir.name}:{b_cfg}", **compare(a, b)}
    print(json.dumps(result, indent=1, ensure_ascii=False))
    if args.blind:
        blind(a_dir, a, b_dir, b, args.blind)
        print(f"Revisión ciega en {args.blind} (la clave está en clave.json: no abrirla antes de elegir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
