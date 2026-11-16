"""Resume `metrics.jsonl` de la etapa eval y propone un checkpoint (docs/10).

    python training/lora_eval_report.py data/lora/<RUN>/eval/<EVAL_RUN>

Valida el resultado contra el manifiesto: exactamente una fila por producto completado, estilo
y configuración, valores finitos e imágenes legibles. La medida de estilo es
el margen CLIP (coseno al centroide del estilo pedido menos el mejor de los otros), pareado
contra el LoRA apagado: primero se promedia por producto y el intervalo bootstrap remuestrea
productos. CLIP es una señal auxiliar: la propuesta es un candidato para la revisión humana
ciega, que puede cambiarlo o quedarse sin LoRA.

Regla, fijada antes de ver resultados:
1. Candidato: checkpoint cuyo IC 95 % de la mejora del margen frente a apagado queda sobre 0.
2. Entre candidatos, el de mayor mejora media; se prefiere uno más temprano si el IC 95 % de
   (mejor - temprano) queda bajo TOLERANCE.
3. Sin candidatos, se propone no usar el LoRA.
4. Con menos de MIN_PRODUCTS productos completos no se propone nada: el resultado es provisional.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OFF = "off"
TOLERANCE = 0.005
RESAMPLES = 10_000
DRIFT_WARN = 12.0  # mismo umbral que G8 en producción (Settings.fidelity_warn_diff)
MIN_PRODUCTS = 15
NUMERIC = ("clip_target", "clip_margin", "dino_product", "protected_mean_abs_diff", "seconds")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected(manifest: dict) -> tuple[list[str], list[str], list[int]]:
    cfg = manifest["config"]
    return [OFF, *cfg.get("variants", cfg["adapters"])], list(cfg["style_prompts"]), list(manifest.get("completed", []))


def validate(eval_dir: Path, manifest: dict) -> list[dict]:
    """Filas de metrics.jsonl comprobadas contra el manifiesto; lanza ValueError si algo no cuadra."""
    from PIL import Image

    configs, styles, completed = expected(manifest)
    rows = load_rows(eval_dir / "metrics.jsonl")
    keys = [(r.get("id"), r.get("style"), r.get("config")) for r in rows]
    wanted = {(i, s, c) for i in completed for s in styles for c in configs}
    if len(keys) != len(set(keys)) or set(keys) != wanted:
        raise ValueError(f"metrics.jsonl no tiene exactamente una fila por combinación ({len(rows)} filas, "
                         f"{len(wanted)} esperadas)")
    for r in rows:
        if not all(isinstance(r.get(k), (int, float)) and np.isfinite(r[k]) for k in NUMERIC):
            raise ValueError(f"Fila con valores faltantes o no finitos: {r}")
        if r.get("clip_top1") not in styles:
            raise ValueError(f"clip_top1 desconocido: {r}")
        with Image.open(eval_dir / "images" / r["image"]) as im:
            im.verify()
    refs = np.load(eval_dir / "style_refs.npz")
    centroids = refs["centroids"].astype(np.float64)
    if refs["styles"].tolist() != styles or centroids.shape[0] != len(styles) or not np.isfinite(centroids).all() \
            or np.abs(np.linalg.norm(centroids, axis=1) - 1).max() > 1e-3:
        raise ValueError("style_refs.npz no tiene un centroide normalizado por estilo.")
    return rows


def per_product(rows: list[dict], key: str, configs: list[str]) -> tuple[list[int], np.ndarray]:
    """Matriz productos x configuraciones con la media de `key` sobre los estilos."""
    ids = sorted({r["id"] for r in rows})
    acc = defaultdict(list)
    for r in rows:
        acc[(r["id"], r["config"])].append(float(r[key]))
    return ids, np.array([[np.mean(acc[(i, c)]) for c in configs] for i in ids])


def bootstrap_ci(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (RESAMPLES, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def choose(margin: np.ndarray, configs: list[str]) -> dict:
    if len(margin) < MIN_PRODUCTS:
        return {"proposal": None, "candidates": [],
                "reason": f"provisional: {len(margin)} productos completos, se necesitan {MIN_PRODUCTS}"}
    deltas = {c: margin[:, k] - margin[:, 0] for k, c in enumerate(configs) if c != OFF}
    cis = {c: bootstrap_ci(d) for c, d in deltas.items()}
    candidates = [c for c in deltas if cis[c][0] > 0]
    if not candidates:
        return {"proposal": OFF, "candidates": [], "reason": "ningún checkpoint mejora el margen con IC sobre 0"}
    best = max(candidates, key=lambda c: deltas[c].mean())
    for c in candidates:  # en orden de pasos: el primero equivalente gana
        gap_hi = bootstrap_ci(deltas[best] - deltas[c])[1]
        if c == best or gap_hi < TOLERANCE:
            reason = "mayor mejora media" if c == best else f"equivalente al mejor ({best}) dentro de {TOLERANCE}"
            return {"proposal": c, "candidates": candidates, "best_mean": best, "reason": reason}
    raise AssertionError("inalcanzable")


def report(rows: list[dict], configs: list[str], styles: list[str]) -> tuple[str, dict]:
    ids, margin = per_product(rows, "clip_margin", configs)
    _, target = per_product(rows, "clip_target", configs)
    _, dino = per_product(rows, "dino_product", configs)
    by_config = defaultdict(list)
    for r in rows:
        by_config[r["config"]].append(r)

    header = ("| Config | Coseno al estilo | Margen | Δ margen vs apagado [IC 95 %] | Gana el estilo pedido "
              "| DINOv2 producto | Deriva media | Deriva > 12 |")
    lines = [f"Productos completos: {len(ids)}; estilos: {len(styles)}; imágenes: {len(rows)}", "",
             header, "|---|---|---|---|---|---|---|---|"]
    for k, c in enumerate(configs):
        rs = by_config[c]
        delta = "" if c == OFF else "{:+.4f} [{:+.4f}, {:+.4f}]".format(
            (margin[:, k] - margin[:, 0]).mean(), *bootstrap_ci(margin[:, k] - margin[:, 0]))
        top1 = sum(r["clip_top1"] == r["style"] for r in rs)
        drift = [r["protected_mean_abs_diff"] for r in rs]
        lines.append(f"| {c} | {target[:, k].mean():.4f} | {margin[:, k].mean():+.4f} | {delta} | "
                     f"{top1}/{len(rs)} | {dino[:, k].mean():.3f} | {np.mean(drift):.1f} | "
                     f"{sum(d > DRIFT_WARN for d in drift)}/{len(rs)} |")

    lines += ["", "Δ margen vs apagado por estilo pedido (media sobre productos; indicativo, n chico):", "",
              "| Estilo | " + " | ".join(configs[1:]) + " |", "|---|" + "---|" * (len(configs) - 1)]
    for s in styles:
        sub = [r for r in rows if r["style"] == s]
        _, m = per_product(sub, "clip_margin", configs)
        lines.append(f"| {s} | " + " | ".join(f"{(m[:, k] - m[:, 0]).mean():+.4f}" for k in range(1, len(configs))) + " |")

    if all("text_margin" in r for r in rows):
        # Señal secundaria: margen CLIP contra el texto de cada preset (no contra BG60k). El LoRA por
        # autodestilación se entrenó con candidatas elegidas por esta medida, así que la favorece.
        _, text = per_product(rows, "text_margin", configs)
        lines += ["", "Δ margen de texto vs apagado (secundario, sesgado a favor del LoRA destilado):", ""]
        for k, c in enumerate(configs[1:], start=1):
            d = text[:, k] - text[:, 0]
            lines.append("- {}: {:+.4f} [{:+.4f}, {:+.4f}]".format(c, d.mean(), *bootstrap_ci(d)))

    decision = choose(margin, configs)
    lines += ["", (f"Propuesta: **{decision['proposal']}** ({decision['reason']}). "
                   f"Candidatos: {decision['candidates'] or 'ninguno'}. Pendiente de la revisión humana ciega.")]
    return "\n".join(lines), decision


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", type=Path)
    args = ap.parse_args(argv)
    manifest = json.loads((args.eval_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = validate(args.eval_dir, manifest)
    configs, styles, _ = expected(manifest)
    text, decision = report(rows, configs, styles)
    (args.eval_dir / "report.md").write_text(text + "\n", encoding="utf-8")
    (args.eval_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
