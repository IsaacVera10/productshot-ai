"""Resultados de la prueba con usuarios (docs/06): métricas descriptivas a partir de registro.csv.

    uv run python evaluation/prueba_usuarios/analizar.py evaluation/prueba_usuarios/registro.csv

Con unas diez personas se reporta mediana, rango y conteos con su intervalo de Wilson al 95 %,
sin pruebas estadísticas. Columnas: ver la cabecera de registro.csv y el guion. `*_ok` y las
respuestas sí/no aceptan "si"/"no" (o 1/0); los vacíos se ignoran.
"""

from __future__ import annotations

import csv
import math
import sys
from statistics import mean, median

TASKS = ("u1", "u2", "u3", "u4")


def sus_score(answers: list[int]) -> float:
    """System Usability Scale: ítems impares suman (r-1), pares suman (5-r); total × 2.5 (0 a 100)."""
    if len(answers) != 10 or any(not 1 <= a <= 5 for a in answers):
        raise ValueError("SUS necesita 10 respuestas de 1 a 5.")
    return 2.5 * sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _yes(v: str) -> bool | None:
    v = (v or "").strip().lower()
    if v in ("si", "sí", "1", "true", "s"):
        return True
    if v in ("no", "0", "false", "n"):
        return False
    return None


def _num(v: str) -> float | None:
    try:
        return float((v or "").replace(",", "."))
    except ValueError:
        return None


def _rate(values: list[bool | None]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "sin datos"
    k, n = sum(vals), len(vals)
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({k / n:.0%}; IC 95 % {lo:.0%}–{hi:.0%})"


def _stats(values: list[float | None]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "sin datos"
    return f"mediana {median(vals):g} (rango {min(vals):g}–{max(vals):g}, n={len(vals)})"


def analyze(rows: list[dict]) -> str:
    rows = [r for r in rows if (r.get("codigo") or "").strip() and _yes(r.get("consentimiento")) is not False]
    out = [f"# Resultados de la prueba con usuarios (n = {len(rows)})", ""]
    modes = {}
    for r in rows:
        modes[r.get("modo", "")] = modes.get(r.get("modo", ""), 0) + 1
    out.append(f"Modos: {', '.join(f'{m or "sin dato"}: {c}' for m, c in modes.items())}")
    out += ["", "| Tarea | Éxito sin ayuda | Tiempo (s) | SEQ (1 a 7) |", "|---|---|---|---|"]
    for t in TASKS:
        out.append(f"| {t.upper()} | {_rate([_yes(r.get(f'{t}_ok')) for r in rows])} | "
                   f"{_stats([_num(r.get(f'{t}_seg')) for r in rows])} | {_stats([_num(r.get(f'{t}_seq')) for r in rows])} |")
    sus = []
    for r in rows:
        ans = [_num(r.get(f"sus{i}")) for i in range(1, 11)]
        if all(a is not None for a in ans):
            sus.append(sus_score([int(a) for a in ans]))
    out += ["", "| Métrica | Resultado | Meta (docs/06) |", "|---|---|---|"]
    out.append(f"| SUS | " + (f"promedio {mean(sus):.1f} (rango {min(sus):.1f}–{max(sus):.1f}, n={len(sus)})"
                               if sus else "sin datos") + " | 68 o más |")
    out.append(f"| El resumen refleja lo que decidió (U3) | {_rate([_yes(r.get('u3_refleja')) for r in rows])} | 10 de 10 |")
    out.append(f"| La ronda 2 tiene en cuenta lo que eligió (U4) | {_rate([_yes(r.get('u4_coherente')) for r in rows])} | — |")
    out.append(f"| Espera aceptable (1 a 5) | {_stats([_num(r.get('espera_aceptable')) for r in rows])} | — |")
    out.append(f"| Espera real de generación (s) | {_stats([_num(r.get('espera_gen_seg')) for r in rows])} | — |")
    out.append(f"| Errores por participante | {_stats([_num(r.get('errores')) for r in rows])} | — |")
    out.append(f"| Pedidos de ayuda | {_stats([_num(r.get('ayudas')) for r in rows])} | — |")
    r1 = [_num(r.get("aceptadas_r1")) for r in rows]
    r2 = [_num(r.get("aceptadas_r2")) for r in rows]
    out.append(f"| Aceptadas de 5, ronda 1 | {_stats(r1)} | — |")
    out.append(f"| Aceptadas de 5, ronda 2 | {_stats(r2)} | — |")
    out += ["", "## Comentarios", ""]
    for r in rows:
        for k, label in (("problemas", "Problemas"), ("mas_confuso", "Lo más confuso"), ("cambiaria", "Cambiaría"),
                         ("usaria", "¿Lo usaría?")):
            if (r.get(k) or "").strip():
                out.append(f"- {r['codigo']} · {label}: {r[k].strip()}")
    return "\n".join(out)


def main(path: str) -> None:
    with open(path, newline="", encoding="utf-8") as fh:
        print(analyze(list(csv.DictReader(fh))))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluation/prueba_usuarios/registro.csv")
