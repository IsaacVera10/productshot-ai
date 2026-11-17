"""Análisis de la prueba con usuarios (evaluation/prueba_usuarios/analizar.py)."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analizar", ROOT / "evaluation/prueba_usuarios/analizar.py")
analizar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analizar)


def test_sus_score_matches_the_standard_formula():
    assert analizar.sus_score([3] * 10) == 50.0
    assert analizar.sus_score([5, 1] * 5) == 100.0  # ideal: de acuerdo en impares, en desacuerdo en pares
    assert analizar.sus_score([1, 5] * 5) == 0.0
    with pytest.raises(ValueError):
        analizar.sus_score([3] * 9)


def test_wilson_interval_of_the_calibration():
    lo, hi = analizar.wilson(8, 9)
    assert round(lo, 2) == 0.56 and round(hi, 2) == 0.98  # 56–98 %, como en ADR-010


def test_analyze_reports_rates_and_skips_rows_without_consent():
    header = (ROOT / "evaluation/prueba_usuarios/registro.csv").read_text().strip().split(",")
    base = {k: "" for k in header}
    rows = [
        {**base, "codigo": "P01", "modo": "intermediate", "consentimiento": "si", "u1_ok": "si", "u1_seg": "40",
         "u1_seq": "6", "u3_refleja": "si", "u4_coherente": "si", **{f"sus{i}": "4" if i % 2 else "2" for i in range(1, 11)}},
        {**base, "codigo": "P02", "modo": "intermediate", "consentimiento": "si", "u1_ok": "no", "u3_refleja": "si"},
        {**base, "codigo": "P03", "consentimiento": "no", "u1_ok": "si"},
    ]
    text = analizar.analyze(rows)
    assert "(n = 2)" in text
    assert "| U1 | 1/2 (50%" in text
    assert "promedio 75.0" in text
    assert "2/2 (100%" in text  # el resumen refleja lo decidido
    assert csv  # el módulo usa csv para leer el archivo real
