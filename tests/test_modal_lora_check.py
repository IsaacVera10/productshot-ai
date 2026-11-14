"""Verificación del LoRA en Modal: funciones puras de deploy/modal/worker.py sin importar modal.

El módulo del worker importa `modal` al cargarse, que la suite no instala; estas pruebas
extraen con `ast` solo las definiciones que no dependen de él.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "deploy" / "modal" / "worker.py"


def load_pure(*names: str) -> dict:
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    nodes = [
        n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in names)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))
    ]
    ns = {"json": json, "Path": Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(WORKER), "exec"), ns)
    return ns


def test_lora_check_orders_unloaded_configs_first_per_engine():
    """En cada motor, lo que va sin LoRA tiene que correr antes de cargarlo (lo exige `Lab`)."""
    sets = load_pure("EXPERIMENT_SETS")["EXPERIMENT_SETS"]
    configs = sets["lora_check"]
    seen_lora = set()
    for name, cfg in configs.items():
        engine = cfg.get("engine", "four")
        assert engine in ("four", "nine")
        if "lora" in cfg:
            seen_lora.add(engine)
        else:
            assert engine not in seen_lora, f"{name} va sin LoRA después de cargarlo"
    engines = [c.get("engine", "four") for c in configs.values()]
    assert engines == sorted(engines, key=lambda e: e == "nine"), "nueve canales debe ir al final"
    # Ningún otro conjunto usa LoRA: `experiment` los lanza sin adaptador.
    assert not any("lora" in c for k, cs in sets.items() if k != "lora_check" for c in cs.values())


def test_pairs_cover_load_repeatability_and_effect():
    ns = load_pure("lora_pairs", "EXPERIMENT_SETS")
    pairs = ns["lora_pairs"](list(ns["EXPERIMENT_SETS"]["lora_check"]))
    assert ("cargar_con_peso_0", "A_sin_lora", "A_lora0") in pairs
    assert ("repetir_peso_0", "G_lora0", "G_lora0_bis") in pairs
    assert {p[1:] for p in pairs if p[0] == "efecto_lora"} == {
        ("A_lora0", "A_lora1"), ("G_lora0", "G_lora1"), ("F_lora0", "F_lora1"), ("K_lora0", "K_lora1"),
    }


def test_background_diff_ignores_the_product():
    ns = load_pure("background_diff")
    mask = Image.new("L", (4, 2), 255)
    mask.putpixel((0, 0), 0)  # producto protegido: no cuenta
    a = Image.new("RGB", (4, 2), (100, 100, 100))
    b = a.copy()
    b.putpixel((0, 0), (255, 255, 255))  # cambia solo el producto
    assert ns["background_diff"](a, b, mask)["mean"] == 0.0
    b.putpixel((1, 0), (130, 130, 130))  # cambia un píxel de fondo de 7
    d = ns["background_diff"](a, b, mask)
    assert d["mean"] == round(30 / 7, 3) and d["frac_gt10"] == round(1 / 7, 4)


def test_verdict_requires_effect_well_above_controls():
    ns = load_pure("lora_pairs", "lora_verdicts")
    configs = ["A_sin_lora", "A_lora0", "A_lora0_bis", "A_lora1", "K_lora0", "K_lora1"]

    def pair(what, a, b, mean):
        return {"measure": what, "a": a, "b": b, "mean": mean}

    out = [
        pair("cargar_con_peso_0", "A_sin_lora", "A_lora0", 0.05),
        pair("repetir_peso_0", "A_lora0", "A_lora0_bis", 0.10),
        pair("efecto_lora", "A_lora0", "A_lora1", 8.0),
        pair("efecto_lora", "K_lora0", "K_lora1", 0.2),  # sin controles propios: usa los de A
    ]
    v = ns["lora_verdicts"](out, configs)
    assert v["A"]["verdict"] == "el LoRA cambia la imagen" and v["A"]["own_controls"]
    assert v["K"]["verdict"] == "inconcluso" and v["K"]["control"] == 0.10 and not v["K"]["own_controls"]
    # Si los controles son comparables al efecto, no se concluye nada.
    out[1] = pair("repetir_peso_0", "A_lora0", "A_lora0_bis", 4.0)
    assert ns["lora_verdicts"](out, configs)["A"]["verdict"] == "inconcluso"
