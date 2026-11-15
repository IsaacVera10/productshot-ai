"""Selección del piloto de BG60k con radio de exclusión de identidad (etapas 7 y 9 de docs/10).

    uv run --group dataset python training/bg60k_pilot.py --run data/bg60k/embeddings/<RUN_ID> --simulate

La selección es en línea: por turnos fijos entre estilos, cada estilo presenta su siguiente mejor
candidato según el z de CLIP que respete los topes (`near` y familia, contados sobre aceptados y
pendientes) y no se parezca a ningún aceptado o pendiente (imagen duplicada, coseno DINOv2,
pHash del recorte, ids cercanos de la misma categoría). La persona acepta o rechaza; un rechazo
deja que la misma cola presente el siguiente. El estado se reconstruye reproduciendo el registro
de decisiones, así que la selección es reproducible y se puede deshacer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from bg60k_manifest import hamming
from bg60k_style import STYLES

FAMILIES = Path(__file__).with_name("bg60k_families.toml")
CONFIG = {"target": 150, "near_max": 30, "per_family": 4, "radius": 0.75, "phash_radius": 7,
          "id_radius": 10, "order": STYLES}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_lookup(path: Path = FAMILIES) -> dict[str, str]:
    families = tomllib.loads(path.read_text(encoding="utf-8"))
    return {member: name for name, members in families.items() for member in members}


def family(row: dict, lookup: dict[str, str]) -> str:
    rule = row["niche_rule"]
    keyword = "" if rule.startswith("exact:") else rule
    return lookup.get(row["category"]) or lookup.get(keyword) or keyword or row["category"]


def load(data: Path, run: Path) -> tuple[list[dict], np.ndarray, dict]:
    style_run = json.loads((data / "style_summary.json").read_text())["run"]
    if style_run != run.name:
        raise SystemExit(f"style_scores.parquet es de la corrida {style_run}, no de {run.name}.")
    candidates = {r["id"]: r for r in pq.read_table(data / "candidates.parquet").to_pylist()}
    scores = {r["id"]: r for r in pq.read_table(data / "style_scores.parquet").to_pylist()}
    dino = np.load(run / "dino.npz")
    emb = dino["dino"].astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    lookup = family_lookup()
    rows = []
    for k, i in enumerate(dino["ids"].tolist()):
        r = candidates[i] | scores[i] | {"row": k}
        if not r["alerts"] and r["niche"] in ("core", "near"):
            rows.append(r | {"family": family(r, lookup)})
    provenance = {"run": run.name, "candidates_sha256": sha256(data / "candidates.parquet"),
                  "style_scores_sha256": sha256(data / "style_scores.parquet"),
                  "dino_sha256": sha256(run / "dino.npz"), "families_sha256": sha256(FAMILIES),
                  "config": CONFIG}
    return rows, emb, provenance


@dataclass
class Pilot:
    rows: list[dict]
    emb: np.ndarray
    config: dict = field(default_factory=lambda: dict(CONFIG))

    def __post_init__(self) -> None:
        self.queues = {s: sorted((r for r in self.rows if r["style_z"] == s), key=lambda r: -r[f"z_{s}"])
                       for s in self.config["order"]}
        self.position = dict.fromkeys(self.config["order"], 0)
        self.status: dict[int, tuple[str, str]] = {}
        self.active: list[dict] = []
        self.skips: list[dict] = []
        self.turn = 0
        self.pending: tuple[str, dict] | None = None

    def accepted(self, style: str) -> int:
        return sum(1 for s, st in self.status.values() if s == style and st == "accepted")

    def holding(self, style: str) -> list[dict]:
        return [r for r in self.active if self.status[r["id"]][0] == style]

    def conflict(self, r: dict) -> tuple[str, int] | None:
        if not self.active:
            return None
        cos = self.emb[[a["row"] for a in self.active]] @ self.emb[r["row"]]
        for a, c in zip(self.active, cos):
            if a["img_sha256"] == r["img_sha256"]:
                return "duplicate", a["id"]
            if c >= self.config["radius"]:
                return "dino", a["id"]
            if hamming(a["phash_cutout"], r["phash_cutout"]) < self.config["phash_radius"]:
                return "phash", a["id"]
            if abs(a["id"] - r["id"]) <= self.config["id_radius"] and a["category"] == r["category"]:
                return "id_near", a["id"]
        return None

    def next_for(self, style: str) -> dict | None:
        queue, mine = self.queues[style], self.holding(style)
        while self.position[style] < len(queue):
            r = queue[self.position[style]]
            self.position[style] += 1
            if r["niche"] == "near" and sum(m["niche"] == "near" for m in mine) >= self.config["near_max"]:
                self.skips.append({"style": style, "id": r["id"], "reason": "near_cap"})
                continue
            if sum(m["family"] == r["family"] for m in mine) >= self.config["per_family"]:
                self.skips.append({"style": style, "id": r["id"], "reason": "family_cap"})
                continue
            if hit := self.conflict(r):
                self.skips.append({"style": style, "id": r["id"], "reason": hit[0], "blocked_by": hit[1]})
                continue
            return r
        return None

    def present(self) -> tuple[str, dict] | None:
        if self.pending:
            return self.pending
        order = self.config["order"]
        for _ in range(len(order)):
            style = order[self.turn % len(order)]
            self.turn += 1
            if self.accepted(style) >= self.config["target"]:
                continue
            if r := self.next_for(style):
                self.status[r["id"]] = (style, "pending")
                self.active.append(r)
                self.pending = (style, r)
                return self.pending
        return None

    def decide(self, item: int, label: str) -> None:
        if not self.pending or self.pending[1]["id"] != item:
            raise ValueError(f"La decisión sobre {item} no corresponde al candidato presentado.")
        style, r = self.pending
        self.pending = None
        if label == "accept":
            self.status[item] = (style, "accepted")
        else:
            self.status[item] = (style, "rejected")
            self.active.remove(r)

    def summary(self) -> dict:
        accepted = [r for r in self.active if self.status[r["id"]][1] == "accepted"]
        by_style = {s: [r for r in accepted if self.status[r["id"]][0] == s] for s in self.config["order"]}
        return {
            "accepted": {s: len(v) for s, v in by_style.items()},
            "near": {s: sum(r["niche"] == "near" for r in v) for s, v in by_style.items()},
            "families": {s: len({r["family"] for r in v}) for s, v in by_style.items()},
            "rejected": sum(1 for _, st in self.status.values() if st == "rejected"),
            "skips": dict(Counter(k["reason"] for k in self.skips)),
            "queue_left": {s: len(self.queues[s]) - self.position[s] for s in self.config["order"]},
        }


def replay(rows: list[dict], emb: np.ndarray, decisions: list[dict]) -> Pilot:
    """Reconstruye el estado; una decisión `undo` anula la anterior."""
    effective: list[dict] = []
    for d in decisions:
        if d["label"] == "undo":
            if effective:
                effective.pop()
        else:
            effective.append(d)
    pilot = Pilot(rows, emb)
    for d in effective:
        style, r = pilot.present()
        if r["id"] != int(d["id"]):
            raise SystemExit(f"El registro presentó {d['id']} y ahora toca {r['id']}: cambió la configuración.")
        pilot.decide(r["id"], "accept" if d["label"] == "accept" else "reject")
    return pilot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Selección del piloto de BG60k.")
    parser.add_argument("--run", type=Path, required=True, help="carpeta de la corrida embed")
    parser.add_argument("--data", type=Path, default=Path("data/bg60k"))
    parser.add_argument("--simulate", action="store_true", help="acepta todo para medir disponibilidad")
    args = parser.parse_args(argv)

    rows, emb, provenance = load(args.data, args.run)
    if args.simulate:
        pilot = Pilot(rows, emb)
        while presented := pilot.present():
            pilot.decide(presented[1]["id"], "accept")
        report = provenance | pilot.summary()
        (args.data / "pilot_simulation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(pilot.summary(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
