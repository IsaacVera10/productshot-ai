"""Split del piloto de BG60k con auditoría de identidad entre particiones (etapa 9 de docs/10).

    uv run --group dataset python training/bg60k_split.py plan --run data/bg60k/embeddings/<RUN_ID>
    uv run --group dataset python training/bg60k_review.py serve --audit split
    uv run --group dataset python training/bg60k_split.py apply --run data/bg60k/embeddings/<RUN_ID>

`plan` toma las aceptadas del piloto menos las exclusiones, recorta cada estilo a `CAP` en orden de
aceptación, reparte train, val y test por estilo con semilla fija y arma los pares a auditar: los
10 vecinos DINOv2 de cada imagen de val y test en las otras particiones, más todo par entre
particiones con ids a 10 o menos o pHash < 15, más una muestra aleatoria de pares restantes para
estimar el riesgo residual. Si hay decisiones de otro revisor (por ejemplo una primera pasada
asistida), las de `--reviewer` mandan sobre ellas. Ante la duda (`unsure`), el par cuenta como
coincidencia y su componente va a train. `apply` junta las coincidencias confirmadas en componentes, mueve cada
componente entero a train y, si aparecen pares nuevos sin revisar, los agrega como otra ronda.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from bg60k_manifest import hamming
from bg60k_pilot import load, replay

REVIEW = Path(__file__).with_name("review")
CAP = 60
FRACTIONS = {"val": 0.1, "test": 0.1}
NEIGHBORS = 10
ID_RADIUS = 10
PHASH_RADIUS = 15
RESIDUAL = 100
MATCH = {"same", "variant", "unsure"}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def selected(rows: list[dict], emb: np.ndarray, decisions: list[dict], excluded: set[int]) -> list[dict]:
    pilot = replay(rows, emb, decisions)
    by_id = {r["id"]: r for r in rows}
    order = [int(d["id"]) for d in decisions if d["label"] == "accept"]
    picked, count = [], {}
    for i in dict.fromkeys(order):
        style, status = pilot.status.get(i, ("", ""))
        if status != "accepted" or i in excluded or count.get(style, 0) >= CAP:
            continue
        count[style] = count.get(style, 0) + 1
        picked.append(by_id[i] | {"style": style})
    return picked


def assign(items: list[dict], seed: int) -> dict[int, str]:
    rng = random.Random(seed)
    split = {}
    for style in sorted({r["style"] for r in items}):
        ids = sorted(r["id"] for r in items if r["style"] == style)
        rng.shuffle(ids)
        cut = {k: max(1, round(f * len(ids))) for k, f in FRACTIONS.items()}
        for k, i in enumerate(ids):
            split[i] = "val" if k < cut["val"] else "test" if k < cut["val"] + cut["test"] else "train"
    return split


def audit_pairs(items: list[dict], emb: np.ndarray, split: dict[int, str]) -> dict[tuple[int, int], str]:
    pairs: dict[tuple[int, int], str] = {}
    vec = emb[[r["row"] for r in items]]
    cos = vec @ vec.T
    part = np.array([split[r["id"]] for r in items])
    for a, r in enumerate(items):
        if part[a] == "train":
            continue
        other = np.flatnonzero(part != part[a])
        for b in other[np.argsort(-cos[a, other])][:NEIGHBORS]:
            pairs.setdefault(tuple(sorted((r["id"], items[b]["id"]))), "dino")
    for a, r in enumerate(items):
        for b in range(a + 1, len(items)):
            s = items[b]
            if part[a] == part[b]:
                continue
            if abs(r["id"] - s["id"]) <= ID_RADIUS or hamming(r["phash_cutout"], s["phash_cutout"]) < PHASH_RADIUS:
                pairs.setdefault(tuple(sorted((r["id"], s["id"]))), "id_phash")
    return pairs


def residual(items: list[dict], split: dict[int, str], audited: set[tuple[int, int]], seed: int):
    cross = [tuple(sorted((r["id"], s["id"]))) for k, r in enumerate(items) for s in items[k + 1:]
             if split[r["id"]] != split[s["id"]]]
    pool = [p for p in cross if p not in audited]
    return random.Random(seed).sample(pool, min(RESIDUAL, len(pool))), len(pool)


def components(matches: list[tuple[int, int]]) -> list[set[int]]:
    parent: dict[int, int] = {}

    def root(x: int) -> int:
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for a, b in matches:
        parent[root(a)] = root(b)
    groups: dict[int, set[int]] = {}
    for x in parent:
        groups.setdefault(root(x), set()).add(x)
    return list(groups.values())


def labels(decisions: list[dict]) -> dict[tuple[int, int], str]:
    out = {}
    for d in decisions:
        out[tuple(sorted((int(d["a"]), int(d["b"]))))] = d["label"]
    return out


def resolve(split: dict[int, str], decided: dict[tuple[int, int], str]) -> tuple[dict[int, str], list[set[int]]]:
    groups = components([p for p, label in decided.items() if label in MATCH and all(i in split for i in p)])
    split = dict(split)
    for group in groups:
        if len({split[i] for i in group}) > 1:
            for i in group:
                split[i] = "train"
    return split, groups


def counts(items: list[dict], split: dict[int, str]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for r in items:
        out.setdefault(r["style"], {"train": 0, "val": 0, "test": 0})[split[r["id"]]] += 1
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split y auditoría del piloto de BG60k.")
    parser.add_argument("command", choices=("plan", "apply"))
    parser.add_argument("--run", type=Path, required=True, help="carpeta de la corrida embed")
    parser.add_argument("--data", type=Path, default=Path("data/bg60k"))
    parser.add_argument("--reviewer", default="isaac")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rows, emb, provenance = load(args.data, args.run)
    excluded = {int(r["id"]) for r in read_csv(REVIEW / "pilot_exclusions.csv")}
    items = selected(rows, emb, read_csv(REVIEW / "pilot_decisions.csv"), excluded)
    audit_path = args.data / "review" / "split_audit.json"
    rows_log = read_csv(args.data / "review" / "split_decisions.csv")
    decisions = [d for d in rows_log if d["reviewer"] != args.reviewer] + \
        [d for d in rows_log if d["reviewer"] == args.reviewer]

    if args.command == "plan":
        if audit_path.exists():
            raise SystemExit(f"{audit_path} ya existe: el split no se rehace para no perder la trazabilidad.")
        split = assign(items, args.seed)
        found = audit_pairs(items, emb, split)
        sample, population = residual(items, split, set(found), args.seed)
        pairs = [{"a": a, "b": b, "source": s, "round": 0} for (a, b), s in found.items()]
        pairs += [{"a": a, "b": b, "source": "residual", "round": 0} for a, b in sample]
        random.Random(args.seed).shuffle(pairs)
        for k, p in enumerate(pairs):
            p["pair"] = k
        audit = {"seed": args.seed, "cap": CAP, "fractions": FRACTIONS, "neighbors": NEIGHBORS,
                 "id_radius": ID_RADIUS, "phash_radius": PHASH_RADIUS, "residual_population": population,
                 "provenance": provenance, "initial_split": {str(i): s for i, s in split.items()}, "pairs": pairs}
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=1))
        print(json.dumps({"items": len(items), "pairs": len(pairs), "residual": len(sample),
                          "split": counts(items, split)}, indent=1, ensure_ascii=False))
        return 0

    audit = json.loads(audit_path.read_text())
    if audit["provenance"] != provenance or set(map(int, audit["initial_split"])) != {r["id"] for r in items}:
        raise SystemExit("Cambiaron las entradas desde el plan (procedencia, selección o exclusiones).")
    reviewed = {tuple(sorted((p["a"], p["b"]))) for p in audit["pairs"]}
    decided = labels(decisions)
    pending = reviewed - set(decided)
    if pending:
        raise SystemExit(f"Faltan {len(pending)} pares por revisar en la página.")
    split, groups = resolve({int(i): s for i, s in audit["initial_split"].items()}, decided)
    new = set(audit_pairs(items, emb, split)) - reviewed
    if new:
        start, rnd = len(audit["pairs"]), max(p["round"] for p in audit["pairs"]) + 1
        audit["pairs"] += [{"a": a, "b": b, "source": "recheck", "round": rnd, "pair": start + k}
                           for k, (a, b) in enumerate(sorted(new))]
        audit_path.write_text(json.dumps(audit, indent=1))
        print(f"{len(new)} pares nuevos tras mover componentes: revísalos y vuelve a correr apply.")
        return 0

    residual_pairs = [p for p in audit["pairs"] if p["source"] == "residual" and split[p["a"]] != split[p["b"]]]
    positives = sum(decided[tuple(sorted((p["a"], p["b"])))] in MATCH for p in residual_pairs)
    out = REVIEW / "split.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "style", "split"])
        for r in sorted(items, key=lambda r: r["id"]):
            writer.writerow([r["id"], r["style"], split[r["id"]]])
    by_reviewer = {r: sum(d["reviewer"] == r for d in decisions) for r in {d["reviewer"] for d in decisions}}
    summary = {"split": counts(items, split), "decisions_by_reviewer": by_reviewer, "matches": sum(label in MATCH for label in decided.values()),
               "moved_components": [sorted(g) for g in groups if len(g) > 1],
               "residual": {"pairs": len(residual_pairs), "positives": positives,
                            "population": audit["residual_population"]}}
    (REVIEW / "split_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
