"""Revisión humana de pares de identidad de BG60k en una página local (etapa 4 de docs/10).

    uv run --group dataset python training/bg60k_review.py sample          # arma la muestra una vez
    uv run --group dataset python training/bg60k_review.py serve --reviewer isaac
    uv run --group dataset python training/bg60k_review.py serve --audit split   # pares del split (etapa 9)

Abre http://127.0.0.1:8765. La página no muestra el coseno ni la banda, para no sesgar. Cada
decisión se agrega a `data/bg60k/review/identity_decisions.csv` (si se corrige un par, vale la
última); la muestra y su semilla quedan en `identity_sample.json`. Las imágenes se leen de los
zip y no salen de la laptop.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import time
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pyarrow.parquet as pq
from PIL import Image

BANDS = [(0.6, 0.7), (0.7, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 0.98), (0.98, 1.01)]
PER_BAND = 40
PHASH_CLOSE = 6
LABELS = {"same": "mismo producto", "variant": "variante del mismo modelo", "different": "distinto",
          "unsure": "no sé"}
FIELDS = ["time_utc", "reviewer", "pair", "a", "b", "label"]


def build_sample(neighbors: Path, seed: int = 0) -> dict:
    rows = [r for r in pq.read_table(neighbors).to_pylist() if r["rank"] == 0]
    seen, unique = set(), []
    for r in rows:
        key = tuple(sorted((r["id"], r["neighbor"])))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    rng = random.Random(seed)
    strata = {f"[{a}, {b})": [r for r in unique if a <= r["cosine"] < b] for a, b in BANDS}
    strata["phash_close"] = [r for r in unique if r["phash_distance"] <= PHASH_CLOSE]
    pairs = []
    for name, pool in strata.items():
        for r in rng.sample(pool, min(PER_BAND, len(pool))):
            pairs.append({"a": r["id"], "b": r["neighbor"], "stratum": name, "cosine": r["cosine"],
                          "phash_distance": r["phash_distance"], "stratum_size": len(pool)})
    rng.shuffle(pairs)
    for k, p in enumerate(pairs):
        p["pair"] = k
    return {"seed": seed, "per_band": PER_BAND, "phash_close": PHASH_CLOSE, "neighbors_rank": 0, "pairs": pairs}


class Images:
    def __init__(self, candidates: Path, raw: Path) -> None:
        self.rows = {r["id"]: r for r in pq.read_table(candidates).to_pylist()}
        self.raw = raw
        self.zips: dict[str, zipfile.ZipFile] = {}

    def jpeg(self, item: int, side: int = 480) -> bytes:
        r = self.rows[item]
        if r["img_zip"] not in self.zips:
            self.zips[r["img_zip"]] = zipfile.ZipFile(self.raw / r["img_zip"])
        img = Image.open(io.BytesIO(self.zips[r["img_zip"]].read(r["img_member"]))).convert("RGB")
        img.thumbnail((side, side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def load_decisions(path: Path, reviewer: str) -> dict[int, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {int(r["pair"]): r["label"] for r in csv.DictReader(f) if r["reviewer"] == reviewer}


def append_decision(path: Path, reviewer: str, pair: dict, label: str) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "reviewer": reviewer,
                         "pair": pair["pair"], "a": pair["a"], "b": pair["b"], "label": label})


PAGE = """<!doctype html><html lang="es"><meta charset="utf-8"><title>Revisión de identidad</title>
<style>
 body{font:15px system-ui,sans-serif;margin:0;background:#f4f4f2;color:#1a1a1a}
 header{padding:12px 20px;background:#fff;border-bottom:1px solid #ddd;display:flex;gap:16px;align-items:center}
 main{display:flex;gap:16px;justify-content:center;padding:20px}
 img{width:480px;height:480px;object-fit:contain;background:#fff;border:1px solid #ccc}
 .bar{display:flex;gap:10px;justify-content:center}
 button{font:inherit;padding:10px 16px;border:1px solid #999;border-radius:6px;background:#fff;cursor:pointer}
 button:hover{background:#eee} .muted{color:#666}
</style>
<header><b>¿Es el mismo producto?</b><span id="progress" class="muted"></span>
<span class="muted">Variante: mismo modelo en otro color, talla o empaque.</span></header>
<main><img id="a" alt=""><img id="b" alt=""></main>
<div class="bar">
 <button data-label="same">1 · Mismo producto</button>
 <button data-label="variant">2 · Variante del mismo modelo</button>
 <button data-label="different">3 · Distinto</button>
 <button data-label="unsure">0 · No sé</button>
 <button id="undo">⌫ Volver al anterior</button>
</div>
<script>
let state;
async function load(back){
  state = await (await fetch('/api/next' + (back ? '?back=1' : ''))).json();
  document.getElementById('progress').textContent = state.done + ' de ' + state.total + ' revisados';
  if(state.pair === null){ document.querySelector('main').innerHTML = '<p>Listo: no quedan pares.</p>'; return; }
  document.getElementById('a').src = '/img/' + state.a;
  document.getElementById('b').src = '/img/' + state.b;
}
async function decide(label){
  if(!state || state.pair === null) return;
  await fetch('/api/decide', {method:'POST', body: JSON.stringify({pair: state.pair, label})});
  load(false);
}
document.querySelectorAll('[data-label]').forEach(b => b.onclick = () => decide(b.dataset.label));
document.getElementById('undo').onclick = () => load(true);
const keys = {'1':'same','2':'variant','3':'different','0':'unsure'};
addEventListener('keydown', e => { if(keys[e.key]) decide(keys[e.key]); if(e.key === 'Backspace') load(true); });
load(false);
</script></html>"""


def make_handler(sample: dict, images: Images, decisions_path: Path, reviewer: str):
    pairs = {p["pair"]: p for p in sample["pairs"]}
    decided = load_decisions(decisions_path, reviewer)
    history: list[int] = sorted(decided, key=lambda k: k)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/":
                return self.send(PAGE.encode(), "text/html; charset=utf-8")
            if url.path.startswith("/img/") and url.path[5:].isdigit() and int(url.path[5:]) in images.rows:
                return self.send(images.jpeg(int(url.path[5:])), "image/jpeg")
            if url.path == "/api/next":
                if "back" in parse_qs(url.query) and history:
                    pending = history.pop()
                    decided.pop(pending, None)
                else:
                    pending = next((k for k in sorted(pairs) if k not in decided), None)
                pair = pairs.get(pending)
                body = {"pair": pending, "a": pair and pair["a"], "b": pair and pair["b"],
                        "done": len(decided), "total": len(pairs)}
                return self.send(json.dumps(body).encode(), "application/json")
            self.send(b"", "text/plain", HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/decide":
                return self.send(b"", "text/plain", HTTPStatus.NOT_FOUND)
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            pair, label = pairs.get(data.get("pair")), data.get("label")
            if pair is None or label not in LABELS:
                return self.send(b"", "text/plain", HTTPStatus.BAD_REQUEST)
            append_decision(decisions_path, reviewer, pair, label)
            decided[pair["pair"]] = label
            history.append(pair["pair"])
            self.send(b"{}", "application/json")

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Revisión de identidad de BG60k.")
    parser.add_argument("command", choices=("sample", "serve"))
    parser.add_argument("--data", type=Path, default=Path("data/bg60k"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/bg60k"))
    parser.add_argument("--reviewer", default="isaac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--audit", choices=("identity", "split"), default="identity")
    args = parser.parse_args(argv)
    review = args.data / "review"
    review.mkdir(parents=True, exist_ok=True)
    sample_path = review / {"identity": "identity_sample.json", "split": "split_audit.json"}[args.audit]

    if args.command == "sample":
        if args.audit == "split":
            raise SystemExit("Los pares del split los arma bg60k_split.py plan.")
        if sample_path.exists():
            raise SystemExit(f"{sample_path} ya existe: la muestra no se rehace para no perder la trazabilidad.")
        sample = build_sample(args.data / "identity_neighbors.parquet", args.seed)
        sample_path.write_text(json.dumps(sample, indent=1))
        print(f"{len(sample['pairs'])} pares en {sample_path}")
        return 0

    sample = json.loads(sample_path.read_text())
    images = Images(args.data / "candidates.parquet", args.raw)
    handler = make_handler(sample, images, review / f"{args.audit}_decisions.csv", args.reviewer)
    server = HTTPServer(("127.0.0.1", args.port), handler)
    print(f"Revisión en http://127.0.0.1:{args.port} (Ctrl+C para terminar)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
