"""Revisión humana del piloto de BG60k en una página local (etapa 7 de docs/10).

    uv run --group dataset python training/bg60k_pilot_review.py --run data/bg60k/embeddings/<RUN_ID>

Abre http://127.0.0.1:8765. Muestra un candidato a la vez con el estilo que se le propone; la
persona lo acepta o lo rechaza con un motivo, y un rechazo hace que ese estilo presente el
siguiente de su cola. Cada decisión se agrega a `data/bg60k/review/pilot_decisions.csv`; al
reiniciar, el estado se reconstruye desde ese registro.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from bg60k_pilot import load, replay
from bg60k_review import Images

LABELS = {
    "accept": "aceptar",
    "reject_style": "no corresponde al estilo",
    "reject_material": "transparente, brillante o reflectante (no mate)",
    "reject_multiple": "más de un producto",
    "reject_food": "comida servida o sin envase",
    "reject_text": "texto o logo grande en el fondo",
    "reject_other": "otro problema",
}
STYLE_ES = {
    "studio": "Estudio: fondo gris claro continuo, luz suave de estudio, sombra suave.",
    "natural": "Natural: exterior o superficie natural, luz de día, vegetación desenfocada.",
    "warm": "Cálido: superficie de madera, luz dorada, tonos ámbar, fondo desenfocado.",
    "lifestyle": "Lifestyle: escena cotidiana de casa (cocina, sala) con algunos objetos de fondo.",
    "minimal": "Minimalista: fondo liso blanco roto, mucho espacio vacío, sin accesorios.",
}
FIELDS = ["time_utc", "reviewer", "style", "id", "label"]

PAGE = """<!doctype html><html lang="es"><meta charset="utf-8"><title>Revisión del piloto</title>
<style>
 body{font:15px system-ui,sans-serif;margin:0;background:#f4f4f2;color:#1a1a1a}
 header{padding:12px 20px;background:#fff;border-bottom:1px solid #ddd}
 main{display:flex;gap:24px;justify-content:center;padding:20px;align-items:flex-start}
 img{width:640px;height:640px;object-fit:contain;background:#fff;border:1px solid #ccc}
 aside{max-width:360px} h2{margin:0 0 6px} .muted{color:#666}
 button{display:block;width:100%;text-align:left;font:inherit;padding:9px 12px;margin:6px 0;
        border:1px solid #999;border-radius:6px;background:#fff;cursor:pointer}
 button:hover{background:#eee} #accept{border-color:#2e7d32;font-weight:600}
</style>
<header><b>Piloto de BG60k</b> <span id="progress" class="muted"></span></header>
<main><img id="img" alt="">
<aside>
 <h2 id="style"></h2><p id="desc" class="muted"></p>
 <p class="muted">¿Sirve como ejemplo de este estilo, con un solo producto mate y sin comida?</p>
 <button id="accept" data-label="accept">A · Aceptar</button>
 <button data-label="reject_style">1 · No corresponde al estilo</button>
 <button data-label="reject_material">2 · Transparente, brillante o reflectante</button>
 <button data-label="reject_multiple">3 · Más de un producto</button>
 <button data-label="reject_food">4 · Comida servida o sin envase</button>
 <button data-label="reject_text">5 · Texto o logo grande en el fondo</button>
 <button data-label="reject_other">6 · Otro problema</button>
 <button id="undo">⌫ Deshacer la última decisión</button>
</aside></main>
<script>
let state;
async function load(){
  state = await (await fetch('/api/next')).json();
  document.getElementById('progress').textContent = state.progress;
  if(state.id === null){ document.querySelector('main').innerHTML = '<p>Listo: el piloto está completo.</p>'; return; }
  document.getElementById('img').src = '/img/' + state.id;
  document.getElementById('style').textContent = state.style_label;
  document.getElementById('desc').textContent = state.description;
}
async function send(label){
  if(!state || (state.id === null && label !== 'undo')) return;
  await fetch('/api/decide', {method:'POST', body: JSON.stringify({id: state.id, label})});
  load();
}
document.querySelectorAll('[data-label]').forEach(b => b.onclick = () => send(b.dataset.label));
document.getElementById('undo').onclick = () => send('undo');
const keys = {'a':'accept','Enter':'accept','1':'reject_style','2':'reject_material','3':'reject_multiple',
              '4':'reject_food','5':'reject_text','6':'reject_other','Backspace':'undo'};
addEventListener('keydown', e => { if(keys[e.key]){ e.preventDefault(); send(keys[e.key]); } });
load();
</script></html>"""


def read_log(path: Path, reviewer: str) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["reviewer"] == reviewer]


def append_log(path: Path, reviewer: str, style: str, item: int, label: str) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "reviewer": reviewer,
                         "style": style, "id": item, "label": label})


def make_handler(rows, emb, images: Images, log_path: Path, reviewer: str):
    state = {"pilot": replay(rows, emb, read_log(log_path, reviewer))}

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
            path = urlparse(self.path).path
            if path == "/":
                return self.send(PAGE.encode(), "text/html; charset=utf-8")
            if path.startswith("/img/") and path[5:].isdigit() and int(path[5:]) in images.rows:
                return self.send(images.jpeg(int(path[5:]), side=640), "image/jpeg")
            if path == "/api/next":
                pilot = state["pilot"]
                presented = pilot.present()
                done = pilot.summary()["accepted"]
                progress = " · ".join(f"{s} {n}/{pilot.config['target']}" for s, n in done.items())
                style, r = presented if presented else (None, None)
                body = {"id": r and r["id"], "style": style, "progress": progress,
                        "style_label": style and STYLE_ES[style].split(":")[0],
                        "description": style and STYLE_ES[style]}
                return self.send(json.dumps(body).encode(), "application/json")
            self.send(b"", "text/plain", HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/decide":
                return self.send(b"", "text/plain", HTTPStatus.NOT_FOUND)
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            label, pilot = data.get("label"), state["pilot"]
            if label == "undo":
                append_log(log_path, reviewer, "", 0, "undo")
                state["pilot"] = replay(rows, emb, read_log(log_path, reviewer))
                return self.send(b"{}", "application/json")
            if label not in LABELS or not pilot.pending or pilot.pending[1]["id"] != data.get("id"):
                return self.send(b"", "text/plain", HTTPStatus.BAD_REQUEST)
            style = pilot.pending[0]
            append_log(log_path, reviewer, style, data["id"], label)
            pilot.decide(data["id"], "accept" if label == "accept" else "reject")
            self.send(b"{}", "application/json")

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Revisión del piloto de BG60k.")
    parser.add_argument("--run", type=Path, required=True, help="carpeta de la corrida embed")
    parser.add_argument("--data", type=Path, default=Path("data/bg60k"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/bg60k"))
    parser.add_argument("--reviewer", default="isaac")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    rows, emb, provenance = load(args.data, args.run)
    review = args.data / "review"
    review.mkdir(parents=True, exist_ok=True)
    (review / "pilot_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    images = Images(args.data / "candidates.parquet", args.raw)
    handler = make_handler(rows, emb, images, review / "pilot_decisions.csv", args.reviewer)
    server = HTTPServer(("127.0.0.1", args.port), handler)
    print(f"Revisión del piloto en http://127.0.0.1:{args.port} (Ctrl+C para terminar)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
