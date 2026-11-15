"""Página local de revisión de identidad de BG60k, con imágenes y vecinos sintéticos."""

from __future__ import annotations

import io
import json
import sys
import threading
import urllib.request
import zipfile
from http.server import HTTPServer
from pathlib import Path

import pytest

pq = pytest.importorskip("pyarrow.parquet")
import pyarrow as pa  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import bg60k_review as review  # noqa: E402


def neighbors(tmp_path: Path) -> Path:
    rows = [{"id": a, "neighbor": b, "rank": 0, "cosine": c, "phash_distance": h}
            for a, b, c, h in [(1, 2, 0.99, 2), (2, 1, 0.99, 2), (3, 4, 0.75, 30), (5, 6, 0.65, 30)]]
    path = tmp_path / "n.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_sample_dedupes_symmetric_pairs_and_records_strata(tmp_path):
    sample = review.build_sample(neighbors(tmp_path), seed=1)
    keys = [tuple(sorted((p["a"], p["b"]))) for p in sample["pairs"]]
    assert keys.count((1, 2)) == 2  # banda [0.98, 1.01) y phash_close
    assert {p["stratum"] for p in sample["pairs"]} == {"[0.98, 1.01)", "phash_close", "[0.7, 0.8)", "[0.6, 0.7)"}
    assert sorted(p["pair"] for p in sample["pairs"]) == list(range(len(sample["pairs"])))
    assert sample["seed"] == 1


def serve(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 50, 50)).save(buf, format="PNG")
    with zipfile.ZipFile(raw / "bg60k_imgs_0.zip", "w") as z:
        for i in (10, 20):
            z.writestr(f"bg60k_imgs_0/{i}.png", buf.getvalue())
    cands = tmp_path / "c.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": i, "img_zip": "bg60k_imgs_0.zip",
                                          "img_member": f"bg60k_imgs_0/{i}.png"} for i in (10, 20)]), cands)
    sample = {"pairs": [{"pair": 0, "a": 10, "b": 20}, {"pair": 1, "a": 20, "b": 10}]}
    decisions = tmp_path / "d.csv"
    handler = review.make_handler(sample, review.Images(cands, raw), decisions, "prueba")
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}", decisions


def get(url: str):
    with urllib.request.urlopen(url) as r:
        return r.read()


def post(url: str, body: dict) -> int:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def test_review_flow_records_decisions_and_undo(tmp_path):
    server, base, decisions = serve(tmp_path)
    try:
        assert b"mismo producto" in get(base + "/").lower()
        assert get(base + "/img/10")[:2] == b"\xff\xd8"
        first = json.loads(get(base + "/api/next"))
        assert first["pair"] == 0 and "cosine" not in first and first["total"] == 2
        assert post(base + "/api/decide", {"pair": 0, "label": "same"}) == 200
        assert json.loads(get(base + "/api/next"))["pair"] == 1
        assert json.loads(get(base + "/api/next?back=1"))["pair"] == 0
        assert post(base + "/api/decide", {"pair": 0, "label": "variant"}) == 200
        assert post(base + "/api/decide", {"pair": 0, "label": "maybe"}) == 400
        assert review.load_decisions(decisions, "prueba") == {0: "variant"}
        with pytest.raises(urllib.error.HTTPError):
            get(base + "/img/../../etc/passwd")
    finally:
        server.shutdown()
