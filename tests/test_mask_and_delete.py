"""Máscara manual (POST /mask, ADR-005) y borrado de la cadena de rondas (DELETE, G11)."""

from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw

from conftest import create_session, decide_all, ready_session

API = "/api/v1/sessions"
SIZE = (320, 240)  # lienzo de trabajo de image_bytes(): menor que work_max_side, no se reduce


def mask_png(size=SIZE, fmt="PNG", fill=None) -> bytes:
    img = Image.new("L", size, 255 if fill is None else fill)
    if fill is None:
        ImageDraw.Draw(img).rectangle([100, 60, 220, 180], fill=0)
    buf = io.BytesIO()
    img.convert("RGB" if fmt == "JPEG" else "L").save(buf, format=fmt)
    return buf.getvalue()


def upload(client, sid, data, filename="mask.png", ctype="image/png"):
    return client.post(f"{API}/{sid}/mask", files={"mask": (filename, data, ctype)})


# --- máscara manual ---


def test_manual_mask_replaces_auto_mask_and_is_used(client, settings):
    sid = create_session(client).json()["id"]
    auto = client.post(f"{API}/{sid}/segment").json()["mask"]
    r = upload(client, sid, mask_png())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "SEGMENTED"
    assert body["mask"]["source"] == "manual" and body["mask"]["simulated"] is False
    # Los archivos de la máscara automática se borraron.
    assert client.get(auto["url"]).status_code == 404
    assert client.get(body["mask"]["preview_url"]).status_code == 200

    assert client.post(f"{API}/{sid}/generate").status_code == 202
    session = client.get(f"{API}/{sid}").json()
    assert session["status"] == "READY_FOR_REVIEW"
    # El generador mock compone con la máscara: el centro del rectángulo negro conserva el producto.
    variation = Image.open(io.BytesIO(client.get(session["variations"][0]["image_url"]).content)).convert("RGB")
    assert variation.getpixel((160, 120)) == (200, 60, 60)


def test_manual_mask_works_without_auto_segmentation(client):
    sid = create_session(client).json()["id"]
    assert upload(client, sid, mask_png()).status_code == 200
    assert client.post(f"{API}/{sid}/generate").status_code == 202


@pytest.mark.parametrize(
    "data, expected",
    [
        (mask_png(size=(100, 100)), "320×240"),
        (mask_png(fmt="JPEG"), "PNG"),
        (mask_png(fill=255), "toda blanca"),
        (mask_png(fill=0), "toda negra"),
        (b"no soy una imagen", "decodificable"),
        (b"", "vacía"),
    ],
)
def test_invalid_manual_mask_is_rejected(client, data, expected):
    sid = create_session(client).json()["id"]
    r = upload(client, sid, data)
    assert r.status_code == 400
    assert expected in r.json()["error"]["message"]
    assert client.get(f"{API}/{sid}").json()["mask"] is None


def test_manual_mask_too_large(client, settings):
    sid = create_session(client).json()["id"]
    r = upload(client, sid, b"\x89PNG" + b"0" * (settings.max_upload_bytes + 10))
    assert r.status_code == 413


def test_manual_mask_locked_after_generation_and_in_child_rounds(client):
    s = ready_session(client)
    assert upload(client, s["id"], mask_png()).status_code == 409
    decide_all(client, s, ["accepted"] * 5)
    client.post(f"{API}/{s['id']}/summary")
    child = client.post(f"{API}/{s['id']}/next-round").json()
    r = upload(client, child["id"], mask_png())
    assert r.status_code == 409 and "ronda 1" in r.json()["error"]["message"]


# --- borrado ---


def _chain_of_two(client) -> tuple[dict, dict]:
    s = ready_session(client)
    decide_all(client, s, ["accepted", "rejected", "rejected", "rejected", "rejected"], ["me gusta", None, None, None, None])
    client.post(f"{API}/{s['id']}/summary")
    child = client.post(f"{API}/{s['id']}/next-round").json()
    client.post(f"{API}/{child['id']}/generate")
    return client.get(f"{API}/{s['id']}").json(), client.get(f"{API}/{child['id']}").json()


@pytest.mark.parametrize("delete_from", ["root", "child"])
def test_delete_removes_the_whole_chain(client, settings, delete_from):
    root, child = _chain_of_two(client)
    other = ready_session(client)  # otra sesión que no debe tocarse
    target = root["id"] if delete_from == "root" else child["id"]

    assert client.delete(f"{API}/{target}").status_code == 204
    for sid in (root["id"], child["id"]):
        assert client.get(f"{API}/{sid}").status_code == 404
        assert not (settings.asset_root / sid).exists()
    assert client.get(root["image_url"]).status_code == 404
    assert client.get(child["variations"][0]["image_url"]).status_code == 404

    assert client.get(f"{API}/{other['id']}").json()["status"] == "READY_FOR_REVIEW"
    assert client.get(other["variations"][0]["image_url"]).status_code == 200
    assert client.delete(f"{API}/{target}").status_code == 404


class _ParkedRunner:
    """Acepta el trabajo y no lo ejecuta: deja la sesión en GENERATING."""

    def submit(self, fn):
        self.fn = fn

    def shutdown(self):
        pass


def test_delete_refuses_while_generating(make_client, settings):
    runner = _ParkedRunner()
    client = make_client(runner=runner)
    sid = create_session(client).json()["id"]
    client.post(f"{API}/{sid}/segment")
    client.post(f"{API}/{sid}/generate")
    r = client.delete(f"{API}/{sid}")
    assert r.status_code == 409
    assert client.get(f"{API}/{sid}").json()["status"] == "GENERATING"
    runner.fn()  # termina la generación
    assert client.delete(f"{API}/{sid}").status_code == 204


# --- UI ---


def test_ui_mask_upload_and_delete(client, monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    import productshot_ui.app as ui

    class _Api(ui.Api):
        def __init__(self, api_base: str):
            super().__init__(api_base)
            self.http = client

    monkeypatch.setattr(ui, "Api", _Api)
    handlers = {f.fn.__name__: f.fn for f in ui.build_ui("http://testserver/api/v1").fns.values()}

    sid = create_session(client).json()["id"]
    path = tmp_path / "m.png"
    path.write_bytes(mask_png())
    _mask, status = handlers["use_mask"](sid, str(path))
    assert "Máscara manual lista" in status
    assert client.get(f"{API}/{sid}").json()["mask"]["source"] == "manual"

    new_sid, _m, status, *_ = handlers["delete"](sid)
    assert new_sid is None and "borrada" in status
    assert client.get(f"{API}/{sid}").status_code == 404
