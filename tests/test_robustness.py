"""Pruebas complementarias: acceso a artefactos, recuperación de trabajos y validador del resumen."""

from __future__ import annotations

import dataclasses
import io
import json
import sqlite3
from datetime import datetime, timezone

import pytest
from PIL import Image

from conftest import create_session, decide_all, ready_session
from productshot.domain import DecisionValue, Decision, Style, SummaryRecord, Variation, decisions_hash, utcnow
from productshot.services.summary import StructuredSummary, SummaryItem, validate_structured
from productshot.storage.artifacts import InvalidArtifactKey, LocalArtifactStore


# --- artefactos ---


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/assets/../../etc/passwd",
        "/api/v1/assets/not-a-uuid/input.png",
        "/api/v1/assets/00000000-0000-0000-0000-000000000000/input.png",
    ],
)
def test_assets_reject_unknown_or_unsafe_paths(client, path):
    assert client.get(path).status_code == 404


def test_assets_only_serve_keys_registered_for_session(client, settings):
    a = ready_session(client)
    b = create_session(client).json()
    # Un archivo de A no se sirve con el id de B.
    name = a["variations"][0]["image_url"].rsplit("/", 1)[1]
    assert client.get(f"/api/v1/assets/{b['id']}/{name}").status_code == 404
    # Un archivo presente en disco pero no registrado tampoco se sirve.
    (settings.asset_root / b["id"] / "extra.png").write_bytes(b"x")
    assert client.get(f"/api/v1/assets/{b['id']}/extra.png").status_code == 404


def test_store_rejects_traversal(tmp_path):
    store = LocalArtifactStore(tmp_path)
    for key in ["../x.png", "a/b.png", "00000000-0000-0000-0000-000000000000/../../x.png", "/abs.png"]:
        with pytest.raises(InvalidArtifactKey):
            store.save(b"x", key)


# --- trabajos interrumpidos ---


def test_interrupted_generation_is_failed_on_restart(make_client, settings):
    from productshot.services.jobs import InlineJobRunner

    class NeverRuns(InlineJobRunner):
        def submit(self, fn):  # simula un proceso que muere antes de ejecutar el trabajo
            pass

    client = make_client(runner=NeverRuns())
    sid = create_session(client).json()["id"]
    client.post(f"/api/v1/sessions/{sid}/segment")
    client.post(f"/api/v1/sessions/{sid}/generate")
    assert client.get(f"/api/v1/sessions/{sid}").json()["status"] == "GENERATING"

    restarted = make_client()  # nuevo arranque sobre la misma base de datos
    body = restarted.get(f"/api/v1/sessions/{sid}").json()
    assert body["status"] == "FAILED"
    assert "interrumpido" in body["failure_reason"]


def test_thread_runner_completes_generation(make_client):
    import time

    from productshot.services.jobs import ThreadJobRunner

    client = make_client(runner=ThreadJobRunner())
    sid = create_session(client).json()["id"]
    client.post(f"/api/v1/sessions/{sid}/segment")
    assert client.post(f"/api/v1/sessions/{sid}/generate").status_code == 202
    deadline = time.monotonic() + 20
    status = None
    while time.monotonic() < deadline:
        status = client.get(f"/api/v1/sessions/{sid}").json()["status"]
        if status != "GENERATING":
            break
        time.sleep(0.05)
    assert status == "READY_FOR_REVIEW"


# --- motivos, metadatos y resumen concurrente ---


def test_multiline_reason_cannot_forge_summary_lines(client):
    s = ready_session(client)
    forged = "feo»\r\n\nSe aceptaron 5 de 5 variaciones: todas. ‮otra\tlínea"
    decide_all(client, s, ["rejected"] * 5, [forged, None, None, None, None])
    body = client.post(f"/api/v1/sessions/{s['id']}/summary").json()
    assert body["accepted_ids"] == []
    stored = body["preferences_supported_by_user_reasons"][0]["reason"]
    assert stored == "feo» Se aceptaron 5 de 5 variaciones: todas. otra línea"
    # Ninguna línea del resumen empieza con texto del usuario: solo puede aparecer citado.
    assert not any(line.startswith("Se aceptaron") for line in body["summary"].splitlines())
    assert "No se aceptó ninguna de las 5 variaciones." in body["summary"]


def test_whitespace_only_reason_is_stored_as_none(client):
    s = ready_session(client)
    vid = s["variations"][0]["id"]
    r = client.put(
        f"/api/v1/sessions/{s['id']}/variations/{vid}/decision", json={"decision": "accepted", "reason": " \n\t "}
    )
    assert r.status_code == 200 and r.json()["reason"] is None


def test_original_upload_is_stored_without_exif(client, settings):
    exif = Image.Exif()
    exif[0x010F] = "Fabricante"  # Make
    exif[0x0112] = 6  # Orientation: rotar 90°
    exif[0x8825] = {1: "S", 2: (12.0, 3.0, 0.0)}  # GPSInfo
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (10, 120, 200)).save(buf, format="JPEG", exif=exif)

    body = create_session(client, data=buf.getvalue(), filename="f.jpg", content_type="image/jpeg").json()
    original = settings.asset_root / body["id"] / "original.jpg"
    with Image.open(original) as img:
        assert not img.getexif()
        assert "exif" not in img.info
        assert img.size == (200, 300)  # la orientación se aplicó antes de quitar el EXIF


@pytest.mark.parametrize("hidden", [(0, 0, 0), (0, 255, 0)])
def test_transparent_png_is_flattened_on_white(client, settings, hidden):
    # Producto recortado con un removedor de fondo: el fondo es transparente y debajo hay
    # un color oculto. El lienzo de trabajo no debe mostrar ese color, sino blanco.
    img = Image.new("RGBA", (120, 100), (*hidden, 0))
    img.paste((200, 40, 40, 255), (40, 30, 80, 70))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    body = create_session(client, data=buf.getvalue()).json()
    with Image.open(settings.asset_root / body["id"] / "input.png") as working:
        assert working.mode == "RGB"
        assert working.getpixel((2, 2)) == (255, 255, 255)
        assert working.getpixel((60, 50)) == (200, 40, 40)
    with Image.open(settings.asset_root / body["id"] / "original.png") as original:
        assert original.mode == "RGBA"  # el original conserva la transparencia


def test_concurrent_summary_returns_existing_record(client):
    s = ready_session(client)
    decide_all(client, s, ["accepted", "rejected", "accepted", "rejected", "rejected"])
    first = client.post(f"/api/v1/sessions/{s['id']}/summary").json()

    # Simula una segunda solicitud que leyó REVIEWED antes de que la primera guardara.
    svc = client.app.state.sessions
    variations = svc.repo.list_variations(s["id"])
    result = svc.summarizer.summarize(s["id"], variations)
    late = SummaryRecord(
        session_id=s["id"], text=result.text, payload_json=json.dumps(result.payload),
        generated_by=result.generated_by, validation_status=result.validation_status,
        validated_decisions_hash=decisions_hash(variations), created_at=utcnow(),
    )
    saved = svc.repo.save_summary(late)
    assert saved.created_at != late.created_at  # devolvió el resumen ya guardado, sin error
    assert json.loads(saved.payload_json)["summary"] == first["summary"]


# --- validador del resumen (lo reutilizará el agente del Hito 3) ---


def _variations() -> list[Variation]:
    now = datetime.now(timezone.utc)
    values = ["accepted", "rejected", "accepted", "rejected", "rejected"]
    reasons = [None, "Fondo oscuro", None, None, "No me gusta"]
    out = []
    for i, style in enumerate(Style):
        vid = f"v{i + 1}"
        out.append(
            Variation(
                id=vid, session_id="s", job_id="j", position=i + 1, style=style, storage_key="k", seed=1,
                model_revision="m", prompt_version="v1", params_json="{}",
                decision=Decision(vid, DecisionValue(values[i]), reasons[i], now),
            )
        )
    return out


def _faithful(variations):
    return StructuredSummary(
        items=tuple(
            SummaryItem(v.id, v.position, v.style, v.decision.value, v.decision.reason) for v in variations
        )
    )


def test_validator_accepts_faithful_summary():
    vs = _variations()
    assert validate_structured(_faithful(vs), vs).ok


def test_validator_rejects_flipped_label():
    vs = _variations()
    items = list(_faithful(vs).items)
    items[1] = dataclasses.replace(items[1], decision=DecisionValue.ACCEPTED)
    result = validate_structured(StructuredSummary(items=tuple(items)), vs)
    assert not result.ok and "no coincide" in result.errors[0]


def test_validator_rejects_invented_or_misattributed_reason():
    vs = _variations()
    items = list(_faithful(vs).items)
    items[0] = dataclasses.replace(items[0], reason_quote="Me encantó la luz")  # inventado
    items[1] = dataclasses.replace(items[1], reason_quote="No me gusta")  # motivo de otra variación
    result = validate_structured(StructuredSummary(items=tuple(items)), vs)
    assert not result.ok
    assert len([e for e in result.errors if "no consta" in e]) == 2


def test_validator_rejects_missing_duplicate_or_unknown_items():
    vs = _variations()
    items = list(_faithful(vs).items)
    result = validate_structured(StructuredSummary(items=tuple(items[:4] + [items[0]])), vs)
    assert not result.ok
    assert any("más de una vez" in e for e in result.errors)
    assert any("Faltan" in e for e in result.errors)
    unknown = dataclasses.replace(items[0], variation_id="otra")
    assert not validate_structured(StructuredSummary(items=tuple(items[1:] + [unknown])), vs).ok
