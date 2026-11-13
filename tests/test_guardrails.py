"""Guardrails sin modelos (ADR-008): G1 (evento), G8, G12, G13 y G14."""

from __future__ import annotations

import io
import logging
import random
import sqlite3

import pytest
from PIL import Image, ImageDraw

from conftest import create_session, ready_session
from productshot.config import Settings
from productshot.services import guardrails
from productshot.services.generation import MockGenerationBackend
from productshot.storage.repository import SqliteRepository

API = "/api/v1/sessions"
S = Settings(_env_file=None)


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def textured(size=(320, 240), base=128, spread=60, seed=1) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", size)
    img.putdata([(v, v, v) for v in (max(0, min(255, base + rng.randint(-spread, spread)))
                                     for _ in range(size[0] * size[1]))])
    return img


def codes(warnings) -> set[str]:
    return {w.code if hasattr(w, "code") else w["code"] for w in warnings}


# --- G12: calidad de la foto ---


@pytest.mark.parametrize(
    "image, expected",
    [
        (textured(), set()),
        (textured(base=15, spread=10), {"photo_dark"}),
        (textured(base=245, spread=8), {"photo_overexposed"}),
        (Image.new("RGB", (320, 240), (128, 128, 128)), {"photo_blurry"}),
    ],
)
def test_photo_quality(image, expected):
    assert codes(guardrails.photo_warnings(guardrails.photo_metrics(image), S)) == expected


def test_photo_warnings_reach_the_api_and_do_not_block(client, caplog):
    caplog.set_level(logging.INFO, logger="productshot.guardrails")
    r = create_session(client, data=png(textured(base=15, spread=10)))
    assert r.status_code == 201
    body = r.json()
    assert "photo_dark" in codes(body["warnings"])
    assert body["photo_quality"]["brightness"] < S.quality_min_brightness
    assert "GUARDRAIL G12 code=photo_dark" in caplog.text
    # Sigue el flujo normal.
    assert client.post(f"{API}/{body['id']}/segment").status_code == 200


# --- G13: heurística de máscara ---


def _mask(boxes, size=(320, 240)) -> Image.Image:
    m = Image.new("L", size, 255)
    draw = ImageDraw.Draw(m)
    for box in boxes:
        draw.rectangle(box, fill=0)
    return m


@pytest.mark.parametrize(
    "mask, expected",
    [
        (_mask([(100, 60, 220, 180)]), set()),
        (_mask([(150, 110, 156, 116)]), {"mask_tiny"}),
        (_mask([(0, 0, 319, 230)]), {"mask_huge"}),
        (_mask([(20, 60, 100, 180), (220, 60, 300, 180)]), {"mask_many_objects"}),
    ],
)
def test_mask_heuristic(mask, expected):
    assert codes(guardrails.mask_warnings(guardrails.mask_metrics(mask, S), S)) == expected


def test_mock_mask_has_no_warnings(client):
    sid = create_session(client, data=png(textured())).json()["id"]
    seg = client.post(f"{API}/{sid}/segment").json()
    assert seg["mask"]["warnings"] == []
    assert 0.2 < seg["mask"]["metrics"]["protected_fraction"] < 0.5


def test_mask_warnings_on_manual_upload(client, caplog):
    caplog.set_level(logging.INFO, logger="productshot.guardrails")
    sid = create_session(client).json()["id"]
    data = png(_mask([(20, 60, 100, 180), (220, 60, 300, 180)]))
    r = client.post(f"{API}/{sid}/mask", files={"mask": ("m.png", data, "image/png")})
    assert r.status_code == 200
    assert codes(r.json()["mask"]["warnings"]) == {"mask_many_objects"}
    assert "mask_many_objects" in codes(client.get(f"{API}/{sid}").json()["warnings"])
    assert "GUARDRAIL G13 code=mask_many_objects" in caplog.text


# --- G8: fidelidad del producto ---


class AlteringBackend(MockGenerationBackend):
    """Simula un modelo que repinta también el producto (lo que G8 debe corregir)."""

    name = "altering"

    def generate(self, request):
        out = super().generate(request)
        return [type(a)(**{**a.__dict__, "image": Image.new("RGB", a.image.size, (20, 40, 200))}) for a in out]


class WrongSizeBackend(MockGenerationBackend):
    def generate(self, request):
        out = super().generate(request)
        return [type(a)(**{**a.__dict__, "image": a.image.resize((10, 10))}) for a in out]


def test_product_is_recomposed_when_the_generator_alters_it(make_client, caplog):
    caplog.set_level(logging.INFO, logger="productshot.guardrails")
    client = make_client(generator=AlteringBackend())
    s = ready_session(client)
    v = s["variations"][0]
    img = Image.open(io.BytesIO(client.get(v["image_url"]).content)).convert("RGB")
    assert img.getpixel((160, 120)) == (200, 60, 60)  # producto original, no el azul
    assert img.getpixel((5, 120)) == (20, 40, 200)  # fondo generado
    fidelity = v["generation"]["fidelity"]
    assert fidelity["drift_warning"] is True and fidelity["protected_mean_abs_diff"] > 50
    assert fidelity["protected_max_diff_after"] == 0
    assert "GUARDRAIL G8 code=product_drift" in caplog.text


def test_mock_generator_keeps_the_product_and_the_simulation_label(client):
    s = ready_session(client)
    v = s["variations"][0]
    assert v["generation"]["fidelity"]["protected_mean_abs_diff"] == 0
    img = Image.open(io.BytesIO(client.get(v["image_url"]).content)).convert("RGB")
    r, g, b = img.getpixel((5, 3))  # franja superior «SIMULACIÓN»
    assert r > 150 and g < 60


def test_soft_mask_edges_are_blended_but_the_core_is_exact():
    working = textured()
    mask = _mask([(100, 60, 220, 180)])
    soft = mask.resize((32, 24)).resize(mask.size, Image.Resampling.BILINEAR)  # bordes grises
    generated = Image.new("RGB", working.size, (0, 255, 0))
    recomposed, metrics = guardrails.enforce_fidelity(working, soft, generated, S)
    assert metrics["protected_max_diff_after"] == 0
    assert recomposed.getpixel((160, 120)) == working.getpixel((160, 120))


def test_wrong_size_output_fails_the_batch(make_client):
    client = make_client(generator=WrongSizeBackend())
    sid = create_session(client).json()["id"]
    client.post(f"{API}/{sid}/segment")
    client.post(f"{API}/{sid}/generate")
    body = client.get(f"{API}/{sid}").json()
    assert body["status"] == "FAILED" and "mide" in body["failure_reason"]


# --- G14: timeout por lote ---


class _ParkedRunner:
    def submit(self, fn):
        self.fn = fn

    def shutdown(self):
        pass


def test_batch_timeout_fails_session_and_discards_late_results(make_client, settings, caplog):
    caplog.set_level(logging.INFO, logger="productshot.guardrails")
    runner = _ParkedRunner()
    client = make_client(runner=runner)
    sid = create_session(client).json()["id"]
    client.post(f"{API}/{sid}/segment")
    client.post(f"{API}/{sid}/generate")
    assert client.get(f"{API}/{sid}").json()["status"] == "GENERATING"

    client.app.state.sessions.settings.generation_timeout_s = 0.0
    body = client.get(f"{API}/{sid}").json()
    assert body["status"] == "FAILED" and "inténtalo más tarde" in body["failure_reason"]
    assert "GUARDRAIL G14 code=batch_timeout" in caplog.text

    runner.fn()  # el trabajo termina tarde: se descarta
    body = client.get(f"{API}/{sid}").json()
    assert body["status"] == "FAILED" and body["variations"] == []
    assert not list((settings.asset_root / sid).glob("variation_*"))


# --- G1: evento contable ---


def test_rejected_upload_logs_g1(client, caplog):
    caplog.set_level(logging.INFO, logger="productshot.guardrails")
    assert create_session(client, data=b"no soy una imagen").status_code == 400
    assert "GUARDRAIL G1 code=invalid_input" in caplog.text


# --- migración 2 -> 3 ---


def test_round_schema_database_gains_guardrail_columns(tmp_path):
    path = tmp_path / "v2.sqlite3"
    SqliteRepository(path)
    con = sqlite3.connect(path)
    con.execute("ALTER TABLE input_images DROP COLUMN quality_json")
    con.execute("ALTER TABLE masks DROP COLUMN warnings_json")
    con.execute("PRAGMA user_version = 2")
    con.commit()
    con.close()

    SqliteRepository(path)
    con = sqlite3.connect(path)
    assert "quality_json" in {r[1] for r in con.execute("PRAGMA table_info(input_images)")}
    assert "warnings_json" in {r[1] for r in con.execute("PRAGMA table_info(masks)")}
    assert con.execute("PRAGMA user_version").fetchone()[0] == 3
    con.close()
