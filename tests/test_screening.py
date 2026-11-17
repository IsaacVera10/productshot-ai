"""Guardrails con modelos: G2 (NSFW en la entrada), G3 (rostros) y G7 (NSFW en la salida)."""

from __future__ import annotations

import httpx
import pytest
from conftest import image_bytes
from PIL import Image
from test_intermediate import make_request
from test_ports import worker_transport

from productshot.services.generation import MockGenerationBackend
from productshot.services.remote_screening import RemoteScreeningService
from productshot.services.screening import (
    LocalScreeningService,
    SafetyCheckedBackend,
    ScreeningResult,
    ScreeningUnavailable,
    UnsafeOutput,
)
from productshot.worker.app import create_worker_app

API = "/api/v1/sessions"


class FixedScreener:
    name = "fijo"
    simulated = False

    def __init__(self, nsfw=0.0, faces=0, fail=False):
        self.result = ScreeningResult(nsfw_score=nsfw, faces=faces, simulated=False)
        self.fail = fail
        self.calls = 0

    def screen(self, image):
        self.calls += 1
        if self.fail:
            raise ScreeningUnavailable("caído")
        return self.result


def post_photo(client):
    return client.post(API, files={"image": ("p.png", image_bytes(), "image/png")}, data={"mode": "intermediate"})


# --- G2 y G3 en la API ---


def test_clean_photo_passes(make_client):
    screener = FixedScreener(nsfw=0.1)
    client = make_client(screener=screener)
    assert post_photo(client).status_code == 201
    assert screener.calls == 1


def test_g2_rejects_inappropriate_photo_without_creating_a_session(make_client, settings):
    client = make_client(screener=FixedScreener(nsfw=0.97))
    r = post_photo(client)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "rejected_content"
    assert not list(settings.asset_root.glob("*/input.png"))  # no quedó ningún archivo


def test_g3_rejects_photo_with_a_face(make_client):
    r = post_photo(make_client(screener=FixedScreener(faces=1)))
    assert r.status_code == 422
    assert "sin personas" in r.json()["error"]["message"]


def test_unavailable_screener_fails_closed(make_client):
    assert post_photo(make_client(screener=FixedScreener(fail=True))).status_code == 503


# --- /v1/screen del worker y su adaptador ---


def remote(screener, token=None):
    app = create_worker_app(MockGenerationBackend(), token=token, screener=screener)
    client = httpx.Client(base_url="http://worker", transport=worker_transport(app))
    return RemoteScreeningService("http://worker", token=token, client=client)


def test_worker_screen_endpoint_round_trip():
    local = LocalScreeningService(classifier=lambda img: 0.42, face_counter=lambda img, f: 2,
                                  category_classifier=lambda img: {"producto": 0.9})
    result = remote(local, token="t").screen(Image.new("RGB", (64, 64)))
    assert (result.nsfw_score, result.faces, result.simulated) == (0.42, 2, False)


def test_worker_without_screener_is_reported_as_unavailable():
    with pytest.raises(ScreeningUnavailable):
        remote(None).screen(Image.new("RGB", (64, 64)))


def test_face_counter_receives_the_configured_fraction():
    seen = []
    local = LocalScreeningService(face_min_fraction=0.2, classifier=lambda img: 0.0, category_classifier=lambda img: {},
                                  face_counter=lambda img, f: seen.append(f) or 0)
    local.screen(Image.new("RGB", (64, 64)))
    assert seen == [0.2]


# --- G7 en la salida ---


def test_g7_regenerates_a_flagged_image_with_another_seed():
    inner = MockGenerationBackend()
    flagged = {"first": True}

    def score(img):
        # Marca solo la primera imagen que revisa (la variación 1 del lote).
        if flagged["first"]:
            flagged["first"] = False
            return 0.99
        return 0.01

    safe = SafetyCheckedBackend(inner, score, threshold=0.85, retries=2)
    request = make_request()
    arts = safe.generate(request)
    assert len(arts) == 5
    assert arts[0].params["g7"]["retries"] == 1 and arts[0].seed == request.specs[0].seed
    assert arts[0].params["g7"]["retry_seed"] != request.specs[0].seed
    assert all(a.params["g7"]["retries"] == 0 for a in arts[1:])
    assert [a.position for a in arts] == [1, 2, 3, 4, 5]


def test_g7_fails_the_whole_batch_when_retries_run_out():
    safe = SafetyCheckedBackend(MockGenerationBackend(), lambda img: 0.99, retries=2)
    with pytest.raises(UnsafeOutput):
        safe.generate(make_request())


def test_g7_failure_leaves_the_session_failed(make_client):
    from conftest import create_session

    worker = create_worker_app(SafetyCheckedBackend(MockGenerationBackend(), lambda img: 0.99, retries=1))
    client_http = httpx.Client(base_url="http://worker", transport=worker_transport(worker))
    from productshot.services.remote_generation import RemoteGenerationBackend

    client = make_client(generator=RemoteGenerationBackend("http://worker", client=client_http))
    sid = create_session(client).json()["id"]
    assert client.post(f"{API}/{sid}/segment").status_code == 200
    client.post(f"{API}/{sid}/generate")
    body = client.get(f"{API}/{sid}").json()
    assert body["status"] == "FAILED"
    assert not body["variations"]


# --- G15: tipo de producto ---


class CategoryScreener(FixedScreener):
    def __init__(self, categories):
        super().__init__()
        self.result = ScreeningResult(nsfw_score=0.0, faces=0, simulated=False, categories=categories)


@pytest.mark.parametrize("group,word", [("tabaco_vape", "vapes"), ("sexual", "adultos"), ("arma", "armas")])
def test_g15_rejects_blocked_categories(make_client, group, word):
    r = post_photo(make_client(screener=CategoryScreener({group: 0.97, "producto": 0.03})))
    assert r.status_code == 422 and word in r.json()["error"]["message"]


def test_g15_warns_out_of_niche_without_blocking(make_client):
    client = make_client(screener=CategoryScreener({"comida": 0.8, "producto": 0.2}))
    r = post_photo(client)
    assert r.status_code == 201
    warnings = client.get(f"{API}/{r.json()['id']}").json()["warnings"]
    assert any(w["code"] == "out_of_niche_comida" for w in warnings)


def test_g15_lets_niche_products_through(make_client):
    # Valores reales del 2026-11-16: el hub da 0.01 en tabaco y la pirámide 0.02 en "sexual".
    client = make_client(screener=CategoryScreener({"producto": 0.97, "tabaco_vape": 0.01, "sexual": 0.02}))
    r = post_photo(client)
    assert r.status_code == 201
    assert not [w for w in client.get(f"{API}/{r.json()['id']}").json()["warnings"] if w["guardrail"] == "G15"]


def test_safety_wrapper_keeps_lora_visible_in_worker_health():
    from fastapi.testclient import TestClient

    from productshot.worker.app import create_worker_app

    inner = MockGenerationBackend()
    inner.lora = {"sha256": "abc", "path": "/cache/lora/x.safetensors"}
    inner.params = type("P", (), {"lora_scale": 1.0})()
    safe = SafetyCheckedBackend(inner, lambda img: 0.0)
    health = TestClient(create_worker_app(safe, token="t")).get(
        "/v1/health", headers={"Authorization": "Bearer t"}).json()
    assert health["lora"] == {"sha256": "abc", "path": "/cache/lora/x.safetensors", "scale": 1.0}
