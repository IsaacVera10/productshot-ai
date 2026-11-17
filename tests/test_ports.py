"""Piezas intercambiables (ADR-009): las mismas pruebas para cada adaptador, sin red ni GPU.

- `Repository`: el orquestador y la API solo usan métodos de la interfaz.
- `GenerationBackend`: el flujo completo pasa igual con el mock en el proceso y con el
  mock detrás del contrato HTTP del worker (`RemoteGenerationBackend`).
- Raíz de composición: las variables de entorno eligen el adaptador y una configuración
  incompleta falla al arrancar.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import create_session, decide_all, ready_session
from productshot.composition import ConfigError, build_generator, build_services, build_summarizer
from productshot.config import Settings
from productshot.domain import Mode, Style
from productshot.services.generation import (
    GenerationRequest,
    GenerationUnavailable,
    MockGenerationBackend,
    VariationSpec,
)
from productshot.services.remote_generation import UNREACHABLE_REVISION, RemoteGenerationBackend
from productshot.storage.ports import Repository
from productshot.storage.repository import SqliteRepository
from productshot.worker.app import create_worker_app
from productshot.worker.contract import CONTRACT_VERSION, GENERATE_PATH, HEALTH_PATH, encode_png

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "token-de-prueba"


# --- Repository ---


def test_sqlite_repository_satisfies_protocol(tmp_path):
    assert isinstance(SqliteRepository(tmp_path / "r.sqlite3"), Repository)


def test_callers_only_use_protocol_methods():
    """Si el orquestador o la API llaman algo que no está en la interfaz, Firestore no podría sustituir a SQLite."""
    allowed = {name for name in dir(Repository) if not name.startswith("_")}
    sources = {
        "services/sessions.py": r"self\.repo\.([a-z_]+)",
        "api/app.py": r"\.repo\.([a-z_]+)",
    }
    for rel, pattern in sources.items():
        text = (ROOT / "backend/productshot" / rel).read_text(encoding="utf-8")
        used = set(re.findall(pattern, text))
        assert used, rel
        assert used <= allowed, f"{rel} usa métodos fuera de Repository: {sorted(used - allowed)}"


# --- worker y RemoteGenerationBackend ---


def worker_transport(worker_app) -> httpx.MockTransport:
    """Transporte httpx que entrega cada solicitud al worker real en memoria.

    El adaptador usa su propio `httpx.Client` igual que en producción; solo cambia el
    transporte, así que se prueban la serialización, los códigos HTTP y los errores reales.
    """
    worker = TestClient(worker_app)

    def handler(request: httpx.Request) -> httpx.Response:
        r = worker.request(request.method, request.url.path, content=request.content, headers=dict(request.headers))
        return httpx.Response(r.status_code, headers=dict(r.headers), content=r.content)

    return httpx.MockTransport(handler)


def remote_backend(backend=None, *, token: str | None = TOKEN, worker_token: str | None = TOKEN):
    """RemoteGenerationBackend que habla con el worker real por HTTP en memoria."""
    transport = worker_transport(create_worker_app(backend or MockGenerationBackend(), token=worker_token))
    client = httpx.Client(base_url="http://worker", transport=transport)
    return RemoteGenerationBackend("http://worker", token=token, client=client)


def _request(size=(64, 48)) -> GenerationRequest:
    mask = Image.new("L", size, 255)
    return GenerationRequest(
        session_id="s", mode=Mode.INTERMEDIATE, image=Image.new("RGB", size, (200, 60, 60)), mask=mask,
        specs=tuple(VariationSpec(position=i + 1, style=s, seed=10 + i) for i, s in enumerate(Style)),
    )


@pytest.fixture(params=["mock", "remote"])
def any_generator(request):
    return MockGenerationBackend() if request.param == "mock" else remote_backend()


def test_full_flow_is_identical_for_each_generator(make_client, any_generator):
    client = make_client(generator=any_generator)
    s = ready_session(client)
    assert [v["style"] for v in s["variations"]] == [st.value for st in Style]
    assert s["simulated"] is True  # el worker con mock se declara simulado y se rotula (G9)
    decide_all(client, s, ["accepted", "rejected", "accepted", "rejected", "accepted"], [None, "Muy oscuro", None, None, None])
    body = client.post(f"/api/v1/sessions/{s['id']}/summary").json()
    assert body["validation_status"] == "structure_verified"
    assert len(body["accepted_ids"]) == 3


def test_remote_reports_worker_identity():
    backend = remote_backend()
    assert backend.model_revision == "mock-v1"
    assert backend.simulated is True
    arts = backend.generate(_request())
    assert len(arts) == 5
    assert all(a.params["worker_backend"] == "mock" and a.params["simulated"] for a in arts)
    assert arts[0].image.size == (64, 48)


def test_remote_unreachable_is_conservative_and_readable():
    def boom(request: httpx.Request):
        raise httpx.ConnectError("sin red", request=request)

    backend = RemoteGenerationBackend("http://worker", client=httpx.Client(base_url="http://worker", transport=httpx.MockTransport(boom)))
    assert backend.simulated is True  # sin saber, se rotula como simulado
    assert backend.model_revision == UNREACHABLE_REVISION
    with pytest.raises(GenerationUnavailable, match="No se pudo conectar"):
        backend.generate(_request())


def test_remote_wrong_token_fails_the_job_with_a_readable_message(make_client):
    client = make_client(generator=remote_backend(token="otro"))
    sid = create_session(client).json()["id"]
    client.post(f"/api/v1/sessions/{sid}/segment")
    assert client.post(f"/api/v1/sessions/{sid}/generate").status_code == 202
    body = client.get(f"/api/v1/sessions/{sid}").json()
    assert body["status"] == "FAILED"
    assert "rechazó el token" in body["failure_reason"]


def test_remote_short_batch_is_rejected_by_the_orchestrator(make_client):
    class FourOnly(MockGenerationBackend):
        def generate(self, request):
            return super().generate(request)[:4]

    client = make_client(generator=remote_backend(FourOnly()))
    sid = create_session(client).json()["id"]
    client.post(f"/api/v1/sessions/{sid}/segment")
    client.post(f"/api/v1/sessions/{sid}/generate")
    body = client.get(f"/api/v1/sessions/{sid}").json()
    assert body["status"] == "FAILED"
    assert "4 variaciones" in body["failure_reason"]
    assert body["variations"] == []


def test_remote_malformed_response_is_unavailable():
    def bad(request: httpx.Request):
        if request.url.path == HEALTH_PATH:
            return httpx.Response(200, json={"backend": "x", "model_revision": "r", "simulated": False})
        return httpx.Response(200, json={"artifacts": "no"})

    backend = RemoteGenerationBackend("http://w", client=httpx.Client(base_url="http://w", transport=httpx.MockTransport(bad)))
    with pytest.raises(GenerationUnavailable, match="no cumple el contrato"):
        backend.generate(_request())


def test_worker_requires_token_and_validates_input():
    worker = TestClient(create_worker_app(MockGenerationBackend(), token=TOKEN))
    assert worker.get(HEALTH_PATH).status_code == 401
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert worker.get(HEALTH_PATH, headers=auth).json()["contract_version"] == CONTRACT_VERSION

    body = {
        "session_id": "s", "mode": "intermediate", "prompt_version": "p",
        "image_png_b64": encode_png(Image.new("RGB", (32, 32))),
        "mask_png_b64": encode_png(Image.new("L", (16, 16), 255)),
        "specs": [{"position": i + 1, "style": s.value, "seed": i} for i, s in enumerate(Style)],
    }
    r = worker.post(GENERATE_PATH, json=body, headers=auth)
    assert r.status_code == 422 and "mismo tamaño" in r.json()["detail"]
    assert worker.post(GENERATE_PATH, json={**body, "contract_version": "99"}, headers=auth).status_code == 409
    assert worker.post(GENERATE_PATH, json={**body, "specs": body["specs"][:4]}, headers=auth).status_code == 422
    jpeg = {**body, "image_png_b64": "bm8gZXMgdW5hIGltYWdlbg=="}
    assert worker.post(GENERATE_PATH, json=jpeg, headers=auth).status_code == 422


# --- raíz de composición ---


def _settings(tmp_path, **kw) -> Settings:
    return Settings(_env_file=None, sqlite_path=tmp_path / "c.sqlite3", asset_root=tmp_path / "a", **kw)


def test_defaults_are_all_local(tmp_path):
    services = build_services(_settings(tmp_path))
    assert services.describe() == {
        "repository": "SqliteRepository", "artifacts": "LocalArtifactStore", "segmentation": "mock",
        "generation": "mock", "summary": "deterministic", "runner": "ThreadJobRunner", "screening": "mock",
    }
    services.runner.shutdown()


def test_environment_variables_choose_adapters(tmp_path, monkeypatch):
    monkeypatch.setenv("GENERATION_BACKEND", "remote")
    monkeypatch.setenv("GENERATION_WORKER_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("SCREENING_BACKEND", "remote")
    monkeypatch.setenv("SUMMARY_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "clave-falsa")
    monkeypatch.setenv("JOB_RUNNER", "inline")
    services = build_services(_settings(tmp_path))
    assert services.describe()["generation"] == "remote"
    assert services.describe()["screening"] == "remote"
    assert services.describe()["summary"] == "gemini"
    assert services.describe()["runner"] == "InlineJobRunner"


def test_incomplete_configuration_fails_at_startup(tmp_path):
    with pytest.raises(ConfigError, match="GENERATION_WORKER_URL"):
        build_generator(_settings(tmp_path, generation_backend="remote"))
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        build_summarizer(_settings(tmp_path, summary_backend="gemini"))


def test_secrets_are_not_exposed(tmp_path, make_client):
    s = _settings(tmp_path, gemini_api_key="AQ.secreto", generation_worker_token="tok-secreto")
    assert "secreto" not in repr(s)
    health = make_client().get("/api/v1/healthz").text
    assert "secreto" not in health and "http" not in health


# --- segmentación por el worker ---


def remote_segmenter(segmenter=None, *, token: str | None = TOKEN):
    from productshot.services.remote_segmentation import RemoteSegmentationService
    from productshot.services.segmentation import MockSegmentationService

    app = create_worker_app(MockGenerationBackend(), token=TOKEN, segmenter=segmenter or MockSegmentationService())
    client = httpx.Client(base_url="http://worker", transport=worker_transport(app))
    return RemoteSegmentationService("http://worker", token=token, client=client)


def test_remote_segmentation_through_the_worker(make_client):
    seg = remote_segmenter()
    assert seg.simulated is True  # el worker declara su segmentador mock
    client = make_client(segmenter=seg)
    s = ready_session(client)
    assert s["mask"]["simulated"] is True and s["mask"]["source"] == "auto"


def test_real_segmenter_mask_is_marked_not_simulated(make_client):
    from productshot.services.segmentation import MaskResult

    class Real:
        name, simulated = "fake-real", False

        def segment(self, image):
            mask = Image.new("L", image.size, 255)
            mask.paste(0, (image.width // 4, image.height // 4, 3 * image.width // 4, 3 * image.height // 4))
            return MaskResult(mask=mask, simulated=False)

    client = make_client(segmenter=remote_segmenter(Real()))
    sid = create_session(client).json()["id"]
    body = client.post(f"/api/v1/sessions/{sid}/segment").json()
    assert body["mask"]["simulated"] is False


def test_worker_without_segmenter_gives_503(make_client):
    from productshot.services.remote_segmentation import RemoteSegmentationService

    app = create_worker_app(MockGenerationBackend(), token=TOKEN)  # sin segmentador
    seg = RemoteSegmentationService("http://worker", token=TOKEN,
                                    client=httpx.Client(base_url="http://worker", transport=worker_transport(app)))
    client = make_client(segmenter=seg)
    sid = create_session(client).json()["id"]
    r = client.post(f"/api/v1/sessions/{sid}/segment")
    assert r.status_code == 503 and "no ofrece segmentación" in r.json()["error"]["message"]
    assert client.get(f"/api/v1/sessions/{sid}").json()["status"] == "UPLOADED"


def test_remote_segmentation_requires_worker_url(tmp_path):
    from productshot.composition import build_segmenter

    with pytest.raises(ConfigError, match="GENERATION_WORKER_URL"):
        build_segmenter(_settings(tmp_path, segmentation_backend="remote"))
