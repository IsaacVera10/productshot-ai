"""Agente de resumen con Gemini, probado con un Gemini falso (sin red, sin gastar cuota).

La prueba real contra la API es opcional: `RUN_LIVE_GEMINI=1 pytest -k live`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from conftest import decide_all, ready_session
from productshot.config import Settings
from productshot.services.gemini_summary import GeminiSummaryService, check_prose

PRIMARY, FALLBACK = "modelo-principal", "modelo-respaldo"
DECISIONS = ["accepted", "rejected", "accepted", "rejected", "accepted"]
REASONS = [None, "El fondo no combina con mi marca", None, None, "Me gusta el espacio vacío"]


class FakeGemini:
    """Responde como generateContent a partir del registro que recibe; cada modelo se configura aparte."""

    def __init__(self, **behaviour):
        self.behaviour = behaviour  # modelo -> "ok" | 503 | 400 | "wrong_decision" | "bad_quote" | "garbage"
        self.calls: list[tuple[str, dict, httpx.Headers]] = []
        self.urls: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        model = request.url.path.rsplit("/", 1)[1].split(":")[0]
        body = json.loads(request.content)
        self.calls.append((model, body, request.headers))
        self.urls.append(str(request.url))
        mode = self.behaviour.get(model, "ok")
        if isinstance(mode, int):
            return httpx.Response(mode, json={"error": {"code": mode}})
        if mode == "garbage":
            return self._wrap("esto no es JSON", model)
        text = body["contents"][0]["parts"][0]["text"]
        records = json.loads(text[text.index("["):])
        items = [
            {
                "variation_id": r["variation_id"], "position": r["position"], "style": r["style"],
                "decision": r["decision"], "reason_quote": r["reason"],
            }
            for r in records
        ]
        narrative = "Se aceptaron tres variaciones. Una se rechazó porque «El fondo no combina con mi marca»."
        if mode == "wrong_decision":
            items[0]["decision"] = "rejected" if items[0]["decision"] == "accepted" else "accepted"
        if mode == "bad_quote":
            narrative = "La persona dijo «odio el color rojo»."
        out = {"items": items, "narrative": narrative, "style_observations": ["Los dos estilos rechazados tienen fondo con textura."]}
        return self._wrap(json.dumps(out, ensure_ascii=False), model)

    @staticmethod
    def _wrap(text: str, model: str) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}], "modelVersion": model + "-001"})


def agent(fake: FakeGemini, retries: int = 1) -> GeminiSummaryService:
    client = httpx.Client(transport=httpx.MockTransport(fake))
    return GeminiSummaryService("clave-falsa", [PRIMARY, FALLBACK], client=client, retries_per_model=retries, backoff_s=0)


def summarize(make_client, fake: FakeGemini, **kw) -> dict:
    client = make_client(summarizer=agent(fake, **kw))
    s = ready_session(client)
    decide_all(client, s, DECISIONS, REASONS)
    r = client.post(f"/api/v1/sessions/{s['id']}/summary")
    assert r.status_code == 200, r.text
    return r.json()


def test_valid_output_is_llm_validated(make_client):
    fake = FakeGemini()
    body = summarize(make_client, fake)
    assert body["generated_by"] == "llm_validated"
    assert body["validation_status"] == "structure_verified"
    assert "Se aceptaron tres variaciones" in body["summary"]
    assert "Observación: Los dos estilos rechazados" in body["summary"]
    assert any("modelo-principal-001" in l for l in body["limitations"])
    assert [c[0] for c in fake.calls] == [PRIMARY]


def test_only_decisions_and_reasons_leave_the_service(make_client):
    fake = FakeGemini()
    summarize(make_client, fake)
    _model, body, headers = fake.calls[0]
    sent = json.dumps(body, ensure_ascii=False)
    assert "image" not in sent and ".png" not in sent and "storage" not in sent and "seed" not in sent
    assert "El fondo no combina con mi marca" in sent
    assert headers["x-goog-api-key"] == "clave-falsa"
    assert "clave-falsa" not in fake.urls[0] and "clave-falsa" not in sent  # nunca en la URL ni en el cuerpo
    assert body["generationConfig"]["responseMimeType"] == "application/json"


def test_overloaded_primary_falls_back_to_second_model(make_client):
    fake = FakeGemini(**{PRIMARY: 503})
    body = summarize(make_client, fake)
    assert body["generated_by"] == "llm_validated"
    assert any("modelo-respaldo-001" in l for l in body["limitations"])
    assert [c[0] for c in fake.calls] == [PRIMARY, PRIMARY, FALLBACK]  # un reintento en el principal


def test_both_models_down_gives_explicit_deterministic_fallback(make_client):
    fake = FakeGemini(**{PRIMARY: 503, FALLBACK: 503})
    body = summarize(make_client, fake)
    assert body["generated_by"] == "deterministic_fallback"
    assert body["validation_status"] == "structure_verified"
    assert any("no es la salida de un agente" in l for l in body["limitations"])
    assert "Se aceptaron 3 de 5" in body["summary"]


def test_client_errors_are_not_retried_on_the_same_model(make_client):
    fake = FakeGemini(**{PRIMARY: 400})
    summarize(make_client, fake)
    assert [c[0] for c in fake.calls] == [PRIMARY, FALLBACK]


def test_output_that_contradicts_the_record_is_never_used(make_client):
    fake = FakeGemini(**{PRIMARY: "wrong_decision", FALLBACK: "wrong_decision"})
    body = summarize(make_client, fake)
    assert body["generated_by"] == "deterministic_fallback"
    assert "Se aceptaron tres variaciones" not in body["summary"]


def test_time_budget_returns_fallback_before_the_ui_gives_up(make_client):
    # Reloj falso: cada lectura avanza 25 s, como si cada llamada agotara su timeout.
    ticks = iter(range(0, 1000, 25))
    fake = FakeGemini(**{PRIMARY: 503, FALLBACK: 503})
    client = make_client(summarizer=GeminiSummaryService(
        "k", [PRIMARY, FALLBACK], client=httpx.Client(transport=httpx.MockTransport(fake)), budget_s=45, backoff_s=0,
        clock=lambda: float(next(ticks))))
    s = ready_session(client)
    decide_all(client, s, DECISIONS, REASONS)
    body = client.post(f"/api/v1/sessions/{s['id']}/summary").json()
    assert body["generated_by"] == "deterministic_fallback"
    assert len(fake.calls) < 4  # no llegó a probar todos los intentos


def test_garbage_output_moves_on(make_client):
    fake = FakeGemini(**{PRIMARY: "garbage"})
    body = summarize(make_client, fake, retries=0)
    assert body["generated_by"] == "llm_validated"
    assert [c[0] for c in fake.calls] == [PRIMARY, FALLBACK]


def test_invented_quote_drops_the_narrative(make_client):
    fake = FakeGemini(**{PRIMARY: "bad_quote"})
    body = summarize(make_client, fake)
    assert body["generated_by"] == "hybrid"
    assert "odio el color rojo" not in body["summary"]
    assert any("cita un texto que no está" in l for l in body["limitations"])
    # la enumeración y los motivos siguen saliendo del registro
    assert "«El fondo no combina con mi marca»" in body["summary"]


@pytest.mark.parametrize(
    "text,problem",
    [
        ("Todo bien.", None),
        ("Dijo «El fondo no combina».", None),
        ("Mira https://ejemplo.com", "contiene enlaces o marcado"),
        ("<script>x</script>", "contiene enlaces o marcado"),
        ("x" * 800, "demasiado larga"),
        ("Dijo \"nunca lo escribió\".", "cita un texto que no está en ningún motivo"),
    ],
)
def test_check_prose(text, problem):
    assert check_prose(text, ["El fondo no combina con mi marca"], 700) == problem


@pytest.mark.skipif(os.environ.get("RUN_LIVE_GEMINI") != "1", reason="prueba real opcional (gasta cuota gratuita)")
def test_live_gemini(make_client):
    settings = Settings(_env_file=Path(__file__).resolve().parents[1] / ".env")
    assert settings.gemini_api_key, "falta GEMINI_API_KEY en .env"
    models = [settings.gemini_model_id] + ([settings.gemini_fallback_model_id] if settings.gemini_fallback_model_id else [])
    service = GeminiSummaryService(settings.gemini_api_key.get_secret_value(), models, timeout_s=60)
    client = make_client(summarizer=service)
    s = ready_session(client)
    decide_all(client, s, DECISIONS, REASONS)
    body = client.post(f"/api/v1/sessions/{s['id']}/summary").json()
    print(json.dumps(body, ensure_ascii=False, indent=2))
    assert body["validation_status"] == "structure_verified"
    assert body["generated_by"] in {"llm_validated", "hybrid", "deterministic_fallback"}
