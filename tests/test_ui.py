"""Lógica de la UI: el resumen debe congelar lo que la persona ve en el formulario.

Se ejecutan los manejadores reales de Gradio contra la API en memoria (TestClient es un
`httpx.Client`), sin abrir un servidor ni un navegador.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gradio")

from conftest import decide_all, ready_session  # noqa: E402
import productshot_ui.app as ui  # noqa: E402

DECISIONS = ["accepted", "rejected", "accepted", "rejected", "rejected"]


@pytest.fixture
def handlers(client, monkeypatch):
    class _Api(ui.Api):
        def __init__(self, api_base: str):
            super().__init__(api_base)
            self.http = client

    monkeypatch.setattr(ui, "Api", _Api)
    demo = ui.build_ui("http://testserver/api/v1")
    return {f.fn.__name__: f.fn for f in demo.fns.values()}


def _form(decisions, reasons=None) -> list:
    reasons = reasons or [""] * 5
    return [x for pair in zip(decisions, reasons) for x in pair]


def test_pending_decisions_detects_changes_and_missing():
    variations = [
        {"id": f"v{i}", "position": i, "decision": None if i == 5 else {"decision": "accepted", "reason": "Me gusta"}}
        for i in range(1, 6)
    ]
    to_save, missing = ui.pending_decisions(
        variations,
        ["accepted", "rejected", "accepted", "accepted", None],
        ["  Me   gusta ", "", "Otro motivo", "Me gusta", ""],
    )
    # v1 solo difiere en espacios (la API los normaliza): no es un cambio.
    assert to_save == [("v2", "rejected", None), ("v3", "accepted", "Otro motivo")]
    assert missing == [5]


def test_summary_saves_unsaved_form_changes_first(client, handlers):
    s = ready_session(client)
    decide_all(client, s, DECISIONS)
    # La persona cambia la variación 2 en el formulario y pulsa «Generar resumen» sin guardar.
    edited = DECISIONS.copy()
    edited[1] = "accepted"
    status, text, *_ = handlers["summarize"](s["id"], *_form(edited))

    assert "congeladas" in status
    body = client.post(f"/api/v1/sessions/{s['id']}/summary").json()
    ids = [v["id"] for v in s["variations"]]
    assert body["accepted_ids"] == [ids[0], ids[1], ids[2]]
    assert "Se aceptaron 3 de 5" in text


def test_summary_with_incomplete_form_does_not_freeze(client, handlers):
    s = ready_session(client)
    decide_all(client, s, DECISIONS)
    edited = DECISIONS.copy()
    edited[4] = None
    status, text, *_ = handlers["summarize"](s["id"], *_form(edited))

    assert "Faltan decisiones en las variaciones 5" in status
    assert text == ""
    assert client.get(f"/api/v1/sessions/{s['id']}").json()["status"] == "REVIEWED"


def test_summary_after_freeze_resets_form_to_record(client, handlers):
    s = ready_session(client)
    decide_all(client, s, DECISIONS)
    handlers["summarize"](s["id"], *_form(DECISIONS))

    edited = ["accepted"] * 5
    status, _text, *form = handlers["summarize"](s["id"], *_form(edited))
    assert "no se aplicaron" in status
    assert [u["value"] for u in form[0::2]] == DECISIONS
