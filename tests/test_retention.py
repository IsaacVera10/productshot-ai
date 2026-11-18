"""Retención (G11): borrado automático de las cadenas de rondas a las N horas de su primera ronda."""

from __future__ import annotations

import time
from datetime import timedelta

import pytest

from conftest import create_session, decide_all, ready_session
from productshot.domain import utcnow
from test_mask_and_delete import _chain_of_two, _ParkedRunner

API = "/api/v1/sessions"


def purge(client, hours_later: float) -> list[str]:
    """Corre una pasada de retención como si hubieran pasado `hours_later` horas."""
    return client.app.state.sessions.purge_expired(now=utcnow() + timedelta(hours=hours_later))


def test_expired_chain_is_deleted_with_its_files(client, settings):
    root, child = _chain_of_two(client)

    assert purge(client, 23.9) == []  # todavía dentro del plazo
    assert sorted(purge(client, 24.1)) == sorted([root["id"], child["id"]])

    for sid in (root["id"], child["id"]):
        assert client.get(f"{API}/{sid}").status_code == 404
        assert not (settings.asset_root / sid).exists()
    assert client.get(child["variations"][0]["image_url"]).status_code == 404


def test_deadline_counts_from_the_first_round(client):
    """Una ronda nueva no extiende la vida de la cadena: el plazo es desde que se subió la foto."""
    s = ready_session(client)
    decide_all(client, s, ["accepted"] * 5)
    client.post(f"{API}/{s['id']}/summary")
    child = client.post(f"{API}/{s['id']}/next-round").json()

    # La hija se creó ahora, pero la raíz ya venció: se borra la cadena completa.
    assert sorted(purge(client, 24.1)) == sorted([s["id"], child["id"]])


def test_only_expired_chains_are_touched(client, settings):
    old = ready_session(client)
    time.sleep(0.01)
    recent = ready_session(client)
    sessions = client.app.state.sessions
    created_old = sessions.repo.get_session(old["id"]).created_at
    created_recent = sessions.repo.get_session(recent["id"]).created_at
    assert created_old < created_recent

    now = created_old + timedelta(hours=24, microseconds=1)
    assert now - timedelta(hours=24) < created_recent  # `recent` sigue dentro del plazo
    assert sessions.purge_expired(now=now) == [old["id"]]
    assert client.get(f"{API}/{recent['id']}").json()["status"] == "READY_FOR_REVIEW"
    assert (settings.asset_root / recent["id"]).exists()


def test_chain_that_is_generating_is_skipped_and_retried(make_client):
    runner = _ParkedRunner()
    client = make_client(runner=runner)
    sid = create_session(client).json()["id"]
    client.post(f"{API}/{sid}/segment")
    client.post(f"{API}/{sid}/generate")

    # La generación sigue dentro de G14: no se puede borrar todavía.
    assert purge(client, 24.1) == []
    assert client.get(f"{API}/{sid}").json()["status"] == "GENERATING"

    runner.fn()  # termina la generación
    assert purge(client, 24.1) == [sid]


def test_hung_generation_does_not_block_retention_forever(make_client, settings):
    """Un lote colgado pasa a FAILED por G14 y la cadena se borra en la misma pasada."""
    settings.generation_timeout_s = 0.0
    client = make_client(runner=_ParkedRunner())
    sid = create_session(client).json()["id"]
    client.post(f"{API}/{sid}/segment")
    client.post(f"{API}/{sid}/generate")
    time.sleep(0.01)

    assert purge(client, 24.1) == [sid]


def test_session_deleted_by_the_person_is_ignored(client):
    s = ready_session(client)
    sessions = client.app.state.sessions
    roots = sessions.repo.chain_roots_created_before(utcnow() + timedelta(hours=1))
    assert client.delete(f"{API}/{s['id']}").status_code == 204

    # Simula la carrera: la lista de vencidas se leyó antes de que la persona borrara.
    sessions.repo.chain_roots_created_before = lambda cutoff: roots
    assert sessions.purge_expired(now=utcnow() + timedelta(hours=25)) == []


def test_zero_disables_retention(client, settings):
    settings.session_retention_hours = 0
    s = ready_session(client)
    assert purge(client, 1000) == []
    assert client.get(f"{API}/{s['id']}").status_code == 200


def test_startup_sweep_deletes_expired_chains(make_client, settings):
    """La pasada al arrancar borra lo que venció mientras el servidor estaba apagado."""
    s = ready_session(make_client())

    settings.session_retention_hours = 1e-9  # todo lo existente ya venció
    second = make_client()  # nuevo arranque sobre la misma base
    for _ in range(200):  # la pasada corre en un hilo aparte
        if second.get(f"{API}/{s['id']}").status_code == 404:
            break
        time.sleep(0.01)
    assert second.get(f"{API}/{s['id']}").status_code == 404


def test_healthz_reports_retention(client):
    assert client.get("/api/v1/healthz").json()["retention_hours"] == 24.0


@pytest.mark.parametrize("hours, expected", [
    (24, "24 horas"),
    (1, "1 hora"),
    (0.5, "0.5 horas"),
])
def test_ui_retention_notice(hours, expected):
    pytest.importorskip("gradio")
    from productshot_ui.app import retention_notice

    assert expected in retention_notice(hours)
    assert retention_notice(0) == ""
