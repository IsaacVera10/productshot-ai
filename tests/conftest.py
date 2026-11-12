"""Fixtures comunes: base de datos y artefactos temporales, sin red ni GPU."""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from productshot.api.app import create_app
from productshot.config import Settings
from productshot.services.jobs import InlineJobRunner


def image_bytes(fmt: str = "PNG", size: tuple[int, int] = (320, 240), color=(200, 60, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        sqlite_path=tmp_path / "test.sqlite3",
        asset_root=tmp_path / "assets",
        max_upload_mb=1,
    )


@pytest.fixture
def make_client(settings) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def _make(**kwargs) -> TestClient:
        kwargs.setdefault("runner", InlineJobRunner())
        app = create_app(settings, mount_ui=False, **kwargs)
        client = TestClient(app)
        client.__enter__()  # ejecuta el lifespan (recuperación de trabajos)
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


# --- ayudas de flujo ---


def create_session(client: TestClient, mode: str = "intermediate", data: bytes | None = None,
                   filename: str = "producto.png", content_type: str = "image/png"):
    return client.post(
        "/api/v1/sessions",
        files={"image": (filename, data if data is not None else image_bytes(), content_type)},
        data={"mode": mode},
    )


def ready_session(client: TestClient, mode: str = "intermediate") -> dict:
    """Sesión con cinco variaciones listas para revisar."""
    sid = create_session(client, mode).json()["id"]
    assert client.post(f"/api/v1/sessions/{sid}/segment").status_code == 200
    assert client.post(f"/api/v1/sessions/{sid}/generate").status_code == 202
    body = client.get(f"/api/v1/sessions/{sid}").json()
    assert body["status"] == "READY_FOR_REVIEW", body
    return body


def decide(client: TestClient, sid: str, vid: str, decision: str, reason: str | None = None):
    return client.put(
        f"/api/v1/sessions/{sid}/variations/{vid}/decision",
        json={"decision": decision, "reason": reason},
    )


def decide_all(client: TestClient, session: dict, decisions: list[str], reasons: list | None = None) -> None:
    reasons = reasons or [None] * 5
    for v, d, r in zip(session["variations"], decisions, reasons):
        assert decide(client, session["id"], v["id"], d, r).status_code == 200
