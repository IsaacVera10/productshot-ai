"""Adaptador `GenerationBackend` que llama a un worker por HTTP (ADR-009).

Es el mismo para Modal, Cloud Run y la laptop: solo cambia `GENERATION_WORKER_URL`. El
orquestador sigue haciendo todas las validaciones (lote de cinco, G8, G9, G14) sobre lo
que devuelve, igual que con el mock: el worker no es de confianza para eso.
"""

from __future__ import annotations

import logging
import threading

import httpx
from pydantic import ValidationError

from productshot.services.generation import GeneratedArtifact, GenerationRequest, GenerationUnavailable
from productshot.worker.contract import (
    CONTRACT_VERSION,
    GENERATE_PATH,
    HEALTH_PATH,
    ContractError,
    GenerateIn,
    GenerateOut,
    HealthOut,
    SpecIn,
    decode_png,
    encode_png,
)

log = logging.getLogger(__name__)

UNREACHABLE_REVISION = "remote:sin-conexion"


class RemoteGenerationBackend:
    name = "remote"

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout_s: float = 540.0,
        health_timeout_s: float = 10.0,
        client: httpx.Client | None = None,
    ):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        # `client` permite inyectar el TestClient del worker en las pruebas (sin red).
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), headers=headers)
        if client is not None and token:
            self._client.headers.update(headers)
        self._timeout = timeout_s
        self._health_timeout = health_timeout_s
        self._health: HealthOut | None = None
        self._lock = threading.Lock()

    # --- identidad del worker (revisión del modelo y si es simulado) ---

    def _info(self) -> HealthOut | None:
        with self._lock:
            if self._health is None:
                try:
                    resp = self._client.get(HEALTH_PATH, timeout=self._health_timeout)
                    resp.raise_for_status()
                    self._health = HealthOut.model_validate(resp.json())
                except (httpx.HTTPError, ValidationError, ValueError) as exc:
                    log.warning("Worker de generación sin respuesta en %s: %s", HEALTH_PATH, type(exc).__name__)
                    return None
            return self._health

    @property
    def model_revision(self) -> str:
        info = self._info()
        return info.model_revision if info else UNREACHABLE_REVISION

    @property
    def simulated(self) -> bool:
        # Si no se sabe, se asume simulado: es preferible rotular de más que presentar
        # como real algo que no lo es (G9).
        info = self._info()
        return True if info is None else info.simulated

    # --- generación ---

    def generate(self, request: GenerationRequest) -> list[GeneratedArtifact]:
        body = GenerateIn(
            session_id=request.session_id,
            mode=request.mode,
            prompt_version=request.prompt_version,
            image_png_b64=encode_png(request.image.convert("RGB")),
            mask_png_b64=encode_png(request.mask.convert("L")),
            specs=[
                SpecIn(position=s.position, style=s.style, seed=s.seed, adjustments=list(s.adjustments))
                for s in request.specs
            ],
            category=request.category,
        )
        try:
            # Modal corta a los 150 s con un 303 al mismo endpoint (con el id de la llamada):
            # se sigue como GET y espera el resultado. Mismo origen: el token se conserva.
            resp = self._client.post(GENERATE_PATH, json=body.model_dump(mode="json"), timeout=self._timeout,
                                     follow_redirects=True)
        except httpx.TimeoutException as exc:
            raise GenerationUnavailable("El worker de generación no respondió a tiempo.") from exc
        except httpx.HTTPError as exc:
            raise GenerationUnavailable("No se pudo conectar con el worker de generación.") from exc
        if resp.status_code == 401:
            raise GenerationUnavailable("El worker de generación rechazó el token configurado.")
        if resp.status_code == 422:
            # El worker explica por qué no entrega el lote (G7 o producto sin verificar): se muestra tal cual.
            try:
                detail = str(resp.json().get("detail") or "")
            except ValueError:
                detail = ""
            raise GenerationUnavailable(detail or "El worker no pudo generar un lote válido.")
        if resp.status_code >= 400:
            raise GenerationUnavailable(f"El worker de generación respondió con error {resp.status_code}.")
        try:
            out = GenerateOut.model_validate(resp.json())
        except (ValidationError, ValueError) as exc:
            raise GenerationUnavailable("El worker devolvió una respuesta que no cumple el contrato.") from exc
        if out.contract_version != CONTRACT_VERSION:
            raise GenerationUnavailable(f"El worker usa la versión de contrato {out.contract_version}.")
        with self._lock:
            # La respuesta trae la identidad vigente del worker (puede haber cambiado de modelo).
            self._health = HealthOut(backend=out.backend, model_revision=out.model_revision, simulated=out.simulated)

        artifacts = []
        for a in out.artifacts:
            try:
                image = decode_png(a.image_png_b64, "RGB")
            except ContractError as exc:
                raise GenerationUnavailable(f"La imagen {a.position} del worker no es válida: {exc}") from exc
            params = {**a.params, "worker_backend": out.backend, "simulated": out.simulated}
            artifacts.append(
                GeneratedArtifact(
                    position=a.position, style=a.style, image=image, seed=a.seed,
                    model_revision=a.model_revision, prompt_version=a.prompt_version,
                    params=params, variant=a.variant,
                )
            )
        return artifacts
