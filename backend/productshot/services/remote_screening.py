"""Adaptador `ScreeningService` que pide G2 y G3 al worker por HTTP (`/v1/screen`).

Usa la misma URL y el mismo token que la generación. Si no se puede revisar la foto, falla
cerrado (`ScreeningUnavailable`): una foto sin revisar no entra al sistema.
"""

from __future__ import annotations

import httpx
from PIL import Image
from pydantic import ValidationError

from productshot.services.screening import ScreeningResult, ScreeningUnavailable
from productshot.worker.contract import CONTRACT_VERSION, SCREEN_PATH, ScreenIn, ScreenOut, encode_png


class RemoteScreeningService:
    name = "remote"
    simulated = False

    def __init__(self, base_url: str, *, token: str | None = None, timeout_s: float = 300.0,
                 client: httpx.Client | None = None):
        # Timeout largo: la primera solicitud puede esperar el arranque en frío del worker.
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), headers=headers)
        if client is not None and token:
            self._client.headers.update(headers)
        self._timeout = timeout_s

    def screen(self, image: Image.Image) -> ScreeningResult:
        body = ScreenIn(image_png_b64=encode_png(image.convert("RGB")))
        try:
            resp = self._client.post(SCREEN_PATH, json=body.model_dump(mode="json"), timeout=self._timeout)
        except httpx.TimeoutException as exc:
            raise ScreeningUnavailable("El revisor de fotos no respondió a tiempo.") from exc
        except httpx.HTTPError as exc:
            raise ScreeningUnavailable("No se pudo conectar con el revisor de fotos.") from exc
        if resp.status_code == 401:
            raise ScreeningUnavailable("El worker rechazó el token configurado.")
        if resp.status_code == 404:
            raise ScreeningUnavailable("El worker configurado no revisa fotos.")
        if resp.status_code >= 400:
            raise ScreeningUnavailable(f"El revisor de fotos respondió con error {resp.status_code}.")
        try:
            out = ScreenOut.model_validate(resp.json())
        except (ValidationError, ValueError) as exc:
            raise ScreeningUnavailable("El revisor devolvió una respuesta que no cumple el contrato.") from exc
        if out.contract_version != CONTRACT_VERSION:
            raise ScreeningUnavailable(f"El worker usa la versión de contrato {out.contract_version}.")
        return ScreeningResult(nsfw_score=out.nsfw_score, faces=out.faces, simulated=out.simulated,
                               seconds=out.seconds, categories=out.categories)
