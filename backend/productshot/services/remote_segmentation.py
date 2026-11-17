"""Adaptador `SegmentationService` que pide la máscara al worker por HTTP (`/v1/segment`).

Usa la misma URL y el mismo token que la generación remota: el worker que genera también
segmenta. La API sigue validando la máscara (tamaño exacto y G13) como con el mock.
"""

from __future__ import annotations

import logging

import httpx
from PIL import Image
from pydantic import ValidationError

from productshot.services.segmentation import MaskResult, SegmentationUnavailable
from productshot.worker.contract import (
    CONTRACT_VERSION,
    HEALTH_PATH,
    SEGMENT_PATH,
    ContractError,
    HealthOut,
    SegmentIn,
    SegmentOut,
    decode_png,
    encode_png,
)

log = logging.getLogger(__name__)


class RemoteSegmentationService:
    name = "remote"

    def __init__(self, base_url: str, *, token: str | None = None, timeout_s: float = 60.0,
                 client: httpx.Client | None = None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), headers=headers)
        if client is not None and token:
            self._client.headers.update(headers)
        self._timeout = timeout_s
        self._simulated: bool | None = None

    @property
    def simulated(self) -> bool:
        """Solo se usa antes de segmentar; ante la duda, simulado (G9)."""
        if self._simulated is None:
            try:
                resp = self._client.get(HEALTH_PATH, timeout=10)
                resp.raise_for_status()
                info = HealthOut.model_validate(resp.json())
                if info.segmenter_simulated is not None:
                    self._simulated = info.segmenter_simulated
            except (httpx.HTTPError, ValidationError, ValueError):
                return True
        return True if self._simulated is None else self._simulated

    def segment(self, image: Image.Image) -> MaskResult:
        body = SegmentIn(image_png_b64=encode_png(image.convert("RGB")))
        try:
            resp = self._client.post(SEGMENT_PATH, json=body.model_dump(mode="json"), timeout=self._timeout,
                                     follow_redirects=True)
        except httpx.TimeoutException as exc:
            raise SegmentationUnavailable("El segmentador no respondió a tiempo.") from exc
        except httpx.HTTPError as exc:
            raise SegmentationUnavailable("No se pudo conectar con el segmentador.") from exc
        if resp.status_code == 401:
            raise SegmentationUnavailable("El worker rechazó el token configurado.")
        if resp.status_code == 404:
            raise SegmentationUnavailable("El worker configurado no ofrece segmentación.")
        if resp.status_code >= 400:
            raise SegmentationUnavailable(f"El segmentador respondió con error {resp.status_code}.")
        try:
            out = SegmentOut.model_validate(resp.json())
            if out.contract_version != CONTRACT_VERSION:
                raise SegmentationUnavailable(f"El worker usa la versión de contrato {out.contract_version}.")
            mask = decode_png(out.mask_png_b64, "L")
        except (ValidationError, ValueError, ContractError) as exc:
            raise SegmentationUnavailable("El segmentador devolvió una respuesta que no cumple el contrato.") from exc
        self._simulated = out.simulated
        log.info("Máscara remota: %s en %s s", out.segmenter, out.seconds)
        return MaskResult(mask=mask, simulated=out.simulated)
