"""Segmentación real del producto con `rembg` (Hito 2).

`rembg` devuelve el ALFA del primer plano (blanco = producto). El proyecto usa la
convención inversa (blanco = fondo editable, negro = producto protegido), así que la
máscara se invierte. Los bordes suaves se conservan: G8 los mezcla al recomponer.

`rembg` y `onnxruntime` se importan al cargar, no al importar el módulo; corre en el
worker (Modal o la laptop). `session_factory` permite probar sin el modelo.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from PIL import Image, ImageOps

from productshot.services.segmentation import MaskResult

log = logging.getLogger(__name__)

DEFAULT_MODEL = "isnet-general-use"


class RembgSegmentationService:
    name = "rembg"
    simulated = False

    def __init__(self, model: str = DEFAULT_MODEL, *, session_factory: Callable | None = None,
                 remove_fn: Callable | None = None):
        self.model = model
        self._session_factory = session_factory
        self._remove = remove_fn
        self._session = None
        self.last_seconds: float | None = None

    def load(self) -> None:
        if self._session is not None:
            return
        if self._session_factory is not None:
            self._session = self._session_factory(self.model)
        else:
            from rembg import new_session, remove

            self._session = new_session(self.model)
            self._remove = remove
        log.info("rembg cargado: %s", self.model)

    def segment(self, image: Image.Image) -> MaskResult:
        self.load()
        t0 = time.perf_counter()
        alpha = self._remove(image.convert("RGB"), session=self._session, only_mask=True, post_process_mask=True)
        alpha = alpha.convert("L")
        if alpha.size != image.size:
            alpha = alpha.resize(image.size, Image.Resampling.BILINEAR)
        self.last_seconds = round(time.perf_counter() - t0, 2)
        # Blanco = fondo editable; negro = producto protegido.
        return MaskResult(mask=ImageOps.invert(alpha), simulated=False)
