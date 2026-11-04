"""Ejecutores de trabajos de generación.

En local el trabajo corre en un hilo aparte del mismo proceso (no es procesamiento
distribuido). La API mantiene la semántica 202 para que el contrato no cambie cuando,
en GCP, Cloud Tasks invoque a un worker GPU.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

log = logging.getLogger(__name__)


class JobRunner(Protocol):
    def submit(self, fn: Callable[[], None]) -> None: ...
    def shutdown(self) -> None: ...


class InlineJobRunner:
    """Ejecuta el trabajo de inmediato, en el mismo hilo. Útil en pruebas deterministas."""

    def submit(self, fn: Callable[[], None]) -> None:
        fn()

    def shutdown(self) -> None:
        pass


class ThreadJobRunner:
    """Un solo hilo de trabajo: una generación a la vez, como un worker GPU con max-instances=1."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="generation")

    def submit(self, fn: Callable[[], None]) -> None:
        def _guarded() -> None:
            try:
                fn()
            except Exception:  # el propio trabajo marca FAILED; esto evita perder la traza
                log.exception("Trabajo de generación terminó con una excepción no controlada")

        self._pool.submit(_guarded)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
