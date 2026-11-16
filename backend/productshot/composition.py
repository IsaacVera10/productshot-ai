"""Raíz de composición (ADR-009): elige el adaptador de cada pieza según la configuración.

Es el ÚNICO lugar que conoce las clases concretas. Dominio, orquestador, API y UI solo
ven interfaces, así que cambiar de entorno (laptop, Modal, GCP) es cambiar variables de
entorno, no código:

| Pieza        | Variable             | Adaptadores hoy                    |
|--------------|----------------------|------------------------------------|
| Generación   | GENERATION_BACKEND   | mock, remote (worker HTTP)         |
| Segmentación | SEGMENTATION_BACKEND | mock, remote (rembg en el worker)  |
| G2 y G3      | SCREENING_BACKEND    | mock, remote (NSFW y rostros en el worker) |
| Resumen      | SUMMARY_BACKEND      | deterministic, gemini              |
| Ejecutor     | JOB_RUNNER           | thread, inline                     |
| Registro     | (fijo)               | SQLite (Firestore cuando haga falta) |
| Archivos     | (fijo)               | carpeta local o volumen            |

Una configuración incompleta (por ejemplo, `remote` sin URL) falla al arrancar con un
mensaje claro, no en la primera solicitud.
"""

from __future__ import annotations

from dataclasses import dataclass

from productshot.config import Settings
from productshot.services.generation import GenerationBackend, MockGenerationBackend
from productshot.services.jobs import InlineJobRunner, JobRunner, ThreadJobRunner
from productshot.services.screening import MockScreeningService, ScreeningService
from productshot.services.segmentation import MockSegmentationService, SegmentationService
from productshot.services.summary import DeterministicSummaryService, SummaryService
from productshot.storage.artifacts import ArtifactStore, LocalArtifactStore
from productshot.storage.ports import Repository
from productshot.storage.repository import SqliteRepository


class ConfigError(RuntimeError):
    """La configuración no alcanza para construir el adaptador pedido."""


@dataclass
class Services:
    repo: Repository
    store: ArtifactStore
    segmenter: SegmentationService
    generator: GenerationBackend
    summarizer: SummaryService
    runner: JobRunner
    screener: ScreeningService

    def describe(self) -> dict[str, str]:
        """Qué adaptador quedó en cada pieza (para el log de arranque y `/health`)."""
        return {
            "repository": type(self.repo).__name__,
            "artifacts": type(self.store).__name__,
            "segmentation": self.segmenter.name,
            "generation": self.generator.name,
            "summary": self.summarizer.name,
            "runner": type(self.runner).__name__,
            "screening": self.screener.name,
        }


def _secret(value) -> str | None:
    return value.get_secret_value() if value is not None else None


def build_generator(settings: Settings) -> GenerationBackend:
    if settings.generation_backend == "mock":
        return MockGenerationBackend()
    if settings.generation_backend == "remote":
        if not settings.generation_worker_url:
            raise ConfigError("GENERATION_BACKEND=remote necesita GENERATION_WORKER_URL.")
        from productshot.services.remote_generation import RemoteGenerationBackend

        return RemoteGenerationBackend(
            settings.generation_worker_url,
            token=_secret(settings.generation_worker_token),
            timeout_s=settings.generation_worker_timeout_s,
        )
    raise ConfigError(f"GENERATION_BACKEND desconocido: {settings.generation_backend!r}.")


def build_segmenter(settings: Settings) -> SegmentationService:
    if settings.segmentation_backend == "mock":
        return MockSegmentationService()
    if settings.segmentation_backend == "remote":
        if not settings.generation_worker_url:
            raise ConfigError("SEGMENTATION_BACKEND=remote necesita GENERATION_WORKER_URL (el mismo worker segmenta).")
        from productshot.services.remote_segmentation import RemoteSegmentationService

        return RemoteSegmentationService(
            settings.generation_worker_url, token=_secret(settings.generation_worker_token)
        )
    raise ConfigError(f"SEGMENTATION_BACKEND desconocido: {settings.segmentation_backend!r}.")


def build_screener(settings: Settings) -> ScreeningService:
    if settings.screening_backend == "mock":
        return MockScreeningService()
    if settings.screening_backend == "remote":
        if not settings.generation_worker_url:
            raise ConfigError("SCREENING_BACKEND=remote necesita GENERATION_WORKER_URL (el mismo worker revisa).")
        from productshot.services.remote_screening import RemoteScreeningService

        return RemoteScreeningService(settings.generation_worker_url, token=_secret(settings.generation_worker_token))
    raise ConfigError(f"SCREENING_BACKEND desconocido: {settings.screening_backend!r}.")


def build_summarizer(settings: Settings) -> SummaryService:
    if settings.summary_backend == "deterministic":
        return DeterministicSummaryService()
    if settings.summary_backend == "gemini":
        key = _secret(settings.gemini_api_key)
        if not key:
            raise ConfigError("SUMMARY_BACKEND=gemini necesita GEMINI_API_KEY (en .env o en los secretos de Modal).")
        from productshot.services.gemini_summary import GeminiSummaryService

        models = [settings.gemini_model_id]
        if settings.gemini_fallback_model_id:
            models.append(settings.gemini_fallback_model_id)
        return GeminiSummaryService(
            key, models, timeout_s=settings.gemini_timeout_s, budget_s=settings.gemini_budget_s
        )
    raise ConfigError(f"SUMMARY_BACKEND desconocido: {settings.summary_backend!r}.")


def build_runner(settings: Settings) -> JobRunner:
    if settings.job_runner == "thread":
        return ThreadJobRunner()
    if settings.job_runner == "inline":
        return InlineJobRunner()
    raise ConfigError(f"JOB_RUNNER desconocido: {settings.job_runner!r}.")


def build_services(
    settings: Settings,
    *,
    repo: Repository | None = None,
    store: ArtifactStore | None = None,
    segmenter: SegmentationService | None = None,
    generator: GenerationBackend | None = None,
    summarizer: SummaryService | None = None,
    runner: JobRunner | None = None,
    screener: ScreeningService | None = None,
) -> Services:
    """Construye todas las piezas. Los argumentos explícitos (pruebas) mandan sobre la configuración."""
    return Services(
        repo=repo or SqliteRepository(settings.sqlite_path),
        store=store or LocalArtifactStore(settings.asset_root),
        segmenter=segmenter or build_segmenter(settings),
        generator=generator or build_generator(settings),
        summarizer=summarizer or build_summarizer(settings),
        runner=runner or build_runner(settings),
        screener=screener or build_screener(settings),
    )
