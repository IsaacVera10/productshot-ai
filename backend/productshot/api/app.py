"""Aplicación FastAPI: rutas `/api/v1`, manejo uniforme de errores y montaje opcional de la UI."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from productshot.api.schemas import (
    API_PREFIX,
    DecisionIn,
    DecisionOut,
    GenerationInfo,
    JobOut,
    MaskOut,
    SegmentOut,
    SessionOut,
    SummaryOut,
    VariationOut,
    asset_url,
)
from productshot.config import Settings
from productshot.domain import DomainError, GenerationJob, Mask, Mode, NotFound, SessionView
from productshot.services.generation import GenerationBackend, MockGenerationBackend
from productshot.services.jobs import JobRunner, ThreadJobRunner
from productshot.services.segmentation import MockSegmentationService
from productshot.services.sessions import SessionService
from productshot.services.summary import DeterministicSummaryService
from productshot.storage.artifacts import InvalidArtifactKey, LocalArtifactStore
from productshot.storage.repository import SqliteRepository

log = logging.getLogger(__name__)


# --- serialización ---


def _mask_out(mask: Mask) -> MaskOut:
    return MaskOut(
        id=mask.id,
        source=mask.source,
        url=asset_url(mask.storage_key),
        preview_url=asset_url(mask.preview_key),
        simulated=mask.simulated,
    )


def _job_out(job: GenerationJob) -> JobOut:
    return JobOut(job_id=job.id, session_id=job.session_id, status=job.status, backend=job.backend, error=job.error)


def _session_out(view: SessionView, simulated: bool) -> SessionOut:
    s = view.session
    return SessionOut(
        id=s.id,
        mode=s.mode,
        status=s.status,
        image_id=s.image_id,
        image_url=asset_url(view.image.storage_key),
        created_at=s.created_at,
        updated_at=s.updated_at,
        failure_reason=s.failure_reason,
        simulated=simulated,
        mask=_mask_out(view.mask) if view.mask else None,
        job=_job_out(view.job) if view.job else None,
        variations=[
            VariationOut(
                id=v.id,
                position=v.position,
                style=v.style,
                image_url=asset_url(v.storage_key),
                generation=GenerationInfo(seed=v.seed, model_revision=v.model_revision, prompt_version=v.prompt_version),
                decision=(
                    DecisionOut(
                        variation_id=v.id,
                        decision=v.decision.value,
                        reason=v.decision.reason,
                        decided_at=v.decision.decided_at,
                    )
                    if v.decision
                    else None
                ),
            )
            for v in view.variations
        ],
        summary_available=view.summary is not None,
    )


def _error(status: int, code: str, message: str, details: list | None = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


# --- rutas ---


def _build_router() -> APIRouter:
    router = APIRouter(prefix=API_PREFIX)

    def svc(request: Request) -> SessionService:
        return request.app.state.sessions

    def simulated(request: Request) -> bool:
        s = svc(request)
        return s.generator.simulated or s.segmenter.name == "mock"

    @router.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @router.post("/sessions", status_code=201, response_model=SessionOut)
    def create_session(
        request: Request,
        image: Annotated[UploadFile, File(description="Foto JPG o PNG del producto")],
        mode: Annotated[Mode, Form()],
    ) -> SessionOut:
        # Ruta síncrona a propósito: FastAPI la ejecuta en el pool de hilos, de modo que
        # decodificar la imagen, escribir archivos y esperar el bloqueo de SQLite no
        # detiene el event loop ni las demás solicitudes (como el sondeo de la UI).
        limit = svc(request).settings.max_upload_bytes
        data = image.file.read(limit + 1)  # leer uno de más basta para detectar el exceso
        view = svc(request).create_session(data, mode)
        return _session_out(view, simulated(request))

    @router.get("/sessions/{session_id}", response_model=SessionOut)
    def get_session(request: Request, session_id: str) -> SessionOut:
        return _session_out(svc(request).get_view(session_id), simulated(request))

    @router.post("/sessions/{session_id}/segment", response_model=SegmentOut)
    def segment(request: Request, session_id: str) -> SegmentOut:
        mask = svc(request).segment(session_id)
        status = svc(request).repo.get_session(session_id).status
        return SegmentOut(session_id=session_id, status=status, mask=_mask_out(mask))

    @router.post("/sessions/{session_id}/generate", status_code=202, response_model=JobOut)
    def generate(
        request: Request,
        session_id: str,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> JobOut:
        job, _created = svc(request).generate(session_id, idempotency_key)
        return _job_out(job)

    @router.put("/sessions/{session_id}/variations/{variation_id}/decision", response_model=DecisionOut)
    def decide(request: Request, session_id: str, variation_id: str, body: DecisionIn) -> DecisionOut:
        d = svc(request).decide(session_id, variation_id, body.decision, body.reason)
        return DecisionOut(variation_id=d.variation_id, decision=d.value, reason=d.reason, decided_at=d.decided_at)

    @router.post("/sessions/{session_id}/summary", response_model=SummaryOut)
    def summary(request: Request, session_id: str) -> SummaryOut:
        record = svc(request).summarize(session_id)
        return SummaryOut(**json.loads(record.payload_json))

    @router.get("/assets/{session_id}/{name}")
    def asset(request: Request, session_id: str, name: str) -> Response:
        key = f"{session_id}/{name}"
        s = svc(request)
        try:
            # Solo se sirven claves que la base de datos asocia a la sesión.
            if key not in s.repo.session_artifact_keys(session_id):
                raise NotFound("Recurso no encontrado.")
            data = s.store.load(key)
        except (InvalidArtifactKey, FileNotFoundError):
            raise NotFound("Recurso no encontrado.") from None
        media = "image/jpeg" if key.endswith(".jpg") else "image/png"
        return Response(content=data, media_type=media, headers={"Cache-Control": "private, max-age=300"})

    return router


# --- fábrica ---


def create_app(
    settings: Settings | None = None,
    *,
    runner: JobRunner | None = None,
    generator: GenerationBackend | None = None,
    mount_ui: bool = True,
) -> FastAPI:
    settings = settings or Settings()
    runner = runner or ThreadJobRunner()
    sessions = SessionService(
        settings=settings,
        repo=SqliteRepository(settings.sqlite_path),
        store=LocalArtifactStore(settings.asset_root),
        segmenter=MockSegmentationService(),
        generator=generator or MockGenerationBackend(),
        summarizer=DeterministicSummaryService(),
        runner=runner,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        sessions.recover_interrupted_jobs()
        yield
        runner.shutdown()

    app = FastAPI(
        title="ProductShot AI",
        version="0.1.0",
        description="Hito 1: backends simulados (mock). No genera imágenes con IA ni usa un LLM.",
        lifespan=lifespan,
    )
    app.state.sessions = sessions
    app.include_router(_build_router())

    @app.exception_handler(DomainError)
    async def _domain_error(_: Request, exc: DomainError) -> JSONResponse:
        return _error(exc.http_status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in exc.errors()]
        return _error(422, "validation_error", "La solicitud no cumple el contrato.", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("Error no controlado", exc_info=exc)
        return _error(500, "internal_error", "Error interno.")

    if mount_ui:
        _mount_ui(app, settings)
    return app


def _mount_ui(app: FastAPI, settings: Settings) -> None:
    try:
        import gradio as gr

        from productshot_ui.app import build_ui
    except ImportError:
        log.warning("Gradio no está instalado: la UI no se monta (ejecuta `uv sync`).")
        return
    host = "127.0.0.1" if settings.app_host in ("0.0.0.0", "::") else settings.app_host
    api_base = f"http://{host}:{settings.app_port}{API_PREFIX}"
    gr.mount_gradio_app(app, build_ui(api_base), path="/ui")

    @app.get("/", include_in_schema=False)
    def _root() -> RedirectResponse:
        return RedirectResponse(url="/ui")
