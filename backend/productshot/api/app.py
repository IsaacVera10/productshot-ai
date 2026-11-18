"""Aplicación FastAPI: rutas `/api/v1`, manejo uniforme de errores y montaje opcional de la UI."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from productshot.api.schemas import (
    API_PREFIX,
    DecisionIn,
    DecisionOut,
    GenerationInfo,
    JobOut,
    MaskOut,
    NextRoundIn,
    PlanItemOut,
    RoundPlanOut,
    SegmentOut,
    SessionOut,
    SummaryOut,
    VariationOut,
    asset_url,
)
from productshot.config import Settings
from productshot.composition import build_services
from productshot.domain import Category, DomainError, GenerationJob, Mask, Mode, NotFound, SessionView
from productshot.services.generation import GenerationBackend
from productshot.services.jobs import JobRunner
from productshot.services.rounds import explain_plan, variant_label
from productshot.services.screening import ScreeningService
from productshot.services.segmentation import SegmentationService
from productshot.services.sessions import SessionService
from productshot.services.summary import SummaryService
from productshot.storage.artifacts import InvalidArtifactKey

log = logging.getLogger(__name__)


# --- serialización ---


def _json(raw: str | None) -> dict:
    return json.loads(raw) if raw else {}


def _mask_out(mask: Mask) -> MaskOut:
    data = _json(mask.warnings_json)
    return MaskOut(
        id=mask.id,
        source=mask.source,
        url=asset_url(mask.storage_key),
        preview_url=asset_url(mask.preview_key),
        simulated=mask.simulated,
        warnings=data.get("warnings", []),
        metrics=data.get("metrics"),
    )


def _job_out(job: GenerationJob) -> JobOut:
    return JobOut(job_id=job.id, session_id=job.session_id, status=job.status, backend=job.backend, error=job.error)


def _session_out(view: SessionView, simulated: bool) -> SessionOut:
    s = view.session
    return SessionOut(
        id=s.id,
        mode=s.mode,
        category=s.category,
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
                variant=v.variant or v.style.value,
                label=variant_label(v.variant or v.style.value),
                image_url=asset_url(v.storage_key),
                generation=GenerationInfo(
                    seed=v.seed, model_revision=v.model_revision, prompt_version=v.prompt_version,
                    fidelity=_json(v.params_json).get("fidelity"),
                ),
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
        warnings=_json(view.image.quality_json).get("warnings", []) + (_json(view.mask.warnings_json).get("warnings", []) if view.mask else []),
        photo_quality=_json(view.image.quality_json).get("metrics"),
        round_index=s.round_index,
        parent_session_id=s.parent_session_id,
        next_session_id=view.next_session_id,
        round_plan=_plan_out(json.loads(s.round_plan_json)) if s.round_plan_json else None,
    )


def _plan_out(plan: dict) -> RoundPlanOut:
    return RoundPlanOut(
        round_index=plan["round_index"],
        parent_session_id=plan["parent_session_id"],
        rule=plan["rule"],
        note=plan["note"],
        note_adjustments=plan["note_adjustments"],
        ignored=plan["ignored"],
        items=[
            PlanItemOut(
                position=it["position"], style=it["style"], variant=it["variant"], label=variant_label(it["variant"]),
                adjustments=it["adjustments"], because=it["because"],
            )
            for it in plan["items"]
        ],
        explanation=explain_plan(plan),
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

    def out(request: Request, view: SessionView) -> SessionOut:
        return _session_out(view, svc(request).is_simulated(view))

    @router.get("/healthz")
    def healthz(request: Request) -> dict:
        # Solo nombres de adaptadores: nunca URLs, claves ni rutas.
        return {
            "status": "ok",
            "adapters": request.app.state.adapters,
            "retention_hours": svc(request).settings.session_retention_hours,
        }

    @router.post("/sessions", status_code=201, response_model=SessionOut)
    def create_session(
        request: Request,
        image: Annotated[UploadFile, File(description="Foto JPG o PNG del producto")],
        mode: Annotated[Mode, Form()],
        category: Annotated[Category | None, Form()] = None,
    ) -> SessionOut:
        # Ruta síncrona a propósito: FastAPI la ejecuta en el pool de hilos, de modo que
        # decodificar la imagen, escribir archivos y esperar el bloqueo de SQLite no
        # detiene el event loop ni las demás solicitudes (como el sondeo de la UI).
        limit = svc(request).settings.max_upload_bytes
        data = image.file.read(limit + 1)  # leer uno de más basta para detectar el exceso
        return out(request, svc(request).create_session(data, mode, category))

    @router.get("/sessions/{session_id}", response_model=SessionOut)
    def get_session(request: Request, session_id: str) -> SessionOut:
        return out(request, svc(request).get_view(session_id))

    @router.post("/sessions/{session_id}/segment", response_model=SegmentOut)
    def segment(request: Request, session_id: str) -> SegmentOut:
        mask = svc(request).segment(session_id)
        status = svc(request).repo.get_session(session_id).status
        return SegmentOut(session_id=session_id, status=status, mask=_mask_out(mask))

    @router.post("/sessions/{session_id}/mask", response_model=SegmentOut)
    def upload_mask(
        request: Request,
        session_id: str,
        mask: Annotated[UploadFile, File(description="PNG del tamaño de la imagen de trabajo: blanco = fondo, negro = producto")],
    ) -> SegmentOut:
        limit = svc(request).settings.max_upload_bytes
        stored = svc(request).upload_mask(session_id, mask.file.read(limit + 1))
        status = svc(request).repo.get_session(session_id).status
        return SegmentOut(session_id=session_id, status=status, mask=_mask_out(stored))

    @router.delete("/sessions/{session_id}", status_code=204)
    def delete_session(request: Request, session_id: str) -> Response:
        svc(request).delete_session(session_id)
        return Response(status_code=204)

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

    @router.post("/sessions/{session_id}/next-round", status_code=201, response_model=SessionOut)
    def next_round(request: Request, session_id: str, body: NextRoundIn | None = None) -> SessionOut:
        note = body.adjustment_note if body else None
        return out(request, svc(request).next_round(session_id, note))

    @router.post("/sessions/{session_id}/next-round/preview", response_model=RoundPlanOut)
    def preview_next_round(request: Request, session_id: str, body: NextRoundIn | None = None) -> RoundPlanOut:
        note = body.adjustment_note if body else None
        return _plan_out(svc(request).preview_next_round(session_id, note))

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
    segmenter: SegmentationService | None = None,
    summarizer: SummaryService | None = None,
    screener: ScreeningService | None = None,
    mount_ui: bool = True,
) -> FastAPI:
    settings = settings or Settings()
    services = build_services(
        settings, runner=runner, generator=generator, segmenter=segmenter, summarizer=summarizer,
        screener=screener,
    )
    runner = services.runner
    sessions = SessionService(
        settings=settings,
        repo=services.repo,
        store=services.store,
        segmenter=services.segmenter,
        generator=services.generator,
        summarizer=services.summarizer,
        runner=runner,
        screener=services.screener,
    )
    adapters = services.describe()
    log.info("Adaptadores: %s", ", ".join(f"{k}={v}" for k, v in adapters.items()))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        sessions.recover_interrupted_jobs()
        sweeper = asyncio.create_task(_retention_loop(sessions, settings))
        yield
        sweeper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweeper
        runner.shutdown()

    app = FastAPI(
        title="ProductShot AI",
        version="0.1.0",
        description="Generación de cinco variaciones, decisiones y resumen. Los adaptadores activos están en /api/v1/healthz.",
        lifespan=lifespan,
    )
    app.state.sessions = sessions
    app.state.adapters = adapters
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

    _mount_web(app)
    if mount_ui:
        _mount_ui(app, settings)
    return app


async def _retention_loop(sessions: SessionService, settings: Settings) -> None:
    """G11: borra las cadenas vencidas al arrancar y después cada `retention_sweep_interval_s`.

    Corre en un hilo para no bloquear el servidor, y un fallo de una pasada no detiene las
    siguientes.
    """
    if settings.session_retention_hours <= 0:
        log.info("Retención desactivada: las sesiones solo se borran a pedido.")
        return
    log.info("Retención: las cadenas se borran a las %s h de su primera ronda", settings.session_retention_hours)
    while True:
        try:
            purged = await asyncio.to_thread(sessions.purge_expired)
            if purged:
                log.info("Retención: %d sesiones borradas", len(purged))
        except Exception:  # noqa: BLE001
            log.exception("Retención: falló la pasada")
        await asyncio.sleep(max(settings.retention_sweep_interval_s, 1.0))


WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def _mount_web(app: FastAPI) -> None:
    """Web app mobile-first (HTML, CSS y JS estáticos) en /app; la raíz lleva ahí.

    Usa la API del mismo origen, así que funciona igual en la laptop que desde un celular en
    la misma red (APP_HOST=0.0.0.0). La UI de Gradio sigue en /ui como panel técnico.
    """
    if not (WEB_DIR / "index.html").exists():
        log.warning("No se encontró la web app en %s", WEB_DIR)
        return
    app.mount("/app", StaticFiles(directory=WEB_DIR, html=True), name="web")

    @app.get("/", include_in_schema=False)
    def _root() -> RedirectResponse:
        return RedirectResponse(url="/app/")


def _mount_ui(app: FastAPI, settings: Settings) -> None:
    try:
        import gradio as gr

        from productshot_ui.app import build_ui
    except ImportError:
        log.warning("Gradio no está instalado: la UI no se monta (ejecuta `uv sync`).")
        return
    host = "127.0.0.1" if settings.app_host in ("0.0.0.0", "::") else settings.app_host
    api_base = f"http://{host}:{settings.app_port}{API_PREFIX}"
    gr.mount_gradio_app(app, build_ui(api_base, retention_hours=settings.session_retention_hours), path="/ui")
