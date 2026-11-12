"""Orquestador del flujo de una sesión: imagen → máscara → 5 variaciones → decisiones → resumen."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import random
import unicodedata
import uuid

from PIL import Image

from productshot.config import Settings
from productshot.domain import (
    CANONICAL_STYLES,
    MASK_EDITABLE_STATES,
    REASON_MAX_CHARS,
    VARIATIONS_PER_BATCH,
    Decision,
    DecisionValue,
    DependencyUnavailable,
    GenerationJob,
    InputImage,
    InvalidInput,
    JobStatus,
    Mask,
    Mode,
    Session,
    SessionStatus,
    SessionView,
    StateConflict,
    Style,
    SummaryRecord,
    Variation,
    VARIATIONS_VISIBLE_STATES,
    decisions_hash,
    utcnow,
)
from productshot.services.generation import GenerationBackend, GenerationRequest, VariationSpec
from productshot.services.images import prepare_upload, to_png
from productshot.services.jobs import JobRunner
from productshot.services.presets import PROMPT_VERSION
from productshot.services.segmentation import SegmentationService, render_mask_preview
from productshot.services.summary import STATUS_VERIFIED, SummaryService
from productshot.storage.artifacts import ArtifactStore
from productshot.storage.repository import SqliteRepository

log = logging.getLogger(__name__)

IDEMPOTENCY_KEY_MAX = 128
INTERRUPTED_REASON = "Trabajo interrumpido por un reinicio del servicio; crea una nueva sesión."


class BatchError(Exception):
    """El backend devolvió un lote que no cumple el contrato de cinco variaciones."""


def _new_id() -> str:
    return str(uuid.uuid4())


def normalize_reason(reason: str) -> str | None:
    """Deja el motivo en una sola línea, sin caracteres de control ni separadores de línea.

    El motivo es un DATO: se cita literalmente en el resumen y nunca se interpreta. Si
    conservara saltos de línea, podría imitar líneas del resumen (por ejemplo «Se aceptaron
    5 de 5») y contradecir el registro aunque la validación estructurada pase.
    """
    flat = "".join(
        "" if ch in _BIDI_CONTROLS else " " if unicodedata.category(ch) in _BREAKING_CATEGORIES else ch
        for ch in reason
    )
    return " ".join(flat.split()) or None


# Control (incluye \n, \r, \t, NUL) y separadores de línea o párrafo Unicode.
_BREAKING_CATEGORIES = frozenset({"Cc", "Zl", "Zp"})
# Controles bidireccionales que reordenan visualmente el texto. Otros caracteres de formato,
# como el ZWJ de los emojis compuestos, se conservan.
_BIDI_CONTROLS = frozenset("‪‫‬‭‮⁦⁧⁨⁩")


class SessionService:
    def __init__(
        self,
        settings: Settings,
        repo: SqliteRepository,
        store: ArtifactStore,
        segmenter: SegmentationService,
        generator: GenerationBackend,
        summarizer: SummaryService,
        runner: JobRunner,
    ):
        self.settings = settings
        self.repo = repo
        self.store = store
        self.segmenter = segmenter
        self.generator = generator
        self.summarizer = summarizer
        self.runner = runner
        self._rng = random.SystemRandom()

    # --- arranque ---

    def recover_interrupted_jobs(self) -> int:
        count = self.repo.fail_interrupted_jobs(INTERRUPTED_REASON)
        if count:
            log.warning("Se marcaron %d trabajos interrumpidos como FAILED", count)
        return count

    # --- 1. sesión ---

    def create_session(self, data: bytes, mode: Mode) -> SessionView:
        prepared = prepare_upload(
            data,
            max_bytes=self.settings.max_upload_bytes,
            max_pixels=self.settings.max_image_pixels,
            work_max_side=self.settings.work_max_side,
        )
        session_id, image_id = _new_id(), _new_id()
        ext = "jpg" if prepared.original_format == "JPEG" else "png"
        original_key = f"{session_id}/original.{ext}"
        working_key = f"{session_id}/input.png"
        now = utcnow()
        session = Session(
            id=session_id,
            mode=mode,
            status=SessionStatus.UPLOADED,
            image_id=image_id,
            base_model_revision=self.generator.model_revision,
            created_at=now,
            updated_at=now,
        )
        image = InputImage(
            id=image_id,
            session_id=session_id,
            storage_key=working_key,
            original_key=original_key,
            sha256=prepared.sha256,
            width=prepared.working.width,
            height=prepared.working.height,
            original_width=prepared.original_size[0],
            original_height=prepared.original_size[1],
            original_format=prepared.original_format,
        )
        written: list[str] = []
        try:
            written.append(self.store.save(prepared.original_bytes, original_key))
            written.append(self.store.save(prepared.working_png, working_key))
            self.repo.create_session(session, image)
        except BaseException:
            for key in written:
                self.store.delete(key)
            raise
        log.info("Sesión creada %s (modo=%s)", session_id, mode.value)
        return self.get_view(session_id)

    # --- 2. máscara ---

    def segment(self, session_id: str) -> Mask:
        session = self.repo.get_session(session_id)
        if session.status not in MASK_EDITABLE_STATES:
            raise StateConflict(f"La máscara ya no se puede cambiar (estado {session.status}).")
        image = self._load_image(self.repo.get_image(session_id).storage_key)
        result = self.segmenter.segment(image)
        if result.mask.size != image.size:
            raise DependencyUnavailable("El segmentador devolvió una máscara con dimensiones incorrectas.")
        mask_png = to_png(result.mask.convert("L"))
        preview_png = to_png(render_mask_preview(image, result.mask, result.simulated))
        suffix = uuid.uuid4().hex[:12]
        mask = Mask(
            id=_new_id(),
            session_id=session_id,
            source="auto",
            storage_key=f"{session_id}/mask_{suffix}.png",
            preview_key=f"{session_id}/mask_{suffix}_preview.png",
            sha256=hashlib.sha256(mask_png).hexdigest(),
            simulated=result.simulated,
            created_at=utcnow(),
        )
        self.store.save(mask_png, mask.storage_key)
        self.store.save(preview_png, mask.preview_key)
        try:
            previous = self.repo.replace_mask(mask, MASK_EDITABLE_STATES)
        except BaseException:
            self.store.delete(mask.storage_key)
            self.store.delete(mask.preview_key)
            raise
        if previous is not None:
            self.store.delete(previous.storage_key)
            self.store.delete(previous.preview_key)
        return mask

    # --- 3. generación ---

    def generate(self, session_id: str, idempotency_key: str | None) -> tuple[GenerationJob, bool]:
        if idempotency_key is not None:
            idempotency_key = idempotency_key.strip()
            if not idempotency_key or len(idempotency_key) > IDEMPOTENCY_KEY_MAX or not idempotency_key.isprintable():
                raise InvalidInput("Idempotency-Key inválida.")
        session = self.repo.get_session(session_id)
        specs = [
            VariationSpec(position=i + 1, style=style, seed=self._rng.randrange(0, 2**31 - 1))
            for i, style in enumerate(CANONICAL_STYLES)
        ]
        config = {
            "backend": self.generator.name,
            "simulated": self.generator.simulated,
            "mode": session.mode.value,
            "model_revision": self.generator.model_revision,
            "prompt_version": PROMPT_VERSION,
            "specs": [{"position": s.position, "style": s.style.value, "seed": s.seed} for s in specs],
        }
        job = GenerationJob(
            id=_new_id(),
            session_id=session_id,
            backend=self.generator.name,
            status=JobStatus.GENERATING,
            idempotency_key=idempotency_key,
            config_json=json.dumps(config),
            error=None,
            created_at=utcnow(),
            finished_at=None,
        )
        job, created = self.repo.start_job(job)
        if created:
            self.runner.submit(lambda: self._run_job(job))
        return self.repo.get_job(session_id) or job, created

    def _run_job(self, job: GenerationJob) -> None:
        written: list[str] = []
        try:
            session = self.repo.get_session(job.session_id)
            image = self._load_image(self.repo.get_image(job.session_id).storage_key)
            mask_record = self.repo.get_mask(job.session_id)
            if mask_record is None:
                raise BatchError("La sesión no tiene máscara.")
            mask = self._load_image(mask_record.storage_key, mode="L")
            config = json.loads(job.config_json)
            specs = tuple(
                VariationSpec(position=s["position"], style=Style(s["style"]), seed=s["seed"])
                for s in config["specs"]
            )
            artifacts = self.generator.generate(
                GenerationRequest(session_id=job.session_id, mode=session.mode, image=image, mask=mask, specs=specs)
            )
            self._check_batch(artifacts, specs)

            variations = []
            for art in sorted(artifacts, key=lambda a: a.position):
                key = f"{job.session_id}/variation_{art.position}.png"
                written.append(self.store.save(to_png(art.image), key))
                variations.append(
                    Variation(
                        id=_new_id(),
                        session_id=job.session_id,
                        job_id=job.id,
                        position=art.position,
                        style=art.style,
                        storage_key=key,
                        seed=art.seed,
                        model_revision=art.model_revision,
                        prompt_version=art.prompt_version,
                        params_json=json.dumps(art.params, ensure_ascii=False),
                    )
                )
            if not all(self.store.exists(k) for k in written):
                raise BatchError("No se pudieron guardar los cinco archivos.")
            self.repo.complete_job(job.id, variations)
            log.info("Lote completo para la sesión %s", job.session_id)
        except Exception as exc:
            for key in written:
                self.store.delete(key)
            message = str(exc) if isinstance(exc, BatchError) else f"Error de generación: {type(exc).__name__}"
            log.exception("Falló la generación de la sesión %s", job.session_id)
            self.repo.fail_job(job.id, message)

    @staticmethod
    def _check_batch(artifacts: list, specs: tuple[VariationSpec, ...]) -> None:
        if len(artifacts) != VARIATIONS_PER_BATCH:
            raise BatchError(f"El backend devolvió {len(artifacts)} variaciones en lugar de 5.")
        expected = {s.position: s for s in specs}
        positions = [a.position for a in artifacts]
        if sorted(positions) != sorted(expected):
            raise BatchError("Las posiciones del lote no son 1..5 sin repetición.")
        if len({a.style for a in artifacts}) != VARIATIONS_PER_BATCH:
            raise BatchError("El lote no tiene cinco estilos distintos.")
        for a in artifacts:
            spec = expected[a.position]
            if a.style != spec.style or a.seed != spec.seed:
                raise BatchError(f"La variación {a.position} no corresponde al estilo o semilla pedidos.")
            if a.image is None or a.image.size[0] == 0:
                raise BatchError(f"La variación {a.position} no tiene imagen.")

    # --- lectura ---

    def get_view(self, session_id: str) -> SessionView:
        session = self.repo.get_session(session_id)
        variations = self.repo.list_variations(session_id) if session.status in VARIATIONS_VISIBLE_STATES else []
        return SessionView(
            session=session,
            image=self.repo.get_image(session_id),
            mask=self.repo.get_mask(session_id),
            job=self.repo.get_job(session_id),
            variations=variations,
            summary=self.repo.get_summary(session_id),
        )

    # --- 4. decisiones ---

    def decide(self, session_id: str, variation_id: str, value: DecisionValue, reason: str | None) -> Decision:
        if reason is not None:
            reason = normalize_reason(reason)
            if reason and len(reason) > REASON_MAX_CHARS:
                raise InvalidInput(f"El motivo supera {REASON_MAX_CHARS} caracteres.")
        return self.repo.upsert_decision(session_id, variation_id, value, reason)

    # --- 5. resumen ---

    def summarize(self, session_id: str) -> SummaryRecord:
        session = self.repo.get_session(session_id)
        if session.status == SessionStatus.SUMMARIZED:
            existing = self.repo.get_summary(session_id)
            if existing is not None:
                return existing
        if session.status != SessionStatus.REVIEWED:
            raise StateConflict("El resumen requiere las cinco decisiones registradas.")
        variations = self.repo.list_variations(session_id)
        result = self.summarizer.summarize(session_id, variations)
        if result.validation_status != STATUS_VERIFIED:
            raise DependencyUnavailable("No se pudo producir un resumen verificado contra el registro.")
        record = SummaryRecord(
            session_id=session_id,
            text=result.text,
            payload_json=json.dumps(result.payload, ensure_ascii=False),
            generated_by=result.generated_by,
            validation_status=result.validation_status,
            validated_decisions_hash=decisions_hash(variations),
            created_at=utcnow(),
        )
        # Si otra solicitud ya guardó el resumen de estas mismas decisiones, se devuelve ese.
        return self.repo.save_summary(record)

    # --- utilidades ---

    def _load_image(self, key: str, mode: str = "RGB") -> Image.Image:
        with Image.open(io.BytesIO(self.store.load(key))) as img:
            img.load()
            return img.convert(mode)
