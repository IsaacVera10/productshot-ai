"""Orquestador del flujo de una sesión: imagen → máscara → 5 variaciones → decisiones → resumen."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import random
import uuid

from PIL import Image

from productshot.config import Settings
from productshot.domain import (
    CANONICAL_STYLES,
    MASK_EDITABLE_STATES,
    MAX_ROUNDS,
    REASON_MAX_CHARS,
    VARIATIONS_PER_BATCH,
    Decision,
    DecisionValue,
    DependencyUnavailable,
    DomainError,
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
    normalize_reason,
    utcnow,
)
from productshot.services import guardrails
from productshot.services.generation import (
    GenerationBackend,
    GenerationRequest,
    GenerationUnavailable,
    VariationSpec,
)
from productshot.services.images import prepare_mask_upload, prepare_upload, to_png
from productshot.services.jobs import JobRunner
from productshot.services.labels import stamp
from productshot.services.presets import PROMPT_VERSION
from productshot.services.rounds import RoundPlan, plan_next_round
from productshot.services.segmentation import SegmentationService, SegmentationUnavailable, render_mask_preview
from productshot.services.summary import STATUS_VERIFIED, SummaryService
from productshot.storage.artifacts import ArtifactStore
from productshot.storage.ports import Repository

log = logging.getLogger(__name__)

IDEMPOTENCY_KEY_MAX = 128
INTERRUPTED_REASON = "Trabajo interrumpido por un reinicio del servicio; crea una nueva sesión."
SUBMIT_FAILED_REASON = "El servicio de generación no está disponible; crea una nueva sesión e inténtalo más tarde."


class BatchError(Exception):
    """El backend devolvió un lote que no cumple el contrato de cinco variaciones."""


def _new_id() -> str:
    return str(uuid.uuid4())


class SessionService:
    def __init__(
        self,
        settings: Settings,
        repo: Repository,
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
        try:
            prepared = prepare_upload(
                data,
                max_bytes=self.settings.max_upload_bytes,
                max_pixels=self.settings.max_image_pixels,
                work_max_side=self.settings.work_max_side,
            )
        except DomainError as exc:
            guardrails.event("G1", None, exc.code)
            raise
        session_id, image_id = _new_id(), _new_id()
        metrics = guardrails.photo_metrics(prepared.working)
        quality = guardrails.photo_warnings(metrics, self.settings)
        for w in quality:
            guardrails.event(w.guardrail, session_id, w.code, **metrics)
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
            quality_json=json.dumps({"metrics": metrics, "warnings": [w.to_json() for w in quality]}, ensure_ascii=False),
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

    def _check_mask_editable(self, session_id: str) -> None:
        session = self.repo.get_session(session_id)
        if session.parent_session_id is not None:
            raise StateConflict("Esta ronda usa la máscara de la ronda 1; no se puede cambiar.")
        if session.status not in MASK_EDITABLE_STATES:
            raise StateConflict(f"La máscara ya no se puede cambiar (estado {session.status}).")

    def segment(self, session_id: str) -> Mask:
        self._check_mask_editable(session_id)
        image = self._load_image(self.repo.get_image(session_id).storage_key)
        try:
            result = self.segmenter.segment(image)
        except SegmentationUnavailable as exc:
            raise DependencyUnavailable(str(exc)) from exc
        if result.mask.size != image.size:
            raise DependencyUnavailable("El segmentador devolvió una máscara con dimensiones incorrectas.")
        return self._store_mask(session_id, image, result.mask.convert("L"), "auto", result.simulated)

    def upload_mask(self, session_id: str, data: bytes) -> Mask:
        """Máscara manual en PNG (ADR-005): blanco = fondo editable, negro = producto protegido."""
        self._check_mask_editable(session_id)
        image = self._load_image(self.repo.get_image(session_id).storage_key)
        mask = prepare_mask_upload(
            data, max_bytes=self.settings.max_upload_bytes, max_pixels=self.settings.max_image_pixels,
            expected_size=image.size,
        )
        return self._store_mask(session_id, image, mask, "manual", simulated=False)

    def _store_mask(self, session_id: str, image: Image.Image, mask_img: Image.Image, source: str,
                    simulated: bool) -> Mask:
        mask_png = to_png(mask_img)
        preview_png = to_png(render_mask_preview(image, mask_img, simulated))
        suffix = uuid.uuid4().hex[:12]
        metrics = guardrails.mask_metrics(mask_img, self.settings)
        warnings = guardrails.mask_warnings(metrics, self.settings)
        for w in warnings:
            guardrails.event(w.guardrail, session_id, w.code, source=source, **metrics)
        mask = Mask(
            id=_new_id(),
            session_id=session_id,
            source=source,
            storage_key=f"{session_id}/mask_{suffix}.png",
            preview_key=f"{session_id}/mask_{suffix}_preview.png",
            sha256=hashlib.sha256(mask_png).hexdigest(),
            simulated=simulated,
            created_at=utcnow(),
            warnings_json=json.dumps({"metrics": metrics, "warnings": [w.to_json() for w in warnings]},
                                     ensure_ascii=False),
        )
        try:
            # Las escrituras van dentro del try: si falla la segunda (o la transacción), no
            # queda un archivo de máscara huérfano. `delete` tolera archivos inexistentes.
            self.store.save(mask_png, mask.storage_key)
            self.store.save(preview_png, mask.preview_key)
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
        if session.round_plan_json:
            # Rondas 2 y 3: el plan ya fijó estilo, ajustes y semilla de cada posición.
            specs = [
                VariationSpec(
                    position=it["position"], style=Style(it["style"]), seed=it["seed"],
                    adjustments=tuple(it["adjustments"]),
                )
                for it in json.loads(session.round_plan_json)["items"]
            ]
        else:
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
            "round_index": session.round_index,
            "specs": [
                {"position": s.position, "style": s.style.value, "seed": s.seed, "adjustments": list(s.adjustments)}
                for s in specs
            ],
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
            try:
                self.runner.submit(lambda: self._run_job(job))
            except Exception as exc:
                # El trabajo ya está confirmado en GENERATING: sin esto quedaría así hasta reiniciar.
                log.exception("No se pudo encolar la generación de la sesión %s", session_id)
                self.repo.fail_job(job.id, SUBMIT_FAILED_REASON)
                raise DependencyUnavailable(SUBMIT_FAILED_REASON) from exc
        return self.repo.get_job(session_id) or job, created

    def is_simulated(self, view: SessionView) -> bool:
        """True si algún componente de ESTA sesión es simulado.

        Usa lo que quedó registrado (máscara y configuración del trabajo), no el backend
        actual del servidor: una sesión creada con mock sigue rotulada como simulada aunque
        después se active un backend real. Solo para los pasos aún no ejecutados se usa el
        backend configurado.
        """
        seg = view.mask.simulated if view.mask else self.segmenter.simulated
        if view.job is not None:
            gen = bool(json.loads(view.job.config_json).get("simulated", True))
        else:
            gen = self.generator.simulated
        return seg or gen

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
                VariationSpec(
                    position=s["position"], style=Style(s["style"]), seed=s["seed"],
                    adjustments=tuple(s.get("adjustments", ())),
                )
                for s in config["specs"]
            )
            artifacts = self.generator.generate(
                GenerationRequest(session_id=job.session_id, mode=session.mode, image=image, mask=mask, specs=specs)
            )
            self._check_batch(artifacts, specs)

            variations = []
            for art in sorted(artifacts, key=lambda a: a.position):
                # G8: el producto se recompone SIEMPRE desde la foto de trabajo; la métrica dice
                # cuánto lo había cambiado el generador.
                try:
                    final, fidelity = guardrails.enforce_fidelity(image, mask, art.image, self.settings)
                except ValueError as exc:
                    raise BatchError(f"Variación {art.position}: {exc}") from exc
                if fidelity["protected_max_diff_after"] != 0:
                    raise BatchError(f"La variación {art.position} alteró el producto tras recomponerlo.")
                if fidelity["drift_warning"]:
                    guardrails.event("G8", job.session_id, "product_drift", position=art.position,
                                     diff=fidelity["protected_mean_abs_diff"])
                if art.params.get("simulated", self.generator.simulated):
                    # G9: el rótulo se pone DESPUÉS de recomponer, para no contarlo como alteración.
                    final = stamp(final, bottom=f"{art.position}/5 · {art.variant or art.style.value} · "
                                              f"modo {session.mode.value}")
                art_params = {**art.params, "fidelity": fidelity}
                key = f"{job.session_id}/variation_{art.position}.png"
                written.append(self.store.save(to_png(final), key))
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
                        params_json=json.dumps(art_params, ensure_ascii=False),
                        variant=art.variant or art.style.value,
                    )
                )
            if not all(self.store.exists(k) for k in written):
                raise BatchError("No se pudieron guardar los cinco archivos.")
            self.repo.complete_job(job.id, variations)
            log.info("Lote completo para la sesión %s", job.session_id)
        except Exception as exc:
            for key in written:
                self.store.delete(key)
            readable = isinstance(exc, (BatchError, GenerationUnavailable))
            message = str(exc) if readable else f"Error de generación: {type(exc).__name__}"
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
        # Ronda 1: cinco estilos distintos. Rondas siguientes: cinco (estilo + ajustes) distintos,
        # porque un estilo aceptado puede repetirse con otra semilla o con otro ajuste.
        if len({a.variant or a.style.value for a in artifacts}) != VARIATIONS_PER_BATCH:
            raise BatchError("El lote no tiene cinco variantes distintas.")
        for a in artifacts:
            spec = expected[a.position]
            if a.style != spec.style or a.seed != spec.seed or (a.variant or a.style.value) != spec.variant:
                raise BatchError(f"La variación {a.position} no corresponde al estilo, ajustes o semilla pedidos.")
            if a.image is None or a.image.size[0] == 0:
                raise BatchError(f"La variación {a.position} no tiene imagen.")

    # --- 6. rondas (ADR-007) ---

    def _plan_for(self, session_id: str, note: str | None) -> tuple[Session, RoundPlan]:
        parent = self.repo.get_session(session_id)
        if parent.status != SessionStatus.SUMMARIZED:
            raise StateConflict("Para pedir otra ronda, primero resume las cinco decisiones de esta.")
        if parent.round_index >= MAX_ROUNDS:
            raise StateConflict(f"Se alcanzó el máximo de {MAX_ROUNDS} rondas.")
        if self.repo.child_session_id(session_id) is not None:
            raise StateConflict("Esta ronda ya tiene una ronda siguiente.")
        # Las decisiones están congeladas desde el resumen, así que leerlas fuera de la
        # transacción es seguro; la creación vuelve a verificar el estado de forma atómica.
        plan = plan_next_round(
            parent_session_id=session_id,
            parent_round=parent.round_index,
            variations=self.repo.list_variations(session_id),
            note=normalize_reason(note) if note else None,
            rng=self._rng,
        )
        return parent, plan

    def preview_next_round(self, session_id: str, note: str | None) -> dict:
        """Plan de la ronda siguiente sin crearla: la persona ve qué se entendió antes de generar."""
        _parent, plan = self._plan_for(session_id, note)
        data = plan.to_json()
        for item in data["items"]:
            item.pop("seed")  # la semilla real se sortea al crear la ronda
        return data

    def next_round(self, session_id: str, note: str | None) -> SessionView:
        parent, plan = self._plan_for(session_id, note)
        parent_image = self.repo.get_image(session_id)
        parent_mask = self.repo.get_mask(session_id)
        if parent_mask is None:  # una sesión resumida siempre tiene máscara
            raise StateConflict("La ronda anterior no tiene máscara.")
        child_id, image_id, now = _new_id(), _new_id(), utcnow()
        child = Session(
            id=child_id,
            mode=parent.mode,
            status=SessionStatus.SEGMENTED,
            image_id=image_id,
            base_model_revision=self.generator.model_revision,
            created_at=now,
            updated_at=now,
            parent_session_id=session_id,
            round_index=plan.round_index,
            round_plan_json=json.dumps(plan.to_json(), ensure_ascii=False),
        )
        # La hija apunta a los MISMOS archivos de la madre: no se copian ni se pueden borrar
        # desde la hija (su máscara está bloqueada).
        image = dataclasses.replace(parent_image, id=image_id, session_id=child_id)
        mask = dataclasses.replace(parent_mask, id=_new_id(), session_id=child_id, created_at=now)
        self.repo.create_child_session(child, image, mask, MAX_ROUNDS)
        log.info("Ronda %d creada: %s (madre %s)", plan.round_index, child_id, session_id)
        return self.get_view(child_id)

    # --- 7. borrado (G11) ---

    def delete_session(self, session_id: str) -> list[str]:
        """Borra la cadena completa de rondas de la sesión: registros y archivos.

        Todas las rondas comparten la foto y la máscara de la ronda 1, así que borrar solo
        una dejaría a las demás apuntando a archivos inexistentes. Devuelve los IDs borrados.
        """
        deleted = self.repo.delete_chain(session_id)
        for sid in deleted:
            self.store.delete_session_dir(sid)
        log.info("Cadena borrada: %d sesiones (pedido sobre %s)", len(deleted), session_id)
        return deleted

    # --- lectura ---

    def _expire_if_stale(self, session: Session) -> Session:
        """G14: un lote que supera el timeout pasa a FAILED. Si el trabajo termina después,
        `complete_job` lo rechaza (ya no está GENERATING) y sus archivos se borran."""
        if session.status != SessionStatus.GENERATING:
            return session
        job = self.repo.get_job(session.id)
        if job is None or (utcnow() - job.created_at).total_seconds() <= self.settings.generation_timeout_s:
            return session
        reason = (f"La generación tardó más de {int(self.settings.generation_timeout_s)} s; "
                  "el servicio está ocupado. Crea una nueva sesión e inténtalo más tarde.")
        self.repo.fail_job(job.id, reason)
        guardrails.event("G14", session.id, "batch_timeout", timeout_s=self.settings.generation_timeout_s)
        return self.repo.get_session(session.id)

    def get_view(self, session_id: str) -> SessionView:
        session = self._expire_if_stale(self.repo.get_session(session_id))
        variations = self.repo.list_variations(session_id) if session.status in VARIATIONS_VISIBLE_STATES else []
        return SessionView(
            session=session,
            image=self.repo.get_image(session_id),
            mask=self.repo.get_mask(session_id),
            job=self.repo.get_job(session_id),
            variations=variations,
            summary=self.repo.get_summary(session_id),
            next_session_id=self.repo.child_session_id(session_id),
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
        plan = json.loads(session.round_plan_json) if session.round_plan_json else None
        result = self.summarizer.summarize(session_id, variations, round_plan=plan)
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
