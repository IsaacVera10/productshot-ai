"""Entidades, enumeraciones y errores del dominio.

El dominio no conoce SQLite, archivos ni HTTP. `mock` nunca es un modo de dominio:
es un backend de ejecución local (ver docs/02_API_Y_MODELO_DE_DATOS.md).
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum


class Mode(StrEnum):
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class SessionStatus(StrEnum):
    UPLOADED = "UPLOADED"
    SEGMENTED = "SEGMENTED"
    GENERATING = "GENERATING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    REVIEWED = "REVIEWED"
    SUMMARIZED = "SUMMARIZED"
    FAILED = "FAILED"


class JobStatus(StrEnum):
    GENERATING = "GENERATING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    FAILED = "FAILED"


class DecisionValue(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Style(StrEnum):
    STUDIO = "studio"
    NATURAL = "natural"
    WARM = "warm"
    LIFESTYLE = "lifestyle"
    MINIMAL = "minimal"


# Lista canónica: la posición 1..5 corresponde a este orden.
CANONICAL_STYLES: tuple[Style, ...] = (
    Style.STUDIO,
    Style.NATURAL,
    Style.WARM,
    Style.LIFESTYLE,
    Style.MINIMAL,
)
VARIATIONS_PER_BATCH = len(CANONICAL_STYLES)
# Única fuente del límite del motivo: la usan el esquema HTTP y el servicio.
REASON_MAX_CHARS = 500
# Tope del texto CRUDO recibido; el límite real (REASON_MAX_CHARS) se mide tras normalizar.
REASON_RAW_MAX_CHARS = 4 * REASON_MAX_CHARS

# Estados en los que todavía se puede reemplazar la máscara.
MASK_EDITABLE_STATES = frozenset({SessionStatus.UPLOADED, SessionStatus.SEGMENTED})
# Estados en los que se aceptan o corrigen decisiones.
DECIDABLE_STATES = frozenset({SessionStatus.READY_FOR_REVIEW, SessionStatus.REVIEWED})
# Estados en los que las variaciones existen y se muestran.
VARIATIONS_VISIBLE_STATES = frozenset(
    {SessionStatus.READY_FOR_REVIEW, SessionStatus.REVIEWED, SessionStatus.SUMMARIZED}
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class InputImage:
    id: str
    session_id: str
    storage_key: str
    original_key: str
    sha256: str
    width: int
    height: int
    original_width: int
    original_height: int
    original_format: str


@dataclass(frozen=True)
class Mask:
    id: str
    session_id: str
    source: str  # "auto" | "manual"
    storage_key: str
    preview_key: str
    sha256: str
    simulated: bool
    created_at: datetime


@dataclass(frozen=True)
class GenerationJob:
    id: str
    session_id: str
    backend: str
    status: JobStatus
    idempotency_key: str | None
    config_json: str
    error: str | None
    created_at: datetime
    finished_at: datetime | None


@dataclass(frozen=True)
class Decision:
    variation_id: str
    value: DecisionValue
    reason: str | None
    decided_at: datetime


@dataclass(frozen=True)
class Variation:
    id: str
    session_id: str
    job_id: str
    position: int
    style: Style
    storage_key: str
    seed: int
    model_revision: str
    prompt_version: str
    params_json: str
    decision: Decision | None = None


@dataclass(frozen=True)
class Session:
    id: str
    mode: Mode
    status: SessionStatus
    image_id: str
    base_model_revision: str | None
    created_at: datetime
    updated_at: datetime
    failure_reason: str | None = None


@dataclass(frozen=True)
class SummaryRecord:
    session_id: str
    text: str
    payload_json: str
    generated_by: str
    validation_status: str
    validated_decisions_hash: str
    created_at: datetime


@dataclass(frozen=True)
class SessionView:
    """Lectura agregada de una sesión para la API."""

    session: Session
    image: InputImage
    mask: Mask | None
    job: GenerationJob | None
    variations: list[Variation] = field(default_factory=list)
    summary: SummaryRecord | None = None


def decisions_hash(variations: list[Variation]) -> str:
    """Huella de las cinco decisiones; permite detectar si cambiaron tras resumir."""
    rows = sorted(
        (v.id, v.decision.value if v.decision else None, v.decision.reason if v.decision else None)
        for v in variations
    )
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()


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


# --- Errores de dominio (la capa HTTP los traduce a códigos de estado) ---


class DomainError(Exception):
    code = "domain_error"
    http_status = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidInput(DomainError):
    code = "invalid_input"
    http_status = 400


class PayloadTooLarge(DomainError):
    code = "payload_too_large"
    http_status = 413


class NotFound(DomainError):
    code = "not_found"
    http_status = 404


class StateConflict(DomainError):
    code = "state_conflict"
    http_status = 409


class DependencyUnavailable(DomainError):
    code = "dependency_unavailable"
    http_status = 503
