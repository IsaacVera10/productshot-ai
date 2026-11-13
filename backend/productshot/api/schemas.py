"""Esquemas HTTP (contrato de docs/02_API_Y_MODELO_DE_DATOS.md)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from productshot.domain import (
    REASON_MAX_CHARS,
    REASON_RAW_MAX_CHARS,
    DecisionValue,
    JobStatus,
    Mode,
    SessionStatus,
    Style,
    normalize_reason,
)

API_PREFIX = "/api/v1"


def asset_url(key: str) -> str:
    return f"{API_PREFIX}/assets/{key}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DecisionIn(StrictModel):
    decision: DecisionValue
    # El tope crudo evita cuerpos enormes; el límite de REASON_MAX_CHARS se aplica al texto
    # ya normalizado, que es el que se guarda y se cita.
    reason: str | None = Field(default=None, max_length=REASON_RAW_MAX_CHARS)

    @field_validator("reason")
    @classmethod
    def _normalized_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = normalize_reason(value)
        if value is not None and len(value) > REASON_MAX_CHARS:
            raise ValueError(f"El motivo supera {REASON_MAX_CHARS} caracteres.")
        return value


class NextRoundIn(StrictModel):
    # Texto libre: nunca llega al prompt. Se traduce a ajustes de un vocabulario cerrado (G5).
    adjustment_note: str | None = Field(default=None, max_length=REASON_RAW_MAX_CHARS)

    @field_validator("adjustment_note")
    @classmethod
    def _normalized_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = normalize_reason(value)
        if value is not None and len(value) > REASON_MAX_CHARS:
            raise ValueError(f"La nota supera {REASON_MAX_CHARS} caracteres.")
        return value


class PlanItemOut(BaseModel):
    position: int
    style: Style
    variant: str
    label: str
    adjustments: list[str]
    because: list[dict]


class RoundPlanOut(BaseModel):
    round_index: int
    parent_session_id: str
    rule: str
    note: str | None
    note_adjustments: list[dict]
    ignored: list[dict]
    items: list[PlanItemOut]
    explanation: list[str] = Field(description="Una línea por variación con su porqué, compuesta por Python.")


class DecisionOut(BaseModel):
    variation_id: str
    decision: DecisionValue
    reason: str | None
    decided_at: datetime


class GenerationInfo(BaseModel):
    seed: int
    model_revision: str
    prompt_version: str
    fidelity: dict | None = Field(default=None, description="G8: métricas de la región protegida.")


class WarningOut(BaseModel):
    guardrail: str
    code: str
    message: str


class VariationOut(BaseModel):
    id: str
    position: int
    style: Style
    variant: str
    label: str
    image_url: str
    generation: GenerationInfo
    decision: DecisionOut | None


class MaskOut(BaseModel):
    id: str
    source: str
    url: str
    preview_url: str
    simulated: bool
    warnings: list[WarningOut] = Field(default_factory=list, description="G13: avisos sobre la máscara.")
    metrics: dict | None = None


class JobOut(BaseModel):
    job_id: str
    session_id: str
    status: JobStatus
    backend: str
    error: str | None = None


class SessionOut(BaseModel):
    id: str
    mode: Mode
    status: SessionStatus
    image_id: str
    image_url: str
    created_at: datetime
    updated_at: datetime
    failure_reason: str | None
    simulated: bool = Field(description="True si la sesión usa backends simulados (mock).")
    mask: MaskOut | None
    job: JobOut | None
    variations: list[VariationOut]
    summary_available: bool
    warnings: list[WarningOut] = Field(description="Avisos de los guardrails G12 y G13; no bloquean.")
    photo_quality: dict | None = Field(description="G12: brillo y nitidez de la foto de trabajo.")
    round_index: int
    parent_session_id: str | None
    next_session_id: str | None = Field(description="Ronda siguiente, si ya se creó.")
    round_plan: RoundPlanOut | None


class SegmentOut(BaseModel):
    session_id: str
    status: SessionStatus
    mask: MaskOut


class ReasonRef(BaseModel):
    variation_id: str
    position: int
    style: Style
    decision: DecisionValue
    reason: str


class SummaryOut(BaseModel):
    session_id: str
    accepted_ids: list[str]
    rejected_ids: list[str]
    summary: str
    preferences_supported_by_user_reasons: list[ReasonRef]
    limitations: list[str]
    generated_by: str
    validation_status: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list | None = None


class ErrorOut(BaseModel):
    error: ErrorBody
