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


class DecisionOut(BaseModel):
    variation_id: str
    decision: DecisionValue
    reason: str | None
    decided_at: datetime


class GenerationInfo(BaseModel):
    seed: int
    model_revision: str
    prompt_version: str


class VariationOut(BaseModel):
    id: str
    position: int
    style: Style
    image_url: str
    generation: GenerationInfo
    decision: DecisionOut | None


class MaskOut(BaseModel):
    id: str
    source: str
    url: str
    preview_url: str
    simulated: bool


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
