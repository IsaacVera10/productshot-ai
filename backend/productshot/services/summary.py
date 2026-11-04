"""Resumen de las cinco decisiones.

Diseño (ADR-005): el agente produce un resumen ESTRUCTURADO; Python verifica cobertura,
decisiones y citas contra la base de datos y renderiza la enumeración. En el Hito 1 no
hay LLM: `DeterministicSummaryService` compone el resumen directamente desde el registro
y se marca `generated_by = "deterministic_fallback"` (no cuenta como agente).

El validador y el renderizador son los mismos que usará el agente Gemini del Hito 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from productshot.domain import VARIATIONS_PER_BATCH, DecisionValue, Style, Variation
from productshot.services.presets import PRESETS

GENERATED_BY_LLM = "llm_validated"
GENERATED_BY_HYBRID = "hybrid"
GENERATED_BY_FALLBACK = "deterministic_fallback"

STATUS_VERIFIED = "structure_verified"
STATUS_NOT_VERIFIED = "not_verified"


@dataclass(frozen=True)
class SummaryItem:
    variation_id: str
    position: int
    style: Style
    decision: DecisionValue
    reason_quote: str | None  # cita LITERAL del motivo registrado, o None


@dataclass(frozen=True)
class StructuredSummary:
    """Salida estructurada que debe producir cualquier generador de resumen (LLM o Python)."""

    items: tuple[SummaryItem, ...]
    narrative: str | None = None  # prosa libre opcional; su revisión automática es parcial
    style_observations: tuple[str, ...] = ()  # asociaciones descriptivas, nunca causales


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SummaryResult:
    payload: dict
    text: str
    generated_by: str
    validation_status: str


def validate_structured(structured: StructuredSummary, variations: list[Variation]) -> ValidationResult:
    """Compara la salida estructurada con el registro. Cualquier discrepancia la invalida."""
    errors: list[str] = []
    by_id = {v.id: v for v in variations}
    if len(variations) != VARIATIONS_PER_BATCH or any(v.decision is None for v in variations):
        errors.append("El registro no tiene cinco decisiones.")

    seen: set[str] = set()
    for item in structured.items:
        v = by_id.get(item.variation_id)
        if v is None:
            errors.append(f"Referencia a una variación inexistente: {item.variation_id}.")
            continue
        if item.variation_id in seen:
            errors.append(f"Variación {v.position} citada más de una vez.")
        seen.add(item.variation_id)
        if item.position != v.position or item.style != v.style:
            errors.append(f"Posición o estilo incorrectos para la variación {v.position}.")
        if v.decision is None or item.decision != v.decision.value:
            errors.append(f"La decisión de la variación {v.position} no coincide con el registro.")
        if item.reason_quote is not None:
            registered = v.decision.reason if v.decision else None
            if registered is None or item.reason_quote != registered:
                errors.append(f"El motivo citado para la variación {v.position} no consta en su registro.")

    missing = set(by_id) - seen
    if missing:
        positions = sorted(by_id[m].position for m in missing)
        errors.append(f"Faltan variaciones en el resumen: {positions}.")
    return ValidationResult(ok=not errors, errors=errors)


def _join_es(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " y " + parts[-1]


def _label(v: Variation) -> str:
    return f"{v.position} ({PRESETS[v.style].label_es.lower()})"


def render_summary(
    session_id: str,
    variations: list[Variation],
    *,
    narrative: str | None,
    style_observations: tuple[str, ...],
    generated_by: str,
    validation_status: str,
    extra_limitations: list[str] | None = None,
) -> SummaryResult:
    """Construye la respuesta final. Las listas y la enumeración salen SIEMPRE de la base de datos."""
    accepted = [v for v in variations if v.decision and v.decision.value == DecisionValue.ACCEPTED]
    rejected = [v for v in variations if v.decision and v.decision.value == DecisionValue.REJECTED]

    lines = []
    if accepted:
        lines.append(f"Se aceptaron {len(accepted)} de 5 variaciones: {_join_es([_label(v) for v in accepted])}.")
    else:
        lines.append("No se aceptó ninguna de las 5 variaciones.")
    if rejected:
        lines.append(f"Se rechazaron {len(rejected)} de 5: {_join_es([_label(v) for v in rejected])}.")
    else:
        lines.append("No se rechazó ninguna variación.")

    with_reason = [v for v in variations if v.decision and v.decision.reason]
    reasons = []
    if with_reason:
        lines.append("")
        lines.append("Motivos escritos por la persona (citados literalmente):")
        for v in with_reason:
            verb = "aceptada" if v.decision.value == DecisionValue.ACCEPTED else "rechazada"
            lines.append(f"- Variación {_label(v)}, {verb}: «{v.decision.reason}»")
            reasons.append(
                {
                    "variation_id": v.id,
                    "position": v.position,
                    "style": v.style.value,
                    "decision": v.decision.value,
                    "reason": v.decision.reason,
                }
            )
    if style_observations:
        lines.append("")
        lines.extend(f"Observación: {o}" for o in style_observations)
    if narrative:
        lines.append("")
        lines.append(narrative)

    limitations = list(extra_limitations or [])
    without_reason = len(variations) - len(with_reason)
    if without_reason:
        limitations.append(
            f"{without_reason} de 5 decisiones no tienen motivo escrito; no se infieren preferencias para ellas."
        )

    text = "\n".join(lines)
    payload = {
        "session_id": session_id,
        "accepted_ids": [v.id for v in accepted],
        "rejected_ids": [v.id for v in rejected],
        "summary": text,
        "preferences_supported_by_user_reasons": reasons,
        "limitations": limitations,
        "generated_by": generated_by,
        "validation_status": validation_status,
    }
    return SummaryResult(payload=payload, text=text, generated_by=generated_by, validation_status=validation_status)


class SummaryService(Protocol):
    name: str

    def summarize(self, session_id: str, variations: list[Variation]) -> SummaryResult: ...


class DeterministicSummaryService:
    """Resumen sin LLM. Sirve para el Hito 1 y como fallback explícito del agente."""

    name = "deterministic"

    def summarize(self, session_id: str, variations: list[Variation]) -> SummaryResult:
        structured = StructuredSummary(
            items=tuple(
                SummaryItem(
                    variation_id=v.id,
                    position=v.position,
                    style=v.style,
                    decision=v.decision.value,
                    reason_quote=v.decision.reason,
                )
                for v in variations
                if v.decision is not None
            )
        )
        check = validate_structured(structured, variations)
        return render_summary(
            session_id,
            variations,
            narrative=None,
            style_observations=(),
            generated_by=GENERATED_BY_FALLBACK,
            validation_status=STATUS_VERIFIED if check.ok else STATUS_NOT_VERIFIED,
            extra_limitations=["Resumen compuesto por Python sin LLM (Hito 1); no es la salida de un agente."],
        )
