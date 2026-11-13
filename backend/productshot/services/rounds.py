"""Planificador de rondas (ADR-007, docs/08_ITERACIONES_Y_GUARDRAILS.md, parte A).

Una ronda nueva es una sesión hija con cinco variaciones derivadas de las decisiones de la
ronda anterior. El plan es DETERMINISTA y auditable: cada variación lleva un `because` que
cita los registros que la produjeron.

Ni los motivos ni la nota de ajuste llegan jamás al prompt como texto libre (G5): se
traducen por reglas a un vocabulario cerrado de modificadores, cada uno con un fragmento
de prompt fijo. En el Hito 3 el agente podrá proponer ajustes, pero solo se aceptarán
valores de este vocabulario.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field

from productshot.domain import (
    CANONICAL_STYLES,
    MAX_ROUNDS,
    VARIATIONS_PER_BATCH,
    DecisionValue,
    Style,
    Variation,
)


# --- vocabulario cerrado ---


@dataclass(frozen=True)
class Adjustment:
    key: str
    label_es: str
    prompt: str  # fragmento fijo y revisado; nunca texto del usuario


_VOCABULARY = (
    Adjustment("brighter", "más luminoso", "brighter well-lit scene, high-key lighting"),
    Adjustment("darker", "más oscuro", "darker moodier scene, low-key lighting"),
    Adjustment("less_saturated", "colores más suaves", "muted soft desaturated colors"),
    Adjustment("more_saturated", "colores más vivos", "vibrant rich saturated colors"),
    Adjustment("warmer", "más cálido", "warmer color temperature, warm tones"),
    Adjustment("cooler", "más frío", "cooler color temperature, cool blue tones"),
    Adjustment("simpler_background", "fondo más simple", "simple uncluttered background"),
    Adjustment("more_props", "más utilería", "a few more tasteful props around the product"),
    Adjustment("fewer_props", "menos utilería", "no props, minimal surroundings"),
    Adjustment("wood_surface", "superficie de madera", "on a wooden surface"),
    Adjustment("stone_surface", "superficie de piedra", "on a stone surface"),
    Adjustment("fabric_surface", "superficie de tela", "on a soft fabric surface"),
    Adjustment("outdoor", "exterior", "outdoor setting"),
    Adjustment("indoor", "interior", "indoor setting"),
)
VOCABULARY: dict[str, Adjustment] = {a.key: a for a in _VOCABULARY}
VOCAB_ORDER: tuple[str, ...] = tuple(a.key for a in _VOCABULARY)

# Ajustes mutuamente excluyentes: dentro de un grupo solo puede quedar uno.
_CONFLICT_GROUPS = (
    frozenset({"brighter", "darker"}),
    frozenset({"less_saturated", "more_saturated"}),
    frozenset({"warmer", "cooler"}),
    frozenset({"more_props", "fewer_props"}),
    frozenset({"more_props", "simpler_background"}),
    frozenset({"wood_surface", "stone_surface", "fabric_surface"}),
    frozenset({"outdoor", "indoor"}),
)

# Variantes que el planificador prueba primero para cada estilo al rellenar huecos.
_STYLE_PREFERENCES: dict[Style, tuple[str, ...]] = {
    Style.STUDIO: ("brighter", "cooler", "stone_surface", "fewer_props", "less_saturated", "warmer"),
    Style.NATURAL: ("brighter", "stone_surface", "wood_surface", "more_saturated", "warmer", "cooler"),
    Style.WARM: ("brighter", "wood_surface", "fabric_surface", "more_saturated", "darker", "fewer_props"),
    Style.LIFESTYLE: ("brighter", "more_props", "wood_surface", "fabric_surface", "warmer", "fewer_props"),
    Style.MINIMAL: ("brighter", "cooler", "warmer", "stone_surface", "fabric_surface", "less_saturated"),
}


def conflicts(a: str, b: str) -> bool:
    return a != b and any(a in g and b in g for g in _CONFLICT_GROUPS)


def variant_key(style: Style, adjustments: tuple[str, ...]) -> str:
    """Identificador estable de una variante: `warm+brighter+wood_surface` (orden del vocabulario)."""
    return "+".join((style.value, *sorted(adjustments, key=VOCAB_ORDER.index)))


def parse_variant(variant: str) -> tuple[Style, tuple[str, ...]]:
    style, *adjustments = variant.split("+")
    unknown = [a for a in adjustments if a not in VOCABULARY]
    if unknown:
        raise ValueError(f"Ajustes fuera del vocabulario: {unknown}")
    return Style(style), tuple(adjustments)


def adjustment_labels(adjustments: tuple[str, ...]) -> list[str]:
    return [VOCABULARY[a].label_es for a in adjustments]


def prompt_fragments(adjustments: tuple[str, ...]) -> list[str]:
    return [VOCABULARY[a].prompt for a in adjustments]


# --- extracción por reglas (Hito 1 y 2) ---


@dataclass(frozen=True)
class Extracted:
    adjustment: str
    matched: str  # fragmento del texto que activó la regla (sin acentos, en minúsculas)


# Orden importa: las frases más específicas van primero y «consumen» el texto que
# reconocen, para que «más oscuro» no active también la regla de «oscuro».
_RULES: tuple[tuple[str, str], ...] = (
    (r"mas oscur\w*|demasiad[oa]s? (?:claro|luz|brillo)\w*|muy clar\w*|quemad\w*|sobreexpuest\w*", "darker"),
    (r"mas clar\w*|mas luz|mas luminos\w*|oscur\w*|poca luz|falta (?:de )?luz|sombri\w*|lugubre\w*", "brighter"),
    (r"mas color\w*|sin vida|desaturad\w*|palid\w*|lavad\w*|apagad\w*", "more_saturated"),
    (r"saturad\w*|chillon\w*|colores (?:muy )?fuertes|muy intens\w*|menos color\w*", "less_saturated"),
    (r"(?:muy|demasiado) calid\w*|naranj\w*|amarillent\w*|mas fresc\w*|mas fri\w*", "cooler"),
    (r"mas calid\w*|fri[oa]s?\b|frialdad|azulad\w*", "warmer"),
    (r"(?:muchos|demasiados) objetos|menos objetos|sobran? (?:cosas|objetos)|menos utileria", "fewer_props"),
    (r"recargad\w*|desorden\w*|distrae\w*|muchas cosas|fondo (?:mas )?simple|mas simple|mas limpi\w*", "simpler_background"),
    (r"vaci[oa]s?\b|le falta algo|aburrid\w*|soso|mas objetos|mas decoracion|mas utileria", "more_props"),
    (r"madera", "wood_surface"),
    (r"piedra|marmol|granito", "stone_surface"),
    (r"tela|mantel|lino|textil", "fabric_surface"),
    (r"exterior|afuera|aire libre|jardin|naturaleza|al sol\b", "outdoor"),
    (r"interior\b|adentro|dentro de casa|cocina|sala\b", "indoor"),
)
_COMPILED = tuple((re.compile(p), adj) for p, adj in _RULES)
# Ajustes que expresan un gusto («madera», «exterior»): una negación cercana los anula.
_NEGATABLE = frozenset({"wood_surface", "stone_surface", "fabric_surface", "outdoor", "indoor", "more_props"})
_NEGATION = re.compile(r"\b(?:no|sin|nada de|odio|evita\w*|ni)\b[^.,;]{0,24}$")


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def extract_adjustments(text: str | None) -> list[Extracted]:
    """Traduce un texto libre a ajustes del vocabulario. Nunca devuelve otra cosa."""
    if not text:
        return []
    folded = _fold(text)
    found: dict[str, Extracted] = {}
    for pattern, adj in _COMPILED:
        for m in pattern.finditer(folded):
            if adj in _NEGATABLE and _NEGATION.search(folded[: m.start()]):
                continue
            found.setdefault(adj, Extracted(adj, m.group(0)))
            # Consumir el fragmento para que una regla posterior no lo reinterprete.
            folded = folded[: m.start()] + " " * (m.end() - m.start()) + folded[m.end():]
    return [found[a] for a in VOCAB_ORDER if a in found]


# --- plan ---


@dataclass(frozen=True)
class PlannedVariation:
    position: int
    style: Style
    adjustments: tuple[str, ...]
    seed: int
    because: tuple[dict, ...]

    @property
    def variant(self) -> str:
        return variant_key(self.style, self.adjustments)


@dataclass(frozen=True)
class RoundPlan:
    round_index: int
    parent_session_id: str
    rule: str  # "accepted" | "none_accepted"
    note: str | None
    note_adjustments: tuple[Extracted, ...]
    items: tuple[PlannedVariation, ...]
    ignored: tuple[dict, ...] = field(default_factory=tuple)

    def to_json(self) -> dict:
        return {
            "round_index": self.round_index,
            "parent_session_id": self.parent_session_id,
            "rule": self.rule,
            "note": self.note,
            "note_adjustments": [{"adjustment": e.adjustment, "matched": e.matched} for e in self.note_adjustments],
            "ignored": list(self.ignored),
            "items": [
                {
                    "position": it.position,
                    "style": it.style.value,
                    "variant": it.variant,
                    "adjustments": list(it.adjustments),
                    "seed": it.seed,
                    "because": list(it.because),
                }
                for it in self.items
            ],
        }


def _merge(
    layers: list[tuple[str, list[tuple[str, dict]]]], ignored: list[dict]
) -> tuple[tuple[str, ...], list[dict]]:
    """Combina capas de ajustes de menor a mayor prioridad.

    Una capa posterior reemplaza a la anterior dentro de un grupo en conflicto (la nota
    manda sobre el motivo, y el motivo sobre lo heredado). Dos ajustes en conflicto dentro
    de la MISMA capa se anulan entre sí y quedan registrados en `ignored`.
    """
    chosen: dict[str, dict] = {}
    for layer_name, entries in layers:
        layer_keys = [k for k, _ in entries]
        for key, because in entries:
            if any(conflicts(key, other) for other in layer_keys):
                ignored.append({"adjustment": key, "source": layer_name, "why": "conflict"})
                continue
            for existing in [k for k in chosen if conflicts(k, key)]:
                ignored.append({"adjustment": existing, "source": "overridden", "why": f"replaced_by_{layer_name}"})
                del chosen[existing]
            chosen.setdefault(key, because)
    keys = tuple(k for k in VOCAB_ORDER if k in chosen)
    return keys, [chosen[k] for k in keys if chosen[k]]


def _reason_entries(v: Variation, parent_round: int) -> list[tuple[str, dict]]:
    reason = v.decision.reason if v.decision else None
    return [
        (e.adjustment, {"round": parent_round, "variation_position": v.position, "adjustment": e.adjustment,
                        "source": "reason", "matched": e.matched})
        for e in extract_adjustments(reason)
    ]


def _fillers(style: Style, base: tuple[str, ...], used: set[str]) -> list[str]:
    """Candidatos para una variante extra del estilo: primero sus preferidos, luego el resto."""
    order = list(_STYLE_PREFERENCES[style]) + [a for a in VOCAB_ORDER if a not in _STYLE_PREFERENCES[style]]
    return [
        a for a in order
        if a not in base and not any(conflicts(a, b) for b in base) and variant_key(style, (*base, a)) not in used
    ]


def _first_filler(style: Style, base: tuple[str, ...], used: set[str]) -> str:
    options = _fillers(style, base, used)
    if not options:  # con 14 ajustes y a lo sumo 5 variantes no debería ocurrir
        raise ValueError(f"No quedan variantes distintas para el estilo {style.value}.")
    return options[0]


def plan_next_round(
    *,
    parent_session_id: str,
    parent_round: int,
    variations: list[Variation],
    note: str | None,
    rng: random.Random,
) -> RoundPlan:
    """Deriva las cinco variaciones de la ronda siguiente a partir de la ronda resumida."""
    if parent_round >= MAX_ROUNDS:
        raise ValueError("Se alcanzó el máximo de rondas.")
    if len(variations) != VARIATIONS_PER_BATCH or any(v.decision is None for v in variations):
        raise ValueError("La ronda anterior no tiene cinco decisiones.")

    ignored: list[dict] = []
    note_found = tuple(extract_adjustments(note))
    note_layer = ("note", [(e.adjustment, {"adjustment": e.adjustment, "source": "note", "matched": e.matched})
                           for e in note_found])
    seeds = [rng.randrange(0, 2**31 - 1) for _ in range(VARIATIONS_PER_BATCH)]

    accepted = sorted(
        (v for v in variations if v.decision.value == DecisionValue.ACCEPTED),
        key=lambda v: (v.decision.decided_at, v.position),
    )
    planned: list[tuple[Style, tuple[str, ...], list[dict]]] = []

    if accepted:
        rule = "accepted"
        bases = []
        for v in accepted:
            style, inherited = parse_variant(v.variant or v.style.value)
            layers = [
                ("inherited", [(a, {}) for a in inherited]),
                ("reason", _reason_entries(v, parent_round)),
                note_layer,
            ]
            keys, adj_because = _merge(layers, ignored)
            ref = {"round": parent_round, "variation_position": v.position, "decision": "accepted"}
            bases.append((style, keys, [ref, *adj_because]))
            planned.append((style, keys, [ref, *adj_because]))
        # Los motivos de variaciones RECHAZADAS no cambian el plan (su estilo no continúa),
        # pero se registran para que la explicación lo diga en vez de ignorarlos en silencio.
        for v in variations:
            if v.decision.value == DecisionValue.REJECTED:
                for adj, _because in _reason_entries(v, parent_round):
                    ignored.append({"adjustment": adj, "source": "reason", "variation_position": v.position,
                                    "why": "rejected_style"})
        used = {variant_key(s, k) for s, k, _ in planned}
        # Huecos: variantes de los estilos aceptados, repartidas en orden de aceptación.
        i = 0
        while len(planned) < VARIATIONS_PER_BATCH:
            style, base, because = bases[i % len(bases)]
            extra = _first_filler(style, base, used)
            keys = tuple(k for k in VOCAB_ORDER if k in {*base, extra})
            used.add(variant_key(style, keys))
            planned.append((style, keys, [*because, {"adjustment": extra, "source": "planner", "why": "variant"}]))
            i += 1
    else:
        rule = "none_accepted"
        # Todos rechazados: los cinco estilos canónicos con una variante alternativa, y los
        # ajustes de TODOS los motivos aplicados a todas.
        all_reasons = [e for v in variations for e in _reason_entries(v, parent_round)]
        keys, adj_because = _merge([("reason", all_reasons), note_layer], ignored)
        used: set[str] = set()
        for style in CANONICAL_STYLES:
            extra = _first_filler(style, keys, used)
            final = tuple(k for k in VOCAB_ORDER if k in {*keys, extra})
            used.add(variant_key(style, final))
            ref = {"round": parent_round, "decision": "all_rejected"}
            planned.append((style, final, [ref, *adj_because, {"adjustment": extra, "source": "planner",
                                                               "why": "alternative"}]))

    items = tuple(
        PlannedVariation(position=i + 1, style=s, adjustments=k, seed=seeds[i], because=tuple(b))
        for i, (s, k, b) in enumerate(planned)
    )
    assert len({it.variant for it in items}) == VARIATIONS_PER_BATCH, "variantes repetidas"
    return RoundPlan(
        round_index=parent_round + 1,
        parent_session_id=parent_session_id,
        rule=rule,
        note=note,
        note_adjustments=note_found,
        items=items,
        ignored=tuple(_dedupe(ignored)),
    )


def _dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = tuple(sorted(r.items()))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# --- explicación en texto (la compone Python desde el plan; no la escribe un LLM) ---


def style_label(style: Style) -> str:
    from productshot.services.presets import PRESETS

    return PRESETS[style].label_es


def variant_label(variant: str) -> str:
    style, adjustments = parse_variant(variant)
    return " + ".join([style_label(style), *adjustment_labels(adjustments)])


def explain_plan(plan: dict) -> list[str]:
    """Una línea por variación con el porqué, citando la ronda anterior."""
    r = plan["round_index"] - 1
    lines = [f"Ronda {plan['round_index']} de {MAX_ROUNDS}, derivada de las decisiones de la ronda {r}."]
    if plan["rule"] == "none_accepted":
        lines.append(f"En la ronda {r} no se aceptó ninguna variación: se prueban los cinco estilos con una variante distinta.")
    if plan.get("note_adjustments"):
        labels = ", ".join(VOCABULARY[e["adjustment"]].label_es for e in plan["note_adjustments"])
        lines.append(f"Ajustes entendidos en la nota: {labels}.")
    elif plan.get("note"):
        lines.append("La nota no contiene ajustes reconocibles; no cambia el plan.")
    for it in plan["items"]:
        parts = []
        for b in it["because"]:
            if b.get("decision") == "accepted":
                parts.append(f"parte de la variación {b['variation_position']}, aceptada en la ronda {b['round']}")
            elif b.get("source") == "reason":
                parts.append(f"«{VOCABULARY[b['adjustment']].label_es}» por el motivo de la variación {b['variation_position']}")
            elif b.get("source") == "note":
                parts.append(f"«{VOCABULARY[b['adjustment']].label_es}» por la nota")
            elif b.get("source") == "planner":
                verb = "variante nueva" if b["why"] == "variant" else "variante alternativa"
                parts.append(f"{verb}: «{VOCABULARY[b['adjustment']].label_es}»")
        lines.append(f"- {it['position']}. {variant_label(it['variant'])}: {'; '.join(parts) or 'repite el estilo'}.")
    unused = [i for i in plan.get("ignored", []) if i.get("why") == "rejected_style"]
    if unused:
        parts = ", ".join(
            f"«{VOCABULARY[i['adjustment']].label_es}» (variación {i['variation_position']})" for i in unused
        )
        lines.append(f"No se aplicaron ajustes de motivos de variaciones rechazadas, porque su estilo no continúa: {parts}.")
    conflicts_ = [i for i in plan.get("ignored", []) if i.get("why") == "conflict"]
    if conflicts_:
        labels = ", ".join(sorted({VOCABULARY[i["adjustment"]].label_es for i in conflicts_}))
        lines.append(f"Ajustes contradictorios que se descartaron: {labels}.")
    return lines
