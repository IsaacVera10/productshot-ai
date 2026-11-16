"""Prompts del modo intermedio: escena según el estilo y la categoría del producto.

Son textos PROPIOS del modo intermedio. Los de `presets.py` no cambian, porque el LoRA del
modo avanzado se entrenó con esas frases exactas.

Historia (experimentos con fotos reales del 2026-11-15, `deploy/modal/`):
- Con los prompts genéricos de los presets aparecían lugares y utilería sin relación con el
  producto (pan junto a un hub USB; focos y trípodes en "estudio"). `scene-v1` separa estilo
  (luz, superficie, ánimo) de categoría (lugar y utilería de `lifestyle`).
- Con el encuadre de catálogo hay más fondo que llenar y `scene-v1` dejaba objetos sueltos
  (libro, lápiz, rocas). FLUX, que no usa negative prompt, convertía las negaciones ("no
  props") en objetos. `scene-v2` describe en POSITIVO una superficie limpia y espacio vacío,
  y deja la utilería de `lifestyle` lejos y desenfocada.

Ningún texto del usuario entra aquí: la categoría sale de una lista cerrada (G5) y los
ajustes de las rondas usan los fragmentos fijos de `rounds.prompt_fragments`.
"""

from __future__ import annotations

from productshot.domain import Category, Style
from productshot.services.rounds import prompt_fragments

SCENE_PROMPT_VERSION = "scene-v2"

SCENE_STYLES: dict[str, dict[Style, str]] = {
    "scene-v1": {
        Style.STUDIO: "professional product photography, seamless light grey paper backdrop, even softbox lighting, "
                      "soft shadow under the product, clean catalog shot",
        Style.NATURAL: "product photo outdoors on a natural stone surface, soft daylight, green foliage softly "
                       "blurred in the background",
        Style.WARM: "product photo on a warm wooden surface, golden hour sunlight, cozy amber tones, soft bokeh "
                    "background",
        Style.LIFESTYLE: "lifestyle product photo on a {setting} in a bright modern home, {props} softly blurred in "
                         "the background, shallow depth of field",
        Style.MINIMAL: "minimalist product photo, plain off-white background, large negative space, no props, "
                       "single soft shadow",
    },
    "scene-v2": {
        Style.STUDIO: "professional e-commerce product photo, the product standing alone on a seamless light grey "
                      "paper backdrop, soft even studio light, soft contact shadow, clean empty background",
        Style.NATURAL: "product photo outdoors, the product standing alone on a flat natural stone slab, soft "
                       "daylight, lush green garden far in the background and out of focus",
        Style.WARM: "product photo, the product standing alone on a clean wooden table, warm golden hour sunlight, "
                    "amber tones, soft bokeh lights far in the background",
        Style.LIFESTYLE: "lifestyle product photo, the product standing alone on a clean {setting} in a bright modern "
                         "home, {props} far in the background and out of focus, shallow depth of field",
        Style.MINIMAL: "minimalist product photo, the product standing alone on a plain off-white surface against a "
                       "plain off-white wall, large empty space around it, single soft shadow",
    },
}
# Compatibilidad: la versión vigente.
SCENE_STYLE = SCENE_STYLES[SCENE_PROMPT_VERSION]

# (lugar, utilería) de la escena lifestyle por categoría.
CATEGORY_SCENES: dict[Category, tuple[str, str]] = {
    Category.TECNOLOGIA: ("home office desk", "a laptop and a notebook"),
    Category.HOGAR: ("living room side table", "a sofa and a small plant"),
    Category.COCINA: ("kitchen breakfast table", "a coffee pot and a small plant"),
    Category.JUGUETES: ("bedroom desk", "a bookshelf and a small plant"),
    Category.OTRO: ("wooden table", "a small plant"),
}

# Solo lo usa SDXL (RealVisXL); FLUX no tiene negative prompt.
SCENE_NEGATIVE = (
    "text, watermark, logo, letters, people, hands, faces, duplicate product, extra objects on product, "
    "light stands, softbox, tripod, studio equipment, camera, food, fruit, bread, "
    "clutter, scattered small objects, debris, rocks, cables, papers, "
    "blurry, low quality, distorted, nsfw"
)


def build_scene_prompt(style: Style, category: Category | None, adjustments: tuple[str, ...] = (),
                       version: str = SCENE_PROMPT_VERSION) -> str:
    setting, props = CATEGORY_SCENES[category or Category.OTRO]
    base = SCENE_STYLES[version][style].format(setting=setting, props=props)
    return ", ".join([base, *prompt_fragments(adjustments)])


# --- Editor por instrucciones (modo intermedio desde el 2026-11-16) ---
#
# Receta "A" de la ronda 4 (`deploy/modal/edit_lab4.py`): escenas escritas a mano por categoría
# y estilo. Estudio y minimalista no tienen contexto (fondo liso y trabajo de luz). Ganó frente
# al "director de arte" con un modelo de visión porque es predecible: el director a veces
# dejaba el fondo original, encendía la vela o proponía escenas raras.

EDIT_PROMPT_VERSION = "edit-v1"

EDIT_FIXED: dict[Style, str] = {
    Style.STUDIO: "Studio shot on a seamless light grey paper backdrop, soft diffused softbox light from the left, "
                  "gentle contact shadow, nothing else in the frame",
    Style.MINIMAL: "Minimalist shot on a plain matte off-white surface against a plain off-white wall, generous "
                   "empty space around the product, one soft natural shadow",
}
EDIT_CURATED: dict[Category, dict[Style, str]] = {
    Category.TECNOLOGIA: {
        Style.NATURAL: "On a light oak desk beside a large window, soft morning daylight, a small succulent far in "
                       "the background, clean and airy",
        Style.WARM: "On a dark walnut desk in a cozy evening workspace, warm desk-lamp glow, soft amber bokeh from "
                    "string lights far behind",
        Style.LIFESTYLE: "On a tidy minimalist desk setup, a closed laptop and a keyboard softly out of focus behind "
                         "it, bright natural light",
    },
    Category.HOGAR: {
        Style.NATURAL: "On a natural linen cloth beside a vase of dried eucalyptus and pampas grass, soft diffused "
                       "window daylight, calm neutral tones",
        Style.WARM: "On a rustic wooden tray next to a chunky wool blanket, a few lit tea lights softly blurred "
                    "behind, warm cozy evening glow",
        Style.LIFESTYLE: "On a shelf in a modern spa-like bathroom, rolled white towels and a small plant softly out "
                         "of focus behind",
    },
    Category.COCINA: {
        Style.NATURAL: "On a wooden breakfast table by a sunny window, a small plate with a croissant and a few "
                       "coffee beans softly out of focus behind",
        Style.WARM: "On a rustic wooden kitchen counter in warm early-morning sunlight, a folded linen napkin and "
                    "coffee beans nearby, amber tones",
        Style.LIFESTYLE: "On a bright Scandinavian kitchen table, a French press and a small plant softly out of "
                         "focus in the background",
    },
    Category.JUGUETES: {
        Style.NATURAL: "On a white shelf next to a window in soft daylight, a few colorful books softly blurred "
                       "behind",
        Style.WARM: "On a wooden bedroom shelf, warm fairy lights softly glowing far in the background",
        Style.LIFESTYLE: "On a tidy collector's display shelf with soft LED accent lighting, other shelves softly "
                         "out of focus behind",
    },
    Category.OTRO: {
        Style.NATURAL: "On a light wooden table beside a window, soft natural daylight, a small plant far in the "
                       "background",
        Style.WARM: "On a warm wooden surface in golden hour sunlight, cozy amber tones, soft bokeh far behind",
        Style.LIFESTYLE: "On a tidy table in a bright modern home, a small plant softly out of focus behind",
    },
}
# Sustantivo genérico por categoría: el vendedor no escribe el nombre del producto (G5).
EDIT_NOUNS: dict[Category, str] = {
    Category.TECNOLOGIA: "tech product",
    Category.HOGAR: "home decor product",
    Category.COCINA: "kitchenware product",
    Category.JUGUETES: "toy or collectible",
    Category.OTRO: "product",
}
EDIT_REPLACE = "Replace the entire original background and surface; nothing from the original surroundings remains."
EDIT_KEEP = (
    "Keep the product identical to the original photo: same shape, proportions, colors, logo and every printed word "
    "and number exactly as shown. Keep the same orientation and viewing angle of the product; do not rotate or "
    "mirror it and do not add or remove parts. The product is the only hero object; no people, no hands. "
    "Photorealistic, sharp focus on the product, high-end commercial product photography."
)


def build_edit_prompt(style: Style, category: Category | None, adjustments: tuple[str, ...] = ()) -> str:
    """Instrucción completa para el editor. Los ajustes de ronda son fragmentos fijos (G5)."""
    cat = category or Category.OTRO
    scene = EDIT_FIXED.get(style) or EDIT_CURATED[cat][style]
    extra = "".join(f" {f}." for f in prompt_fragments(adjustments))
    return (f"Turn this into a professional e-commerce photo to sell this exact {EDIT_NOUNS[cat]}. {scene}.{extra} "
            f"{EDIT_REPLACE} {EDIT_KEEP}")
