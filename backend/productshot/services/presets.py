"""Presets de los cinco estilos.

Los prompts describen el ENTORNO, no el producto: los píxeles del producto se preservan
con la máscara. Ambos modos usan los mismos prompts para que la comparación aísle el
efecto del LoRA (ver docs/06_EVALUACION_Y_ETICA.md). En el Hito 1 el backend mock solo
usa la paleta; los prompts quedan versionados para el Hito 2 y todavía no se han probado.
"""

from __future__ import annotations

from dataclasses import dataclass

from productshot.domain import Style

PROMPT_VERSION = "v1"

NEGATIVE_PROMPT = (
    "text, watermark, logo, letters, people, hands, faces, duplicate product, extra objects on product, "
    "blurry, low quality, distorted, nsfw"
)


@dataclass(frozen=True)
class StylePreset:
    style: Style
    label_es: str
    prompt: str
    # Paleta del mock: (color superior, color inferior) del degradado de fondo.
    mock_colors: tuple[tuple[int, int, int], tuple[int, int, int]]


PRESETS: dict[Style, StylePreset] = {
    Style.STUDIO: StylePreset(
        Style.STUDIO,
        "Estudio",
        "professional product photography, seamless light grey studio backdrop, softbox lighting, "
        "subtle soft shadow under the product, clean commercial catalog shot",
        ((236, 236, 240), (178, 180, 188)),
    ),
    Style.NATURAL: StylePreset(
        Style.NATURAL,
        "Natural",
        "product on a natural stone surface outdoors, soft daylight, green leaves softly blurred in the "
        "background, fresh organic commercial photo",
        ((198, 222, 190), (120, 150, 104)),
    ),
    Style.WARM: StylePreset(
        Style.WARM,
        "Cálido",
        "product on a warm wooden surface, golden hour ambient light, cozy amber tones, soft bokeh "
        "background, inviting commercial photo",
        ((250, 214, 160), (196, 120, 62)),
    ),
    Style.LIFESTYLE: StylePreset(
        Style.LIFESTYLE,
        "Lifestyle",
        "product on a kitchen table in a bright modern home, everyday lifestyle scene, a few tasteful props "
        "in the background, shallow depth of field",
        ((214, 200, 186), (150, 112, 86)),
    ),
    Style.MINIMAL: StylePreset(
        Style.MINIMAL,
        "Minimalista",
        "minimalist product photo, flat solid off-white background, large negative space, no props, "
        "single soft shadow, clean geometric composition",
        ((250, 249, 246), (250, 249, 246)),
    ),
}


# v2: prompts del modo avanzado escritos como los pediría un fotógrafo de producto (lente, luz
# principal, sombra de contacto y acabado). v1 queda para reproducir lo ya entrenado y evaluado.
PROMPTS: dict[str, dict[Style, str]] = {
    "v1": {style: preset.prompt for style, preset in PRESETS.items()},
    "v2": {
        Style.STUDIO: (
            "professional studio product photograph, seamless light grey paper sweep backdrop, large diffused "
            "key light from the left with soft fill, gentle contact shadow under the product, 100mm lens at f/8, "
            "crisp focus, high-end e-commerce catalog image"
        ),
        Style.NATURAL: (
            "professional product photograph outdoors, product resting on a flat natural stone slab, soft "
            "morning daylight, lush green foliage softly out of focus behind, 85mm lens at f/2.8, natural "
            "contact shadow, fresh organic brand campaign"
        ),
        Style.WARM: (
            "professional product photograph, product on a rustic oak tabletop, warm golden hour sunlight from "
            "the side, cozy amber tones, creamy bokeh lights in the background, 85mm lens at f/2, soft "
            "realistic shadow, premium advertising image"
        ),
        Style.LIFESTYLE: (
            "professional lifestyle product photograph, product on a kitchen countertop in a bright modern "
            "home, natural window light, a few tasteful props softly out of focus, 50mm lens at f/2.2, "
            "realistic scale and contact shadow, editorial magazine look"
        ),
        Style.MINIMAL: (
            "professional minimalist product photograph, product on a seamless off-white surface, large "
            "negative space, single soft directional shadow, no props, 100mm lens at f/8, clean geometric "
            "composition, high-end brand aesthetic"
        ),
    },
}
NEGATIVE_PROMPTS = {
    "v1": NEGATIVE_PROMPT,
    "v2": NEGATIVE_PROMPT + (
        ", tripod, light stand, softbox, umbrella light, studio equipment, camera, cables, floating object, "
        "extra parts attached to the product, cluttered, oversaturated, cartoon, illustration, cgi render"
    ),
}
