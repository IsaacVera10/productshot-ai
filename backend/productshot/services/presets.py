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
