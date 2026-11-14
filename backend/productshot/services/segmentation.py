"""Segmentación del producto.

Semántica de la máscara (docs/00_ALCANCE_Y_REQUISITOS.md): BLANCO (255) = área editable
(fondo), NEGRO (0) = región protegida (producto). En el Hito 2 se añadirá `rembg`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image, ImageDraw

from productshot.services.labels import stamp


@dataclass(frozen=True)
class MaskResult:
    mask: Image.Image  # modo "L", mismas dimensiones que el lienzo de trabajo
    simulated: bool


class SegmentationUnavailable(Exception):
    """El segmentador no respondió o respondió mal. El mensaje se muestra a la persona."""


class SegmentationService(Protocol):
    name: str
    simulated: bool

    def segment(self, image: Image.Image) -> MaskResult: ...


class MockSegmentationService:
    """Máscara de demostración: una elipse central protegida. No detecta el producto."""

    name = "mock"
    simulated = True

    def segment(self, image: Image.Image) -> MaskResult:
        w, h = image.size
        mask = Image.new("L", (w, h), 255)
        draw = ImageDraw.Draw(mask)
        mx, my = int(w * 0.22), int(h * 0.15)
        draw.ellipse([mx, my, w - mx, h - my], fill=0)
        return MaskResult(mask=mask, simulated=True)


def render_mask_preview(image: Image.Image, mask: Image.Image, simulated: bool) -> Image.Image:
    """Vista previa: el área editable (fondo) se tiñe; el producto protegido queda intacto."""
    tint = Image.new("RGB", image.size, (40, 120, 230))
    tinted = Image.blend(image.convert("RGB"), tint, 0.55)
    preview = Image.composite(tinted, image.convert("RGB"), mask)
    if simulated:
        return stamp(preview, bottom="Máscara simulada: azul = fondo editable")
    return preview
