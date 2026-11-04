"""Rótulos visibles para resultados simulados: un mock nunca debe parecer una imagen IA real."""

from __future__ import annotations

import unicodedata

from PIL import Image, ImageDraw, ImageFont

SIMULATION_TEXT = "SIMULACION (MOCK) - NO GENERADO POR IA"


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _ascii(text: str) -> str:
    """La fuente por defecto de Pillow no tiene tildes ni rayas: se dibuja en ASCII."""
    text = text.replace("—", "-").replace("·", "|")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def stamp(image: Image.Image, top: str = SIMULATION_TEXT, bottom: str | None = None) -> Image.Image:
    """Devuelve una copia con una franja superior (y opcionalmente inferior) de texto."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    size = max(12, w // 32)
    font = _font(size)
    band = int(size * 1.9)

    draw.rectangle([0, 0, w, band], fill=(180, 20, 40, 230))
    draw.text((w / 2, band / 2), _ascii(top), font=font, fill="white", anchor="mm")
    if bottom:
        draw.rectangle([0, h - band, w, h], fill=(20, 20, 20, 200))
        draw.text((w / 2, h - band / 2), _ascii(bottom), font=font, fill="white", anchor="mm")
    return img
