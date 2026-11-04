"""Validación de la imagen de entrada y preparación del lienzo de trabajo."""

from __future__ import annotations

import hashlib
import io
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from productshot.domain import InvalidInput, PayloadTooLarge

ALLOWED_FORMATS = {"JPEG", "PNG"}


@dataclass(frozen=True)
class PreparedImage:
    original_bytes: bytes
    original_format: str
    original_size: tuple[int, int]
    working: Image.Image  # RGB, lado mayor <= work_max_side, proporción conservada
    working_png: bytes
    sha256: str  # del archivo original


def to_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def prepare_upload(data: bytes, *, max_bytes: int, max_pixels: int, work_max_side: int) -> PreparedImage:
    """Valida tamaño, formato real (no la extensión ni el MIME declarado) y decodificación."""
    if len(data) == 0:
        raise InvalidInput("El archivo está vacío.")
    if len(data) > max_bytes:
        raise PayloadTooLarge(f"La imagen supera el máximo de {max_bytes // (1024 * 1024)} MB.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                fmt = probe.format
                if fmt not in ALLOWED_FORMATS:
                    raise InvalidInput("Formato no admitido: solo JPG o PNG.")
                if probe.width * probe.height > max_pixels:
                    raise InvalidInput("La imagen tiene demasiados píxeles.")
                probe.verify()
            # verify() invalida el objeto: se reabre para decodificar de verdad.
            with Image.open(io.BytesIO(data)) as img:
                img.load()
                img = ImageOps.exif_transpose(img)
                original_size = img.size
                rgb = img.convert("RGB")
    except InvalidInput:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise InvalidInput("El archivo no es una imagen JPG/PNG decodificable.") from exc

    working = rgb.copy()
    working.thumbnail((work_max_side, work_max_side), Image.Resampling.LANCZOS)
    return PreparedImage(
        original_bytes=data,
        original_format=fmt,
        original_size=original_size,
        working=working,
        working_png=to_png(working),
        sha256=hashlib.sha256(data).hexdigest(),
    )
