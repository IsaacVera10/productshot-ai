"""Reencuadre del modo avanzado: el producto se achica y se apoya en el tercio inferior.

En la foto del usuario el producto suele ocupar el centro y casi todo el cuadro, así que
cambiar solo el fondo lo deja «pegado». Antes de generar se arma un lienzo del mismo tamaño
con el producto más chico y apoyado más abajo, para que el modelo dibuje la superficie, la
sombra de contacto y la escena con profundidad. G8 compara contra este lienzo, no contra
la foto original: el producto conserva sus píxeles, solo reducidos.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

FILL = (128, 128, 128)  # el fondo es editable entero: el relleno solo fija el punto de partida


@dataclass(frozen=True)
class Layout:
    height: float = 0.45  # alto máximo del producto, como fracción del alto del lienzo
    width: float = 0.6  # ancho máximo del producto, como fracción del ancho del lienzo
    bottom: float = 0.85  # dónde queda la base del producto, como fracción del alto


def product_box(mask: Image.Image) -> tuple[int, int, int, int]:
    """Caja del producto en una máscara con blanco = editable y negro = protegido."""
    box = mask.convert("L").point(lambda v: 255 if v < 128 else 0).getbbox()
    if box is None:
        raise ValueError("La máscara no protege ningún píxel.")
    return box


def compose(image: Image.Image, mask: Image.Image, layout: Layout) -> tuple[Image.Image, Image.Image]:
    """Lienzo y máscara del mismo tamaño que la foto, con el producto reencuadrado.

    El producto nunca se agranda (perdería nitidez): si ya es más chico que el objetivo,
    solo se mueve.
    """
    if image.size != mask.size:
        raise ValueError(f"La foto mide {image.size} y la máscara {mask.size}.")
    w, h = image.size
    box = product_box(mask)
    bw, bh = box[2] - box[0], box[3] - box[1]
    scale = min(1.0, layout.height * h / bh, layout.width * w / bw)
    size = (max(1, round(bw * scale)), max(1, round(bh * scale)))
    product = image.convert("RGB").crop(box).resize(size, Image.Resampling.LANCZOS)
    product_mask = mask.convert("L").crop(box).resize(size, Image.Resampling.BILINEAR)

    x = (w - size[0]) // 2
    y = min(h - size[1], max(0, round(layout.bottom * h) - size[1]))
    canvas = Image.new("RGB", (w, h), FILL)
    canvas_mask = Image.new("L", (w, h), 255)
    canvas.paste(product, (x, y))
    canvas_mask.paste(product_mask, (x, y))
    return canvas, canvas_mask
