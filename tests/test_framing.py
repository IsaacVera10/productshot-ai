"""Encuadre de catálogo y armonización de luz (preparación común a los dos modos)."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from productshot.services.framing import harmonize, product_bbox, reframe


def photo(size=(1200, 900), box=(100, 500, 300, 800), color=(200, 30, 30)):
    img = Image.new("RGB", size, (60, 80, 200))
    mask = Image.new("L", size, 255)
    ImageDraw.Draw(img).rectangle(box, fill=color)
    ImageDraw.Draw(mask).rectangle(box, fill=0)
    return img, mask


@pytest.mark.parametrize("aspect,ratio", [("4:5", 4 / 5), ("1:1", 1.0)])
def test_reframe_centers_the_product_with_the_requested_aspect(aspect, ratio):
    img, mask = photo()
    out, out_mask = reframe(img, mask, aspect=aspect, fill=0.6, out_long=1000)
    assert out.size == out_mask.size and max(out.size) == 1000
    assert abs(out.width / out.height - ratio) < 0.01
    x0, y0, x1, y1 = product_bbox(out_mask)
    assert abs((x0 + x1) / 2 - out.width / 2) <= 2  # centrado en horizontal
    assert abs((y1 - y0) / out.height - 0.6) < 0.02 or abs((x1 - x0) / out.width - 0.6) < 0.02


def test_reframe_keeps_product_pixels_and_marks_the_new_border_as_editable():
    img, mask = photo()
    out, out_mask = reframe(img, mask, fill=0.6)
    x0, y0, x1, y1 = product_bbox(out_mask)
    assert out.getpixel(((x0 + x1) // 2, (y0 + y1) // 2)) == (200, 30, 30)
    # La esquina cae fuera de la foto original (producto pegado al borde): fondo editable.
    assert out_mask.getpixel((0, 0)) == 255


def test_reframe_rejects_bad_input():
    img, mask = photo()
    with pytest.raises(ValueError):
        reframe(img, Image.new("L", (10, 10), 255))
    with pytest.raises(ValueError):
        reframe(img, Image.new("L", img.size, 255))  # sin producto
    with pytest.raises(ValueError):
        reframe(img, mask, aspect="16:9")


def test_harmonize_moves_product_toward_the_scene_and_is_bounded():
    # Producto gris neutro sobre una escena muy cálida.
    img = Image.new("RGB", (400, 400), (230, 150, 60))
    mask = Image.new("L", (400, 400), 255)
    ImageDraw.Draw(img).rectangle((150, 150, 250, 250), fill=(120, 120, 120))
    ImageDraw.Draw(mask).rectangle((150, 150, 250, 250), fill=0)
    out, info = harmonize(img, mask, max_gain=1.25)
    r, g, b = out.getpixel((200, 200))
    assert info["applied"] and r > g > b  # se calentó
    assert all(1 / 1.25 - 1e-6 <= k <= 1.25 + 1e-6 for k in info["gains"])
    assert out.getpixel((10, 10)) == img.getpixel((10, 10))  # el fondo no cambia


def test_harmonize_without_product_is_a_no_op():
    img = Image.new("RGB", (100, 100), (10, 20, 30))
    out, info = harmonize(img, Image.new("L", (100, 100), 255))
    assert not info["applied"] and out.getpixel((5, 5)) == (10, 20, 30)
