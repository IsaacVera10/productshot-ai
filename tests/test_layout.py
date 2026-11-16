"""Reencuadre del modo avanzado."""

import io

import pytest
from PIL import Image, ImageChops

from productshot.services.generation import MockGenerationBackend
from productshot.services.layout import Layout, compose, product_box
from conftest import create_session


def photo_with_product(size=(400, 300), box=(100, 50, 300, 250)):
    image = Image.new("RGB", size, (200, 200, 200))
    mask = Image.new("L", size, 255)
    image.paste((10, 120, 30), box)
    mask.paste(0, box)
    return image, mask


def test_product_is_scaled_down_and_rests_on_the_lower_third():
    image, mask = photo_with_product()
    canvas, canvas_mask = compose(image, mask, Layout(height=0.4, bottom=0.85))
    assert canvas.size == image.size and canvas_mask.size == mask.size
    x0, y0, x1, y1 = product_box(canvas_mask)
    assert (y1 - y0) == 120 and (x1 - x0) == 120
    assert y1 == round(0.85 * 300) and abs((x0 + x1) / 2 - 200) <= 1


def test_protected_pixels_come_from_the_product():
    image, mask = photo_with_product()
    canvas, canvas_mask = compose(image, mask, Layout())
    inside = canvas.crop(product_box(canvas_mask))
    assert {color for _, color in inside.getcolors()} == {(10, 120, 30)}


def test_small_products_are_not_enlarged():
    image, mask = photo_with_product(box=(190, 140, 210, 160))
    canvas, canvas_mask = compose(image, mask, Layout())
    x0, y0, x1, y1 = product_box(canvas_mask)
    assert (x1 - x0, y1 - y0) == (20, 20)
    assert not ImageChops.difference(canvas.crop((x0, y0, x1, y1)), image.crop((190, 140, 210, 160))).getbbox()


def test_wide_products_are_limited_by_width():
    image, mask = photo_with_product(box=(0, 100, 400, 200))
    _, canvas_mask = compose(image, mask, Layout(width=0.5))
    x0, _, x1, _ = product_box(canvas_mask)
    assert x1 - x0 == 200


class RecordingBackend(MockGenerationBackend):
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return super().generate(request)


@pytest.mark.parametrize("mode", ["intermediate", "advanced"])
def test_only_advanced_mode_reframes_the_product(make_client, mode):
    backend = RecordingBackend()
    client = make_client(generator=backend)
    sid = create_session(client, mode).json()["id"]
    original = Image.open(io.BytesIO(client.get(client.get(f"/api/v1/sessions/{sid}").json()["image_url"]).content))
    client.post(f"/api/v1/sessions/{sid}/segment")
    assert client.post(f"/api/v1/sessions/{sid}/generate").status_code == 202
    assert client.get(f"/api/v1/sessions/{sid}").json()["status"] == "READY_FOR_REVIEW"
    request = backend.requests[0]
    assert request.image.size == original.size
    reframed = not ImageChops.difference(request.image.convert("RGB"), original.convert("RGB")).getbbox()
    assert reframed is (mode == "intermediate")
