"""Categoría del producto de punta a punta y el backend de inpainting del experimento RealVisXL (descartado)."""

from __future__ import annotations

import dataclasses

import httpx
import pytest
from PIL import Image, ImageDraw

from productshot.domain import Category, Mode, Style
from productshot.services.generation import GenerationRequest, MockGenerationBackend, VariationSpec
from productshot.services.presets import NEGATIVE_PROMPT, PRESETS
from productshot.services.remote_generation import RemoteGenerationBackend
from productshot.services.scenes import SCENE_NEGATIVE, SCENE_PROMPT_VERSION, build_scene_prompt
from productshot.services.sdxl_generation import (
    REALVIS_INPAINT_MODEL_ID,
    REALVIS_INPAINT_PARAMS,
    SdxlInpaintBackend,
    SdxlParams,
    prefill_background,
)
from productshot.worker.app import create_worker_app
from test_ports import worker_transport
from test_sdxl_backend import FakePipe


def product_photo(size=(320, 240)):
    """Foto con fondo azul y un "producto" rojo al centro; máscara: negro = producto."""
    img = Image.new("RGB", size, (40, 60, 200))
    mask = Image.new("L", size, 255)
    box = [size[0] // 3, size[1] // 3, 2 * size[0] // 3, 2 * size[1] // 3]
    ImageDraw.Draw(img).rectangle(box, fill=(200, 20, 20))
    ImageDraw.Draw(mask).rectangle(box, fill=0)
    return img, mask


def make_request(mode=Mode.INTERMEDIATE, category=Category.TECNOLOGIA):
    img, mask = product_photo()
    specs = tuple(VariationSpec(position=i + 1, style=s, seed=7 + i) for i, s in enumerate(Style))
    return GenerationRequest(session_id="s", mode=mode, image=img, mask=mask, specs=specs, category=category)


def intermediate_backend():
    pipe = FakePipe()
    params = dataclasses.replace(REALVIS_INPAINT_PARAMS, steps=4)
    return SdxlInpaintBackend(params, pipeline_factory=lambda p: (pipe, "rv123")), pipe


# --- prompts de escena ---


def test_scene_prompt_follows_category_and_leaves_presets_untouched():
    tech = build_scene_prompt(Style.LIFESTYLE, Category.TECNOLOGIA)
    kitchen = build_scene_prompt(Style.LIFESTYLE, Category.COCINA)
    assert "laptop" in tech and "office desk" in tech
    assert "kitchen" in kitchen and "laptop" not in kitchen
    # Sin categoría: escena genérica, nunca un error.
    assert "wooden table" in build_scene_prompt(Style.LIFESTYLE, None)
    # Los textos del LoRA (presets.py) siguen siendo los originales.
    assert "kitchen table" in PRESETS[Style.LIFESTYLE].prompt


def test_scene_prompt_appends_round_adjustments():
    prompt = build_scene_prompt(Style.STUDIO, Category.HOGAR, ("brighter",))
    assert prompt.startswith("professional e-commerce product photo")
    assert len(prompt) > len(build_scene_prompt(Style.STUDIO, Category.HOGAR))


def test_scene_negative_drops_studio_gear_and_food():
    for word in ("tripod", "softbox", "food", "bread", "clutter", "debris"):
        assert word in SCENE_NEGATIVE


# --- mezcla de fondo ---


@pytest.mark.parametrize("how", ["blend", "palette"])
def test_prefill_changes_background_and_keeps_product(how):
    img, mask = product_photo()
    out = prefill_background(img, mask, Style.WARM, how)
    assert out.getpixel((160, 120)) == img.getpixel((160, 120))  # producto intacto
    assert out.getpixel((5, 5)) != img.getpixel((5, 5))  # el fondo azul ya no está


def test_prefill_rejects_unknown_mode():
    img, mask = product_photo()
    with pytest.raises(ValueError):
        prefill_background(img, mask, Style.WARM, "otro")


# --- backend de inpainting del experimento RealVisXL ---


def test_realvis_params_are_the_measured_configuration():
    assert REALVIS_INPAINT_PARAMS.model_id == REALVIS_INPAINT_MODEL_ID
    assert (REALVIS_INPAINT_PARAMS.strength, REALVIS_INPAINT_PARAMS.prefill, REALVIS_INPAINT_PARAMS.prompt_set) == (
        0.90, "blend", "scene")


def test_intermediate_backend_uses_scene_prompts_and_prefilled_input():
    b, pipe = intermediate_backend()
    arts = b.generate(make_request())
    assert len(arts) == 5 and len(pipe.calls) == 5
    lifestyle = next(c for c, s in zip(pipe.calls, Style) if s == Style.LIFESTYLE)
    assert "laptop" in lifestyle["prompt"]
    assert all(c["negative_prompt"] == SCENE_NEGATIVE for c in pipe.calls)
    # Cada estilo recibe su propia entrada con el fondo cambiado; el producto no se toca.
    corners = {c["image"].getpixel((2, 2)) for c in pipe.calls}
    assert len(corners) == 5
    first = arts[0]
    assert first.prompt_version == SCENE_PROMPT_VERSION
    assert first.params["category"] == "tecnologia" and first.params["prefill"] == "blend"
    assert first.params["strength"] == 0.90
    assert first.image.size == (320, 240)


def test_default_params_keep_the_preset_pipeline():
    pipe = FakePipe()
    b = SdxlInpaintBackend(SdxlParams(steps=4), pipeline_factory=lambda p: (pipe, "abc"))
    req = make_request(mode=Mode.ADVANCED)
    b.generate(req)
    assert all(c["negative_prompt"] == NEGATIVE_PROMPT for c in pipe.calls)
    assert PRESETS[Style.STUDIO].prompt in pipe.calls[0]["prompt"]
    # Sin prefill, todos los estilos reciben la misma foto.
    assert len({c["image"].getpixel((2, 2)) for c in pipe.calls}) == 1


# --- contrato HTTP ---


class RecordingBackend(MockGenerationBackend):
    def __init__(self):
        super().__init__()
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return super().generate(request)


@pytest.mark.parametrize("category", [Category.JUGUETES, None])
def test_category_travels_through_the_worker_contract(category):
    rec = RecordingBackend()
    client = httpx.Client(base_url="http://worker", transport=worker_transport(create_worker_app(rec)))
    remote = RemoteGenerationBackend("http://worker", client=client)
    arts = remote.generate(make_request(category=category))
    assert len(arts) == 5
    assert rec.requests[0].category == category


# --- categoría de punta a punta (API -> sesión -> generador -> ronda siguiente) ---


def test_category_is_stored_sent_to_the_generator_and_inherited(make_client):
    from conftest import decide_all, image_bytes

    rec = RecordingBackend()
    client = make_client(generator=rec)
    r = client.post("/api/v1/sessions", files={"image": ("p.png", image_bytes(), "image/png")},
                    data={"mode": "intermediate", "category": "hogar"})
    assert r.status_code == 201 and r.json()["category"] == "hogar"
    sid = r.json()["id"]
    assert client.post(f"/api/v1/sessions/{sid}/segment").status_code == 200
    assert client.post(f"/api/v1/sessions/{sid}/generate").status_code == 202
    assert rec.requests[-1].category == Category.HOGAR
    session = client.get(f"/api/v1/sessions/{sid}").json()
    decide_all(client, session, ["accepted", "rejected", "accepted", "rejected", "rejected"])
    assert client.post(f"/api/v1/sessions/{sid}/summary").status_code == 200
    child = client.post(f"/api/v1/sessions/{sid}/next-round").json()
    assert child["category"] == "hogar"
    assert client.post(f"/api/v1/sessions/{child['id']}/generate").status_code == 202
    assert rec.requests[-1].category == Category.HOGAR


def test_category_is_optional_and_validated(client):
    from conftest import image_bytes

    ok = client.post("/api/v1/sessions", files={"image": ("p.png", image_bytes(), "image/png")},
                     data={"mode": "advanced"})
    assert ok.status_code == 201 and ok.json()["category"] is None
    bad = client.post("/api/v1/sessions", files={"image": ("p.png", image_bytes(), "image/png")},
                      data={"mode": "intermediate", "category": "ropa"})
    assert bad.status_code == 422
