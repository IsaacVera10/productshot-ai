"""Backend SDXL sin GPU: un pipeline falso registra cómo se lo llama."""

from __future__ import annotations

from PIL import Image

from productshot.domain import Mode, Style
from productshot.services.generation import GenerationRequest, VariationSpec, build_prompt
from productshot.services.sdxl_generation import SdxlInpaintBackend, SdxlParams, generation_size


class FakePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        img = Image.new("RGB", (kw["width"], kw["height"]), (10, 200, 10))
        return type("Out", (), {"images": [img]})()


def backend():
    pipe = FakePipe()
    b = SdxlInpaintBackend(SdxlParams(steps=4), pipeline_factory=lambda p: (pipe, "abc123"))
    return b, pipe


def request(size=(640, 480)):
    specs = tuple(VariationSpec(position=i + 1, style=s, seed=7 + i, adjustments=("brighter",) if i == 0 else ())
                  for i, s in enumerate(Style))
    return GenerationRequest(session_id="s", mode=Mode.INTERMEDIATE, image=Image.new("RGB", size, "red"),
                             mask=Image.new("L", size, 255), specs=specs)


def test_generation_size_targets_1024_and_multiples_of_8():
    assert generation_size((640, 480), 1024) == (1024, 768)
    assert generation_size((1000, 333), 1024) == (1024, 344)
    w, h = generation_size((321, 777), 1024)
    assert h == 1024 and w % 8 == 0


def test_generates_five_at_work_size_with_pinned_revision():
    b, pipe = backend()
    arts = b.generate(request())
    assert len(arts) == 5 and all(a.image.size == (640, 480) for a in arts)
    assert b.model_revision == "stabilityai/stable-diffusion-xl-base-1.0@abc123"
    assert all(not a.params["simulated"] for a in arts) and b.simulated is False
    assert [c["generator"] for c in pipe.calls] == [7, 8, 9, 10, 11]  # una semilla por variación
    assert pipe.calls[0]["width"] == 1024 and pipe.calls[0]["height"] == 768
    assert pipe.calls[0]["prompt"] == build_prompt(Style.STUDIO, ("brighter",))
    assert arts[0].variant == "studio+brighter"


def test_works_behind_the_orchestrator(make_client):
    from conftest import ready_session

    b, _ = backend()
    client = make_client(generator=b)
    s = ready_session(client)
    assert len(s["variations"]) == 5
    assert s["simulated"] is True  # la máscara sigue siendo la elipse simulada
    assert s["variations"][0]["generation"]["model_revision"].endswith("@abc123")


def test_mitigation_knobs_change_prompt_and_mask():
    pipe = FakePipe()
    b = SdxlInpaintBackend(SdxlParams(steps=4, mask_blur=0, mask_dilate_px=5, prompt_prefix="a ceramic mug"),
                           pipeline_factory=lambda p: (pipe, "abc"))
    req = request()
    mask = Image.new("L", (640, 480), 255)
    mask.paste(0, (300, 200, 340, 280))
    req = GenerationRequest(session_id="s", mode=req.mode, image=req.image, mask=mask, specs=req.specs)
    arts = b.generate(req)
    assert pipe.calls[0]["prompt"].startswith("a ceramic mug, ")
    sent = pipe.calls[0]["mask_image"]
    black = sum(1 for v in sent.getdata() if v == 0)
    assert black > (40 * 1.6) * (80 * 1.6)  # el producto escalado a 1024 px, más el margen
    assert arts[0].params["mask_dilate_px"] == 5
