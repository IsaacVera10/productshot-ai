"""Bloque del modo intermedio (QwenEditBackend) y G8 por modo, sin GPU."""

from __future__ import annotations

import itertools

import pytest
from conftest import create_session
from PIL import Image, ImageDraw

from productshot.domain import Category, Mode, Style
from productshot.services import fidelity
from productshot.services.generation import GenerationRequest, VariationSpec
from productshot.services.qwen_edit import EDIT_PROMPT_VERSION, QwenEditBackend, UnverifiedBatch, edit_size

API = "/api/v1/sessions"


def photo(size=(400, 300)):
    img = Image.new("RGB", size, (40, 60, 200))
    mask = Image.new("L", size, 255)
    box = (150, 90, 250, 210)
    ImageDraw.Draw(img).rectangle(box, fill=(200, 20, 20))
    ImageDraw.Draw(mask).rectangle(box, fill=0)
    return img, mask


def request(category=Category.HOGAR, size=(400, 300)):
    img, mask = photo(size)
    specs = tuple(VariationSpec(position=i + 1, style=s, seed=10 + i) for i, s in enumerate(Style))
    return GenerationRequest(session_id="s", mode=Mode.INTERMEDIATE, image=img, mask=mask, specs=specs,
                             category=category)


class FakeEditor:
    """Devuelve una imagen distinta por llamada y registra lo que recibió."""

    def __init__(self):
        self.calls = []

    def __call__(self, *, prompt, image, num_inference_steps, generator):
        self.calls.append({"prompt": prompt, "size": image.size, "seed": generator, "steps": num_inference_steps})
        out = Image.new("RGB", (image.width * 2, image.height * 2), (len(self.calls) % 255, 120, 90))
        return type("Out", (), {"images": [out]})()


GOOD = {"aligned": True, "matches": 300, "inlier_ratio": 0.8, "edges": 0.8, "structure": 0.6, "scene_change": 60.0}
BAD = {**GOOD, "inlier_ratio": 0.1, "edges": 0.1}


def backend(verdicts, *, candidates=3, budget_s=420.0, clock=None):
    """`verdicts`: iterador de métricas, una por candidata generada."""
    editor = FakeEditor()
    it = iter(verdicts)
    b = QwenEditBackend(candidates=candidates, budget_s=budget_s, pipeline_factory=lambda m: (editor, "sha1"),
                        verifier=lambda o, c, m: dict(next(it)), clock=clock or (lambda: 0.0))
    return b, editor


def test_returns_five_verified_images_at_photo_size():
    b, editor = backend(itertools.repeat(GOOD))
    req = request()
    arts = b.generate(req)
    assert [a.position for a in arts] == [1, 2, 3, 4, 5]
    assert len(editor.calls) == 15  # 3 candidatas × 5 estilos
    assert all(a.image.size == req.image.size for a in arts)
    assert editor.calls[0]["size"] == edit_size(req.image.size)
    a = arts[0]
    assert a.prompt_version == EDIT_PROMPT_VERSION and a.model_revision.endswith("@sha1")
    f = a.params["fidelity"]
    assert f["method"] == "verificado" and f["candidates"] == 3 and f["retries"] == 0
    assert a.params["category"] == "hogar"


def test_prompt_uses_category_scene_and_preservation_clause():
    b, editor = backend(itertools.repeat(GOOD))
    b.generate(request(category=Category.COCINA))
    lifestyle = editor.calls[3]["prompt"]  # posición 4 = lifestyle en la primera ronda
    assert "French press" in lifestyle and "do not rotate or mirror" in lifestyle
    assert "seamless light grey" in editor.calls[0]["prompt"]  # estudio: sin contexto


def test_picks_the_best_passing_candidate():
    better = {**GOOD, "inlier_ratio": 0.95, "edges": 0.9}
    # Ronda base: el estilo 1 recibe BAD, GOOD y better (una por vuelta); el resto, GOOD.
    seq = [BAD, GOOD, GOOD, GOOD, GOOD, GOOD, GOOD, GOOD, GOOD, GOOD, better, GOOD, GOOD, GOOD, GOOD]
    b, editor = backend(seq)
    arts = b.generate(request())
    assert arts[0].params["fidelity"]["inlier_ratio"] == 0.95
    assert arts[0].params["fidelity"]["passed_candidates"] == 2
    assert arts[0].params["candidate_seed"] == editor.calls[10]["seed"]
    assert [a.seed for a in arts] == [s.seed for s in request().specs]  # semillas pedidas


def test_retries_only_failing_styles_while_time_allows():
    # Estilo 1 falla en sus 3 candidatas y pasa en el primer reintento.
    seq = [BAD, GOOD, GOOD, GOOD, GOOD] * 3 + [GOOD]
    b, editor = backend(seq)
    arts = b.generate(request())
    assert len(editor.calls) == 16
    assert arts[0].params["fidelity"]["retries"] == 1 and arts[1].params["fidelity"]["retries"] == 0


def test_fails_the_whole_batch_when_time_runs_out():
    now = {"t": 0.0}
    b, editor = backend(itertools.chain([BAD] * 1, itertools.repeat(GOOD, 4), [BAD], itertools.repeat(GOOD, 4),
                                        [BAD], itertools.repeat(GOOD, 4), itertools.repeat(BAD)),
                        budget_s=100.0, clock=lambda: now["t"])

    original = b._candidate

    def slow(*args, **kwargs):
        now["t"] += 9.0  # cada generación tarda 9 s del reloj
        return original(*args, **kwargs)

    b._candidate = slow
    with pytest.raises(UnverifiedBatch) as exc:
        b.generate(request())
    assert "studio" in str(exc.value)
    assert now["t"] <= 100.0 + 9.0  # se detuvo al no caber otra generación


def test_real_verifier_rejects_a_mirrored_product_and_accepts_the_same_one():
    pytest.importorskip("cv2")
    img = Image.new("RGB", (400, 300), (30, 30, 30))
    mask = Image.new("L", (400, 300), 255)
    d = ImageDraw.Draw(img)
    d.rectangle((120, 60, 280, 240), fill=(230, 230, 230))
    for i, word in enumerate(("WD_BLACK", "SN850X", "1TB 7300")):
        d.text((130, 80 + i * 40), word, fill=(10, 10, 10))
    d.ellipse((230, 190, 270, 230), fill=(200, 30, 30))
    ImageDraw.Draw(mask).rectangle((120, 60, 280, 240), fill=0)
    scene = Image.new("RGB", (400, 300), (180, 140, 90))
    same = scene.copy()
    same.paste(img.crop((120, 60, 280, 240)), (125, 62))
    mirrored = scene.copy()
    mirrored.paste(img.crop((120, 60, 280, 240)).transpose(Image.Transpose.FLIP_LEFT_RIGHT), (125, 62))
    assert fidelity.passes(fidelity.verify(img, same, mask))
    assert not fidelity.passes(fidelity.verify(img, mirrored, mask))
    assert not fidelity.passes(fidelity.verify(img, img, mask))  # no cambió la escena


# --- G8 por modo en la API ---


class VerifiedGenerator:
    """Simula el worker del modo intermedio: imágenes ya verificadas, del tamaño de la foto."""

    name = "fake-edit"
    simulated = False
    model_revision = "fake@1"

    def __init__(self, method="verificado"):
        self.method = method

    def generate(self, req):
        from productshot.services.generation import GeneratedArtifact

        out = []
        for s in req.specs:
            img = Image.new("RGB", req.image.size, (250, 200, 10))  # el editor re-dibuja todo, producto incluido
            out.append(GeneratedArtifact(position=s.position, style=s.style, image=img, seed=s.seed,
                                         model_revision=self.model_revision, prompt_version="edit-v1",
                                         params={"fidelity": {"method": self.method, **GOOD}}, variant=s.variant))
        return out


def run_batch(client, mode):
    sid = create_session(client, mode).json()["id"]
    assert client.post(f"{API}/{sid}/segment").status_code == 200
    assert client.post(f"{API}/{sid}/generate").status_code == 202
    return client.get(f"{API}/{sid}").json()


def test_intermediate_verified_images_are_not_recomposed(make_client):
    client = make_client(generator=VerifiedGenerator())
    body = run_batch(client, "intermediate")
    assert body["status"] == "READY_FOR_REVIEW"
    img = client.get(body["variations"][0]["image_url"]).content
    pic = Image.open(__import__("io").BytesIO(img)).convert("RGB")
    w, h = pic.size
    assert pic.getpixel((w // 2, h // 2)) == (250, 200, 10)  # el centro (producto) no se re-pegó


def test_advanced_or_unverified_images_are_recomposed(make_client):
    for mode, method in (("advanced", "verificado"), ("intermediate", "otro")):
        client = make_client(generator=VerifiedGenerator(method))
        body = run_batch(client, mode)
        assert body["status"] == "READY_FOR_REVIEW"
        pic = Image.open(__import__("io").BytesIO(client.get(body["variations"][0]["image_url"]).content)).convert("RGB")
        w, h = pic.size
        assert pic.getpixel((w // 2, h // 2)) != (250, 200, 10)  # el producto se recompuso desde la foto


def test_block_passes_the_orchestrator_end_to_end(make_client):
    """El bloque real (editor falso) por la API: lote listo, G8 verificado y semillas pedidas."""
    b, _ = backend(itertools.cycle([BAD, GOOD]))  # la mitad de las candidatas falla
    client = make_client(generator=b)
    body = run_batch(client, "intermediate")
    assert body["status"] == "READY_FOR_REVIEW", body.get("failure_reason")
    assert len(body["variations"]) == 5


def test_remote_generation_follows_modal_303_and_keeps_the_token():
    """Modal responde 303 pasados 150 s; el cliente sigue la redirección como GET con el token."""
    import httpx

    from productshot.services.generation import MockGenerationBackend
    from productshot.services.remote_generation import RemoteGenerationBackend
    from productshot.worker.app import create_worker_app
    from productshot.worker.contract import GENERATE_PATH
    from test_ports import worker_transport

    inner = worker_transport(create_worker_app(MockGenerationBackend(), token="t"))
    stash, seen = {}, []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path, req.headers.get("authorization")))
        if req.method == "POST" and req.url.path == GENERATE_PATH:
            stash["body"] = req.content
            return httpx.Response(303, headers={"location": f"{GENERATE_PATH}?__modal_function_call_id=fc-1"})
        if req.method == "GET" and req.url.path == GENERATE_PATH:
            # Modal entrega el resultado de la llamada original.
            return inner.handle_request(httpx.Request("POST", req.url.copy_with(query=None), headers=req.headers,
                                                      content=stash["body"]))
        return inner.handle_request(req)

    client = httpx.Client(base_url="http://worker", transport=httpx.MockTransport(handler))
    arts = RemoteGenerationBackend("http://worker", token="t", client=client).generate(request())
    assert len(arts) == 5
    assert ("GET", GENERATE_PATH, "Bearer t") in seen


def test_unverified_batch_reaches_the_user_with_its_message(make_client):
    """El lote sin verificar viaja por HTTP como 422 y la sesión muestra el mensaje útil."""
    import httpx

    from productshot.services.remote_generation import RemoteGenerationBackend
    from productshot.worker.app import create_worker_app
    from test_ports import worker_transport

    b, _ = backend(itertools.repeat(BAD), budget_s=0.0)  # nunca pasa y no hay tiempo para reintentos
    client_http = httpx.Client(base_url="http://w", transport=worker_transport(create_worker_app(b)))
    client = make_client(generator=RemoteGenerationBackend("http://w", client=client_http))
    body = run_batch(client, "intermediate")
    assert body["status"] == "FAILED"
    assert "cables sueltos" in body["failure_reason"]


# --- un worker por modo ---


def test_qwen_worker_refuses_advanced_requests():
    b, editor = backend(itertools.repeat(GOOD))
    req = request()
    adv = GenerationRequest(session_id="s", mode=Mode.ADVANCED, image=req.image, mask=req.mask, specs=req.specs)
    with pytest.raises(Exception, match="solo genera el modo intermedio"):
        b.generate(adv)
    assert not editor.calls


def test_api_routes_each_mode_to_its_worker(make_client):
    import httpx

    from productshot.services.generation import MockGenerationBackend, ModeRoutedGenerator
    from productshot.services.remote_generation import RemoteGenerationBackend
    from productshot.worker.app import create_worker_app
    from test_ports import worker_transport

    class Named(MockGenerationBackend):
        def __init__(self, name):
            super().__init__()
            self.name, self.calls = name, []

        def generate(self, req):
            self.calls.append(req.mode)
            return super().generate(req)

    workers = {Mode.INTERMEDIATE: Named("qwen-edit"), Mode.ADVANCED: Named("sdxl-lora")}
    remotes = {m: RemoteGenerationBackend("http://w", client=httpx.Client(
        base_url="http://w", transport=worker_transport(create_worker_app(w)))) for m, w in workers.items()}
    client = make_client(generator=ModeRoutedGenerator(remotes))
    for mode in ("intermediate", "advanced"):
        assert run_batch(client, mode)["status"] == "READY_FOR_REVIEW"
    assert workers[Mode.INTERMEDIATE].calls == [Mode.INTERMEDIATE]
    assert workers[Mode.ADVANCED].calls == [Mode.ADVANCED]


def test_advanced_worker_url_enables_routing(tmp_path):
    from productshot.composition import build_generator
    from productshot.config import Settings
    from productshot.services.generation import ModeRoutedGenerator

    base = dict(_env_file=None, generation_backend="remote", generation_worker_url="http://inter")
    single = build_generator(Settings(**base))
    routed = build_generator(Settings(**base, advanced_worker_url="http://adv", advanced_worker_token="t"))
    assert not isinstance(single, ModeRoutedGenerator)
    assert isinstance(routed, ModeRoutedGenerator)
    assert routed.for_mode(Mode.ADVANCED)._client.base_url.host == "adv"
    assert routed.for_mode(Mode.INTERMEDIATE)._client.base_url.host == "inter"


# --- correcciones de la revisión de Isaac (2026-11-16) ---


@pytest.mark.parametrize("bad", [
    {"inlier_ratio": 0.0},
    {"edges": 0.0},
    {"edges": float("nan")},
    {"inlier_ratio": float("inf")},
    {"scene_change": "60"},
    {"aligned": False},
])
def test_api_rejects_verified_images_with_invalid_metrics(make_client, bad):
    class Liar(VerifiedGenerator):
        def generate(self, req):
            arts = super().generate(req)
            for a in arts:
                a.params["fidelity"].update(bad)
            return arts

    body = run_batch(make_client(generator=Liar()), "intermediate")
    assert body["status"] == "FAILED"
    assert "verificación" in body["failure_reason"]


def test_api_rejects_verified_images_with_missing_metrics(make_client):
    class Partial(VerifiedGenerator):
        def generate(self, req):
            arts = super().generate(req)
            for a in arts:
                a.params["fidelity"] = {"method": "verificado", "inlier_ratio": 0.9}
            return arts

    body = run_batch(make_client(generator=Partial()), "intermediate")
    assert body["status"] == "FAILED" and "faltan métricas" in body["failure_reason"]


def test_remote_generator_with_mock_screening_does_not_start(tmp_path):
    from productshot.composition import ConfigError, build_services
    from productshot.config import Settings

    s = Settings(_env_file=None, sqlite_path=tmp_path / "db.sqlite3", asset_root=tmp_path / "a",
                 generation_backend="remote", generation_worker_url="http://w")
    with pytest.raises(ConfigError, match="SCREENING_BACKEND=remote"):
        build_services(s)


def test_mock_screening_is_visible_in_the_session(client):
    r = create_session(client, "intermediate")
    codes = [w["code"] for w in client.get(f"{API}/{r.json()['id']}").json()["warnings"]]
    assert "screening_simulated" in codes


def test_gemini_defaults_are_the_responsive_models():
    from productshot.config import Settings

    s = Settings(_env_file=None)
    assert (s.gemini_model_id, s.gemini_fallback_model_id) == ("gemini-3.5-flash", "gemini-3.5-flash-lite")
