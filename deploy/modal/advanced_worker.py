"""Worker del modo avanzado en Modal: UNet de inpainting de nueve canales con el LoRA propio.

La API lo usa con ADVANCED_WORKER_URL y ADVANCED_WORKER_TOKEN; el intermedio sigue en
`worker.py`. Solo sirve /v1/health y /v1/generate: la API recibe el lienzo ya reencuadrado
(`services/layout.py`) y recompone el producto con G8.

    modal volume put productshot-hf-cache <adaptador> lora/<nombre>.safetensors
    modal run deploy/modal/advanced_worker.py::smoke     # un lote de cinco con un producto sintético
    modal deploy deploy/modal/advanced_worker.py

Secreto requerido: `productshot-advanced` con `GENERATION_WORKER_TOKEN` (el mismo valor va en
ADVANCED_WORKER_TOKEN de la API). El adaptador se verifica por sha256 antes de servir.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
CACHE = "/cache"

# v5distill checkpoint 1000 con prompts v2: ganó la revisión ciega y no muestra equipo de estudio (docs/10).
LORA_PATH = f"{CACHE}/lora/v5distill-ckpt1000.safetensors"
LORA_SHA256 = "d1ccc94a466886eddba02706a9133a6e7d993dc30a580ae607e4b05803aba531"
STRENGTH = 1.0
MODEL_REVISION = "115134f363124c53c7d878647567d04daf26e41e"  # la misma revisión evaluada en val y test
PROMPT_VERSION = "v2"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "diffusers==0.31.0",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "safetensors==0.4.5",
        "huggingface_hub==0.26.2",
        "peft==0.13.2",
        "fastapi==0.162.0",
        "pydantic==2.14.2",
        "pydantic-settings==2.16.2",
        "pillow==12.3.0",
        "httpx==0.28.1",
    )
    .env({"HF_HOME": f"{CACHE}/hf", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_dir(ROOT / "backend" / "productshot", "/root/productshot")
)

app = modal.App("productshot-advanced", image=image)
cache = modal.Volume.from_name("productshot-hf-cache", create_if_missing=True)


@app.cls(
    gpu=["L4", "A10G"],  # A10G si no hay L4 libres: Modal llegó a no tener capacidad L4
    volumes={CACHE: cache},
    secrets=[modal.Secret.from_name("productshot-advanced")],
    timeout=600,
    scaledown_window=120,
    max_containers=1,
)
class AdvancedWorker:
    @modal.enter()
    def load(self) -> None:
        if not os.environ.get("GENERATION_WORKER_TOKEN"):
            raise RuntimeError("Falta GENERATION_WORKER_TOKEN: el worker no se sirve sin autenticación.")
        from productshot.services.sdxl_generation import NINE_CHANNEL_MODEL_ID, SdxlInpaintBackend, SdxlParams

        from productshot.config import Settings
        from productshot.services.screening import SafetyCheckedBackend, _load_nsfw_classifier

        sdxl = SdxlInpaintBackend(SdxlParams(
            model_id=NINE_CHANNEL_MODEL_ID, revision=MODEL_REVISION, strength=STRENGTH,
            prompt_version=PROMPT_VERSION,
        ))
        sdxl.load()
        loaded = sdxl.attach_lora(LORA_PATH)
        if loaded["sha256"] != LORA_SHA256:
            raise RuntimeError(f"El LoRA del volumen no coincide: {loaded['sha256']}")
        # G7: cada salida pasa por el mismo clasificador NSFW que el worker principal.
        self.backend = SafetyCheckedBackend(sdxl, _load_nsfw_classifier("cuda"),
                                            threshold=Settings(_env_file=None).nsfw_threshold)
        cache.commit()

    @modal.asgi_app()
    def web(self):
        from productshot.worker.app import create_worker_app

        return create_worker_app(self.backend, token=os.environ.get("GENERATION_WORKER_TOKEN"))

    @modal.method()
    def smoke(self, image_png: bytes, mask_png: bytes) -> dict:
        from PIL import Image

        from productshot.domain import Mode, Style
        from productshot.services.generation import GenerationRequest, VariationSpec

        image = Image.open(io.BytesIO(image_png)).convert("RGB")
        mask = Image.open(io.BytesIO(mask_png)).convert("L")
        specs = tuple(VariationSpec(position=i + 1, style=s, seed=100 + i) for i, s in enumerate(Style))
        arts = self.backend.generate(GenerationRequest(session_id="smoke", mode=Mode.ADVANCED, image=image,
                                                       mask=mask, specs=specs))
        out = {"model_revision": self.backend.model_revision, "lora": self.backend.lora, "images": {}, "params": {}}
        for art in arts:
            buf = io.BytesIO()
            art.image.save(buf, format="PNG")
            out["images"][art.style.value] = buf.getvalue()
            out["params"][art.style.value] = art.params | {"prompt_version": art.prompt_version}
        return out


@app.local_entrypoint()
def smoke(out: str = "data/experiments/advanced_smoke") -> None:
    """Un lote de cinco con un producto sintético reencuadrado como lo hace la API."""
    from PIL import Image, ImageDraw

    from productshot.services.layout import Layout, compose

    photo = Image.new("RGB", (1024, 768), (200, 200, 200))
    ImageDraw.Draw(photo).rounded_rectangle((420, 180, 600, 560), radius=40, fill=(245, 245, 240))
    mask = Image.new("L", photo.size, 255)
    ImageDraw.Draw(mask).rounded_rectangle((420, 180, 600, 560), radius=40, fill=0)
    canvas, canvas_mask = compose(photo, mask, Layout())

    def png(im) -> bytes:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    result = AdvancedWorker().smoke.remote(png(canvas), png(canvas_mask))
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    for style, data in result["images"].items():
        (dest / f"{style}.png").write_bytes(data)
    manifest = {k: result[k] for k in ("model_revision", "lora", "params")}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    params = result["params"].values()
    if (len(result["images"]) != 5 or result["lora"]["sha256"] != LORA_SHA256
            or any(p["prompt_version"] != PROMPT_VERSION or (p["lora"] or {}).get("sha256") != LORA_SHA256
                   or p["lora"]["scale"] != 1.0
                   or p["strength"] != STRENGTH or "g7" not in p for p in params)):
        raise SystemExit(f"El smoke no coincide con la configuración del worker: revisar {dest / 'manifest.json'}")
    seconds = sorted(p["seconds"] for p in params)
    print(f"{result['model_revision']} · LoRA {LORA_SHA256[:12]} · prompts {PROMPT_VERSION} · strength {STRENGTH} "
          f"· {seconds[2]} s por imagen (mediana) · {dest}")
