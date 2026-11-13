"""Backend real de generación: SDXL base con el pipeline de inpainting de cuatro canales.

Decisión (ADR-004, P3 y P4): el mismo checkpoint, VAE, scheduler, prompts y
posprocesamiento en ambos modos; el modo avanzado solo agrega el LoRA (Hito 5). La
máscara sigue la convención de diffusers y del proyecto: BLANCO = área que se repinta
(fondo), NEGRO = producto protegido. El orquestador recompone el producto desde la foto
original después (G8), así que aquí solo importa el fondo y su integración.

`torch` y `diffusers` se importan al cargar, no al importar el módulo: la API y las
pruebas no los necesitan. `pipeline_factory` permite probar todo sin GPU.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image

from productshot.services.generation import GeneratedArtifact, GenerationRequest, build_prompt
from productshot.services.presets import NEGATIVE_PROMPT

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


@dataclass(frozen=True)
class SdxlParams:
    model_id: str = DEFAULT_MODEL_ID
    revision: str = "main"  # se resuelve al SHA exacto al cargar
    steps: int = 30
    guidance_scale: float = 7.0
    # Con el UNet de cuatro canales, strength < 1 conserva algo del fondo original;
    # 0.99 lo reemplaza casi por completo y deja que el modelo integre bordes y sombras.
    strength: float = 0.99
    target_side: int = 1024  # SDXL rinde mejor cerca de 1024 px
    mask_blur: int = 8  # suaviza el borde de la máscara para la integración (px a target_side)


def generation_size(size: tuple[int, int], target_side: int) -> tuple[int, int]:
    """Tamaño de trabajo para SDXL: lado mayor = target_side, ambos lados múltiplos de 8."""
    w, h = size
    scale = target_side / max(w, h)
    return max(8, round(w * scale / 8) * 8), max(8, round(h * scale / 8) * 8)


class SdxlInpaintBackend:
    name = "sdxl-inpaint"
    simulated = False

    def __init__(self, params: SdxlParams | None = None, *, pipeline_factory: Callable | None = None,
                 device: str = "cuda"):
        self.params = params or SdxlParams()
        self._factory = pipeline_factory
        self._device = device
        self._pipe = None
        self._resolved_revision: str | None = None
        self.load_seconds: float | None = None

    @property
    def model_revision(self) -> str:
        rev = self._resolved_revision or self.params.revision
        return f"{self.params.model_id}@{rev}"

    # --- carga ---

    def load(self) -> None:
        if self._pipe is not None:
            return
        t0 = time.perf_counter()
        if self._factory is not None:
            self._pipe, self._resolved_revision = self._factory(self.params)
        else:
            self._pipe, self._resolved_revision = _load_diffusers(self.params, self._device)
        self.load_seconds = round(time.perf_counter() - t0, 1)
        log.info("SDXL cargado en %.1f s: %s", self.load_seconds, self.model_revision)

    # --- GenerationBackend ---

    def generate(self, request: GenerationRequest) -> list[GeneratedArtifact]:
        self.load()
        p = self.params
        work_size = request.image.size
        gen_size = generation_size(work_size, p.target_side)
        image = request.image.convert("RGB").resize(gen_size, Image.Resampling.LANCZOS)
        mask = request.mask.convert("L").resize(gen_size, Image.Resampling.BILINEAR)
        if p.mask_blur and hasattr(self._pipe, "mask_processor"):
            mask = self._pipe.mask_processor.blur(mask, blur_factor=p.mask_blur)

        results = []
        for spec in request.specs:
            prompt = build_prompt(spec.style, spec.adjustments)
            t0 = time.perf_counter()
            _reset_peak_memory()
            out = self._pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=image,
                mask_image=mask,
                height=gen_size[1],
                width=gen_size[0],
                num_inference_steps=p.steps,
                guidance_scale=p.guidance_scale,
                strength=p.strength,
                generator=_generator(self._device, spec.seed, self._factory is not None),
            ).images[0]
            elapsed = round(time.perf_counter() - t0, 2)
            # De vuelta al tamaño de la foto de trabajo: G8 compara píxel a píxel.
            final = out.convert("RGB").resize(work_size, Image.Resampling.LANCZOS)
            results.append(
                GeneratedArtifact(
                    position=spec.position,
                    style=spec.style,
                    image=final,
                    seed=spec.seed,
                    model_revision=self.model_revision,
                    prompt_version=request.prompt_version,
                    params={
                        "backend": self.name,
                        "simulated": False,
                        "mode": request.mode.value,
                        "lora": None,  # el LoRA del modo avanzado llega en el Hito 5
                        "prompt": prompt,
                        "negative_prompt": NEGATIVE_PROMPT,
                        "adjustments": list(spec.adjustments),
                        "steps": p.steps,
                        "guidance_scale": p.guidance_scale,
                        "strength": p.strength,
                        "generation_size": list(gen_size),
                        "seconds": elapsed,
                        "peak_vram_gb": _peak_memory_gb(),
                    },
                    variant=spec.variant,
                )
            )
            log.info("Variación %d (%s) en %.1f s", spec.position, spec.variant, elapsed)
        return results


# --- dependencias pesadas (solo en el worker con GPU) ---


def _load_diffusers(params: SdxlParams, device: str):
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline
    from huggingface_hub import HfApi

    sha = HfApi().model_info(params.model_id, revision=params.revision).sha
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        params.model_id, revision=sha, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, sha


def _generator(device: str, seed: int, fake: bool):
    if fake:
        return seed
    import torch

    return torch.Generator(device=device).manual_seed(seed)


def _reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _peak_memory_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    except ImportError:
        pass
    return None
