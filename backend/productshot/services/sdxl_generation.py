"""Backend real de generación: SDXL base con el pipeline de inpainting de cuatro canales.

Decisión (ADR-004, P3 y P4): el mismo checkpoint, VAE, scheduler, prompts y
posprocesamiento en ambos modos; el modo avanzado solo agrega el LoRA (Hito 5): con un
adaptador cargado (`attach_lora`), las solicitudes `advanced` lo usan con `lora_scale` y las
`intermediate` con peso 0, así que un mismo pipeline sirve ambos modos. La
máscara sigue la convención de diffusers y del proyecto: BLANCO = área que se repinta
(fondo), NEGRO = producto protegido. El orquestador recompone el producto desde la foto
original después (G8), así que aquí solo importa el fondo y su integración.

`torch` y `diffusers` se importan al cargar, no al importar el módulo: la API y las
pruebas no los necesitan. `pipeline_factory` permite probar todo sin GPU.

Experimento del 2026-11-15 (descartado para el modo intermedio): `REALVIS_INPAINT_PARAMS` usa
un checkpoint fotorrealista de inpainting de nueve canales (RealVisXL V4.0), una mezcla del
fondo antes de generar (`prefill`) y prompts de escena por categoría (`scenes.py`). Con fotos
reales el producto se veía "pegado" (la luz no coincide) y el modo intermedio pasó a un editor
por instrucciones (`qwen_edit.py`). Se conserva porque lo usan los scripts de experimento.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter

from productshot.domain import Mode, Style
from productshot.services.generation import GeneratedArtifact, GenerationRequest, build_prompt
from productshot.services.presets import NEGATIVE_PROMPTS, PRESETS, PROMPT_VERSION
from productshot.services.scenes import SCENE_NEGATIVE, SCENE_PROMPT_VERSION, build_scene_prompt

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# UNet entrenado para inpainting (nueve canales de entrada). Candidato de P4, en evaluación.
NINE_CHANNEL_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
# Fine-tune fotorrealista de SDXL para inpainting (nueve canales, openrail++). Solo experimentos.
REALVIS_INPAINT_MODEL_ID = "OzzyGT/RealVisXL_V4.0_inpainting"
LORA_ADAPTER_NAME = "estilo"


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
    # Agranda la región protegida N px (a target_side) para que el borde editable no toque el
    # producto. Mitigación candidata contra la expansión de silueta; 0 = desactivada.
    mask_dilate_px: int = 0
    # Texto FIJO antepuesto al prompt (p. ej., la categoría del producto). Nunca texto libre
    # del usuario (G5). Mitigación candidata contra la expansión de silueta.
    prompt_prefix: str = ""
    # Peso del LoRA en las solicitudes `advanced` cuando hay un adaptador cargado.
    lora_scale: float = 1.0
    # Versión de los prompts y del negativo (presets.PROMPTS); el LoRA se entrena con una sola.
    prompt_version: str = PROMPT_VERSION
    # Fondo de la ENTRADA antes de generar. Con strength < 1 los latentes parten de la foto y
    # el fondo original (pared, mantel) ancla los colores. "blend" mezcla `prefill_mix` del
    # degradado del preset con el fondo original difuminado (conserva el plano de la mesa);
    # "palette" usa solo el degradado; "" deja la foto tal cual.
    prefill: str = ""
    prefill_blur: int = 24  # px a la resolución de trabajo
    prefill_mix: float = 0.5
    # "preset": textos de presets.py (los del LoRA); "scene": escena por categoría (scenes.py).
    prompt_set: str = "preset"


# Configuración W del experimento con fotos reales del 2026-11-15 (descartada).
REALVIS_INPAINT_PARAMS = SdxlParams(
    model_id=REALVIS_INPAINT_MODEL_ID, strength=0.90, prefill="blend", prompt_set="scene",
)


def prefill_background(image: Image.Image, mask: Image.Image, style: Style, how: str,
                       blur: int = 24, mix: float = 0.5) -> Image.Image:
    """Reemplaza el fondo (blanco en la máscara) por el degradado del preset o una mezcla."""
    if how not in ("blend", "palette"):
        raise ValueError(f"prefill desconocido: {how!r}")
    top, bottom = PRESETS[style].mock_colors
    w, h = image.size
    column = Image.new("RGB", (1, h))
    for y in range(h):
        t = y / max(1, h - 1)
        column.putpixel((0, y), tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)))
    gradient = column.resize((w, h))
    if how == "blend":
        gradient = Image.blend(image.filter(ImageFilter.GaussianBlur(blur)), gradient, mix)
    return Image.composite(gradient, image, mask.convert("L"))


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
        self.lora: dict | None = None  # {"path", "sha256"} del adaptador cargado

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

    def attach_lora(self, path: str | Path) -> dict:
        """Carga un adaptador LoRA (`.safetensors` de diffusers) en el UNet del pipeline."""
        self.load()
        if self.lora is not None:
            raise RuntimeError(f"Ya hay un LoRA cargado: {self.lora['path']}")
        path = Path(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            self._pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=LORA_ADAPTER_NAME)
        except Exception:
            # Una carga a medias puede haber inyectado capas: se revierten antes de propagar.
            try:
                self._pipe.unload_lora_weights()
            except Exception:  # noqa: BLE001 - el error original es el que importa
                log.exception("No se pudo revertir una carga parcial del LoRA")
            raise
        self.lora = {"path": str(path), "sha256": sha}
        log.info("LoRA cargado: %s (%s)", path, sha[:12])
        return self.lora

    def detach_lora(self) -> None:
        """Descarga el adaptador y deja el UNet como el checkpoint original."""
        if self.lora is not None:
            self._pipe.unload_lora_weights()
            self.lora = None

    def _lora_for(self, mode: Mode) -> dict | None:
        """Activa el peso que corresponde al modo y devuelve lo que se registra en la variación."""
        if self.lora is None:
            return None
        scale = self.params.lora_scale if mode == Mode.ADVANCED else 0.0
        self._pipe.set_adapters([LORA_ADAPTER_NAME], adapter_weights=[scale])
        return {**self.lora, "scale": scale}

    # --- GenerationBackend ---

    def generate(self, request: GenerationRequest) -> list[GeneratedArtifact]:
        self.load()
        p = self.params
        work_size = request.image.size
        gen_size = generation_size(work_size, p.target_side)
        work_image = request.image.convert("RGB")
        image = work_image.resize(gen_size, Image.Resampling.LANCZOS)
        mask = request.mask.convert("L").resize(gen_size, Image.Resampling.BILINEAR)
        if p.mask_dilate_px:
            # Blanco = editable: un filtro de mínimo agranda lo negro (protegido).
            mask = mask.filter(ImageFilter.MinFilter(2 * p.mask_dilate_px + 1))
        if p.mask_blur and hasattr(self._pipe, "mask_processor"):
            mask = self._pipe.mask_processor.blur(mask, blur_factor=p.mask_blur)
        lora = self._lora_for(request.mode)

        scene = p.prompt_set == "scene"
        negative = SCENE_NEGATIVE if scene else NEGATIVE_PROMPTS[p.prompt_version]
        prompt_version = SCENE_PROMPT_VERSION if scene else p.prompt_version
        results = []
        for spec in request.specs:
            if scene:
                prompt = build_scene_prompt(spec.style, request.category, spec.adjustments)
            else:
                prompt = build_prompt(spec.style, spec.adjustments, p.prompt_version)
            if p.prompt_prefix:
                prompt = f"{p.prompt_prefix}, {prompt}"
            spec_image = image
            if p.prefill:
                # A la resolución de trabajo, igual que en el experimento; luego al tamaño de SDXL.
                spec_image = prefill_background(work_image, request.mask, spec.style, p.prefill,
                                                p.prefill_blur, p.prefill_mix).resize(gen_size, Image.Resampling.LANCZOS)
            t0 = time.perf_counter()
            _reset_peak_memory()
            out = self._pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=spec_image,
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
                    prompt_version=prompt_version,
                    params={
                        "backend": self.name,
                        "simulated": False,
                        "mode": request.mode.value,
                        "lora": lora,  # None si no hay adaptador; si no, sha256 y peso aplicado
                        "prompt": prompt,
                        "negative_prompt": negative,
                        "prompt_version": prompt_version,
                        "prompt_set": p.prompt_set,
                        "category": request.category.value if request.category else None,
                        "prefill": p.prefill,
                        "adjustments": list(spec.adjustments),
                        "steps": p.steps,
                        "guidance_scale": p.guidance_scale,
                        "strength": p.strength,
                        "mask_blur": p.mask_blur,
                        "mask_dilate_px": p.mask_dilate_px,
                        "prompt_prefix": p.prompt_prefix,
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
