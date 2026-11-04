"""Backends de generación de las cinco variaciones.

El orquestador (services/sessions.py) es quien valida que el lote tenga exactamente
cinco variaciones con los cinco estilos canónicos; un backend defectuoso no puede
dejar la sesión en READY_FOR_REVIEW.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image, ImageDraw

from productshot.domain import Mode, Style
from productshot.services.labels import stamp
from productshot.services.presets import NEGATIVE_PROMPT, PRESETS, PROMPT_VERSION


@dataclass(frozen=True)
class VariationSpec:
    position: int
    style: Style
    seed: int


@dataclass(frozen=True)
class GenerationRequest:
    session_id: str
    mode: Mode
    image: Image.Image  # lienzo de trabajo RGB
    mask: Image.Image  # "L": blanco editable, negro protegido
    specs: tuple[VariationSpec, ...]
    prompt_version: str = PROMPT_VERSION


@dataclass(frozen=True)
class GeneratedArtifact:
    position: int
    style: Style
    image: Image.Image
    seed: int
    model_revision: str
    prompt_version: str
    params: dict = field(default_factory=dict)


class GenerationBackend(Protocol):
    name: str
    model_revision: str
    simulated: bool

    def generate(self, request: GenerationRequest) -> list[GeneratedArtifact]: ...


def _gradient(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    w, h = size
    column = Image.new("RGB", (1, h))
    for y in range(h):
        t = y / max(1, h - 1)
        column.putpixel((0, y), tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)))
    return column.resize((w, h))


class MockGenerationBackend:
    """Compone el producto sobre fondos planos por estilo. No usa ningún modelo generativo."""

    name = "mock"
    model_revision = "mock-v1"
    simulated = True

    def _background(self, spec: VariationSpec, size: tuple[int, int]) -> Image.Image:
        preset = PRESETS[spec.style]
        bg = _gradient(size, *preset.mock_colors)
        rng = random.Random(spec.seed)
        draw = ImageDraw.Draw(bg, "RGBA")
        w, h = size
        if spec.style == Style.LIFESTYLE:
            # "Mesa": franja inferior más oscura.
            draw.rectangle([0, int(h * 0.68), w, h], fill=(110, 78, 56, 200))
        if spec.style != Style.MINIMAL:
            # Ruido suave dependiente de la semilla para que la semilla tenga efecto visible.
            for _ in range(12):
                x, y, r = rng.randrange(w), rng.randrange(h), rng.randrange(w // 20 + 1, w // 6 + 2)
                draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 28))
        return bg

    def generate(self, request: GenerationRequest) -> list[GeneratedArtifact]:
        results = []
        for spec in request.specs:
            bg = self._background(spec, request.image.size)
            # Blanco en la máscara = fondo nuevo; negro = píxeles originales del producto.
            composed = Image.composite(bg, request.image.convert("RGB"), request.mask)
            labeled = stamp(composed, bottom=f"{spec.position}/5 · {spec.style.value} · modo {request.mode.value}")
            results.append(
                GeneratedArtifact(
                    position=spec.position,
                    style=spec.style,
                    image=labeled,
                    seed=spec.seed,
                    model_revision=self.model_revision,
                    prompt_version=request.prompt_version,
                    params={
                        "backend": self.name,
                        "simulated": True,
                        "prompt": PRESETS[spec.style].prompt,
                        "negative_prompt": NEGATIVE_PROMPT,
                    },
                )
            )
        return results
