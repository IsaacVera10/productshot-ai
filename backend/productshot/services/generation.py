"""Backends de generación de las cinco variaciones.

El orquestador (services/sessions.py) es quien valida que el lote tenga exactamente
cinco variaciones con los cinco estilos canónicos; un backend defectuoso no puede
dejar la sesión en READY_FOR_REVIEW.
"""

from __future__ import annotations

import colorsys
import random
from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image, ImageDraw

from productshot.domain import Mode, Style
from productshot.services.presets import NEGATIVE_PROMPT, PRESETS, PROMPT_VERSION, PROMPTS
from productshot.services.rounds import prompt_fragments, variant_key


@dataclass(frozen=True)
class VariationSpec:
    position: int
    style: Style
    seed: int
    # Ajustes del vocabulario cerrado (rondas 2 y 3). En la ronda 1 está vacío.
    adjustments: tuple[str, ...] = ()

    @property
    def variant(self) -> str:
        return variant_key(self.style, self.adjustments)


def build_prompt(style: Style, adjustments: tuple[str, ...], version: str = PROMPT_VERSION) -> str:
    """Prompt del preset más los fragmentos FIJOS de cada ajuste. Ningún texto del usuario entra aquí."""
    return ", ".join([PROMPTS[version][style], *prompt_fragments(adjustments)])


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
    variant: str = ""


class GenerationUnavailable(Exception):
    """El servicio de generación no respondió o respondió mal. El mensaje se muestra a la persona."""


class GenerationBackend(Protocol):
    name: str
    model_revision: str
    simulated: bool

    def generate(self, request: GenerationRequest) -> list[GeneratedArtifact]: ...


def _shift(rgb: tuple[int, int, int], adjustments: tuple[str, ...]) -> tuple[int, int, int]:
    """Versión mock de cada ajuste de color: basta para que la variante se vea distinta."""
    r, g, b = (c / 255 for c in rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if "brighter" in adjustments:
        l = l + (1 - l) * 0.45
    if "darker" in adjustments:
        l = l * 0.6
    if "more_saturated" in adjustments:
        s = min(1.0, s * 1.6 + 0.15)
    if "less_saturated" in adjustments:
        s = s * 0.35
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    out = [r * 255, g * 255, b * 255]
    if "warmer" in adjustments:
        out = [out[0] + 28, out[1] + 6, out[2] - 26]
    if "cooler" in adjustments:
        out = [out[0] - 24, out[1] + 2, out[2] + 30]
    return tuple(max(0, min(255, int(c))) for c in out)


_SURFACES = {"wood_surface": (120, 82, 50, 220), "stone_surface": (128, 128, 132, 220), "fabric_surface": (214, 196, 170, 220)}


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
        adj = spec.adjustments
        top, bottom = (_shift(c, adj) for c in preset.mock_colors)
        if "outdoor" in adj:
            top = _shift((170, 205, 235), adj)  # cielo
        bg = _gradient(size, top, bottom)
        rng = random.Random(spec.seed)
        draw = ImageDraw.Draw(bg, "RGBA")
        w, h = size
        surface = next((_SURFACES[a] for a in adj if a in _SURFACES), None)
        if surface or spec.style == Style.LIFESTYLE:
            # "Mesa": franja inferior.
            draw.rectangle([0, int(h * 0.68), w, h], fill=surface or (110, 78, 56, 200))
        quiet = spec.style == Style.MINIMAL or "simpler_background" in adj or "fewer_props" in adj
        if not quiet:
            # Ruido suave dependiente de la semilla para que la semilla tenga efecto visible.
            for _ in range(12):
                x, y, r = rng.randrange(w), rng.randrange(h), rng.randrange(w // 20 + 1, w // 6 + 2)
                draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 28))
        if "more_props" in adj:
            # "Utilería": bloques en los bordes, lejos del producto.
            for x0 in (0.04, 0.82):
                bw, bh = int(w * 0.12), int(h * rng.uniform(0.12, 0.22))
                x = int(w * x0)
                draw.rectangle([x, int(h * 0.68) - bh, x + bw, int(h * 0.68)], fill=(90, 110, 140, 200))
        return bg

    def generate(self, request: GenerationRequest) -> list[GeneratedArtifact]:
        results = []
        for spec in request.specs:
            bg = self._background(spec, request.image.size)
            # Blanco en la máscara = fondo nuevo; negro = píxeles originales del producto.
            composed = Image.composite(bg, request.image.convert("RGB"), request.mask)
            # El rótulo «SIMULACIÓN» lo pone el orquestador después de recomponer (G8, G9).
            results.append(
                GeneratedArtifact(
                    position=spec.position,
                    style=spec.style,
                    image=composed,
                    seed=spec.seed,
                    model_revision=self.model_revision,
                    prompt_version=request.prompt_version,
                    params={
                        "backend": self.name,
                        "simulated": True,
                        "prompt": build_prompt(spec.style, spec.adjustments),
                        "negative_prompt": NEGATIVE_PROMPT,
                        "adjustments": list(spec.adjustments),
                    },
                    variant=spec.variant,
                )
            )
        return results
