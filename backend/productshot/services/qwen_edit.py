"""Bloque del modo intermedio: editor por instrucciones con candidatas verificadas.

Entrada y salida iguales a las de cualquier `GenerationBackend`: foto de trabajo + máscara
(blanco = fondo, negro = producto) + cinco especificaciones de estilo → cinco imágenes del
mismo tamaño que la foto.

Cómo genera (experimentos del 2026-11-16, `deploy/modal/edit_lab*.py`):

1. Qwen-Image-2.1-Turbo (8 pasos) re-dibuja la foto completa a partir de una instrucción
   (`scenes.build_edit_prompt`): escena curada según el estilo y la categoría, y la orden de
   conservar el producto, su orientación y todo su texto. A diferencia del inpainting, re-ilumina
   el producto con la luz de la escena, así que no se ve "pegado".
2. Por estilo genera `candidates` versiones con semillas distintas y verifica cada una contra la
   foto original (`fidelity.verify`: alineación ORB y bordes dentro del producto, más cambio de
   escena). Se queda con la mejor que pasa.
3. Si un estilo no tiene ninguna candidata válida, genera más, una por vez,
   mientras quepa en el presupuesto de tiempo (`budget_s`) según la latencia p95 observada.
   Si aun así falta, el lote falla entero ("cinco o nada").

G8 en este modo es "verificado", no "recompuesto": re-pegar los píxeles originales arruinaría la
re-iluminación. Cada imagen registra el método y las métricas en `params["fidelity"]`.

Licencia del modelo: Qwen Research License (uso de investigación y evaluación, no comercial).
`torch` y `diffusers` se importan al cargar; `pipeline_factory` y `verifier` permiten probar
todo sin GPU.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable

from PIL import Image

from productshot.services import fidelity
from productshot.services.generation import (
    GeneratedArtifact,
    GenerationRequest,
    GenerationUnavailable,
    VariationSpec,
)
from productshot.services.scenes import EDIT_PROMPT_VERSION, build_edit_prompt

log = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen-Image-2.1-Turbo"
STEPS = 8
SEED_STEP = 100_003  # separa las semillas de las candidatas de un mismo estilo


class UnverifiedBatch(GenerationUnavailable):
    """Algún estilo no tuvo ninguna candidata que conserve el producto dentro del tiempo."""


def edit_size(size: tuple[int, int], megapixels: float = 1.0) -> tuple[int, int]:
    """~1 MP con la proporción de la foto, lados múltiplos de 16."""
    w, h = size
    scale = math.sqrt(megapixels * 1024 * 1024 / (w * h))
    return max(16, round(w * scale / 16) * 16), max(16, round(h * scale / 16) * 16)


def _p95(values: list[float]) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)]


class QwenEditBackend:
    name = "qwen-edit"
    simulated = False

    def __init__(self, *, model_id: str = MODEL_ID, candidates: int = 3, budget_s: float = 420.0,
                 thresholds: fidelity.FidelityThresholds | None = None,
                 pipeline_factory: Callable | None = None,
                 verifier: Callable[[Image.Image, Image.Image, Image.Image], dict] | None = None,
                 clock: Callable[[], float] = time.monotonic, device: str = "cuda"):
        self.model_id = model_id
        self.candidates = candidates
        self.budget_s = budget_s
        self.thresholds = thresholds or fidelity.FidelityThresholds()
        self._factory = pipeline_factory
        self._verify = verifier or fidelity.verify
        self._clock = clock
        self._device = device
        self._pipe = None
        self._revision: str | None = None
        self.load_seconds: float | None = None
        self.latencies: list[float] = []  # de todas las llamadas, para estimar la p95

    @property
    def model_revision(self) -> str:
        return f"{self.model_id}@{self._revision or 'main'}"

    def load(self) -> None:
        if self._pipe is not None:
            return
        t0 = time.perf_counter()
        if self._factory is not None:
            self._pipe, self._revision = self._factory(self.model_id)
        else:
            self._pipe, self._revision = _load_qwen(self.model_id, self._device)
        self.load_seconds = round(time.perf_counter() - t0, 1)
        log.info("Editor cargado en %.1f s: %s", self.load_seconds, self.model_revision)

    # --- generación y verificación de una candidata ---

    def _candidate(self, request: GenerationRequest, spec: VariationSpec, attempt: int, src: Image.Image,
                   prompt: str) -> dict:
        seed = (spec.seed + attempt * SEED_STEP) % 2**32
        t0 = time.perf_counter()
        out = self._pipe(prompt=prompt, image=src, num_inference_steps=STEPS,
                         generator=_generator(self._device, seed, self._factory is not None)).images[0]
        out = out.convert("RGB").resize(request.image.size, Image.Resampling.LANCZOS)
        seconds = round(time.perf_counter() - t0, 2)
        self.latencies.append(seconds)
        metrics = self._verify(request.image, out, request.mask)
        return {"image": out, "seed": seed, "seconds": seconds, "metrics": metrics,
                "passed": fidelity.passes(metrics, self.thresholds)}

    def generate(self, request: GenerationRequest) -> list[GeneratedArtifact]:
        self.load()
        start = self._clock()
        src = request.image.convert("RGB").resize(edit_size(request.image.size), Image.Resampling.LANCZOS)
        prompts = {s.position: build_edit_prompt(s.style, request.category, s.adjustments) for s in request.specs}
        tried: dict[int, list[dict]] = {s.position: [] for s in request.specs}

        def fits() -> bool:
            """Otra generación solo si cabe en el presupuesto según la p95 observada (G14)."""
            return not self.latencies or self._clock() - start + _p95(self.latencies) <= self.budget_s

        # 1) Ronda base: `candidates` por estilo (el caso típico termina aquí).
        for attempt in range(self.candidates):
            for spec in request.specs:
                if not fits():
                    break
                tried[spec.position].append(self._candidate(request, spec, attempt, src, prompts[spec.position]))

        # 2) Reintentos solo para los estilos sin candidata válida, mientras quepan en el tiempo.
        retries = {s.position: 0 for s in request.specs}
        pending = [s for s in request.specs if not any(c["passed"] for c in tried[s.position])]
        while pending:
            if not fits():
                break
            spec = pending[0]
            attempt = len(tried[spec.position])
            cand = self._candidate(request, spec, attempt, src, prompts[spec.position])
            tried[spec.position].append(cand)
            retries[spec.position] += 1
            if cand["passed"]:
                pending.pop(0)
            else:
                pending.append(pending.pop(0))  # turno rotativo entre los estilos pendientes

        missing = [s.style.value for s in request.specs if not any(c["passed"] for c in tried[s.position])]
        if missing:
            log.warning("Lote sin verificar en %s tras %.0f s", missing, self._clock() - start)
            raise UnverifiedBatch(
                "No pudimos generar fotos que conserven tu producto en: " + ", ".join(missing)
                + ". Prueba con otra foto: el producto completo, bien iluminado y sin cables sueltos.")

        results = []
        for spec in request.specs:
            ok = [c for c in tried[spec.position] if c["passed"]]
            best = max(ok, key=lambda c: fidelity.score(c["metrics"]))
            results.append(GeneratedArtifact(
                # La variación se identifica por la semilla PEDIDA (el orquestador la verifica);
                # la de la candidata elegida queda en `candidate_seed`.
                position=spec.position, style=spec.style, image=best["image"], seed=spec.seed,
                model_revision=self.model_revision, prompt_version=EDIT_PROMPT_VERSION, variant=spec.variant,
                params={
                    "backend": self.name, "simulated": False, "mode": request.mode.value,
                    "category": request.category.value if request.category else None,
                    "prompt": prompts[spec.position], "adjustments": list(spec.adjustments),
                    "steps": STEPS, "seconds": best["seconds"], "candidate_seed": best["seed"],
                    "fidelity": {
                        "method": "verificado", **best["metrics"],
                        "candidates": len(tried[spec.position]), "retries": retries[spec.position],
                        "passed_candidates": len(ok),
                        "thresholds": {"inlier_ratio": self.thresholds.min_inlier_ratio,
                                       "edges": self.thresholds.min_edges,
                                       "scene_change": self.thresholds.min_scene_change},
                    },
                },
            ))
        log.info("Lote verificado en %.0f s (%d generaciones)", self._clock() - start,
                 sum(len(v) for v in tried.values()))
        return results


# --- dependencias pesadas (solo en el worker con GPU) ---


def _load_qwen(model_id: str, device: str):
    import torch
    from diffusers import QwenImage21Pipeline
    from huggingface_hub import HfApi

    sha = HfApi().model_info(model_id).sha
    pipe = QwenImage21Pipeline.from_pretrained(model_id, revision=sha, torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, sha


def _generator(device: str, seed: int, fake: bool):
    if fake:
        return seed
    import torch

    return torch.Generator(device=device).manual_seed(seed)
