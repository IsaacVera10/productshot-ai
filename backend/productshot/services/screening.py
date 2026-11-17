"""Guardrails con modelos (ADR-008): G2 y G7 (contenido inapropiado) y G3 (rostros).

- G2: la foto subida pasa por el clasificador NSFW antes de crear la sesión; se rechaza con 422.
- G3: un detector clásico de rostros (Haar de OpenCV) rechaza las fotos con una cara
  prominente con 422: la herramienta es para productos y evita manipular la identidad
  visual de personas (decisión del 2026-11-12, `docs/08`).
- G7: cada una de las cinco salidas pasa por el mismo clasificador; si una lo supera se
  regenera con otra semilla hasta `retries` veces, y si sigue fallando el lote entero falla
  (cinco o nada).

Clasificador: `Falconsai/nsfw_image_detection` (ViT-base, Apache-2.0, 343 MB, etiquetas
normal/nsfw). Se eligió frente a `AdamCodd/vit-base-nsfw-detector` (misma arquitectura y
licencia) por uso: ~1.6 M de descargas frente a ~0.7 M. Corre en el worker junto a SDXL
(menos de 0.5 GB de VRAM). `torch`, `transformers` y `cv2` se importan al cargar.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from productshot.services.generation import GeneratedArtifact, GenerationBackend, GenerationRequest

log = logging.getLogger(__name__)

NSFW_MODEL_ID = "Falconsai/nsfw_image_detection"
NSFW_MODEL_REVISION = "96cb0d0342c7"

# G15: tipo de producto con CLIP zero-shot. Criterio: se bloquea lo que Instagram y TikTok no
# permiten anunciar (el nicho vende ahí) y lo que facilita fraude; se avisa lo que está fuera
# del nicho y puede salir mal. Cada grupo suma la probabilidad de sus etiquetas; "producto" son
# anclas del nicho para que una taza o una vela no queden forzadas a elegir un grupo riesgoso.
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
CATEGORY_LABELS: dict[str, list[str]] = {
    "arma": ["a photo of a gun", "a photo of a pistol", "a photo of a rifle", "a photo of bullets"],
    "tabaco_vape": ["a photo of cigarettes", "a photo of a vape", "a photo of an electronic cigarette"],
    "drogas": ["a photo of marijuana", "a photo of illegal drugs", "a photo of drug paraphernalia"],
    "dinero_documentos": ["a photo of banknotes", "a photo of paper money", "a photo of an identity card",
                          "a photo of a passport", "a photo of a credit card"],
    "sexual": ["a photo of a sex toy", "a photo of an adult toy"],
    "comida": ["a photo of a plate of food", "a photo of a meal"],
    "transparente": ["a photo of a transparent glass object", "a photo of a mirror"],
    "grande": ["a photo of a piece of furniture", "a photo of a large appliance"],
    "animal": ["a photo of a live animal", "a photo of a pet"],
    "pantalla_documento": ["a screenshot", "a photo of a computer screen", "a photo of a printed document"],
    "producto": ["a photo of a mug", "a photo of a candle", "a photo of a bar of soap", "a photo of a ceramic vase",
                 "a photo of an electronic gadget", "a photo of a toy", "a photo of a figurine",
                 "a photo of a cosmetic product", "a photo of jewelry", "a photo of a product box",
                 "a photo of a home decoration item", "a photo of a canned drink", "a photo of a jar",
                 "a photo of a circuit board", "a photo of a bag",
                 # Productos comunes que CLIP confundía con «sexual» o «dinero» (evaluation/calibrar_g15.py).
                 "a photo of a video game controller", "a photo of a computer mouse", "a photo of a keyboard",
                 "a photo of headphones", "a photo of a speaker", "a photo of a microphone",
                 "a photo of a remote control", "a photo of a smartphone", "a photo of a ball",
                 "a photo of an umbrella", "a photo of a tool", "a photo of a garden tool",
                 "a photo of a wallet", "a photo of a water bottle", "a photo of a kitchen utensil"],
}
BLOCKED_CATEGORIES = ("arma", "tabaco_vape", "drogas", "dinero_documentos", "sexual")
WARNED_CATEGORIES = ("comida", "transparente", "grande", "animal", "pantalla_documento")


@dataclass(frozen=True)
class ScreeningResult:
    nsfw_score: float  # probabilidad de "nsfw", 0 a 1
    faces: int  # rostros prominentes detectados
    simulated: bool
    seconds: float | None = None
    # G15: probabilidad por grupo de CATEGORY_LABELS (vacío si no hay clasificador).
    categories: dict[str, float] = dataclasses.field(default_factory=dict)


class ScreeningUnavailable(Exception):
    """No se pudo revisar la foto (worker caído, timeout): se falla cerrado."""


class ScreeningService(Protocol):
    name: str

    @property
    def simulated(self) -> bool: ...

    def screen(self, image: Image.Image) -> ScreeningResult: ...


class MockScreeningService:
    """Desarrollo y pruebas: deja pasar todo y lo declara simulado."""

    name = "mock"
    simulated = True

    def screen(self, image: Image.Image) -> ScreeningResult:
        return ScreeningResult(nsfw_score=0.0, faces=0, simulated=True, seconds=0.0)


class LocalScreeningService:
    """Clasificador NSFW y detector de rostros reales (en el worker)."""

    name = "nsfw-vit+haar"
    simulated = False

    def __init__(self, *, face_min_fraction: float = 0.15, device: str = "cuda",
                 classifier: Callable[[Image.Image], float] | None = None,
                 face_counter: Callable[[Image.Image, float], int] | None = None,
                 category_classifier: Callable[[Image.Image], dict[str, float]] | None = None):
        self.face_min_fraction = face_min_fraction
        self._device = device
        self._classifier = classifier
        self._face_counter = face_counter
        self._categories = category_classifier

    def load(self) -> None:
        if self._classifier is None:
            self._classifier = _load_nsfw_classifier(self._device)
        if self._face_counter is None:
            self._face_counter = _load_face_counter()
        if self._categories is None:
            self._categories = _load_category_classifier(self._device)

    def nsfw_score(self, image: Image.Image) -> float:
        self.load()
        return self._classifier(image.convert("RGB"))

    def screen(self, image: Image.Image) -> ScreeningResult:
        self.load()
        t0 = time.perf_counter()
        score = self.nsfw_score(image)
        faces = self._face_counter(image.convert("RGB"), self.face_min_fraction)
        categories = {k: round(v, 4) for k, v in self._categories(image.convert("RGB")).items()}
        return ScreeningResult(nsfw_score=round(score, 4), faces=faces, simulated=False,
                               seconds=round(time.perf_counter() - t0, 2), categories=categories)


class UnsafeOutput(Exception):
    """G7: una variación siguió siendo inapropiada tras los reintentos."""


class SafetyCheckedBackend:
    """G7: envuelve un backend y revisa cada salida con el clasificador NSFW.

    Una salida que supera el umbral se regenera sola, con otra semilla, hasta `retries`
    veces. Si ninguna pasa, el lote falla entero (el orquestador nunca ve cuatro imágenes).
    Cada artefacto registra su puntaje y los reintentos en `params["g7"]`.
    """

    def __init__(self, inner: GenerationBackend, score: Callable[[Image.Image], float], *,
                 threshold: float = 0.85, retries: int = 2, seed_step: int = 100_003):
        self.inner = inner
        self._score = score
        self.threshold = threshold
        self.retries = retries
        self.seed_step = seed_step

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def simulated(self) -> bool:
        return self.inner.simulated

    @property
    def model_revision(self) -> str:
        return self.inner.model_revision

    def __getattr__(self, name: str):
        # El resto (lora, params, attach_lora) es del backend envuelto: el health lo informa.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def load(self) -> None:
        if hasattr(self.inner, "load"):
            self.inner.load()

    def generate(self, request: GenerationRequest) -> list[GeneratedArtifact]:
        artifacts = self.inner.generate(request)
        specs = {s.position: s for s in request.specs}
        checked = []
        for art in artifacts:
            score = self._score(art.image)
            attempts = 0
            while score >= self.threshold and attempts < self.retries:
                attempts += 1
                spec = specs[art.position]
                retry_spec = dataclasses.replace(spec, seed=(spec.seed + attempts * self.seed_step) % 2**32)
                log.warning("G7: variación %d con puntaje %.2f; reintento %d", art.position, score, attempts)
                art = self.inner.generate(dataclasses.replace(request, specs=(retry_spec,)))[0]
                score = self._score(art.image)
            if score >= self.threshold:
                raise UnsafeOutput(f"La variación {art.position} no pasó el filtro de contenido (G7).")
            # La variación conserva la semilla pedida (el orquestador la verifica); la del
            # reintento de G7, si lo hubo, queda registrada.
            g7 = {"nsfw_score": round(score, 4), "retries": attempts, "threshold": self.threshold}
            if attempts:
                g7["retry_seed"] = art.seed
            checked.append(dataclasses.replace(art, seed=specs[art.position].seed, params={**art.params, "g7": g7}))
        return checked


# --- dependencias pesadas (solo en el worker) ---


def _load_nsfw_classifier(device: str) -> Callable[[Image.Image], float]:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(NSFW_MODEL_ID, revision=NSFW_MODEL_REVISION)
    model = AutoModelForImageClassification.from_pretrained(NSFW_MODEL_ID, revision=NSFW_MODEL_REVISION)
    model.to(device).eval()
    nsfw_index = {v: int(k) for k, v in model.config.id2label.items()}["nsfw"]

    def score(image: Image.Image) -> float:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = model(**inputs).logits.softmax(-1)[0]
        return float(probs[nsfw_index])

    return score


def _features(out):
    """transformers 4 devuelve el tensor; transformers 5, un objeto con `pooler_output`."""
    return out if hasattr(out, "norm") else out.pooler_output


def _load_category_classifier(device: str) -> Callable[[Image.Image], dict[str, float]]:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=torch.float16).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    groups = [(g, t) for g, texts in CATEGORY_LABELS.items() for t in texts]
    with torch.no_grad():
        tokens = proc(text=[t for _, t in groups], return_tensors="pt", padding=True).to(device)
        text = _features(model.get_text_features(**tokens))
        text = text / text.norm(dim=-1, keepdim=True)

    def classify(image: Image.Image) -> dict[str, float]:
        with torch.no_grad():
            px = proc(images=image, return_tensors="pt")["pixel_values"].to(device, torch.float16)
            emb = _features(model.get_image_features(pixel_values=px))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            probs = (model.logit_scale.exp() * emb @ text.T).float().softmax(-1)[0].tolist()
        out: dict[str, float] = {}
        for (g, _), p in zip(groups, probs):
            out[g] = out.get(g, 0.0) + p
        return out

    return classify


def _load_face_counter() -> Callable[[Image.Image, float], int]:
    import cv2
    import numpy as np

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("No se pudo cargar el detector de rostros de OpenCV.")

    def count(image: Image.Image, min_fraction: float) -> int:
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        side = max(24, int(min(gray.shape) * min_fraction))
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(side, side))
        return len(faces)

    return count
