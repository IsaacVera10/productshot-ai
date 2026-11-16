"""Contrato HTTP entre la API y el worker de generación (ADR-009).

El mismo contrato sirve en Modal, en Cloud Run y en la laptop: lo único que cambia es la
URL. La API manda la foto de trabajo, la máscara y las cinco especificaciones; el worker
devuelve las cinco imágenes. El worker no guarda nada ni conoce la base de datos, así que
repetir una solicitud (reintento de `spawn` o de Cloud Tasks) es inofensivo: con la misma
semilla produce el mismo lote, y la API solo confirma el primero.

Las imágenes viajan como PNG en base64 dentro del JSON. Para cinco variaciones de 1024 px
son unos pocos MB por lote, aceptable para el volumen del proyecto.

Rutas:
- `GET  /v1/health`   → `HealthOut`
- `POST /v1/generate` → `GenerateOut` (cuerpo `GenerateIn`)
- `POST /v1/segment`  → `SegmentOut` (cuerpo `SegmentIn`); 404 si el worker no segmenta
- `POST /v1/screen`   → `ScreenOut` (cuerpo `ScreenIn`): G2 y G3; 404 si el worker no revisa

Autenticación: `Authorization: Bearer <GENERATION_WORKER_TOKEN>` si el worker tiene token.
"""

from __future__ import annotations

import base64
import binascii
import io

from PIL import Image
from pydantic import BaseModel, Field

from productshot.domain import VARIATIONS_PER_BATCH, Category, Mode, Style

CONTRACT_VERSION = "1"
HEALTH_PATH = "/v1/health"
GENERATE_PATH = "/v1/generate"
SEGMENT_PATH = "/v1/segment"
SCREEN_PATH = "/v1/screen"
# Tope de decodificación para las imágenes que cruzan el contrato (en ambos sentidos).
MAX_CONTRACT_PIXELS = 4096 * 4096


class ContractError(ValueError):
    """Una imagen o un campo del contrato no es válido."""


class SpecIn(BaseModel):
    position: int = Field(ge=1, le=VARIATIONS_PER_BATCH)
    style: Style
    seed: int = Field(ge=0, le=2**32 - 1)
    adjustments: list[str] = Field(default_factory=list, max_length=4)


class GenerateIn(BaseModel):
    contract_version: str = CONTRACT_VERSION
    session_id: str = Field(max_length=64)
    mode: Mode
    prompt_version: str = Field(max_length=64)
    image_png_b64: str
    mask_png_b64: str
    specs: list[SpecIn] = Field(min_length=VARIATIONS_PER_BATCH, max_length=VARIATIONS_PER_BATCH)
    # Opcional (compatible con clientes anteriores): tipo de producto, lista cerrada.
    category: Category | None = None


class ArtifactOut(BaseModel):
    position: int
    style: Style
    seed: int
    variant: str = ""
    image_png_b64: str
    model_revision: str
    prompt_version: str
    params: dict = Field(default_factory=dict)


class GenerateOut(BaseModel):
    contract_version: str = CONTRACT_VERSION
    backend: str
    model_revision: str
    simulated: bool
    artifacts: list[ArtifactOut]


class HealthOut(BaseModel):
    contract_version: str = CONTRACT_VERSION
    backend: str
    model_revision: str
    simulated: bool
    segmenter: str | None = None  # None: el worker no ofrece /v1/segment
    segmenter_simulated: bool | None = None
    screener: str | None = None  # None: el worker no ofrece /v1/screen


class SegmentIn(BaseModel):
    contract_version: str = CONTRACT_VERSION
    image_png_b64: str


class SegmentOut(BaseModel):
    contract_version: str = CONTRACT_VERSION
    segmenter: str
    simulated: bool
    mask_png_b64: str  # "L": blanco = fondo editable, negro = producto
    seconds: float | None = None


class ScreenIn(BaseModel):
    contract_version: str = CONTRACT_VERSION
    image_png_b64: str


class ScreenOut(BaseModel):
    contract_version: str = CONTRACT_VERSION
    screener: str
    simulated: bool
    nsfw_score: float = Field(ge=0, le=1)
    faces: int = Field(ge=0)
    seconds: float | None = None
    categories: dict[str, float] = Field(default_factory=dict)  # G15, opcional


def encode_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_png(data: str, mode: str) -> Image.Image:
    """Decodifica una imagen del contrato. Solo PNG y con tope de píxeles."""
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ContractError("Imagen en base64 inválida.") from exc
    try:
        with Image.open(io.BytesIO(raw)) as img:
            if img.format != "PNG":
                raise ContractError("La imagen del contrato debe ser PNG.")
            if img.width * img.height > MAX_CONTRACT_PIXELS:
                raise ContractError("La imagen del contrato es demasiado grande.")
            img.load()
            return img.convert(mode)
    except ContractError:
        raise
    except Exception as exc:  # Pillow lanza varios tipos según el daño del archivo
        raise ContractError("No se pudo decodificar la imagen del contrato.") from exc
