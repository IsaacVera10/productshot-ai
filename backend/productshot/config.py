"""Configuración por entorno (variables de entorno o archivo `.env`)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    sqlite_path: Path = Path("./data/productshot.sqlite3")
    asset_root: Path = Path("./data/assets")

    # Piezas intercambiables (ADR-009): `composition.py` elige el adaptador de cada una.
    generation_backend: Literal["mock", "remote"] = "mock"
    segmentation_backend: Literal["mock", "remote"] = "mock"
    # G2 y G3 (clasificador NSFW y rostros) sobre la foto subida; remote = el mismo worker.
    screening_backend: Literal["mock", "remote"] = "mock"
    summary_backend: Literal["deterministic", "gemini"] = "deterministic"
    job_runner: Literal["thread", "inline"] = "thread"

    # Worker de generación por HTTP (GENERATION_BACKEND=remote). Mismo contrato en Modal,
    # Cloud Run y la laptop; el token lo comparten la API y el worker.
    generation_worker_url: str | None = None
    generation_worker_token: SecretStr | None = None
    generation_worker_timeout_s: float = 540.0  # menor que G14 para que falle antes el HTTP
    # Un worker por modo (acordado el 2026-11-16): GENERATION_WORKER_URL sirve el intermedio (y
    # segmenta y revisa fotos); si se define ADVANCED_WORKER_URL, las sesiones avanzadas van ahí.
    advanced_worker_url: str | None = None
    advanced_worker_token: SecretStr | None = None
    worker_host: str = "127.0.0.1"
    worker_port: int = 8001

    # Agente de resumen (SUMMARY_BACKEND=gemini). Solo recibe decisiones y motivos.
    gemini_api_key: SecretStr | None = None
    # 3.5-flash y su lite: los más nuevos (3.7 y 3.8) respondían 503 por saturación el 2026-11-16.
    gemini_model_id: str = "gemini-3.5-flash"
    gemini_fallback_model_id: str | None = "gemini-3.5-flash-lite"
    gemini_timeout_s: float = 20.0  # por llamada
    gemini_budget_s: float = 45.0  # todos los intentos juntos; la UI espera 60 s

    max_upload_mb: float = 10.0
    # Lado mayor del lienzo de trabajo. La imagen original se conserva aparte.
    work_max_side: int = 1024
    # Límite de píxeles al decodificar (protección contra "decompression bombs").
    max_image_pixels: int = 40_000_000

    # Retención (G11): cada cadena de rondas se borra, con sus fotos, cuando su primera ronda
    # cumple este plazo. Con 0 no se borra nada automáticamente.
    session_retention_hours: float = 24.0
    retention_sweep_interval_s: float = 600.0  # cada cuánto se buscan cadenas vencidas

    # Guardrails (ADR-008). Umbrales PROVISIONALES: se calibran con los casos del Hito 2.
    quality_min_brightness: float = 45.0  # G12: brillo medio 0 a 255
    quality_max_brightness: float = 230.0
    quality_min_sharpness: float = 25.0  # G12: varianza del laplaciano a 512 px
    mask_min_protected: float = 0.03  # G13: fracción de la foto protegida
    mask_max_protected: float = 0.85
    mask_min_component_fraction: float = 0.01  # G13: tamaño mínimo para contar un objeto
    nsfw_threshold: float = 0.85  # G2 (entrada) y G7 (salida, en el worker): probabilidad 0 a 1
    # G3: lado mínimo del rostro como fracción del lado menor de la foto. Con 0.08 la taza
    # estampada con personajes de anime daba 2 "rostros" (8 % y 12 %, medido el 2026-11-15):
    # sería rechazar el producto típico del nicho. 0.15 deja pasar estampados y rechaza a una
    # persona en primer plano.
    face_min_fraction: float = 0.15
    # G15 (CLIP, tipo de producto). Calibrado el 2026-11-16 con 22 fotos reales: los productos
    # del nicho no pasan de 0.04 en ningún grupo bloqueado; el vape da 0.99 y el juguete sexual 0.97.
    category_block_threshold: float = 0.5
    category_warn_threshold: float = 0.5
    fidelity_warn_diff: float = 12.0  # G8: diferencia media por canal antes de recomponer
    generation_timeout_s: float = 600.0  # G14: tiempo máximo de un lote
    # Reencuadre del modo avanzado (services/layout.py): alto máximo y base del producto como
    # fracción del lienzo. Con 0 en el alto, el producto queda donde está en la foto.
    advanced_layout_height: float = 0.45
    advanced_layout_bottom: float = 0.85

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)
