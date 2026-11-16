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
    summary_backend: Literal["deterministic", "gemini"] = "deterministic"
    job_runner: Literal["thread", "inline"] = "thread"

    # Worker de generación por HTTP (GENERATION_BACKEND=remote). Mismo contrato en Modal,
    # Cloud Run y la laptop; el token lo comparten la API y el worker.
    generation_worker_url: str | None = None
    generation_worker_token: SecretStr | None = None
    generation_worker_timeout_s: float = 540.0  # menor que G14 para que falle antes el HTTP
    worker_host: str = "127.0.0.1"
    worker_port: int = 8001

    # Agente de resumen (SUMMARY_BACKEND=gemini). Solo recibe decisiones y motivos.
    gemini_api_key: SecretStr | None = None
    gemini_model_id: str = "gemini-3.8-flash"
    gemini_fallback_model_id: str | None = "gemini-3.5-flash-lite"
    gemini_timeout_s: float = 20.0  # por llamada
    gemini_budget_s: float = 45.0  # todos los intentos juntos; la UI espera 60 s

    max_upload_mb: float = 10.0
    # Lado mayor del lienzo de trabajo. La imagen original se conserva aparte.
    work_max_side: int = 1024
    # Límite de píxeles al decodificar (protección contra "decompression bombs").
    max_image_pixels: int = 40_000_000

    # Guardrails (ADR-008). Umbrales PROVISIONALES: se calibran con los casos del Hito 2.
    quality_min_brightness: float = 45.0  # G12: brillo medio 0 a 255
    quality_max_brightness: float = 230.0
    quality_min_sharpness: float = 25.0  # G12: varianza del laplaciano a 512 px
    mask_min_protected: float = 0.03  # G13: fracción de la foto protegida
    mask_max_protected: float = 0.85
    mask_min_component_fraction: float = 0.01  # G13: tamaño mínimo para contar un objeto
    fidelity_warn_diff: float = 12.0  # G8: diferencia media por canal antes de recomponer
    generation_timeout_s: float = 600.0  # G14: tiempo máximo de un lote
    # Reencuadre del modo avanzado (services/layout.py): alto máximo y base del producto como
    # fracción del lienzo. Con 0 en el alto, el producto queda donde está en la foto.
    advanced_layout_height: float = 0.45
    advanced_layout_bottom: float = 0.85

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)
