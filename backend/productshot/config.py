"""Configuración por entorno (variables de entorno o archivo `.env`)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    sqlite_path: Path = Path("./data/productshot.sqlite3")
    asset_root: Path = Path("./data/assets")

    # En el Hito 1 solo existen los backends simulados. Los reales llegan en los Hitos 2 y 3.
    generation_backend: Literal["mock"] = "mock"
    segmentation_backend: Literal["mock"] = "mock"
    summary_backend: Literal["deterministic"] = "deterministic"

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

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)
