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

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)
