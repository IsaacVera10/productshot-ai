"""Punto de entrada: `productshot` o `python -m productshot.main`."""

from __future__ import annotations

import logging

import uvicorn

from productshot.api.app import create_app
from productshot.config import Settings


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)


if __name__ == "__main__":
    run()
