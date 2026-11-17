"""Calienta el worker del modo intermedio antes de cada participante (arranque en frío ~1 min).

    uv run python evaluation/prueba_usuarios/calentar.py

Pide `/v1/health` al worker configurado en `.env` hasta que responda. El contenedor se apaga
solo a los 2 minutos sin solicitudes (`scaledown_window`), así que conviene correrlo justo
antes de empezar la sesión: mantenerlo encendido todo el día costaría varios dólares.
"""

from __future__ import annotations

import time

import httpx

from productshot.config import Settings
from productshot.worker.contract import HEALTH_PATH


def main() -> None:
    s = Settings()
    if not s.generation_worker_url:
        raise SystemExit("Falta GENERATION_WORKER_URL en .env.")
    headers = {}
    if s.generation_worker_token:
        headers["Authorization"] = f"Bearer {s.generation_worker_token.get_secret_value()}"
    t0 = time.perf_counter()
    deadline = t0 + 300
    while True:
        try:
            r = httpx.get(s.generation_worker_url.rstrip("/") + HEALTH_PATH, headers=headers, timeout=150,
                          follow_redirects=True)
            if r.status_code == 200:
                info = r.json()
                print(f"Worker listo en {time.perf_counter() - t0:.0f} s: {info.get('backend')} · "
                      f"{info.get('model_revision')} · revisor {info.get('screener')}")
                return
            print(f"Respondió {r.status_code}; reintento…")
        except httpx.HTTPError as exc:
            print(f"Todavía arrancando ({type(exc).__name__}); reintento…")
        if time.perf_counter() > deadline:
            raise SystemExit("El worker no respondió en 5 minutos: revisa el panel de Modal.")
        time.sleep(5)


if __name__ == "__main__":
    main()
