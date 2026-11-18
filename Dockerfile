# Imagen de la API + UI (FastAPI, web app y Gradio) para Railway u otro host de contenedores.
# La generación corre en los workers de Modal; esta imagen no lleva GPU ni modelos.
#
#   docker build -t productshot .
#   docker run --rm -p 8000:8000 --env-file .env -e APP_HOST=0.0.0.0 -v ./data:/data \
#     -e SQLITE_PATH=/data/productshot.sqlite3 -e ASSET_ROOT=/data/assets productshot
#
# Datos persistentes: montar un volumen en /data (SQLite e imágenes).

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False

# Dependencias primero: esta capa solo se rehace si cambian pyproject.toml o uv.lock.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# Código del servidor (backend con la web app estática, y la UI de Gradio).
COPY backend ./backend
COPY ui ./ui
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    APP_HOST=0.0.0.0 \
    SQLITE_PATH=/data/productshot.sqlite3 \
    ASSET_ROOT=/data/assets

EXPOSE 8000

# Railway inyecta PORT; la app lee APP_PORT.
CMD ["sh", "-c", "APP_PORT=${PORT:-8000} exec productshot"]
