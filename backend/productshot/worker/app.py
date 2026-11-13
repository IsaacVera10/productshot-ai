"""Worker de generación: envuelve cualquier `GenerationBackend` detrás del contrato HTTP.

Hoy envuelve el mock, que basta para probar el camino completo API → HTTP → worker sin
GPU. En el Hito 2 envuelve el backend de SDXL; el despliegue de Modal y el de la laptop
importan esta misma aplicación.

Uso local: `productshot-worker` (lee `WORKER_HOST`, `WORKER_PORT` y
`GENERATION_WORKER_TOKEN` del entorno o de `.env`).
"""

from __future__ import annotations

import hmac
import logging

from fastapi import FastAPI, Header, HTTPException

from productshot.services.generation import GenerationBackend, GenerationRequest, VariationSpec
from productshot.worker.contract import (
    CONTRACT_VERSION,
    GENERATE_PATH,
    HEALTH_PATH,
    ArtifactOut,
    ContractError,
    GenerateIn,
    GenerateOut,
    HealthOut,
    decode_png,
    encode_png,
)

log = logging.getLogger(__name__)


def create_worker_app(backend: GenerationBackend, token: str | None = None) -> FastAPI:
    app = FastAPI(title="ProductShot AI worker", version=CONTRACT_VERSION, docs_url=None, redoc_url=None)

    def _check_auth(authorization: str | None) -> None:
        if not token:
            return
        expected = f"Bearer {token}"
        if authorization is None or not hmac.compare_digest(authorization.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="Token del worker inválido.")

    def _health() -> HealthOut:
        return HealthOut(backend=backend.name, model_revision=backend.model_revision, simulated=backend.simulated)

    @app.get(HEALTH_PATH, response_model=HealthOut)
    def health(authorization: str | None = Header(default=None)) -> HealthOut:
        _check_auth(authorization)
        return _health()

    @app.post(GENERATE_PATH, response_model=GenerateOut)
    def generate(body: GenerateIn, authorization: str | None = Header(default=None)) -> GenerateOut:
        _check_auth(authorization)
        if body.contract_version != CONTRACT_VERSION:
            raise HTTPException(status_code=409, detail=f"Versión de contrato no soportada: {body.contract_version}.")
        try:
            image = decode_png(body.image_png_b64, "RGB")
            mask = decode_png(body.mask_png_b64, "L")
        except ContractError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if image.size != mask.size:
            raise HTTPException(status_code=422, detail="La foto y la máscara no tienen el mismo tamaño.")
        specs = tuple(
            VariationSpec(position=s.position, style=s.style, seed=s.seed, adjustments=tuple(s.adjustments))
            for s in body.specs
        )
        request = GenerationRequest(
            session_id=body.session_id, mode=body.mode, image=image, mask=mask, specs=specs,
            prompt_version=body.prompt_version,
        )
        artifacts = backend.generate(request)
        log.info("Lote generado: sesión %s, %d imágenes, backend %s", body.session_id, len(artifacts), backend.name)
        info = _health()
        return GenerateOut(
            backend=info.backend,
            model_revision=info.model_revision,
            simulated=info.simulated,
            artifacts=[
                ArtifactOut(
                    position=a.position, style=a.style, seed=a.seed, variant=a.variant,
                    image_png_b64=encode_png(a.image), model_revision=a.model_revision,
                    prompt_version=a.prompt_version, params=a.params,
                )
                for a in artifacts
            ],
        )

    return app


def run() -> None:
    """Worker local (laptop). En Modal se importa `create_worker_app` desde su propio archivo."""
    import os

    import uvicorn

    from productshot.config import Settings
    from productshot.services.generation import MockGenerationBackend

    def sdxl():
        # Necesita torch y diffusers con CUDA (laptop con la RTX 4060 o cualquier GPU).
        from productshot.services.sdxl_generation import SdxlInpaintBackend

        backend = SdxlInpaintBackend()
        backend.load()
        return backend

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    backends = {"mock": MockGenerationBackend, "sdxl": sdxl}
    name = os.environ.get("WORKER_BACKEND", "mock")
    if name not in backends:
        raise SystemExit(f"WORKER_BACKEND desconocido: {name!r}. Disponibles: {sorted(backends)}.")
    token = settings.generation_worker_token.get_secret_value() if settings.generation_worker_token else None
    app = create_worker_app(backends[name](), token=token)
    uvicorn.run(app, host=settings.worker_host, port=settings.worker_port)


if __name__ == "__main__":
    run()
