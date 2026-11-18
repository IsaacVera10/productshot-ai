# Backend

Paquete `productshot`. Contiene la API, el orquestador de sesiones, los backends de generación, el agente de resumen, el registro de decisiones y el worker de generación. La forma de ejecutarlo está en el [README principal](../README.md).

## Módulos

| Módulo | Responsabilidad |
|---|---|
| `domain.py` | Estados de sesión y transiciones válidas, estilos, categorías, entidades y errores de dominio |
| `config.py` | Configuración (`Settings`) leída del entorno o de `.env`, incluidos los umbrales de los guardrails |
| `composition.py` | Raíz de composición: es el único módulo que conoce las clases concretas y elige un adaptador por pieza |
| `main.py` | Punto de entrada `productshot` (uvicorn) |
| `api/` | Rutas `/api/v1` (`app.py`) y esquemas Pydantic (`schemas.py`). Monta además la web app en `/app` y Gradio en `/ui` |
| `services/sessions.py` | Orquestador: imagen → máscara → cinco variaciones → decisiones → resumen → siguiente ronda. Valida el lote de cinco y cada transición de estado. También borra las cadenas de rondas, a pedido o al vencer el plazo de retención (`purge_expired`, que la API ejecuta al arrancar y cada 10 minutos) |
| `services/generation.py` | Interfaz `GenerationBackend` y backend simulado |
| `services/qwen_edit.py`, `scenes.py`, `fidelity.py` | Modo intermedio: editor Qwen-Image, prompts de escena por estilo y categoría, y verificación de fidelidad de cada candidata |
| `services/sdxl_generation.py`, `layout.py` | Modo avanzado: inpainting SDXL con LoRA y reencuadre del producto |
| `services/framing.py`, `images.py`, `presets.py` | Validación de la foto, lienzo de trabajo, encuadre de catálogo y presets de los cinco estilos |
| `services/segmentation.py`, `rembg_segmentation.py` | Interfaz de segmentación, mock y BiRefNet (rembg) |
| `services/screening.py` | Guardrails con modelos: G2 (NSFW de entrada), G3 (rostros), G7 (NSFW de salida) y G15 (categoría permitida) |
| `services/guardrails.py` | Guardrails sin modelos: G8 (fidelidad), G12 (calidad de la foto), G13 (máscara) y el formato de los eventos en el log |
| `services/summary.py` | Interfaz `SummaryService`, validador del resumen estructurado contra el registro y resumen determinista de respaldo |
| `services/gemini_summary.py` | Agente de resumen con Gemini: modelo principal y de respaldo, y validación antes de mostrar |
| `services/rounds.py` | Planificador determinista de las rondas 2 y 3 sobre un vocabulario cerrado de ajustes |
| `services/remote_*.py` | Adaptadores HTTP de generación, segmentación y revisión que llaman al worker |
| `services/jobs.py` | Ejecutores de trabajos: hilo (`thread`) o en línea (`inline`, para las pruebas) |
| `services/labels.py` | Franja «SIMULACIÓN» en los resultados del mock |
| `storage/ports.py` | Interfaz `Repository` |
| `storage/repository.py` | Repositorio SQLite (sesiones, imágenes, máscaras, trabajos, variaciones, decisiones y resúmenes) con transiciones atómicas y migraciones de esquema |
| `storage/artifacts.py` | Almacenamiento de imágenes en una carpeta local, protegido contra path traversal |
| `worker/contract.py` | Contrato HTTP API ↔ worker: `/v1/health`, `/v1/generate`, `/v1/segment` y `/v1/screen` |
| `worker/app.py` | Servidor del worker: envuelve cualquier `GenerationBackend`. `productshot-worker` lo levanta con el mock; Modal lo usa con los modelos reales |
| `web/` | Web app en HTML, CSS y JS, sin dependencias |

## Piezas intercambiables

| Pieza | Variable | Adaptadores |
|---|---|---|
| Generación | `GENERATION_BACKEND` | `mock`, `remote` |
| Segmentación | `SEGMENTATION_BACKEND` | `mock`, `remote` |
| Revisión de la foto (G2, G3, G15) | `SCREENING_BACKEND` | `mock`, `remote` |
| Resumen | `SUMMARY_BACKEND` | `deterministic`, `gemini` |
| Ejecutor de trabajos | `JOB_RUNNER` | `thread`, `inline` |

Con `GENERATION_BACKEND=remote`, las sesiones del modo intermedio van a `GENERATION_WORKER_URL` y las del avanzado a `ADVANCED_WORKER_URL`.

Para agregar un adaptador nuevo:

1. Implementar la interfaz correspondiente (`GenerationBackend`, `SegmentationService`, `ScreeningService`, `SummaryService`, `Repository` o `ArtifactStore`).
2. Registrarlo en `composition.py`.
3. Agregarlo a las pruebas parametrizadas de `tests/test_ports.py`.

El orquestador valida lo que devuelve cualquier backend: lote de cinco, fidelidad (G8) y tiempo máximo (G14). Un adaptador nuevo no puede saltarse esas reglas.

## Worker local sin GPU

Para probar el camino API → HTTP → worker con el mock:

```bash
uv run productshot-worker                                                    # http://127.0.0.1:8001
GENERATION_BACKEND=remote GENERATION_WORKER_URL=http://127.0.0.1:8001 uv run productshot
```

Los workers con los modelos reales están en [`deploy/modal/`](../deploy/modal/).
