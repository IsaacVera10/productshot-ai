# Pruebas

```bash
uv sync
uv run pytest
```

Las pruebas no usan red, GPU ni credenciales:

- Cada prueba usa su propia base de datos y su propia carpeta de imágenes temporales.
- Los modelos (Qwen, SDXL, rembg, clasificadores y Gemini) se reemplazan por dobles que registran cómo se los llama.
- La generación corre en línea (`InlineJobRunner`) para que el resultado sea determinista. La excepción es una prueba que usa el ejecutor en hilo real.

`uv sync` instala también los grupos `dev` (pytest) y `dataset` (numpy, pyarrow y scipy), que necesitan las pruebas de `training/`.

En Windows:

- Si los permisos sobre `%TEMP%` están restringidos, usar `uv run pytest --basetemp=.pytest_tmp`.
- Las pruebas del driver de Colab que ejecutan su CLI falso se omiten, porque ese driver solo corre en Linux o macOS.

Prueba opcional contra Gemini real (gasta cuota y necesita `GEMINI_API_KEY`): `RUN_LIVE_GEMINI=1 uv run pytest -k live`.

## Sistema (API, flujo y registro)

| Archivo | Cubre |
|---|---|
| `test_hito1_api.py` | Flujo completo por la API con backends mock: crear sesión, segmentar, generar exactamente cinco, decidir, resumir. También la validación de entradas, las transiciones de estado inválidas, la idempotencia, los motivos maliciosos tratados como datos y el aislamiento entre sesiones |
| `test_robustness.py` | Acceso a las imágenes (path traversal, pertenencia a la sesión), recuperación de trabajos interrumpidos, ejecutor en hilo y validador del resumen (etiquetas invertidas, motivos inventados o mal atribuidos, variaciones faltantes o duplicadas) |
| `test_ports.py` | Piezas intercambiables: las mismas pruebas para cada adaptador del repositorio y de la generación (en el proceso y detrás del contrato HTTP del worker), y la elección de adaptador por variables de entorno |
| `test_mask_and_delete.py` | Máscara manual (válida, tamaño incorrecto, no PNG, toda blanca o negra, bloqueada tras generar) y borrado de la cadena de rondas sin tocar otras sesiones |
| `test_retention.py` | Borrado automático a las 24 horas. Comprueba que el plazo cuenta desde la primera ronda y que borra la cadena con sus archivos sin tocar las sesiones vigentes. Una generación en curso se reintenta y un lote colgado no bloquea el borrado. También cubre la pasada al arrancar, el valor 0 que lo desactiva y los avisos en `/healthz` y en la UI |
| `test_rounds.py` | Rondas: extracción de ajustes, planificador (las 32 combinaciones de aceptar o rechazar dan cinco variantes distintas), `next-round` y su vista previa, máximo de 3 rondas, nota que nunca llega al prompt y migración de la base |
| `test_ui.py` | Manejadores reales de Gradio contra la API en memoria: el resumen guarda antes los cambios pendientes del formulario y respeta las decisiones congeladas |

## Agente LLM

| Archivo | Cubre |
|---|---|
| `test_gemini_summary.py` | Agente de resumen con un Gemini falso. Comprueba que solo salen decisiones y motivos (nunca fotos) y que se descarta la salida que contradice el registro, la salida inválida y las citas inventadas. También el cambio al modelo de respaldo si el principal está saturado, el presupuesto de tiempo y el respaldo determinista rotulado |

## Generación y guardrails

| Archivo | Cubre |
|---|---|
| `test_qwen_edit.py` | Modo intermedio: mejor candidata por estilo, verificación de fidelidad (rechaza un producto espejado), reintentos dentro del presupuesto de tiempo y «cinco o nada». También que la API envía cada modo a su worker y sirve la web app |
| `test_sdxl_backend.py` | Backend SDXL con un pipeline falso: tamaño de generación, revisión fijada del modelo y LoRA aplicado solo al modo avanzado (con reversión si falla la carga) |
| `test_intermediate.py` | Categoría del producto de punta a punta y el backend RealVisXL (experimento descartado) |
| `test_framing.py`, `test_layout.py` | Encuadre de catálogo, armonización de luz y reencuadre del modo avanzado |
| `test_rembg_segmentation.py` | Convención de la máscara del segmentador (blanco = fondo, negro = producto) |
| `test_guardrails.py` | G8 (fidelidad), G12 (foto oscura, quemada o borrosa), G13 (máscara diminuta, enorme o con varios objetos), G14 (timeout y resultado tardío descartado) y eventos de G1 |
| `test_screening.py` | G2 (NSFW en la entrada), G3 (rostros) y G7 (NSFW en la salida) |
| `test_modal_lora_check.py` | Verificación del LoRA en el worker de Modal, sin importar `modal` |

## Entrenamiento y evaluación

| Archivo | Cubre |
|---|---|
| `test_bg60k_*.py` | Curación del dataset BG60k con datos sintéticos: manifiesto, filtros por categoría, embeddings, piloto, split, puntajes de estilo e identidad, revisión y captions |
| `test_distill_select.py` | Selección de candidatas para el LoRA por autodestilación |
| `test_training_colab.py` | Entrenamiento en Colab: configuración, driver y rutas de falla con un CLI `colab` falso |
| `test_lora_eval.py`, `test_lora_compare.py` | Evaluación del LoRA y comparación entre corridas con revisión ciega |
| `test_prueba_usuarios.py` | Análisis de la prueba con usuarios (`evaluation/prueba_usuarios/analizar.py`) |

Última ejecución: 371 pasadas y 16 omitidas (2026-11-18, Windows 11, Python 3.12, versiones exactas de `uv.lock`).
