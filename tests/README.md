# Pruebas

```bash
uv run pytest
```

Cada prueba usa una base de datos y un directorio de artefactos temporales; no hay red ni GPU. La generación corre en línea (`InlineJobRunner`) para que el resultado sea determinista, salvo una prueba que usa el ejecutor en hilo real.

| Archivo | Cubre |
|---|---|
| `test_hito1_api.py` | Casos T01–T14 de [`docs/04_ROADMAP_Y_TESTS.md`](../docs/04_ROADMAP_Y_TESTS.md), más reintento de máscara, generación sin máscara y validación del cuerpo de la decisión |
| `test_robustness.py` | Acceso a artefactos (path traversal, pertenencia a la sesión), recuperación de trabajos interrumpidos, ejecutor en hilo y validador del resumen (etiquetas invertidas, motivos inventados o mal atribuidos, variaciones faltantes o duplicadas) PNG con transparencia compuesto sobre blanco, sin máscara huérfana si falla una escritura, trabajo `FAILED` si no se puede encolar, rótulo `simulated` tomado del registro y límite del motivo medido tras normalizar |
| `test_rounds.py` | Rondas (ADR-007): extracción de ajustes por reglas, planificador (las 32 combinaciones de aceptar o rechazar dan cinco variantes distintas y sin contradicciones), `next-round` y su vista previa, máscara heredada bloqueada, máximo de 3 rondas, nota que nunca llega al prompt, migración de una base del Hito 1 y botón de la UI |
| `test_mask_and_delete.py` | Máscara manual (válida, tamaño incorrecto, no PNG, toda blanca o negra, demasiado grande, bloqueada tras generar y en las rondas hijas) y borrado de la cadena completa, que no toca otras sesiones y espera a que termine una generación |
| `test_ui.py` | Manejadores reales de la UI contra la API en memoria: «Generar resumen» guarda antes los cambios pendientes del formulario, no congela con decisiones incompletas y, ya resumida la sesión, devuelve el formulario al registro |

Última ejecución: 125 pruebas pasadas (2026-11-12, Linux, entorno conda con Python 3.11 y las versiones exactas de `uv.lock` instaladas con `uv pip`).

> En Windows con permisos restringidos sobre `%TEMP%`, usar `uv run pytest --basetemp=.pytest_tmp`.
