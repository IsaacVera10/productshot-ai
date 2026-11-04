# Pruebas

```bash
pytest
```

Cada prueba usa una base de datos y un directorio de artefactos temporales; no hay red ni GPU. La generación corre en línea (`InlineJobRunner`) para que el resultado sea determinista, salvo una prueba que usa el ejecutor en hilo real.

| Archivo | Cubre |
|---|---|
| `test_hito1_api.py` | Casos T01–T14 de [`docs/04_ROADMAP_Y_TESTS.md`](../docs/04_ROADMAP_Y_TESTS.md), más reintento de máscara, generación sin máscara y validación del cuerpo de la decisión |
| `test_robustness.py` | Acceso a artefactos (path traversal, pertenencia a la sesión), recuperación de trabajos interrumpidos, ejecutor en hilo y validador del resumen (etiquetas invertidas, motivos inventados o mal atribuidos, variaciones faltantes o duplicadas) |

Última ejecución: 39 pruebas pasadas (2026-11-04, Windows 11, Python 3.12.10).

> En Windows con permisos restringidos sobre `%TEMP%`, usar `pytest --basetemp=.pytest_tmp`.
