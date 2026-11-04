# Backend

API y servicios de dominio para sesiones, segmentación, generación, persistencia, evaluación y resumen (paquete `productshot`).

**Estado:** Hito 1 implementado con backends simulados (`mock`). Contrato en [`docs/02_API_Y_MODELO_DE_DATOS.md`](../docs/02_API_Y_MODELO_DE_DATOS.md).

## Dónde enchufar los hitos siguientes

| Hito | Punto de extensión |
|---|---|
| 2 — Inpainting real | Implementar `GenerationBackend` y `SegmentationService` (`services/generation.py`, `services/segmentation.py`) y registrarlos en `api/app.py`. Los prompts v1 están en `services/presets.py`, sin probar |
| 3 — Agente Gemini | Implementar `SummaryService` que produzca un `StructuredSummary` y pasarlo por `validate_structured` y `render_summary` (`services/summary.py`) |
| 6 — GCP | Adaptadores de `ArtifactStore` (Cloud Storage) y del repositorio (Firestore) con las mismas garantías de transición atómica; `JobRunner` que encole en Cloud Tasks |

El orquestador (`services/sessions.py`) valida el lote de cinco y las transiciones de estado, de modo que un backend nuevo no puede saltarse esas reglas.
