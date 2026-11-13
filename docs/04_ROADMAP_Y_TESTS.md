# 04 — Plan incremental de implementación y pruebas

**Prioridad:** demostrar un flujo completo, offline y reproducible antes de consumir GPUs.

## Responsables y fechas

| Hito | Responsable | Fecha objetivo |
|---|---|---|
| 0 — Preparación | Por definir | Por definir |
| 1 — Vertical slice mock | Por definir | Por definir |
| 2 — Inpainting real | Por definir | Por definir |
| 3 — Agente | Por definir | Por definir |
| 4 — Dataset y análisis exploratorio | Por definir | Por definir |
| 5 — Fine-tuning LoRA | Por definir | Por definir |
| 6 — GCP y respaldo | Por definir | Por definir |
| 7 — Evaluación, video e informe | Por definir | Por definir |

Los Hitos 2 y 3 pueden avanzar en paralelo. El Hito 7 depende de los modelos reales, no de que el Hito 6 esté terminado.

## Hito 0 — Preparación y validación (sin GPU)

- [ ] Compartir documento de alcance y registrar feedback del equipo.   
- [ ] Crear repo Git privado y ramas/convenciones.
- [ ] No subir fotografías privadas ni claves a Git.
- [ ] Planificar la solicitud de cuota de GPU en GCP, que puede tardar, junto con el presupuesto.

## Hito 1 — Primer PR: vertical slice mock

Objetivo: **demostración local offline** con cinco imágenes simuladas y resumen determinista; *no afirmar generación real ni agente LLM*.

**Estado (2026-11-12):** implementado y revisado en la rama `feat/hito-1-mock` (PR 1). La revisión corrigió el motivo multilínea, el EXIF del original, el resumen concurrente y el bloqueo del event loop; `pytest` pasa con 43 pruebas (Windows, Python 3.12, `uv sync --locked`). Queda P10 para el Hito 6.

- [x] `pyproject.toml` con dependencias ligeras y versiones fijables; separar extras pesados `gpu` y `training` para después.
- [x] FastAPI + Pydantic, rutas de `02_API_Y_MODELO_DE_DATOS.md` (salvo `POST /mask`, que es del Hito 2).
- [x] SQLite + almacenamiento local; acceso por adaptadores/repositorios.
- [x] Gradio para imagen, selector de modo y visualización de cinco tarjetas.
- [x] Segmentador `MockSegmentationService` (resultado marcado como simulado).
- [x] `MockGenerationBackend`: cinco imágenes **claramente identificadas como MOCK**, distintas por estilo, con IDs y seeds registrados.
- [x] Flujo de estado sin éxito parcial de generación y protección de duplicados.
- [x] Decisiones individuales, motivos opcionales y bloqueo del resumen hasta tener cinco.
- [x] `DeterministicSummaryService`: no usar ni fingir LLM en este hito.
- [x] Tests `pytest` con base de datos temporal y sin red.
- [x] README local: instalación, ejecución, pruebas y límites conocidos.

### Pruebas mínimas automatizadas

| ID | Caso | Resultado esperado |
|---|---|---|
| T01 | Imagen JPG/PNG válida | Sesión creada, metadatos registrados |
| T02 | Archivo inválido o muy grande | Error controlado; nada persistido inconsistente |
| T03 | Elegir `intermediate` y `advanced` | Cada uno disponible como modo en dominio, ambos en mock local |
| T04 | Generación nominal | Exactamente **5** variaciones, 5 estilos distintos, IDs únicos |
| T05 | Backend produce 4 o 6 variaciones | `FAILED`; nunca `READY_FOR_REVIEW` |
| T06 | POST `/generate` duplicado | No duplica trabajos o variaciones |
| T07 | Decidir sobre ID que no pertenece a sesión | 404/409 y ningún cambio |
| T08 | Solicitar resumen con 4 decisiones | 409, ninguna explicación emitida |
| T09 | 5 decisiones (3 aceptar/2 rechazar) | Resumen 5/5 concordante |
| T10 | 5 rechazos | Resumen válido, lista aceptadas vacía |
| T11 | Comentario malicioso/inyectado | Tratado como texto; sin ejecuciones ni supuestos |
| T12 | Modificar decisión antes de resumen | Última decisión persistida y resumida |
| T13 | Intentar modificar decisión tras `SUMMARIZED` | 409 (sesión congelada) |
| T14 | Nueva sesión con imagen propia | Sin contaminación de decisiones entre sesiones |

### Definition of Done del primer PR

1. `pytest -q` pasa **realmente** en entorno limpio (adjuntar salida de ejecución).
2. Demo UI permite: subir una imagen, elegir modo, ver cinco mocks, guardar cinco decisiones, ver resumen correcto.
3. Documentación no atribuye a SDXL/Gemini capacidades que aún no se implementaron.
4. Sin tokens ni pesos pesados en Git y sin gasto cloud.
5. PR pequeño, revisable por otro integrante.

## Hito 2 — Inpainting real (intermedio)

- [ ] Puerta de decisión: medir memoria, arranque y lote completo en GCP, GPU local y Colab con benchmarks autorizados; elegir entornos según el resultado.
- [ ] Probar `rembg` y la máscara; implementar reintento y `POST /sessions/{id}/mask` para máscaras manuales.
- [ ] Ejecutar el candidato SDXL base con el pipeline de inpainting de cuatro canales; medir VRAM, latencia y calidad de integración.
- [ ] Integrar preservación del producto y manejo explícito de sombras/bordes.
- [ ] Generar cinco estilos por sesión; guardar checkpoint/revisión/seeds/prompts.
- [ ] Probar fallos en tazas, recipientes y decorativos; registrar errores.
- [ ] Casos obligatorios: taza con logotipo (hueco del asa, brillo y logo curvo), producto mate sencillo, producto blanco sobre fondo claro y producto con texto pequeño visible. Evaluar el logotipo contra la foto original, sin exigir reconstruir su parte oculta.
- [ ] Guardrails del Hito 2 (ADR-008, `08_ITERACIONES_Y_GUARDRAILS.md`): G2 y G7 (clasificador NSFW en entrada y salida), G3 (rostros: rechazar), G8 (fidelidad de la región protegida), G12 (brillo y nitidez), G13 (heurística de máscara) y G14 (timeout por lote). Calibrar sus umbrales con los casos obligatorios.
- [ ] `DELETE /sessions/{id}` para el borrado de fotos (G11).
- [x] Iteraciones (ADR-007): `POST /sessions/{id}/next-round` y su vista previa, sesiones hijas (máximo 3 rondas), planificador determinista y vocabulario cerrado de ajustes por reglas. Adelantado sobre el mock el 2026-11-12; falta comprobarlo con imágenes reales.

## Hito 3 — Agente con Gemini en Vertex AI

- [ ] Herramienta `get_session_decisions(session_id)` de solo lectura.
- [ ] Herramienta opcional `get_variation_metadata(variation_id)` de solo lectura.
- [ ] LLM produce salida estructurada con referencias a variaciones y motivos citados literalmente.
- [ ] Python verifica cobertura, decisiones y citas, y renderiza la enumeración; `generated_by` y `validation_status` reflejan el resultado. Fallback explícito ante timeout/error, sin disfrazar fallback como agente.
- [ ] IAM mínimo, sin secretos en `.env` versionado, pruebas con respuestas Gemini simuladas.
- [ ] Reutilizar `validate_structured` y `render_summary` de `backend/productshot/services/summary.py` (ya probados en el Hito 1).
- [ ] *(ADR-007)* Extracción de ajustes desde los motivos con salida estructurada y resumen de la cadena de rondas.

## Hito 4 — Dataset y análisis exploratorio

- [ ] Revisar la respuesta docente a la propuesta; si no se acepta el LoRA, aplicar el plan B (ADR-002).
- [ ] Definir el protocolo experimental antes de reservar el conjunto de prueba.
- [ ] Consultar a los autores de BG60k sobre la publicación de ejemplos y de los pesos, y confirmar un acceso reproducible (ADR-006).
- [ ] Inspeccionar una muestra estratificada y medir la cobertura por estilo; usar CommonCatalog solo ante un déficit demostrado, y ABO + CommonCatalog si BG60k no se confirma.
- [ ] Filtrar por producto único, máscara íntegra, fondo visible, resolución y ausencia de collages, con OCR fuera del producto y revisión visual.
- [ ] Definir la rúbrica de estilos y armar un piloto de 100–200 imágenes por estilo, con captions asistidos por BLIP y plantilla, revisados a mano.
- [ ] Definir particiones sin fuga de productos y conseguir la autorización de las fotos de prueba.

## Hito 5 — Fine-tuning LoRA (Colab Pro)

- [ ] Smoke test sobre SDXL base: entrenamiento corto, exportación, recarga e inferencia con el pipeline de inpainting (ADR-004).
- [ ] Checkpoints reanudables y exportación de pesos versionados.
- [ ] Inferencia con LoRA y comparación de memoria/estilos, evaluando cada estilo por separado.

## Hito 6 — GCP y respaldo

- [ ] Adaptadores Firestore/Cloud Storage.
- [ ] Cloud Run CPU y worker GPU desplegados mediante imágenes Docker.
- [ ] Credenciales/IAM, auth, límites de gasto, logs y manejo de estados de job.
- [ ] Probar el respaldo local con modelos reales y los mismos contratos (ADR-003).
- [ ] Ensayar la demo completa en GCP, en menos de seis minutos y con el modelo precargado, y grabar el video de respaldo.

## Hito 7 — Evaluación, video e informe

- [ ] Evaluación comparativa y análisis crítico en ejemplos reservados.
- [ ] Prueba con usuarios según `06_EVALUACION_Y_ETICA.md`: reclutar 10 o más personas ajenas a CC/DS, preferentemente del nicho; puede empezar apenas termine el Hito 2.
- [ ] Piloto del guion con una o dos personas y ajuste antes de las sesiones.
- [ ] Sesiones con SUS, SEQ, éxito y tiempo por tarea; corregir los problemas que bloquean y reportar el resto.
- [ ] Video y demostración e informe.

## Alcance técnico del primer hito

Implementar exclusivamente el **vertical slice offline** descrito en los requisitos anteriores. Se esperan:

- FastAPI + Gradio + SQLite + almacenamiento local de artefactos.
- Carga validada de una sola imagen y selección de `intermediate | advanced`.
- Segmentación simulada identificada como tal y cinco imágenes de demostración por sesión.
- Cinco decisiones editables antes del resumen y persistencia consistente.
- Resumen determinista de las decisiones sin afirmar integración con Gemini.
- Manejo de errores e idempotencia definidos en `02_API_Y_MODELO_DE_DATOS.md`.
- Pruebas T01–T14, ejecución reproducible y documentación de limitaciones.

No incluir todavía modelos generativos descargados, endpoints en nube ni claves reales. Una función solo puede marcarse como probada cuando exista evidencia de su ejecución.
