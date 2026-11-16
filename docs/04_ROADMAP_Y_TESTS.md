# 04 — Plan incremental de implementación y pruebas

**Prioridad:** demostrar un flujo completo, offline y reproducible antes de consumir GPUs.

## Responsables y fechas

| Hito | Responsable | Fecha objetivo |
|---|---|---|
| 0 — Preparación | Por definir | Por definir |
| 1 — Vertical slice mock | Por definir | Por definir |
| 2 — Modo intermedio | Aaron Camacho (ADR-010) | Congelado el 2026-11-16 |
| 3 — Agente | Por definir | Por definir |
| 4 — Dataset y análisis exploratorio | Por definir | Por definir |
| 5 — Fine-tuning LoRA | Por definir | Por definir |
| 6 — GCP y respaldo | Aaron Camacho (`09_PLAN_AARON_MODAL.md`) | Por definir |
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
- [x] FastAPI + Pydantic, rutas de `02_API_Y_MODELO_DE_DATOS.md`.
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

## Hito 2 — Modo intermedio (ADR-010)

**Estado (2026-11-16):** cerrado y congelado. El inpainting con el producto fijo se descartó con fotos reales (se ve «pegado»); el intermedio es un bloque independiente con Qwen-Image-2.1-Turbo, escenas curadas por categoría y candidatas verificadas. Evidencia en ADR-010 y en `05_COLAB_Y_GCP.md`.

- [x] Puerta de decisión: Modal con L40S (respaldo A100/H100); medido en H100: p95 de 2.07 s por generación, lote de 5 con 15 candidatas en 45–67 s, 40.2 GB de VRAM pico.
- [x] Segmentación: BiRefNet (`birefnet-general`) en GPU, 1.9 s por foto, con limpieza de fragmentos sueltos; reintento y máscara manual siguen disponibles. G13 detecta caja + producto (dos objetos) como aviso.
- [x] Comparar inpainting (SDXL base, nueve canales, RealVisXL, FLUX.1 Fill) contra editores (FLUX.2 klein, Kontext, Qwen-Image-2.1 y su Turbo) con fotos reales: `deploy/modal/real_photos.py`, `quality_lab.py` y `edit_lab*.py`.
- [x] Preservación del producto: verificación por candidata (alineación ORB, bordes dentro del producto y cambio de escena) en vez de recomposición; G8 «verificado».
- [x] Cinco estilos por sesión, con modelo, revisión, semillas (pedida y de la candidata), prompt (`edit-v1`) y métricas guardados.
- [x] Casos obligatorios y del nicho: taza estampada, vela (mate con texto), caja de SSD con texto pequeño, impresora con su caja, figura, incienso, lata, entre otros (12 productos). Límite: productos con cables sueltos.
- [x] Guardrails: G2 y G7 (NSFW), G3 (rostros, calibrado con la taza estampada), G15 (tipo de producto con CLIP), G8 por modo, G12, G13 y G14. Umbrales calibrados con 22 fotos reales.
- [x] `DELETE /sessions/{id}` para el borrado de fotos (G11): borra la cadena completa de rondas.
- [x] Iteraciones (ADR-007): probadas con imágenes reales el 2026-11-16 (ronda 2 de la vela con «más luminoso», 10/10 verificadas).

## Hito 3 — Agente con Gemini (AI Studio) y un segundo LLM

- [ ] Herramienta `get_session_decisions(session_id)` de solo lectura.
- [ ] Herramienta opcional `get_variation_metadata(variation_id)` de solo lectura.
- [x] LLM produce salida estructurada con referencias a variaciones y motivos citados literalmente (Gemini con esquema JSON, 2026-11-13).
- [x] Python verifica cobertura, decisiones y citas, y renderiza la enumeración; `generated_by` y `validation_status` reflejan el resultado. Fallback explícito ante timeout/error, sin disfrazar fallback como agente. Con modelo de respaldo si el principal está saturado (2026-11-13).
- [x] Clave de Gemini por AI Studio en `.env` o en un secreto de Modal, nunca versionada; pruebas con respuestas de Gemini simuladas (ADR-009). Prueba real opcional con `RUN_LIVE_GEMINI=1`.
- [ ] Segundo LLM detrás del mismo `SummaryService` (modelo abierto gratuito u Ollama) para comparar resúmenes con las mismas métricas.
- [x] Reutilizar `validate_structured` y `render_summary` de `backend/productshot/services/summary.py` (ya probados en el Hito 1).
- [ ] *(ADR-007)* Extracción de ajustes desde los motivos con salida estructurada y resumen de la cadena de rondas.

## Hito 4 — Dataset y análisis exploratorio

- [x] Respuesta docente sobre el LoRA: el equipo decidió no esperarla (2026-11-14); el plan B (ADR-002) queda solo para inviabilidad técnica.
- [ ] Definir el protocolo experimental antes de reservar el conjunto de prueba.
- [ ] Confirmar un acceso reproducible a BG60k (ADR-006). No se consulta a sus autores por permisos: es un trabajo académico (decisión del 2026-11-14).
    Descarga del 2026-11-14 desde el JD Whale Disk del repositorio de BG60k: llegaron 19 de 22 zip, todos íntegros. Faltan `bg60k_imgs_2`, `bg60k_masks_1` y `bg60k_masks_4`. Los zip se reparten por `id mod 10`, así que quedan 44 304 de 63 293 pares imagen y máscara completos (70 %, grupos 0, 3, 5, 6, 7, 8 y 9), sin sesgo de categoría. Alcanza para el piloto; los grupos 1, 2 y 4 se excluyen del manifiesto.
- [ ] Inspeccionar una muestra estratificada y medir la cobertura por estilo; usar CommonCatalog solo ante un déficit demostrado, y ABO + CommonCatalog si BG60k no se confirma.
- [ ] Filtrar por producto único, máscara íntegra, fondo visible, resolución y ausencia de collages, con OCR fuera del producto y revisión visual.
- [ ] Definir la rúbrica de estilos y armar un piloto de 100–200 imágenes por estilo, con captions asistidos por BLIP y plantilla, revisados a mano.
- [ ] Definir particiones sin fuga de productos y conseguir la autorización de las fotos de prueba.

## Hito 5 — Fine-tuning LoRA (Colab Pro de Pedro; Kaggle de respaldo)

- [x] Smoke test sobre SDXL base: entrenamiento corto en Colab, exportación, recarga e inferencia con los pipelines de inpainting de cuatro y nueve canales, verificada en Modal (2026-11-14, `docs/05`). Faltan memoria y latencia del lote completo con LoRA.
- [ ] Confirmar con Pedro cuándo vencen sus unidades de Colab Pro (217.84 el 2026-11-14; la L4 gasta 1.94 por hora).
- [ ] Checkpoints reanudables y exportación de pesos versionados a un repositorio privado de Hugging Face Hub, con manifiesto (ADR-009).
- [ ] Inferencia con LoRA y comparación de memoria/estilos, evaluando cada estilo por separado. El backend ya carga el adaptador (`attach_lora`; modo avanzado con `lora_scale`, intermedio con peso 0 en el mismo pipeline); falta que el worker HTTP lo cargue desde `LORA_ADAPTER_URI`.

## Hito 6 — Despliegue en Modal y piezas intercambiables (ADR-009)

- [x] Piezas intercambiables: interfaz `Repository`, raíz de composición por variables de entorno y contrato HTTP del worker con `RemoteGenerationBackend` (2026-11-13). Falta el ejecutor de Modal (P10), con el despliegue.
- [x] Worker del modo intermedio en Modal (clase `Worker`, L40S) con secretos y límite de gasto (2026-11-16). La API y la UI corren en la laptop contra el worker (acordado el 2026-11-16: las tareas 1 a 4 del doc 09 se descartan); el avanzado tendrá su propio worker.
- [ ] Cancelar la inferencia remota cuando vence G14: hoy la API abandona a los 540 s, pero el worker sigue generando (y cobrando GPU) hasta terminar el lote o llegar a su tope de 10 minutos.
- [x] (2026-11-16) Los adaptadores remotos siguen el 303 de Modal como GET con el token (prueba con transporte simulado). Antes: manejar las solicitudes largas al endpoint web de Modal: pasados 150 s Modal responde con una redirección 303 y `RemoteGenerationBackend` no la sigue (HTTPX no sigue redirecciones por defecto), así que intentaría leerla como la respuesta del contrato. Probarlo con un lote lento, con autenticación y dentro del presupuesto total de espera.
- [ ] Probar el worker en la laptop (RTX 4060) con los mismos contratos.
- [ ] Ensayar la demo completa en Modal, en menos de seis minutos y con el worker caliente, y grabar el video de respaldo.
- [ ] Alternativa GCP (solo si hay créditos): adaptadores de Firestore y Cloud Storage, Dockerfile y Cloud Run, según `05_COLAB_Y_GCP.md`.

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
