# 03 — Registro de decisiones arquitectónicas (ADR) y riesgos

**Fecha:** 2026-11-01 (Perú). **Estado:** diseño inicial, sin implementación verificada. No transformar hipótesis en hechos.

## Decisiones tomadas por el equipo en conversación

| ID | Decisión | Racional | Estado |
|---|---|---|---|
| D-01 | Caso B: fotografía comercial de productos para emprendedores | Problema concreto, compatible con input imagen y cinco variaciones | Elegida |
| D-02 | Entrada v1: fotografía única, sin cambio de perspectiva | Minimiza alucinaciones geométricas/textuales | Elegida |
| D-03 | Cinco presets por ejecución: studio/natural/warm/lifestyle/minimal (ver ADR-001) | Facilita diversidad y evaluación | Propuesta implementable |
| D-04 | Modo intermedio: segmentación + inpainting preentrenado | Control de región editable sin reentrenamiento | Elegido, pendiente de pruebas |
| D-05 | Modo avanzado: inpainting + LoRA entrenado | Adaptación y comparación del estilo comercial | Elegido; plan B en ADR-002 |
| D-06 | Colab Pro para entrenamiento offline | Evita factura de entrenamiento en GCP | Elegido, condicionado a recursos |
| D-07 | GCP para despliegue; Cloud Run CPU/GPU + Storage/Firestore | Separación web/inferencia y portabilidad | Elegido como principal, con respaldo local (ADR-003) |
| D-08 | Gemini mediante Vertex AI para agente LLM | Consumo gestionado, sin alojar LLM en nuestra GPU | Elegido |
| D-09 | MVP local offline con adaptador mock y resumen determinista | Probar el flujo sin costosos modelos ni credenciales | Elegido para primer PR |
| D-10 | Una sesión = imagen + modo + generación de cinco; sin iteraciones | Evita estado ambiguo y repetición de costes | Decisión v1 revisable |

## Pendientes que NO bloquean el primer PR

**P1 — Validación docente (prioridad alta):** dejar constancia en la propuesta de que el modo intermedio y el LoRA reentrenado se presentan como los dos niveles elegidos. Si no se aceptan, aplicar el plan B de ADR-002.

**P3 — Checkpoint final (puede esperar a la prueba de GPU):** candidato inicial: `stabilityai/stable-diffusion-xl-base-1.0` con el pipeline de inpainting de cuatro canales en ambos modos (solo **candidato**; ver ADR-004). Ambos modos comparten también VAE, scheduler, prompts y posprocesamiento, y no se mezclan checkpoints en una comparación. Registrar ID, **commit SHA de revisión**, licencia, exactitud de las dependencias y checksum de cualquier LoRA.

**P4 — Compatibilidad LoRA + inpainting (riesgo técnico mayor):** el UNet específico de SDXL inpainting tiene **cinco canales de entrada adicionales**, y `train_text_to_image_lora_sdxl.py` es para text-to-image de SDXL. La ruta inicial entrena sobre SDXL base e infiere con el pipeline de inpainting, que admite UNet de cuatro canales y mezcla latentes con la máscara. Verificar con pasos cortos el entrenamiento, la exportación, la recarga y la inferencia antes de curar un dataset grande. La calidad de integración con cuatro canales es una hipótesis a medir; si es insuficiente, evaluar el UNet de nueve canales y documentar por qué.

**P5 — GPU Colab Pro / L4:** GPU de Colab no garantizada; hacer medición de VRAM pico, memoria de sistema, tiempo por inferencia de cinco imágenes y tiempo de inicio en GPU. Probar pesos `fp16`/optimizaciones solo si compatibles. Definir budget máximo real para cloud después de medir.

**P6 — Dataset:** BG60k es el candidato principal, condicionado a auditoría (ADR-006). Aún deben confirmarse los permisos con sus autores, el acceso reproducible, la calidad de las máscaras, los duplicados y la cobertura por estilo. La estructura de conjuntos está en `00_ALCANCE_Y_REQUISITOS.md`.

**P7 — Preservación de producto:** una máscara protege los rasgos visibles, pero la integración de sombras/reflejos puede fallar. Medir fallos. No prometer preservación perfecta ni cambios físicos de iluminación sobre toda la superficie.

**P8 — Hosting / autenticación:** escoger región, cuota de GPU, acceso de prueba (IAM/autenticación), política de retención y límites de consumo antes de desplegar. No desplegar públicamente por defecto.

## Decisiones que deben salir de pruebas (no de preferencias)

1. Modelo/checkpoint y revisión exacta, formato/precisión de pesos, dimensiones efectivas.
2. Segmentador inicial: `rembg` por sencillez; `SAM 2` solo si corrige fallos de forma significativa y cabe en recursos.
3. Adaptador LoRA: confirmación de la arquitectura de ADR-004, parámetros `rank`, LR, resolución y memoria.
4. Plataforma del worker GPU: Cloud Run GPU como preferencia, Compute Engine como contingencia si inferencia o startup no se ajustan a demo y presupuesto.
5. Selección final de datos + política de licencia/atribución.

## Fuentes oficiales para verificaciones técnicas

- Enunciado `Evaluación_Enunciado Proyecto2.pdf`, §2.3–2.7, §3 y §4.
- SDXL inpainting, model card: https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1
- Explicación de canales del UNet inpainting: https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/blob/main/README.md
- Ejemplo de entrenamiento **text-to-image** LoRA SDXL (NO asumir inpainting): https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py
- Pipeline SDXL inpainting con UNet de cuatro o nueve canales: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_inpaint.py
- Licencia y contenido de ABO: https://amazon-berkeley-objects.s3.amazonaws.com/index.html
- BG60k, paper y repositorio: https://arxiv.org/abs/2312.13309 y https://github.com/Whileherham/BG60k
- CommonCatalog CC-BY: https://huggingface.co/datasets/common-canvas/commoncatalog-cc-by
- Finegrain Product Masks Lite: https://huggingface.co/datasets/finegrain/finegrain-product-masks-lite
- Soporte Cloud Run GPU: https://docs.cloud.google.com/run/docs/configuring/services/gpu
- Límites Colab: https://research.google.com/colaboratory/faq.html?hl=es

## Cómo registrar cambios

Ante un cambio importante, agregar un bloque:

```text
ADR-XXX — [título]
Fecha: YYYY-MM-DD
Estado: propuesta / aceptada / descartada
Contexto: ...
Opciones: ...
Decisión: ...
Consecuencias: ...
Evidencia: prueba / enlace / decisión docente
```

No cambiar silenciosamente contratos ya usados por más de un módulo; abrir issue/PR y documentar migración.

## ADR registrados

```text
ADR-001 — Sustituir el preset brand por minimal
Fecha: 2026-11-01
Estado: aceptada
Contexto: brand promete adaptar la imagen a una marca, pero el sistema no recibe datos de marca ni puede evaluar ese resultado.
Opciones: conservar brand con paleta configurable; brand_palette con colores extraídos del producto; minimal.
Decisión: usar minimal. La paleta extraída del producto queda como experimento opcional bajo el nombre product_palette.
Consecuencias: no se añaden datos de entrada, API ni interfaz. Se debe distinguir minimal de studio con rasgos observables. Un logotipo en el producto se conserva con cualquier preset.
Evidencia: revisión de diseño; sin pruebas de generación todavía. Extraer colores no garantiza que la generación los respete, y la distancia cromática no mide armonía ni identidad de marca.
```

```text
ADR-002 — Niveles intermedio y avanzado, con plan B
Fecha: 2026-11-01
Estado: aceptada
Contexto: el enunciado pide dos de tres enfoques. Ambos modos comparten flujo y difieren en el LoRA reentrenado por el equipo.
Opciones: intermedio + avanzado; básico + intermedio.
Decisión: intermedio + avanzado. El modo advanced es utilizable en la app y además un experimento con pregunta propia. Básico + intermedio queda como plan B.
Consecuencias: el documento debe mostrar el trabajo propio del avanzado: datos, entrenamiento y análisis de estabilidad. El plan B solo se activa si la docente no acepta el LoRA o si este no resulta viable técnicamente.
Evidencia: un compañero presentó a la docente un avance con LoRA reentrenado sin objeciones; es un indicio informal, no una confirmación escrita para este proyecto.
```

```text
ADR-003 — GCP como despliegue principal, con respaldo local
Fecha: 2026-11-01
Estado: aceptada
Contexto: el enunciado no exige nube, pero el equipo quiere desplegar en GCP. La demo en vivo no debe depender de un único entorno.
Opciones: solo local; GCP opcional; GCP desde el inicio con respaldo local.
Decisión: comprometer GCP como despliegue principal y mantener una arquitectura local equivalente como respaldo, con los mismos contratos.
Consecuencias: aumenta el trabajo de infraestructura y requiere presupuesto, cuota de GPU y dos conjuntos de adaptadores. La evaluación, el video y el informe no deben quedar bloqueados por el despliegue.
Evidencia: decisión del equipo; faltan mediciones de memoria, latencia, arranque y costo en GCP, GPU local y Colab.
```

```text
ADR-004 — Ruta del LoRA y conjuntos de datos
Fecha: 2026-11-01
Estado: aceptada
Contexto: el LoRA no tenía objetivo definido, el script de referencia es text-to-image y el UNet de inpainting tiene nueve canales.
Opciones: adaptar el entrenamiento a nueve canales; cargar en nueve canales un LoRA entrenado sobre base; entrenar sobre SDXL base e inferir con el pipeline de cuatro canales; un LoRA por estilo o uno solo.
Decisión: entrenar un solo LoRA sobre SDXL base, condicionado por captions, e inferir con el pipeline de cuatro canales en ambos modos. Separar datos de entrenamiento/validación estética y prueba casera.
Consecuencias: no cambia la API ni los estados. Cambia P3, P4, manifiestos y benchmarks. Con ambos modos en la misma base, la comparación aísla el efecto del LoRA.
Evidencia: revisión del código del pipeline de diffusers, que admite cuatro o nueve canales y carga de LoRA. Sin pruebas de entrenamiento ni de calidad todavía.
```

```text
ADR-005 — Recuperación de máscara y garantías del resumen
Fecha: 2026-11-01
Estado: aceptada
Contexto: un fallo de segmentación bloqueaba el flujo, y un validador determinista no puede garantizar la fidelidad de la prosa libre.
Opciones: editor de máscaras o carga de PNG; narrativa libre validada o resumen estructurado verificado.
Decisión: permitir reintento y carga de máscara en PNG antes de generar. El agente LLM produce un resumen estructurado que Python verifica y renderiza; la narrativa libre es opcional y su revisión automática es parcial.
Consecuencias: se añaden POST /sessions/{id}/mask, generated_by y validation_status. Se elimina is_validated. La fidelidad de la narrativa se mide con revisión humana.
Evidencia: revisión de diseño; sin pruebas con Gemini todavía.
```

```text
ADR-006 — Selección del dataset
Fecha: 2026-11-01
Estado: aceptada por consenso del equipo; BG60k condicionado a auditoría
Contexto: el dataset es obligatorio y el LoRA necesita estética de escena comercial. ABO es de catálogo y su cobertura de estilos está sin demostrar.
Opciones: BG60k; ABO; CommonCatalog; Unsplash Lite; Products-10K; PxHere; muestra de Getty; Pexels Advertisement Photography.
Decisión: BG60k como candidato principal, CommonCatalog solo ante un déficit demostrado, Finegrain Lite para evaluar segmentación y fotos caseras autorizadas para prueba. Alternativa: ABO + CommonCatalog.
Consecuencias: BG60k solo permite uso académico, sin licencia con nombre; debe aclararse con sus autores la publicación de ejemplos y de los pesos. Su filtrado por CTR introduce un sesgo publicitario, y no consta que excluya marcas. Las máscaras sirven para filtrar y medir; no aíslan el aprendizaje de un LoRA text-to-image estándar.
Descartes: Pexels por términos que restringen la extracción para machine learning y la copia masiva; Products-10K por estar orientado a reconocimiento y ser no comercial; PxHere por ser general, sin captions declarados y con contenido adulto aparente; Getty por su tamaño y sus términos con acceso condicionado. Unsplash Lite queda en reserva, porque sus términos restringen publicar partes de los datos.
Evidencia: revisión de papers, páginas y términos oficiales. Sin descarga: siguen sin comprobar el acceso, la integridad, las máscaras, los duplicados, la cobertura y el rendimiento.
```

```text
ADR-007 — Iteraciones guiadas por las decisiones del usuario
Fecha: 2026-11-04
Estado: propuesta (pendiente de revisión del equipo)
Contexto: la rúbrica pide «correspondencia entre las decisiones del usuario y las iteraciones» y D-10 impide que las decisiones influyan en una generación posterior.
Opciones: regenerar las rechazadas dentro de la misma sesión; que el LLM reescriba los prompts; nueva ronda como sesión hija con un planificador determinista.
Decisión propuesta: ronda siguiente = sesión hija (misma imagen y máscara, cinco variaciones) cuyo plan deriva de las decisiones previas mediante un planificador determinista y un vocabulario cerrado de ajustes. D-10 se mantiene por ronda.
Consecuencias: campos parent_session_id, round_index y round_plan_json; la invariante de lote pasa a cinco (style, variant) distintos; nueva ruta POST /sessions/{id}/next-round; resumen de cadena con citas verificables. Detalle en 08_ITERACIONES_Y_GUARDRAILS.md.
Evidencia: lectura de la rúbrica; sin implementación ni validación docente.
```

```text
ADR-008 — Guardrails ejecutables
Fecha: 2026-11-04
Estado: propuesta (pendiente de revisión del equipo); G1, G4, G10 (validador) y el rótulo mock de G9 ya están en el Hito 1
Contexto: la rúbrica de ética (2 pts) exige guardrails implementados y visibles en la demo, no solo reflexión.
Opciones: solo reflexión; filtros externos de contenido; guardrails por capas (entrada, texto, generación, salida, fidelidad, transparencia, agente, operación).
Decisión propuesta: guardrails por capas G1–G11, priorizando fidelidad del producto (G8), clasificador NSFW en entrada y salida (G2, G7) y detector de rostros (G3).
Consecuencias: más VRAM y latencia en el worker (clasificadores), umbrales que calibrar y casos de demo preparados. Detalle en 08_ITERACIONES_Y_GUARDRAILS.md.
Evidencia: lectura de la rúbrica; sin mediciones todavía.
```
