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
| D-06 | Colab Pro para entrenamiento offline | Evita factura de entrenamiento en GCP | Vigente (ADR-009): Colab Pro de Pedro (~140 unidades) como principal, Kaggle de respaldo |
| D-07 | GCP para despliegue; Cloud Run CPU/GPU + Storage/Firestore | Separación web/inferencia y portabilidad | Reemplazada por ADR-009: Modal como principal; GCP queda como alternativa compatible |
| D-08 | Gemini mediante Vertex AI para agente LLM | Consumo gestionado, sin alojar LLM en nuestra GPU | Ajustada por ADR-009: Gemini por la API de AI Studio (plan gratuito); Vertex AI solo con la alternativa GCP |
| D-09 | MVP local offline con adaptador mock y resumen determinista | Probar el flujo sin costosos modelos ni credenciales | Elegido para primer PR |
| D-10 | Una sesión = imagen + modo + generación de cinco; sin iteraciones | Evita estado ambiguo y repetición de costes | Vigente por ronda: las iteraciones de ADR-007 son sesiones hijas, cada una con sus cinco |
| D-11 | Nicho: emprendedores en Perú que venden productos pequeños y rígidos por redes sociales (ver `00_ALCANCE_Y_REQUISITOS.md`) | Lineamiento de clase; acota los productos a los que el pipeline maneja mejor y da acceso a usuarios de prueba | Elegida (2026-11-12) |
| D-12 | Guardrails por capas G1 a G14, con mensaje y acción siguiente para cada fallo (ADR-008, `08_ITERACIONES_Y_GUARDRAILS.md`) | Lineamiento de clase y rúbrica de ética | Elegida (2026-11-12) |
| D-13 | Prueba de usabilidad con 10 o más personas ajenas a CC/DS, con SUS, éxito y tiempo por tarea (ver `06_EVALUACION_Y_ETICA.md`) | Lineamiento de clase | Elegida (2026-11-12) |

## Pendientes que NO bloquean el primer PR

**P1: validación docente (cerrada el 2026-11-14).** la propuesta presenta el modo intermedio y el LoRA reentrenado como los dos niveles elegidos. El equipo decidió no esperar una confirmación escrita de la docente; el plan B de ADR-002 queda solo para el caso de que el LoRA no resulte viable técnicamente.

**P3 — Checkpoint final (puede esperar a la prueba de GPU):** candidato inicial: `stabilityai/stable-diffusion-xl-base-1.0` con el pipeline de inpainting de cuatro canales en ambos modos (solo **candidato**; ver ADR-004). Ambos modos comparten también VAE, scheduler, prompts y posprocesamiento, y no se mezclan checkpoints en una comparación. Registrar ID, **commit SHA de revisión**, licencia, exactitud de las dependencias y checksum de cualquier LoRA.

**P4: compatibilidad LoRA + inpainting (riesgo técnico mayor):** el UNet específico de SDXL inpainting tiene **cinco canales de entrada adicionales**, y `train_text_to_image_lora_sdxl.py` es para text-to-image de SDXL. La ruta inicial entrena sobre SDXL base e infiere con el pipeline de inpainting, que admite UNet de cuatro canales y mezcla latentes con la máscara. Verificar con pasos cortos el entrenamiento, la exportación, la recarga y la inferencia antes de curar un dataset grande. La calidad de integración con cuatro canales es una hipótesis a medir; si es insuficiente, evaluar el UNet de nueve canales y documentar por qué. **Medido (2026-11-13 y 14, `05_COLAB_Y_GCP.md`):** con cuatro canales el modelo prolonga la silueta del producto en el fondo y ni el prompt, ni un margen en la máscara, ni bajar `strength` lo resuelven; con los cuatro casos completos, el checkpoint de inpainting de nueve canales con `strength` 0.85 tiene el menor valor del indicador de expansión entre A, F y G (media 0.054 frente a 0.117 de la base; gana 13 de 16 pares) y la mejor integración observada en los ejemplos de la caja y el frasco, pero pierde estilo (conserva más del fondo original) y con `strength` 0.99 es irregular. Con 0.90 y 0.95 el indicador queda entre ambos (0.078 y 0.087), pero en algunos casos cumplen mejor el estilo que 0.85, así que el `strength` no se elige sin medir el estilo. Es un candidato, no una decisión: falta medir el cumplimiento del estilo con revisión ciega, probar con fotos caseras reales y demostrar en el smoke test que un LoRA entrenado sobre SDXL base funciona cargado en nueve canales. Si se adopta, va en ambos modos y hay que decidir si el LoRA se entrena sobre SDXL base o sobre el UNet de nueve canales.

**P5 — GPU Colab Pro / L4:** GPU de Colab no garantizada; hacer medición de VRAM pico, memoria de sistema, tiempo por inferencia de cinco imágenes y tiempo de inicio en GPU. Probar pesos `fp16`/optimizaciones solo si compatibles. Definir budget máximo real para cloud después de medir.

**P6: dataset.** BG60k es el candidato principal, condicionado a auditoría (ADR-006). No se consultará a sus autores por permisos de publicación (decisión del equipo del 2026-11-14: es un trabajo académico, de uso académico como el dataset). Falta comprobar el acceso reproducible, la calidad de las máscaras, los duplicados y la cobertura por estilo. La estructura de conjuntos está en `00_ALCANCE_Y_REQUISITOS.md`.

**P7 — Preservación de producto:** una máscara protege los rasgos visibles, pero la integración de sombras/reflejos puede fallar. Medir fallos. No prometer preservación perfecta ni cambios físicos de iluminación sobre toda la superficie.

**P8 — Hosting / autenticación:** escoger región, cuota de GPU, acceso de prueba (IAM/autenticación) y límites de consumo antes de desplegar. No desplegar públicamente por defecto. La retención ya está decidida en `08_ITERACIONES_Y_GUARDRAILS.md`.

**P9: informar a la docente la interpretación de «iteraciones».** ADR-007 ya está aceptada con la lectura más exigente (rondas de cinco derivadas de las decisiones previas). En la próxima entrega de avance se le comunica para que quede constancia; si ella entiende otra cosa, esa lectura ya queda cubierta y solo se ajusta la presentación. No bloquea ningún hito.

**P10: trabajos en curso con varias instancias (Hito 6).** Al arrancar, `fail_interrupted_jobs` marca FAILED cualquier trabajo en `GENERATING`. Con un solo proceso local es correcto; con varias instancias o con Cloud Tasks hará falta registrar quién ejecuta cada trabajo y un heartbeat, y recuperar solo los trabajos vencidos.

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
Consecuencias: el documento debe mostrar el trabajo propio del avanzado: datos, entrenamiento y análisis de estabilidad. El plan B solo se activa si el LoRA no resulta viable técnicamente; el equipo decidió (2026-11-14) no esperar una confirmación escrita de la docente.
Evidencia: un compañero presentó a la docente un avance con LoRA reentrenado sin objeciones; es un indicio informal, no una confirmación escrita para este proyecto.
```

```text
ADR-003 — GCP como despliegue principal, con respaldo local
Fecha: 2026-11-01
Estado: reemplazada por ADR-009 (2026-11-12); se conserva la idea de mismos contratos en todos los entornos
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
Consecuencias: BG60k solo permite uso académico, sin licencia con nombre; el equipo decidió (2026-11-14) no consultar a sus autores por la publicación de ejemplos y pesos, porque el proyecto es académico; ejemplos y pesos se comparten solo en ese ámbito. Su filtrado por CTR introduce un sesgo publicitario, y no consta que excluya marcas. Las máscaras sirven para filtrar y medir; no aíslan el aprendizaje de un LoRA text-to-image estándar.
Descartes: Pexels por términos que restringen la extracción para machine learning y la copia masiva; Products-10K por estar orientado a reconocimiento y ser no comercial; PxHere por ser general, sin captions declarados y con contenido adulto aparente; Getty por su tamaño y sus términos con acceso condicionado. Unsplash Lite queda en reserva, porque sus términos restringen publicar partes de los datos.
Evidencia: revisión de papers, páginas y términos oficiales. Sin descarga: siguen sin comprobar el acceso, la integridad, las máscaras, los duplicados, la cobertura y el rendimiento.
```

```text
ADR-007 — Iteraciones guiadas por las decisiones del usuario
Fecha: 2026-11-04
Estado: aceptada (2026-11-12), con la lectura más exigente de la rúbrica; máximo 3 rondas, la ronda siguiente exige la madre en SUMMARIZED. Se informa a la docente (P9) sin esperar su respuesta
Contexto: la rúbrica pide «correspondencia entre las decisiones del usuario y las iteraciones» y D-10 impide que las decisiones influyan en una generación posterior.
Opciones: regenerar las rechazadas dentro de la misma sesión; que el LLM reescriba los prompts; nueva ronda como sesión hija con un planificador determinista.
Decisión: ronda siguiente = sesión hija (misma imagen y máscara, cinco variaciones) cuyo plan deriva de las decisiones previas mediante un planificador determinista y un vocabulario cerrado de ajustes. D-10 se mantiene por ronda.
Consecuencias: campos parent_session_id, round_index y round_plan_json; la invariante de lote pasa a cinco (style, variant) distintos; nueva ruta POST /sessions/{id}/next-round; resumen de cadena con citas verificables. Detalle en 08_ITERACIONES_Y_GUARDRAILS.md.
Evidencia: lectura de la rúbrica. Implementada sobre el backend mock el 2026-11-12 (`services/rounds.py`, `tests/test_rounds.py`); sin validación docente.
```

```text
ADR-008 — Guardrails ejecutables
Fecha: 2026-11-04
Estado: aceptada (2026-11-12); G1, G4, G10 (validador) y el rótulo mock de G9 ya están en el Hito 1. Se agregan G12 a G14 y se resuelven las preguntas abiertas (rostro: rechazar; NSFW y umbrales: se eligen y calibran en el Hito 2; retención: siete días en GCP)
Contexto: la rúbrica de ética (2 pts) exige guardrails implementados y visibles en la demo, no solo reflexión.
Opciones: solo reflexión; filtros externos de contenido; guardrails por capas (entrada, texto, generación, salida, fidelidad, transparencia, agente, operación).
Decisión: guardrails por capas G1–G14, priorizando fidelidad del producto (G8), clasificador NSFW en entrada y salida (G2, G7) y detector de rostros (G3).
Consecuencias: más VRAM y latencia en el worker (clasificadores), umbrales que calibrar y casos de demo preparados. Detalle en 08_ITERACIONES_Y_GUARDRAILS.md.
Evidencia: lectura de la rúbrica; sin mediciones todavía.
```

```text
ADR-009 — Infraestructura sin créditos de GCP: Modal como principal y piezas intercambiables
Fecha: 2026-11-12
Estado: aceptada
Contexto: el equipo no tiene créditos de GCP (las cuentas de facturación de Isaac están cerradas y el proyecto planifica-proyecto no tiene facturación). Cloud Run con GPU no tiene capa gratuita, se factura por instancia y sigue cobrando unos minutos después de cada solicitud; una instancia olvidada cuesta unos USD 25 al día y las alertas de presupuesto no cortan el gasto. La cuota de GPU exige solicitud y aprobación. Dejar la demo en la laptop obliga a tenerla encendida. Pedro tiene Colab Pro con unas 140 unidades de cómputo.
Opciones: GCP con tarjeta (estimado USD 25 a 30 para todo el proyecto, con riesgo de cobros por olvido); solo local; Cloudflare Workers AI (inpainting solo con SD 1.5 y sin LoRA propio para imágenes: deja fuera el modo avanzado); Hugging Face Spaces (Gradio en hardware gratuito pide PRO, disco no persistente, poca GPU gratis); Modal (USD 30 de cómputo gratis al mes, L4 cerca de USD 0.80 la hora, escala a cero, Python nativo, admite SDXL y LoRA propio).
Decisión:
1. Despliegue principal en Modal: UI y API como endpoint web sin GPU, y un worker con GPU L4 para segmentar y generar en ambos modos. Límite de gasto de la cuenta en USD 30 si Modal pide tarjeta.
2. Agente con Gemini por la API de AI Studio (plan gratuito, clave en .env o en un secreto de Modal). Segundo LLM para comparar resúmenes: un modelo abierto (Ollama en local, o uno gratuito en la nube) detrás de la misma interfaz SummaryService.
3. LoRA en Colab Pro (cuenta de Pedro) como principal y Kaggle como respaldo gratuito. El smoke test de ADR-004 se completa cargando el adaptador en el worker de Modal, que es donde se servirá.
4. Pesos del LoRA y dataset curado en repositorios privados de Hugging Face Hub, con manifiesto de revisión y checksum; el worker los toma de LORA_ADAPTER_URI.
5. Arquitectura de piezas intercambiables: cada componente (worker GPU, UI y API, base de datos, archivos, agente, ejecutor de trabajos, entrenamiento, pesos) tiene una interfaz y adaptadores para Modal (A), GCP (B) y la laptop (C). El worker se escribe una sola vez y se aloja en cualquiera de los tres detrás del mismo contrato HTTP. Pasar de A a B debe ser configuración más los adaptadores de Firestore y Cloud Storage, sin tocar dominio, UI ni pruebas.
Consecuencias: el informe describe Modal y no Cloud Run. Faltan cuatro piezas para que el intercambio sea real: interfaz Repository (hoy SessionService depende de SqliteRepository), una raíz de composición que elija adaptadores por variables de entorno, el contrato HTTP del worker con un RemoteGenerationBackend, y ejecutores de trabajos por plataforma (P10). El plan gratuito de Gemini permite a Google usar lo enviado para mejorar sus productos: al agente solo se envían decisiones y motivos, nunca fotos, y se declara en el consentimiento de la prueba con usuarios. SQLite en un volumen de Modal exige un solo contenedor de API escribiendo. Las unidades de Colab Pro vencen (habitualmente a los 90 días de su compra): confirmar la fecha con Pedro antes de planificar el entrenamiento.
Evidencia: consulta de precios y planes del 2026-11-12 (Cloud Run, Modal, Workers AI y Spaces; varias cifras de fuentes secundarias, a confirmar al registrarse); revisión de gcloud sobre las cuentas de facturación y proyectos del equipo. Sin mediciones de GPU todavía.
```
