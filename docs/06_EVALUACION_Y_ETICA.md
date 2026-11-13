# 06 — Evaluación experimental, usabilidad y ética

## Preguntas de evaluación

- ¿Se generan **cinco variaciones** y funcionan los controles de decisión del usuario?
- ¿Las variaciones son visualmente atractivas, diversas y acordes a su estilo?
- ¿El producto visible conserva forma, color, inscripciones y logo originales?
- ¿El modelo adaptado con LoRA aporta calidad/estilo comercial respecto al baseline?
- ¿El agente comunica las decisiones **sin contradicciones ni motivos inventados**?
- ¿La interacción es comprensible, usable y suficientemente rápida para una demo?

## Diseño experimental propuesto (no requisito textual del enunciado)

- Fijar un conjunto de prueba **separado del entrenamiento por identidad del producto**; empezar con piloto de ~5–10 productos y ampliar según presupuesto/tiempo (n final por definir y justificar).
- Una imagen por producto, misma vista y estilos predefinidos. Ejecutar ambos modos con el mismo checkpoint base, VAE, scheduler, prompts/estilos, posprocesamiento, resolución y número de pasos, de modo que solo varíe el LoRA; registrar seeds y versiones.
- Evaluación **ciega**: personas evaluadoras no saben qué modo generó la muestra; mezclar orden de las salidas para evitar sesgo.
- Evaluar por producto y por estilo, no inferir superioridad por un único ejemplo.
- Registrar fallos además de casos exitosos. No entrenar/ajustar sobre imágenes reservadas de prueba.

## Métricas propuestas

**Principales:** fidelidad e integración visual mediante rúbricas; cinco resultados y concordancia 5/5 del resumen; usabilidad con SUS, éxito y tiempo por tarea (sección «Usabilidad»). **Secundaria:** porcentaje de aceptadas, que mide preferencia declarada y no detecta por sí solo un logotipo alterado; indicar si participan emprendedores reales y reportarlo por producto.

| Dimensión | Métrica / procedimiento | Observaciones |
|---|---|---|
| Cumplimiento | `n_variaciones_exitosas == 5` | Obligatorio en generación exitosa |
| Fidelidad | Rúbrica humana por forma/color/logos/texto, bordes y sombras | Un producto recompuesto tendrá similitud pixel trivial; usar también integración visual |
| Calidad comercial | Escala ordinal ciega (p. ej. 1–5), con rúbrica | Reportar mediana, variabilidad y casos |
| Estilo | Rúbrica o similitud imagen-texto (si se valida) | Métricas automáticas no sustituyen criterio humano |
| Diversidad | LPIPS promedio entre pares y valoración humana | Mayor diversidad no siempre es mejor |
| Preferencia | % aceptadas por modo | Exploratoria; controlar asignación aleatoria a evaluadores |
| Consistencia del agente | Exactitud de los cinco labels (objetivo 5/5), cobertura, razones no inventadas | Validar con datos de la DB |
| Usabilidad | SUS, éxito por tarea, tiempo por tarea, errores y SEQ (ver sección «Usabilidad») | No confundir juicio de la imagen con calidad UX |
| Operación | Segundos por lote, VRAM pico, errores, cold starts | Fundamental para presupuesto GCP |
| Guardrails | Veces que se activa cada guardrail y fallos que no detectó | Registrar desde los logs (ver `08_ITERACIONES_Y_GUARDRAILS.md`, parte B) |

## Usabilidad

Mide si una persona del nicho puede completar el flujo sola, no si le gustan las imágenes. Las metas son propuestas del equipo; con unas diez personas se reportan de forma descriptiva (mediana, rango y casos), sin pruebas estadísticas.

| Métrica | Cómo se mide | Meta propuesta |
|---|---|---|
| Éxito por tarea | Porcentaje de participantes que completan cada tarea sin ayuda | 80 % o más en cada tarea |
| Tiempo por tarea | Cronómetro desde que empieza la tarea hasta que la termina; la espera de generación se mide aparte | Se reporta la mediana; sin meta hasta tener la primera medición |
| Errores y pedidos de ayuda | Conteo por participante, anotado por quien observa (clic equivocado, retroceso, pregunta) | Se reportan y se clasifican por gravedad |
| SEQ | Tras cada tarea: «¿Qué tan fácil o difícil fue esta tarea?», de 1 a 7 | Mediana de 5 o más |
| SUS | Cuestionario System Usability Scale de 10 ítems al final, en español | Promedio de 68 o más (el promedio de referencia de la escala) |
| Comprensión del resumen | La persona dice si el resumen refleja lo que decidió (sí o no, y por qué) | 10 de 10 |
| Espera percibida | «¿El tiempo de espera fue aceptable?», de 1 a 5 | Se reporta junto al tiempo real de generación |

Cada problema observado se registra con la tarea, cuántas personas lo tuvieron y su gravedad (bloquea, molesta o cosmético). Los problemas que bloquean se corrigen antes de la evaluación final.

## Prueba con usuarios

**Participantes:** 10 como mínimo, que no estudien Ciencia de la Computación ni Data Science (familiares, amistades o conocidos). Se priorizan personas del nicho, es decir que venden o quieren vender productos por redes sociales. Se registra solo un perfil no identificable: código P01 a P10, rango de edad, si emprende y su experiencia con apps de edición de fotos.

**Piloto:** antes de las diez sesiones, se prueba el guion con una o dos personas para ajustar instrucciones y tiempos. El piloto no cuenta dentro de las diez.

**Cuándo:** después del Hito 2, con el modo intermedio generando imágenes reales. Antes se puede hacer una prueba informal de la interfaz con el backend mock, pero no reemplaza a esta.

**Material:** cada persona usa una foto de un producto suyo, tomada con su celular. Si no tiene, se le presta uno de un conjunto de productos del nicho preparado por el equipo. El modo (`intermediate` o `advanced`) se asigna alternando entre participantes y se registra; esta prueba no compara modos.

**Sesión (15 a 20 minutos):**

1. Consentimiento informado: qué se registra, que la foto se borra al terminar el proyecto salvo autorización expresa, que las decisiones y los motivos escritos (nunca la foto) se envían a Gemini, cuyo plan gratuito permite a Google usarlos para mejorar sus productos (ADR-009), y que puede retirarse cuando quiera.
2. Contexto en una frase, sin enseñar la interfaz: «Imagina que quieres publicar este producto en tu Instagram».
3. Tareas, cada una seguida de la pregunta SEQ:
   - U1: subir la foto y elegir el modo indicado.
   - U2: revisar las cinco imágenes y aceptar o rechazar cada una, con el motivo de al menos una.
   - U3: pedir el resumen y decir si refleja lo que decidió.
4. Durante las tareas, la persona piensa en voz alta y quien observa no ayuda. Si queda bloqueada más de dos minutos, se le ayuda y la tarea cuenta como no completada.
5. Al final: cuestionario SUS y tres preguntas abiertas: qué fue lo más confuso, qué cambiaría y si usaría la herramienta para sus ventas.

**Registro:** una hoja por participante con tiempos, errores, respuestas y observaciones. No se graba el rostro; si se graba la pantalla, con consentimiento. Las decisiones de estas sesiones también alimentan la métrica de preferencia.

**Relación con el video:** la persona de otra carrera que exige el video puede ser una de las participantes, siempre que dé su consentimiento al inicio.

## Protocolo de aceptación / rechazo

Cada evaluador acepta o rechaza individualmente cinco variaciones. El motivo es opcional. **No inferir automáticamente preferencias causales**. Si el usuario rechaza cinco, el resumen debe incluir las cinco rechazadas y cero aceptadas. Registrar la diferencia entre observaciones de estilo (p. ej. «se aceptó warm») y motivos declarados (p. ej. «me gustó el fondo»).

## Experimento del agente

1. Construir escenarios de 5 decisiones con combinaciones: 0/5, 2/3, 5/0, comentarios vacíos y texto adversarial.
2. Pedir al agente que recupere registros mediante herramientas **read-only**.
3. Rechazar cualquier resumen estructurado que cambie un label o cite un motivo que no conste en el registro correcto. Revisar con personas la narrativa libre, porque su revisión automática es parcial.
4. Registrar por separado la salida original del LLM, las correcciones (`hybrid`) y el uso de fallback (`deterministic_fallback`). No llamar «agente Gemini» a un placeholder determinista.

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Se altera color/logo/etiqueta | Máscaras + recomposición del producto + inspección visual; rotular fallos |
| Producto diferente de la fotografía real | No crear vistas ocultas; usar fotografías que el usuario puede autorizar; advertencia de imagen generada |
| Derechos de autor/licencias | Auditoría de dataset y licencias de modelos/adaptadores, atribución cuando proceda |
| Datos personales/empresariales en imágenes | Almacenamiento privado, autenticación, retención limitada y eliminación |
| Sesgo de estilos | Datos y test de múltiples categorías; reconocer cobertura limitada. BG60k se filtró por CTR en una plataforma concreta, lo que favorece esa estética publicitaria |
| Prompt injection desde motivos | Motivos como datos no confiables, herramientas read-only, salida validada |
| Opacidad del agente | Listas de decisiones verificadas, motivos solo explícitos, trazabilidad |
| Sobreajuste del LoRA | Train/validation/test separados por producto, pruebas fuera de distribución |

## Entregables académicos

El informe final (máximo 10 páginas) debe incluir justificación de dataset, arquitectura, implementación, método de evaluación, resultados reales y reflexión ética. Este documento establece el protocolo técnico de evaluación; el informe final debe ser elaborado por el equipo conforme al enunciado.
