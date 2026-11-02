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

**Principales:** fidelidad e integración visual mediante rúbricas; cinco resultados y concordancia 5/5 del resumen; finalización y errores de usuario. **Secundaria:** porcentaje de aceptadas, que mide preferencia declarada y no detecta por sí solo un logotipo alterado; indicar si participan emprendedores reales y reportarlo por producto.

| Dimensión | Métrica / procedimiento | Observaciones |
|---|---|---|
| Cumplimiento | `n_variaciones_exitosas == 5` | Obligatorio en generación exitosa |
| Fidelidad | Rúbrica humana por forma/color/logos/texto, bordes y sombras | Un producto recompuesto tendrá similitud pixel trivial; usar también integración visual |
| Calidad comercial | Escala ordinal ciega (p. ej. 1–5), con rúbrica | Reportar mediana, variabilidad y casos |
| Estilo | Rúbrica o similitud imagen-texto (si se valida) | Métricas automáticas no sustituyen criterio humano |
| Diversidad | LPIPS promedio entre pares y valoración humana | Mayor diversidad no siempre es mejor |
| Preferencia | % aceptadas por modo | Exploratoria; controlar asignación aleatoria a evaluadores |
| Consistencia del agente | Exactitud de los cinco labels (objetivo 5/5), cobertura, razones no inventadas | Validar con datos de la DB |
| Usabilidad | Tasa de finalización, errores y cuestionario breve | No confundir juicio de la imagen con calidad UX |
| Operación | Segundos por lote, VRAM pico, errores, cold starts | Fundamental para presupuesto GCP |

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
