# Diagramas

Cada diagrama tiene su fuente editable en draw.io (`.drawio`) y su exportación en PNG. Representan el diseño propuesto: ningún componente está implementado todavía. Al editar una fuente, volver a exportar su PNG. El diagrama de arquitectura GCP usa los [iconos oficiales de Google Cloud](https://cloud.google.com/icons), incrustados en el `.drawio`.

| Diagrama | Qué muestra | Archivos |
|---|---|---|
| Arquitectura GCP | Alternativa B desde ADR-009 (el principal es Modal; ver `01_ARQUITECTURA.md`, sección F) | [PNG](01-arquitectura-gcp.png) · [draw.io](01-arquitectura-gcp.drawio) |
| Arquitectura local | Diseño anterior a ADR-009; hoy la laptop es el entorno de desarrollo | [PNG](02-arquitectura-local.png) · [draw.io](02-arquitectura-local.drawio) |
| Flujo del sistema | Flujo de información y decisiones | [PNG](03-flujo-sistema.png) · [draw.io](03-flujo-sistema.drawio) |
| Secuencia de sesión | Recorrido completo, de la foto al resumen | [PNG](04-secuencia-sesion.png) · [draw.io](04-secuencia-sesion.drawio) |
| Secuencia de generación | Generación asíncrona en GCP, con éxito y fallo | [PNG](05-secuencia-generacion-gcp.png) · [draw.io](05-secuencia-generacion-gcp.drawio) |
| Estados de sesión | Transiciones de `UPLOADED` a `SUMMARIZED` o `FAILED` | [PNG](06-estados-sesion.png) · [draw.io](06-estados-sesion.drawio) |
| Modelo de datos | Entidades y relaciones | [PNG](07-modelo-datos.png) · [draw.io](07-modelo-datos.drawio) |
| Pipeline de generación | De la foto a las cinco variaciones, por modo | [PNG](08-pipeline-generacion.png) · [draw.io](08-pipeline-generacion.drawio) |
| Agente y resumen | Consulta, verificación y fallback | [PNG](09-agente-resumen.png) · [draw.io](09-agente-resumen.drawio) |
| Datos y LoRA | Auditoría del dataset, entrenamiento y evaluación | [PNG](10-datos-lora.png) · [draw.io](10-datos-lora.drawio) |
