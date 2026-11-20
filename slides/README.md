# Presentación en Overleaf

1. En Overleaf, elige **New Project → Upload Project** y sube el contenido de `slides/` conservando la carpeta `fig/`.
2. Selecciona `main.tex` como documento principal y el compilador **pdfLaTeX**. No requiere shell-escape.
3. Compila dos veces, para que el pie muestre el total de diapositivas (`n/17`). Revisa el log: no debe haber errores ni `Overfull` mayores a 5 pt.
4. Para ensayar con el guion, cambia `hide notes` por `show only notes` en `preamble.tex`. Antes de exponer, vuelve a `hide notes`.

## Reparto y tiempos (15:00, 3:00 por integrante)

Para cambiar el reparto, edita el nombre en los `\note{...}` de cada `sec-*.tex`. Los nombres no aparecen en las diapositivas.

| Bloque | Expone | Diapositivas | Tiempo |
|---|---|---|---|
| Problema y Design Thinking: perfil de Vivian, pregunta guía, lienzo de propuesta de valor | Pedro Mori | 1–4 | 0:00–3:00 |
| Cómo funciona por dentro: flujo, interfaz, arquitectura | Aaron Camacho | 5–7 | 3:00–6:00 |
| Modelos generativos y entrenamiento: dos modos, verificación G8, SDXL + LoRA y BG60k | Isaac Vera | 8–10 | 6:00–9:00 |
| Agente LLM, rondas, guardrails y privacidad | Nicolas Castañeda | 11–13 | 9:00–12:00 |
| Resultados, evaluación con usuarios (pendiente), limitaciones y conclusión | Juan Diego Castro | 14–17 | 12:00–15:00 |

Cada diapositiva tiene su guion y su minuto exacto en `\note{...}`.

Cuando se hagan las entrevistas, actualiza la diapositiva 16 (`sec-05-resultados.tex`) con los resultados y 2–3 hallazgos que cambiaron el diseño.

Esta presentación usa `fig/f01-flujo.tex`, `fig/f02-arquitectura.tex`, `fig/f03-agente.tex` y las imágenes `.png` de `fig/`. Los demás `fig/f*.tex` (`f01-exchange` a `f15-niveles`) y `RelayBook-presentacion.pdf` son de la presentación anterior y ya no se usan; se pueden borrar.
