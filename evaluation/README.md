# Evaluación

Esta carpeta tiene dos cosas: los scripts para calibrar uno de los guardrails y el material de la prueba con usuarios. La evaluación del LoRA del modo avanzado (validación, test y revisión ciega) está en [`training/`](../training/README.md).

| Archivo | Qué es |
|---|---|
| [`calibrar_g15.py`](calibrar_g15.py) | Calibra G15, el guardrail de categoría permitida |
| [`prueba_usuarios/guion.md`](prueba_usuarios/guion.md) | Protocolo de la sesión con cada participante: preparación, consentimiento, tareas U1–U4, cuestionario SUS, preguntas abiertas y video |
| [`prueba_usuarios/registro.csv`](prueba_usuarios/registro.csv) | Hoja de registro, con una fila por participante. Por ahora solo tiene la cabecera |
| [`prueba_usuarios/calentar.py`](prueba_usuarios/calentar.py) | Despierta el worker de Modal antes de cada participante, para que no espere el arranque en frío |
| [`prueba_usuarios/analizar.py`](prueba_usuarios/analizar.py) | Calcula las métricas a partir del registro |

## Calibración de G15

G15 bloquea las fotos cuya categoría no está permitida. El script aplica a una o más carpetas de fotos el mismo CLIP, las mismas etiquetas y la misma reducción a 1024 px que usa [`services/screening.py`](../backend/productshot/services/screening.py). Para cada foto imprime los grupos bloqueados que superan el umbral y al final muestra un resumen. Corre en GPU si hay una disponible y, si no, en CPU.

```bash
uv run --no-project --with torch --with transformers==4.46.3 --with pillow \
    python evaluation/calibrar_g15.py <carpeta> [<carpeta> ...] [--tar archivo.tar]
```

Los umbrales resultantes van en `.env` como `CATEGORY_BLOCK_THRESHOLD` y `CATEGORY_WARN_THRESHOLD`.

## Prueba con usuarios

Participan unas diez personas, con sesiones de 15 a 20 minutos. Se miden:

- Si cada tarea se completó.
- El tiempo y la facilidad percibida de cada tarea (SEQ, de 1 a 7).
- La espera de la generación y si resultó aceptable.
- Los errores.
- La puntuación SUS.
- Si el resumen del agente refleja las decisiones (U3) y si la segunda ronda las tiene en cuenta (U4).

1. Preparar el sistema completo, como se indica en «Ejecución 2» del [README principal](../README.md), con `SUMMARY_BACKEND=gemini`.
2. Antes de cada participante, despertar el worker:

   ```bash
   uv run python evaluation/prueba_usuarios/calentar.py
   ```

3. Seguir [`guion.md`](prueba_usuarios/guion.md) y anotar una fila en `registro.csv` por participante. Las columnas están descritas en el guion.
4. Al terminar, calcular las métricas:

   ```bash
   uv run python evaluation/prueba_usuarios/analizar.py evaluation/prueba_usuarios/registro.csv
   ```

   Con unas diez personas, el análisis reporta mediana, rango y conteos con intervalo de Wilson al 95 %. No hace pruebas estadísticas. El cálculo lo cubre `tests/test_prueba_usuarios.py`.

**Privacidad:** el registro usa códigos (P01, P02, …) en vez de nombres. La sesión de cada participante se borra al terminar, salvo que haya autorizado guardar la foto. De todos modos, el servidor la borra solo a las 24 horas, así que lo autorizado hay que descargarlo antes. Gemini solo recibe las decisiones y los motivos, nunca las fotos.
