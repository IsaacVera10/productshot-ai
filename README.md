# ProductShot AI

**Proyecto 2 — Planificación y Toma de Decisiones en IA (CS5363, UTEC, 2026-2)**  
**Estado actual:** Hito 1 (vertical slice offline) implementado: API, UI, persistencia y pruebas con segmentación, generación y resumen **simulados**. Todavía no hay generación con IA ni agente LLM.

## Objetivo

Generar, a partir de una foto casera de un producto, cinco propuestas de ambientación para catálogo o redes sociales, priorizando la conservación de sus rasgos visibles, para que la persona acepte o rechace cada una y reciba un resumen fiel de sus decisiones.

## Descripción

ProductShot AI transforma **una fotografía de producto** en **exactamente cinco variaciones visuales** para catálogos y redes sociales, priorizando la fidelidad de sus características visibles. El usuario evalúa cada variación mediante una decisión de aceptación o rechazo; el sistema registra las cinco decisiones y un agente basado en un LLM presenta un resumen fiel de ellas.

El caso inicial se limita a productos pequeños, rígidos y preferiblemente opacos. No se reconstruyen ángulos ocultos ni geometría tridimensional.

## Modos de generación

| Modo | Arquitectura | Estado |
|---|---|---|
| `intermediate` | Segmentación + inpainting con modelo de difusión preentrenado | Propuesto |
| `advanced` | Segmentación + inpainting con modelo adaptado mediante LoRA | Propuesto; adaptación a verificar |

Ambos modos deben poder elegirse y utilizarse por separado. Corresponden a los niveles intermedio y avanzado del enunciado; si el LoRA no se acepta o no resulta viable, el plan B es básico + intermedio. La evaluación A/B es un diseño experimental del proyecto, no una exigencia literal del enunciado.

## Flujo funcional

1. Cargar una imagen JPG/PNG de un producto.
2. Seleccionar el modo de generación.
3. Obtener una máscara automática de producto, reintentarla o subir una máscara manual.
4. Generar exactamente cinco variaciones (`studio`, `natural`, `warm`, `lifestyle`, `minimal`).
5. Evaluar individualmente cada variación y registrar la decisión.
6. Solicitar un resumen estructurado y validado de las cinco decisiones.

## Tecnologías previstas

| Área | Desarrollo local | Despliegue objetivo |
|---|---|---|
| API y UI | FastAPI, Gradio | Cloud Run CPU |
| Generación | PyTorch, Diffusers, pipeline local/mock | Cloud Run GPU |
| Segmentación | rembg / SAM 2 | Servicio GPU |
| Datos operativos | SQLite | Firestore |
| Artefactos | Directorio local | Cloud Storage |
| Agente de explicación | Servicio simulado (MVP) | Gemini en Vertex AI |
| Entrenamiento LoRA | Prototipos / scripts reproducibles | Colab Pro para entrenamiento offline |

GCP es el despliegue principal y la arquitectura local su respaldo; el mapeo de ambas está en [01 — Arquitectura, sección F](docs/01_ARQUITECTURA.md#f-mapeo-de-arquitecturas-gcp-y-local). El checkpoint exacto de difusión sigue pendiente de pruebas de memoria, compatibilidad con inpainting y entrenamiento LoRA.

## Documentación

- [00 — Alcance y requisitos](docs/00_ALCANCE_Y_REQUISITOS.md)
- [01 — Arquitectura](docs/01_ARQUITECTURA.md)
- [02 — API y modelo de datos](docs/02_API_Y_MODELO_DE_DATOS.md)
- [03 — Decisiones y riesgos](docs/03_DECISIONES_Y_RIESGOS.md)
- [04 — Roadmap y pruebas](docs/04_ROADMAP_Y_TESTS.md)
- [05 — Colab y GCP](docs/05_COLAB_Y_GCP.md)
- [06 — Evaluación y ética](docs/06_EVALUACION_Y_ETICA.md)
- [07 — Enunciado Proyecto 2](docs/07_ENUNCIADO_PROYECTO_2.md)
- [08 — Propuesta: iteraciones y guardrails](docs/08_ITERACIONES_Y_GUARDRAILS.md) (guardrails aceptados; iteraciones pendientes de consulta docente)
- [Rúbrica de evaluación](docs/Evaluación_Rúbrica_Proyecto2.md)

**Fuente primaria:** `Evaluación_Enunciado Proyecto2.pdf`. El archivo `docs/07_ENUNCIADO_PROYECTO_2.md` reproduce el contenido textual del documento original en Markdown para consulta. Ante discrepancias prevalece el PDF oficial.

## Ejecución local (Hito 1)

El proyecto se gestiona con [uv](https://docs.astral.sh/uv/). No necesita GPU, red (salvo para instalar) ni credenciales.

```bash
uv sync                # crea .venv con Python 3.12 e instala las versiones exactas de uv.lock
uv run productshot     # API + UI en http://127.0.0.1:8000  (UI en /ui, OpenAPI en /docs)
uv run pytest          # pruebas T01–T14 y complementarias
```

`uv run` usa siempre el `.venv` del proyecto, así que no hace falta activarlo. Para añadir una dependencia: `uv add <paquete>` (o `uv add --dev <paquete>` si solo es para desarrollo); después versionar `pyproject.toml` y `uv.lock` juntos.

Sin uv (alternativa): crear y activar un entorno con `python -m venv .venv`, luego `pip install -e .` y `pip install pytest`. Así no se respetan las versiones exactas del lockfile.

La configuración se lee de variables de entorno o de un `.env` (copiar `.env.example`). Los datos se guardan en `data/`, que Git ignora.

Estructura:

| Ruta | Contenido |
|---|---|
| `backend/productshot/domain.py` | Estados, estilos, entidades y errores de dominio |
| `backend/productshot/storage/` | Repositorio SQLite y almacenamiento de artefactos (adaptadores) |
| `backend/productshot/services/` | Orquestador de sesión, segmentación, generación, resumen y validador |
| `backend/productshot/api/` | FastAPI: rutas `/api/v1` y esquemas |
| `ui/productshot_ui/` | Gradio; consume la API por HTTP |
| `tests/` | Pruebas `pytest` sin red ni GPU |

### Límites conocidos del Hito 1

- La máscara es una elipse fija y las cinco imágenes son fondos planos con el producto recortado. Todo lleva la franja «SIMULACIÓN (MOCK) - NO GENERADO POR IA»; **no son resultados de IA**.
- El resumen lo compone Python (`generated_by: deterministic_fallback`); no hay agente LLM todavía.
- `POST /sessions/{id}/mask` (máscara manual) queda para el Hito 2.
- Sin autenticación: cualquiera con el ID de sesión puede ver sus imágenes. Solo para uso local.
- El ejecutor de trabajos es un hilo del mismo proceso; un reinicio marca como `FAILED` las generaciones en curso.

Consultar [roadmap y casos de prueba](docs/04_ROADMAP_Y_TESTS.md) antes de desarrollar. La ejecución de pruebas y la disponibilidad de GPU deben verificarse antes de declarar una función como operativa.

## Entregables académicos

Sistema reproducible, informe técnico de máximo diez páginas, diagramas, ejemplos visuales, video de máximo tres minutos con el participante indicado en el enunciado y demostración de máximo seis minutos. Consultar el enunciado para el detalle formal.
