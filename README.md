# ProductShot AI

**Proyecto 2 — Planificación y Toma de Decisiones en IA (CS5363, UTEC, 2026-2)**  
**Estado actual:** especificación técnica y estructura inicial de repositorio; los servicios aún no están implementados.

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

**Fuente primaria:** `Evaluación_Enunciado Proyecto2.pdf`. El archivo `docs/07_ENUNCIADO_PROYECTO_2.md` reproduce el contenido textual del documento original en Markdown para consulta. Ante discrepancias prevalece el PDF oficial.

## Primer hito: prototipo offline

Construir un flujo verificable sin modelos pesados: una foto, dos modos simulados identificados como tales, cinco imágenes de demostración, cinco decisiones persistidas y resumen determinista. No se debe confundir este hito con el sistema generativo final.

Consultar [roadmap y casos de prueba](docs/04_ROADMAP_Y_TESTS.md) antes de desarrollar. La ejecución de pruebas y la disponibilidad de GPU deben verificarse antes de declarar una función como operativa.

## Entregables académicos

Sistema reproducible, informe técnico de máximo diez páginas, diagramas, ejemplos visuales, video de máximo tres minutos con el participante indicado en el enunciado y demostración de máximo seis minutos. Consultar el enunciado para el detalle formal.
