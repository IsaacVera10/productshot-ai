# ProductShot AI

**Proyecto 2 — Planificación y Toma de Decisiones en IA (CS5363, UTEC, 2026-2)**

A partir de una foto casera de un producto, ProductShot AI genera **exactamente cinco variaciones visuales** para catálogos y redes sociales. Prioriza que el producto conserve sus rasgos visibles. La persona acepta o rechaza cada variación. El sistema registra las cinco decisiones y un **agente basado en un LLM** (Gemini) redacta un resumen fiel de ellas, que Python valida contra el registro. Opcionalmente, se pide otra ronda de cinco derivada de esas decisiones, hasta tres rondas.

El caso de uso se limita a productos pequeños, rígidos y preferiblemente opacos. No se reconstruyen ángulos ocultos ni geometría 3D.

## Componentes del sistema

| Componente | Implementación | Código |
|---|---|---|
| Generación de variaciones, modo **intermedio** | Editor por instrucciones **Qwen-Image-2.1-Turbo** con escenas por estilo y categoría. Genera varias candidatas por estilo y verifica la fidelidad del producto (alineación ORB y bordes) para elegir la mejor | [`services/qwen_edit.py`](backend/productshot/services/qwen_edit.py), [`services/scenes.py`](backend/productshot/services/scenes.py), [`services/fidelity.py`](backend/productshot/services/fidelity.py) |
| Generación de variaciones, modo **avanzado** | Inpainting **SDXL de nueve canales** (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`) con un **LoRA propio** entrenado sobre BG60k. Después recompone el producto original encima (G8) | [`services/sdxl_generation.py`](backend/productshot/services/sdxl_generation.py), [`services/layout.py`](backend/productshot/services/layout.py), [`training/`](training/) |
| Segmentación | BiRefNet (rembg) en el worker; también se puede subir una máscara manual | [`services/rembg_segmentation.py`](backend/productshot/services/rembg_segmentation.py) |
| Interfaz gráfica | Web app (HTML/JS, en `/app`) y panel técnico en Gradio (`/ui`). Ambas hablan con la API por HTTP | [`backend/productshot/web/`](backend/productshot/web/), [`ui/productshot_ui/`](ui/productshot_ui/) |
| Registro de decisiones | SQLite con tablas `sessions`, `variations`, `decisions` (`accepted`/`rejected`, motivo y fecha) y `summaries`. Cada transición de estado es atómica | [`storage/repository.py`](backend/productshot/storage/repository.py) |
| Agente LLM | Gemini recibe solo las decisiones y los motivos (nunca las fotos) y devuelve un resumen **estructurado**. Python verifica cobertura, decisiones y citas contra el registro antes de mostrarlo. Si el modelo falla, prueba con uno de respaldo y, si ese también falla, usa un resumen determinista rotulado como tal | [`services/gemini_summary.py`](backend/productshot/services/gemini_summary.py), [`services/summary.py`](backend/productshot/services/summary.py) |
| Rondas 2 y 3 | Un planificador determinista traduce las decisiones y una nota de ajuste a un vocabulario cerrado de modificadores. Cada variación nueva explica qué decisión la originó | [`services/rounds.py`](backend/productshot/services/rounds.py) |
| Privacidad | Se quitan los metadatos EXIF y GPS de la foto. La sesión se puede borrar a pedido, y si no, se borra automáticamente con todas sus rondas 24 horas después de subir la foto | [`services/images.py`](backend/productshot/services/images.py), [`services/sessions.py`](backend/productshot/services/sessions.py) (`purge_expired`) |
| Guardrails | Calidad de la foto (G12), máscara (G13), contenido y rostros (G2, G3, G7), categoría permitida (G15), fidelidad del producto (G8) y tiempo máximo del lote (G14) | [`services/guardrails.py`](backend/productshot/services/guardrails.py), [`services/screening.py`](backend/productshot/services/screening.py) |

## Arquitectura

```
Navegador ──► API FastAPI (/api/v1) ──► Orquestador de sesión ──► SQLite + carpeta de imágenes
 /app, /ui        │                        (services/sessions.py)
                  ├── HTTP ──► Worker intermedio (Modal, H100): Qwen-Image + BiRefNet + G2/G3/G7/G15
                  ├── HTTP ──► Worker avanzado (Modal, L4):    SDXL inpainting + LoRA
                  └── HTTPS ─► Gemini (AI Studio): agente de resumen
```

La API no necesita GPU. Cada pieza se elige con variables de entorno en [`composition.py`](backend/productshot/composition.py), así que el mismo código corre de dos formas:

- **Simulado** (`mock`): sin GPU, sin red y sin credenciales. Sirve para desarrollar y corre las pruebas. Las imágenes son fondos planos con la franja «SIMULACIÓN (MOCK)».
- **Real** (`remote` + `gemini`): generación en los workers de Modal y resumen con Gemini.

## Requisitos

- [uv](https://docs.astral.sh/uv/) (instala Python 3.12 y las versiones exactas de `uv.lock`).
- Solo para el modo real: una cuenta de [Modal](https://modal.com) y una clave de Gemini de [AI Studio](https://aistudio.google.com/apikey).

## Ejecución 1: modo simulado (sin GPU ni credenciales)

```bash
uv sync                # crea .venv e instala dependencias y herramientas de desarrollo
uv run productshot     # API + interfaces en http://127.0.0.1:8000
```

| URL | Qué es |
|---|---|
| http://127.0.0.1:8000/app | Web app |
| http://127.0.0.1:8000/ui | Panel en Gradio |
| http://127.0.0.1:8000/docs | API (OpenAPI) |
| http://127.0.0.1:8000/api/v1/healthz | Qué adaptador usa cada pieza |

Los datos se guardan en `data/`, que Git ignora.

## Ejecución 2: sistema completo (generación real y agente LLM)

### 1. Desplegar los workers en Modal

```bash
uv run --with modal modal setup                      # una sola vez: autentica la cuenta

# Secretos: un token por worker. La API envía el mismo valor.
uv run --with modal modal secret create productshot-worker   GENERATION_WORKER_TOKEN=<token-intermedio>
uv run --with modal modal secret create productshot-advanced GENERATION_WORKER_TOKEN=<token-avanzado>

# Worker del modo intermedio (también segmenta y revisa las fotos)
uv run --with modal modal deploy deploy/modal/worker.py

# Worker del modo avanzado: primero sube el LoRA al volumen (se verifica por sha256 al arrancar)
uv run --with modal modal volume put productshot-hf-cache <adaptador>.safetensors lora/v5distill-ckpt1000.safetensors
uv run --with modal modal deploy deploy/modal/advanced_worker.py
```

`modal deploy` imprime la URL de cada worker. Los pesos de los modelos base se descargan la primera vez y quedan en el volumen. Los workers se apagan solos tras uno o dos minutos sin solicitudes, así que la GPU solo se cobra mientras se usa. El primer lote después de un apagado tarda alrededor de un minuto más.

El adaptador LoRA no se versiona porque es un archivo binario pesado. Se obtiene con el pipeline de [`training/`](training/) (curación de BG60k, entrenamiento en Colab y evaluación); las instrucciones están en [training/README.md](training/README.md).

### 2. Configurar la API

```bash
cp .env.example .env
```

En `.env`:

```ini
GENERATION_BACKEND=remote
SEGMENTATION_BACKEND=remote
SCREENING_BACKEND=remote
SUMMARY_BACKEND=gemini

GENERATION_WORKER_URL=<URL del worker intermedio>
GENERATION_WORKER_TOKEN=<token-intermedio>
ADVANCED_WORKER_URL=<URL del worker avanzado>
ADVANCED_WORKER_TOKEN=<token-avanzado>

GEMINI_API_KEY=<clave de AI Studio>
```

Sin `ADVANCED_WORKER_URL`, el modo avanzado responde «todavía no está disponible» y el intermedio sigue funcionando. Si la configuración está incompleta (por ejemplo, `remote` sin URL), la API se niega a arrancar y explica qué falta.

### 3. Arrancar

```bash
uv run productshot
```

`/api/v1/healthz` debe mostrar los adaptadores remotos y el de Gemini.

### Alternativa: contenedor

El [`Dockerfile`](Dockerfile) empaqueta la API y las interfaces (sin GPU) para Railway u otro host de contenedores. Los comandos están en su encabezado.

## Flujo de uso

1. Subir una foto JPG o PNG, elegir la categoría del producto y el modo (`intermediate` o `advanced`).
2. Revisar la máscara automática del producto: reintentarla o subir una propia.
3. Generar las cinco variaciones: `studio`, `natural`, `warm`, `lifestyle` y `minimal`.
4. Aceptar o rechazar cada una, con un motivo opcional.
5. Pedir el resumen del agente.
6. Opcional: pedir otra ronda con una nota de ajuste (hasta tres rondas).
7. Borrar la sesión, con todas sus rondas y fotos, en cualquier momento. Si nadie la borra, el servidor lo hace solo 24 horas después de subir la foto (`SESSION_RETENTION_HOURS`).

## Pruebas

```bash
uv run pytest
```

Las pruebas no usan red ni GPU: los modelos se sustituyen por dobles de prueba y cada prueba usa una base de datos temporal. Cubren la API y el flujo completo, el registro de decisiones, el validador del resumen del agente, las rondas, los guardrails, los backends de generación y los scripts de entrenamiento y evaluación. El detalle por archivo está en [tests/README.md](tests/README.md).

En Windows:

- Si los permisos sobre `%TEMP%` están restringidos, usar `uv run pytest --basetemp=.pytest_tmp`.
- Las pruebas del driver de Colab se omiten, porque ese driver solo corre en Linux o macOS.

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `backend/productshot/domain.py` | Estados, estilos, categorías, entidades y errores de dominio |
| `backend/productshot/composition.py` | Elige el adaptador de cada pieza según las variables de entorno |
| `backend/productshot/api/` | FastAPI: rutas `/api/v1` y esquemas |
| `backend/productshot/services/` | Orquestador de sesión, generación, segmentación, agente de resumen, rondas y guardrails |
| `backend/productshot/storage/` | Interfaz `Repository`, repositorio SQLite y almacenamiento de imágenes |
| `backend/productshot/worker/` | Servidor y contrato HTTP del worker de generación |
| `backend/productshot/web/` | Web app (HTML, CSS y JS sin dependencias) |
| `ui/productshot_ui/` | Panel en Gradio |
| `deploy/modal/` | `worker.py` y `advanced_worker.py` son los workers que usa el sistema. El resto son experimentos documentados de selección de modelos y calibración, cada uno con su uso en el encabezado |
| `training/` | Curación del dataset BG60k, entrenamiento del LoRA en Colab y evaluación ([README](training/README.md)) |
| `evaluation/` | Calibración de G15 y prueba con usuarios: guion, registro y análisis |
| `tests/` | Pruebas `pytest` ([README](tests/README.md)) |

## Límites conocidos

- **Licencias:** Qwen-Image-2.1 tiene licencia de investigación (no comercial). Por eso el modo intermedio solo sirve para uso académico.
- **Autenticación:** no hay. Cualquiera con el ID de una sesión (un UUID aleatorio) puede ver sus imágenes mientras existan. El borrado automático a las 24 horas limita esa exposición, pero no la elimina.
- **Trabajos en curso:** el ejecutor de trabajos es un hilo del mismo proceso, y un reinicio de la API marca como `FAILED` las generaciones que estaban en curso.
- **Rondas:** el agente LLM solo redacta el resumen. Las rondas siguientes las planifica Python con reglas, de modo que ningún texto libre del usuario llega al prompt de generación.
- **Fidelidad:** las verificaciones reducen el riesgo de que el producto cambie, pero no lo eliminan. Productos transparentes, reflectantes o deformables quedan fuera del alcance.

## Entregables académicos

Sistema reproducible, informe técnico de máximo diez páginas, diagramas, ejemplos visuales, video de máximo tres minutos con el participante indicado en el enunciado y demostración de máximo seis minutos. El detalle formal está en el enunciado.
