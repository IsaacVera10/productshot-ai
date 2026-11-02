# 01 — Arquitectura y diagramas

Las versiones editables en draw.io y exportadas en PNG están en [`diagrams/`](diagrams/README.md).

## Criterios de diseño

Separar dominio/servicios de implementaciones de infraestructura. El mismo orquestador invoca backends intercambiables: `mock`, `pretrained_inpainting`, `lora_inpainting`. **`mock` es solo para desarrollo y pruebas y no cuenta como enfoque académico.** El procesamiento de GPU debe ser independiente de la API/web. En local usar persistencia SQLite + carpetas; en nube Firestore + Cloud Storage mediante adaptadores.

## A. Componentes — local y GCP

```mermaid
flowchart TD
    UI["Gradio UI"] --> API["FastAPI: API / Orquestador"]
    API --> REP["SessionRepository / DecisionRepository"]
    REP --> LOCAL[("SQLite local")]
    REP -. "adaptador GCP" .-> FS[("Firestore")]
    API --> ART["ArtifactStore"]
    ART --> FILES[("Archivos locales")]
    ART -. "adaptador GCP" .-> GCS[("Cloud Storage")]
    API --> SEG["SegmentationService: rembg -> máscara"]
    API --> GEN["GenerationService: 5 estilos"]
    SEG --> GEN
    GEN --> SEL{"Backend elegido"}
    SEL --> MOCK["Mock: pruebas offline"]
    SEL --> B["Inpainting preentrenado"]
    SEL --> L["Inpainting + LoRA"]
    B --> QA["Postproceso, preservación y QA"]
    L --> QA
    MOCK --> QA
    QA --> ART
    API --> SUMMARY["SummaryService"]
    SUMMARY --> TOOL["Herramientas de consulta de decisiones"]
    TOOL --> REP
    SUMMARY --> GEM["Gemini en Vertex AI (fase agente)"]
    SUMMARY --> CHECK["Validador determinista"]
    CHECK --> API
    COLAB["Colab Pro: entrenamiento LoRA offline"] --> WEIGHTS["LoRA versionado"]
    WEIGHTS --> L
```

**Nota:** componentes GCP dibujados como adaptadores futuros; el primer PR NO establece conexiones reales con servicios cloud ni usa pesos GPU.

## B. Flujo de sesión (v1)

```mermaid
sequenceDiagram
    actor User as Usuario
    participant UI as Gradio
    participant API as FastAPI
    participant Repo as Repositorio
    participant Worker as Generador
    participant Agent as Agente LLM

    User->>UI: 1 foto + modo
    UI->>API: POST /api/v1/sessions (multipart)
    API->>Repo: Persiste imagen/sesión
    UI->>API: POST /sessions/{id}/segment
    API-->>UI: Mascara y vista previa
    UI->>API: POST /sessions/{id}/generate
    API-->>UI: job_id, estado GENERATING
    API->>Worker: Trabajo (local mock o worker GPU)
    Worker->>Repo: Guarda SOLO lote completo de 5
    loop Consultar estado
        UI->>API: GET /sessions/{id}
        API-->>UI: Estado / variaciones
    end
    loop Para cada una de las 5
        User->>UI: Acepta o rechaza (+ motivo opcional)
        UI->>API: PUT /sessions/{id}/variations/{vid}/decision
        API->>Repo: Upsert de decisión
    end
    User->>UI: Solicita resumen
    UI->>API: POST /sessions/{id}/summary
    API->>Agent: Objetivo: explicar 5 decisiones
    Agent->>Repo: Consulta datos con herramienta read-only
    Agent-->>API: Resumen estructurado
    API->>API: Validación exacta 5/5
    API-->>UI: Resumen verificable
```

### Regla de consistencia

Un lote **solo** está `READY_FOR_REVIEW` si existen exactamente cinco variaciones verificadas y archivos disponibles. Si falla a mitad de la inferencia, marcar `FAILED`, limpiar artefactos transitorios y NO mostrar tres o cuatro resultados como éxito. Concurrencia/idempotencia: un `session_id` solo admite un lote iniciado; un reintento con la misma llave no debe duplicarlo.

## C. Estados de dominio

```mermaid
stateDiagram-v2
    [*] --> UPLOADED
    UPLOADED --> SEGMENTED: máscara lista
    SEGMENTED --> GENERATING: generar
    GENERATING --> READY_FOR_REVIEW: cinco resultados válidos
    GENERATING --> FAILED: error o lote incompleto
    READY_FOR_REVIEW --> REVIEWED: cinco decisiones
    REVIEWED --> SUMMARIZED: resumen validado
    FAILED --> [*]
    SUMMARIZED --> [*]
```

**Nota:** cambiar una decisión después del resumen debe invalidarlo y obligar a recomputarlo, o congelar las decisiones cuando la sesión termine; en v1 se **congelan después de SUMMARIZED**. Hasta entonces se permite corregir aceptación/rechazo.

## D. Entidades y relaciones

```mermaid
erDiagram
    SESSION ||--|| INPUT_IMAGE : contains
    SESSION ||--|| MASK : uses
    SESSION ||--|| GENERATION_JOB : starts
    GENERATION_JOB ||--|{ VARIATION : produces
    VARIATION ||--o| DECISION : receives
    SESSION ||--o| SUMMARY : has
    SESSION {
      string id PK
      string mode
      string status
      string base_model_revision
      datetime created_at
    }
    INPUT_IMAGE {
      string id PK
      string storage_key
      string sha256
      int width
      int height
    }
    MASK {
      string id PK
      string source
      string storage_key
      string sha256
    }
    GENERATION_JOB {
      string id PK
      string backend
      string status
      string idempotency_key
      string config_json
      datetime created_at
    }
    VARIATION {
      string id PK
      int position
      string style
      string storage_key
      int seed
      string model_revision
      string prompt_version
    }
    DECISION {
      string variation_id PK
      string value
      string reason
      datetime decided_at
    }
    SUMMARY {
      string session_id PK
      string text
      string generated_by
      string validation_status
      string validated_decisions_hash
    }
```

## E. Interfaces portables

```python
class SegmentationService(Protocol):
    def segment(self, image: ImageArtifact) -> MaskArtifact: ...

class GenerationBackend(Protocol):
    def generate(self, request: GenerationRequest) -> list[GeneratedArtifact]: ...

class ArtifactStore(Protocol):
    def save(self, data: bytes, key: str) -> str: ...

class DecisionRepository(Protocol):
    def list_for_session(self, session_id: str) -> list[Decision]: ...

class SummaryService(Protocol):
    def summarize(self, session_id: str) -> Summary: ...
```

*Especificaciones conceptuales*: La implementación deberá concretar los tipos en el repositorio. No copiar interfaces sin implementar semánticas de errores, configuración e idempotencia definidas en el contrato.

## F. Mapeo de arquitecturas: GCP y local

GCP es el despliegue principal y la arquitectura local es su respaldo (ver ADR-003). Ambas usan el mismo dominio, la misma API `/api/v1` y las interfaces de la sección E; cambian los adaptadores. Ninguna está implementada todavía.

| Componente | GCP (principal) | Local (respaldo) | Interfaz común |
|---|---|---|---|
| UI y API | Cloud Run CPU: Gradio + FastAPI | Uvicorn: Gradio + FastAPI | Contrato `/api/v1` |
| Ejecución de trabajos | Cloud Tasks invoca al worker | Ejecutor aislado en el mismo proceso | Semántica `202` y estados de sesión |
| Segmentación e inferencia | Cloud Run GPU (L4 inicial) | GPU local o Colab, según medición | `SegmentationService`, `GenerationBackend` |
| Sesiones y decisiones | Firestore | SQLite | `SessionRepository`, `DecisionRepository` |
| Imágenes y máscaras | Cloud Storage | `data/assets/` | `ArtifactStore` |
| Pesos LoRA | Cloud Storage | Copia local versionada | Manifiesto con revisión y checksum |
| Entrenamiento LoRA | Colab Pro, fuera de GCP | Colab Pro, fuera de GCP | Mismo adaptador exportado |
| Agente | Gemini en Vertex AI con identidad de servicio | Gemini en Vertex AI con ADC; requiere internet | `SummaryService` y validador determinista |
| Credenciales | Identidades de servicio y Secret Manager | ADC y `.env` no versionado | Configuración por entorno |
| Contenedores | Artifact Registry | Build local, si se usa Docker | Mismo código |
| Registros | Cloud Logging y Monitoring | stdout estructurado | Mismos eventos |

La configuración de desarrollo del Hito 1 (`GENERATION_BACKEND=mock`, `SUMMARY_BACKEND=deterministic`) es distinta del respaldo: el mock no cuenta como enfoque académico ni el resumen determinista como agente.

### GCP (principal)

```mermaid
flowchart TD
    USER["Navegador"] --> WEB["Cloud Run CPU: Gradio + FastAPI"]
    WEB --> STORE[("Cloud Storage")]
    WEB --> META[("Firestore")]
    WEB --> TASK["Cloud Tasks"]
    TASK --> GPU["Cloud Run GPU: segmentación e inferencia"]
    GPU --> STORE
    GPU --> META
    WEB --> AGENT["Agente Python de solo lectura"]
    AGENT --> GEM["Vertex AI: Gemini"]
    AGENT --> META
    COLAB["Colab Pro: LoRA offline"] --> ART["LoRA versionado"]
    ART --> STORE
    STORE --> GPU
```

### Local (respaldo)

```mermaid
flowchart TD
    USER["Navegador"] --> WEB["Uvicorn: Gradio + FastAPI"]
    WEB --> FILES[("data/assets/")]
    WEB --> DB[("SQLite")]
    WEB --> EXEC["Ejecutor aislado"]
    EXEC --> GPU["GPU local o Colab: segmentación e inferencia"]
    GPU --> FILES
    GPU --> DB
    WEB --> AGENT["Agente Python de solo lectura"]
    AGENT --> GEM["Vertex AI: Gemini"]
    AGENT --> DB
    COLAB["Colab Pro: LoRA offline"] --> ART["LoRA versionado"]
    ART --> FILES
    FILES --> GPU
```

### Niveles de respaldo para la demo

1. **GCP:** modelo precargado antes de presentar.
2. **Local:** misma aplicación con modelos reales, si la medición confirma que la GPU es suficiente.
3. **Video grabado y sesión ya generada:** sirven como evidencia si fallan los anteriores, pero no sustituyen la demostración en vivo.

### Diferencias que ambos adaptadores deben cubrir

- **Lote atómico:** una sesión solo pasa a `READY_FOR_REVIEW` con cinco artefactos y registros, sin depender de transacciones SQL.
- **Reintentos:** Cloud Tasks puede repetir entregas, así que el worker debe ser idempotente; el ejecutor local debe cumplir el mismo contrato.
- **Trabajos interrumpidos:** un reinicio del proceso o del worker no debe dejar la sesión en `GENERATING` indefinidamente.
- **Almacenamiento:** Cloud Run tiene disco efímero, por lo que no debe depender de rutas locales.
