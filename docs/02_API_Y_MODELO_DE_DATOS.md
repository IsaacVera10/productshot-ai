# 02 — API v1 y contratos de datos

**Prefijo**: `/api/v1`. **Estado de estas rutas**: implementadas con backends simulados; ver la sección «Implementación del Hito 1» al final. La documentación interactiva (OpenAPI) está en `/docs` con el servidor en marcha. Preferir UUID v4/UUID7, timestamps ISO-8601 UTC y validación estricta con Pydantic. Objetos con identificadores opacos; no exponer rutas absolutas locales ni credenciales.

## Endpoints

| Método | Ruta | Entrada | Salida | Notas |
|---|---|---|---|---|
| GET | `/healthz` | — | 200 `{"status":"ok"}` | No invoca LLM/GPU |
| POST | `/sessions` | `multipart/form-data` `image`, `mode` | 201 Session | Tamaño / MIME / decodificación válidos |
| POST | `/sessions/{id}/segment` | `{}` | 200 máscara/URI y estado `SEGMENTED` | En mock puede ser máscara de demostración rotulada |
| POST | `/sessions/{id}/mask` | `multipart/form-data` `mask` (PNG) | 200 máscara/URI, vista previa y estado `SEGMENTED` | Alternativa a `/segment`; solo en `UPLOADED` o `SEGMENTED` y en la ronda 1 |
| POST | `/sessions/{id}/generate` | `{}` + `Idempotency-Key` opcional | 202 `job_id`, estado | Solo tras máscara; el modo fue elegido al crear sesión |
| GET | `/sessions/{id}` | — | 200 sesión + estado + variaciones (si completas) | Polling para UI |
| PUT | `/sessions/{id}/variations/{vid}/decision` | Decision | 200 decisión actualizada | Solo si variación pertenece a sesión y está lista |
| DELETE | `/sessions/{id}` | — | 204 | Borra la cadena completa de rondas (registros y archivos); 409 si alguna ronda está generando (G11) |
| POST | `/sessions/{id}/summary` | `{}` | 200 Summary | Solo tras cinco decisiones; requiere validar |
| POST | `/sessions/{id}/next-round/preview` | `{"adjustment_note": str \| null}` | 200 RoundPlan | Plan de la ronda siguiente sin crearla (ADR-007) |
| POST | `/sessions/{id}/next-round` | `{"adjustment_note": str \| null}` | 201 Session hija en `SEGMENTED` | Solo con la madre `SUMMARIZED`; 409 si ya tiene hija o es la ronda 3 |

La máscara manual debe ser un PNG decodificable, dentro de los límites de tamaño, con las dimensiones del lienzo de trabajo y la semántica blanco editable/negro protegido. Reintentar la segmentación o subir una máscara reemplaza la anterior solo en `UPLOADED` o `SEGMENTED`; al iniciar la generación, la máscara queda bloqueada de forma atómica. Se registran su origen (`auto | manual`) y su hash.

En el modo local, la generación mock podrá completarse en el mismo proceso mediante un ejecutor aislado, **sin prometer** procesamiento distribuido. La API seguirá la semántica `202` para evitar modificar el contrato al desplegar un worker en GCP. Para cloud, especificar integración de Cloud Tasks o equivalente en una fase separada.

## Entidades y ejemplos

### Session

```json
{
  "id": "30bb650c-4cb0-48c4-99ef-e10e920212a6",
  "mode": "intermediate",
  "status": "UPLOADED",
  "image_id": "7d3ee36b-7341-411e-804a-5dc653fd4dc6",
  "created_at": "2026-11-02T00:00:00Z",
  "variations": []
}
```

`mode` es `intermediate | advanced`; **nunca** `mock` en dominio: `mock` se configura como backend de ejecución local. Debe quedar rotulado visualmente que es simulación.

### Job

```json
{
  "job_id": "9d652ba5-d320-4a75-a8c7-59f03796bb6c",
  "session_id": "30bb650c-4cb0-48c4-99ef-e10e920212a6",
  "status": "GENERATING"
}
```

### Variation

```json
{
  "id": "a743e227-c1eb-41b9-9391-010aee7a17d1",
  "position": 1,
  "style": "studio",
  "image_url": "/api/v1/assets/safe-relative-identifier",
  "generation": {"seed": 12345, "model_revision": "mock-v1", "prompt_version": "v1"},
  "decision": null
}
```

El motor debe producir exactamente una variación para cada estilo de la lista canónica `studio, natural, warm, lifestyle, minimal`, en posiciones 1–5, con IDs únicos. La semilla puede ser distinta para cada variación, pero registrarla. Una nueva sesión es necesaria para otro modo.

### Decision

```json
{"decision": "rejected", "reason": "El fondo no combina con mi marca"}
```

`decision`: `accepted | rejected`. `reason`: `null` o cadena opcional con límite razonable, tratada solo como **dato**; ninguna cadena proporcionada por el usuario puede instruir al agente a ejecutar operaciones no autorizadas.

### Summary

```json
{
  "session_id": "30bb650c-4cb0-48c4-99ef-e10e920212a6",
  "accepted_ids": ["id1", "id3", "id5"],
  "rejected_ids": ["id2", "id4"],
  "summary": "Se aceptaron las variaciones 1, 3 y 5; se rechazaron 2 y 4.",
  "preferences_supported_by_user_reasons": [],
  "limitations": ["No se registraron motivos para todas las decisiones"],
  "generated_by": "llm_validated",
  "validation_status": "structure_verified"
}
```

Los IDs de arriba son **ilustrativos**, no UUID válidos; la implementación los generará automáticamente. El agente consulta los registros y produce un resumen estructurado con referencias a variaciones y motivos. Python construye `accepted_ids` y `rejected_ids` desde la DB, verifica cobertura, decisiones y citas asociadas al registro correcto, y renderiza la enumeración. Los motivos se citan literalmente.

`generated_by` indica la procedencia:

| Valor | Significado |
|---|---|
| `llm_validated` | La salida estructurada del LLM pasó la verificación sin correcciones |
| `hybrid` | Se usó la salida del LLM con partes corregidas o sustituidas por Python |
| `deterministic_fallback` | El LLM falló o su salida se descartó; el resumen lo compuso Python y no cuenta como agente |

`validation_status` es `structure_verified` cuando la parte estructurada coincide con los registros, o `not_verified` en caso contrario; un resumen `not_verified` no se presenta como válido. Este estado no garantiza la narrativa libre: la revisión automática no detecta negaciones, alusiones ni contradicciones arbitrarias en prosa.

## Máquina de estados e invariantes

- `UPLOADED -> SEGMENTED -> GENERATING -> READY_FOR_REVIEW -> REVIEWED -> SUMMARIZED`. `GENERATING -> FAILED` ante cualquier error.
- **Éxito de generación** = cinco archivos realmente existentes + cinco registros persistidos + estilos distintos. Si no, `FAILED`.
- **Una sesión = una generación**; POST duplicados devuelven 409 salvo reintento idempotente de la misma operación. Otra ronda es otra sesión (hija), con su propia generación.
- Se permite cambiar las decisiones mientras no esté `SUMMARIZED`; tras resumen quedan congeladas.
- Un resumen **no se solicita** hasta cinco decisiones registradas, incluyendo caso `0 accepted`.
- En errores HTTP: 400 para contenido inválido no capturado por validación, 404 ID inexistente, 409 conflicto de estado, 413 archivo excesivo, 422 payload inválido o foto rechazada por los guardrails de entrada, 429 límite de generaciones alcanzado, 503 dependencia LLM/GPU caída; usar estructura de error coherente con un mensaje en español para la persona usuaria y la acción siguiente (ver `08_ITERACIONES_Y_GUARDRAILS.md`, parte B). 422 y 429 por guardrails llegan en el Hito 2.

## Persistencia local

- `data/productshot.sqlite3` (SQLite, migraciones preferibles cuando crezca).
- `data/assets/<session_id>/input.png`, `mask.png`, `variation_1.png` ... `variation_5.png`.
- `data/` siempre ignorado por Git y creado en ejecución.
- Los archivos deben servirse por una ruta controlada que valide pertenencia/autorización; **no** concatenar paths arbitrarios a partir de input del usuario.
- En cloud, `SessionRepository` -> Firestore y `ArtifactStore` -> Cloud Storage. Nunca persistir sesiones/archivos críticos únicamente en disco efímero de Cloud Run.

## Implementación del Hito 1 (2026-11-04)

Código en `backend/productshot/`. Decisiones de implementación que concretan este contrato:

- **Errores:** todos con la forma `{"error": {"code": "...", "message": "...", "details": [...]}}`. Códigos: `invalid_input` (400), `not_found` (404), `state_conflict` (409), `payload_too_large` (413), `validation_error` (422), `dependency_unavailable` (503). Un formato real distinto de JPG/PNG o un archivo no decodificable devuelve 400; un `mode` inválido, 422.
- **Artefactos:** `GET /api/v1/assets/{session_id}/{nombre}` sirve **solo** claves que la base de datos asocia a esa sesión; cualquier otra ruta responde 404. Las claves se validan con una expresión estricta; nunca se concatena input del usuario a una ruta. No hay autenticación en el Hito 1. El original se guarda sin metadatos (EXIF, GPS, texto PNG) y con la orientación aplicada; `sha256` corresponde al archivo tal como se subió.
- **Session** añade `image_url`, `updated_at`, `failure_reason`, `simulated` (true si la máscara o el trabajo de ESTA sesión se hicieron con un backend mock; se toma del registro, no de la configuración actual del servidor), `mask` (`id`, `source`, `url`, `preview_url`, `simulated`), `job` y `summary_available`. `variations` solo se rellena desde `READY_FOR_REVIEW`.
- **Máscara:** cada reintento de `/segment` crea archivos nuevos y borra los anteriores; la máscara queda bloqueada en cuanto la sesión sale de `SEGMENTED`.
- **Generación:** `202` con `{job_id, session_id, status, backend, error}`. Sin `Idempotency-Key`, un segundo POST da 409; con la misma llave devuelve el mismo trabajo. En local el trabajo corre en un hilo aparte (un trabajo a la vez). Si el trabajo no se puede encolar, se marca `FAILED` y la respuesta es 503. Al arrancar, los trabajos que quedaron en `GENERATING` se marcan `FAILED`.
- **Lote:** el orquestador exige 5 artefactos, posiciones 1–5, estilos canónicos en orden y las semillas pedidas; si no, borra los archivos escritos y marca `FAILED`.
- **Decisión:** `reason` se guarda en una sola línea (saltos de línea, tabulaciones y caracteres de control pasan a un espacio, se colapsan los espacios y se quitan los controles bidireccionales), y una cadena vacía equivale a `null`. El límite de 500 caracteres se mide sobre el texto ya normalizado; el texto crudo tiene un tope de 2000. Superar cualquiera de los dos, o enviar campos extra en el cuerpo, da 422.
- **Máscara manual:** PNG real (no basta la extensión), del tamaño exacto del lienzo de trabajo, en cualquier modo de color (se pasa a gris y los bordes suaves se conservan). Se rechaza con 400 si es toda blanca (no protege nada) o toda negra (no deja fondo). Queda con `source: manual` y `simulated: false`, y reemplaza a la automática borrando sus archivos.
- **Borrado:** `DELETE` sobre cualquier ronda borra toda la cadena, porque las rondas comparten la foto y la máscara de la ronda 1; borrar solo una dejaría a las otras sin archivos. Después de borrar, la sesión y sus imágenes responden 404.
- **Rondas (ADR-007):** Session añade `round_index` (1 a 3), `parent_session_id`, `next_session_id` y `round_plan` (`items` con `position`, `style`, `variant`, `label`, `adjustments` y `because`, más `explanation`, una línea por variación). Variation añade `variant` (estilo más ajustes del vocabulario cerrado, por ejemplo `warm+brighter`; en la ronda 1 es el estilo) y `label`. La invariante del lote pasa a cinco `variant` distintos. La nota de ajuste sigue las reglas del motivo (una línea, 500 caracteres tras normalizar) y nunca llega al prompt. Detalle en `08_ITERACIONES_Y_GUARDRAILS.md`, parte A, «Implementación».
- **Summary:** `preferences_supported_by_user_reasons` es una lista de `{variation_id, position, style, decision, reason}` con los motivos **literales**. En el Hito 1 `generated_by` es siempre `deterministic_fallback` (no hay LLM) y se agrega una limitación que lo declara. Un segundo POST devuelve el mismo resumen, también si llega en paralelo al primero. Al guardar se verifica que las decisiones no cambiaron mientras se calculaba (hash).
