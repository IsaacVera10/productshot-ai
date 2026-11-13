# 05 — Entrenamiento (Colab Pro, Kaggle) y despliegue (Modal, alternativa GCP)

**Estado (2026-11-12):** decidido en ADR-009; nada desplegado todavía. Los precios son de una consulta del 2026-11-12, varios de fuentes secundarias: confirmarlos al registrarse.

## Resumen

| Necesidad | Principal | Respaldo o alternativa | Costo esperado |
|---|---|---|---|
| Entrenar el LoRA | Colab Pro de Pedro (~140 unidades de cómputo) | Kaggle (gratis, unas 30 h de GPU por semana) | Unidades ya pagadas |
| Servir la generación (ambos modos) | Modal, GPU L4 de 24 GB, escala a cero | Laptop con RTX 4060 de 8 GB; Cloud Run con GPU | Dentro de los USD 30 gratis al mes de Modal |
| UI y API | Modal, endpoint web sin GPU | Laptop; Cloud Run sin GPU | Dentro de los mismos créditos |
| Agente | Gemini por AI Studio (plan gratuito) | Gemini en Vertex AI (solo con GCP) | 0 |
| Segundo LLM | Modelo abierto gratuito en la nube | Ollama en la laptop | 0 |
| Pesos y dataset | Repositorios privados de Hugging Face Hub | Cloud Storage | 0 |

El mapeo completo por pieza está en [01 — Arquitectura, sección F](01_ARQUITECTURA.md#f-mapeo-de-arquitecturas-modal-gcp-y-local).

## Entrenamiento del LoRA

### Dónde

- **Colab Pro (principal).** Cuenta de Pedro, con unas 140 unidades de cómputo. Permite GPU de 24 GB o más (L4, A100), cómodas para SDXL a 1024 px. El consumo por hora depende de la GPU y es mucho mayor en A100 que en L4 o T4: revisar en el panel de Colab cuántas unidades por hora gasta la GPU asignada antes de cada corrida larga. **Las unidades vencen** (habitualmente a los 90 días de la compra): confirmar la fecha con Pedro y planificar el entrenamiento antes.
- **Kaggle (respaldo gratuito).** T4 o P100 de 16 GB, unas 30 h por semana y sesiones de hasta 12 h. SDXL a 1024 px entra con `fp16`, *gradient checkpointing*, Adam en 8 bits y lote de 1, pero es lento. Sirve para el piloto o si se agotan las unidades.
- **Modal (integración).** El smoke test de ADR-004 termina cargando el adaptador en el worker de Modal, que es donde se servirá: si no carga o no se ve en el pipeline de inpainting, se detecta ahí. Entrenar en Modal también es posible con los créditos (unos USD 1.5 a 3 por corrida en L4), pero compite con la inferencia.
- **Laptop:** 8 GB no alcanzan para entrenar SDXL.

**Colab Pro no garantiza tipo de GPU ni disponibilidad ilimitada.** FAQ oficial: https://research.google.com/colaboratory/faq.html?hl=es

### Flujo

1. Descargar solo datos permitidos y definir el split train/validation/test separado **por producto/identidad** para reducir fuga.
2. Guardar el manifiesto `datasets.csv` (id, origen, licencia, hash, split, categoría, caption).
3. Probar el checkpoint y el entorno; verificar la VRAM real antes de iniciar el entrenamiento.
4. Registrar `base_model_id`, **revisión exacta**, scripts y dependencias, rank, alpha, LR, pasos, semilla, curvas de pérdida, checkpoint y resultados en validación.
5. Guardar checkpoints para reanudar (Kaggle corta a las 12 h y Colab por inactividad) y **exportar solo los adaptadores**.
6. Publicar el adaptador y su manifiesto (revisión, checksum, checkpoint base) en un **repositorio privado de Hugging Face Hub**. El worker lo carga desde `LORA_ADAPTER_URI`, así que da igual dónde se entrenó.
7. El dataset curado también va privado: BG60k es solo para uso académico y falta confirmar con sus autores si se pueden publicar ejemplos y pesos (P6).

## Despliegue principal: Modal

1. Crear la cuenta en modal.com (puede ser con GitHub) y `modal setup`. Si pide tarjeta, **fijar el límite de gasto en USD 30**, para no pasar de los créditos gratuitos.
2. Worker con GPU L4: el mismo código para ambos modos (SDXL base con inpainting de cuatro canales; el LoRA solo en `advanced`), pesos cacheados en un volumen para no descargarlos en cada arranque.
3. UI y API como endpoint web sin GPU, con `max_containers=1` mientras la base sea SQLite en un volumen.
4. La API encola la generación con `spawn`; el worker guarda los cinco artefactos y la API confirma `READY_FOR_REVIEW` con las mismas validaciones de hoy (lote de cinco o nada, G8, G14).
5. Secretos de Modal para la clave de Gemini y el token de Hugging Face.
6. Para la demo: un contenedor mínimo del worker encendido solo durante la presentación, y volver a 0 al terminar.

### Comandos (implementado el 2026-11-13)

El worker está en `deploy/modal/worker.py` y usa `SdxlInpaintBackend` (`backend/productshot/services/sdxl_generation.py`) detrás del mismo contrato HTTP que el mock.

```bash
# una vez: el token lo comparten la API (.env) y el worker (secreto de Modal)
modal secret create productshot-worker GENERATION_WORKER_TOKEN=<token>

PYTHONPATH=backend modal run deploy/modal/worker.py      # prueba de humo: un lote de cinco, deja PNG y métricas en data/smoke/
PYTHONPATH=backend modal deploy deploy/modal/worker.py   # endpoint permanente; escala a cero a los 2 minutos sin uso
```

Con el endpoint desplegado, la API local o la de Modal lo usa con `GENERATION_BACKEND=remote` y `GENERATION_WORKER_URL=<URL del endpoint>`. Los pesos quedan en el volumen `productshot-hf-cache` y la revisión exacta del checkpoint (SHA) se registra en cada variación.

### Prueba de humo en Modal (2026-11-13)

Un lote de cinco con `stabilityai/stable-diffusion-xl-base-1.0@462165984030d82259a11f4367a4eed129e94a7b`, pipeline de inpainting de cuatro canales, fp16, 30 pasos, guidance 7.0, strength 0.99, a 1024×768, sobre un producto sintético (una taza roja plana dibujada con su máscara exacta):

| Medida | Resultado |
|---|---|
| GPU | NVIDIA L4, 22.1 GB utilizables |
| Pico de VRAM | 9.5 GB por imagen (sobra margen en la L4; no entra tal cual en los 8 GB de la RTX 4060: haría falta `enable_model_cpu_offload`) |
| Carga del modelo | 55 s con los pesos ya en el volumen (la primera descarga suma unos 45 s) |
| Lote de cinco | 50 s, unos 10 s por imagen |
| Deriva del producto antes de recomponer (G8) | 3.1 a 6.2 de 255; la recomposición la lleva a 0 |
| Costo | Unos minutos de L4, menos de USD 0.10 |

Hallazgo de calidad: con el producto sintético, el modelo **prolonga la silueta del producto** en el fondo (un telón rojo detrás en estudio, un vaso en natural, una tetera en lifestyle, una tarjeta blanca en minimalista). Es el problema conocido de expansión de objeto en inpainting y es lo que el Hito 2 debe medir con fotos reales de los casos obligatorios. Opciones a probar en ese orden: describir el producto en el prompt, agrandar un poco la máscara protegida o difuminar menos su borde, bajar `strength` y, si no alcanza, el UNet de nueve canales (P4). Las imágenes y `metrics.json` quedan en `data/smoke/` (no versionado).

Costo esperado: unas 20 h de L4 para todo el proyecto (desarrollo, prueba con 10 usuarios, ensayos y demo) a cerca de USD 0.80 por hora, dentro de los USD 30 gratis de un mes. Es una estimación: el smoke test medirá el costo real por lote.

## Alternativa: GCP

Se mantiene compatible por si el equipo consigue créditos (por ejemplo, educativos) o quiere la demo en Cloud Run. **No hay créditos hoy:** las cuentas de facturación de Isaac están cerradas y `planifica-proyecto` no tiene facturación vinculada.

1. Proyecto con presupuesto y alertas; no asumir créditos gratuitos. La prueba gratuita no da GPU: hay que pasar a cuenta de pago para pedir cuota de L4, y la aprobación puede tardar días.
2. Bucket privado y Firestore con identidades de servicio de permisos mínimos.
3. Imágenes de contenedor reproducibles (API y worker por separado) en Artifact Registry.
4. Cloud Run sin GPU para la API y la UI, con almacenamiento externo (el disco de Cloud Run es efímero).
5. Cloud Run con GPU L4 sin redundancia zonal, **min instances = 0**, **max instances = 1**, timeout según el benchmark.
6. `POST /generate` crea el trabajo y Cloud Tasks invoca al worker, protegido con IAM e idempotente.
7. Gemini por Vertex AI con identidad de servicio (el mismo SDK, `vertexai=True`).

Costo estimado (consulta del 2026-11-12): L4 a USD 0.0001867 por segundo, cerca de USD 1.05 por hora con el CPU y la memoria mínimos; la instancia sigue cobrándose unos minutos tras la última solicitud. Para el uso del proyecto, unos USD 25 a 30. **El riesgo es una instancia olvidada:** unos USD 25 al día.

Documentación: https://docs.cloud.google.com/run/docs/configuring/services/gpu y https://cloud.google.com/run/pricing

## Protección de costos (cualquier opción)

- Las alertas de presupuesto **no apagan nada**: en GCP hace falta una función que desactive la facturación al pasar el tope; en Modal, el límite de gasto de la cuenta.
- Nunca dejar contenedores o instancias mínimas encendidas fuera de una demo.
- Máximo de un worker con GPU a la vez y endpoints protegidos, para que nadie genere en masa.
- Cachear pesos en un volumen o en la imagen; no descargarlos en cada solicitud.

## Secretos e identidad

- Nunca versionar claves de Gemini, tokens de Hugging Face, JSON de cuentas de servicio ni refresh tokens.
- En la laptop: `.env` no versionado. En Modal: secretos de Modal. En GCP: identidades de servicio, ADC y Secret Manager.
- El plan gratuito de Gemini permite a Google usar lo enviado para mejorar sus productos: al agente solo van decisiones y motivos, nunca fotos, y se declara en el consentimiento de la prueba con usuarios (`06_EVALUACION_Y_ETICA.md`).
- Repositorios de pesos y datos privados; sin acceso público a archivos de sesiones de otros usuarios; borrado con `DELETE /sessions/{id}`.

## Punto de decisión

Si el lote de cinco no entra en memoria o tarda demasiado en una L4, probar menos pasos, menor resolución o una GPU mayor en Modal antes de cambiar de checkpoint. **La infraestructura no determina por adelantado el checkpoint.**
