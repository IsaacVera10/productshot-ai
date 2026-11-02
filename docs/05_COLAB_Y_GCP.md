# 05 — Entrenamiento Colab Pro y despliegue GCP

## Objetivo de despliegue (no implementado todavía)

| Servicio | Propósito | Alternativa local |
|---|---|---|
| Colab Pro | Entrenar LoRA offline, exportar pesos y registro | GPU propia/Khipu autorizada |
| Cloud Run CPU | FastAPI + Gradio, agente/tool orchestration | Uvicorn local |
| Cloud Run GPU (L4 inicial) | Segmentar y generar cinco imágenes | Worker local con GPU/Colab para pruebas |
| Cloud Storage | Fotos originales, máscaras, variantes, pesos LoRA | Directorio `data/assets/` |
| Firestore | Sesiones, trabajos, decisiones, metadatos | SQLite |
| Vertex AI + Gemini | Modelo fundacional que usa el agente | Servicio mock/LLM local de prueba |
| Artifact Registry | Contenedores Docker | Build local |
| Secret Manager | Claves y configuración sensible | `.env` local no versionado / ADC |
| Cloud Logging/Monitoring | Métricas, errores y diagnóstico | stdout estructurado |

## Diagramas de despliegue

El mapeo completo y los diagramas de las arquitecturas GCP (principal) y local (respaldo) están en [01 — Arquitectura, sección F](01_ARQUITECTURA.md#f-mapeo-de-arquitecturas-gcp-y-local).

## Flujo de entrenamiento — fuera de GCP

1. Descargar solo datos permitidos y definir split train/validation/test separado **por producto/identidad** para reducir fuga.
2. Guardar manifiesto `datasets.csv` (id, origen, licencia, hash, split, categoría, caption).
3. Probar checkpoint compatible y entorno Colab; verificar VRAM real antes de iniciar entrenamiento.
4. Registrar `base_model_id`, **revisión exacta**, scripts/dependencias, LoRA rank/alpha/LR/steps/seed, curvas de pérdida, checkpoint y resultados en validación.
5. Guardar checkpoints para reanudar y **exportar únicamente los adaptadores necesarios**, con licencias y documentación. No depender de una sesión de Colab encendida.
6. Publicar un paquete de pesos/manifest en Cloud Storage (o copiar localmente para pruebas), **solo tras verificar compatibilidad** de `load_lora_weights` con el pipeline elegido.

**Colab Pro no garantiza tipo de GPU ni disponibilidad ilimitada.** Ver FAQ oficial: https://research.google.com/colaboratory/faq.html?hl=es

## Flujo de GCP — segunda etapa

1. Establecer proyecto de prueba, región, presupuesto y alertas; no asumir créditos gratuitos.
2. Crear bucket privado y Firestore, definir identidades de servicio de permisos mínimos.
3. Crear imágenes de contenedor reproducibles (API CPU y worker GPU por separado), publicar en Artifact Registry.
4. Configurar Cloud Run CPU con API/UI; almacenamiento permanente externo (Cloud Run tiene filesystem efímero).
5. Configurar GPU Cloud Run con L4 u otra GPU compatible, **min instances = 0** cuando aplique, **max instances = 1** inicialmente, CPU/RAM mínimos del tipo GPU, límite de concurrencia y timeout basados en benchmark. Confirmar región y cuotas.
6. Generación asíncrona con trabajos: `POST /generate` crea job -> Cloud Tasks invoca worker -> worker valida y guarda cinco artefactos -> confirma estado `READY_FOR_REVIEW`. Proteger endpoint worker con IAM, implementar reintentos seguros (idempotentes).
7. Activar Gemini a través de Vertex AI con identidad de servicio; el LLM no corre en GPU propia.
8. Validar una demo de extremo a extremo y apagar los recursos no necesarios tras pruebas.

Cloud Run GPU admite GPU L4 de 24 GB, puede reducir instancias a cero, pero **la factibilidad de nuestro modelo y sus tiempos solo se conocen tras benchmarks**. Documentación: https://docs.cloud.google.com/run/docs/configuring/services/gpu

## Costos y protección

- **Antes de crear GPU**: revisar precio efectivo por región, cuota, costos de inicio/carga e inactividad, facturación de Gemini, GCS y red. No poner valores monetarios inventados en propuesta.
- Presupuesto + alertas **no son un apagado automático**. Configurar topes de instancias, autenticación y protección del endpoint para minimizar generación maliciosa/repetitiva.
- La cuota y `max-instances=1` limitan escalado, pero **no garantizan un tope monetario total** si la instancia opera de forma sostenida.
- Cachear adaptadores/modelos cuando sea seguro; evitar bajar pesos enormes en cada request.
- En la demo, considerar solicitudes preautorizadas para evitar abuso público.

## Secretos e identidad

- Nunca versionar JSON de service account, refresh tokens ni claves Gemini.
- En GCP usar identidades de servicio y ADC, roles mínimos en Storage, Firestore y Vertex AI.
- En local usar ADC de desarrollo si se autoriza; `SUMMARY_BACKEND=deterministic` en Hito 1.
- Impedir acceso público a buckets y a archivos de sesiones de otros usuarios. Crear borrado/retención de datos de usuario.

## Punto de decisión

Si la prueba de inferencia de cinco fotos supera memoria/latencia/costo tolerable de Cloud Run GPU, considerar otra GPU/tamaño de modelo o despliegue por batch en Compute Engine como contingencia. **La nube no determina por adelantado el checkpoint.**
