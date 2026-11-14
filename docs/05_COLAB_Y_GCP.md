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

- **Colab Pro (principal).** Cuenta de Pedro: 217.84 unidades el 2026-11-14; la L4 gasta 1.94 por hora. Se maneja desde la laptop con el Google Colab CLI (`training/README.md`). Permite GPU de 24 GB o más (L4, A100), cómodas para SDXL a 1024 px. El consumo por hora depende de la GPU y es mucho mayor en A100 que en L4 o T4: revisar en el panel de Colab cuántas unidades por hora gasta la GPU asignada antes de cada corrida larga. **Las unidades vencen** (habitualmente a los 90 días de la compra): confirmar la fecha con Pedro y planificar el entrenamiento antes.
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
7. El dataset curado también va privado: BG60k es solo para uso académico, y el proyecto lo es; no se consultará a sus autores (P6).

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

PYTHONPATH=backend modal run deploy/modal/worker.py::main  # prueba de humo: un lote de cinco, deja PNG y métricas en data/smoke/
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

### Experimento de expansión de silueta (2026-11-13)

`PYTHONPATH=backend modal run deploy/modal/worker.py::experiment`. Cuatro fotos de producto generadas por el mismo SDXL (taza con logo, jarrón mate, frasco blanco sobre fondo claro, caja con texto), máscara de `rembg` (`isnet-general-use`), estilos estudio y lifestyle, una sola semilla. Métrica: fracción del área del producto que `rembg` detecta como primer plano NUEVO en una franja de 4 a 40 px alrededor del producto, ya recompuesto (0 = no se prolongó la silueta).

| Configuración | Expansión media | Máxima | Lo que se ve |
|---|---|---|---|
| A: base (strength 0.99, borde difuminado 8 px) | 0.130 | 0.252 | Siluetas extra pegadas al producto en varios casos |
| B: categoría del producto al inicio del prompt | 0.164 | 0.360 | **Peor en estudio:** dibuja un segundo producto detrás (taza doble, jarrón doble) |
| C: margen protegido de 12 px sin difuminar | 0.139 | 0.255 | Menos siluetas, pero queda un halo del fondo original alrededor del producto |
| D: strength 0.85 | 0.116 | 0.214 | Conserva más el fondo original; a veces agrega objetos detrás (una tapa dorada) |
| E: B + C | 0.121 | 0.279 | Muy bien en lifestyle, mal en estudio |

Conclusiones, con la cautela de 8 imágenes por configuración y una semilla: **ninguna mitigación sencilla resuelve la expansión con el UNet de cuatro canales**; nombrar el producto en el prompt la empeora en estudio, y el margen la cambia por un halo. Siguiente experimento: el UNet de inpainting de nueve canales (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, P4) y, como referencia, generar el fondo sin producto y componer encima. Si el de nueve canales gana con claridad, hay que revisar ADR-004 (el LoRA se entrena sobre SDXL base). `rembg` separa bien la taza y el frasco, pero en el jarrón incluyó las flores y un cuenco aparte (G13 cuenta componentes y avisa). Las imágenes quedan en `data/experiments/expansion/` (no versionado); el experimento costó unos 15 minutos de L4.

### UNet de nueve canales y fondo compuesto (2026-11-14)

`PYTHONPATH=backend modal run --detach deploy/modal/worker.py::experiment --config-set nueve`, con las mismas cuatro fotos, dos semillas (500 y 900) y los estilos estudio y lifestyle. La corrida llegó al tope de 15 minutos de la función cuando iba por el UNet de nueve canales: base, `strength` 0.85 y fondo compuesto quedaron completos (16 imágenes cada uno) y el de nueve canales con 6 (taza y jarrón). Desde entonces cada resultado se guarda en el volumen apenas sale y el tope es de 60 minutos.

| Configuración | Expansión media (n) | Mismo subconjunto que nueve canales (n = 6) | Lo que se ve |
|---|---|---|---|
| A: base, cuatro canales | 0.117 (16) | 0.104 | Siluetas extra y objetos pegados detrás del producto |
| D: `strength` 0.85 | 0.091 (16) | 0.114 | Igual que A; a veces una tapa u objeto detrás |
| H: fondo generado sin producto y producto pegado encima | 0.005 (16) | 0.008 | Sin expansión por construcción, pero **el producto flota**: sin contacto ni sombra, con la perspectiva de la mesa equivocada |
| F: UNet de nueve canales (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`) | 0.054 (6) | 0.054 | Taza: sin siluetas extra y con sombra creíble. Jarrón: el cuerpo no se duplica, pero **el ramo crece** más allá de su contorno original |
| G: nueve canales con `strength` 0.85 | No se corrió | | |

Por producto, la media esconde resultados opuestos:

| Caso | A: base | F: nueve canales |
|---|---|---|
| Taza (2 estilos × 2 semillas) | 0.160, 0.134, 0.122, 0.033 | 0.000, 0.001, 0.015, 0.003 |
| Jarrón (2 estilos, solo el lote con semilla base 500) | 0.069, 0.107 | 0.144, 0.162 |

En la taza, nueve canales gana con claridad. En el jarrón pierde: la métrica sube y a la vista el ramo de flores se extiende hacia los lados y hacia arriba. La máscara de `rembg` incluye las flores, así que ese crecimiento también es expansión del "producto". Con dos productos, imágenes generadas por el mismo SDXL y una o dos semillas, la diferencia es descriptiva, no concluyente.

Problemas de esta corrida, corregidos después en `worker.py`: la semilla registrada en `lifestyle` era la del lote y no la real (`seed + 1`: las imágenes marcadas `s500` y `s900` usaron 501 y 901); las fotos y máscaras se guardaron reducidas a 512 px; los SHA de los checkpoints solo se escribían al terminar, así que **no quedó registrada la revisión del UNet de nueve canales** usada aquí, y la de la base se supone igual a la de la prueba de humo (`4621659…`) pero ningún archivo de esta corrida lo confirma; y el experimento dejaba el pipeline base en CPU dentro del mismo contenedor que sirve la API. Ahora los experimentos corren en una clase aparte (`Lab`), siempre sobre SDXL base (verifica cuatro y nueve canales), escriben `manifest.json` antes de generar, guardan entrada a resolución completa y la semilla real de cada imagen, e intentan devolver la base a la GPU en un `finally` (si eso falla, el contenedor se descarta al apagarse).

### Nueve canales completo (2026-11-14, corrida `nueve_completo`)

`PYTHONPATH=backend modal run --detach deploy/modal/worker.py::experiment --config-set nueve_completo`: A, F y G sobre los cuatro casos, estilos estudio y lifestyle, semillas base 500 y 900 (n = 16 por configuración). Primera corrida con la clase `Lab` y `manifest.json`: SDXL base `462165984030d82259a11f4367a4eed129e94a7b`, nueve canales `115134f363124c53c7d878647567d04daf26e41e`, torch 2.5.1, diffusers 0.31.0, L4. Reproducible: A coincide en 16 de 16 imágenes con la corrida anterior y F en las 6 que ya existían. La app estuvo viva 28 minutos (10:21 a 10:49; el inpainting suma 7.8 min y el resto, sin medir por separado, incluye generar las fotos, cargar los modelos, `rembg` en CPU, guardar y bajar los resultados), así que la GPU costó cerca de USD 0.37 si la L4 estuvo activa todo ese intervalo; Modal también cobra CPU y memoria, y el costo total no está verificado en la facturación. Al terminar no quedó ningún contenedor (`modal container list`).

| Caso (media de 4) | A: base, cuatro canales, 0.99 | F: nueve canales, 0.99 | G: nueve canales, 0.85 |
|---|---|---|---|
| Taza | 0.112 | 0.005 | 0.019 |
| Jarrón (con flores) | 0.108 | 0.194 | 0.102 |
| Frasco blanco | 0.123 | 0.225 | 0.042 |
| Caja con texto | 0.123 | 0.062 | 0.051 |
| **Media (mediana)** | 0.117 (0.118) | 0.122 (0.070) | 0.054 (0.055) |
| Pares que gana frente a A | | 9 de 16 | 13 de 16 |

Contra D de la corrida anterior (cuatro canales con el mismo `strength` 0.85, media 0.091), G gana 12 pares, empata 1 y pierde 3; en el jarrón queda un poco peor (0.102 frente a 0.097). Bajar `strength` no explica toda la mejora, pero G cambia el checkpoint completo, no solo la capa de entrada: la ventaja es del **checkpoint de inpainting**, no se puede atribuir solo a los canales.

Lo que se ve en las grillas:

- **F es irregular:** bien en la taza y la caja, pero en lifestyle llena la escena de objetos pegados al producto (frasco, 0.52) y en el jarrón agranda mucho el ramo.
- **G integra mejor en estos ejemplos:** la caja y el frasco se apoyan en la superficie con sombra y sin siluetas extra. El ramo del jarrón todavía crece.
- **A inventa escenas que no cuadran con el producto:** el frasco acostado en diagonal delante de un estudio con focos, la caja que parece flotar delante de trípodes. F y G conservaron mejor el ángulo y la superficie en estos casos (observación, no propiedad general).
- **G pierde estilo:** conserva más del fondo original, y en la taza sus dos imágenes "estudio" son interiores domésticos en vez del fondo gris continuo del preset. Es el costo de bajar `strength`.

Límites: la métrica cuenta cualquier primer plano en la franja (también objetos ajenos y pliegues de tela) y **no penaliza conservar el fondo original**, así que favorece justo el comportamiento conservador de G. Es un indicador auxiliar hasta contrastarlo con anotación humana. Son cuatro productos sintéticos con observaciones repetidas, no 16 independientes, y la máscara del jarrón protege también las flores y un cuenco separado. Estas imágenes son de desarrollo: sirven para elegir parámetros y no pueden usarse en la evaluación final (`06_EVALUACION_Y_ETICA.md`). Esta corrida guardó las salidas a 512 px y sin la segmentación final. El código actualizado guardará en las próximas corridas las salidas a resolución completa, la segmentación final, la franja, numerador y denominador de la métrica, el prompt efectivo de cada imagen, el checksum del modelo de `rembg`, y el commit del código con su diff si había cambios sin commit.

### `strength` intermedio en nueve canales (2026-11-14, corrida `nueve_strength`)

`PYTHONPATH=backend modal run --detach deploy/modal/worker.py::experiment --config-set nueve_strength`, desde el commit limpio `5567737`: I (0.90) y J (0.95) con los mismos casos, estilos y semillas. A, F y G no se repitieron porque A y F ya se reprodujeron imagen a imagen. Primera corrida con la evidencia completa: la métrica se recalcula exacta desde los archivos guardados (segmentación final × franja) en 32 de 32 imágenes. La app estuvo viva 18.6 minutos (cerca de USD 0.25 de GPU si la L4 estuvo activa todo el intervalo; total sin verificar).

| Configuración | Pasos efectivos (de 30) | Taza | Jarrón | Frasco | Caja | Media (mediana) | Gana a A |
|---|---|---|---|---|---|---|---|
| A: base, 0.99 | 29 | 0.112 | 0.108 | 0.123 | 0.123 | 0.117 (0.118) | |
| F: nueve canales, 0.99 | 29 | 0.005 | 0.194 | 0.225 | 0.062 | 0.122 (0.070) | 9 de 16 |
| J: nueve canales, 0.95 | 28 | 0.026 | 0.166 | 0.093 | 0.062 | 0.087 (0.075) | 10 de 16 |
| I: nueve canales, 0.90 | 27 | 0.024 | 0.138 | 0.100 | 0.052 | 0.078 (0.061) | 12 de 16 |
| G: nueve canales, 0.85 | 25 | 0.019 | 0.102 | 0.042 | 0.051 | 0.054 (0.055) | 13 de 16 |

En conjunto, a menor `strength`, menor indicador, pero no en cada caso: en el frasco G queda muy por debajo de I y J, y en la taza y la caja las variantes de nueve canales están todas cerca. El pipeline usa `int(30 × strength)` pasos, así que las configuraciones también difieren en pasos, no solo en cuánto ruido parten.

Lo que se ve:

- **El estilo sí puede separar G de I y J.** En la taza "estudio" con semilla base 900, I y J dan fondo gris con focos y G una cocina con lámpara. En el frasco "estudio" con semilla 500, F elimina casi toda la tela de la foto original y G, I y J conservan sus pliegues.
- **Las variantes de nueve canales tienden a conservar la composición de la foto:** ninguna logra en el frasco el fondo gris continuo del preset de estudio, y en la caja I y J casi no cambian entre estilos. Es plausible que venga de los latentes iniciales, que con `strength` < 1 parten de la imagen completa codificada más ruido (la imagen enmascarada que recibe el UNet, en cambio, borra el fondo editable). No está demostrado. El model card recomienda `strength` < 1 y advierte menor nitidez con 1.0, que serviría como diagnóstico.
- No se puede decir que el estilo dependa más de la semilla que del `strength`: estudio usa las semillas 500 y 900 y lifestyle 501 y 901, así que al comparar estilos cambian a la vez prompt y ruido.
- El ramo del jarrón crece en todas las variantes.

**Estado:** con el indicador de expansión, G (0.85) sigue siendo el mejor, pero el cumplimiento del estilo puede invertir el orden frente a I (0.90). No se elige `strength` sin medir el estilo. Antes de proponer el cambio de P3 y ADR-004 falta:

1. Medir el cumplimiento del estilo. Revisión ciega como medida principal: A, F, G, I y J al mismo tamaño y en orden aleatorio, evaluando por separado cumplimiento del preset (distinguir "estudio con focos visibles" de "fondo continuo de catálogo", que es lo que pide el preset), contacto y sombras, y expansión. CLIP en CPU como apoyo, con el protocolo fijado antes de calcular: el mismo modelo para todas las imágenes, cada una contra cinco descripciones paralelas basadas en los presets (sin categoría de producto ni adjetivos de calidad), el margen frente a los demás estilos y no solo el más parecido, las fotos originales como control y paráfrasis para comprobar estabilidad.
2. Repetir con fotos caseras reales de los casos obligatorios y los cinco estilos, cruzando estilos y semillas, con un caso mate sencillo, sin flores.
3. Demostrar en el smoke test del LoRA (P4) entrenamiento breve, exportación, recarga e inferencia del adaptador sobre nueve canales, con memoria y latencia del lote completo.

La evidencia de las corridas está en el volumen `productshot-hf-cache` (`experiments/<RUN_ID>`) y en `data/experiments/expansion/` de quien la bajó; `data/` no se versiona.

Para el modo avanzado, si se adopta nueve canales, hay dos caminos: cargar el LoRA entrenado sobre SDXL base (el script oficial lo aplica solo en las proyecciones de atención, que tienen la misma forma; pero los pesos del UNet de nueve canales siguen entrenados aparte, así que la compatibilidad se demuestra en el smoke test, no se supone) o entrenarlo sobre el UNet de nueve canales con objetivo de inpainting (más trabajo). Ambos modos deben usar el mismo checkpoint para que la comparación aísle el LoRA.

### Smoke test del LoRA en Colab (2026-11-14)

`PYTHONPATH=backend python training/run_colab.py --stage smoke`, desde el commit `0cd76d1`, en el Colab Pro de Pedro manejado con el Google Colab CLI (`training/README.md`). L4 de 22 GB a **1.94 unidades por hora**; toda la corrida gastó unas 0.23 unidades (de 217.84). La laptop se congeló a mitad de corrida y el `finally` del driver no corrió: el VM siguió vivo, se bajó el resultado y se liberó a mano (`colab_assignments.py list` vacío, consumo 0). Es el caso que cubre `--cleanup`.

- **Entorno:** torch 2.14.1+cu130 de Colab intacto; diffusers 0.31.0, transformers 4.46.3, peft 0.13.2, accelerate 1.1.1. `pip check` falla por paquetes de Colab que no se usan (gradio y gcsfs piden otras versiones de huggingface_hub y fsspec).
- **Entrenamiento:** 20 imágenes generadas (4 productos × 5 estilos), 100 pasos a 1024 px, rank 16, bf16: 200 s, 11.0 GB de VRAM pico, adaptador de 44.5 MB con 560 capas LoRA en el UNet. Pesos finitos, no nulos y con el sha256 registrado. Sin pérdida de validación, 100 pasos no dicen si aprendió algo útil (la pérdida por paso pasó de 0.130 a 0.160 en promedio, sin significado con lote 1).

| Verificación (misma semilla, LoRA en 0 y en 1) | Cuatro canales, 0.99 | Nueve canales, 0.85 |
|---|---|---|
| Capas cargadas | 560 de 560 | 560 de 560 |
| Tensores (A y B) frente al archivo | 1120, sin faltantes, error relativo máx. 2.4e-5 | 1120, sin faltantes, error relativo máx. 2.4e-5 |
| Cambio medio del fondo al encender el LoRA | 8.7 de 255 | **0.15 de 255** (0.6 % de los píxeles cambia más de 1/255) |

**Lo que queda resuelto:** un LoRA entrenado sobre SDXL base carga completo en el UNet de nueve canales, sin claves faltantes y con los pesos dentro de tolerancia. La parte mecánica de P4 funciona.

**Lo que no:** en nueve canales el LoRA casi no cambia la imagen, y la prueba no permite saber por qué. La imagen de verificación es una taza sintética sobre fondo liso, y con `strength` 0.85 ese modelo tiende a conservar la composición original; también es posible que el LoRA se transfiera débilmente, porque los pesos del UNet de nueve canales son distintos. Que `set_adapters` con peso 0 no apague el LoRA, o que el UNet de nueve canales no use esas capas, no tiene respaldo en el código de diffusers 0.31 y peft 0.13.2. El umbral de 0.5/255 del smoke test es heurístico. La verificación sigue en Modal, donde se servirá (ADR-009): con controles de repetibilidad (LoRA apagado dos veces), fotos con fondo texturizado y `strength` 0.85, 0.99 y 1.0.

### Verificación del LoRA en Modal (2026-11-14, corrida `lora_check`)

`PYTHONPATH=backend modal run --detach deploy/modal/worker.py::lora_check --adapter data/lora/20261114-121343-smoke/pytorch_lora_weights.safetensors`, desde el commit limpio `d71cdf1`. El mismo adaptador del smoke test (sha256 `a16c35b6…`) cargado con `SdxlInpaintBackend.attach_lora` en la clase `Lab`, con las cuatro fotos de los experimentos (fondos con textura), estudio y lifestyle, semilla base 500: 12 configuraciones, 96 imágenes. Carga comprobada en ambos UNet: 560 capas y 1120 tensores, sin faltantes, error relativo máximo 2.4e-5 (torch 2.5.1, diffusers 0.31.0, peft 0.13.2).

| Diferencia media del fondo (0 a 255), n = 8 | A: cuatro canales, 0.99 | G: nueve canales, 0.85 | F: nueve canales, 0.99 | K: nueve canales, 1.0 |
|---|---|---|---|---|
| Control: cargar el LoRA con peso 0 frente a sin LoRA | 0.94 (máx. 2.23) | 0.17 (máx. 0.23) | | |
| Control: repetir peso 0 | 0.00 | 0.00 | | |
| **Efecto del LoRA (peso 0 frente a 1)** | **13.3** (10.1 a 16.9) | **2.6** (1.0 a 3.9) | **6.2** (1.1 a 19.1) | **9.2** (3.8 a 16.1) |
| Cambio frente al fondo de la foto original | 46 | 19 | 29 | 47 |

- **El LoRA entrenado sobre SDXL base cambia la imagen en el UNet de nueve canales** con estas fotos. En A y G el efecto medio supera más de tres veces el mayor valor de sus controles; F y K no tienen controles propios (los de A son de otro checkpoint), aunque su efecto es del mismo orden.
- El efecto medio en nueve canales crece con `strength`, igual que el cambio frente a la foto, pero no en cada imagen (G < F < K en 5 de los 8 pares). Con `strength` 1.0 la misma semilla no garantiza el mismo ruido inicial que con 0.99 (diffusers omite una codificación previa), así que F y K no son pares exactos entre sí; sí lo son el LoRA en 0 y en 1 dentro de cada configuración.
- El resultado de Colab (0.15/255) queda compatible con una dependencia de la entrada (fondo liso) y del anclaje, pero su causa no está aislada: entre ambas corridas cambiaron imagen, máscara, semilla y versión de torch.
- Cargar el adaptador con peso 0 no es exactamente neutro (0.94/255 en A), aunque es pequeño frente al efecto. Por eso el modo intermedio debe correr en el mismo pipeline con el LoRA en peso 0, como hace el backend, y no en un pipeline sin adaptador: así la comparación de `docs/06` solo cambia la escala del LoRA.
- A la vista, el LoRA cambia detalles (accesorios del fondo, una ventana, la forma de un objeto) sin un cambio de estilo claro: es esperable con 100 pasos sobre 20 imágenes generadas. Si mejora la estética se mide con el dataset real y la evaluación de `docs/06`; el tamaño del cambio en píxeles no dice si es mejor.

La app estuvo viva 41 minutos (12:50 a 13:31): la subida del adaptador con `modal volume put` tardó unos 6 minutos antes de pedir la GPU, y la generación sumó unos 16. El costo de GPU no está verificado; con unos 30 minutos de contenedor serían cerca de USD 0.40, por encima de los USD 0.30 estimados. Al terminar no quedó ningún contenedor.

**Estado de P4:** compatibilidad de carga e inferencia demostrada en ambos UNet, con efecto observable. Faltan: memoria y latencia del lote completo de cinco estilos con el LoRA, que el worker HTTP cargue el adaptador (`LORA_ADAPTER_URI`, hoy arranca sin él) y la evaluación estética. Elegir `strength` sigue dependiendo del estilo y la integración, no del tamaño del efecto.

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
- Máximo de un worker con GPU a la vez y endpoints protegidos, para que nadie genere en masa. En Modal, `Worker` (API) y `Lab` (experimentos) son clases distintas con un contenedor como máximo cada una: no lanzar experimentos durante una demo, un ensayo o la prueba con usuarios, para no tener dos GPU encendidas a la vez.
- Cachear pesos en un volumen o en la imagen; no descargarlos en cada solicitud.

## Secretos e identidad

- Nunca versionar claves de Gemini, tokens de Hugging Face, JSON de cuentas de servicio ni refresh tokens.
- En la laptop: `.env` no versionado. En Modal: secretos de Modal. En GCP: identidades de servicio, ADC y Secret Manager.
- El plan gratuito de Gemini permite a Google usar lo enviado para mejorar sus productos: al agente solo van decisiones y motivos, nunca fotos, y se declara en el consentimiento de la prueba con usuarios (`06_EVALUACION_Y_ETICA.md`).
- Repositorios de pesos y datos privados; sin acceso público a archivos de sesiones de otros usuarios; borrado con `DELETE /sessions/{id}`.

## Punto de decisión

Si el lote de cinco no entra en memoria o tarda demasiado en una L4, probar menos pasos, menor resolución o una GPU mayor en Modal antes de cambiar de checkpoint. **La infraestructura no determina por adelantado el checkpoint.**
