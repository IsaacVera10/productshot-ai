# 09: Plan de Aaron para el despliegue en Modal

> **Actualización 2026-11-16:** acordado con Isaac. Aaron toma el Hito 2 completo (modo intermedio como bloque independiente, ADR-010). Las tareas 1 a 4 se descartan: la demo corre con la API y la UI en la laptop y cada modo en su propio worker de Modal; el 303 de Modal se resolvió en los adaptadores remotos. La tarea 5 (LoRA en el worker) pasa a Isaac, junto con su worker del modo avanzado.

**Fecha:** 2026-11-14 · **Responsable:** Aaron Camacho · **Revisa:** Isaac

Aaron toma el Hito 6 (Modal) mientras Isaac sigue con el dataset BG60k y el LoRA (Hitos 4 y 5). Las dos partes solo se cruzan en el contrato HTTP del worker, así que pueden avanzar en paralelo.

## Contexto

El worker de generación ya corre en Modal. Falta servir allí la app completa y cerrar los huecos del Hito 6 (`04_ROADMAP_Y_TESTS.md`).

- **Ya funciona:** `deploy/modal/worker.py` sirve SDXL y `rembg` en una L4 detrás del contrato HTTP de `productshot.worker`. Un lote de cinco tarda unos 50 s y usa 9.5 GB de VRAM (`05_COLAB_Y_GCP.md`).
- **Ya funciona en la laptop:** la API y la UI hablan con ese worker con `GENERATION_BACKEND=remote` y `SEGMENTATION_BACKEND=remote`.
- **Falta:**
    - la API y la UI como endpoint web en Modal;
    - su ejecutor de trabajos (P10);
    - el manejo de la redirección 303 de Modal;
    - la cancelación al vencer G14;
    - que el worker cargue el LoRA desde `LORA_ADAPTER_URI`.

Nada de esto necesita BG60k ni el LoRA final: se prueba con el mock y con SDXL base.

## Antes de empezar

El primer día se dedica a preparar una cuenta de Modal propia y repetir la prueba de humo. No hace falta escribir código.

- [ ] Clonar el repo y correr `uv sync` y `uv run pytest`. Todo pasa sin red ni GPU: 204 pruebas al 2026-11-14.
- [ ] Crear la cuenta en modal.com y correr `modal setup`. Si pide tarjeta, fijar el **límite de gasto en USD 30**.
- [ ] Crear el secreto con `modal secret create productshot-worker GENERATION_WORKER_TOKEN=<token>`. El mismo token va en el `.env` de la API.
- [ ] Correr la prueba de humo: `PYTHONPATH=backend modal run deploy/modal/worker.py::main`. Deja los PNG y `metrics.json` en `data/smoke/`. La primera vez baja unos 7 GB de pesos al volumen `productshot-hf-cache`.
- [ ] Comprobar con `modal container list` que no quedó ningún contenedor vivo.

Las corridas largas van siempre con `modal run --detach` y guardan en el volumen. Así, si se corta la laptop, la GPU no queda cobrando sin dejar resultado.

## Tareas

Son cinco tareas en orden de prioridad. Cada una cabe en un PR propio, y la 1 desbloquea a las demás.

| # | Tarea | Archivos principales | Hecho cuando | Depende de |
|---|---|---|---|---|
| 1 | API y UI como endpoint web en Modal, sin GPU. SQLite y artefactos en un volumen, `max_containers=1`. Apunta al worker ya desplegado | `deploy/modal/api.py` (nuevo), `composition.py`, `config.py` | La demo completa (subir, generar, decidir y resumir) funciona desde la URL de Modal | Antes de empezar |
| 2 | Ejecutor de trabajos de Modal (P10). La generación se lanza con `spawn` y no en un hilo del proceso, así que un reinicio del contenedor de la API no deja lotes en `FAILED` | `services/jobs.py`, `composition.py` (`JOB_RUNNER=modal`) | Pruebas sin red con un runner falso y una corrida real en Modal | 1 |
| 3 | Redirección 303 de Modal. Pasados 150 s, el endpoint responde con un 303 que `RemoteGenerationBackend` no sigue | `services/remote_generation.py` | Prueba unitaria con un transporte que devuelve 303 y prueba real con un worker lento (el mock con una espera) | 1 |
| 4 | Cancelar el lote en el worker cuando vence G14. Hoy la API abandona a los 540 s, pero la GPU sigue generando hasta 10 minutos | `remote_generation.py`, `deploy/modal/worker.py` | Al vencer el plazo, el contenedor del worker deja de trabajar (se ve en `modal container list`) | 2 |
| 5 | El worker HTTP carga el LoRA desde `LORA_ADAPTER_URI` (repo privado de Hugging Face, con el token en un secreto de Modal). `advanced` usa `lora_scale` e `intermediate` usa peso 0 en el mismo pipeline | `deploy/modal/worker.py` (clase `Worker`); `sdxl_generation.py` ya tiene `attach_lora` | `/healthz` informa el sha256 del adaptador, y la memoria y la latencia del lote de cinco con LoRA quedan anotadas en `05_COLAB_Y_GCP.md` | Que Isaac publique el adaptador |

Notas:

1. **Tarea 1:** la base de datos solo puede tener un contenedor escribiendo, porque es SQLite en un volumen. Hacer `commit` del volumen después de cada escritura, o documentar por qué no hace falta.
2. **Tarea 3:** hay dos caminos. Uno es seguir la redirección con `follow_redirects` sin perder el token. El otro es pasar a `spawn` y consultar el resultado. Se elige con la prueba real, sin pasarse del presupuesto de G14.
3. **Tarea 5:** para desarrollarla basta el adaptador del smoke test (`data/lora/20261114-121343-smoke/`), que Isaac comparte. No hay que esperar al LoRA final.

Más adelante, con el LoRA final, queda ensayar la demo en Modal en menos de 6 minutos con el worker caliente y grabar el video de respaldo.

## Qué no tocar

Isaac sigue con el dataset, el LoRA y la elección del checkpoint. Estas partes quedan fuera para no pisarse:

- La clase `Lab` y los entrypoints `experiment`, `lora_check` y `collect` de `deploy/modal/worker.py`. En ese archivo solo se cambia la clase `Worker`.
- `training/` y `data/raw/bg60k/`.
- Los parámetros de `SdxlParams` (checkpoint y `strength`), porque P3 y ADR-004 siguen abiertos. Avisar antes de tocar `sdxl_generation.py`.

## Forma de trabajo

Cada tarea es un PR chico que revisa Isaac, con la evidencia de la corrida en la descripción.

- Rama `feat/<tarea>` desde `main`. Antes de abrir el PR, `git pull --rebase`.
- Commits en español con el formato `<tipo>: <descripción en minúsculas>`, por ejemplo `feat: endpoint web de la api en modal`.
- Cada PR deja `uv run pytest` en verde. Lo que necesita Modal lleva una prueba simulada sin red, como las de `tests/test_ports.py`.
- En cada corrida de Modal se anota cuánto duró y su costo aproximado, y se termina con `modal container list` vacío.
- El estado se actualiza en `05_COLAB_Y_GCP.md` (resultados) y en las casillas del Hito 6 de `04_ROADMAP_Y_TESTS.md`.
- Nunca lanzar experimentos durante una demo, un ensayo o la prueba con usuarios, porque quedarían dos GPU encendidas a la vez.

## Decisiones abiertas

- [ ] ¿Aaron usa una cuenta de Modal propia o entra como miembro del workspace de Isaac? Recomendado: cuenta propia para desarrollar (con sus propios USD 30 al mes, volumen y secreto) y un solo workspace para la demo final.
- [ ] ¿Quién crea el repo privado de Hugging Face del adaptador y cómo se comparte el token con el worker?
- [ ] ¿Aaron tiene GPU local para probar el worker en la laptop con los mismos contratos? Es opcional: la RTX 4060 de 8 GB necesita `enable_model_cpu_offload`.
