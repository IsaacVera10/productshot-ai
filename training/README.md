# Entrenamiento del LoRA en Colab Pro

El LoRA del modo `advanced` se entrena sobre SDXL base (ADR-004) en un VM de Colab Pro que se maneja desde la laptop con el [Google Colab CLI](https://github.com/googlecolab/google-colab-cli). Nadie tiene que abrir un notebook: un solo comando alquila la GPU, entrena, verifica, baja el resultado y libera el VM.

| Archivo | Dónde corre | Qué hace |
|---|---|---|
| `run_colab.py` | Laptop | Driver: comprueba que no haya VM asignados → `colab new` → `upload` → `exec` → `download` → `stop` y conciliación con el servidor (siempre, en un `finally`) |
| `colab_job.py` | VM de Colab | Instalación (sin tocar el torch del VM), dataset, entrenamiento con el script oficial de diffusers 0.31.0 (verificado por sha256), verificación y empaquetado |
| `colab_assignments.py` | Laptop, con el Python del CLI | Lista o libera VM directamente en el servidor (respaldo para VM sin nombre local) |
| `requirements-colab.txt` | VM de Colab | Mismas versiones que el worker de Modal; torch es el de Colab |

## Requisitos (una sola vez)

1. Linux o macOS (el CLI no soporta Windows; en Windows, usar WSL).
2. Instalar el CLI: `uv tool install google-colab-cli`.
3. Autenticarlo con la cuenta que tiene las unidades de Colab Pro: `colab --auth=oauth2 whoami`. Abre un enlace de Google; quien sea dueño de la cuenta lo aprueba y pega el código en la terminal. **Abrir el enlace que imprime la terminal**, no el que queda en la barra del navegador después de la redirección (ese está atado a otra sesión y falla). La autorización incluye `cloud-platform`: revocarla en myaccount.google.com/permissions al terminar el proyecto.
4. Comprobar el saldo: `colab --auth=oauth2 usage`.

## Uso

Desde la raíz del repo:

```bash
PYTHONPATH=backend python training/run_colab.py --stage smoke --dry-run   # solo muestra los comandos
PYTHONPATH=backend python training/run_colab.py --stage smoke             # gasta unidades
```

El resultado queda en `data/lora/<RUN_ID>/` (no versionado): `pytorch_lora_weights.safetensors`, `manifest.json` (instalación, versiones, revisiones, parámetros, pérdida por paso, VRAM pico, verificación), los eventos de tensorboard, los pesos del último checkpoint e imágenes `verify_*.png`. El driver termina con código 0 solo si el manifiesto es de esta corrida, dice `status: ok`, la verificación pasó y el adaptador bajado coincide con el sha256 registrado en el VM; el CLI devuelve 0 aunque el código remoto falle, así que no basta con su código de salida.

### Etapa `smoke` (P4)

Comprueba la mecánica, no la calidad. Genera 20 imágenes con SDXL base (cuatro productos × cinco estilos, captions con el preset de cada estilo), entrena 100 pasos a 1024 px con rank 16 y después carga el adaptador en el pipeline de inpainting de cuatro canales (`strength` 0.99) y en el de nueve canales (0.85). Para cada uno genera con el LoRA apagado y encendido, con la misma semilla, y registra:

- `lora_layers_in_unet`: capas LoRA inyectadas (deben ser iguales en ambos pipelines);
- `loaded_vs_file`: cada tensor A y B cargado frente al del archivo (faltantes, forma y error relativo máximo);
- `background_mean_abs_diff_lora_on_vs_off`: cuánto cambia el fondo (0 = el LoRA no se aplica);
- `warnings`: avisos de diffusers o peft al cargar.

Falla (y el driver termina con código 1) si algún pipeline no tiene los canales esperados (4 y 9), no carga todas las capas del archivo, algún tensor cargado falta, cambia de forma o difiere más de 1 % del archivo, hay avisos de claves inesperadas o faltantes, ambos pipelines no tienen el mismo número de capas, o el fondo cambia menos de 0.5/255 al encender el LoRA. Que cambie no dice que mejore: eso se mide con el dataset real y la evaluación de `docs/06`. Este smoke test cubre Colab; falta cargar el adaptador en el worker de Modal (ADR-009), que es donde se sirve.

### Etapa `train`

```bash
PYTHONPATH=backend python training/run_colab.py --stage train --dataset-tar dataset.tar --steps 1500
```

`dataset.tar` es una carpeta imagefolder: las imágenes y un `metadata.jsonl` con `{"file_name": ..., "text": ...}` por línea, donde `text` es el preset del estilo más la descripción del producto. Guarda checkpoints cada cuarto del entrenamiento y el resultado incluye los pesos del último. Cada corrida pide un VM nuevo, así que no reanuda un entrenamiento cortado: para eso habría que subir el checkpoint (pendiente). Sin probar todavía: el dataset real está pendiente (BG60k solo se descarga con una cuenta de JD Cloud).

## Cuidado con las unidades

- No arranca si la cuenta ya tiene un VM asignado (evita duplicar gasto o confundir sesiones).
- El driver muestra el saldo antes y después, y la tasa de consumo de la GPU apenas la asigna.
- `--max-minutes` (50 por defecto) es el presupuesto desde que se pide el VM: cada llamada al CLI tiene timeout, el trabajo remoto recibe su propio plazo (al vencer mata el entrenamiento con todos sus procesos) y se reservan 10 minutos para bajar el resultado y apagar. No es un máximo estricto: la descarga, el apagado y la conciliación tienen timeouts propios y pueden excederlo unos minutos.
- El `stop` se intenta aunque `colab new` falle a medias, y después se consulta al servidor: si queda un VM de esta corrida, se libera directamente. Si no se puede confirmar, termina con código 3.
- Si la laptop se apaga a mitad de corrida, el apagado no corre: `PYTHONPATH=backend python training/run_colab.py --cleanup` libera **todos** los VM de la cuenta (incluido uno abierto desde el navegador).
- `--keep` deja el VM vivo y gastando: después, `colab --auth=oauth2 --config data/lora/<RUN_ID>/colab-state.json stop -s productshot-lora`.
- Al terminar, `colab --auth=oauth2 sessions` debe decir que no hay sesiones activas.
- Colab Pro no garantiza la GPU: si `colab new --gpu L4` falla, probar `--gpu A100` (gasta bastante más) o esperar.
