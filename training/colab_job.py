"""Trabajo de entrenamiento LoRA que corre DENTRO del VM de Colab (ADR-004, P4).

No se ejecuta en la laptop: lo lanza `training/run_colab.py` con `colab exec -f`, después
de subir la configuración a `/content/job.json`. Es autocontenido (no importa el paquete
`productshot`) porque en el VM solo existe este archivo.

Etapas:
0. Instalación: `requirements-colab.txt` (subido por el driver) con torch y torchvision
   restringidos a los que trae el VM; después `pip check` (se registra) y una prueba de imports
   en un intérprete nuevo, que sí hace fallar el trabajo.
1. Entorno: GPU, versiones y el script oficial de diffusers, verificado por sha256.
2. Dataset: `generate` crea uno mínimo con SDXL base (un producto por estilo, captions con el
   preset del estilo); `upload` usa una carpeta imagefolder subida como tar.
3. Entrenamiento: `train_text_to_image_lora_sdxl.py` de diffusers 0.31.0 sobre SDXL base,
   con checkpoints para reanudar (`--resume_from_checkpoint latest`).
4. Verificación (smoke test de P4): carga el adaptador en el pipeline de inpainting de cuatro
   canales y en el de nueve, genera con el LoRA apagado y encendido con la misma semilla y
   mide cuánto cambia el fondo. Falla si algún pipeline no tiene los canales esperados, si no
   carga todas las capas del archivo, si los pesos cargados no coinciden con los del archivo,
   si hay avisos de claves inesperadas o faltantes, o si el LoRA no cambia la imagen.
5. Resultado: `/content/out/result.tar.gz` con el adaptador, el manifiesto, la pérdida, los
   eventos de tensorboard, el último checkpoint y las imágenes de verificación. El manifiesto
   se reescribe en `/content/out/manifest.json` después de cada etapa.

El trabajo respeta `deadline_unix` de la configuración: si se pasa, mata el entrenamiento y
termina con error, así el driver alcanza a bajar lo hecho y detener el VM.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

CONTENT = Path("/content")
JOB = CONTENT / "job.json"
REQUIREMENTS = CONTENT / "requirements-colab.txt"
OUT = CONTENT / "out"
DATA = CONTENT / "dataset"
TRAIN_OUT = CONTENT / "lora_train"

SCRIPT_URL = (
    "https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0/"
    "examples/text_to_image/train_text_to_image_lora_sdxl.py"
)
SCRIPT_SHA256 = "5860e4a50f5d7a75b3f5c73d1df153ea604069e2f90bdd0994308749541fe959"


def log(msg: str) -> None:
    print(f"[job {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GpuPeak:
    """Muestrea la memoria usada de la GPU con nvidia-smi (el entrenamiento es otro proceso)."""

    def __init__(self) -> None:
        self.peak_mb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip().splitlines()
                if out:
                    self.peak_mb = max(self.peak_mb, int(out[0]))
            except Exception:  # noqa: BLE001 - la medición es auxiliar
                pass
            self._stop.wait(3)

    def __enter__(self) -> "GpuPeak":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


DEADLINE = float("inf")


def check_deadline(stage: str) -> None:
    if time.time() > DEADLINE:
        raise TimeoutError(f"Se acabó el tiempo del trabajo antes de: {stage}")


def run_streaming(cmd: list[str], env: dict | None = None) -> None:
    """Ejecuta un comando mostrando su salida en vivo (así `colab exec` ve actividad).

    Si se pasa el plazo del trabajo, mata el GRUPO de procesos: `accelerate launch` crea otro
    proceso para entrenar, y matar solo al lanzador lo dejaría vivo (y la tubería abierta).
    """
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    assert proc.stdout is not None

    def kill_group() -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    killer = threading.Timer(max(1.0, DEADLINE - time.time()), kill_group) if DEADLINE != float("inf") else None
    if killer:
        killer.daemon = True
        killer.start()
    try:
        for chunk in iter(lambda: proc.stdout.read(512), b""):
            sys.stdout.write(chunk.decode("utf-8", "replace"))
            sys.stdout.flush()
        code = proc.wait()
    finally:
        if killer:
            killer.cancel()
        if proc.poll() is None:  # salida inesperada del bucle: no dejar procesos huérfanos
            kill_group()
            proc.wait()
    if time.time() > DEADLINE:
        raise TimeoutError(f"Se acabó el tiempo del trabajo durante: {cmd[0]}")
    if code != 0:
        raise RuntimeError(f"El comando terminó con código {code}: {' '.join(cmd[:4])}")


def install() -> dict:
    """Instala las dependencias sin tocar el torch del VM y comprueba que todo importa."""
    from importlib.metadata import version

    before = {p: version(p) for p in ("torch", "torchvision")}
    constraints = CONTENT / "constraints-colab.txt"
    constraints.write_text("".join(f"{p}=={v}\n" for p, v in before.items()), encoding="utf-8")
    base = ["-r", str(REQUIREMENTS), "-c", str(constraints)]
    try:
        run_streaming(["uv", "pip", "install", "--system", *base])
    except (FileNotFoundError, RuntimeError):
        run_streaming([sys.executable, "-m", "pip", "install", "-q", *base])
    pip_check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    probe = subprocess.run(
        [sys.executable, "-c",
         "import json, importlib.metadata as m, torch, torchvision, diffusers, transformers, peft, accelerate, "
         "datasets; print(json.dumps({p: m.version(p) for p in ('torch', 'torchvision', 'diffusers', "
         "'transformers', 'peft', 'accelerate', 'datasets', 'huggingface_hub')}))"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"Las dependencias no importan: {probe.stderr[-2000:]}")
    after = json.loads(probe.stdout.strip().splitlines()[-1])
    for p, v in before.items():
        if after[p] != v:
            raise RuntimeError(f"La instalación cambió {p}: {v} -> {after[p]}")
    log(f"Dependencias: {after}")
    return {"before": before, "after": after, "pip_check_ok": pip_check.returncode == 0,
            "pip_check": (pip_check.stdout + pip_check.stderr)[-3000:]}


def environment(cfg: dict) -> dict:
    import torch

    info = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
        "python": sys.version.split()[0],
        "packages": {},
    }
    from importlib.metadata import version

    for pkg in ("torch", "diffusers", "transformers", "accelerate", "peft", "datasets", "huggingface_hub"):
        info["packages"][pkg] = version(pkg)
    if info["packages"]["diffusers"] != cfg["diffusers_version"]:
        raise RuntimeError(f"diffusers {info['packages']['diffusers']} != {cfg['diffusers_version']}")
    script = CONTENT / "train_text_to_image_lora_sdxl.py"
    if not script.exists():
        urllib.request.urlretrieve(SCRIPT_URL, script)
    if sha256(script) != SCRIPT_SHA256:
        raise RuntimeError("El script oficial de diffusers no coincide con el sha256 esperado.")
    info["train_script"] = {"url": SCRIPT_URL, "sha256": SCRIPT_SHA256}
    log(f"Entorno: {info['gpu']} ({info['vram_total_gb']} GB), {info['packages']}")
    return info


def build_dataset(cfg: dict) -> dict:
    """Carpeta imagefolder con `metadata.jsonl` (columnas `file_name` y `text`)."""
    ds = cfg["dataset"]
    if ds["kind"] == "upload":
        tar_path = CONTENT / ds["remote_tar"]
        DATA.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(DATA, filter="data")
        n = sum(1 for _ in (DATA / "metadata.jsonl").open())
        return {"kind": "upload", "images": n, "tar_sha256": sha256(tar_path)}

    # kind == "generate": dataset mínimo para comprobar la mecánica, no la calidad.
    import torch
    from diffusers import StableDiffusionXLPipeline

    meta = DATA / "metadata.jsonl"
    if meta.exists():
        return {"kind": "generate", "images": sum(1 for _ in meta.open()), "reused": True}
    DATA.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg["base_model"], revision=cfg["base_revision"], torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True,
    ).to("cuda")
    rows = []
    for i, product in enumerate(ds["products"]):
        for j, (style, prompt) in enumerate(cfg["style_prompts"].items()):
            caption = f"{prompt}, {product}"
            image = pipe(
                prompt=caption, negative_prompt=cfg["negative_prompt"], width=1024, height=1024,
                num_inference_steps=25, generator=torch.Generator("cuda").manual_seed(1000 + 10 * i + j),
            ).images[0]
            name = f"{i:02d}_{style}.png"
            image.save(DATA / name)
            rows.append({"file_name": name, "text": caption, "style": style})
            log(f"Imagen {len(rows)}: {name}")
    meta.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    del pipe
    torch.cuda.empty_cache()
    return {"kind": "generate", "images": len(rows), "seed_rule": "1000 + 10 * producto + estilo"}


def train(cfg: dict) -> dict:
    t = cfg["train"]
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch", "--num_processes", "1",
        "--mixed_precision", t["mixed_precision"],
        str(CONTENT / "train_text_to_image_lora_sdxl.py"),
        "--pretrained_model_name_or_path", cfg["base_model"],
        "--revision", cfg["base_revision"],
        "--variant", "fp16",  # los mismos archivos que la generación: una sola descarga
        "--train_data_dir", str(DATA),
        "--caption_column", "text",
        "--resolution", str(t["resolution"]),
        "--random_flip",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", str(t["grad_accum"]),
        "--gradient_checkpointing",
        "--max_train_steps", str(t["steps"]),
        "--checkpointing_steps", str(t["checkpointing_steps"]),
        "--checkpoints_total_limit", "2",
        "--resume_from_checkpoint", "latest",
        "--learning_rate", str(t["learning_rate"]),
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--rank", str(t["rank"]),
        "--mixed_precision", t["mixed_precision"],
        "--seed", str(t["seed"]),
        "--dataloader_num_workers", "2",
        "--output_dir", str(TRAIN_OUT),
        "--report_to", "tensorboard",
    ]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t0 = time.perf_counter()
    with GpuPeak() as peak:
        run_streaming(cmd, env=env)
    weights = TRAIN_OUT / "pytorch_lora_weights.safetensors"
    if not weights.exists():
        raise RuntimeError("El entrenamiento no produjo pytorch_lora_weights.safetensors.")
    return {
        "args": t, "seconds": round(time.perf_counter() - t0, 1), "peak_vram_mb": peak.peak_mb,
        "adapter_sha256": sha256(weights), "adapter_mb": round(weights.stat().st_size / 1024**2, 1),
        "loss": read_losses(),
    }


def read_losses() -> dict:
    """Pérdida por paso desde los eventos de tensorboard que escribe el script oficial."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {"error": "tensorboard no disponible"}
    values = []
    for events in sorted((TRAIN_OUT / "logs").rglob("events.out.tfevents.*")):
        acc = EventAccumulator(str(events))
        acc.Reload()
        if "train_loss" in acc.Tags().get("scalars", []):
            values += [(e.step, round(e.value, 5)) for e in acc.Scalars("train_loss")]
    if not values:
        return {"error": "sin eventos de train_loss"}
    first = [v for _, v in values[:10]]
    last = [v for _, v in values[-10:]]
    return {"steps": len(values), "first10_mean": round(sum(first) / len(first), 5),
            "last10_mean": round(sum(last) / len(last), 5), "series": values}


def demo_product(size=(1024, 768)):
    """Producto sintético (una taza) con máscara exacta: blanco = editable, negro = producto."""
    from PIL import Image, ImageDraw

    w, h = size
    img = Image.new("RGB", size, (214, 210, 202))
    mask = Image.new("L", size, 255)
    for target, body, handle in ((img, (176, 38, 44), (150, 30, 36)), (mask, 0, 0)):
        d = ImageDraw.Draw(target)
        d.rounded_rectangle([w * 0.38, h * 0.30, w * 0.60, h * 0.78], radius=18, fill=body)
        d.ellipse([w * 0.56, h * 0.40, w * 0.70, h * 0.62], outline=handle, width=int(w * 0.022))
    return img, mask


def adapter_file_tensors(weights: Path) -> dict:
    """Tensores LoRA del UNet en el archivo, por nombre de módulo PEFT esperado.

    diffusers 0.31 guarda `unet.<módulo>.lora.down.weight` (A) y `.lora.up.weight` (B); al
    cargar con el nombre `estilo` quedan como `<módulo>.lora_A.estilo` y `.lora_B.estilo`.
    """
    from safetensors.torch import load_file

    state = load_file(str(weights))
    tensors = {}
    for key, value in state.items():
        if not key.startswith("unet."):
            continue
        name = key[len("unet."):]
        for old, new in ((".lora.down.weight", ".lora_A.estilo"), (".lora_A.weight", ".lora_A.estilo"),
                         (".lora.up.weight", ".lora_B.estilo"), (".lora_B.weight", ".lora_B.estilo")):
            if name.endswith(old):
                tensors[name[: -len(old)] + new] = value.float()
                break
    return tensors


def compare_loaded(unet, file_tensors: dict) -> dict:
    """Compara cada tensor A y B cargado con el del archivo (forma y valores)."""
    modules = dict(unet.named_modules())
    missing, shape_mismatch, worst = [], [], 0.0
    for name, expected in file_tensors.items():
        module = modules.get(name)
        if module is None:
            missing.append(name)
            continue
        got = module.weight.detach().float().cpu()
        if tuple(got.shape) != tuple(expected.shape):
            shape_mismatch.append(name)
            continue
        scale = float(expected.abs().max()) or 1.0
        worst = max(worst, float((got - expected).abs().max()) / scale)
    return {"tensors": len(file_tensors), "missing": missing[:20], "missing_count": len(missing),
            "shape_mismatch": shape_mismatch[:20], "max_rel_err": worst}


def verify(cfg: dict) -> dict:
    """Carga el adaptador en cada pipeline de inpainting, comprueba la carga y mide su efecto."""
    import numpy as np
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline

    warnings: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                warnings.append(record.getMessage()[:500])

    handler = Capture()
    for name in ("diffusers", "peft"):
        logging.getLogger(name).addHandler(handler)

    weights = TRAIN_OUT / "pytorch_lora_weights.safetensors"
    file_tensors = adapter_file_tensors(weights)
    file_layers = sum(1 for k in file_tensors if k.endswith("lora_A.estilo"))
    file_b_nonzero = any(float(v.abs().max()) > 0 for k, v in file_tensors.items() if k.endswith("lora_B.estilo"))
    min_diff = cfg["verify"]["min_background_diff"]
    image, mask = demo_product()
    background = np.array(mask) > 127
    prompt = cfg["style_prompts"][cfg["verify"]["style"]]
    results: dict = {"adapter_file": {"unet_layers": file_layers, "unet_tensors": len(file_tensors),
                                      "lora_b_nonzero": file_b_nonzero}, "pipelines": {}}
    failures: list[str] = []
    try:
        for name, spec in cfg["verify"]["pipelines"].items():
            check_deadline(f"verificación {name}")
            log(f"Verificación: {name} ({spec['model']})")
            start = len(warnings)
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                spec["model"], revision=spec["revision"], torch_dtype=torch.float16,
                variant="fp16", use_safetensors=True,
            ).to("cuda")
            in_channels = pipe.unet.config.in_channels
            pipe.load_lora_weights(str(TRAIN_OUT), weight_name=weights.name, adapter_name="estilo")
            lora_a = [n for n, _ in pipe.unet.named_modules() if n.endswith("lora_A.estilo")]
            loaded = compare_loaded(pipe.unet, file_tensors)
            images = {}
            for scale in (0.0, 1.0):
                pipe.set_adapters(["estilo"], adapter_weights=[scale])
                out = pipe(
                    prompt=prompt, negative_prompt=cfg["negative_prompt"], image=image, mask_image=mask,
                    width=1024, height=768, strength=spec["strength"], num_inference_steps=30,
                    guidance_scale=7.0, generator=torch.Generator("cuda").manual_seed(cfg["verify"]["seed"]),
                ).images[0]
                images[scale] = np.asarray(out, dtype=np.float32)
                out.save(OUT / f"verify_{name}_lora{scale:.1f}.png")
            diff = float(np.abs(images[1.0] - images[0.0])[background].mean())
            own_warnings = warnings[start:]
            checks = {
                "canales_esperados": in_channels == spec["in_channels"],
                "pesos_entrenados_no_nulos": file_b_nonzero,
                "todas_las_capas_cargadas": len(lora_a) == file_layers > 0,
                # fp16 en el pipeline frente a fp32 en el archivo: tolerancia relativa de 1 %.
                "pesos_cargados_iguales_al_archivo": (
                    loaded["missing_count"] == 0 and not loaded["shape_mismatch"] and loaded["max_rel_err"] < 0.01
                ),
                "sin_claves_inesperadas_o_faltantes": not any(
                    w for w in own_warnings if "unexpected" in w.lower() or "missing" in w.lower()
                ),
                "el_lora_cambia_la_imagen": diff >= min_diff,
            }
            results["pipelines"][name] = {
                "model": f"{spec['model']}@{spec['revision']}", "unet_in_channels": in_channels,
                "strength": spec["strength"], "lora_layers_in_unet": len(lora_a),
                "loaded_vs_file": loaded,
                "background_mean_abs_diff_lora_on_vs_off": round(diff, 3),
                "warnings": own_warnings, "checks": checks,
            }
            failures += [f"{name}: {k}" for k, ok in checks.items() if not ok]
            log(f"  {in_channels} canales, {len(lora_a)}/{file_layers} capas, error máx. "
                f"{loaded['max_rel_err']:.2e}, fondo {diff:.2f}/255, fallas: {[k for k, ok in checks.items() if not ok]}")
            del pipe
            torch.cuda.empty_cache()
    finally:
        for name in ("diffusers", "peft"):
            logging.getLogger(name).removeHandler(handler)
    layers = {r["lora_layers_in_unet"] for r in results["pipelines"].values()}
    if len(layers) > 1:
        failures.append(f"distinto número de capas entre pipelines: {sorted(layers)}")
    results["passed"] = not failures
    results["failures"] = failures
    return results


def package(manifest_path: Path) -> None:
    """Empaqueta lo que haya, también si el trabajo falló a mitad."""
    with tarfile.open(OUT / "result.tar.gz", "w:gz") as tar:
        tar.add(manifest_path, arcname="manifest.json")
        for png in sorted(OUT.glob("verify_*.png")):
            tar.add(png, arcname=png.name)
        meta = DATA / "metadata.jsonl"
        if meta.exists():
            tar.add(meta, arcname="dataset_metadata.jsonl")
        weights = TRAIN_OUT / "pytorch_lora_weights.safetensors"
        if weights.exists():
            tar.add(weights, arcname=weights.name)
        if (TRAIN_OUT / "logs").exists():
            tar.add(TRAIN_OUT / "logs", arcname="logs")
        checkpoints = sorted(TRAIN_OUT.glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[1]))
        if checkpoints:
            last = checkpoints[-1] / "pytorch_lora_weights.safetensors"
            if last.exists():
                tar.add(last, arcname=f"{checkpoints[-1].name}_pytorch_lora_weights.safetensors")


def main() -> None:
    global DEADLINE
    cfg = json.loads(JOB.read_text(encoding="utf-8"))
    DEADLINE = float(cfg.get("deadline_unix") or float("inf"))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"run_id": cfg["run_id"], "stage": cfg["stage"], "status": "running",
                      "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": cfg}
    manifest_path = OUT / "manifest.json"

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    try:
        save()
        manifest["install"] = install()
        save()
        manifest["environment"] = environment(cfg)
        save()
        check_deadline("dataset")
        manifest["dataset"] = build_dataset(cfg)
        save()
        check_deadline("entrenamiento")
        manifest["train"] = train(cfg)
        save()
        if cfg["verify"]["enabled"]:
            manifest["verify"] = verify(cfg)
            if not manifest["verify"]["passed"]:
                raise RuntimeError(f"El smoke test falló: {manifest['verify']['failures']}")
        manifest["status"] = "ok"
    except BaseException as exc:
        manifest["status"] = f"error: {exc!r}"
        raise
    finally:
        manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()
        package(manifest_path)
        log(f"Resultado en {OUT / 'result.tar.gz'} (estado: {manifest['status']})")


if __name__ == "__main__":
    main()
