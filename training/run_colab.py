"""Lanza el entrenamiento LoRA en Colab con el Google Colab CLI y siempre libera el VM.

Corre en la laptop (Linux o macOS; el CLI no soporta Windows). Necesita el CLI instalado
(`uv tool install google-colab-cli`) y autenticado con la cuenta que tiene las unidades de
Colab Pro (`colab --auth=oauth2 whoami`).

    PYTHONPATH=backend python training/run_colab.py --stage smoke --dry-run   # solo muestra el plan
    PYTHONPATH=backend python training/run_colab.py --stage smoke             # gasta unidades
    PYTHONPATH=backend python training/run_colab.py --stage train --dataset-tar ds.tar --steps 1500
    PYTHONPATH=backend python training/run_colab.py --stage embed --pack data/bg60k/pack.tar --max-minutes 30
    PYTHONPATH=backend python training/run_colab.py --stage eval --adapters data/lora/<RUN_ID> --max-minutes 120
    PYTHONPATH=backend python training/run_colab.py --stage eval --adapters data/lora/<RUN_ID> \
        --checkpoints ckpt300 --scales 0.3,0.6 --max-products 17 --max-minutes 90
    PYTHONPATH=backend python training/run_colab.py --cleanup                 # libera VM huérfanos

Pasos: comprobar que la cuenta no tiene VM asignados -> `colab new --gpu L4` -> subir la
configuración, las dependencias (y el dataset) -> `colab exec -f training/colab_job.py`
(instala, entrena y verifica dentro del VM) -> bajar el resultado -> `colab stop` y
conciliación con el servidor en un `finally`.

Garantías y límites:
- `--max-minutes` es el presupuesto TOTAL desde que se pide el VM: cada llamada al CLI tiene
  su timeout, el trabajo remoto recibe un plazo propio (`deadline_unix`) y se reservan
  minutos para bajar el resultado y apagar.
- El `stop` se intenta aunque `colab new` falle a medias. Después se consulta al servidor: si
  queda un VM que no estaba antes de empezar, se libera directamente (`colab_assignments.py`).
- Si la laptop se apaga en medio, el `finally` no corre: lanzar `--cleanup` apenas se pueda.
- Termina con código 0 solo si el resultado bajó, corresponde a esta corrida y su manifiesto
  dice `status: ok` (el CLI devuelve 0 aunque el código remoto falle).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB_SCRIPT = ROOT / "training" / "colab_job.py"
EMBED_SCRIPT = ROOT / "training" / "colab_embed.py"
EVAL_SCRIPT = ROOT / "training" / "colab_eval.py"
EVAL_SPLIT = ROOT / "data" / "bg60k" / "export" / "eval" / "val"
TRAIN_TAR = ROOT / "data" / "bg60k" / "export" / "dataset.tar"
STYLE_PROMPTS = ROOT / "training" / "bg60k_style_prompts.toml"
REQUIREMENTS = ROOT / "training" / "requirements-colab.txt"
ASSIGNMENTS = ROOT / "training" / "colab_assignments.py"

# Mismos checkpoints y revisiones que el worker y los experimentos (docs/05).
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
BASE_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
NINE_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
NINE_REVISION = "115134f363124c53c7d878647567d04daf26e41e"
DIFFUSERS_VERSION = "0.31.0"

# Productos del dataset mínimo del smoke test: solo prueban la mecánica, no la calidad.
SMOKE_PRODUCTS = [
    "a white ceramic mug with a small printed logo",
    "a small cardboard tea box",
    "a white plastic lotion bottle",
    "a matte black ceramic vase",
]

CLIP_MODEL = "openai/clip-vit-large-patch14"
DINO_MODEL = "facebook/dinov2-base"
# Las mismas revisiones que los embeddings de BG60k (20261115-022838-embed).
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
DINO_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"

RESERVE_S = 10 * 60
# El CLI sube cada archivo en una sola petición en base64: los grandes van en partes.
PART_BYTES = 32 * 1024 * 1024  # para bajar el resultado y apagar después del trabajo


class Failure(Exception):
    """Error que termina la corrida con un código distinto de cero."""


def build_config(stage: str, run_id: str, steps: int | None = None, rank: int = 16,
                 learning_rate: float = 1e-4, dataset_remote: str | None = None,
                 deadline_unix: float | None = None, pack_sha256: str | None = None,
                 adapters: dict | None = None, scales: list[float] | None = None,
                 max_products: int | None = None, masked_loss: bool = False) -> dict:
    """Configuración que se sube como /content/job.json. Pura: se prueba sin Colab."""
    from productshot.domain import Style
    from productshot.services.presets import NEGATIVE_PROMPT, PRESETS

    if stage == "embed":
        import tomllib
        return {
            "run_id": run_id,
            "stage": stage,
            "deadline_unix": deadline_unix,
            "pack_sha256": pack_sha256,
            "models": {"clip": {"id": CLIP_MODEL, "revision": "main"},
                       "dino": {"id": DINO_MODEL, "revision": "main"}},
            "prompts": tomllib.loads(STYLE_PROMPTS.read_text(encoding="utf-8")),
        }
    if stage == "eval":
        return eval_config(run_id, adapters or {}, deadline_unix, scales, max_products)
    if stage not in ("smoke", "train"):
        raise ValueError(f"Etapa desconocida: {stage}")
    if stage == "train" and not dataset_remote:
        raise ValueError("La etapa train necesita un dataset (--dataset-tar).")
    smoke = stage == "smoke"
    steps = steps or (100 if smoke else 1500)
    return {
        "run_id": run_id,
        "stage": stage,
        "deadline_unix": deadline_unix,
        "diffusers_version": DIFFUSERS_VERSION,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "negative_prompt": NEGATIVE_PROMPT,
        # Caption = preset del estilo (el mismo prompt que usa la generación) + producto.
        "style_prompts": {s.value: PRESETS[s].prompt for s in Style},
        "dataset": (
            {"kind": "generate", "products": SMOKE_PRODUCTS} if smoke
            else {"kind": "upload", "remote_tar": dataset_remote}
        ),
        "train": {
            "steps": steps,
            "checkpointing_steps": max(10, steps // 4),
            "resolution": 1024,
            "rank": rank,
            "learning_rate": learning_rate,
            "grad_accum": 1,
            "mixed_precision": "bf16",
            "seed": 42,
            "masked_loss": masked_loss,
        },
        "verify": {
            "enabled": True,
            "style": "studio",
            "seed": 1234,
            "min_background_diff": 0.5,  # de 255: por debajo, el LoRA no hace nada visible
            # La taza sintética comprueba la carga; en train el efecto se mide con val, no aquí.
            "effect_required": smoke,
            # Cada pipeline con su configuración candidata (docs/05): A y G.
            "pipelines": {
                "cuatro_canales": {"model": BASE_MODEL, "revision": BASE_REVISION,
                                   "strength": 0.99, "in_channels": 4},
                "nueve_canales": {"model": NINE_MODEL, "revision": NINE_REVISION,
                                  "strength": 0.85, "in_channels": 9},
            },
        },
    }


def eval_config(run_id: str, adapters: dict, deadline_unix: float | None, scales: list[float] | None = None,
                max_products: int | None = None) -> dict:
    """Mismo pipeline, prompts y parámetros que producción (SdxlParams); solo cambia el LoRA."""
    from productshot.domain import Style
    from productshot.services.presets import NEGATIVE_PROMPT, PRESETS
    from productshot.services.sdxl_generation import DEFAULT_MODEL_ID, SdxlParams

    if not adapters:
        raise ValueError("La etapa eval necesita adaptadores (--adapters).")
    if max_products is not None and max_products < 1:
        raise ValueError("--max-products debe ser positivo.")
    if scales and not all(math.isfinite(x) and 0 < x <= 2 for x in scales):
        raise ValueError("Los pesos deben ser finitos y estar entre 0 y 2.")
    p = SdxlParams()
    # Orden de desempate del reporte: checkpoint más temprano y después menor peso.
    order = sorted(adapters, key=lambda a: int(a.removeprefix("ckpt")))
    variants = ({a: {"adapter": a, "scale": p.lora_scale} for a in order} if not scales else
                {f"{a}@{x:g}": {"adapter": a, "scale": x} for a in order for x in sorted(scales)})
    if scales and len(variants) != len(order) * len(scales):
        raise ValueError("Hay pesos repetidos o que se confunden al nombrar las variantes.")
    if p.model_id != DEFAULT_MODEL_ID or p.model_id != BASE_MODEL or p.mask_dilate_px or p.prompt_prefix:
        raise ValueError("La configuración de producción cambió: revisar la evaluación antes de correrla.")
    return {
        "run_id": run_id,
        "stage": "eval",
        "deadline_unix": deadline_unix,
        "diffusers_version": DIFFUSERS_VERSION,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        # Producción resuelve esta revisión al cargar; la evaluación fija BASE_REVISION (docs/05).
        "production_revision": p.revision,
        "negative_prompt": NEGATIVE_PROMPT,
        "style_prompts": {s.value: PRESETS[s].prompt for s in Style},
        "generation": {"steps": p.steps, "guidance_scale": p.guidance_scale, "strength": p.strength,
                       "target_side": p.target_side, "mask_blur": p.mask_blur, "lora_scale": p.lora_scale},
        "adapters": adapters,
        # Sin --scales, cada checkpoint con el peso de producción; con --scales, cada par checkpoint@peso.
        "variants": variants,
        "max_products": max_products,
        "models": {"clip": {"id": CLIP_MODEL, "revision": CLIP_REVISION},
                   "dino": {"id": DINO_MODEL, "revision": DINO_REVISION}},
        # Igual que bg60k_pack.py, para que CLIP y DINOv2 vean lo mismo que en los embeddings.
        "preprocess": {"side": 224, "neutral": 128, "jpeg_quality": 92},
        "order_seed": 0,
        "seed_rule": "int(sha256(f'{id}:{style}')[:8], 16)",
    }


def collect_adapters(run_dir: Path) -> dict[str, Path]:
    """Checkpoints de una corrida de entrenamiento, del más temprano al más tardío."""
    found = {}
    for path in run_dir.glob("checkpoint-*_pytorch_lora_weights.safetensors"):
        found[f"ckpt{int(path.name.split('_')[0].split('-')[1])}"] = path
    if not found:
        raise SystemExit(f"No hay checkpoints en {run_dir}")
    return dict(sorted(found.items(), key=lambda kv: int(kv[0][4:])))


def build_eval_tar(dest: Path, split_dir: Path, train_tar: Path, adapters: dict[str, Path]) -> dict:
    """val (foto, recorte y eval.jsonl), fotos de train con su estilo y los adaptadores."""
    with tarfile.open(dest, "w") as tar:
        for path in sorted(split_dir.iterdir()):
            tar.add(path, arcname=f"val/{path.name}")
        with tarfile.open(train_tar) as src:
            for member in src.getmembers():
                if member.isfile():
                    info = tarfile.TarInfo(f"train/{member.name}")
                    info.size, info.mtime, info.mode = member.size, member.mtime, member.mode
                    tar.addfile(info, src.extractfile(member))
        for name, path in adapters.items():
            tar.add(path, arcname=f"adapters/{name}.safetensors")
    return {name: {"file": f"adapters/{name}.safetensors", "source": path.name,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in adapters.items()}


def eval_report():
    spec = importlib.util.spec_from_file_location("lora_eval_report", ROOT / "training" / "lora_eval_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_result(out_dir: Path, run_id: str) -> dict:
    """Exige el resultado completo de ESTA corrida. Pura: se prueba sin Colab.

    No basta con `status: ok`: el adaptador tiene que haber bajado, coincidir con el sha256
    que registró el trabajo y, si había verificación, haberla pasado.
    """
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise Failure("No se bajó el manifiesto: el trabajo no terminó o no se pudo descargar.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != run_id:
        raise Failure(f"El manifiesto es de otra corrida: {manifest.get('run_id')} != {run_id}")
    if manifest.get("status") != "ok":
        raise Failure(f"El trabajo remoto no terminó bien: {manifest.get('status')}")
    if manifest.get("stage") == "embed":
        check_embeddings(out_dir, manifest)
        return manifest
    if manifest.get("stage") == "eval":
        for name, expected in manifest.get("files", {}).items():
            path = out_dir / name
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise Failure(f"{name} falta o no coincide con el sha256 registrado en el VM.")
        if set(manifest.get("files", {})) != {"metrics.jsonl", "style_refs.npz"}:
            raise Failure("El manifiesto de eval no lista metrics.jsonl y style_refs.npz.")
        if not manifest.get("completed"):
            raise Failure("La evaluación no completó ningún producto.")
        try:
            eval_report().validate(out_dir, manifest)
        except (ValueError, OSError, KeyError) as exc:
            raise Failure(f"Resultado de eval inválido: {exc}") from exc
        return manifest
    weights = out_dir / "pytorch_lora_weights.safetensors"
    if not weights.exists():
        raise Failure("Falta el adaptador: solo se recuperó el manifiesto.")
    expected = manifest.get("train", {}).get("adapter_sha256")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != expected:
        raise Failure("El adaptador bajado no coincide con el sha256 registrado en el VM.")
    if manifest.get("config", {}).get("verify", {}).get("enabled") and manifest.get("verify", {}).get("passed") is not True:
        raise Failure(f"La verificación no pasó: {manifest.get('verify', {}).get('failures')}")
    return manifest


def check_embeddings(out_dir: Path, manifest: dict) -> None:
    """Integridad y contenido de los embeddings: ids, dimensiones, valores finitos y normas."""
    import numpy as np

    for name, expected in manifest.get("files", {}).items():
        path = out_dir / name
        if not path.exists():
            raise Failure(f"Falta {name}: solo se recuperó el manifiesto.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise Failure(f"{name} no coincide con el sha256 registrado en el VM.")
    if set(manifest.get("files", {})) != {"clip.npz", "dino.npz", "index.json"}:
        raise Failure("El manifiesto no lista los tres archivos esperados.")

    index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    ids = [item["id"] for item in index["items"]]
    cut_ids = [item["id"] for item in index["items"] if item["cutout"]]
    clip, dino = np.load(out_dir / "clip.npz"), np.load(out_dir / "dino.npz")
    if clip["ids"].tolist() != ids or dino["ids"].tolist() != cut_ids or len(set(ids)) != len(ids):
        raise Failure("Los ids de los embeddings no coinciden con el índice del paquete.")
    blocks = {"clip_image": clip["clip_image"], "clip_text": clip["clip_text"], "dino": dino["dino"]}
    if clip["clip_image"].shape[1] != clip["clip_text"].shape[1]:
        raise Failure("CLIP de imagen y de texto tienen dimensiones distintas.")
    if len(clip["prompt_group"]) != len(clip["clip_text"]):
        raise Failure("Las descripciones no coinciden con sus embeddings.")
    for name, block in blocks.items():
        block = block.astype(np.float32)
        norms = np.linalg.norm(block, axis=1)
        if not np.isfinite(block).all() or np.abs(norms - 1).max() > 1e-2:
            raise Failure(f"{name}: valores no finitos o normas lejos de 1.")


def split_parts(path: Path, folder: Path, size: int = PART_BYTES) -> list[Path]:
    parts = []
    with path.open("rb") as src:
        while chunk := src.read(size):
            part = folder / f"{path.name}.part{len(parts):03d}"
            part.write_bytes(chunk)
            parts.append(part)
    return parts


def colab_bin() -> str:
    path = os.environ.get("COLAB_BIN") or shutil.which("colab")
    if not path:
        raise SystemExit("No encuentro el CLI `colab` (instalar: uv tool install google-colab-cli).")
    return path


def cli_python(bin_path: str) -> str:
    """Intérprete del CLI (primera línea del ejecutable), para `colab_assignments.py`."""
    first = Path(bin_path).read_text(encoding="utf-8", errors="replace").splitlines()[0]
    return first[2:].strip().split()[0] if first.startswith("#!") else sys.executable


class Colab:
    def __init__(self, session: str, state: Path, deadline: float, dry_run: bool) -> None:
        self.session = session
        self.state = state
        self.deadline = deadline
        self.dry_run = dry_run
        self.bin = "colab" if dry_run else colab_bin()

    def remaining(self, reserve: float = 0.0) -> float:
        return self.deadline - reserve - time.time()

    def __call__(self, *args: str, session: bool = True, timeout: float = 120.0, check: bool = True,
                 budget: bool = True) -> subprocess.CompletedProcess | None:
        """Una llamada al CLI con timeout. `budget` la limita también al presupuesto restante."""
        cmd = [self.bin, "--auth=oauth2", "--config", str(self.state), *args]
        if session:
            cmd[5:5] = ["-s", self.session]
        if budget:
            timeout = min(timeout, max(5.0, self.remaining()))
        print("+ " + " ".join(cmd), flush=True)
        if self.dry_run:
            return None
        try:
            proc = subprocess.run(cmd, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            if check:
                raise Failure(f"`colab {args[0]}` superó {timeout:.0f} s") from exc
            print(f"[driver] `colab {args[0]}` superó {timeout:.0f} s", flush=True)
            return None
        if check and proc.returncode != 0:
            raise Failure(f"`colab {args[0]}` terminó con código {proc.returncode}")
        return proc

    def assignments(self) -> list[dict] | None:
        """Asignaciones en el servidor; None si no se pudo consultar."""
        if self.dry_run:
            print(f"+ <python del CLI> {ASSIGNMENTS.name} list", flush=True)
            return []
        try:
            out = subprocess.run([cli_python(self.bin), str(ASSIGNMENTS), "list"], capture_output=True,
                                 text=True, timeout=90)
            return json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else None
        except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError, OSError):
            return None

    def unassign(self, endpoints: list[str]) -> bool:
        """True si el auxiliar terminó bien; nunca lanza (el apagado debe seguir)."""
        if self.dry_run or not endpoints:
            return True
        try:
            proc = subprocess.run([cli_python(self.bin), str(ASSIGNMENTS), "unassign", *endpoints], timeout=180)
            return proc.returncode == 0
        except (subprocess.TimeoutExpired, OSError) as exc:
            print(f"[driver] no se pudo liberar {endpoints}: {exc!r}", flush=True)
            return False


def shutdown(colab: Colab, before: set[str], keep: bool) -> bool:
    """Detiene el VM de esta corrida y concilia con el servidor. True si no quedó nada nuestro."""
    if keep:
        print("[driver] --keep: el VM sigue vivo y gastando unidades.", flush=True)
        return True
    for attempt in range(2):
        proc = colab("stop", check=False, timeout=120, budget=False)
        if colab.dry_run or (proc is not None and proc.returncode == 0):
            break
        print(f"[driver] stop falló (intento {attempt + 1}); reintento", flush=True)
    for _ in range(3):
        try:
            now = colab.assignments()
        except Exception as exc:  # noqa: BLE001 - el apagado no puede cortarse aquí
            print(f"[driver] no se pudo consultar el servidor: {exc!r}", flush=True)
            now = None
        if now is None:
            time.sleep(10)
            continue
        ours = [a["endpoint"] for a in now if a["endpoint"] not in before]
        if not ours:
            return True
        print(f"[driver] quedan VM de esta corrida en el servidor: {ours}; liberándolos", flush=True)
        colab.unassign(ours)
        time.sleep(10)
    print("[driver] ATENCIÓN: no pude confirmar que el VM se liberó. Correr `--cleanup` y revisar "
          "`colab --auth=oauth2 sessions`.", flush=True)
    return False


def cleanup(dry_run: bool) -> int:
    colab = Colab("productshot-lora", Path(tempfile.gettempdir()) / "colab-cleanup.json", time.time() + 600, dry_run)
    found = colab.assignments()
    if found is None:
        print("No se pudo consultar el servidor.")
        return 1
    print(f"Asignaciones: {found}")
    colab.unassign([a["endpoint"] for a in found])
    left = colab.assignments()
    print(f"Quedan: {left}")
    return 0 if left == [] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("smoke", "train", "embed", "eval"))
    ap.add_argument("--cleanup", action="store_true", help="libera TODOS los VM asignados a la cuenta")
    ap.add_argument("--gpu", default="L4", help="T4, L4, A100 o H100; cada una gasta distinto")
    ap.add_argument("--session", default="productshot-lora")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--dataset-tar", type=Path, help="tar de una carpeta imagefolder con metadata.jsonl")
    ap.add_argument("--masked-loss", action="store_true", help="train: pérdida solo sobre el fondo (el tar trae máscaras)")
    ap.add_argument("--pack", type=Path, help="paquete de bg60k_pack.py (etapa embed)")
    ap.add_argument("--adapters", type=Path, help="carpeta de una corrida train con sus checkpoints (etapa eval)")
    ap.add_argument("--checkpoints", type=lambda v: v.split(","), help="eval: solo estos, p. ej. ckpt300,ckpt1200")
    ap.add_argument("--scales", type=lambda v: [float(x) for x in v.split(",")], help="eval: pesos del LoRA a probar")
    ap.add_argument("--max-products", type=int, help="eval: primeros N productos del orden fijo")
    ap.add_argument("--max-minutes", type=float, default=50.0, help="presupuesto total desde que se pide el VM")
    ap.add_argument("--idle-timeout", type=float, default=900.0,
                    help="segundos sin salida antes de que `colab exec` deje de esperar")
    ap.add_argument("--out", type=Path, help="por defecto data/lora o, en embed, data/bg60k/embeddings")
    ap.add_argument("--keep", action="store_true", help="no detener el VM al terminar (gasta unidades)")
    ap.add_argument("--dry-run", action="store_true", help="muestra los comandos sin ejecutarlos")
    args = ap.parse_args()

    if args.cleanup:
        return cleanup(args.dry_run)
    if not args.stage:
        ap.error("falta --stage (o --cleanup)")
    if args.stage == "embed" and not args.pack:
        ap.error("la etapa embed necesita --pack")
    if args.stage == "eval" and not args.adapters:
        ap.error("la etapa eval necesita --adapters")
    if args.max_minutes * 60 <= RESERVE_S + 5 * 60:
        ap.error("--max-minutes debe dejar al menos 5 minutos además de la reserva de 10")

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{args.stage}"
    embed = args.stage == "embed"
    evaluate = args.stage == "eval"
    out_root = args.out or (args.adapters / "eval" if evaluate else ROOT / "data" / ("bg60k/embeddings" if embed else "lora"))
    out_dir = out_root / run_id
    state_dir = Path(tempfile.mkdtemp(prefix="colab-state-")) if args.dry_run else out_dir
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.max_minutes * 60
    colab = Colab(args.session, state_dir / "colab-state.json", deadline, args.dry_run)
    remote_tar = "dataset.tar" if args.dataset_tar else None
    pack_sha = hashlib.sha256(args.pack.read_bytes()).hexdigest() if embed else None
    parts_dir = Path(tempfile.mkdtemp(prefix="colab-parts-"))
    adapters, eval_parts = None, []
    if evaluate:
        eval_tar = parts_dir / "eval.tar"
        found = collect_adapters(args.adapters)
        if args.checkpoints:
            missing = set(args.checkpoints) - set(found)
            if missing:
                ap.error(f"checkpoints inexistentes: {sorted(missing)} (hay {list(found)})")
            found = {k: v for k, v in found.items() if k in args.checkpoints}
        adapters = build_eval_tar(eval_tar, EVAL_SPLIT, TRAIN_TAR, found)
    cfg = build_config(args.stage, run_id, args.steps, args.rank, args.learning_rate, remote_tar,
                       deadline_unix=deadline - RESERVE_S, pack_sha256=pack_sha, adapters=adapters,
                       scales=args.scales, max_products=args.max_products, masked_loss=args.masked_loss)
    if evaluate:
        eval_parts = split_parts(eval_tar, parts_dir) if not args.dry_run else []
        cfg["eval_tar"] = {"parts": len(eval_parts), "sha256": hashlib.sha256(eval_tar.read_bytes()).hexdigest()}
        cfg["adapters_run"] = args.adapters.name
    pack_parts = split_parts(args.pack, parts_dir) if embed and not args.dry_run else []
    if embed:
        cfg["pack_parts"] = len(pack_parts)
    dataset_parts = split_parts(args.dataset_tar, parts_dir) if args.dataset_tar and not args.dry_run else []
    if args.dataset_tar:
        cfg["dataset"]["parts"] = len(dataset_parts)
        cfg["dataset"]["sha256"] = hashlib.sha256(args.dataset_tar.read_bytes()).hexdigest()
    print(f"RUN_ID={run_id}", flush=True)

    before_list = colab.assignments()
    if before_list is None:
        print("No se pudo consultar el servidor de Colab; no se pide ningún VM.")
        return 1
    if before_list:
        print(f"La cuenta ya tiene VM asignados: {before_list}. Liberarlos (`--cleanup`) o esperar.")
        return 1
    before = {a["endpoint"] for a in before_list}
    colab("usage", session=False, check=False, timeout=60)

    code = 0
    released = True
    try:
        # El stop se intenta aunque `new` falle a medias: el servidor pudo asignar el VM igual.
        colab("new", "--gpu", args.gpu, timeout=600)
        colab("usage", session=False, check=False, timeout=60)  # tasa de consumo de esta GPU
        if not embed:
            colab("upload", str(REQUIREMENTS), "/content/requirements-colab.txt", timeout=180)
        if evaluate:
            colab("upload", str(JOB_SCRIPT), "/content/colab_job.py", timeout=180)
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / "job.json"
            job.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            colab("upload", str(job), "/content/job.json", timeout=180)
        for part in dataset_parts:
            colab("upload", str(part), f"/content/{part.name.replace(args.dataset_tar.name, remote_tar)}", timeout=300)
        for part in eval_parts:
            colab("upload", str(part), f"/content/{part.name}", timeout=300)
        for part in pack_parts:
            colab("upload", str(part), f"/content/{part.name.replace(args.pack.name, 'pack.tar')}", timeout=300)
        if embed and args.dry_run:
            colab("upload", f"{args.pack}.part000", "/content/pack.tar.part000", timeout=300)
        try:
            colab("exec", "-f", str(EMBED_SCRIPT if embed else EVAL_SCRIPT if evaluate else JOB_SCRIPT), "--timeout", str(args.idle_timeout), check=False,
                  timeout=colab.remaining(reserve=RESERVE_S - 120))
        finally:
            # Lo que haya, aunque el trabajo falle: el manifiesto trae el error.
            colab("download", "/content/out/result.tar.gz", str(out_dir / "result.tar.gz"),
                  check=False, timeout=900 if evaluate else 420, budget=False)
            if (out_dir / "result.tar.gz").exists():
                with tarfile.open(out_dir / "result.tar.gz") as tar:
                    tar.extractall(out_dir, filter="data")
            elif not args.dry_run:
                colab("download", "/content/out/manifest.json", str(out_dir / "manifest.json"),
                      check=False, timeout=120, budget=False)
        if not args.dry_run:
            manifest = check_result(out_dir, run_id)
            if embed:
                summary = {k: manifest[k] for k in ("pack", "reference_check", "clip_image", "dino")}
                print(f"[driver] OK. Embeddings: {json.dumps(summary, ensure_ascii=False)[:2000]}")
            elif evaluate:
                print(f"[driver] OK. {len(manifest['completed'])} productos evaluados (parcial: {manifest.get('partial')}), "
                      f"referencias de estilo: {json.dumps(manifest.get('style_references'), ensure_ascii=False)[:600]}")
            else:
                verify = manifest.get("verify", {})
                print(f"[driver] OK. Verificación: {json.dumps(verify.get('pipelines', {}), ensure_ascii=False)[:2000]}")
            print(f"Resultados en {out_dir}", flush=True)
    except Failure as exc:
        print(f"[driver] FALLA: {exc}", flush=True)
        code = 1
    except KeyboardInterrupt:
        print("[driver] Interrumpido: apagando el VM.", flush=True)
        code = 130
    finally:
        try:
            released = shutdown(colab, before, args.keep)
        except Exception as exc:  # noqa: BLE001 - informar en vez de salir sin aviso
            print(f"[driver] ATENCIÓN: el apagado falló ({exc!r}). Correr `--cleanup`.", flush=True)
            released = False
        colab("sessions", session=False, check=False, timeout=60, budget=False)
        colab("usage", session=False, check=False, timeout=60, budget=False)
        shutil.rmtree(parts_dir, ignore_errors=True)
    return code if released else 3


if __name__ == "__main__":
    sys.exit(main())
