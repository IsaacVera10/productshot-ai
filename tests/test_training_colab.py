"""Entrenamiento LoRA en Colab: configuración, driver y rutas de falla, sin Colab ni GPU.

Las rutas de falla usan un CLI `colab` falso (un script que simula el VM con carpetas) y
sustituyen la consulta de asignaciones al servidor.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from productshot.domain import Style
from productshot.services.presets import NEGATIVE_PROMPT, PRESETS

ROOT = Path(__file__).resolve().parents[1]

FAKE_COLAB = textwrap.dedent('''\
    #!{python}
    """CLI colab falso: el VM es la carpeta FAKE_REMOTE."""
    import hashlib, json, os, shutil, sys, tarfile
    from pathlib import Path

    args = [a for a in sys.argv[1:] if not a.startswith("--auth")]
    if args[0] == "--config":
        args = args[2:]
    cmd = args[0]
    rest = args[1:]
    if rest[:1] == ["-s"]:
        rest = rest[2:]
    remote = Path(os.environ["FAKE_REMOTE"])
    with open(os.environ["FAKE_LOG"], "a") as fh:
        fh.write(cmd + "\\n")
    fail = os.environ.get("FAKE_FAIL", "")
    if cmd == fail:
        sys.exit(1)
    if cmd == "upload":
        dest = remote / rest[1].lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(rest[0], dest)
    elif cmd == "exec":
        job = json.loads((remote / "content/job.json").read_text())
        out = remote / "content/out"
        out.mkdir(parents=True, exist_ok=True)
        status = os.environ.get("FAKE_STATUS", "ok")
        run_id = os.environ.get("FAKE_RUN_ID", job["run_id"])
        weights = out / "pytorch_lora_weights.safetensors"
        weights.write_bytes(b"pesos-falsos")
        manifest = {{
            "run_id": run_id, "status": status, "config": job,
            "train": {{"adapter_sha256": hashlib.sha256(b"pesos-falsos").hexdigest()}},
            "verify": {{"passed": os.environ.get("FAKE_VERIFY", "1") == "1", "failures": []}},
        }}
        (out / "manifest.json").write_text(json.dumps(manifest))
        with tarfile.open(out / "result.tar.gz", "w:gz") as tar:
            tar.add(out / "manifest.json", arcname="manifest.json")
            if not os.environ.get("FAKE_NO_ADAPTER"):
                tar.add(weights, arcname=weights.name)
        # Como el CLI real: devuelve 0 aunque el código remoto falle.
    elif cmd == "download":
        src = remote / rest[0].lstrip("/")
        if not src.exists():
            sys.exit(1)
        Path(rest[1]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, rest[1])
''')


def load_driver():
    spec = importlib.util.spec_from_file_location("run_colab", ROOT / "training" / "run_colab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REAL_UNASSIGN = load_driver().Colab.unassign


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Driver con CLI falso. Devuelve (driver, ejecutar, comandos registrados, asignaciones)."""
    drv = load_driver()
    bin_path = tmp_path / "colab"
    bin_path.write_text(FAKE_COLAB.format(python=sys.executable))
    bin_path.chmod(0o755)
    log = tmp_path / "calls.log"
    monkeypatch.setenv("COLAB_BIN", str(bin_path))
    monkeypatch.setenv("FAKE_REMOTE", str(tmp_path / "vm"))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setattr(drv.time, "sleep", lambda s: None)
    server = {"lists": [], "unassigned": []}

    def assignments(self):
        return server["lists"].pop(0) if server["lists"] else []

    monkeypatch.setattr(drv.Colab, "assignments", assignments)
    monkeypatch.setattr(drv.Colab, "unassign", lambda self, eps: server["unassigned"].extend(eps))

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["run_colab.py", "--stage", "smoke", "--out", str(tmp_path / "out"), *argv])
        code = drv.main()
        calls = log.read_text().split() if log.exists() else []
        return code, calls

    return drv, run, server


def test_smoke_config_uses_production_prompts_and_pinned_models():
    drv = load_driver()
    cfg = drv.build_config("smoke", "r1", deadline_unix=123.0)
    assert cfg["style_prompts"] == {s.value: PRESETS[s].prompt for s in Style}
    assert cfg["negative_prompt"] == NEGATIVE_PROMPT
    assert cfg["dataset"]["kind"] == "generate"
    assert cfg["train"]["steps"] == 100 and cfg["train"]["checkpointing_steps"] == 25
    assert cfg["deadline_unix"] == 123.0
    pipes = cfg["verify"]["pipelines"]
    # Las revisiones son SHA completos, iguales a las de los experimentos (docs/05).
    assert all(len(p["revision"]) == 40 for p in pipes.values())
    assert {p["in_channels"] for p in pipes.values()} == {4, 9}
    assert cfg["base_revision"] == pipes["cuatro_canales"]["revision"]
    json.dumps(cfg)  # se sube como JSON


def test_train_stage_requires_dataset():
    drv = load_driver()
    with pytest.raises(ValueError):
        drv.build_config("train", "r2")
    cfg = drv.build_config("train", "r2", steps=800, dataset_remote="dataset.tar")
    assert cfg["dataset"] == {"kind": "upload", "remote_tar": "dataset.tar"}
    assert cfg["train"]["steps"] == 800


def test_requirements_match_modal_worker_pins():
    """El adaptador se entrena y se sirve con la misma pila de generación."""
    worker = (ROOT / "deploy" / "modal" / "worker.py").read_text(encoding="utf-8")
    reqs = (ROOT / "training" / "requirements-colab.txt").read_text(encoding="utf-8")
    for line in reqs.splitlines():
        line = line.strip()
        if "==" in line and not line.startswith("#"):
            name = line.split("==")[0]
            if f'"{name}==' in worker:
                assert f'"{line}"' in worker, f"{line} difiere del worker de Modal"


def test_colab_job_is_self_contained():
    """En el VM solo existe ese archivo: no puede importar el paquete del proyecto."""
    tree = ast.parse((ROOT / "training" / "colab_job.py").read_text(encoding="utf-8"))
    imported = {
        (node.module if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "productshot" not in imported


def test_dry_run_prints_plan_with_stop_after_exec(tmp_path):
    out = subprocess.run(
        [sys.executable, str(ROOT / "training" / "run_colab.py"), "--stage", "smoke", "--dry-run"],
        capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "backend"), "TMPDIR": str(tmp_path)},
    ).stdout
    cmds = [line[2:] for line in out.splitlines() if line.startswith("+ ")]
    idx = {name: next(i for i, c in enumerate(cmds) if f" {name} " in f"{c} ") for name in ("new", "exec", "stop")}
    assert idx["new"] < idx["exec"] < idx["stop"]


def test_success_downloads_validates_and_stops(fake):
    drv, run, server = fake
    code, calls = run()
    assert code == 0
    assert calls.index("stop") > calls.index("exec") > calls.index("new")


def test_remote_error_fails_even_if_cli_returns_zero(fake, monkeypatch):
    drv, run, server = fake
    monkeypatch.setenv("FAKE_STATUS", "error: RuntimeError('smoke test falló')")
    code, calls = run()
    assert code == 1 and "stop" in calls


def test_manifest_from_another_run_is_rejected(fake, monkeypatch):
    drv, run, server = fake
    monkeypatch.setenv("FAKE_RUN_ID", "otra-corrida")
    code, calls = run()
    assert code == 1 and "stop" in calls


def test_failed_new_still_tries_to_stop(fake, monkeypatch):
    drv, run, server = fake
    monkeypatch.setenv("FAKE_FAIL", "new")
    code, calls = run()
    assert code == 1 and "stop" in calls and "exec" not in calls


def test_orphan_vm_is_unassigned_and_reported(fake, monkeypatch):
    drv, run, server = fake
    monkeypatch.setenv("FAKE_FAIL", "stop")
    # Antes de empezar: nada. Después del stop fallido: queda el VM. Luego: liberado.
    server["lists"] = [[], [{"endpoint": "vm-1", "accelerator": "L4"}], []]
    code, calls = run()
    assert server["unassigned"] == ["vm-1"]
    assert code == 0


def test_unconfirmed_release_exits_with_3(fake, monkeypatch):
    drv, run, server = fake
    monkeypatch.setenv("FAKE_FAIL", "stop")
    stuck = [{"endpoint": "vm-1", "accelerator": "L4"}]
    server["lists"] = [[], stuck, stuck, stuck]
    code, calls = run()
    assert code == 3


def test_refuses_to_start_if_account_has_a_vm(fake):
    drv, run, server = fake
    server["lists"] = [[{"endpoint": "vm-0", "accelerator": "T4"}]]
    code, calls = run()
    assert code == 1 and "new" not in calls and server["unassigned"] == []


def test_missing_adapter_fails(fake, monkeypatch):
    drv, run, server = fake
    monkeypatch.setenv("FAKE_NO_ADAPTER", "1")
    code, calls = run()
    assert code == 1 and "stop" in calls


def test_failed_verification_fails(fake, monkeypatch):
    drv, run, server = fake
    monkeypatch.setenv("FAKE_VERIFY", "0")
    code, calls = run()
    assert code == 1 and "stop" in calls


def test_unassign_timeout_reports_and_exits_3(fake, monkeypatch):
    """Un auxiliar colgado no puede cortar el apagado sin aviso."""
    drv, run, server = fake
    monkeypatch.setenv("FAKE_FAIL", "stop")
    stuck = [{"endpoint": "vm-1", "accelerator": "L4"}]
    server["lists"] = [[], stuck, stuck, stuck]
    real_run = drv.subprocess.run

    def run_or_hang(cmd, *a, **kw):
        if "unassign" in cmd:
            raise drv.subprocess.TimeoutExpired(cmd, 180)
        return real_run(cmd, *a, **kw)

    # El fixture reemplaza `unassign`; aquí se usa el real, con el auxiliar colgado.
    monkeypatch.setattr(drv.Colab, "unassign", REAL_UNASSIGN)
    monkeypatch.setattr(drv, "cli_python", lambda b: sys.executable)
    monkeypatch.setattr(drv.subprocess, "run", run_or_hang)
    code, calls = run()
    assert code == 3



def test_deadline_kills_the_whole_process_group(tmp_path):
    """El plazo mata también al hijo que crea el lanzador (como `accelerate launch`)."""
    spec = importlib.util.spec_from_file_location("colab_job", ROOT / "training" / "colab_job.py")
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    pid_file = tmp_path / "child.pid"
    parent = (
        "import subprocess, sys, time\n"
        f"c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(c.pid))\n"
        "time.sleep(60)\n"
    )
    job.DEADLINE = time.time() + 2
    t0 = time.time()
    with pytest.raises(TimeoutError):
        job.run_streaming([sys.executable, "-c", parent])
    assert time.time() - t0 < 20
    child = int(pid_file.read_text())
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)
