"""Calibración de la verificación de fidelidad (CPU, OpenCV). Uso:
    PYTHONPATH=backend uv run --with modal modal run deploy/modal/calib_fidelity.py
"""
import io
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==2.2.6", "pillow==12.3.0", "opencv-python-headless==4.10.0.84")
         .add_local_dir(ROOT / "backend" / "productshot", "/root/productshot"))
app = modal.App("productshot-calib", image=image)


@app.function(cpu=8, memory=8192, timeout=1800)
def calibrate(cases: list[dict], files: dict[str, bytes]) -> list[dict]:
    from PIL import Image, ImageFilter, ImageOps

    from productshot.services.fidelity import product_box, verify

    img = {k: Image.open(io.BytesIO(v)).convert("RGB") for k, v in files.items()}
    out = []
    for c in cases:
        o, m = img[c["orig"]], img[c["mask"]].convert("L")
        cand = img[c["cand"]].resize(o.size)
        variants = [(c["kind"], cand)]
        if c["kind"] == "pos" and c.get("synthetic"):
            box = product_box(m)
            mir = cand.copy(); mir.paste(ImageOps.mirror(cand.crop(box)), box[:2])
            blur = cand.copy(); blur.paste(cand.crop(box).filter(ImageFilter.GaussianBlur(6)), box[:2])
            variants += [("neg_mirror", mir), ("neg_blur", blur)]
        for kind, im in variants:
            out.append({**verify(o, im, m), "kind": kind, "tag": c["tag"], "dino": c.get("dino")})
    return out


@app.local_entrypoint()
def main():
    E = ROOT / "data/experiments/edit"
    R4, R1 = E / "20261116-034958-edit4", E / "20261116-030100-edit"
    sel = json.loads((R4 / "selected.json").read_text())
    files, cases = {}, []

    def add(path):
        files[str(path)] = Path(path).read_bytes()
        return str(path)

    for s in sel:
        cases.append({"orig": add(E / "inputs" / f"{s['photo']}__full.png"), "mask": add(E / "inputs" / f"{s['photo']}__mask.png"),
                      "cand": add(R4 / f"{s['image']}.jpg"), "kind": "pos", "synthetic": s["mode"] == "A",
                      "tag": f"{s['photo']} {s['mode']} {s['style']}", "dino": s["fidelity"]})
    for cfg, photo, styles in [("K2", "002_cup", ["warm", "lifestyle", "minimal"]), ("KX", "002_cup", ["studio", "lifestyle", "minimal"]),
                               ("K2", "001_candle", ["studio", "natural", "minimal"])]:
        for st in styles:
            cases.append({"orig": add(E / "inputs" / f"{photo}__full.png"), "mask": add(E / "inputs" / f"{photo}__mask.png"),
                          "cand": add(R1 / f"{photo}__{cfg}__full__{st}.jpg"), "kind": "neg_real", "tag": f"{photo} {cfg} {st}"})
    rows = calibrate.remote(cases, files)
    (E / "calibracion_fidelidad.json").write_text(json.dumps(rows, indent=1))
    import statistics as stt
    for k in ("pos", "neg_real", "neg_mirror", "neg_blur"):
        xs = [r for r in rows if r["kind"] == k]
        def q(v):
            v = sorted(v)
            return f"min {v[0]:.2f} p10 {v[len(v)//10]:.2f} med {stt.median(v):.2f}"
        print(f"{k:10s} n={len(xs):3d} aligned {sum(r['aligned'] for r in xs)}/{len(xs)} | matches {q([r['matches'] for r in xs])} | "
              f"inl {q([r['inlier_ratio'] for r in xs])} | structure {q([r['structure'] for r in xs])} | edges {q([r['edges'] for r in xs])}")
    print("negativas reales:")
    for r in rows:
        if r["kind"] == "neg_real":
            print("  ", r["tag"], r["matches"], r["inlier_ratio"], r["structure"], r["edges"])
    print("positivas más bajas:")
    for r in sorted([r for r in rows if r["kind"] == "pos"], key=lambda r: (r["matches"], r["structure"]))[:8]:
        print("  ", r["tag"], r["matches"], r["inlier_ratio"], r["structure"], r["edges"], r["dino"])
