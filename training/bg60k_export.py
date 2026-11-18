"""Captions y exportación del piloto de BG60k (etapas 8 y 10 de docs/10).

    PYTHONPATH=backend uv run --group dataset python training/bg60k_export.py

Lee el split congelado (`review/split.csv`) y escribe:
- `review/captions.csv`: id, split, estilo, producto y caption, para revisarlos como texto;
- `data/bg60k/export/<name>.tar`: solo train, como imagefolder con `metadata.jsonl`. Con
  `--captions preset` el caption es solo el preset (igual que el prompt de inferencia) y con
  `--masks` cada foto lleva `<id>.mask`, la máscara del producto en PNG, para la pérdida con máscara;
- `data/bg60k/export/eval/<split>/`: val y test con la foto, el recorte RGBA y `eval.jsonl`;
- `data/bg60k/export/<name>.json`: conteos y sha256 (`export.json` para `dataset`).

Las fotos se copian sin recomprimir desde los zip de BG60k.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from productshot.domain import Style
from productshot.services.presets import PRESETS, PROMPT_VERSION

REVIEW = Path(__file__).with_name("review")
NOUNS = Path(__file__).with_name("bg60k_captions.toml")


def noun_lookup(path: Path = NOUNS) -> tuple[dict[str, str], dict[int, str]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return data["categories"], {int(k): v for k, v in data.get("ids", {}).items()}


def caption(style: str, noun: str) -> str:
    return f"{PRESETS[Style(style)].prompt}, {noun}"


def rows_for(split_csv: Path, candidates: Path) -> list[dict]:
    with split_csv.open(encoding="utf-8") as f:
        split = {int(r["id"]): r for r in csv.DictReader(f)}
    cands = {r["id"]: r for r in pq.read_table(candidates).to_pylist() if r["id"] in split}
    categories, ids = noun_lookup()
    rows = []
    for i, s in sorted(split.items()):
        c = cands[i]
        noun = ids.get(i) or categories.get(c["category"])
        if not noun:
            raise SystemExit(f"Falta el producto de la categoría {c['category']!r} en {NOUNS.name}.")
        rows.append(c | {"split": s["split"], "style": s["style"], "noun": noun, "caption": caption(s["style"], noun)})
    return rows


class Zips:
    def __init__(self, raw: Path) -> None:
        self.raw, self.open = raw, {}

    def read(self, name: str, member: str) -> bytes:
        if name not in self.open:
            self.open[name] = zipfile.ZipFile(self.raw / name)
        return self.open[name].read(member)


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime = len(data), 0
    tar.addfile(info, io.BytesIO(data))


def product_mask(cutout: bytes, size: tuple[int, int]) -> bytes:
    alpha = Image.open(io.BytesIO(cutout)).getchannel("A")
    if alpha.size != size:
        raise SystemExit(f"El recorte mide {alpha.size} y la foto {size}.")
    buf = io.BytesIO()
    alpha.point(lambda v: 255 if v >= 128 else 0).save(buf, format="PNG")
    return buf.getvalue()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(rows: list[dict], raw: Path, out: Path, name: str = "dataset", captions: str = "noun",
           masks: bool = False) -> dict:
    zips = Zips(raw)
    out.mkdir(parents=True, exist_ok=True)
    train = [r for r in rows if r["split"] == "train"]
    tmp = out / f"{name}.tar.tmp"
    with tarfile.open(tmp, "w") as tar:
        meta = []
        for r in train:
            text = r["caption"] if captions == "noun" else PRESETS[Style(r["style"])].prompt
            meta.append({"file_name": f"{r['id']}.png", "text": text, "style": r["style"]}
                        | ({"mask": f"{r['id']}.mask"} if masks else {}))
        metadata = "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta).encode()
        add_bytes(tar, "metadata.jsonl", metadata)
        for r in train:
            photo = zips.read(r["img_zip"], r["img_member"])
            add_bytes(tar, f"{r['id']}.png", photo)
            if masks:
                size = Image.open(io.BytesIO(photo)).size
                add_bytes(tar, f"{r['id']}.mask", product_mask(zips.read(r["mask_zip"], r["mask_member"]), size))
    tmp.replace(out / f"{name}.tar")

    for part in ("val", "test"):
        folder = out / "eval" / part
        folder.mkdir(parents=True, exist_ok=True)
        lines = []
        for r in (r for r in rows if r["split"] == part):
            (folder / f"{r['id']}.png").write_bytes(zips.read(r["img_zip"], r["img_member"]))
            (folder / f"{r['id']}_cutout.png").write_bytes(zips.read(r["mask_zip"], r["mask_member"]))
            lines.append(json.dumps({"id": r["id"], "style": r["style"], "noun": r["noun"],
                                     "caption": r["caption"]}, ensure_ascii=False))
        (folder / "eval.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    counts = {}
    for r in rows:
        counts.setdefault(r["style"], {"train": 0, "val": 0, "test": 0})[r["split"]] += 1
    return {"prompt_version": PROMPT_VERSION, "counts": counts, "dataset": name, "captions": captions,
            "masks": masks, "dataset_sha256": sha256(out / f"{name}.tar"),
            "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
            "split_sha256": sha256(REVIEW / "split.csv"), "captions_sha256": sha256(NOUNS)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Captions y exportación del piloto de BG60k.")
    parser.add_argument("--data", type=Path, default=Path("data/bg60k"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/bg60k"))
    parser.add_argument("--name", default="dataset", help="nombre del tar de train")
    parser.add_argument("--captions", choices=("noun", "preset"), default="noun")
    parser.add_argument("--masks", action="store_true", help="incluye la máscara del producto de cada foto")
    args = parser.parse_args(argv)

    rows = rows_for(REVIEW / "split.csv", args.data / "candidates.parquet")
    with (REVIEW / "captions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "split", "style", "category", "noun", "caption"])
        for r in rows:
            writer.writerow([r["id"], r["split"], r["style"], r["category"], r["noun"], r["caption"]])
    summary = export(rows, args.raw, args.data / "export", args.name, args.captions, args.masks)
    report = "export.json" if args.name == "dataset" else f"{args.name}.json"
    (args.data / "export" / report).write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
