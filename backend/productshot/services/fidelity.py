"""Verificación de fidelidad del producto para el modo intermedio (G8 "verificado").

El modo intermedio re-ilumina el producto, así que no se pueden recomponer los píxeles
originales (G8 "recompuesto"). En su lugar, cada candidata se compara con la foto original
dentro de la región del producto, con métricas que toleran el cambio de luz:

- `structure`: SSIM sobre la luminancia normalizada (z-score dentro de la caja del producto).
  Un cambio de iluminación global se cancela al normalizar; un logo deformado, un texto
  ilegible o un producto espejado bajan la SSIM.
- `edges`: correlación de Pearson entre los mapas de bordes (Sobel) dentro de la máscara. El
  texto impreso y los contornos del logo son, sobre todo, bordes.
- `matches` y `inlier_ratio`: el editor re-dibuja el producto y lo mueve o reescala un poco, así
  que primero se ALINEA la candidata con la foto (puntos ORB + similitud estimada con RANSAC sobre
  la región del producto). Un producto espejado o cambiado casi no tiene puntos que calcen.
  `structure` y `edges` se miden después de alinear.
- `scene_change`: diferencia media del FONDO (0 a 255) entre la candidata y la foto. Si es
  baja, el modelo no editó la escena y la candidata se descarta.

La similitud semántica DINOv2 se calcula en el worker (necesita torch) y se suma a estas.
La alineación usa OpenCV (`cv2`, ya presente en el worker por G3); el resto, numpy y Pillow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageStat

SIDE = 256  # resolución de comparación del recorte (lado mayor)


@dataclass(frozen=True)
class FidelityThresholds:
    """Calibrados el 2026-11-16 (`deploy/modal/calib_fidelity.py`, IC de Wilson al 95 %).

    Con 3 candidatas por estilo: pasan 106/120 buenas [81–93 %] y se rechazan 60/60 espejadas
    [94–100 %], 54/60 con texto o logo borroso [80–95 %] y 8/9 fallas reales [56–98 %].
    `structure` (SSIM) queda registrada pero fuera de la regla: falla en superficies lisas.
    """

    min_inlier_ratio: float = 0.45
    min_edges: float = 0.40
    min_scene_change: float = 25.0


def product_box(mask: Image.Image, pad: float = 0.05) -> tuple[int, int, int, int]:
    """Caja del producto (negro en la máscara del proyecto) con un margen relativo."""
    prod = mask.convert("L").point(lambda v: 255 if v < 128 else 0)
    box = prod.getbbox()
    if box is None:
        raise ValueError("La máscara no tiene producto.")
    x0, y0, x1, y1 = box
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    w, h = mask.size
    return max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)


def _gray(img: Image.Image, box, size) -> np.ndarray:
    return np.asarray(img.convert("L").crop(box).resize(size, Image.Resampling.BILINEAR), dtype=np.float64)


def _box_filter(a: np.ndarray, r: int) -> np.ndarray:
    """Media en ventanas de (2r+1)² con sumas acumuladas (sin scipy)."""
    p = np.pad(a, r + 1, mode="edge")
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s[: a.shape[0], : a.shape[1]] / (k * k)


def _ssim(a: np.ndarray, b: np.ndarray, weight: np.ndarray | None, r: int = 3) -> float:
    # z-score: tolera brillo y contraste globales distintos (re-iluminación).
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    c1, c2 = 0.01, 0.03
    ma, mb = _box_filter(a, r), _box_filter(b, r)
    va = _box_filter(a * a, r) - ma * ma
    vb = _box_filter(b * b, r) - mb * mb
    cov = _box_filter(a * b, r) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    if weight is not None and weight.sum() > 0:
        return float((s * weight).sum() / weight.sum())
    return float(s.mean())


def _sobel(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    gx = (p[1:-1, 2:] - p[1:-1, :-2]) * 2 + (p[:-2, 2:] - p[:-2, :-2]) + (p[2:, 2:] - p[2:, :-2])
    gy = (p[2:, 1:-1] - p[:-2, 1:-1]) * 2 + (p[2:, :-2] - p[:-2, :-2]) + (p[2:, 2:] - p[:-2, 2:])
    return np.hypot(gx, gy)


def structure_metrics(original: Image.Image, candidate: Image.Image, mask: Image.Image) -> dict:
    """`structure`, `edges` y `scene_change` de una candidata frente a la foto original."""
    if candidate.size != original.size:
        candidate = candidate.resize(original.size, Image.Resampling.LANCZOS)
    mask = mask.convert("L").resize(original.size, Image.Resampling.BILINEAR)
    box = product_box(mask)
    bw, bh = box[2] - box[0], box[3] - box[1]
    scale = SIDE / max(bw, bh)
    size = (max(16, round(bw * scale)), max(16, round(bh * scale)))
    a, b = _gray(original, box, size), _gray(candidate, box, size)
    weight = (255 - np.asarray(mask.crop(box).resize(size, Image.Resampling.BILINEAR), dtype=np.float64)) / 255
    structure = _ssim(a, b, weight)
    ea, eb = _sobel(a) * weight, _sobel(b) * weight
    edges = float(np.corrcoef(ea.ravel(), eb.ravel())[0, 1]) if ea.std() > 0 and eb.std() > 0 else 0.0
    small = (128, 128)
    bg = mask.resize(small, Image.Resampling.NEAREST).point(lambda v: 255 if v >= 128 else 0)
    change = ImageStat.Stat(ImageChops.difference(candidate.convert("L").resize(small),
                                                  original.convert("L").resize(small)), mask=bg).mean[0]
    return {"structure": round(structure, 4), "edges": round(edges, 4), "scene_change": round(change, 1)}


def passes(metrics: dict, t: FidelityThresholds = FidelityThresholds()) -> bool:
    return (bool(metrics.get("aligned")) and metrics["inlier_ratio"] >= t.min_inlier_ratio
            and metrics["edges"] >= t.min_edges and metrics["scene_change"] >= t.min_scene_change)


def score(metrics: dict) -> float:
    """Orden entre candidatas que pasan: alineación por bordes (mayor es mejor)."""
    return metrics["inlier_ratio"] * max(0.0, metrics["edges"])


def align_to_original(original: Image.Image, candidate: Image.Image, mask: Image.Image,
                      min_inliers: int = 8) -> tuple[Image.Image | None, dict]:
    """Alinea la candidata al encuadre de la foto original usando la región del producto.

    Devuelve la candidata re-proyectada (o None si no hay suficientes puntos que calcen) y
    `matches` (inliers) e `inlier_ratio`. Usa una transformación de similitud (escala,
    rotación y traslación): un espejo no se puede representar, así que no alinea.
    """
    import cv2

    if candidate.size != original.size:
        candidate = candidate.resize(original.size, Image.Resampling.LANCZOS)
    a = np.asarray(original.convert("L"))
    b = np.asarray(candidate.convert("L"))
    prod = (np.asarray(mask.convert("L").resize(original.size)) < 128).astype(np.uint8) * 255
    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=7)
    ka, da = orb.detectAndCompute(a, prod)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < min_inliers or len(kb) < min_inliers:
        return None, {"matches": 0, "inlier_ratio": 0.0}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < min_inliers:
        return None, {"matches": len(good), "inlier_ratio": 0.0}
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(dst, src, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    n_in = int(inl.sum()) if inl is not None else 0
    info = {"matches": n_in, "inlier_ratio": round(n_in / len(good), 3)}
    if M is None or n_in < min_inliers:
        return None, info
    warped = cv2.warpAffine(np.asarray(candidate.convert("RGB")), M, original.size, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped), info


def body_mask(mask: Image.Image, frac: float = 0.03) -> Image.Image:
    """Quita partes delgadas del producto (cables, antenas) para verificar solo el cuerpo.

    Apertura morfológica (erosión + dilatación) con un tamaño relativo al lado menor de la
    caja: un cable re-dibujado con otra forma no es un cambio del producto. Si la apertura
    borra casi todo (un producto delgado de por sí), se devuelve la máscara original.
    """
    from PIL import ImageFilter

    m = mask.convert("L")
    x0, y0, x1, y1 = product_box(m, pad=0)
    k = max(3, int(min(x1 - x0, y1 - y0) * frac) * 2 + 1)
    prod = m.point(lambda v: 255 if v < 128 else 0)
    opened = prod.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k))
    before = sum(prod.histogram()[1:])
    after = sum(opened.histogram()[1:])
    if before == 0 or after < 0.3 * before:
        return m
    return opened.point(lambda v: 0 if v else 255)  # vuelve a la convención: negro = producto


def verify(original: Image.Image, candidate: Image.Image, mask: Image.Image) -> dict:
    """Métricas completas de una candidata: alineación, estructura y bordes alineados, cambio de escena."""
    if candidate.size != original.size:
        candidate = candidate.resize(original.size, Image.Resampling.LANCZOS)
    full_mask = mask
    mask = body_mask(mask.resize(original.size))
    scene = structure_metrics(original, candidate, full_mask)["scene_change"]
    aligned, info = align_to_original(original, candidate, mask)
    if aligned is None:
        return {**info, "structure": 0.0, "edges": 0.0, "scene_change": scene, "aligned": False}
    m = structure_metrics(original, aligned, mask)
    return {**info, "structure": m["structure"], "edges": m["edges"], "scene_change": scene, "aligned": True}
