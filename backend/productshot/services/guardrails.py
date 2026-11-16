"""Guardrails que no necesitan modelos (ADR-008, docs/08_ITERACIONES_Y_GUARDRAILS.md, parte B).

- G8  fidelidad: los píxeles del producto se recomponen desde la foto de trabajo con la
      máscara, y se mide cuánto los había cambiado el generador.
- G12 calidad de la foto: brillo y nitidez. Solo advierte.
- G13 heurística de máscara: área protegida fuera de rango o varios objetos. Solo advierte.
- G14 el timeout por lote vive en el orquestador (services/sessions.py).

Cada activación deja un evento `GUARDRAIL <código>` en los logs, para contar en la
evaluación cuántas veces actuó cada uno. Los umbrales están en `Settings` y son
provisionales: se calibran con los casos obligatorios del Hito 2.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import asdict, dataclass

from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

from productshot.config import Settings

log = logging.getLogger("productshot.guardrails")


@dataclass(frozen=True)
class GuardrailWarning:
    guardrail: str  # "G12", "G13", ...
    code: str  # identificador estable para la UI y las métricas
    message: str  # español, con la acción siguiente

    def to_json(self) -> dict:
        return asdict(self)


def event(guardrail: str, session_id: str | None, code: str, **data) -> None:
    """Un evento por activación, con formato estable para contarlos desde los logs."""
    extras = " ".join(f"{k}={v}" for k, v in data.items())
    log.info("GUARDRAIL %s code=%s session=%s %s", guardrail, code, session_id, extras)


# --- G12: calidad de la foto ---

_LAPLACIAN = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128)


def photo_metrics(image: Image.Image) -> dict:
    """Brillo medio (0 a 255) y nitidez (varianza del laplaciano a 512 px de lado mayor)."""
    gray = image.convert("L")
    brightness = ImageStat.Stat(gray).mean[0]
    small = gray.copy()
    small.thumbnail((512, 512))
    sharpness = ImageStat.Stat(small.filter(_LAPLACIAN)).var[0]
    return {"brightness": round(brightness, 1), "sharpness": round(sharpness, 1)}


def photo_warnings(metrics: dict, settings: Settings) -> list[GuardrailWarning]:
    out = []
    if metrics["brightness"] < settings.quality_min_brightness:
        out.append(GuardrailWarning("G12", "photo_dark",
                                    "La foto está muy oscura: puedes seguir, pero conviene tomarla con más luz."))
    elif metrics["brightness"] > settings.quality_max_brightness:
        out.append(GuardrailWarning("G12", "photo_overexposed",
                                    "La foto está muy clara o quemada: puedes seguir, pero conviene tomarla con menos luz."))
    if metrics["sharpness"] < settings.quality_min_sharpness:
        out.append(GuardrailWarning("G12", "photo_blurry",
                                    "La foto se ve borrosa o sin detalle: puedes seguir, pero conviene enfocar el producto."))
    return out


# --- G13: heurística de máscara ---


def _protected(mask: Image.Image) -> Image.Image:
    """Máscara binaria del producto (negro en la máscara de inpainting) como 255."""
    return ImageOps.invert(mask.convert("L")).point(lambda v: 255 if v > 127 else 0)


def _components(binary: Image.Image, min_pixels: int) -> list[int]:
    """Tamaños de las componentes conexas (4-vecinos) con al menos `min_pixels` píxeles."""
    w, h = binary.size
    data = binary.load()
    seen = bytearray(w * h)
    sizes = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0 * w + x0] or not data[x0, y0]:
                continue
            seen[y0 * w + x0] = 1
            queue, size = deque([(x0, y0)]), 0
            while queue:
                x, y = queue.popleft()
                size += 1
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny * w + nx] and data[nx, ny]:
                        seen[ny * w + nx] = 1
                        queue.append((nx, ny))
            if size >= min_pixels:
                sizes.append(size)
    return sorted(sizes, reverse=True)


def mask_metrics(mask: Image.Image, settings: Settings) -> dict:
    small = _protected(mask)
    small.thumbnail((96, 96), Image.Resampling.NEAREST)
    total = small.width * small.height
    protected = sum(small.histogram()[1:])  # binaria: todo lo que no es 0 es producto
    big = _components(small, max(1, int(total * settings.mask_min_component_fraction)))
    return {"protected_fraction": round(protected / total, 3), "large_components": len(big)}


def mask_warnings(metrics: dict, settings: Settings) -> list[GuardrailWarning]:
    out = []
    if metrics["protected_fraction"] < settings.mask_min_protected:
        out.append(GuardrailWarning("G13", "mask_tiny",
                                    "La máscara protege muy poco: quizá no detectó el producto. Reintenta o sube tu máscara."))
    elif metrics["protected_fraction"] > settings.mask_max_protected:
        out.append(GuardrailWarning("G13", "mask_huge",
                                    "La máscara protege casi toda la foto y deja poco fondo para cambiar. Reintenta o sube tu máscara."))
    if metrics["large_components"] >= 2:
        out.append(GuardrailWarning("G13", "mask_many_objects",
                                    "La máscara tiene varios objetos separados; la herramienta funciona mejor con un solo producto."))
    return out


# --- G15: tipo de producto (CLIP zero-shot en el worker) ---

# Criterio: se bloquea lo que Instagram y TikTok no permiten anunciar y lo que facilita fraude;
# se avisa lo que está fuera del nicho y puede salir mal. Los grupos vienen de screening.py.
CATEGORY_REJECT_MESSAGES = {
    "arma": "No trabajamos con armas ni municiones.",
    "tabaco_vape": "No trabajamos con cigarrillos ni vapes: su publicidad está restringida en las redes del nicho.",
    "drogas": "No trabajamos con drogas ni con sus accesorios.",
    "dinero_documentos": "No trabajamos con dinero, tarjetas ni documentos de identidad, por riesgo de falsificación.",
    "sexual": "No trabajamos con productos para adultos: Instagram y TikTok no permiten anunciarlos.",
}
CATEGORY_WARN_MESSAGES = {
    "comida": "Parece comida: la herramienta está pensada para objetos rígidos y el resultado puede salir mal.",
    "transparente": "Parece un objeto transparente o espejado: el fondo nuevo puede verse a través y salir mal.",
    "grande": "Parece un producto grande (mueble o electrodoméstico): la herramienta está pensada para objetos de mesa.",
    "animal": "Parece un animal: la herramienta está pensada para productos.",
    "pantalla_documento": "Parece una captura o un documento, no la foto de un producto.",
}


def category_decision(session_id: str | None, categories: dict[str, float], settings: Settings) -> list[GuardrailWarning]:
    """Rechaza (422) si un grupo bloqueado supera el umbral; si no, devuelve los avisos."""
    from productshot.domain import RejectedContent

    blocked = [(categories.get(g, 0.0), g) for g in CATEGORY_REJECT_MESSAGES]
    score, group = max(blocked, default=(0.0, None))
    if group and score >= settings.category_block_threshold:
        event("G15", session_id, f"blocked_{group}", score=round(score, 4))
        raise RejectedContent(CATEGORY_REJECT_MESSAGES[group])
    warnings = []
    for g, message in CATEGORY_WARN_MESSAGES.items():
        if categories.get(g, 0.0) >= settings.category_warn_threshold:
            event("G15", session_id, f"warn_{g}", score=round(categories[g], 4))
            warnings.append(GuardrailWarning("G15", f"out_of_niche_{g}", message))
    return warnings


# --- G8: fidelidad del producto ---


VERIFIED_KEYS = ("inlier_ratio", "edges", "scene_change", "aligned")


def verified_fidelity(mode, params: dict) -> dict | None:
    """G8 por modo: devuelve las métricas si la imagen viene VERIFICADA del modo intermedio.

    Solo se acepta en el modo intermedio y si trae el método y todas las métricas; si no, la
    API recompone el producto como siempre ("recompuesto"), que es lo seguro.
    """
    from productshot.domain import Mode

    fid = params.get("fidelity") if isinstance(params, dict) else None
    if mode != Mode.INTERMEDIATE or not isinstance(fid, dict) or fid.get("method") != "verificado":
        return None
    if not all(k in fid for k in VERIFIED_KEYS) or not fid["aligned"]:
        return None
    return dict(fid)


def enforce_fidelity(working: Image.Image, mask: Image.Image, generated: Image.Image,
                     settings: Settings) -> tuple[Image.Image, dict]:
    """Recompone el producto desde la foto de trabajo y mide cuánto lo había alterado el generador.

    Blanco en la máscara = fondo generado; negro = píxeles originales. Con una máscara de
    bordes suaves, la transición se mezcla. Devuelve la imagen recompuesta y las métricas:
    `protected_mean_abs_diff` (0 a 255, diferencia media por canal en la región protegida
    ANTES de recomponer) y `protected_max_diff_after` (máxima diferencia donde la máscara es
    negro puro DESPUÉS de recomponer; debe ser 0).
    """
    if generated.size != working.size:
        raise ValueError(f"El generador devolvió {generated.size} y la foto de trabajo mide {working.size}.")
    working, generated, mask = working.convert("RGB"), generated.convert("RGB"), mask.convert("L")
    region = _protected(mask)
    if not region.getbbox():
        raise ValueError("La máscara no protege ningún píxel.")
    before = ImageStat.Stat(ImageChops.difference(generated, working), mask=region).mean
    recomposed = Image.composite(generated, working, mask)
    # Donde la máscara es negro puro el resultado debe ser idéntico al original; en el borde
    # suave (gris) la mezcla es intencional y no cuenta.
    exact = mask.point(lambda v: 255 if v == 0 else 0)
    after_max = 0
    if exact.getbbox():
        after = ImageChops.difference(recomposed, working)
        after_max = max(hi for _lo, hi in ImageStat.Stat(after, mask=exact).extrema)
    drift = round(sum(before) / 3, 2)
    return recomposed, {
        "method": "recompuesto",
        "protected_mean_abs_diff": drift,
        "protected_max_diff_after": after_max,
        "recomposed": True,
        "drift_warning": drift > settings.fidelity_warn_diff,
    }
