"""Encuadre de catálogo y armonización de luz (preparación común a los dos modos).

Encuadre: en la foto casera el producto queda donde cayó (descentrado, chico, con media mesa
vacía). `reframe` arma un lienzo con la proporción pedida (4:5 por defecto, la de Instagram)
con el producto centrado y apoyado un poco por debajo del centro, ocupando `fill` del lado
que limita. Se recorta desde la foto ORIGINAL, así el producto no pierde resolución. Lo que
cae fuera de la foto queda como fondo editable (blanco en la máscara) y el generador lo
completa. Los píxeles del producto no cambian: solo se recortan y escalan.

Armonización: el producto conserva la luz de la foto casera (fría y plana) y la escena
generada puede ser cálida o dorada, así que se ve "pegado". `harmonize` corrige suavemente
el balance de blancos y el brillo del producto hacia los de su entorno inmediato. No toca
forma, texto ni logotipo; la magnitud se limita y se registra (G8 la reporta aparte).
"""

from __future__ import annotations

from PIL import Image, ImageChops, ImageFilter, ImageStat

ASPECTS = {"4:5": (4, 5), "1:1": (1, 1), "original": None}


def product_bbox(mask: Image.Image) -> tuple[int, int, int, int] | None:
    """Caja del producto (negro en la máscara del proyecto)."""
    return mask.convert("L").point(lambda v: 255 if v < 128 else 0).getbbox()


def reframe(image: Image.Image, mask: Image.Image, *, aspect: str = "4:5", fill: float = 0.6,
            out_long: int = 1024, anchor: float = 0.55) -> tuple[Image.Image, Image.Image]:
    """Lienzo centrado en el producto. `image` y `mask` deben tener el mismo tamaño.

    Devuelve (imagen, máscara) al tamaño de salida, con el lado mayor `out_long`. El centro
    del producto queda en la mitad horizontal y a `anchor` del alto. Si la foto no alcanza,
    el borde nuevo se rellena con una versión difuminada de la foto y queda editable.
    """
    if image.size != mask.size:
        raise ValueError("La foto y la máscara deben tener el mismo tamaño.")
    if aspect not in ASPECTS:
        raise ValueError(f"Proporción desconocida: {aspect!r}")
    box = product_bbox(mask)
    if box is None:
        raise ValueError("La máscara no tiene producto.")
    ratio = ASPECTS[aspect]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    ar = (ratio[0] / ratio[1]) if ratio else image.width / image.height
    # El producto ocupa `fill` del lado que limita.
    ch = max(bh / fill, bw / (fill * ar))
    cw = ch * ar
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = round(cx - cw / 2), round(cy - anchor * ch)
    cw, ch = round(cw), round(ch)

    # Fondo de relleno: la foto escalada y muy difuminada (solo orienta colores; es editable).
    base = image.convert("RGB").resize((max(1, cw // 8), max(1, ch // 8)), Image.Resampling.BILINEAR)
    canvas = base.filter(ImageFilter.GaussianBlur(4)).resize((cw, ch), Image.Resampling.BILINEAR)
    canvas.paste(image.convert("RGB"), (-left, -top))
    canvas_mask = Image.new("L", (cw, ch), 255)
    canvas_mask.paste(mask.convert("L"), (-left, -top))

    scale = out_long / max(cw, ch)
    size = (max(8, round(cw * scale)), max(8, round(ch * scale)))
    return (canvas.resize(size, Image.Resampling.LANCZOS),
            canvas_mask.resize(size, Image.Resampling.BILINEAR))


def harmonize(final: Image.Image, mask: Image.Image, *, color: float = 0.35, light: float = 0.2,
              max_gain: float = 1.25, ring_px: int = 48) -> tuple[Image.Image, dict]:
    """Acerca el balance de blancos y el brillo del producto a los de su entorno inmediato.

    `final` es la imagen ya recompuesta (producto original y fondo generado). Solo cambia la
    región del producto, con ganancias por canal acotadas a [1/max_gain, max_gain].
    """
    final = final.convert("RGB")
    product = mask.convert("L").point(lambda v: 255 - v)  # 255 = producto (bordes suaves)
    hard = product.point(lambda v: 255 if v >= 128 else 0)
    ring = ImageChops.subtract(hard.filter(ImageFilter.MaxFilter(2 * (ring_px // 2) + 1)), hard)
    if not hard.getbbox() or not ring.getbbox():
        return final, {"applied": False}
    p_mean = ImageStat.Stat(final, mask=hard).mean
    r_mean = ImageStat.Stat(final, mask=ring).mean
    p_lum = sum(p_mean) / 3 or 1.0
    r_lum = sum(r_mean) / 3 or 1.0
    gains = []
    for pc, rc in zip(p_mean, r_mean):
        chroma = ((rc / r_lum) / max(pc / p_lum, 1e-3)) ** color  # balance de blancos
        lum = (r_lum / p_lum) ** light  # brillo
        gains.append(min(max_gain, max(1 / max_gain, chroma * lum)))
    adjusted = Image.merge("RGB", [ch.point(lambda v, g=g: min(255, round(v * g)))
                                   for ch, g in zip(final.split(), gains)])
    out = Image.composite(adjusted, final, product)
    return out, {"applied": True, "gains": [round(g, 3) for g in gains]}


def clean_mask(mask: Image.Image, *, min_ratio: float = 0.15, grid: int = 256) -> tuple[Image.Image, dict]:
    """Quita de la máscara los fragmentos sueltos (un pedazo de cable, una mancha).

    Conserva los objetos protegidos cuya área es al menos `min_ratio` del más grande: una
    caja junto al producto se queda y un trozo de cable en la esquina pasa a ser fondo. Las
    componentes se buscan en una versión reducida (`grid` px) y el resultado se aplica a la
    máscara original, así los bordes suaves no cambian.
    """
    mask = mask.convert("L")
    small = mask.copy()
    small.thumbnail((grid, grid), Image.Resampling.NEAREST)
    w, h = small.size
    px = small.load()
    seen = [[False] * w for _ in range(h)]
    comps: list[list[tuple[int, int]]] = []
    for y in range(h):
        for x in range(w):
            if seen[y][x] or px[x, y] >= 128:
                continue
            stack, comp = [(x, y)], []
            seen[y][x] = True
            while stack:
                cx, cy = stack.pop()
                comp.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and px[nx, ny] < 128:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            comps.append(comp)
    if not comps:
        return mask, {"components": 0, "removed": 0}
    largest = max(len(c) for c in comps)
    drop = [c for c in comps if len(c) < min_ratio * largest]
    if not drop:
        return mask, {"components": len(comps), "removed": 0}
    erase = Image.new("L", (w, h), 0)
    ep = erase.load()
    for comp in drop:
        for x, y in comp:
            ep[x, y] = 255
    # Un poco de margen para que el borde suave del fragmento también desaparezca.
    erase = erase.filter(ImageFilter.MaxFilter(3)).resize(mask.size, Image.Resampling.NEAREST)
    return ImageChops.lighter(mask, erase), {"components": len(comps), "removed": len(drop)}
