"""Segmentador rembg sin el modelo: se verifica la convención de la máscara."""

from __future__ import annotations

from PIL import Image

from productshot.services.rembg_segmentation import RembgSegmentationService


def fake_remove(image, session, only_mask, post_process_mask):
    assert only_mask and post_process_mask and session == "sesion-falsa"
    alpha = Image.new("L", image.size, 0)  # rembg: blanco = producto
    alpha.paste(255, (10, 10, 30, 30))
    return alpha


def test_mask_is_inverted_to_project_convention():
    seg = RembgSegmentationService(session_factory=lambda model: "sesion-falsa", remove_fn=fake_remove)
    result = seg.segment(Image.new("RGB", (40, 40), "white"))
    assert result.simulated is False and seg.simulated is False
    assert result.mask.getpixel((20, 20)) == 0  # producto protegido
    assert result.mask.getpixel((2, 2)) == 255  # fondo editable
    assert seg.last_seconds is not None
