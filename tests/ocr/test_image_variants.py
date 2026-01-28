"""Tests for OCR image variant generation."""

from PIL import Image

from cola_label_verification.ocr.image_variants import (
    _apply_cylindrical_warp,
    _iter_image_variants,
    _resize_image,
)


def test_iter_image_variants_without_enhance_yields_only_original() -> None:
    image = Image.new("RGBA", (10, 8), (10, 20, 30, 40))

    variants = list(_iter_image_variants(image, enhance=False))

    assert variants == [image]
    assert variants[0] is image
    assert variants[0].mode == "RGBA"


def test_iter_image_variants_enhance_geometry_safe_base_variants_only() -> None:
    image = Image.new("RGBA", (12, 9), (100, 120, 140, 255))

    variants = list(
        _iter_image_variants(
            image,
            enhance=True,
            geometry_safe=True,
        )
    )

    assert len(variants) == 5
    assert variants[0] is image
    assert all(variant.size == image.size for variant in variants)
    modes = {variant.mode for variant in variants}
    assert "RGB" in modes
    assert "L" in modes


def test_iter_image_variants_enhance_includes_geometry_variants() -> None:
    image = Image.new("RGB", (20, 10), (10, 20, 30))

    variants = list(
        _iter_image_variants(
            image,
            enhance=True,
            geometry_safe=False,
        )
    )

    assert len(variants) == 13
    assert variants[0] is image
    sizes = {variant.size for variant in variants}
    assert image.size in sizes
    assert any(size != image.size for size in sizes)


def test_resize_image_scales_using_int_rounding() -> None:
    image = Image.new("RGB", (3, 2), (1, 2, 3))

    resized = _resize_image(image, scale=1.5)

    assert resized.size == (4, 3)
    assert resized.mode == image.mode
    assert resized is not image


def test_apply_cylindrical_warp_preserves_size_and_uniform_color() -> None:
    image = Image.new("RGB", (12, 8), (12, 34, 56))

    warped = _apply_cylindrical_warp(image, strength=-0.25)

    assert warped.size == image.size
    assert warped.mode == image.mode
    colors = warped.getcolors(maxcolors=image.size[0] * image.size[1])
    assert colors == [(image.size[0] * image.size[1], (12, 34, 56))]
