from collections.abc import Iterable

from PIL import Image, ImageFilter, ImageOps


def _iter_image_variants(
    image: Image.Image,
    *,
    enhance: bool,
    geometry_safe: bool = False,
) -> Iterable[Image.Image]:
    yield image
    if not enhance:
        return

    # ImageOps.autocontrast doesn't support RGBA; normalize to RGB for variants.
    base = image.convert("RGB") if image.mode != "RGB" else image

    base_variants = [
        ImageOps.autocontrast(base),
        base.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)),
        ImageOps.autocontrast(ImageOps.grayscale(base)),
    ]
    inverted = ImageOps.invert(ImageOps.autocontrast(base))
    base_variants.append(inverted)
    for variant in base_variants:
        yield variant

    if geometry_safe:
        return

    for variant in base_variants[:2]:
        yield _resize_image(variant, scale=1.5)

    for angle in (-8, 8):
        yield base_variants[0].rotate(angle, expand=True, fillcolor=(255, 255, 255))

    for strength in (-0.25, -0.45):
        dewarped = _apply_cylindrical_warp(base_variants[0], strength=strength)
        yield dewarped
        yield ImageOps.autocontrast(dewarped)


def _resize_image(image: Image.Image, *, scale: float) -> Image.Image:
    width, height = image.size
    resized = image.resize(
        (int(width * scale), int(height * scale)),
        resample=Image.Resampling.LANCZOS,
    )
    return resized


def _apply_cylindrical_warp(image: Image.Image, *, strength: float) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        return image
    slices = min(24, width)
    mesh: list[tuple[tuple[int, int, int, int], tuple[float, ...]]] = []
    for i in range(slices):
        x0 = width * i / slices
        x1 = width * (i + 1) / slices
        center = (x0 + x1) / 2
        norm = abs((center - width / 2) / (width / 2))
        scale = 1 + strength * (norm**2)
        in_x0 = center - (x1 - x0) * scale / 2
        in_x1 = center + (x1 - x0) * scale / 2
        in_x0 = max(0.0, in_x0)
        in_x1 = min(float(width), in_x1)
        bbox = (int(x0), 0, int(x1), height)
        quad = (
            in_x0,
            0.0,
            in_x1,
            0.0,
            in_x1,
            float(height),
            in_x0,
            float(height),
        )
        mesh.append((bbox, quad))
    return image.transform(
        image.size,
        Image.Transform.MESH,
        mesh,
        resample=Image.Resampling.BICUBIC,
    )
