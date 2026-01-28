#!/usr/bin/env python3
"""
Generate synthetic COLA fixtures with image effects for OCR testing.

Outputs a fixtures directory with the same per-sample structure as real data:
  <out>/<sample_id>/data.json
  <out>/<sample_id>/meta.json
  <out>/<sample_id>/images/<image files>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


WARNING_TEXT = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic "
    "beverages during pregnancy because of the risk of birth defects. (2) Consumption of "
    "alcoholic beverages impairs your ability to drive a car or operate machinery, and may "
    "cause health problems."
)

TEXT_TAGS = [
    "brand.case",
    "brand.punctuation",
    "class.abbrev",
    "abv.proof",
    "net_contents.units",
    "warning.case_mismatch",
    "warning.punctuation",
    "warning.linebreaks",
]

IMAGE_TAGS = [
    "image.curvature.mild",
    "image.curvature.strong",
    "image.blur.gaussian",
    "image.blur.heavy",
    "image.grain.low",
    "image.grain.high",
    "image.glare",
    "image.low_contrast",
    "image.skew",
    "image.rotation",
    "image.compression.low",
    "image.compression.high",
]

ALL_TAGS = TEXT_TAGS + IMAGE_TAGS

BACKGROUND = (245, 243, 238)
TEXT_COLOR = (28, 26, 24)
BORDER_COLOR = (40, 40, 40)
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
]


@dataclass(frozen=True)
class LabelSpec:
    ttb_id: str
    brand_name: str
    class_type: str
    abv: float
    proof: int | None
    net_contents_ml: int
    type_of_product: str
    source_of_product: str


@dataclass(frozen=True)
class LabelText:
    brand_name: str
    class_type: str
    alcohol_content: str
    net_contents: str
    warning: str
    font_path: str | None


@dataclass(frozen=True)
class ImageInfo:
    filename: str
    image: Image.Image


def available_fonts() -> list[str]:
    return [path for path in FONT_CANDIDATES if Path(path).exists()]


def load_font(size: int, font_path: str | None) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_width(text: str, font: ImageFont.ImageFont) -> float:
    if hasattr(font, "getlength"):
        return float(font.getlength(text))
    bbox = font.getbbox(text)
    return float(bbox[2] - bbox[0])


def wrap_text_pixels(text: str, max_width: int, font: ImageFont.ImageFont) -> str:
    lines: list[str] = []
    for paragraph in text.splitlines():
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if text_width(candidate, font) <= max_width:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return "\n".join(lines)


def format_alcohol_content(spec: LabelSpec) -> str:
    abv_text = f"{spec.abv:.1f}% Alc./Vol."
    if spec.proof:
        return f"{abv_text} ({spec.proof} Proof)"
    return abv_text


def format_net_contents_ml(ml: int) -> str:
    if ml >= 1000:
        return f"{ml / 1000:.2f} L".replace(".00", "")
    return f"{ml} mL"


def format_net_contents_oz(ml: int) -> str:
    ounces = ml / 29.5735
    return f"{ounces:.1f} FL OZ"


def generate_brand_name(rng: random.Random) -> str:
    prefixes = [
        "Stone's",
        "Old",
        "Black",
        "Highland",
        "Copper",
        "River",
        "Juniper",
        "Bright",
        "North",
        "Royal",
    ]
    suffixes = [
        "Reserve",
        "Creek",
        "Forge",
        "Valley",
        "Ridge",
        "No. 9",
        "Barrel",
        "Estate",
        "Select",
        "Craft",
    ]
    return f"{rng.choice(prefixes)} {rng.choice(suffixes)}"


def generate_class_type(rng: random.Random, product_type: str) -> str:
    options = {
        "distilled_spirits": [
            "Kentucky Straight Bourbon Whiskey",
            "Rye Whiskey",
            "London Dry Gin",
            "American Vodka",
            "Aged Rum",
        ],
        "wine": [
            "California Red Wine",
            "Oregon Pinot Noir",
            "Washington Riesling",
            "Sonoma Chardonnay",
        ],
        "malt_beverage": [
            "India Pale Ale",
            "American Lager",
            "Pilsner",
            "Hazy Pale Ale",
        ],
    }
    return rng.choice(options[product_type])


def generate_spec(sample_id: str, rng: random.Random) -> LabelSpec:
    type_of_product = rng.choice(["distilled_spirits", "wine", "malt_beverage"])
    source_of_product = rng.choice(["domestic", "imported"])
    brand_name = generate_brand_name(rng)
    class_type = generate_class_type(rng, type_of_product)

    if type_of_product == "distilled_spirits":
        abv = rng.uniform(35.0, 55.0)
    elif type_of_product == "wine":
        abv = rng.uniform(10.5, 15.5)
    else:
        abv = rng.uniform(4.0, 9.0)

    abv = round(abv * 2) / 2
    proof = int(round(abv * 2)) if type_of_product == "distilled_spirits" else None

    net_contents_ml = rng.choice([50, 200, 375, 500, 700, 750, 1000, 1750])

    return LabelSpec(
        ttb_id=sample_id,
        brand_name=brand_name,
        class_type=class_type,
        abv=abv,
        proof=proof,
        net_contents_ml=net_contents_ml,
        type_of_product=type_of_product,
        source_of_product=source_of_product,
    )


def abbreviate_class_type(value: str) -> str:
    replacements = {
        "Kentucky": "KY",
        "Straight": "Str.",
        "Bourbon": "Bbn.",
        "Whiskey": "Whsky",
        "American": "Am.",
        "California": "CA",
        "Oregon": "OR",
        "Washington": "WA",
        "Estate": "Est.",
    }
    parts = value.split()
    return " ".join(replacements.get(part, part) for part in parts)


def apply_text_edge_cases(spec: LabelSpec, tags: list[str], font_path: str | None) -> LabelText:
    brand_name = spec.brand_name
    class_type = spec.class_type
    alcohol_content = format_alcohol_content(spec)
    net_contents = format_net_contents_ml(spec.net_contents_ml)
    warning = WARNING_TEXT

    if "brand.punctuation" in tags:
        brand_name = brand_name.replace("'", "")
        brand_name = brand_name.replace("&", "AND")
    if "brand.case" in tags:
        brand_name = brand_name.title() if brand_name.isupper() else brand_name.upper()
    if "class.abbrev" in tags:
        class_type = abbreviate_class_type(class_type)
    if "abv.proof" in tags and spec.proof:
        alcohol_content = f"{spec.proof} Proof"
    if "net_contents.units" in tags:
        net_contents = format_net_contents_oz(spec.net_contents_ml)
    if "warning.case_mismatch" in tags:
        warning = warning.replace("GOVERNMENT WARNING", "Government Warning")
    if "warning.punctuation" in tags:
        warning = warning.replace(",", "").replace(".", "")
    if "warning.linebreaks" in tags:
        warning = warning.replace(": ", ":\n")

    return LabelText(
        brand_name=brand_name,
        class_type=class_type,
        alcohol_content=alcohol_content,
        net_contents=net_contents,
        warning=warning,
        font_path=font_path,
    )


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    font: ImageFont.ImageFont,
    width: int,
) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    x = int((width - text_width) / 2)
    draw.text((x, y), text, font=font, fill=TEXT_COLOR)
    return bbox[3] - bbox[1]


def render_front_label(text: LabelText, size: tuple[int, int]) -> Image.Image:
    width, height = size
    img = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(img)
    margin = int(min(width, height) * 0.05)
    draw.rectangle(
        (margin, margin, width - margin, height - margin),
        outline=BORDER_COLOR,
        width=2,
    )

    brand_font = load_font(int(height * 0.08), font_path=text.font_path)
    class_font = load_font(int(height * 0.045), font_path=text.font_path)
    info_font = load_font(int(height * 0.03), font_path=text.font_path)

    y = margin + int(height * 0.08)
    y += draw_centered_text(draw, text.brand_name, y, brand_font, width) + 10
    y += draw_centered_text(draw, text.class_type, y, class_font, width) + 12
    y += draw_centered_text(draw, text.alcohol_content, y, info_font, width) + 6
    draw_centered_text(draw, text.net_contents, y, info_font, width)

    return img


def render_back_label(text: LabelText, size: tuple[int, int]) -> Image.Image:
    width, height = size
    img = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(img)
    margin = int(min(width, height) * 0.06)
    draw.rectangle(
        (margin, margin, width - margin, height - margin),
        outline=BORDER_COLOR,
        width=2,
    )

    body_font = load_font(int(height * 0.028), font_path=text.font_path)

    y = margin + int(height * 0.06)
    x = margin + 15
    max_width = width - x - margin
    wrapped = wrap_text_pixels(text.warning, max_width=max_width, font=body_font)
    draw.multiline_text(
        (x, y),
        wrapped,
        font=body_font,
        fill=TEXT_COLOR,
        spacing=6,
    )

    return img


def apply_cylindrical_warp(img: Image.Image, strength: float) -> Image.Image:
    width, height = img.size
    slices = 24
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
        quad = (in_x0, 0.0, in_x1, 0.0, in_x1, float(height), in_x0, float(height))
        mesh.append((bbox, quad))
    return img.transform(img.size, Image.Transform.MESH, mesh, resample=Image.BICUBIC)


def apply_cylindrical_shading(img: Image.Image, strength: float) -> Image.Image:
    width, height = img.size
    values = []
    for x in range(width):
        t = abs((x - width / 2) / (width / 2))
        shade = 1 - strength * (t**2)
        values.append(int(255 * shade))
    mask = Image.new("L", (width, 1))
    mask.putdata(values)
    mask = mask.resize((width, height))
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay.putalpha(ImageOps.invert(mask).point(lambda v: int(v * 0.6)))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def apply_skew(img: Image.Image, rng: random.Random, max_shear: float = 0.12) -> Image.Image:
    shear = rng.uniform(-max_shear, max_shear)
    xshift = abs(shear) * img.size[1]
    matrix = (1, shear, -xshift if shear > 0 else 0, 0, 1, 0)
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        matrix,
        resample=Image.BICUBIC,
        fillcolor=BACKGROUND,
    )


def apply_glare(img: Image.Image, rng: random.Random, intensity: float) -> Image.Image:
    width, height = img.size
    gradient = Image.new("L", (width, 1))
    gradient.putdata([int(255 * (x / max(1, width - 1))) for x in range(width)])
    gradient = gradient.resize((width, height))
    angle = rng.choice([-35, -20, 20, 35])
    rotated = gradient.rotate(angle, expand=True)
    cropped = ImageOps.fit(rotated, (width, height), method=Image.BICUBIC)
    alpha = cropped.point(lambda v: int(v * intensity))
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    overlay.putalpha(alpha)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def apply_grain(img: Image.Image, sigma: float, alpha: float) -> Image.Image:
    noise = Image.effect_noise(img.size, sigma).convert("RGB")
    return Image.blend(img, noise, alpha)


def apply_image_effects(img: Image.Image, tags: list[str], rng: random.Random) -> Image.Image:
    if "image.curvature.strong" in tags:
        img = apply_cylindrical_warp(img, strength=0.45)
        img = apply_cylindrical_shading(img, strength=0.25)
    elif "image.curvature.mild" in tags:
        img = apply_cylindrical_warp(img, strength=0.25)
        img = apply_cylindrical_shading(img, strength=0.15)

    if "image.skew" in tags:
        img = apply_skew(img, rng)

    if "image.rotation" in tags:
        angle = rng.uniform(-2.0, 2.0)
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=BACKGROUND)

    if "image.glare" in tags:
        img = apply_glare(img, rng, intensity=0.35)

    if "image.low_contrast" in tags:
        img = ImageEnhance.Contrast(img).enhance(0.75)

    if "image.grain.high" in tags:
        img = apply_grain(img, sigma=22.0, alpha=0.18)
    elif "image.grain.low" in tags:
        img = apply_grain(img, sigma=12.0, alpha=0.1)

    if "image.blur.heavy" in tags:
        img = img.filter(ImageFilter.GaussianBlur(radius=2.2))
    elif "image.blur.gaussian" in tags:
        img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

    return img


def save_image(img: Image.Image, path: Path, tags: list[str]) -> None:
    quality = 85
    if "image.compression.high" in tags:
        quality = 18
    elif "image.compression.low" in tags:
        quality = 35
    img.save(path, format="JPEG", quality=quality, optimize=True)


def select_edge_cases(
    rng: random.Random,
    min_tags: int,
    max_tags: int,
    include_tags: list[str],
) -> list[str]:
    tags = list(dict.fromkeys(include_tags))
    available = [tag for tag in ALL_TAGS if tag not in tags]
    target = rng.randint(min_tags, max_tags)

    if not any(tag in IMAGE_TAGS for tag in tags):
        image_choices = [tag for tag in IMAGE_TAGS if tag not in tags]
        if image_choices:
            tags.append(rng.choice(image_choices))

    if len(tags) < target and available:
        extra_count = min(target - len(tags), len(available))
        tags.extend(rng.sample(available, k=extra_count))

    return tags


def build_fields(spec: LabelSpec) -> dict[str, object]:
    fields: dict[str, object] = {
        "ttb_id": spec.ttb_id,
        "brand_name": spec.brand_name,
        "class_type_description": spec.class_type,
        "alcohol_content": format_alcohol_content(spec),
        "net_contents": format_net_contents_ml(spec.net_contents_ml),
        "type_of_product": [spec.type_of_product],
        "source_of_product": [spec.source_of_product],
        "status": "APPROVED",
        "date_issued": dt.date.today().strftime("%m/%d/%Y"),
    }
    return fields


def build_images(text: LabelText, size: tuple[int, int], tags: list[str], rng: random.Random) -> list[ImageInfo]:
    front = render_front_label(text, size)
    back = render_back_label(text, size)

    front = apply_image_effects(front, tags, rng)
    back = apply_image_effects(back, tags, rng)

    return [
        ImageInfo(filename="front.jpg", image=front),
        ImageInfo(filename="back.jpg", image=back),
    ]


def list_tags() -> None:
    print("Available tags:")
    for tag in ALL_TAGS:
        print(f"- {tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/fixtures/samples_synthetic")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=1600)
    ap.add_argument("--min-tags", type=int, default=2)
    ap.add_argument("--max-tags", type=int, default=4)
    ap.add_argument("--include-tag", action="append", default=[])
    ap.add_argument("--list-tags", action="store_true")
    args = ap.parse_args()

    if args.list_tags:
        list_tags()
        return

    invalid = [tag for tag in args.include_tag if tag not in ALL_TAGS]
    if invalid:
        raise SystemExit(f"Unknown tag(s): {', '.join(invalid)}")

    if args.min_tags < 0 or args.max_tags < args.min_tags:
        raise SystemExit("--min-tags must be >= 0 and --max-tags must be >= --min-tags")

    rng = random.Random(args.seed)
    font_paths = available_fonts()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    size = (args.width, args.height)

    for i in range(1, args.count + 1):
        sample_id = f"99{i:012d}"
        spec = generate_spec(sample_id, rng)
        font_path = rng.choice(font_paths) if font_paths else None
        tags = select_edge_cases(rng, args.min_tags, args.max_tags, args.include_tag)
        label_text = apply_text_edge_cases(spec, tags, font_path)
        images = build_images(label_text, size, tags, rng)

        sample_dir = out_root / sample_id
        images_dir = sample_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for info in images:
            save_image(info.image, images_dir / info.filename, tags)

        data = {
            "id": sample_id,
            "source": "synthetic",
            "synthetic": True,
            "edge_cases": tags,
            "fields": build_fields(spec),
            "fields_raw_printable": {},
            "fields_raw_detail": {},
            "checkboxes": {},
            "images": [info.filename for info in images],
            "images_detail": [
                {"file": info.filename, "source_name": f"synthetic_{info.filename}"}
                for info in images
            ],
        }
        (sample_dir / "data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

        meta = {
            "ttbid": sample_id,
            "status": "ok",
            "synthetic": True,
            "generated_at": dt.datetime.now(dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {args.count} synthetic fixture(s) to {out_root}")


if __name__ == "__main__":
    main()
