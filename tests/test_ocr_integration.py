import json
import re
import warnings
from pathlib import Path

import pytest
from PIL import Image

from cola_label_verification.ocr import extract_label_info_from_application_images

FIXTURE_ROOTS = (Path("tests/fixtures/samples"),)

_ABV_RE = re.compile(r"(\d+(?:\.\d+)?)")
_NET_CONTENTS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(milliliter|milliliters|ml|liter|liters|l|fl\.?\s*oz|oz)",
    re.IGNORECASE,
)


def _iter_fixture_dirs() -> list[Path]:
    fixture_dirs: list[Path] = []
    for root in FIXTURE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            if path.is_dir() and (path / "data.json").exists():
                fixture_dirs.append(path)
    return fixture_dirs


def _load_fixture_data(fixture_dir: Path) -> dict[str, object]:
    return json.loads((fixture_dir / "data.json").read_text(encoding="utf-8"))


def _open_images(fixture_dir: Path, data: dict[str, object]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for name in data.get("images", []):
        image_path = fixture_dir / "images" / name
        if not image_path.exists():
            warnings.warn(f"Missing image file: {image_path}", stacklevel=2)
            continue
        images.append(Image.open(image_path))
    return images


def _parse_abv(value: str | None) -> float | None:
    if not value:
        return None
    match = _ABV_RE.search(value)
    if not match:
        return None
    return float(match.group(1))


def _parse_net_contents(value: str | None) -> float | None:
    if not value:
        return None
    match = _NET_CONTENTS_RE.search(value)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower().replace(".", "").replace(" ", "")
    if unit in {"ml", "milliliter", "milliliters"}:
        return amount
    if unit in {"l", "liter", "liters"}:
        return amount * 1000
    if unit in {"floz", "oz"}:
        return amount * 29.5735
    return None


def _close_images(images: list[Image.Image]) -> None:
    for image in images:
        image.close()


def _assert_close(actual: float, expected: float, *, tolerance: float) -> None:
    assert abs(actual - expected) <= tolerance


def _warn_missing(fixture_dir: Path, field_name: str) -> None:
    warnings.warn(
        f"Fixture {fixture_dir.name} missing `{field_name}` for verification.",
        stacklevel=2,
    )


FIXTURE_DIRS = _iter_fixture_dirs()


def _fixture_id(fixture_dir: Path) -> str:
    return f"{fixture_dir.parent.name}/{fixture_dir.name}"


def test_fixtures_present() -> None:
    assert FIXTURE_DIRS, "No fixtures found to test."


@pytest.mark.parametrize(
    "fixture_dir",
    FIXTURE_DIRS,
    ids=[_fixture_id(fixture_dir) for fixture_dir in FIXTURE_DIRS],
)
def test_ocr_against_fixture(fixture_dir: Path) -> None:
    data = _load_fixture_data(fixture_dir)
    fields = data.get("fields", {})
    if not isinstance(fields, dict):
        raise AssertionError(f"Fixture {fixture_dir} has invalid fields payload.")

    images = _open_images(fixture_dir, data)
    if not images:
        warnings.warn(
            f"Fixture {fixture_dir.name} has no readable images.",
            stacklevel=2,
        )
        return

    try:
        label_info = extract_label_info_from_application_images(
            images,
        )
    finally:
        _close_images(images)

    expected_abv = _parse_abv(fields.get("alcohol_content"))
    expected_net = _parse_net_contents(fields.get("net_contents"))
    expected_warning = fields.get("warning_text")

    if expected_abv is None:
        _warn_missing(fixture_dir, "alcohol_content")
    else:
        extracted_abv = _parse_abv(label_info.alcohol_content.value)
        assert extracted_abv is not None, (
            f"{fixture_dir.name}: expected ABV {expected_abv}, got None"
        )
        _assert_close(extracted_abv, expected_abv, tolerance=0.6)

    if expected_net is None:
        _warn_missing(fixture_dir, "net_contents")
    else:
        extracted_net = _parse_net_contents(label_info.net_contents.value)
        assert extracted_net is not None, (
            f"{fixture_dir.name}: expected net contents {expected_net}mL, got None"
        )
        tolerance = max(5.0, expected_net * 0.02)
        _assert_close(extracted_net, expected_net, tolerance=tolerance)

    if expected_warning is None:
        _warn_missing(fixture_dir, "warning_text")
    else:
        extracted_warning = label_info.warning_text.value
        assert extracted_warning is not None, (
            f"{fixture_dir.name}: expected warning text, got None"
        )
        assert "GOVERNMENT WARNING" in extracted_warning.upper()
