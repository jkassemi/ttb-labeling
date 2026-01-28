import json
import warnings
from pathlib import Path

import pytest
from PIL import Image

from cola_label_verification.models import LabelInfo
from cola_label_verification.ocr import extract_label_info_from_application_images

CRYSTAL_SPRINGS_DIR = Path("tests/fixtures/samples/18295001000454")


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


def _close_images(images: list[Image.Image]) -> None:
    for image in images:
        image.close()


@pytest.fixture(scope="session")
def crystal_springs_extraction() -> LabelInfo:
    fixture_dir = CRYSTAL_SPRINGS_DIR
    if not fixture_dir.exists():
        pytest.skip(f"Fixture directory missing: {fixture_dir}")
    data = _load_fixture_data(fixture_dir)
    images = _open_images(fixture_dir, data)
    if not images:
        pytest.skip(f"Fixture {fixture_dir.name} has no readable images.")
    try:
        return extract_label_info_from_application_images(images)
    finally:
        _close_images(images)
