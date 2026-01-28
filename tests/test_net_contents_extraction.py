import json
import warnings
from pathlib import Path

import pytest
from PIL import Image

from cola_label_verification.models import LabelInfo
from cola_label_verification.ocr import extract_label_info_with_spans
from cola_label_verification.rules import evaluate_checklist

FIXTURE_DIR = Path("tests/fixtures/samples/25100001000415")


def _find_finding(result, rule_id: str):
    for finding in result.findings:
        if finding.rule_id == rule_id:
            return finding
    raise AssertionError(f"{rule_id} finding not found")


def test_net_contents_extraction(
    crystal_springs_extraction: LabelInfo,
) -> None:
    result = evaluate_checklist(
        crystal_springs_extraction,
    )
    finding = _find_finding(result, "net_contents_presence")
    assert finding.status == "pass"


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


def test_net_contents_extraction_uses_vlm_fixture() -> None:
    if not FIXTURE_DIR.exists():
        pytest.skip(f"Fixture directory missing: {FIXTURE_DIR}")
    data = _load_fixture_data(FIXTURE_DIR)
    images = _open_images(FIXTURE_DIR, data)
    if not images:
        pytest.skip(f"Fixture {FIXTURE_DIR.name} has no readable images.")

    try:
        extraction = extract_label_info_with_spans(images)
        label_info = extraction.label_info
    finally:
        _close_images(images)

    result = evaluate_checklist(
        label_info,
        spans=extraction.spans,
    )
    finding = _find_finding(result, "net_contents_presence")
    assert finding.status == "pass"
