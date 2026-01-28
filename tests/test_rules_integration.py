import json
import warnings
from pathlib import Path
from typing import Literal

import pytest
from PIL import Image

from cola_label_verification.ocr import extract_label_info_with_spans
from cola_label_verification.rules import ApplicationFields, evaluate_checklist
from cola_label_verification.rules.models import FindingStatus

FIXTURE_ROOT = Path("tests/fixtures/samples")
STATUS_SET: set[FindingStatus] = {
    "pass",
    "fail",
    "needs_review",
    "not_applicable",
    "not_evaluated",
}


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


def _beverage_type(
    fields: dict[str, object],
) -> Literal["distilled_spirits", "wine"] | None:
    types = fields.get("type_of_product", [])
    if isinstance(types, list):
        if "distilled_spirits" in types:
            return "distilled_spirits"
        if "wine" in types:
            return "wine"
    return None


def _application_fields(fields: dict[str, object]) -> ApplicationFields:
    source_of_product: tuple[str, ...] | None = None
    raw_source = fields.get("source_of_product")
    if isinstance(raw_source, list):
        source_of_product = tuple(item for item in raw_source if isinstance(item, str))
    return ApplicationFields(
        brand_name=fields.get("brand_name"),
        class_type=fields.get("class_type_description")
        or fields.get("class_type_code"),
        alcohol_content=fields.get("alcohol_content"),
        net_contents=fields.get("net_contents"),
        name_and_address=fields.get("applicant_name_address"),
        grape_varietals=fields.get("grape_varietals"),
        appellation_of_origin=fields.get("wine_appellation"),
        source_of_product=source_of_product,
    )


@pytest.mark.parametrize(
    "fixture_id",
    [
        "11054001000049",  # wine
        "14323001000602",  # distilled spirits
    ],
)
def test_rules_engine_integration(fixture_id: str) -> None:
    fixture_dir = FIXTURE_ROOT / fixture_id
    data = _load_fixture_data(fixture_dir)
    fields = data.get("fields", {})
    if not isinstance(fields, dict):
        raise AssertionError(f"Fixture {fixture_id} has invalid fields payload.")

    images = _open_images(fixture_dir, data)
    if not images:
        pytest.skip(f"Fixture {fixture_id} has no readable images.")

    try:
        extraction = extract_label_info_with_spans(images)
        label_info = extraction.label_info
    finally:
        _close_images(images)

    result = evaluate_checklist(
        label_info,
        application_fields=_application_fields(fields),
        spans=extraction.spans,
    )

    assert result.findings, "Checklist produced no findings."
    assert all(finding.status in STATUS_SET for finding in result.findings)
    assert any(f.rule_id == "warning_text" for f in result.findings)
    if _beverage_type(fields) == "wine":
        assert any(f.rule_id == "grape_varietals" for f in result.findings)


def test_warning_text_boldness_thresholds_todo() -> None:
    pytest.xfail("TODO: calibrate boldness thresholds against labeled samples.")


def test_appellation_additional_triggers_todo() -> None:
    pytest.xfail(
        "TODO: cover appellation triggers beyond varietal (vintage, semi-generic, "
        "estate bottling)."
    )


def test_varietal_blend_percentage_todo() -> None:
    pytest.xfail("TODO: check multi-varietal percentage disclosure rules.")
