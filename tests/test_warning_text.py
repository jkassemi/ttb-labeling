import json
import warnings
from pathlib import Path

import pytest
from PIL import Image

from cola_label_verification.ocr import extract_label_info_with_spans
from cola_label_verification.rules import evaluate_checklist
from cola_label_verification.rules.models import ChecklistResult, Finding
from cola_label_verification.rules.warning_text import CANONICAL_WARNING_TEXT

FIXTURE_DIR = Path("tests/fixtures/samples/25100001000415")
FIXTURE_ROOTS = (Path("tests/fixtures/samples"),)
NON_BOLD_FIXTURES = {
    # Approved, but the warning header does not appear bold in the label art.
    "23244001000241": "Warning header looks non-bold; keep needs_review.",
}


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


def _close_images(images: list[Image.Image]) -> None:
    for image in images:
        image.close()


def _warning_finding(result: ChecklistResult) -> Finding:
    for finding in result.findings:
        if finding.rule_id == "warning_text":
            return finding
    raise AssertionError("warning_text finding not found")


FIXTURE_DIRS = _iter_fixture_dirs()


def _fixture_id(fixture_dir: Path) -> str:
    return f"{fixture_dir.parent.name}/{fixture_dir.name}"


def test_warning_text_exactness_all_caps_matches_canonical() -> None:
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

    warning = label_info.warning_text
    assert warning.value is not None, "Warning text missing in fixture extraction"
    caps_warning = warning.model_copy(update={"value": CANONICAL_WARNING_TEXT.upper()})
    label_info = label_info.model_copy(update={"warning_text": caps_warning})

    result = evaluate_checklist(
        label_info,
        spans=extraction.spans,
    )
    finding = _warning_finding(result)
    assert finding.evidence is not None, "Warning text evidence missing"
    assert finding.evidence.get("warning_text_exact_match") is True


@pytest.mark.parametrize(
    "fixture_dir",
    FIXTURE_DIRS,
    ids=[_fixture_id(fixture_dir) for fixture_dir in FIXTURE_DIRS],
)
def test_warning_text_boldness_across_fixtures(fixture_dir: Path) -> None:
    data = _load_fixture_data(fixture_dir)
    images = _open_images(fixture_dir, data)
    if not images:
        pytest.skip(f"Fixture {fixture_dir.name} has no readable images.")

    try:
        extraction = extract_label_info_with_spans(images)
        label_info = extraction.label_info
        result = evaluate_checklist(
            label_info,
            images=images,
            spans=extraction.spans,
        )
    finally:
        _close_images(images)

    warning = label_info.warning_text
    assert warning.value is not None, f"{fixture_dir.name}: warning text missing"

    normalized = warning.normalized or {}
    if (
        "warning_header_bbox" not in normalized
        or "warning_header_image_index" not in normalized
    ):
        raise AssertionError(
            f"{fixture_dir.name}: warning header bbox not detected for boldness check"
        )

    warning_finding = _warning_finding(result)
    boldness = warning_finding.evidence
    assert isinstance(boldness, dict), (
        f"{fixture_dir.name}: warning boldness evidence missing"
    )
    status = boldness.get("status")
    if fixture_dir.name in NON_BOLD_FIXTURES:
        assert status == "needs_review", (
            f"{fixture_dir.name}: expected needs_review for non-bold header"
        )
    else:
        assert status in {"pass", "needs_review"}, (
            f"{fixture_dir.name}: unexpected boldness status={status!r}"
        )
