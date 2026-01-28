import json
import re
import warnings
from functools import lru_cache
from pathlib import Path

import pytest
from PIL import Image

from cola_label_verification.ocr import extract_label_info_from_application_images

FIXTURE_ROOT = Path("tests/fixtures/samples")
MALT_FIXTURE = FIXTURE_ROOT / "23244001000241"
SPIRIT_FIXTURE = FIXTURE_ROOT / "18295001000454"
CONTEXT_ROOT = Path("context")
MALT_CONTEXT = CONTEXT_ROOT / "malt-beverage-class-types.txt"
SPIRIT_CONTEXT = CONTEXT_ROOT / "spirits-class-types.txt"

_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
_SUPERSCRIPT_RE = re.compile(r"[\u00b9\u00b2\u00b3]+")
_TERM_CLEAN_RE = re.compile(r"[^A-Z0-9]+")
_HEADING_EXCLUDE = {
    "CHAPTER 4",
    "CLASS AND TYPE DESIGNATION",
    "GENERAL FEATURES",
    "DEFINITIONS",
    "DEFINITION",
    "DEFINITION OF MALT BEVERAGE",
    "GENERAL CLASSIFICATION DEFINITION",
    "CLASSES OF MALT BEVERAGES",
    "TYPES OF MALT BEVERAGES",
    "CLASSES AND TYPES",
    "CLASS",
    "TYPE",
    "GENERAL CLASS",
    "GENERAL TYPE",
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


def _assert_fixture_type(fields: dict[str, object], expected: str) -> None:
    raw_types = fields.get("type_of_product", [])
    if not isinstance(raw_types, list):
        raise AssertionError("Fixture `type_of_product` must be a list.")
    assert expected in raw_types


def _normalize_heading(text: str) -> str:
    cleaned = _SUPERSCRIPT_RE.sub("", text)
    cleaned = _NON_ASCII_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _heading_from_line(line: str) -> str | None:
    tokens = line.split()
    if not tokens:
        return None
    heading_tokens: list[str] = []
    for token in tokens:
        if any(char.isalpha() and char.islower() for char in token):
            break
        heading_tokens.append(token)
    if not heading_tokens:
        return None
    heading = _normalize_heading(" ".join(heading_tokens))
    if not heading:
        return None
    if heading.upper().startswith("VOL "):
        return None
    if heading in _HEADING_EXCLUDE:
        return None
    if "NO TYPE UNDER THIS CLASS" in heading:
        return None
    return heading


def _normalize_term(value: str) -> str:
    cleaned = _NON_ASCII_RE.sub("", value)
    cleaned = _TERM_CLEAN_RE.sub(" ", cleaned.upper()).strip()
    return re.sub(r"\s+", " ", cleaned)


@lru_cache(maxsize=1)
def _context_terms(path: Path) -> set[str]:
    if not path.exists():
        return set()
    terms: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = _heading_from_line(line.strip())
        if not heading:
            continue
        for part in re.split(r"\s+OR\s+|/", heading):
            normalized = _normalize_term(part.strip(" -"))
            if normalized and normalized not in _HEADING_EXCLUDE:
                terms.add(normalized)
    return terms


def _matches_allowed_term(value: str, allowed_terms: set[str]) -> bool:
    normalized_value = _normalize_term(value)
    padded = f" {normalized_value} "
    return any(f" {term} " in padded for term in allowed_terms)


def _assert_class_type(value: str | None, allowed_terms: set[str]) -> None:
    assert value is not None
    assert value == value.strip()
    assert value[0].isalnum()
    assert value[-1].isalnum()
    assert _matches_allowed_term(value, allowed_terms)


def test_class_type_extraction_spirits_strips_punctuation(
    crystal_springs_extraction,
) -> None:
    if not SPIRIT_FIXTURE.exists():
        pytest.skip(f"Fixture directory missing: {SPIRIT_FIXTURE}")
    data = _load_fixture_data(SPIRIT_FIXTURE)
    fields = data.get("fields", {})
    if not isinstance(fields, dict):
        raise AssertionError("Fixture has invalid fields payload.")
    _assert_fixture_type(fields, "distilled_spirits")

    allowed_terms = _context_terms(SPIRIT_CONTEXT)
    assert allowed_terms, "No distilled spirits class/type terms loaded."
    _assert_class_type(crystal_springs_extraction.class_type.value, allowed_terms)


def test_class_type_extraction_malt_beverage() -> None:
    if not MALT_FIXTURE.exists():
        pytest.skip(f"Fixture directory missing: {MALT_FIXTURE}")
    data = _load_fixture_data(MALT_FIXTURE)
    fields = data.get("fields", {})
    if not isinstance(fields, dict):
        raise AssertionError("Fixture has invalid fields payload.")
    _assert_fixture_type(fields, "malt_beverage")

    images = _open_images(MALT_FIXTURE, data)
    if not images:
        pytest.skip(f"Fixture {MALT_FIXTURE.name} has no readable images.")

    try:
        label_info = extract_label_info_from_application_images(images)
    finally:
        _close_images(images)

    allowed_terms = _context_terms(MALT_CONTEXT)
    assert allowed_terms, "No malt beverage class/type terms loaded."
    _assert_class_type(label_info.class_type.value, allowed_terms)
