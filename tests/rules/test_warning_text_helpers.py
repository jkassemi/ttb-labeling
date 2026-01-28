import pytest

from cola_label_verification.models import FieldCandidate
from cola_label_verification.ocr.types import OcrSpan
from cola_label_verification.rules.warning_text_helpers import (
    attach_warning_header,
    looks_like_warning_text,
)


def _candidate(
    value: str,
    *,
    normalized: dict[str, object] | None = None,
) -> FieldCandidate:
    return FieldCandidate(
        value=value,
        confidence=0.5,
        evidence="test",
        normalized=normalized,
        numeric_value=None,
        unit=None,
    )


def _span(
    text: str,
    bbox: tuple[float, float, float, float],
    *,
    image_index: int = 0,
) -> OcrSpan:
    return OcrSpan(
        text=text,
        confidence=0.9,
        bbox=bbox,
        image_index=image_index,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Government Warning", True),
        ("surgeon general", True),
        ("(1) and (2)", True),
        ("nothing to see here", False),
    ],
)
def test_looks_like_warning_text_detects_tokens(text: str, expected: bool) -> None:
    assert looks_like_warning_text(text) is expected


def test_attach_warning_header_returns_none_for_missing_candidate() -> None:
    assert attach_warning_header(None, []) is None


def test_attach_warning_header_prefers_span_with_both_tokens() -> None:
    spans = [
        _span("GOVERNMENT WARNING", (10, 10, 50, 20), image_index=2),
        _span("WARNING", (100, 100, 140, 120), image_index=2),
    ]
    candidate = _candidate("warning text", normalized={"source": "qwen"})
    updated = attach_warning_header(candidate, spans)
    assert updated is not None
    assert updated.normalized is not None
    assert updated.normalized["source"] == "qwen"
    assert updated.normalized["warning_header_bbox"] == (10, 10, 50, 20)
    assert updated.normalized["warning_header_image_index"] == 2


def test_attach_warning_header_selects_closest_pair() -> None:
    spans = [
        _span("GOVERNMENT", (0, 0, 10, 10), image_index=0),
        _span("WARNING", (12, 0, 22, 10), image_index=0),
        _span("WARNING", (200, 200, 210, 210), image_index=0),
    ]
    updated = attach_warning_header(_candidate("warning text"), spans)
    assert updated is not None
    assert updated.normalized is not None
    assert updated.normalized["warning_header_bbox"] == (0, 0, 22, 10)
    assert updated.normalized["warning_header_image_index"] == 0


def test_attach_warning_header_falls_back_to_first_match() -> None:
    spans = [
        _span("WARNING", (5, 5, 25, 15), image_index=1),
        _span("GOVERNMENT", (0, 0, 20, 10), image_index=0),
    ]
    updated = attach_warning_header(_candidate("warning text"), spans)
    assert updated is not None
    assert updated.normalized is not None
    assert updated.normalized["warning_header_bbox"] == (5, 5, 25, 15)
    assert updated.normalized["warning_header_image_index"] == 1


def test_attach_warning_header_uses_warning_text_when_no_header_tokens() -> None:
    spans = [
        _span("SURGEON", (0, 10, 30, 30), image_index=1),
        _span("GENERAL", (40, 12, 80, 32), image_index=1),
        _span("PREGNANCY", (0, 200, 60, 220), image_index=1),
        _span("SURGEON", (0, 50, 30, 70), image_index=0),
    ]
    candidate = _candidate(
        "Surgeon General warning: Drinking alcoholic beverages during pregnancy",
    )
    updated = attach_warning_header(candidate, spans)
    assert updated is not None
    assert updated.normalized is not None
    assert updated.normalized["warning_header_bbox"] == (0, 10, 80, 32)
    assert updated.normalized["warning_header_image_index"] == 1


def test_attach_warning_header_returns_candidate_when_header_unresolvable() -> None:
    spans = [
        _span("SURGEON", (0, 10, 30, 10), image_index=0),
        _span("GENERAL", (40, 12, 80, 12), image_index=0),
    ]
    candidate = _candidate("Surgeon General warning", normalized={"source": "qwen"})
    updated = attach_warning_header(candidate, spans)
    assert updated is not None
    assert updated.normalized == {"source": "qwen"}


def test_attach_warning_header_no_spans_returns_candidate() -> None:
    candidate = _candidate("warning text", normalized={"source": "qwen"})
    updated = attach_warning_header(candidate, [])
    assert updated is not None
    assert updated.normalized == {"source": "qwen"}
