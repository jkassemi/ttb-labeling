from PIL import Image

import cola_label_verification.rules.warning_text as warning_text_module
from cola_label_verification.models import FieldExtraction, LabelInfo
from cola_label_verification.ocr.types import OcrSpan
from cola_label_verification.rules.models import RuleContext
from cola_label_verification.rules.warning_text import (
    BoldnessMetrics,
    CANONICAL_WARNING_TEXT,
    estimate_boldness,
    warning_text,
)


def _context_with_warning_text(
    value: str | None,
    *,
    normalized: dict[str, object] | None = None,
    images: list[Image.Image] | None = None,
) -> RuleContext:
    warning = FieldExtraction(value=value, normalized=normalized)
    label_info = LabelInfo(warning_text=warning)
    return RuleContext(
        label_info=label_info,
        application_fields=None,
        images=images,
    )


def test_warning_text_missing_value_fails() -> None:
    context = _context_with_warning_text(None)
    finding = warning_text(context)

    assert finding.status == "fail"
    assert finding.field == "warning_text"
    assert "not detected" in finding.message


def test_warning_text_header_missing_needs_review() -> None:
    text = CANONICAL_WARNING_TEXT.replace("GOVERNMENT WARNING: ", "")
    context = _context_with_warning_text(text)
    finding = warning_text(context)

    assert finding.status == "needs_review"
    assert finding.severity == "warning"
    assert "header text may be incomplete" in finding.message
    assert finding.evidence is not None
    assert finding.evidence.get("warning_header_present") is False


def test_warning_text_exact_but_boldness_unavailable() -> None:
    context = _context_with_warning_text(CANONICAL_WARNING_TEXT)
    finding = warning_text(context)

    assert finding.status == "needs_review"
    assert finding.severity == "info"
    assert finding.message.startswith(
        "Warning statement text matches the required wording"
    )
    assert finding.evidence is not None
    assert finding.evidence.get("status") == "needs_review"
    assert finding.evidence.get("reason") == "boldness_unavailable"


def test_warning_text_passes_with_boldness_pass() -> None:
    normalized = {"warning_boldness": {"status": "pass", "score": 1.2}}
    context = _context_with_warning_text(
        CANONICAL_WARNING_TEXT,
        normalized=normalized,
    )
    finding = warning_text(context)

    assert finding.status == "pass"
    assert finding.severity == "info"
    assert finding.evidence is not None
    assert finding.evidence.get("status") == "pass"
    assert finding.evidence.get("warning_text_exact_match") is True


def test_warning_text_uppercase_mismatch_flags_wording() -> None:
    text = CANONICAL_WARNING_TEXT.upper().replace(
        "THE SURGEON GENERAL",
        "SURGEON GENERAL",
    )
    normalized = {"warning_boldness": {"status": "pass"}}
    context = _context_with_warning_text(text, normalized=normalized)
    finding = warning_text(context)

    assert finding.status == "needs_review"
    assert finding.severity == "warning"
    assert "wording does not exactly match" in finding.message
    assert finding.evidence is not None
    assert finding.evidence.get("warning_text_exact_match") is False


def test_estimate_boldness_fallback_without_peers(monkeypatch) -> None:
    metrics = BoldnessMetrics(
        foreground_ratio=0.05,
        edge_ratio=0.1,
        stroke_ratio=4.0,
        contrast=0.2,
    )
    monkeypatch.setattr(
        warning_text_module,
        "_measure_metrics",
        lambda _image: metrics,
    )

    image = Image.new("RGB", (10, 10), "white")
    result = estimate_boldness(image, (0, 0, 10, 10), [])

    assert result is not None
    assert result.get("status") == "pass"
    assert result.get("reason") == "no_peer_spans"
    assert result.get("header_metrics") == {
        "foreground_ratio": 0.05,
        "edge_ratio": 0.1,
        "stroke_ratio": 4.0,
        "contrast": 0.2,
    }


def test_estimate_boldness_invalid_peer_metrics(monkeypatch) -> None:
    metrics_queue = [
        BoldnessMetrics(
            foreground_ratio=0.1,
            edge_ratio=0.1,
            stroke_ratio=2.0,
            contrast=0.2,
        ),
        BoldnessMetrics(
            foreground_ratio=0.0,
            edge_ratio=0.1,
            stroke_ratio=0.0,
            contrast=0.2,
        ),
        BoldnessMetrics(
            foreground_ratio=0.0,
            edge_ratio=0.1,
            stroke_ratio=0.0,
            contrast=0.2,
        ),
    ]

    def _fake_measure(_image: Image.Image) -> BoldnessMetrics:
        return metrics_queue.pop(0)

    monkeypatch.setattr(warning_text_module, "_measure_metrics", _fake_measure)

    image = Image.new("RGB", (10, 10), "white")
    spans = [
        OcrSpan(text="peer-1", confidence=None, bbox=(0, 0, 2, 2), image_index=0),
        OcrSpan(text="peer-2", confidence=None, bbox=(2, 2, 4, 4), image_index=0),
    ]
    result = estimate_boldness(image, (0, 0, 4, 4), spans)

    assert result is not None
    assert result.get("status") == "needs_review"
    assert result.get("reason") == "invalid_peer_metrics"
    assert result.get("header_metrics") == {
        "foreground_ratio": 0.1,
        "edge_ratio": 0.1,
        "stroke_ratio": 2.0,
        "contrast": 0.2,
    }
