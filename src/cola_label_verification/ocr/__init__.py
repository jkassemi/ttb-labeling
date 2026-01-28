import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from PIL import Image

from cola_label_verification.ocr.clients import _get_default_ocr_client
from cola_label_verification.ocr.lines import _extract_text_lines_and_spans
from cola_label_verification.ocr.types import DEFAULT_OCR_OPTIONS, OcrSpan
from cola_label_verification.models import (
    BeverageTypeClassification,
    FieldCandidate,
    FieldExtraction,
    LabelInfo,
    QwenFieldValue,
    TokenVerification,
)
from cola_label_verification.rules import field_of_vision
from cola_label_verification.rules.beverage_type_presence import (
    beverage_type_from_qwen,
)
from cola_label_verification.rules.warning_text_helpers import attach_warning_header
from cola_label_verification.vlm import extract_qwen_field_values

_LABEL_FIELDS = tuple(
    name
    for name, field in LabelInfo.model_fields.items()
    if field.annotation is FieldExtraction
)
_VERIFICATION_TOKEN_RE = re.compile(r"[A-Z0-9]+")


@dataclass(frozen=True)
class OcrExtractionResult:
    """Structured label fields plus the OCR spans used for verification."""

    label_info: LabelInfo
    spans: Sequence[OcrSpan]


def _resolve_field(
    candidate: FieldCandidate | None,
) -> FieldExtraction:
    if candidate:
        status = _verification_status(candidate)
        return FieldExtraction(
            value=candidate.value,
            confidence=candidate.confidence,
            evidence=candidate.evidence,
            source="qwen",
            status=status,
            normalized=candidate.normalized,
            numeric_value=candidate.numeric_value,
            unit=candidate.unit,
        )
    return FieldExtraction(
        value=None,
        confidence=None,
        evidence=None,
        source="qwen",
        status="missing",
        normalized=None,
        numeric_value=None,
        unit=None,
    )


def _verification_status(candidate: FieldCandidate) -> str:
    verification = (candidate.normalized or {}).get("verification")
    token_count: int | None = None
    matched_token_count: int | None = None
    if isinstance(verification, TokenVerification):
        token_count = verification.token_count
        matched_token_count = verification.matched_token_count
    elif isinstance(verification, Mapping):
        token_count = verification.get("token_count")
        matched_token_count = verification.get("matched_token_count")
    else:
        return "needs_review"
    if (
        isinstance(token_count, int)
        and isinstance(matched_token_count, int)
        and token_count > 0
        and matched_token_count == token_count
    ):
        return "verified"
    return "needs_review"


def _tokenize_for_verification(text: str) -> list[str]:
    """Tokenize text for span-based verification in build_field_candidates."""
    tokens = _VERIFICATION_TOKEN_RE.findall(text.upper())
    return [token for token in tokens if len(token) > 1 or token.isdigit()]


def _candidate_from_qwen(
    field: str,
    qwen_fields: Mapping[str, QwenFieldValue | None],
) -> FieldCandidate | None:
    """Build a candidate from Qwen output for build_field_candidates."""
    field_value = qwen_fields.get(field)
    if field_value is None:
        return None
    cleaned = (field_value.text or "").strip()
    if not cleaned:
        return None
    normalized = {
        "source": "qwen_vl",
        "qwen_field": field,
    }
    return FieldCandidate(
        value=cleaned,
        confidence=None,
        evidence=cleaned,
        normalized=normalized,
        numeric_value=field_value.numeric_value,
        unit=field_value.unit,
    )


def _candidate_image_index(candidate: FieldCandidate) -> int | None:
    """Extract the image index from candidate metadata for build_field_candidates."""
    normalized = candidate.normalized or {}
    span_index = normalized.get("image_index")
    if isinstance(span_index, int):
        return span_index
    return None


def _span_token_set(
    spans: Sequence[OcrSpan],
    *,
    image_index: int | None = None,
) -> set[str]:
    """Collect verification tokens from spans for build_field_candidates."""
    tokens: set[str] = set()
    for span in spans:
        if image_index is not None and span.image_index != image_index:
            continue
        tokens.update(_tokenize_for_verification(span.text))
    return tokens


def _best_span_for_tokens(
    tokens: Sequence[str],
    spans: Sequence[OcrSpan],
    *,
    image_index: int | None = None,
) -> OcrSpan | None:
    """Select the best matching span for build_field_candidates."""
    if not tokens:
        return None
    token_set = set(tokens)
    best: OcrSpan | None = None
    best_score = 0.0
    for span in spans:
        if image_index is not None and span.image_index != image_index:
            continue
        span_tokens = set(_tokenize_for_verification(span.text))
        if not span_tokens:
            continue
        overlap = len(token_set & span_tokens)
        if overlap == 0:
            continue
        score = overlap / len(token_set)
        if score > best_score:
            best_score = score
            best = span
    return best


def _verify_tokens_with_spans(
    tokens: Sequence[str],
    spans: Sequence[OcrSpan],
    *,
    image_index: int | None = None,
) -> TokenVerification:
    """Compute token coverage against spans for build_field_candidates."""
    if not tokens:
        return TokenVerification(
            matched=False,
            coverage=0.0,
            token_count=0,
            matched_token_count=0,
            source="span",
        )
    span_tokens = _span_token_set(spans, image_index=image_index)
    matched = [token for token in tokens if token in span_tokens]
    coverage = len(matched) / len(tokens)
    return TokenVerification(
        matched=bool(matched),
        coverage=round(coverage, 3),
        token_count=len(tokens),
        matched_token_count=len(matched),
        source="span",
    )


def _attach_span_verification(
    candidate: FieldCandidate | None,
    spans: Sequence[OcrSpan],
) -> FieldCandidate | None:
    """Attach span verification and location metadata for build_field_candidates."""
    if candidate is None:
        return None
    tokens = _tokenize_for_verification(candidate.value)
    image_index = _candidate_image_index(candidate)
    span = _best_span_for_tokens(tokens, spans, image_index=image_index)
    verification = _verify_tokens_with_spans(
        tokens,
        spans,
        image_index=image_index,
    )
    normalized: dict[str, object] = {}
    if candidate.normalized:
        normalized.update(candidate.normalized)
    normalized["verification"] = verification
    if span is not None:
        normalized["bbox"] = span.bbox
        normalized["image_index"] = span.image_index
    confidence = candidate.confidence
    token_count = verification.token_count
    matched_token_count = verification.matched_token_count
    if (
        confidence is None
        and isinstance(token_count, int)
        and isinstance(matched_token_count, int)
        and token_count > 0
    ):
        confidence = round(matched_token_count / token_count, 3)
    return FieldCandidate(
        value=candidate.value,
        confidence=confidence,
        evidence=candidate.evidence,
        normalized=normalized,
        numeric_value=candidate.numeric_value,
        unit=candidate.unit,
    )


def build_field_candidates(
    spans: Sequence[OcrSpan],
    *,
    qwen_fields: Mapping[str, QwenFieldValue | None] | None = None,
) -> dict[str, FieldCandidate | None]:
    """Assemble per-field candidates using Qwen outputs and OCR spans.

    Used by the OCR pipeline to attach span-based verification and warning header
    metadata before resolving fields into `LabelInfo`.
    """
    qwen_fields = qwen_fields or {}
    candidates: dict[str, FieldCandidate | None] = {}
    for field_name in _LABEL_FIELDS:
        candidate = _candidate_from_qwen(field_name, qwen_fields)
        candidate = _attach_span_verification(candidate, spans)
        if field_name == "warning_text":
            candidate = attach_warning_header(candidate, spans)
        candidates[field_name] = candidate
    return candidates


def _field_of_vision_metadata(
    candidates: Mapping[str, FieldCandidate | None],
    image_sizes: Sequence[tuple[int, int]],
) -> dict[str, object] | None:
    return field_of_vision._field_of_vision_metadata(candidates, image_sizes)


def _apply_field_of_vision(
    label_info: LabelInfo,
    candidates: Mapping[str, FieldCandidate | None],
    image_sizes: Sequence[tuple[int, int]],
) -> LabelInfo:
    return field_of_vision._apply_field_of_vision(label_info, candidates, image_sizes)


def _resolve_fields(
    candidates: Mapping[str, FieldCandidate | None],
    *,
    beverage_type: BeverageTypeClassification | None = None,
) -> LabelInfo:
    resolved: dict[str, FieldExtraction] = {}
    for field_name in _LABEL_FIELDS:
        resolved[field_name] = _resolve_field(candidates.get(field_name))
    return LabelInfo(**resolved, beverage_type=beverage_type)


def extract_label_info_from_application_images(
    images: Sequence[Image.Image],
    *,
    ocr_client: object | None = None,
) -> LabelInfo:
    """Extract label fields from images associated with one application.

    Args:
        images: Application images to scan (front/back, neck, etc.).
        ocr_client: Optional OCR backend implementation for dependency injection.
    Returns:
        LabelInfo with structured fields.
    """
    return extract_label_info_with_spans(
        images,
        ocr_client=ocr_client,
    ).label_info


def extract_label_info_with_spans(
    images: Sequence[Image.Image],
    *,
    ocr_client: object | None = None,
) -> OcrExtractionResult:
    """Extract label fields and OCR spans from images associated with one application.

    Args:
        images: Application images to scan (front/back, neck, etc.).
        ocr_client: Optional OCR backend implementation for dependency injection.
    Returns:
        Structured label fields plus the OCR spans used for verification.
    """
    assert images, "No images provided for OCR extraction."

    resolved_client = ocr_client or _get_default_ocr_client()

    qwen_result = extract_qwen_field_values(images)

    paddle_spans: list[OcrSpan] = []
    _, paddle_spans = _extract_text_lines_and_spans(
        images,
        resolved_client,
        DEFAULT_OCR_OPTIONS,
    )

    beverage_prediction = beverage_type_from_qwen(qwen_result.beverage_type)
    combined_spans = paddle_spans

    image_sizes = [image.size for image in images]
    if not combined_spans:
        _, enhanced_spans = _extract_text_lines_and_spans(
            images,
            resolved_client,
            DEFAULT_OCR_OPTIONS,
            enhance_images=True,
            geometry_safe=True,
        )
        combined_spans = enhanced_spans

    if not combined_spans:
        raise RuntimeError("OCR completed but no text was detected.")
    candidates_by_field = build_field_candidates(
        combined_spans,
        qwen_fields=qwen_result.fields,
    )
    label_info = _resolve_fields(
        candidates_by_field,
        beverage_type=beverage_prediction,
    )
    label_info = _apply_field_of_vision(label_info, candidates_by_field, image_sizes)

    return OcrExtractionResult(label_info=label_info, spans=combined_spans)
