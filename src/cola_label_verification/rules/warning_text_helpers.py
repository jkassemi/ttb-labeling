import logging
from collections.abc import Sequence

from cola_label_verification.models import FieldCandidate
from cola_label_verification.ocr.types import OcrSpan
from cola_label_verification.text import normalize_for_match

logger = logging.getLogger(__name__)

WARNING_HEADER_TOKENS = ("GOVERNMENT", "WARNING")
WARNING_BODY_TOKENS = (
    "PREGNANCY",
    "SURGEON",
    "BIRTH",
    "DEFECTS",
    "ALCOHOLIC",
    "BEVERAGES",
    "IMPAIRS",
    "MACHINERY",
    "HEALTH",
)


def looks_like_warning_text(text: str) -> bool:
    upper = text.upper()
    if any(token in upper for token in WARNING_HEADER_TOKENS):
        return True
    if any(token in upper for token in WARNING_BODY_TOKENS):
        return True
    return "(1)" in upper or "(2)" in upper


def attach_warning_header(
    candidate: FieldCandidate | None,
    spans: Sequence[OcrSpan],
) -> FieldCandidate | None:
    if candidate is None:
        return None
    header = _find_warning_header_bbox(spans)
    if header is None and candidate.value:
        header = _find_warning_header_from_text(spans, candidate.value)
    if header is None:
        if spans:
            token_spans = [
                span
                for span in spans
                if any(
                    token in span.text.upper()
                    for token in ("GOVERNMENT", "WARNING", "WARN")
                )
            ]
            sample_spans = token_spans[:10] if token_spans else spans[:10]
            sample_payload = [
                {
                    "text": span.text,
                    "bbox": span.bbox,
                    "image_index": span.image_index,
                }
                for span in sample_spans
            ]
            logger.debug(
                "Warning header bbox not found; spans=%d token_spans=%d sample=%s",
                len(spans),
                len(token_spans),
                sample_payload,
            )
        return candidate
    bbox, image_index = header
    normalized: dict[str, object] = {}
    if candidate.normalized:
        normalized.update(candidate.normalized)
    normalized["warning_header_bbox"] = bbox
    normalized["warning_header_image_index"] = image_index
    return FieldCandidate(
        value=candidate.value,
        confidence=candidate.confidence,
        evidence=candidate.evidence,
        normalized=normalized,
        numeric_value=candidate.numeric_value,
        unit=candidate.unit,
    )


def _find_warning_header_bbox(
    spans: Sequence[OcrSpan],
) -> tuple[tuple[float, float, float, float], int] | None:
    matches: list[OcrSpan] = []
    for span in spans:
        upper = span.text.upper()
        if "GOVERNMENT" in upper or "WARNING" in upper:
            matches.append(span)
        if "GOVERNMENT" in upper and "WARNING" in upper:
            return span.bbox, span.image_index
    if not matches:
        return None
    gov_spans = [span for span in matches if "GOVERNMENT" in span.text.upper()]
    warn_spans = [span for span in matches if "WARNING" in span.text.upper()]
    best_score = float("inf")
    best_bbox: tuple[float, float, float, float] | None = None
    best_index: int | None = None
    for gov in gov_spans:
        for warn in warn_spans:
            if gov.image_index != warn.image_index:
                continue
            score = _pair_span_score(gov, warn)
            if score < best_score:
                best_score = score
                best_bbox = _union_bbox(gov.bbox, warn.bbox)
                best_index = gov.image_index
    if best_bbox is not None and best_index is not None:
        return best_bbox, best_index
    for span in matches:
        return span.bbox, span.image_index
    return None


def _find_warning_header_from_text(
    spans: Sequence[OcrSpan],
    warning_text: str,
) -> tuple[tuple[float, float, float, float], int] | None:
    target = normalize_for_match(warning_text)
    if not target:
        return None
    matches: list[OcrSpan] = []
    for span in spans:
        span_norm = normalize_for_match(span.text)
        if not span_norm:
            continue
        if span_norm in target:
            matches.append(span)
    if not matches:
        return None
    matches_by_image: dict[int, list[OcrSpan]] = {}
    for span in matches:
        matches_by_image.setdefault(span.image_index, []).append(span)
    image_index, image_spans = max(
        matches_by_image.items(),
        key=lambda item: len(item[1]),
    )
    min_y = min(span.bbox[1] for span in image_spans)
    heights = [
        span.bbox[3] - span.bbox[1]
        for span in image_spans
        if span.bbox[3] > span.bbox[1]
    ]
    if not heights:
        return None
    median_height = sorted(heights)[len(heights) // 2]
    band = median_height * 1.6
    header_spans = [span for span in image_spans if span.bbox[1] <= min_y + band]
    if not header_spans:
        return None
    bbox = header_spans[0].bbox
    for span in header_spans[1:]:
        bbox = _union_bbox(bbox, span.bbox)
    return bbox, image_index


def _pair_span_score(span_a: OcrSpan, span_b: OcrSpan) -> float:
    center_a = (
        (span_a.bbox[0] + span_a.bbox[2]) / 2,
        (span_a.bbox[1] + span_a.bbox[3]) / 2,
    )
    center_b = (
        (span_b.bbox[0] + span_b.bbox[2]) / 2,
        (span_b.bbox[1] + span_b.bbox[3]) / 2,
    )
    width = (span_a.bbox[2] - span_a.bbox[0] + span_b.bbox[2] - span_b.bbox[0]) / 2
    height = (span_a.bbox[3] - span_a.bbox[1] + span_b.bbox[3] - span_b.bbox[1]) / 2
    width = max(width, 1.0)
    height = max(height, 1.0)
    return (
        abs(center_a[0] - center_b[0]) / width + abs(center_a[1] - center_b[1]) / height
    )


def _union_bbox(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (
        min(bbox_a[0], bbox_b[0]),
        min(bbox_a[1], bbox_b[1]),
        max(bbox_a[2], bbox_b[2]),
        max(bbox_a[3], bbox_b[3]),
    )
