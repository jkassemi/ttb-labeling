from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Final, cast

from PIL import Image, ImageFilter, ImageOps

from cola_label_verification.ocr.types import OcrSpan
from cola_label_verification.rules.common import build_finding
from cola_label_verification.rules.models import Finding, RuleContext

CANONICAL_WARNING_TEXT: Final[str] = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women "
    "should not drink alcoholic beverages during pregnancy because of the risk of "
    "birth defects. (2) Consumption of alcoholic beverages impairs your ability to "
    "drive a car or operate machinery, and may cause health problems."
)


@dataclass(frozen=True)
class BoldnessMetrics:
    foreground_ratio: float
    edge_ratio: float
    stroke_ratio: float
    contrast: float


def warning_text(context: RuleContext) -> Finding:
    """Evaluate government warning text presence, wording, and formatting."""
    label_info = context.label_info
    value = label_info.warning_text.value
    if not value:
        return build_finding(
            "warning_text",
            "fail",
            "Government warning statement not detected.",
            field="warning_text",
        )

    normalized_text = _normalize_warning_text(value)
    header_present = "GOVERNMENT WARNING" in value.upper()
    exact_match = _matches_canonical_warning(
        value
    ) or _matches_canonical_warning_all_caps(value)

    normalized_fields = label_info.warning_text.normalized or {}
    boldness = normalized_fields.get("warning_boldness")
    if not isinstance(boldness, dict):
        boldness = _compute_warning_boldness(context, normalized_fields)

    boldness_evidence = _boldness_evidence(boldness)
    evidence = {
        **boldness_evidence,
        "warning_text_normalized": normalized_text,
        "warning_header_present": header_present,
        "warning_text_exact_match": exact_match,
    }

    issues: list[str] = []
    has_text_issue = False

    if not header_present:
        issues.append("header text may be incomplete")
        has_text_issue = True
    else:
        exactness_issue = _exactness_issue(normalized_text, exact_match)
        if exactness_issue:
            issues.append(exactness_issue)
            has_text_issue = True

    if header_present:
        boldness_issue = _boldness_issue(boldness_evidence)
        if boldness_issue:
            issues.append(boldness_issue)

    if issues:
        if not has_text_issue and issues == ["header boldness could not be confirmed"]:
            message = (
                "Warning statement text matches the required wording, but "
                "header boldness could not be confirmed."
            )
        else:
            message = "Warning statement detected, but " + "; ".join(issues) + "."
        severity = "warning" if has_text_issue else "info"
        return build_finding(
            "warning_text",
            "needs_review",
            message,
            field="warning_text",
            severity=severity,
            evidence=evidence,
        )

    return build_finding(
        "warning_text",
        "pass",
        "Warning statement matches the required text and header appears bold.",
        field="warning_text",
        severity="info",
        evidence=evidence,
    )


def _normalize_warning_text(text: str) -> str:
    """Normalize warning text for strict comparison."""
    return " ".join(text.split())


def _matches_canonical_warning(text: str) -> bool:
    """Return True when the warning text matches the canonical statement."""
    return _normalize_warning_text(text) == CANONICAL_WARNING_TEXT


def _matches_canonical_warning_all_caps(text: str) -> bool:
    """Return True when the warning text matches the canonical statement in all caps."""
    normalized = _normalize_warning_text(text)
    if not normalized.isupper():
        return False
    return normalized == CANONICAL_WARNING_TEXT.upper()


def _exactness_issue(normalized_text: str, exact_match: bool) -> str | None:
    if exact_match:
        return None
    if "GOVERNMENT WARNING:" not in normalized_text:
        return "header text may not be correctly formatted"
    if "Surgeon General" not in normalized_text:
        if normalized_text.isupper() and "SURGEON GENERAL" in normalized_text:
            return "wording does not exactly match the required statement"
        return "capitalization for Surgeon General may be incorrect"
    return "wording does not exactly match the required statement"


def _boldness_evidence(boldness: dict[str, object] | None) -> dict[str, object]:
    if isinstance(boldness, dict):
        return dict(boldness)
    return {"status": "needs_review", "reason": "boldness_unavailable"}


def _boldness_issue(boldness_evidence: dict[str, object]) -> str | None:
    if boldness_evidence.get("status") == "pass":
        return None
    return "header boldness could not be confirmed"


def _compute_warning_boldness(
    context: RuleContext,
    normalized: dict[str, object],
) -> dict[str, object] | None:
    images = context.images
    if not images:
        return None
    header_index = normalized.get("warning_header_image_index")
    if not isinstance(header_index, int):
        return None
    if header_index < 0 or header_index >= len(images):
        return None
    header_bbox = _warning_header_bbox(normalized)
    if header_bbox is None:
        return None
    boldness = estimate_boldness(images[header_index], header_bbox, [])
    if isinstance(boldness, dict) and boldness.get("reason") == "no_peer_spans":
        boldness = {**boldness, "status": "needs_review"}
    return boldness


def _warning_header_bbox(
    normalized: dict[str, object],
) -> tuple[float, float, float, float] | None:
    raw_bbox = normalized.get("warning_header_bbox")
    if not isinstance(raw_bbox, Sequence) or len(raw_bbox) != 4:
        return None
    try:
        left, top, right, bottom = (float(item) for item in raw_bbox)
    except (TypeError, ValueError):
        return None
    return (left, top, right, bottom)


def estimate_boldness(
    image: Image.Image,
    header_bbox: tuple[float, float, float, float],
    peer_spans: Sequence[OcrSpan],
) -> dict[str, object] | None:
    header_metrics = _measure_metrics(_crop_with_padding(image, header_bbox))
    if header_metrics is None:
        return None
    peer_metrics = [
        _measure_metrics(_crop_with_padding(image, span.bbox)) for span in peer_spans
    ]
    peer_metrics = [metric for metric in peer_metrics if metric is not None]
    if not peer_metrics:
        return _fallback_without_peers(header_metrics)
    median_fg = median(metric.foreground_ratio for metric in peer_metrics)
    median_stroke = median(metric.stroke_ratio for metric in peer_metrics)
    if median_fg <= 0 or median_stroke <= 0:
        return {
            "status": "needs_review",
            "reason": "invalid_peer_metrics",
            "header_metrics": _metrics_payload(header_metrics),
        }
    score = (
        header_metrics.foreground_ratio / median_fg
        + header_metrics.stroke_ratio / median_stroke
    ) / 2
    status = "pass" if score >= 1.1 else "needs_review"
    return {
        "status": status,
        "score": round(score, 3),
        "header_metrics": _metrics_payload(header_metrics),
        "peer_median": {
            "foreground_ratio": round(median_fg, 4),
            "stroke_ratio": round(median_stroke, 4),
        },
        "peer_count": len(peer_metrics),
    }


def _crop_with_padding(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    *,
    padding_ratio: float = 0.1,
) -> Image.Image:
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    crop = (
        max(0, int(left - pad_x)),
        max(0, int(top - pad_y)),
        min(image.width, int(right + pad_x)),
        min(image.height, int(bottom + pad_y)),
    )
    return image.crop(crop)


def _measure_metrics(image: Image.Image) -> BoldnessMetrics | None:
    if image.width == 0 or image.height == 0:
        return None
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    contrast = _contrast_ratio(gray)
    if contrast < 0.05:
        return None
    threshold = _otsu_threshold(gray)
    pixels = _pixel_values(gray)
    total = len(pixels)
    if total == 0:
        return None
    below_mask = [value <= threshold for value in pixels]
    below = sum(1 for is_foreground in below_mask if is_foreground)
    above = total - below
    if below == 0 or above == 0:
        return None
    if below <= above:
        foreground_mask = below_mask
        foreground_pixels = below
    else:
        foreground_mask = [not value for value in below_mask]
        foreground_pixels = above
    foreground_ratio = foreground_pixels / total
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_pixels = _pixel_values(edges)
    edge_mean = sum(edge_pixels) / total
    edge_threshold = max(10.0, edge_mean * 1.5)
    edge_in_foreground = sum(
        1
        for edge_value, is_foreground in zip(edge_pixels, foreground_mask, strict=False)
        if is_foreground and edge_value > edge_threshold
    )
    edge_ratio = edge_in_foreground / total
    stroke_ratio = (
        foreground_pixels / edge_in_foreground if edge_in_foreground > 0 else 0.0
    )
    return BoldnessMetrics(
        foreground_ratio=foreground_ratio,
        edge_ratio=edge_ratio,
        stroke_ratio=stroke_ratio,
        contrast=contrast,
    )


def _contrast_ratio(image: Image.Image) -> float:
    pixels = sorted(_pixel_values(image))
    if not pixels:
        return 0.0
    p5 = pixels[int(len(pixels) * 0.05)]
    p95 = pixels[int(len(pixels) * 0.95)]
    return (p95 - p5) / 255.0


def _otsu_threshold(image: Image.Image) -> int:
    hist = image.histogram()
    total = sum(hist)
    sum_total = sum(index * count for index, count in enumerate(hist))
    sum_background = 0
    weight_background = 0
    max_variance = 0.0
    threshold = 128
    for index, count in enumerate(hist):
        weight_background += count
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += index * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )
        if variance > max_variance:
            max_variance = variance
            threshold = index
    return threshold


def _pixel_values(image: Image.Image) -> list[int]:
    return list(cast(Iterable[int], image.getdata()))


def _metrics_payload(metrics: BoldnessMetrics) -> dict[str, float]:
    return {
        "foreground_ratio": round(metrics.foreground_ratio, 4),
        "edge_ratio": round(metrics.edge_ratio, 4),
        "stroke_ratio": round(metrics.stroke_ratio, 4),
        "contrast": round(metrics.contrast, 4),
    }


def _fallback_without_peers(metrics: BoldnessMetrics) -> dict[str, object]:
    status = "needs_review"
    if (
        metrics.contrast >= 0.15
        and metrics.stroke_ratio >= 3
        and metrics.foreground_ratio >= 0.02
    ):
        status = "pass"
    return {
        "status": status,
        "reason": "no_peer_spans",
        "header_metrics": _metrics_payload(metrics),
    }
