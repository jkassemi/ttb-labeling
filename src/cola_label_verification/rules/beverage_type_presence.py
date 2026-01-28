from collections.abc import Sequence
from typing import Final

from cola_label_verification import taxonomy
from cola_label_verification.ocr.types import OcrSpan
from cola_label_verification.models import BeverageTypeClassification
from cola_label_verification.rules.common import build_finding
from cola_label_verification.rules.models import Finding, RuleContext

_WINE_KEYWORDS: Final = (
    "WINE",
    "RED WINE",
    "WHITE WINE",
    "SPARKLING",
    "ROSE",
    "CABERNET",
    "SAUVIGNON",
    "MERLOT",
    "PINOT",
    "CHARDONNAY",
    "RIESLING",
    "MALBEC",
    "SYRAH",
    "SHIRAZ",
    "ZINFANDEL",
    "TEMPRANILLO",
    "SANGIOVESE",
    "PETIT VERDOT",
)
_MIN_CLASSIFY_SCORE: Final = 1.2
_MIN_AUTO_CONFIDENCE: Final = 0.6


def classify_beverage_type(
    text_blocks: Sequence[str],
) -> BeverageTypeClassification | None:
    """Classify beverage type using extracted text."""
    if not text_blocks:
        return None
    wine_score, wine_hits = _score_keywords(
        text_blocks,
        _WINE_KEYWORDS,
        weight=1.5,
    )
    spirits_score, spirits_hits = _score_keywords(
        text_blocks,
        tuple(keyword.upper() for keyword in taxonomy.SPIRITS_CLASS_KEYWORDS),
        weight=1.2,
    )

    if wine_score < _MIN_CLASSIFY_SCORE and spirits_score < _MIN_CLASSIFY_SCORE:
        return None

    if wine_score >= spirits_score:
        beverage_type = "wine"
        best_score = wine_score
        evidence = {
            "type": "wine",
            "matched_terms": wine_hits,
        }
    else:
        beverage_type = "distilled_spirits"
        best_score = spirits_score
        evidence = {
            "type": "distilled_spirits",
            "matched_terms": spirits_hits,
        }

    total = wine_score + spirits_score
    confidence = best_score / total if total else 0.0
    return BeverageTypeClassification(
        beverage_type=beverage_type,
        confidence=confidence,
        evidence=evidence,
    )


def should_auto_apply_classification(
    prediction: BeverageTypeClassification | None,
) -> bool:
    if prediction is None:
        return False
    return prediction.confidence >= _MIN_AUTO_CONFIDENCE


def _score_keywords(
    text_blocks: Sequence[str],
    keywords: tuple[str, ...],
    *,
    weight: float,
) -> tuple[float, list[str]]:
    score = 0.0
    hits: list[str] = []
    for text in text_blocks:
        if not text:
            continue
        upper = text.upper()
        for keyword in keywords:
            if keyword in upper:
                score += weight
                if keyword not in hits:
                    hits.append(keyword)
    return score, hits


def serialize_prediction(
    prediction: BeverageTypeClassification | None,
) -> dict[str, object] | None:
    if prediction is None:
        return None
    return prediction.model_dump()


def beverage_type_from_qwen(
    value: str | None,
) -> BeverageTypeClassification | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    normalized = normalized.replace("-", " ").replace("_", " ")
    beverage_type: str | None = None
    if normalized == "wine" or " wine" in normalized or normalized.startswith("wine"):
        beverage_type = "wine"
    elif "distilled spirits" in normalized or "distilled spirit" in normalized:
        beverage_type = "distilled_spirits"
    elif normalized in {"spirits", "spirit"}:
        beverage_type = "distilled_spirits"
    if beverage_type is None:
        return None
    return BeverageTypeClassification(
        beverage_type=beverage_type,
        confidence=1.0,
        evidence={"source": "qwen", "raw_value": value},
    )


def _predict_from_spans(
    spans: Sequence[OcrSpan] | None,
) -> BeverageTypeClassification | None:
    if not spans:
        return None
    text_blocks = [span.text for span in spans if span.text]
    if not text_blocks:
        return None
    return classify_beverage_type(text_blocks)


def beverage_type_presence(context: RuleContext) -> Finding:
    selected: str | None = None
    if context.application_fields is not None:
        selected = context.application_fields.beverage_type
    prediction = context.label_info.beverage_type
    source = "label_info"
    if prediction is None:
        prediction = _predict_from_spans(context.spans)
        source = "ocr_spans"
    if selected:
        if prediction is None:
            return build_finding(
                "beverage_type_presence",
                "needs_review",
                "Beverage type selected, but none was detected on the label.",
                field="beverage_type",
                severity="warning",
                evidence={
                    "prediction_source": source,
                    "selected": selected,
                    "prediction": None,
                },
            )
        if prediction.beverage_type != selected:
            return build_finding(
                "beverage_type_presence",
                "needs_review",
                "Selected beverage type does not match the detected label type.",
                field="beverage_type",
                severity="warning",
                evidence={
                    "prediction_source": source,
                    "selected": selected,
                    "prediction": prediction.model_dump(),
                },
            )
        return build_finding(
            "beverage_type_presence",
            "pass",
            "Selected beverage type matches the detected label type.",
            field="beverage_type",
            severity="info",
            evidence={
                "prediction_source": source,
                "selected": selected,
                "prediction": prediction.model_dump(),
            },
        )
    if prediction is None:
        return build_finding(
            "beverage_type_presence",
            "fail",
            "Beverage type not detected on the label.",
            field="beverage_type",
        )
    status = "pass" if should_auto_apply_classification(prediction) else "needs_review"
    message = (
        "Beverage type detected on the label."
        if status == "pass"
        else "Beverage type detected, but confidence is low."
    )
    severity = "info" if status == "pass" else "warning"
    return build_finding(
        "beverage_type_presence",
        status,
        message,
        field="beverage_type",
        severity=severity,
        evidence={
            "prediction_source": source,
            "prediction": prediction.model_dump(),
        },
    )
