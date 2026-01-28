from cola_label_verification.models import BeverageTypeClassification, LabelInfo
from cola_label_verification.rules.beverage_type_presence import (
    beverage_type_presence,
    classify_beverage_type,
)
from cola_label_verification.rules.models import ApplicationFields, RuleContext


def _context_with_beverage_type(
    beverage_type: str | None,
    *,
    selected: str | None = None,
) -> RuleContext:
    classification = None
    if beverage_type is not None:
        classification = BeverageTypeClassification(
            beverage_type=beverage_type,
            confidence=0.9,
            evidence={"source": "test"},
        )
    application_fields = (
        ApplicationFields(beverage_type=selected) if selected is not None else None
    )
    return RuleContext(
        label_info=LabelInfo(beverage_type=classification),
        application_fields=application_fields,
    )


def test_beverage_type_presence_fails_without_prediction() -> None:
    finding = beverage_type_presence(_context_with_beverage_type(None))
    assert finding.status == "fail"
    assert finding.field == "beverage_type"


def test_beverage_type_presence_passes_with_prediction() -> None:
    finding = beverage_type_presence(_context_with_beverage_type("wine"))
    assert finding.status == "pass"
    assert finding.field == "beverage_type"


def test_beverage_type_presence_matches_selected_value() -> None:
    finding = beverage_type_presence(
        _context_with_beverage_type("wine", selected="wine")
    )
    assert finding.status == "pass"
    assert finding.field == "beverage_type"


def test_beverage_type_presence_flags_mismatch_with_selection() -> None:
    finding = beverage_type_presence(
        _context_with_beverage_type("wine", selected="distilled_spirits")
    )
    assert finding.status == "needs_review"
    assert finding.field == "beverage_type"


def test_beverage_type_presence_needs_review_without_prediction_when_selected() -> None:
    finding = beverage_type_presence(_context_with_beverage_type(None, selected="wine"))
    assert finding.status == "needs_review"
    assert finding.field == "beverage_type"


def test_beverage_type_classifies_wine_from_varietal() -> None:
    prediction = classify_beverage_type(["Cabernet Sauvignon"])
    assert prediction is not None
    assert prediction.beverage_type == "wine"
    assert prediction.confidence >= 0.6


def test_beverage_type_classifies_spirits_from_class() -> None:
    prediction = classify_beverage_type(["VODKA"])
    assert prediction is not None
    assert prediction.beverage_type == "distilled_spirits"
    assert prediction.confidence >= 0.6
