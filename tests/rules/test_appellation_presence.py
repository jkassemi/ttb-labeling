import pytest

from cola_label_verification.models import (
    BeverageTypeClassification,
    FieldExtraction,
    LabelInfo,
)
from cola_label_verification.rules.appellation_presence import appellation_presence
from cola_label_verification.rules.models import RuleContext


def _context_with_appellation(
    value: str | None,
    *,
    beverage_type: str | None,
) -> RuleContext:
    classification = None
    if beverage_type is not None:
        classification = BeverageTypeClassification(
            beverage_type=beverage_type,
            confidence=0.9,
            evidence={"source": "test"},
        )
    label_info = LabelInfo(
        appellation_of_origin=FieldExtraction(value=value),
        beverage_type=classification,
    )
    return RuleContext(label_info=label_info, application_fields=None)


def test_appellation_presence_not_evaluated_without_beverage_type() -> None:
    finding = appellation_presence(
        _context_with_appellation("Napa Valley", beverage_type=None)
    )
    assert finding.rule_id == "appellation_presence"
    assert finding.status == "not_evaluated"
    assert finding.field == "appellation_of_origin"
    assert finding.severity == "info"


def test_appellation_presence_not_applicable_for_non_wine() -> None:
    finding = appellation_presence(
        _context_with_appellation("Napa Valley", beverage_type="distilled_spirits")
    )
    assert finding.rule_id == "appellation_presence"
    assert finding.status == "not_applicable"
    assert finding.field == "appellation_of_origin"
    assert finding.severity == "info"


@pytest.mark.parametrize("value", [None, ""])
def test_appellation_presence_fails_without_value(value: str | None) -> None:
    finding = appellation_presence(
        _context_with_appellation(value, beverage_type="wine")
    )
    assert finding.rule_id == "appellation_presence"
    assert finding.status == "fail"
    assert finding.field == "appellation_of_origin"
    assert finding.severity == "warning"


def test_appellation_presence_passes_with_value() -> None:
    finding = appellation_presence(
        _context_with_appellation("Napa Valley", beverage_type="wine")
    )
    assert finding.rule_id == "appellation_presence"
    assert finding.status == "pass"
    assert finding.field == "appellation_of_origin"
    assert finding.severity == "info"
