from cola_label_verification.models import (
    BeverageTypeClassification,
    FieldExtraction,
    LabelInfo,
)
from cola_label_verification.rules.commodity_statement_distilled_from_presence import (
    commodity_statement_distilled_from_presence,
)
from cola_label_verification.rules.models import RuleContext


def _context_with_distilled_from(
    value: str | None,
    *,
    beverage_type: str | None,
) -> RuleContext:
    classification = None
    if beverage_type is not None:
        classification = BeverageTypeClassification(
            beverage_type=beverage_type,
            confidence=0.92,
            evidence={"source": "test"},
        )
    label_info = LabelInfo(
        beverage_type=classification,
        commodity_statement_distilled_from=FieldExtraction(value=value),
    )
    return RuleContext(label_info=label_info, application_fields=None)


def test_distilled_from_presence_not_evaluated_without_beverage_type() -> None:
    finding = commodity_statement_distilled_from_presence(
        _context_with_distilled_from("Distilled from corn", beverage_type=None)
    )
    assert finding.status == "not_evaluated"
    assert finding.rule_id == "commodity_statement_distilled_from_presence"
    assert finding.field == "commodity_statement_distilled_from"
    assert finding.message == "Beverage type not selected; rule not evaluated."


def test_distilled_from_presence_not_applicable_for_wine() -> None:
    finding = commodity_statement_distilled_from_presence(
        _context_with_distilled_from("Distilled from rye", beverage_type="wine")
    )
    assert finding.status == "not_applicable"
    assert finding.field == "commodity_statement_distilled_from"
    assert finding.message == "Rule not applicable to the selected beverage type."


def test_distilled_from_presence_passes_with_value() -> None:
    finding = commodity_statement_distilled_from_presence(
        _context_with_distilled_from(
            "Distilled from wheat", beverage_type="distilled_spirits"
        )
    )
    assert finding.status == "pass"
    assert finding.severity == "info"
    assert finding.field == "commodity_statement_distilled_from"
    assert finding.message == "Distilled-from commodity statement detected."


def test_distilled_from_presence_not_evaluated_when_missing() -> None:
    finding = commodity_statement_distilled_from_presence(
        _context_with_distilled_from(None, beverage_type="distilled_spirits")
    )
    assert finding.status == "not_evaluated"
    assert finding.field == "commodity_statement_distilled_from"
    assert (
        finding.message
        == "Distilled-from commodity statement not detected; requirement depends on "
        "formulation."
    )


def test_distilled_from_presence_treats_empty_string_as_missing() -> None:
    finding = commodity_statement_distilled_from_presence(
        _context_with_distilled_from("", beverage_type="distilled_spirits")
    )
    assert finding.status == "not_evaluated"
    assert finding.field == "commodity_statement_distilled_from"
