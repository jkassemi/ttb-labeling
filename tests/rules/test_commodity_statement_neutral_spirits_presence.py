from cola_label_verification.models import (
    BeverageTypeClassification,
    FieldExtraction,
    LabelInfo,
)
from cola_label_verification.rules.commodity_statement_neutral_spirits_presence import (
    commodity_statement_neutral_spirits_presence,
)
from cola_label_verification.rules.models import RuleContext


def _context_for_rule(
    value: str | None,
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
        commodity_statement_neutral_spirits=FieldExtraction(value=value),
        beverage_type=classification,
    )
    return RuleContext(label_info=label_info, application_fields=None)


def test_neutral_spirits_presence_not_evaluated_without_beverage_type() -> None:
    finding = commodity_statement_neutral_spirits_presence(
        _context_for_rule("Neutral spirits statement", None)
    )
    assert finding.status == "not_evaluated"
    assert finding.field == "commodity_statement_neutral_spirits"
    assert finding.message == "Beverage type not selected; rule not evaluated."


def test_neutral_spirits_presence_not_applicable_for_wine() -> None:
    finding = commodity_statement_neutral_spirits_presence(
        _context_for_rule("Neutral spirits statement", "wine")
    )
    assert finding.status == "not_applicable"
    assert finding.field == "commodity_statement_neutral_spirits"
    assert finding.message == "Rule not applicable to the selected beverage type."


def test_neutral_spirits_presence_passes_when_present_for_spirits() -> None:
    finding = commodity_statement_neutral_spirits_presence(
        _context_for_rule("Neutral spirits statement", "distilled_spirits")
    )
    assert finding.status == "pass"
    assert finding.field == "commodity_statement_neutral_spirits"
    assert finding.message == "Neutral spirits commodity statement detected."


def test_neutral_spirits_presence_not_evaluated_when_missing() -> None:
    finding = commodity_statement_neutral_spirits_presence(
        _context_for_rule(None, "distilled_spirits")
    )
    assert finding.status == "not_evaluated"
    assert finding.field == "commodity_statement_neutral_spirits"
    assert (
        finding.message
        == "Neutral spirits commodity statement not detected; requirement depends on formulation."
    )


def test_neutral_spirits_presence_not_evaluated_for_empty_string() -> None:
    finding = commodity_statement_neutral_spirits_presence(
        _context_for_rule("", "distilled_spirits")
    )
    assert finding.status == "not_evaluated"
    assert finding.field == "commodity_statement_neutral_spirits"
