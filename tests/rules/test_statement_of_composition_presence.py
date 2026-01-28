from cola_label_verification.models import (
    BeverageTypeClassification,
    FieldExtraction,
    LabelInfo,
)
from cola_label_verification.rules.models import RuleContext
from cola_label_verification.rules.statement_of_composition_presence import (
    statement_of_composition_presence,
)


def _context_with_statement(
    beverage_type: str | None,
    statement_value: str | None,
) -> RuleContext:
    classification = None
    if beverage_type is not None:
        classification = BeverageTypeClassification(
            beverage_type=beverage_type,
            confidence=0.9,
            evidence={"source": "test"},
        )
    label_info = LabelInfo(
        beverage_type=classification,
        statement_of_composition=FieldExtraction(value=statement_value),
    )
    return RuleContext(label_info=label_info, application_fields=None)


def test_statement_of_composition_not_evaluated_without_beverage_type() -> None:
    finding = statement_of_composition_presence(_context_with_statement(None, None))
    assert finding.status == "not_evaluated"
    assert finding.severity == "info"
    assert finding.field == "statement_of_composition"
    assert finding.message == "Beverage type not selected; rule not evaluated."


def test_statement_of_composition_not_applicable_for_wine() -> None:
    finding = statement_of_composition_presence(
        _context_with_statement("wine", "Gin")
    )
    assert finding.status == "not_applicable"
    assert finding.severity == "info"
    assert finding.field == "statement_of_composition"
    assert finding.message == "Rule not applicable to the selected beverage type."


def test_statement_of_composition_passes_with_value_for_spirits() -> None:
    finding = statement_of_composition_presence(
        _context_with_statement("distilled_spirits", "Distilled from apples")
    )
    assert finding.status == "pass"
    assert finding.severity == "info"
    assert finding.field == "statement_of_composition"
    assert finding.message == "Statement of composition detected."


def test_statement_of_composition_not_evaluated_when_missing_value() -> None:
    finding = statement_of_composition_presence(
        _context_with_statement("distilled_spirits", "")
    )
    assert finding.status == "not_evaluated"
    assert finding.severity == "info"
    assert finding.field == "statement_of_composition"
    assert (
        finding.message
        == "Statement of composition not detected; required for some distinctive "
        "or fanciful designations."
    )
