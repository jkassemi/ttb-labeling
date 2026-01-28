from cola_label_verification.models import (
    BeverageTypeClassification,
    FieldExtraction,
    LabelInfo,
)
from cola_label_verification.rules.models import RuleContext
from cola_label_verification.rules.statement_of_age_presence import (
    statement_of_age_presence,
)

NOT_EVALUATED_MESSAGE = (
    "Statement of age not detected; requirement depends on product type and aging."
)
NOT_SELECTED_MESSAGE = "Beverage type not selected; rule not evaluated."
NOT_APPLICABLE_MESSAGE = "Rule not applicable to the selected beverage type."
PRESENT_MESSAGE = "Statement of age detected."


def _context(
    beverage_type: str | None,
    *,
    statement_of_age: str | None = None,
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
        statement_of_age=FieldExtraction(value=statement_of_age),
    )
    return RuleContext(label_info=label_info, application_fields=None)


def test_statement_of_age_presence_not_evaluated_without_beverage_type() -> None:
    finding = statement_of_age_presence(_context(None, statement_of_age="12 years"))
    assert finding.status == "not_evaluated"
    assert finding.severity == "info"
    assert finding.field == "statement_of_age"
    assert finding.rule_id == "statement_of_age_presence"
    assert finding.message == NOT_SELECTED_MESSAGE


def test_statement_of_age_presence_not_applicable_for_wine() -> None:
    finding = statement_of_age_presence(_context("wine"))
    assert finding.status == "not_applicable"
    assert finding.severity == "info"
    assert finding.field == "statement_of_age"
    assert finding.rule_id == "statement_of_age_presence"
    assert finding.message == NOT_APPLICABLE_MESSAGE


def test_statement_of_age_presence_passes_when_present() -> None:
    finding = statement_of_age_presence(
        _context("distilled_spirits", statement_of_age="12 years")
    )
    assert finding.status == "pass"
    assert finding.severity == "info"
    assert finding.field == "statement_of_age"
    assert finding.rule_id == "statement_of_age_presence"
    assert finding.message == PRESENT_MESSAGE


def test_statement_of_age_presence_not_evaluated_when_missing() -> None:
    finding = statement_of_age_presence(_context("distilled_spirits"))
    assert finding.status == "not_evaluated"
    assert finding.severity == "info"
    assert finding.field == "statement_of_age"
    assert finding.rule_id == "statement_of_age_presence"
    assert finding.message == NOT_EVALUATED_MESSAGE

    finding = statement_of_age_presence(
        _context("distilled_spirits", statement_of_age="")
    )
    assert finding.status == "not_evaluated"
    assert finding.severity == "info"
    assert finding.field == "statement_of_age"
    assert finding.rule_id == "statement_of_age_presence"
    assert finding.message == NOT_EVALUATED_MESSAGE
