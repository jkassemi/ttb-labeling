import pytest

from cola_label_verification.models import (
    BeverageTypeClassification,
    FieldExtraction,
    LabelInfo,
)
from cola_label_verification.rules.models import RuleContext
from cola_label_verification.rules.percentage_of_foreign_wine_presence import (
    percentage_of_foreign_wine_presence,
)


def _context_with_percentage(
    beverage_type: str | None,
    percentage_value: str | None,
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
        percentage_of_foreign_wine=FieldExtraction(value=percentage_value),
    )
    return RuleContext(label_info=label_info, application_fields=None)


def test_percentage_of_foreign_wine_presence_not_evaluated_without_beverage_type() -> None:
    finding = percentage_of_foreign_wine_presence(_context_with_percentage(None, "5%"))
    assert finding.status == "not_evaluated"
    assert finding.message == "Beverage type not selected; rule not evaluated."
    assert finding.field == "percentage_of_foreign_wine"
    assert finding.rule_id == "percentage_of_foreign_wine_presence"
    assert finding.severity == "info"


def test_percentage_of_foreign_wine_presence_not_applicable_for_spirits() -> None:
    finding = percentage_of_foreign_wine_presence(
        _context_with_percentage("distilled_spirits", "5%")
    )
    assert finding.status == "not_applicable"
    assert finding.message == "Rule not applicable to the selected beverage type."
    assert finding.field == "percentage_of_foreign_wine"
    assert finding.rule_id == "percentage_of_foreign_wine_presence"
    assert finding.severity == "info"


def test_percentage_of_foreign_wine_presence_passes_when_present() -> None:
    finding = percentage_of_foreign_wine_presence(
        _context_with_percentage("wine", "75% foreign wine")
    )
    assert finding.status == "pass"
    assert finding.message == "Percentage of foreign wine statement detected."
    assert finding.field == "percentage_of_foreign_wine"
    assert finding.rule_id == "percentage_of_foreign_wine_presence"
    assert finding.severity == "info"


@pytest.mark.parametrize("missing_value", [None, ""])
def test_percentage_of_foreign_wine_presence_not_evaluated_when_missing_for_wine(
    missing_value: str | None,
) -> None:
    finding = percentage_of_foreign_wine_presence(
        _context_with_percentage("wine", missing_value)
    )
    assert finding.status == "not_evaluated"
    assert (
        finding.message
        == "Percentage of foreign wine statement not detected; requirement depends on "
        "labeling."
    )
    assert finding.field == "percentage_of_foreign_wine"
    assert finding.rule_id == "percentage_of_foreign_wine_presence"
    assert finding.severity == "info"
