import pytest

from cola_label_verification.models import (
    BeverageTypeClassification,
    FieldExtraction,
    LabelInfo,
)
from cola_label_verification.rules.models import RuleContext
from cola_label_verification.rules.wine_designation_present import (
    wine_designation_present,
)


def _context_with_fields(
    *,
    beverage_type: str | None,
    class_type: str | None = None,
    grape_varietals: str | None = None,
    statement_of_composition: str | None = None,
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
        class_type=FieldExtraction(value=class_type),
        grape_varietals=FieldExtraction(value=grape_varietals),
        statement_of_composition=FieldExtraction(value=statement_of_composition),
    )
    return RuleContext(label_info=label_info, application_fields=None)


def test_wine_designation_present_not_evaluated_without_beverage_type() -> None:
    finding = wine_designation_present(
        _context_with_fields(
            beverage_type=None,
            class_type="Red Wine",
        )
    )

    assert finding.rule_id == "wine_designation_presence"
    assert finding.status == "not_evaluated"
    assert finding.severity == "info"
    assert finding.field == "class_type"


def test_wine_designation_present_not_applicable_for_distilled_spirits() -> None:
    finding = wine_designation_present(
        _context_with_fields(
            beverage_type="distilled_spirits",
            class_type="Red Wine",
        )
    )

    assert finding.rule_id == "wine_designation_presence"
    assert finding.status == "not_applicable"
    assert finding.severity == "info"
    assert finding.field == "class_type"


@pytest.mark.parametrize(
    ("class_type", "grape_varietals", "statement_of_composition"),
    [
        ("Red Wine", None, None),
        (None, "Cabernet Sauvignon", None),
        (None, None, "Contains natural flavors"),
    ],
)
def test_wine_designation_present_passes_when_any_field_present(
    class_type: str | None,
    grape_varietals: str | None,
    statement_of_composition: str | None,
) -> None:
    finding = wine_designation_present(
        _context_with_fields(
            beverage_type="wine",
            class_type=class_type,
            grape_varietals=grape_varietals,
            statement_of_composition=statement_of_composition,
        )
    )

    assert finding.rule_id == "wine_designation_presence"
    assert finding.status == "pass"
    assert finding.severity == "info"
    assert finding.field == "class_type"


@pytest.mark.parametrize(
    ("class_type", "grape_varietals", "statement_of_composition"),
    [
        (None, None, None),
        ("", "", ""),
    ],
)
def test_wine_designation_present_fails_when_missing(
    class_type: str | None,
    grape_varietals: str | None,
    statement_of_composition: str | None,
) -> None:
    finding = wine_designation_present(
        _context_with_fields(
            beverage_type="wine",
            class_type=class_type,
            grape_varietals=grape_varietals,
            statement_of_composition=statement_of_composition,
        )
    )

    assert finding.rule_id == "wine_designation_presence"
    assert finding.status == "fail"
    assert finding.field == "class_type"
