import pytest

from cola_label_verification.models import (
    BeverageTypeClassification,
    FieldExtraction,
    LabelInfo,
)
from cola_label_verification.rules.class_type_presence import class_type_presence
from cola_label_verification.rules.models import RuleContext


def _field(value: str | None) -> FieldExtraction:
    return FieldExtraction(value=value)


def _context_with_class_type(
    beverage_type: str | None,
    *,
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
        class_type=_field(class_type),
        grape_varietals=_field(grape_varietals),
        statement_of_composition=_field(statement_of_composition),
    )
    return RuleContext(label_info=label_info, application_fields=None)


def test_class_type_presence_not_evaluated_without_beverage_type() -> None:
    finding = class_type_presence(
        _context_with_class_type(None, class_type="VODKA")
    )
    assert finding.status == "not_evaluated"
    assert finding.field == "class_type"
    assert finding.severity == "info"


def test_class_type_presence_fails_for_spirits_with_noise_value() -> None:
    finding = class_type_presence(
        _context_with_class_type("distilled_spirits", class_type="___")
    )
    assert finding.status == "fail"
    assert finding.field == "class_type"
    assert finding.severity == "warning"


def test_class_type_presence_passes_for_spirits_with_trailing_abv() -> None:
    finding = class_type_presence(
        _context_with_class_type(
            "distilled_spirits",
            class_type="VODKA 40% ALC/VOL",
        )
    )
    assert finding.status == "pass"
    assert finding.field == "class_type"
    assert finding.severity == "info"


@pytest.mark.parametrize(
    "grape_varietals,statement_of_composition",
    [
        ("Cabernet Sauvignon", None),
        (None, "Grape wine"),
    ],
)
def test_class_type_presence_not_applicable_for_wine_with_designation_fields(
    grape_varietals: str | None,
    statement_of_composition: str | None,
) -> None:
    finding = class_type_presence(
        _context_with_class_type(
            "wine",
            class_type=None,
            grape_varietals=grape_varietals,
            statement_of_composition=statement_of_composition,
        )
    )
    assert finding.status == "not_applicable"
    assert finding.field == "class_type"
    assert finding.severity == "info"


def test_class_type_presence_fails_for_wine_without_designation() -> None:
    finding = class_type_presence(
        _context_with_class_type("wine", class_type=None)
    )
    assert finding.status == "fail"
    assert finding.field == "class_type"
    assert finding.severity == "warning"


def test_class_type_presence_passes_for_wine_with_value_without_keyword() -> None:
    finding = class_type_presence(
        _context_with_class_type("wine", class_type="Estate Reserve")
    )
    assert finding.status == "pass"
    assert finding.field == "class_type"
    assert finding.severity == "info"
