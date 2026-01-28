import pytest

from cola_label_verification.models import (
    BeverageTypeClassification,
    FieldExtraction,
    LabelInfo,
)
from cola_label_verification.rules.coloring_materials_presence import (
    coloring_materials_presence,
)
from cola_label_verification.rules.models import RuleContext


def _context_with_beverage_type(
    beverage_type: str | None,
    *,
    coloring_value: str | None = None,
) -> RuleContext:
    classification = None
    if beverage_type is not None:
        classification = BeverageTypeClassification(
            beverage_type=beverage_type,
            confidence=0.9,
            evidence={"source": "test"},
        )
    coloring_materials = FieldExtraction(value=coloring_value)
    label_info = LabelInfo(
        beverage_type=classification,
        coloring_materials=coloring_materials,
    )
    return RuleContext(label_info=label_info, application_fields=None)


def test_coloring_materials_presence_requires_beverage_type() -> None:
    finding = coloring_materials_presence(
        _context_with_beverage_type(None, coloring_value="Caramel color")
    )
    assert finding.status == "not_evaluated"
    assert finding.field == "coloring_materials"
    assert finding.severity == "info"


def test_coloring_materials_presence_not_applicable_for_wine() -> None:
    finding = coloring_materials_presence(
        _context_with_beverage_type("wine", coloring_value="Caramel color")
    )
    assert finding.status == "not_applicable"
    assert finding.field == "coloring_materials"
    assert finding.severity == "info"


def test_coloring_materials_presence_passes_with_value() -> None:
    finding = coloring_materials_presence(
        _context_with_beverage_type("distilled_spirits", coloring_value="Caramel")
    )
    assert finding.status == "pass"
    assert finding.field == "coloring_materials"
    assert finding.severity == "info"
    assert finding.message == "Coloring materials disclosure detected."


@pytest.mark.parametrize("coloring_value", [None, ""])
def test_coloring_materials_presence_not_evaluated_when_missing(
    coloring_value: str | None,
) -> None:
    finding = coloring_materials_presence(
        _context_with_beverage_type("distilled_spirits", coloring_value=coloring_value)
    )
    assert finding.status == "not_evaluated"
    assert finding.field == "coloring_materials"
    assert finding.severity == "info"
    assert (
        finding.message
        == "Coloring materials disclosure not detected; requirement depends on "
        "formulation."
    )
