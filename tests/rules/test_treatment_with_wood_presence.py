from cola_label_verification.models import (
    BeverageTypeClassification,
    FieldExtraction,
    LabelInfo,
)
from cola_label_verification.rules.models import RuleContext
from cola_label_verification.rules.treatment_with_wood_presence import (
    treatment_with_wood_presence,
)


def _context_for(
    beverage_type: str | None,
    *,
    treatment_value: str | None = None,
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
        treatment_with_wood=FieldExtraction(value=treatment_value),
    )
    return RuleContext(label_info=label_info, application_fields=None)


def test_treatment_with_wood_not_evaluated_without_beverage_type() -> None:
    finding = treatment_with_wood_presence(_context_for(None))

    assert finding.status == "not_evaluated"
    assert finding.field == "treatment_with_wood"
    assert finding.severity == "info"
    assert finding.message == "Beverage type not selected; rule not evaluated."


def test_treatment_with_wood_not_applicable_for_wine() -> None:
    finding = treatment_with_wood_presence(_context_for("wine"))

    assert finding.status == "not_applicable"
    assert finding.field == "treatment_with_wood"
    assert finding.severity == "info"
    assert finding.message == "Rule not applicable to the selected beverage type."


def test_treatment_with_wood_passes_for_distilled_spirits() -> None:
    finding = treatment_with_wood_presence(
        _context_for("distilled_spirits", treatment_value="Aged in oak barrels")
    )

    assert finding.status == "pass"
    assert finding.field == "treatment_with_wood"
    assert finding.severity == "info"
    assert finding.message == "Treatment with wood disclosure detected."


def test_treatment_with_wood_not_evaluated_when_missing() -> None:
    finding = treatment_with_wood_presence(
        _context_for("distilled_spirits", treatment_value=None)
    )

    assert finding.status == "not_evaluated"
    assert finding.field == "treatment_with_wood"
    assert finding.severity == "info"
    assert (
        finding.message
        == "Treatment with wood disclosure not detected; requirement depends on "
        "production method."
    )


def test_treatment_with_wood_not_evaluated_with_empty_string() -> None:
    finding = treatment_with_wood_presence(
        _context_for("distilled_spirits", treatment_value="")
    )

    assert finding.status == "not_evaluated"
    assert finding.field == "treatment_with_wood"
    assert finding.severity == "info"
