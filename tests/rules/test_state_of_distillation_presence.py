from cola_label_verification.models import (
    BeverageTypeClassification,
    FieldExtraction,
    LabelInfo,
)
from cola_label_verification.rules.models import RuleContext
from cola_label_verification.rules.state_of_distillation_presence import (
    state_of_distillation_presence,
)


def _context_with_state_of_distillation(
    *,
    beverage_type: str | None,
    state_of_distillation: str | None,
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
        state_of_distillation=FieldExtraction(value=state_of_distillation),
    )
    return RuleContext(label_info=label_info, application_fields=None)


def test_state_of_distillation_not_evaluated_without_beverage_type() -> None:
    finding = state_of_distillation_presence(
        _context_with_state_of_distillation(
            beverage_type=None,
            state_of_distillation="Texas",
        )
    )

    assert finding.rule_id == "state_of_distillation_presence"
    assert finding.status == "not_evaluated"
    assert finding.field == "state_of_distillation"
    assert finding.message == "Beverage type not selected; rule not evaluated."


def test_state_of_distillation_not_applicable_for_wine() -> None:
    finding = state_of_distillation_presence(
        _context_with_state_of_distillation(
            beverage_type="wine",
            state_of_distillation="Texas",
        )
    )

    assert finding.rule_id == "state_of_distillation_presence"
    assert finding.status == "not_applicable"
    assert finding.field == "state_of_distillation"
    assert finding.message == "Rule not applicable to the selected beverage type."


def test_state_of_distillation_passes_when_present_for_spirits() -> None:
    finding = state_of_distillation_presence(
        _context_with_state_of_distillation(
            beverage_type="distilled_spirits",
            state_of_distillation="Texas",
        )
    )

    assert finding.rule_id == "state_of_distillation_presence"
    assert finding.status == "pass"
    assert finding.field == "state_of_distillation"
    assert finding.message == "State of distillation statement detected."


def test_state_of_distillation_not_evaluated_when_missing() -> None:
    finding = state_of_distillation_presence(
        _context_with_state_of_distillation(
            beverage_type="distilled_spirits",
            state_of_distillation=None,
        )
    )

    assert finding.rule_id == "state_of_distillation_presence"
    assert finding.status == "not_evaluated"
    assert finding.field == "state_of_distillation"
    assert (
        finding.message
        == "State of distillation statement not detected; requirement depends on "
        "product type."
    )


def test_state_of_distillation_treats_empty_string_as_missing() -> None:
    finding = state_of_distillation_presence(
        _context_with_state_of_distillation(
            beverage_type="distilled_spirits",
            state_of_distillation="",
        )
    )

    assert finding.rule_id == "state_of_distillation_presence"
    assert finding.status == "not_evaluated"
    assert finding.field == "state_of_distillation"
