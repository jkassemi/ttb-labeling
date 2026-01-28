from dataclasses import FrozenInstanceError

import pytest

from cola_label_verification.models import LabelInfo
from cola_label_verification.rules.models import (
    ApplicationFields,
    ChecklistResult,
    Finding,
    RuleContext,
    RulesConfig,
)


def test_application_fields_defaults_are_none() -> None:
    fields = ApplicationFields()
    assert all(value is None for value in vars(fields).values())


def test_application_fields_accepts_tuple_source() -> None:
    fields = ApplicationFields(
        brand_name="Acme",
        source_of_product=("Imported", "Domestic"),
    )
    assert fields.brand_name == "Acme"
    assert fields.source_of_product == ("Imported", "Domestic")


def test_application_fields_is_frozen() -> None:
    fields = ApplicationFields()
    with pytest.raises(FrozenInstanceError):
        fields.brand_name = "New Brand"


def test_rules_config_defaults_and_overrides() -> None:
    default_config = RulesConfig()
    assert default_config.verification_threshold == 0.65

    custom_config = RulesConfig(verification_threshold=0.9)
    assert custom_config.verification_threshold == 0.9


def test_rule_context_default_config_is_unique() -> None:
    context_one = RuleContext(label_info=LabelInfo(), application_fields=None)
    context_two = RuleContext(label_info=LabelInfo(), application_fields=None)

    assert context_one.rules_config == RulesConfig()
    assert context_one.rules_config is not context_two.rules_config


def test_rule_context_keeps_explicit_inputs() -> None:
    rules_config = RulesConfig(verification_threshold=0.8)
    application_fields = ApplicationFields(brand_name="Acme")
    spans = [object()]
    images = [object()]

    context = RuleContext(
        label_info=LabelInfo(),
        application_fields=application_fields,
        rules_config=rules_config,
        spans=spans,
        images=images,
    )

    assert context.rules_config is rules_config
    assert context.application_fields is application_fields
    assert context.spans is spans
    assert context.images is images


def test_finding_and_checklist_result_are_frozen() -> None:
    finding = Finding(
        rule_id="warning_text",
        status="pass",
        message="Found warning text.",
        severity="info",
        field="warning_text",
        evidence={"matched": True},
    )
    result = ChecklistResult(findings=[finding])

    assert result.findings == [finding]
    assert result.findings[0].evidence == {"matched": True}

    with pytest.raises(FrozenInstanceError):
        finding.message = "Updated"

    with pytest.raises(FrozenInstanceError):
        result.findings = []
