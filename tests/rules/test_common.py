import pytest

from cola_label_verification.models import BeverageTypeClassification, LabelInfo
from cola_label_verification.rules.common import (
    build_finding,
    is_imported,
    presence_rule,
    require_beverage_type,
    resolve_beverage_type,
)
from cola_label_verification.rules.models import ApplicationFields, RuleContext


def _context_with_prediction(beverage_type: str | None) -> RuleContext:
    classification = None
    if beverage_type is not None:
        classification = BeverageTypeClassification(
            beverage_type=beverage_type,
            confidence=0.9,
            evidence={"source": "test"},
        )
    return RuleContext(
        label_info=LabelInfo(beverage_type=classification),
        application_fields=None,
    )


def test_resolve_beverage_type_returns_none_without_prediction() -> None:
    assert resolve_beverage_type(_context_with_prediction(None)) is None


@pytest.mark.parametrize("beverage_type", ["wine", "distilled_spirits"])
def test_resolve_beverage_type_returns_prediction(beverage_type: str) -> None:
    assert resolve_beverage_type(_context_with_prediction(beverage_type)) == beverage_type


def test_build_finding_defaults_to_warning() -> None:
    finding = build_finding("rule_id", "fail", "Missing field")
    assert finding.rule_id == "rule_id"
    assert finding.status == "fail"
    assert finding.message == "Missing field"
    assert finding.severity == "warning"
    assert finding.field is None
    assert finding.evidence is None


def test_build_finding_accepts_field_and_evidence() -> None:
    evidence = {"key": "value"}
    finding = build_finding(
        "rule_id",
        "pass",
        "Present",
        severity="info",
        field="field",
        evidence=evidence,
    )
    assert finding.severity == "info"
    assert finding.field == "field"
    assert finding.evidence == evidence


def test_require_beverage_type_returns_not_evaluated_when_missing() -> None:
    finding = require_beverage_type(
        _context_with_prediction(None),
        allowed={"wine"},
        rule_id="rule_id",
        field="field",
    )
    assert finding is not None
    assert finding.status == "not_evaluated"
    assert finding.message == "Beverage type not selected; rule not evaluated."
    assert finding.severity == "info"
    assert finding.field == "field"


def test_require_beverage_type_returns_not_applicable_when_not_allowed() -> None:
    finding = require_beverage_type(
        _context_with_prediction("wine"),
        allowed={"distilled_spirits"},
        rule_id="rule_id",
        field="field",
    )
    assert finding is not None
    assert finding.status == "not_applicable"
    assert (
        finding.message
        == "Rule not applicable to the selected beverage type."
    )
    assert finding.severity == "info"
    assert finding.field == "field"


def test_require_beverage_type_returns_none_when_allowed() -> None:
    assert (
        require_beverage_type(
            _context_with_prediction("wine"),
            allowed={"wine"},
            rule_id="rule_id",
            field="field",
        )
        is None
    )


def test_presence_rule_passes_when_value_present() -> None:
    finding = presence_rule(
        "present",
        rule_id="rule_id",
        field="field",
        present_message="Present",
        missing_message="Missing",
    )
    assert finding.status == "pass"
    assert finding.message == "Present"
    assert finding.severity == "info"


def test_presence_rule_fails_when_required_and_missing() -> None:
    finding = presence_rule(
        None,
        rule_id="rule_id",
        field="field",
        present_message="Present",
        missing_message="Missing",
        required=True,
    )
    assert finding.status == "fail"
    assert finding.message == "Missing"
    assert finding.severity == "warning"


def test_presence_rule_not_applicable_with_override_message() -> None:
    finding = presence_rule(
        None,
        rule_id="rule_id",
        field="field",
        present_message="Present",
        missing_message="Missing",
        required=False,
        not_applicable_message="Not required",
    )
    assert finding.status == "not_applicable"
    assert finding.message == "Not required"
    assert finding.severity == "info"


def test_presence_rule_not_evaluated_with_default_message() -> None:
    finding = presence_rule(
        None,
        rule_id="rule_id",
        field="field",
        present_message="Present",
        missing_message="Missing",
        required=None,
    )
    assert finding.status == "not_evaluated"
    assert finding.message == "Missing"
    assert finding.severity == "info"


@pytest.mark.parametrize(
    ("application_fields", "expected"),
    [
        (None, None),
        (ApplicationFields(), None),
        (ApplicationFields(source_of_product=()), None),
        (ApplicationFields(source_of_product=(" Imported ", "Other")), True),
        (ApplicationFields(source_of_product=("domestic",)), False),
        (ApplicationFields(source_of_product=("Domestic", "Imported")), True),
        (ApplicationFields(source_of_product=("International",)), None),
    ],
)
def test_is_imported_normalizes_values(
    application_fields: ApplicationFields | None,
    expected: bool | None,
) -> None:
    assert is_imported(application_fields) is expected
