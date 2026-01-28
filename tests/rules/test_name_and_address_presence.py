from cola_label_verification.models import FieldExtraction, LabelInfo
from cola_label_verification.rules.models import RuleContext
from cola_label_verification.rules.name_and_address_presence import (
    name_and_address_presence,
)


def _context_with_name_and_address(value: str | None) -> RuleContext:
    return RuleContext(
        label_info=LabelInfo(name_and_address=FieldExtraction(value=value)),
        application_fields=None,
    )


def test_name_and_address_presence_passes_with_value() -> None:
    finding = name_and_address_presence(
        _context_with_name_and_address("Acme Spirits, 123 Main St.")
    )
    assert finding.rule_id == "name_and_address_presence"
    assert finding.status == "pass"
    assert finding.severity == "info"
    assert finding.field == "name_and_address"
    assert finding.message == "Name and address statement detected."


def test_name_and_address_presence_fails_with_default_missing_value() -> None:
    finding = name_and_address_presence(
        RuleContext(label_info=LabelInfo(), application_fields=None)
    )
    assert finding.rule_id == "name_and_address_presence"
    assert finding.status == "fail"
    assert finding.severity == "warning"
    assert finding.field == "name_and_address"
    assert finding.message == "Name and address statement not detected."


def test_name_and_address_presence_fails_with_empty_string() -> None:
    finding = name_and_address_presence(_context_with_name_and_address(""))
    assert finding.status == "fail"
    assert finding.field == "name_and_address"
