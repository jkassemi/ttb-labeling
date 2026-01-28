from cola_label_verification.models import FieldExtraction, LabelInfo
from cola_label_verification.rules.models import RuleContext
from cola_label_verification.rules.sulfite_declaration_presence import (
    sulfite_declaration_presence,
)


def _context_with_sulfite(value: str | None) -> RuleContext:
    return RuleContext(
        label_info=LabelInfo(sulfite_declaration=FieldExtraction(value=value)),
        application_fields=None,
    )


def test_sulfite_declaration_presence_passes_when_value_present() -> None:
    finding = sulfite_declaration_presence(_context_with_sulfite("Contains sulfites"))
    assert finding.status == "pass"
    assert finding.severity == "info"
    assert finding.field == "sulfite_declaration"
    assert finding.rule_id == "sulfite_declaration_presence"
    assert finding.message == "Sulfite declaration detected."


def test_sulfite_declaration_presence_not_evaluated_when_missing() -> None:
    finding = sulfite_declaration_presence(_context_with_sulfite(None))
    assert finding.status == "not_evaluated"
    assert finding.severity == "info"
    assert finding.field == "sulfite_declaration"
    assert finding.rule_id == "sulfite_declaration_presence"
    assert (
        finding.message
        == "Sulfite declaration not detected; requirement depends on formulation."
    )


def test_sulfite_declaration_presence_treats_empty_string_as_missing() -> None:
    finding = sulfite_declaration_presence(_context_with_sulfite(""))
    assert finding.status == "not_evaluated"
    assert finding.severity == "info"
    assert finding.field == "sulfite_declaration"
