from cola_label_verification.models import FieldExtraction, LabelInfo
from cola_label_verification.rules.carmine_presence import carmine_presence
from cola_label_verification.rules.models import RuleContext

PRESENT_MESSAGE = "Carmine/cochineal disclosure detected."
MISSING_MESSAGE = (
    "Carmine/cochineal disclosure not detected; requirement depends on "
    "formulation."
)


def _context_with_carmine(value: str | None) -> RuleContext:
    label_info = LabelInfo(carmine=FieldExtraction(value=value))
    return RuleContext(label_info=label_info, application_fields=None)


def test_carmine_presence_passes_when_value_present() -> None:
    finding = carmine_presence(_context_with_carmine("Contains Carmine"))
    assert finding.status == "pass"
    assert finding.field == "carmine"
    assert finding.rule_id == "carmine_presence"
    assert finding.severity == "info"
    assert finding.message == PRESENT_MESSAGE


def test_carmine_presence_not_evaluated_when_missing() -> None:
    finding = carmine_presence(_context_with_carmine(None))
    assert finding.status == "not_evaluated"
    assert finding.field == "carmine"
    assert finding.rule_id == "carmine_presence"
    assert finding.severity == "info"
    assert finding.message == MISSING_MESSAGE


def test_carmine_presence_not_evaluated_when_empty_string() -> None:
    finding = carmine_presence(_context_with_carmine(""))
    assert finding.status == "not_evaluated"
    assert finding.field == "carmine"
    assert finding.rule_id == "carmine_presence"
    assert finding.severity == "info"
    assert finding.message == MISSING_MESSAGE
