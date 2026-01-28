from cola_label_verification.models import FieldExtraction, LabelInfo
from cola_label_verification.rules.models import RuleContext
from cola_label_verification.rules.net_contents_presence import net_contents_presence


def _context_with_net_contents(
    *,
    value: str | None,
    numeric_value: float | None,
) -> RuleContext:
    extraction = FieldExtraction(value=value, numeric_value=numeric_value)
    label_info = LabelInfo(net_contents=extraction)
    return RuleContext(label_info=label_info, application_fields=None)


def test_net_contents_presence_fails_when_missing_value_and_numeric() -> None:
    finding = net_contents_presence(
        _context_with_net_contents(value=None, numeric_value=None)
    )
    assert finding.status == "fail"
    assert finding.rule_id == "net_contents_presence"
    assert finding.field == "net_contents"
    assert finding.severity == "warning"
    assert finding.message == "Net contents statement not detected."


def test_net_contents_presence_fails_with_blank_value_and_missing_numeric() -> None:
    finding = net_contents_presence(
        _context_with_net_contents(value="", numeric_value=None)
    )
    assert finding.status == "fail"


def test_net_contents_presence_passes_with_value() -> None:
    finding = net_contents_presence(
        _context_with_net_contents(value="750 mL", numeric_value=None)
    )
    assert finding.status == "pass"
    assert finding.rule_id == "net_contents_presence"
    assert finding.field == "net_contents"
    assert finding.severity == "info"
    assert finding.message == "Net contents statement detected."


def test_net_contents_presence_passes_with_numeric_value_only() -> None:
    finding = net_contents_presence(
        _context_with_net_contents(value=None, numeric_value=750.0)
    )
    assert finding.status == "pass"


def test_net_contents_presence_passes_with_zero_numeric_value() -> None:
    finding = net_contents_presence(
        _context_with_net_contents(value="", numeric_value=0.0)
    )
    assert finding.status == "pass"
