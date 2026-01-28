import pytest

from cola_label_verification.models import FieldExtraction, LabelInfo
from cola_label_verification.rules.fd_and_c_yellow_5_presence import (
    fd_and_c_yellow_5_presence,
)
from cola_label_verification.rules.models import RuleContext


def _context_with_value(value: str | None) -> RuleContext:
    label_info = LabelInfo(fd_and_c_yellow_5=FieldExtraction(value=value))
    return RuleContext(label_info=label_info, application_fields=None)


def test_fd_and_c_yellow_5_presence_passes_with_value() -> None:
    finding = fd_and_c_yellow_5_presence(
        _context_with_value("Contains FD&C Yellow #5.")
    )
    assert finding.status == "pass"
    assert finding.severity == "info"
    assert finding.field == "fd_and_c_yellow_5"
    assert finding.rule_id == "fd_and_c_yellow_5_presence"
    assert finding.message == "FD&C Yellow #5 disclosure detected."


@pytest.mark.parametrize("value", [None, ""])
def test_fd_and_c_yellow_5_presence_not_evaluated_without_value(
    value: str | None,
) -> None:
    finding = fd_and_c_yellow_5_presence(_context_with_value(value))
    assert finding.status == "not_evaluated"
    assert finding.severity == "info"
    assert finding.field == "fd_and_c_yellow_5"
    assert finding.rule_id == "fd_and_c_yellow_5_presence"
    assert (
        finding.message
        == "FD&C Yellow #5 disclosure not detected; requirement depends on formulation."
    )
