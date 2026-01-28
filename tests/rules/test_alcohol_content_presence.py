from cola_label_verification.models import FieldExtraction, LabelInfo
from cola_label_verification.rules.alcohol_content_presence import (
    alcohol_content_presence,
)
from cola_label_verification.rules.models import RuleContext


def _context_with_alcohol_content(
    value: str | None,
    numeric_value: float | None,
) -> RuleContext:
    return RuleContext(
        label_info=LabelInfo(
            alcohol_content=FieldExtraction(
                value=value,
                numeric_value=numeric_value,
            )
        ),
        application_fields=None,
    )


def test_alcohol_content_presence_fails_when_missing() -> None:
    finding = alcohol_content_presence(_context_with_alcohol_content(None, None))
    assert finding.rule_id == "alcohol_content_presence"
    assert finding.status == "fail"
    assert finding.field == "alcohol_content"
    assert finding.severity == "warning"


def test_alcohol_content_presence_fails_with_empty_string_value() -> None:
    finding = alcohol_content_presence(_context_with_alcohol_content("", None))
    assert finding.status == "fail"
    assert finding.message == "Alcohol content not detected."


def test_alcohol_content_presence_passes_with_value() -> None:
    finding = alcohol_content_presence(_context_with_alcohol_content("12% ABV", None))
    assert finding.status == "pass"
    assert finding.field == "alcohol_content"
    assert finding.severity == "info"


def test_alcohol_content_presence_passes_with_numeric_value_only() -> None:
    finding = alcohol_content_presence(_context_with_alcohol_content(None, 0.0))
    assert finding.status == "pass"
    assert finding.message == "Alcohol content detected."
