import pytest

from cola_label_verification.models import FieldExtraction, LabelInfo
from cola_label_verification.rules.brand_name_presence import brand_name_presence
from cola_label_verification.rules.models import RuleContext


def _context_with_brand_name(value: str | None) -> RuleContext:
    label_info = LabelInfo(brand_name=FieldExtraction(value=value))
    return RuleContext(label_info=label_info, application_fields=None)


@pytest.mark.parametrize("value", [None, ""])
def test_brand_name_presence_fails_when_missing(value: str | None) -> None:
    finding = brand_name_presence(_context_with_brand_name(value))
    assert finding.rule_id == "brand_name_presence"
    assert finding.status == "fail"
    assert finding.message == "Brand name not detected on the label."
    assert finding.field == "brand_name"
    assert finding.severity == "warning"
    assert finding.evidence is None


def test_brand_name_presence_passes_when_present() -> None:
    finding = brand_name_presence(_context_with_brand_name("Coca-Cola"))
    assert finding.rule_id == "brand_name_presence"
    assert finding.status == "pass"
    assert finding.message == "Brand name detected."
    assert finding.field == "brand_name"
    assert finding.severity == "info"
    assert finding.evidence is None


def test_brand_name_presence_treats_whitespace_as_present() -> None:
    finding = brand_name_presence(_context_with_brand_name("  "))
    assert finding.status == "pass"
    assert finding.field == "brand_name"
