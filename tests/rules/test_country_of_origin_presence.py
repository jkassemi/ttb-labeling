from cola_label_verification.models import FieldExtraction, LabelInfo
from cola_label_verification.rules.country_of_origin_presence import (
    country_of_origin_presence,
)
from cola_label_verification.rules.models import ApplicationFields, RuleContext


def _context(
    country_value: str | None,
    *,
    source_of_product: tuple[str, ...] | None = None,
    include_application_fields: bool = True,
) -> RuleContext:
    label_info = LabelInfo(
        country_of_origin=FieldExtraction(value=country_value),
    )
    application_fields = None
    if include_application_fields:
        application_fields = ApplicationFields(source_of_product=source_of_product)
    return RuleContext(
        label_info=label_info,
        application_fields=application_fields,
    )


def test_country_of_origin_presence_passes_with_value() -> None:
    finding = country_of_origin_presence(
        _context("Italy", include_application_fields=False)
    )
    assert finding.status == "pass"
    assert finding.field == "country_of_origin"
    assert finding.rule_id == "country_of_origin_presence"
    assert finding.severity == "info"


def test_country_of_origin_presence_fails_when_imported_missing() -> None:
    finding = country_of_origin_presence(
        _context(None, source_of_product=("  Imported ",))
    )
    assert finding.status == "fail"
    assert finding.field == "country_of_origin"
    assert finding.message == "Country of origin statement not detected."
    assert finding.severity == "warning"


def test_country_of_origin_presence_not_applicable_when_domestic_missing() -> None:
    finding = country_of_origin_presence(
        _context(None, source_of_product=("domestic",))
    )
    assert finding.status == "not_applicable"
    assert finding.field == "country_of_origin"
    assert (
        finding.message
        == "Country of origin statement not required for domestic products."
    )
    assert finding.severity == "info"


def test_country_of_origin_presence_not_evaluated_without_source() -> None:
    finding = country_of_origin_presence(
        _context(None, include_application_fields=False)
    )
    assert finding.status == "not_evaluated"
    assert finding.field == "country_of_origin"
    assert (
        finding.message
        == "Source of product not provided; cannot determine if country of origin "
        "statement is required."
    )
    assert finding.severity == "info"


def test_country_of_origin_presence_treats_empty_string_as_missing() -> None:
    finding = country_of_origin_presence(
        _context("", source_of_product=("imported",))
    )
    assert finding.status == "fail"
    assert finding.field == "country_of_origin"


def test_country_of_origin_presence_not_evaluated_with_unknown_source() -> None:
    finding = country_of_origin_presence(
        _context(None, source_of_product=("unknown",))
    )
    assert finding.status == "not_evaluated"
    assert finding.field == "country_of_origin"
