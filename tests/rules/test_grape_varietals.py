import pytest

from cola_label_verification.models import (
    BeverageTypeClassification,
    FieldExtraction,
    LabelInfo,
)
from cola_label_verification.rules.grape_varietals import (
    ADMINISTRATIVELY_APPROVED_VARIETALS,
    APPELLATION_FIELD,
    DEFAULT_FIELD,
    grape_varietals,
    normalize_grape_name,
    split_grape_varietals,
)
from cola_label_verification.rules.models import ApplicationFields, RuleContext


def _field(value: str | None) -> FieldExtraction:
    return FieldExtraction(value=value)


def _label_info(
    *,
    beverage_type: str | None,
    grape_varietals_value: str | None,
    appellation_value: str | None,
) -> LabelInfo:
    classification = None
    if beverage_type is not None:
        classification = BeverageTypeClassification(
            beverage_type=beverage_type,
            confidence=0.9,
            evidence={"source": "test"},
        )
    return LabelInfo(
        grape_varietals=_field(grape_varietals_value),
        appellation_of_origin=_field(appellation_value),
        beverage_type=classification,
    )


def _context(
    *,
    beverage_type: str | None,
    grape_varietals_value: str | None,
    appellation_value: str | None = None,
    source_of_product: tuple[str, ...] | None = None,
) -> RuleContext:
    application_fields = None
    if source_of_product is not None:
        application_fields = ApplicationFields(source_of_product=source_of_product)
    return RuleContext(
        label_info=_label_info(
            beverage_type=beverage_type,
            grape_varietals_value=grape_varietals_value,
            appellation_value=appellation_value,
        ),
        application_fields=application_fields,
    )


def test_normalize_grape_name_strips_diacritics_and_symbols() -> None:
    assert normalize_grape_name("Torrontés Riojano") == "torrontesriojano"
    assert normalize_grape_name("Xarel·lo") == "xarello"


def test_split_grape_varietals_handles_multiple_separators() -> None:
    value = "Merlot, Syrah / Cabernet Sauvignon and Petit Verdot"
    assert split_grape_varietals(value) == [
        "Merlot",
        "Syrah",
        "Cabernet Sauvignon",
        "Petit Verdot",
    ]


@pytest.mark.parametrize(
    ("beverage_type", "status", "message"),
    [
        (None, "not_evaluated", "Beverage type not selected; rule not evaluated."),
        (
            "distilled_spirits",
            "not_applicable",
            "Rule not applicable to the selected beverage type.",
        ),
    ],
)
def test_grape_varietals_gates_on_beverage_type(
    beverage_type: str | None,
    status: str,
    message: str,
) -> None:
    finding = grape_varietals(
        _context(
            beverage_type=beverage_type,
            grape_varietals_value=None,
            appellation_value=None,
        )
    )
    assert finding.status == status
    assert finding.message == message
    assert finding.field == DEFAULT_FIELD
    assert finding.severity == "info"
    assert finding.evidence is None


def test_grape_varietals_fails_without_label_value() -> None:
    finding = grape_varietals(
        _context(
            beverage_type="wine",
            grape_varietals_value=None,
            appellation_value="Napa Valley",
            source_of_product=("domestic",),
        )
    )
    assert finding.status == "fail"
    assert finding.message == "Grape varietals not detected on the label."
    assert finding.field == DEFAULT_FIELD
    assert finding.severity == "warning"
    assert finding.evidence is None


def test_grape_varietals_requires_appellation_with_approved_domestic_varietal() -> None:
    approved = ADMINISTRATIVELY_APPROVED_VARIETALS[0]
    finding = grape_varietals(
        _context(
            beverage_type="wine",
            grape_varietals_value=approved,
            appellation_value=None,
            source_of_product=("domestic",),
        )
    )
    assert finding.status == "needs_review"
    assert (
        finding.message
        == "Appellation not detected with a grape varietal designation."
    )
    assert finding.field == APPELLATION_FIELD
    assert finding.severity == "warning"
    assert finding.evidence == {
        "appellation_present": False,
        "imported": False,
        "approval_status": "pass",
    }


def test_grape_varietals_flags_unknown_domestic_varietals() -> None:
    approved = ADMINISTRATIVELY_APPROVED_VARIETALS[0]
    label_value = f"{approved}, Mystery Grape"
    finding = grape_varietals(
        _context(
            beverage_type="wine",
            grape_varietals_value=label_value,
            appellation_value="Napa Valley",
            source_of_product=("domestic",),
        )
    )
    assert finding.status == "needs_review"
    assert (
        finding.message
        == "Grape varietal designation may not be approved for domestic wine."
    )
    assert finding.field == DEFAULT_FIELD
    assert finding.severity == "warning"
    assert finding.evidence == {
        "appellation_present": True,
        "imported": False,
        "unknown_varietals": ["Mystery Grape"],
        "approval_status": "needs_review",
    }


def test_grape_varietals_combines_missing_appellation_and_unknown_domestic() -> None:
    finding = grape_varietals(
        _context(
            beverage_type="wine",
            grape_varietals_value="Mystery Grape",
            appellation_value=None,
            source_of_product=("domestic",),
        )
    )
    assert finding.status == "needs_review"
    assert (
        finding.message
        == "Appellation not detected with a grape varietal designation, and one or "
        "more varietals may not be approved for domestic wine."
    )
    assert finding.field == DEFAULT_FIELD
    assert finding.severity == "warning"
    assert finding.evidence == {
        "appellation_present": False,
        "imported": False,
        "unknown_varietals": ["Mystery Grape"],
        "approval_status": "needs_review",
    }


def test_grape_varietals_imported_with_appellation_passes() -> None:
    finding = grape_varietals(
        _context(
            beverage_type="wine",
            grape_varietals_value="Mystery Grape",
            appellation_value="Mendoza",
            source_of_product=("imported",),
        )
    )
    assert finding.status == "pass"
    assert (
        finding.message
        == "Imported wines are not restricted to the domestic varietal list."
    )
    assert finding.field == DEFAULT_FIELD
    assert finding.severity == "info"
    assert finding.evidence == {
        "appellation_present": True,
        "imported": True,
        "approval_status": "not_applicable",
    }


def test_grape_varietals_not_evaluated_without_source_of_product() -> None:
    approved = ADMINISTRATIVELY_APPROVED_VARIETALS[0]
    finding = grape_varietals(
        _context(
            beverage_type="wine",
            grape_varietals_value=approved,
            appellation_value="Willamette Valley",
        )
    )
    assert finding.status == "not_evaluated"
    assert (
        finding.message
        == "Source of product not provided; cannot validate varietal approval."
    )
    assert finding.field == DEFAULT_FIELD
    assert finding.severity == "info"
    assert finding.evidence == {
        "appellation_present": True,
        "approval_status": "not_evaluated",
    }
