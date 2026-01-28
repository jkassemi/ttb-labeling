from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from cola_label_verification.models import (
    BeverageTypeClassification,
    FieldCandidate,
    FieldExtraction,
    LabelInfo,
    QwenExtractionResult,
    QwenFieldValue,
    TokenVerification,
)

EXPECTED_FIELD_NAMES = (
    "brand_name",
    "class_type",
    "statement_of_composition",
    "grape_varietals",
    "appellation_of_origin",
    "percentage_of_foreign_wine",
    "alcohol_content",
    "net_contents",
    "name_and_address",
    "warning_text",
    "country_of_origin",
    "sulfite_declaration",
    "coloring_materials",
    "fd_and_c_yellow_5",
    "carmine",
    "treatment_with_wood",
    "commodity_statement_neutral_spirits",
    "commodity_statement_distilled_from",
    "state_of_distillation",
    "statement_of_age",
)


def test_field_candidate_frozen_and_payload() -> None:
    candidate = FieldCandidate(
        value="40% Alc./Vol.",
        confidence=0.87,
        evidence="40% Alc./Vol.",
        normalized={"source": "qwen"},
        numeric_value=40.0,
        unit="%",
    )

    assert candidate.value == "40% Alc./Vol."
    assert candidate.confidence == 0.87
    assert candidate.evidence == "40% Alc./Vol."
    assert candidate.normalized == {"source": "qwen"}
    assert candidate.numeric_value == 40.0
    assert candidate.unit == "%"

    with pytest.raises(FrozenInstanceError):
        candidate.value = "42%"


def test_qwen_field_value_and_result_allow_optional_fields() -> None:
    field_value = QwenFieldValue(text="Brand Name", numeric_value=None, unit=None)
    result = QwenExtractionResult(
        fields={
            "brand_name": field_value,
            "net_contents": None,
        },
        beverage_type=None,
    )

    assert result.fields["brand_name"] == field_value
    assert result.fields["net_contents"] is None
    assert result.beverage_type is None


def test_field_extraction_defaults() -> None:
    extraction = FieldExtraction()

    assert extraction.value is None
    assert extraction.numeric_value is None
    assert extraction.unit is None
    assert extraction.confidence is None
    assert extraction.evidence is None
    assert extraction.source == "qwen"
    assert extraction.status == "missing"
    assert extraction.normalized is None


def test_field_extraction_rejects_invalid_literals() -> None:
    with pytest.raises(ValidationError):
        FieldExtraction(source="tesseract")

    with pytest.raises(ValidationError):
        FieldExtraction(status="unknown")


def test_token_verification_rejects_invalid_source() -> None:
    verification = TokenVerification(
        matched=True,
        coverage=1.0,
        token_count=3,
        matched_token_count=3,
        source="span",
    )
    assert verification.source == "span"

    with pytest.raises(ValidationError):
        TokenVerification(
            matched=False,
            coverage=0.0,
            token_count=0,
            matched_token_count=0,
            source="ocr",
        )


def test_beverage_type_classification_validation() -> None:
    classification = BeverageTypeClassification(
        beverage_type="wine",
        confidence=0.77,
        evidence={"source": "test"},
    )

    assert classification.beverage_type == "wine"
    assert classification.confidence == 0.77
    assert classification.evidence == {"source": "test"}

    with pytest.raises(ValidationError):
        BeverageTypeClassification(
            beverage_type="beer",
            confidence=0.5,
            evidence=None,
        )


def test_label_info_defaults_and_fields_are_distinct() -> None:
    label_info = LabelInfo()

    assert label_info.beverage_type is None

    field_names = [
        name
        for name, field in LabelInfo.model_fields.items()
        if field.annotation is FieldExtraction
    ]
    assert tuple(field_names) == EXPECTED_FIELD_NAMES

    for name in field_names:
        field_value = getattr(label_info, name)
        assert isinstance(field_value, FieldExtraction)
        assert field_value.status == "missing"

    instances = {id(getattr(label_info, name)) for name in field_names}
    assert len(instances) == len(field_names)


def test_label_info_is_frozen() -> None:
    label_info = LabelInfo()

    with pytest.raises((TypeError, AttributeError)):
        label_info.brand_name = FieldExtraction(value="New")
