from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class FieldCandidate:
    """Field candidate extracted from a backend."""

    value: str
    confidence: float | None
    evidence: str | None
    normalized: dict[str, object] | None
    numeric_value: float | None = None
    unit: str | None = None


@dataclass(frozen=True)
class QwenFieldValue:
    """Structured field payload returned by Qwen."""

    text: str | None
    numeric_value: float | None
    unit: str | None


@dataclass(frozen=True)
class QwenExtractionResult:
    """Normalized Qwen response payload."""

    fields: dict[str, QwenFieldValue | None]
    beverage_type: str | None


class FieldExtraction(BaseModel):
    """Single extracted field value with confidence and evidence."""

    model_config = ConfigDict(frozen=True)

    value: str | None = None
    numeric_value: float | None = None
    unit: str | None = None
    confidence: float | None = None
    evidence: str | None = None
    source: Literal["paddle", "qwen"] = "qwen"
    status: Literal["verified", "needs_review", "missing"] = "missing"
    normalized: dict[str, object] | None = None


class TokenVerification(BaseModel):
    """Token verification summary derived from OCR spans."""

    model_config = ConfigDict(frozen=True)

    matched: bool
    coverage: float
    token_count: int
    matched_token_count: int
    source: Literal["span"]


class BeverageTypeClassification(BaseModel):
    """Beverage type prediction inferred from VLM blocks."""

    model_config = ConfigDict(frozen=True)

    beverage_type: Literal["distilled_spirits", "wine"]
    confidence: float
    evidence: dict[str, object] | None


class LabelInfo(BaseModel):
    """Structured label fields extracted from a single application."""

    model_config = ConfigDict(frozen=True)

    brand_name: FieldExtraction = Field(default_factory=FieldExtraction)
    class_type: FieldExtraction = Field(default_factory=FieldExtraction)
    statement_of_composition: FieldExtraction = Field(default_factory=FieldExtraction)
    grape_varietals: FieldExtraction = Field(default_factory=FieldExtraction)
    appellation_of_origin: FieldExtraction = Field(default_factory=FieldExtraction)
    percentage_of_foreign_wine: FieldExtraction = Field(default_factory=FieldExtraction)
    alcohol_content: FieldExtraction = Field(default_factory=FieldExtraction)
    net_contents: FieldExtraction = Field(default_factory=FieldExtraction)
    name_and_address: FieldExtraction = Field(default_factory=FieldExtraction)
    warning_text: FieldExtraction = Field(default_factory=FieldExtraction)
    country_of_origin: FieldExtraction = Field(default_factory=FieldExtraction)
    sulfite_declaration: FieldExtraction = Field(default_factory=FieldExtraction)
    coloring_materials: FieldExtraction = Field(default_factory=FieldExtraction)
    fd_and_c_yellow_5: FieldExtraction = Field(default_factory=FieldExtraction)
    carmine: FieldExtraction = Field(default_factory=FieldExtraction)
    treatment_with_wood: FieldExtraction = Field(default_factory=FieldExtraction)
    commodity_statement_neutral_spirits: FieldExtraction = Field(
        default_factory=FieldExtraction
    )
    commodity_statement_distilled_from: FieldExtraction = Field(
        default_factory=FieldExtraction
    )
    state_of_distillation: FieldExtraction = Field(default_factory=FieldExtraction)
    statement_of_age: FieldExtraction = Field(default_factory=FieldExtraction)
    beverage_type: BeverageTypeClassification | None = None
