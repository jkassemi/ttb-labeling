from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from cola_label_verification.models import LabelInfo

if TYPE_CHECKING:
    from PIL import Image

    from cola_label_verification.ocr.types import OcrSpan


FindingStatus = Literal[
    "pass",
    "fail",
    "needs_review",
    "not_applicable",
    "not_evaluated",
]
FindingSeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class ApplicationFields:
    """Optional application fields used for context-specific checks."""

    beverage_type: Literal["distilled_spirits", "wine"] | None = None
    brand_name: str | None = None
    class_type: str | None = None
    alcohol_content: str | None = None
    net_contents: str | None = None
    name_and_address: str | None = None
    warning_text: str | None = None
    grape_varietals: str | None = None
    appellation_of_origin: str | None = None
    source_of_product: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RulesConfig:
    """Configuration for checklist evaluation."""

    verification_threshold: float = 0.65


@dataclass(frozen=True)
class RuleContext:
    """Inputs passed to checklist rules."""

    label_info: LabelInfo
    application_fields: ApplicationFields | None
    rules_config: RulesConfig = field(default_factory=RulesConfig)
    spans: Sequence["OcrSpan"] | None = None
    images: Sequence["Image.Image"] | None = None


@dataclass(frozen=True)
class Finding:
    """Single checklist outcome."""

    rule_id: str
    status: FindingStatus
    message: str
    severity: FindingSeverity
    field: str | None = None
    evidence: dict[str, object] | None = None


@dataclass(frozen=True)
class ChecklistResult:
    """Checklist evaluation output."""

    findings: list[Finding]
