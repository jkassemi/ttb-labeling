from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from cola_label_verification.models import LabelInfo
from cola_label_verification.rules.alcohol_content_presence import (
    alcohol_content_presence,
)
from cola_label_verification.rules.appellation_presence import appellation_presence
from cola_label_verification.rules.beverage_type_presence import (
    beverage_type_presence,
)
from cola_label_verification.rules.brand_name_presence import brand_name_presence
from cola_label_verification.rules.carmine_presence import carmine_presence
from cola_label_verification.rules.class_type_presence import class_type_presence
from cola_label_verification.rules.coloring_materials_presence import (
    coloring_materials_presence,
)
from cola_label_verification.rules.commodity_statement_distilled_from_presence import (
    commodity_statement_distilled_from_presence,
)
from cola_label_verification.rules.commodity_statement_neutral_spirits_presence import (
    commodity_statement_neutral_spirits_presence,
)
from cola_label_verification.rules.country_of_origin_presence import (
    country_of_origin_presence,
)
from cola_label_verification.rules.fd_and_c_yellow_5_presence import (
    fd_and_c_yellow_5_presence,
)
from cola_label_verification.rules.field_of_vision import field_of_vision_check
from cola_label_verification.rules.grape_varietals import grape_varietals
from cola_label_verification.rules.models import (
    ApplicationFields,
    ChecklistResult,
    Finding,
    RuleContext,
    RulesConfig,
)
from cola_label_verification.rules.name_and_address_presence import (
    name_and_address_presence,
)
from cola_label_verification.rules.net_contents_presence import net_contents_presence
from cola_label_verification.rules.percentage_of_foreign_wine_presence import (
    percentage_of_foreign_wine_presence,
)
from cola_label_verification.rules.state_of_distillation_presence import (
    state_of_distillation_presence,
)
from cola_label_verification.rules.statement_of_age_presence import (
    statement_of_age_presence,
)
from cola_label_verification.rules.statement_of_composition_presence import (
    statement_of_composition_presence,
)
from cola_label_verification.rules.sulfite_declaration_presence import (
    sulfite_declaration_presence,
)
from cola_label_verification.rules.treatment_with_wood_presence import (
    treatment_with_wood_presence,
)
from cola_label_verification.rules.warning_text import warning_text
from cola_label_verification.rules.wine_designation_present import (
    wine_designation_present,
)

if TYPE_CHECKING:
    from PIL import Image

    from cola_label_verification.ocr.types import OcrSpan

RuleFn = Callable[[RuleContext], Finding]

COMMON_RULES: tuple[RuleFn, ...] = (
    beverage_type_presence,
    warning_text,
    name_and_address_presence,
    brand_name_presence,
    class_type_presence,
    alcohol_content_presence,
    net_contents_presence,
    country_of_origin_presence,
    sulfite_declaration_presence,
    fd_and_c_yellow_5_presence,
    carmine_presence,
)

DISTILLED_SPIRITS_RULES: tuple[RuleFn, ...] = (
    field_of_vision_check,
    statement_of_composition_presence,
    coloring_materials_presence,
    treatment_with_wood_presence,
    commodity_statement_neutral_spirits_presence,
    commodity_statement_distilled_from_presence,
    state_of_distillation_presence,
    statement_of_age_presence,
)

WINE_RULES: tuple[RuleFn, ...] = (
    wine_designation_present,
    grape_varietals,
    appellation_presence,
    percentage_of_foreign_wine_presence,
)

ALL_RULES: tuple[RuleFn, ...] = COMMON_RULES + DISTILLED_SPIRITS_RULES + WINE_RULES


def evaluate_checklist(
    label_info: LabelInfo,
    *,
    application_fields: ApplicationFields | None = None,
    rules_config: RulesConfig | None = None,
    images: Sequence["Image.Image"] | None = None,
    spans: Sequence["OcrSpan"] | None = None,
) -> ChecklistResult:
    """Evaluate checklist rules for a label and optional application fields."""
    resolved_config = rules_config or RulesConfig()
    context = RuleContext(
        label_info=label_info,
        application_fields=application_fields,
        rules_config=resolved_config,
        spans=spans,
        images=images,
    )
    findings = [rule(context) for rule in ALL_RULES]
    return ChecklistResult(findings=findings)
