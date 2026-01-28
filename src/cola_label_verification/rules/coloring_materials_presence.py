from cola_label_verification.rules.common import presence_rule, require_beverage_type
from cola_label_verification.rules.models import Finding, RuleContext


def coloring_materials_presence(context: RuleContext) -> Finding:
    gate = require_beverage_type(
        context,
        allowed={"distilled_spirits"},
        rule_id="coloring_materials_presence",
        field="coloring_materials",
    )
    if gate is not None:
        return gate
    label_info = context.label_info
    return presence_rule(
        label_info.coloring_materials.value,
        rule_id="coloring_materials_presence",
        field="coloring_materials",
        present_message="Coloring materials disclosure detected.",
        missing_message=(
            "Coloring materials disclosure not detected; requirement depends on "
            "formulation."
        ),
        required=None,
    )
