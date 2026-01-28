from cola_label_verification.rules.common import presence_rule, require_beverage_type
from cola_label_verification.rules.models import Finding, RuleContext


def treatment_with_wood_presence(context: RuleContext) -> Finding:
    gate = require_beverage_type(
        context,
        allowed={"distilled_spirits"},
        rule_id="treatment_with_wood_presence",
        field="treatment_with_wood",
    )
    if gate is not None:
        return gate
    label_info = context.label_info
    return presence_rule(
        label_info.treatment_with_wood.value,
        rule_id="treatment_with_wood_presence",
        field="treatment_with_wood",
        present_message="Treatment with wood disclosure detected.",
        missing_message=(
            "Treatment with wood disclosure not detected; requirement depends on "
            "production method."
        ),
        required=None,
    )
