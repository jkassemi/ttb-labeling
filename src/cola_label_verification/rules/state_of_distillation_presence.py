from cola_label_verification.rules.common import presence_rule, require_beverage_type
from cola_label_verification.rules.models import Finding, RuleContext


def state_of_distillation_presence(context: RuleContext) -> Finding:
    gate = require_beverage_type(
        context,
        allowed={"distilled_spirits"},
        rule_id="state_of_distillation_presence",
        field="state_of_distillation",
    )
    if gate is not None:
        return gate
    label_info = context.label_info
    return presence_rule(
        label_info.state_of_distillation.value,
        rule_id="state_of_distillation_presence",
        field="state_of_distillation",
        present_message="State of distillation statement detected.",
        missing_message=(
            "State of distillation statement not detected; requirement depends on "
            "product type."
        ),
        required=None,
    )
