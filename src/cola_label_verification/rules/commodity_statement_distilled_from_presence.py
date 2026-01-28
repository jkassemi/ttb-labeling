from cola_label_verification.rules.common import presence_rule, require_beverage_type
from cola_label_verification.rules.models import Finding, RuleContext


def commodity_statement_distilled_from_presence(context: RuleContext) -> Finding:
    gate = require_beverage_type(
        context,
        allowed={"distilled_spirits"},
        rule_id="commodity_statement_distilled_from_presence",
        field="commodity_statement_distilled_from",
    )
    if gate is not None:
        return gate
    label_info = context.label_info
    return presence_rule(
        label_info.commodity_statement_distilled_from.value,
        rule_id="commodity_statement_distilled_from_presence",
        field="commodity_statement_distilled_from",
        present_message="Distilled-from commodity statement detected.",
        missing_message=(
            "Distilled-from commodity statement not detected; requirement depends on "
            "formulation."
        ),
        required=None,
    )
