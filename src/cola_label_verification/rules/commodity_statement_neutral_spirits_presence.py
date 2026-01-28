from cola_label_verification.rules.common import presence_rule, require_beverage_type
from cola_label_verification.rules.models import Finding, RuleContext


def commodity_statement_neutral_spirits_presence(context: RuleContext) -> Finding:
    gate = require_beverage_type(
        context,
        allowed={"distilled_spirits"},
        rule_id="commodity_statement_neutral_spirits_presence",
        field="commodity_statement_neutral_spirits",
    )
    if gate is not None:
        return gate
    label_info = context.label_info
    return presence_rule(
        label_info.commodity_statement_neutral_spirits.value,
        rule_id="commodity_statement_neutral_spirits_presence",
        field="commodity_statement_neutral_spirits",
        present_message="Neutral spirits commodity statement detected.",
        missing_message=(
            "Neutral spirits commodity statement not detected; requirement depends on "
            "formulation."
        ),
        required=None,
    )
