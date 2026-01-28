from cola_label_verification.rules.common import presence_rule, require_beverage_type
from cola_label_verification.rules.models import Finding, RuleContext


def percentage_of_foreign_wine_presence(context: RuleContext) -> Finding:
    gate = require_beverage_type(
        context,
        allowed={"wine"},
        rule_id="percentage_of_foreign_wine_presence",
        field="percentage_of_foreign_wine",
    )
    if gate is not None:
        return gate
    label_info = context.label_info
    return presence_rule(
        label_info.percentage_of_foreign_wine.value,
        rule_id="percentage_of_foreign_wine_presence",
        field="percentage_of_foreign_wine",
        present_message="Percentage of foreign wine statement detected.",
        missing_message=(
            "Percentage of foreign wine statement not detected; requirement depends on "
            "labeling."
        ),
        required=None,
    )
