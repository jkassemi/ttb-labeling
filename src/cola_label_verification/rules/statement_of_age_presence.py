from cola_label_verification.rules.common import presence_rule, require_beverage_type
from cola_label_verification.rules.models import Finding, RuleContext


def statement_of_age_presence(context: RuleContext) -> Finding:
    gate = require_beverage_type(
        context,
        allowed={"distilled_spirits"},
        rule_id="statement_of_age_presence",
        field="statement_of_age",
    )
    if gate is not None:
        return gate
    label_info = context.label_info
    return presence_rule(
        label_info.statement_of_age.value,
        rule_id="statement_of_age_presence",
        field="statement_of_age",
        present_message="Statement of age detected.",
        missing_message=(
            "Statement of age not detected; requirement depends on product type and "
            "aging."
        ),
        required=None,
    )
