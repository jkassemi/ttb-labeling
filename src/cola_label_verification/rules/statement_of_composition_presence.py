from cola_label_verification.rules.common import presence_rule, require_beverage_type
from cola_label_verification.rules.models import Finding, RuleContext


def statement_of_composition_presence(context: RuleContext) -> Finding:
    gate = require_beverage_type(
        context,
        allowed={"distilled_spirits"},
        rule_id="statement_of_composition_presence",
        field="statement_of_composition",
    )
    if gate is not None:
        return gate
    label_info = context.label_info
    return presence_rule(
        label_info.statement_of_composition.value,
        rule_id="statement_of_composition_presence",
        field="statement_of_composition",
        present_message="Statement of composition detected.",
        missing_message=(
            "Statement of composition not detected; required for some distinctive "
            "or fanciful designations."
        ),
        required=None,
    )
