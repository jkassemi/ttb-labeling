from cola_label_verification.rules.common import presence_rule
from cola_label_verification.rules.models import Finding, RuleContext


def fd_and_c_yellow_5_presence(context: RuleContext) -> Finding:
    label_info = context.label_info
    return presence_rule(
        label_info.fd_and_c_yellow_5.value,
        rule_id="fd_and_c_yellow_5_presence",
        field="fd_and_c_yellow_5",
        present_message="FD&C Yellow #5 disclosure detected.",
        missing_message=(
            "FD&C Yellow #5 disclosure not detected; requirement depends on "
            "formulation."
        ),
        required=None,
    )
