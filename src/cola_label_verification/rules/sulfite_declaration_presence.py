from cola_label_verification.rules.common import presence_rule
from cola_label_verification.rules.models import Finding, RuleContext


def sulfite_declaration_presence(context: RuleContext) -> Finding:
    label_info = context.label_info
    return presence_rule(
        label_info.sulfite_declaration.value,
        rule_id="sulfite_declaration_presence",
        field="sulfite_declaration",
        present_message="Sulfite declaration detected.",
        missing_message=(
            "Sulfite declaration not detected; requirement depends on formulation."
        ),
        required=None,
    )
