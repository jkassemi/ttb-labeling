from cola_label_verification.rules.common import presence_rule
from cola_label_verification.rules.models import Finding, RuleContext


def carmine_presence(context: RuleContext) -> Finding:
    label_info = context.label_info
    return presence_rule(
        label_info.carmine.value,
        rule_id="carmine_presence",
        field="carmine",
        present_message="Carmine/cochineal disclosure detected.",
        missing_message=(
            "Carmine/cochineal disclosure not detected; requirement depends on "
            "formulation."
        ),
        required=None,
    )
