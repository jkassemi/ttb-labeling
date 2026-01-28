from cola_label_verification.rules.common import is_imported, presence_rule
from cola_label_verification.rules.models import Finding, RuleContext


def country_of_origin_presence(context: RuleContext) -> Finding:
    label_info = context.label_info
    required = is_imported(context.application_fields)
    return presence_rule(
        label_info.country_of_origin.value,
        rule_id="country_of_origin_presence",
        field="country_of_origin",
        present_message="Country of origin statement detected.",
        missing_message="Country of origin statement not detected.",
        required=required,
        not_applicable_message=(
            "Country of origin statement not required for domestic products."
        ),
        not_evaluated_message=(
            "Source of product not provided; cannot determine if country of origin "
            "statement is required."
        ),
    )
