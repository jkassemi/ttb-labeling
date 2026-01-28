from cola_label_verification.rules.common import build_finding, require_beverage_type
from cola_label_verification.rules.models import Finding, RuleContext


def appellation_presence(context: RuleContext) -> Finding:
    gate = require_beverage_type(
        context,
        allowed={"wine"},
        rule_id="appellation_presence",
        field="appellation_of_origin",
    )
    if gate is not None:
        return gate
    label_info = context.label_info
    label_value = label_info.appellation_of_origin.value
    if not label_value:
        return build_finding(
            "appellation_presence",
            "fail",
            "Appellation not detected on the label.",
            field="appellation_of_origin",
        )
    return build_finding(
        "appellation_presence",
        "pass",
        "Appellation detected on the label.",
        field="appellation_of_origin",
        severity="info",
    )
