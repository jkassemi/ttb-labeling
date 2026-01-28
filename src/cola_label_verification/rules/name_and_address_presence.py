from cola_label_verification.rules.common import build_finding
from cola_label_verification.rules.models import Finding, RuleContext


def name_and_address_presence(context: RuleContext) -> Finding:
    label_info = context.label_info
    if label_info.name_and_address.value:
        return build_finding(
            "name_and_address_presence",
            "pass",
            "Name and address statement detected.",
            field="name_and_address",
            severity="info",
        )
    return build_finding(
        "name_and_address_presence",
        "fail",
        "Name and address statement not detected.",
        field="name_and_address",
    )
