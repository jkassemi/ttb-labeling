from cola_label_verification.rules.common import build_finding
from cola_label_verification.rules.models import Finding, RuleContext


def net_contents_presence(context: RuleContext) -> Finding:
    label_info = context.label_info
    field = label_info.net_contents
    if not field.value and field.numeric_value is None:
        return build_finding(
            "net_contents_presence",
            "fail",
            "Net contents statement not detected.",
            field="net_contents",
        )
    return build_finding(
        "net_contents_presence",
        "pass",
        "Net contents statement detected.",
        field="net_contents",
        severity="info",
    )
