from cola_label_verification.rules.common import build_finding
from cola_label_verification.rules.models import Finding, RuleContext


def alcohol_content_presence(context: RuleContext) -> Finding:
    label_info = context.label_info
    field = label_info.alcohol_content
    if not field.value and field.numeric_value is None:
        return build_finding(
            "alcohol_content_presence",
            "fail",
            "Alcohol content not detected.",
            field="alcohol_content",
        )
    return build_finding(
        "alcohol_content_presence",
        "pass",
        "Alcohol content detected.",
        field="alcohol_content",
        severity="info",
    )
