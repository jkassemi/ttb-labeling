from cola_label_verification.rules.common import build_finding
from cola_label_verification.rules.models import Finding, RuleContext


def brand_name_presence(context: RuleContext) -> Finding:
    label_info = context.label_info
    label_value = label_info.brand_name.value
    if not label_value:
        return build_finding(
            "brand_name_presence",
            "fail",
            "Brand name not detected on the label.",
            field="brand_name",
        )
    return build_finding(
        "brand_name_presence",
        "pass",
        "Brand name detected.",
        field="brand_name",
        severity="info",
    )
