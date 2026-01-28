from cola_label_verification.rules.common import build_finding, require_beverage_type
from cola_label_verification.rules.models import Finding, RuleContext


def wine_designation_present(context: RuleContext) -> Finding:
    gate = require_beverage_type(
        context,
        allowed={"wine"},
        rule_id="wine_designation_presence",
        field="class_type",
    )
    if gate is not None:
        return gate
    label_info = context.label_info
    if (
        label_info.class_type.value
        or label_info.grape_varietals.value
        or label_info.statement_of_composition.value
    ):
        return build_finding(
            "wine_designation_presence",
            "pass",
            "Wine designation detected on the label.",
            field="class_type",
            severity="info",
        )
    return build_finding(
        "wine_designation_presence",
        "fail",
        "Wine designation not detected on the label.",
        field="class_type",
    )
