from typing import Literal

from cola_label_verification.rules.models import (
    ApplicationFields,
    Finding,
    FindingSeverity,
    FindingStatus,
    RuleContext,
)

BeverageType = Literal["distilled_spirits", "wine"]


def resolve_beverage_type(context: RuleContext) -> BeverageType | None:
    prediction = context.label_info.beverage_type
    if prediction is None:
        return None
    return prediction.beverage_type


def build_finding(
    rule_id: str,
    status: FindingStatus,
    message: str,
    *,
    severity: FindingSeverity = "warning",
    field: str | None = None,
    evidence: dict[str, object] | None = None,
) -> Finding:
    return Finding(
        rule_id=rule_id,
        status=status,
        message=message,
        severity=severity,
        field=field,
        evidence=evidence,
    )


def require_beverage_type(
    context: RuleContext,
    *,
    allowed: set[BeverageType],
    rule_id: str,
    field: str,
) -> Finding | None:
    beverage_type = resolve_beverage_type(context)
    if beverage_type is None:
        return build_finding(
            rule_id,
            "not_evaluated",
            "Beverage type not selected; rule not evaluated.",
            field=field,
            severity="info",
        )
    if beverage_type not in allowed:
        return build_finding(
            rule_id,
            "not_applicable",
            "Rule not applicable to the selected beverage type.",
            field=field,
            severity="info",
        )
    return None


def presence_rule(
    value: str | None,
    *,
    rule_id: str,
    field: str,
    present_message: str,
    missing_message: str,
    required: bool | None = True,
    not_applicable_message: str | None = None,
    not_evaluated_message: str | None = None,
) -> Finding:
    if value:
        return build_finding(
            rule_id,
            "pass",
            present_message,
            field=field,
            severity="info",
        )
    if required is True:
        return build_finding(rule_id, "fail", missing_message, field=field)
    if required is False:
        return build_finding(
            rule_id,
            "not_applicable",
            not_applicable_message or missing_message,
            field=field,
            severity="info",
        )
    return build_finding(
        rule_id,
        "not_evaluated",
        not_evaluated_message or missing_message,
        field=field,
        severity="info",
    )


def is_imported(application_fields: ApplicationFields | None) -> bool | None:
    if application_fields is None or not application_fields.source_of_product:
        return None
    normalized = {item.strip().lower() for item in application_fields.source_of_product}
    if "imported" in normalized:
        return True
    if "domestic" in normalized:
        return False
    return None
