import re

from cola_label_verification import taxonomy
from cola_label_verification.rules.common import build_finding, resolve_beverage_type
from cola_label_verification.rules.models import Finding, RuleContext

_CLASS_TYPE_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")
_ABV_LINE_RE = re.compile(
    r"\b\d{1,3}(?:\.\d+)?\s*%?\s*.*\b(?:ALC|ALCOHOL|ABV|VOL)\b",
    re.IGNORECASE,
)
_NET_CONTENTS_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*"
    r"(m\s*l|ml|milliliter|milliliters|l|liter|liters|fl\.?\s*oz|oz)\b",
    re.IGNORECASE,
)
_CLASS_TYPE_RE = re.compile(
    r"\b("
    + "|".join(re.escape(keyword) for keyword in taxonomy.CLASS_KEYWORDS)
    + r")\b",
    re.IGNORECASE,
)
_NAME_ADDRESS_PREFIX_RE = re.compile(
    r"\b(BOTTLED|DISTILLED|PRODUCED|IMPORTED|MANUFACTURED|RECTIFIED)\s+BY\b",
    re.IGNORECASE,
)
_CLASS_TYPE_STOP_RE = re.compile(
    r"\b(ALC|ALCOHOL|ABV|PROOF|BY\s+VOL|VOL\.?|NET\s+CONTENTS?)\b",
    re.IGNORECASE,
)
_COUNTRY_ORIGIN_RE = re.compile(
    r"\b(PRODUCT OF|MADE IN|PRODUCED IN|IMPORTED FROM)\b",
    re.IGNORECASE,
)
_CLASS_TYPE_STOP_PATTERNS = (
    _ABV_LINE_RE,
    _NET_CONTENTS_RE,
    _COUNTRY_ORIGIN_RE,
    _NAME_ADDRESS_PREFIX_RE,
    _CLASS_TYPE_STOP_RE,
)


def _clean_class_type_value(value: str) -> str:
    cleaned = _CLASS_TYPE_EDGE_RE.sub("", value).strip()
    return cleaned


def _trim_class_type_value(value: str) -> str:
    match = _CLASS_TYPE_RE.search(value)
    if not match:
        return _clean_class_type_value(value)
    start = match.start()
    stop = len(value)
    for pattern in _CLASS_TYPE_STOP_PATTERNS:
        for stop_match in pattern.finditer(value):
            if stop_match.start() > start:
                stop = min(stop, stop_match.start())
                break
    candidate = _clean_class_type_value(value[:stop])
    return candidate or _clean_class_type_value(value)


def class_type_presence(context: RuleContext) -> Finding:
    beverage_type = resolve_beverage_type(context)
    if beverage_type is None:
        return build_finding(
            "class_type_presence",
            "not_evaluated",
            "Beverage type not selected; class/type not evaluated.",
            field="class_type",
            severity="info",
        )
    if beverage_type not in {"distilled_spirits", "wine"}:
        return build_finding(
            "class_type_presence",
            "not_applicable",
            "Class/type rule not applicable to the selected beverage type.",
            field="class_type",
            severity="info",
        )
    label_info = context.label_info
    label_value = label_info.class_type.value
    normalized_label_value = (
        _trim_class_type_value(label_value) if label_value else label_value
    )
    if beverage_type == "wine":
        if not normalized_label_value:
            if (
                label_info.grape_varietals.value
                or label_info.statement_of_composition.value
            ):
                return build_finding(
                    "class_type_presence",
                    "not_applicable",
                    "Class/type designation omitted; varietal or composition serves as "
                    "designation.",
                    field="class_type",
                    severity="info",
                )
            return build_finding(
                "class_type_presence",
                "fail",
                "Class/type designation not detected.",
                field="class_type",
            )
    else:
        if not normalized_label_value:
            return build_finding(
                "class_type_presence",
                "fail",
                "Class/type designation not detected.",
                field="class_type",
            )
    return build_finding(
        "class_type_presence",
        "pass",
        "Class/type designation detected.",
        field="class_type",
        severity="info",
    )
