from collections.abc import Mapping, Sequence

from cola_label_verification.models import FieldCandidate
from cola_label_verification.models import FieldExtraction, LabelInfo
from cola_label_verification.rules.common import build_finding, require_beverage_type
from cola_label_verification.rules.models import Finding, RuleContext


def _field_of_vision_metadata(
    candidates: Mapping[str, FieldCandidate | None],
    image_sizes: Sequence[tuple[int, int]],
) -> dict[str, object] | None:
    required = ("brand_name", "class_type", "alcohol_content")
    spans: list[tuple[float, float, float, float, int]] = []
    for field in required:
        candidate = candidates.get(field)
        if candidate is None or not candidate.normalized:
            return {"status": "unknown", "reason": "missing_fields"}
        bbox = candidate.normalized.get("bbox")
        image_index = candidate.normalized.get("image_index")
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            return {"status": "unknown", "reason": "missing_bbox"}
        if not isinstance(image_index, int) or image_index < 0:
            return {"status": "unknown", "reason": "missing_image_index"}
        spans.append(
            (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
                image_index,
            )
        )

    image_indices = {item[4] for item in spans}
    if len(image_indices) != 1:
        return {"status": "needs_review", "reason": "multiple_images"}
    image_index = next(iter(image_indices))
    if image_index >= len(image_sizes):
        return {"status": "unknown", "reason": "image_index_out_of_range"}
    image_width, _ = image_sizes[image_index]
    if image_width <= 0:
        return {"status": "unknown", "reason": "invalid_image_size"}
    min_x = min(item[0] for item in spans)
    min_y = min(item[1] for item in spans)
    max_x = max(item[2] for item in spans)
    max_y = max(item[3] for item in spans)
    span_ratio = (max_x - min_x) / image_width
    status = "pass" if span_ratio <= 0.4 else "needs_review"
    return {
        "status": status,
        "image_index": image_index,
        "span_ratio": round(span_ratio, 4),
        "bbox_union": [
            round(min_x, 2),
            round(min_y, 2),
            round(max_x, 2),
            round(max_y, 2),
        ],
    }


def _apply_field_of_vision(
    label_info: LabelInfo,
    candidates: Mapping[str, FieldCandidate | None],
    image_sizes: Sequence[tuple[int, int]],
) -> LabelInfo:
    metadata = _field_of_vision_metadata(candidates, image_sizes)
    if metadata is None:
        return label_info

    def update(field: FieldExtraction) -> FieldExtraction:
        normalized: dict[str, object] = {}
        if field.normalized:
            normalized.update(field.normalized)
        normalized["field_of_vision"] = metadata
        return FieldExtraction(
            value=field.value,
            confidence=field.confidence,
            evidence=field.evidence,
            source=field.source,
            status=field.status,
            normalized=normalized,
            numeric_value=field.numeric_value,
            unit=field.unit,
        )

    return LabelInfo(
        brand_name=update(label_info.brand_name),
        class_type=update(label_info.class_type),
        statement_of_composition=label_info.statement_of_composition,
        grape_varietals=label_info.grape_varietals,
        appellation_of_origin=label_info.appellation_of_origin,
        percentage_of_foreign_wine=label_info.percentage_of_foreign_wine,
        alcohol_content=update(label_info.alcohol_content),
        net_contents=label_info.net_contents,
        name_and_address=label_info.name_and_address,
        warning_text=label_info.warning_text,
        country_of_origin=label_info.country_of_origin,
        sulfite_declaration=label_info.sulfite_declaration,
        coloring_materials=label_info.coloring_materials,
        fd_and_c_yellow_5=label_info.fd_and_c_yellow_5,
        carmine=label_info.carmine,
        treatment_with_wood=label_info.treatment_with_wood,
        commodity_statement_neutral_spirits=label_info.commodity_statement_neutral_spirits,
        commodity_statement_distilled_from=label_info.commodity_statement_distilled_from,
        state_of_distillation=label_info.state_of_distillation,
        statement_of_age=label_info.statement_of_age,
        beverage_type=label_info.beverage_type,
    )


def field_of_vision_check(context: RuleContext) -> Finding:
    gate = require_beverage_type(
        context,
        allowed={"distilled_spirits"},
        rule_id="field_of_vision",
        field="field_of_vision",
    )
    if gate is not None:
        return gate
    label_info = context.label_info
    metadata = label_info.brand_name.normalized or {}
    field_of_vision = metadata.get("field_of_vision")
    if not isinstance(field_of_vision, dict):
        return build_finding(
            "field_of_vision",
            "needs_review",
            "Field-of-vision metadata unavailable.",
            field="field_of_vision",
        )
    status = field_of_vision.get("status")
    if status == "pass":
        return build_finding(
            "field_of_vision",
            "pass",
            "Brand, class/type, and alcohol content appear in the same field "
            "of vision.",
            field="field_of_vision",
            severity="info",
            evidence=field_of_vision,
        )
    if status == "needs_review":
        return build_finding(
            "field_of_vision",
            "needs_review",
            "Brand, class/type, and alcohol content may not be in the same "
            "field of vision.",
            field="field_of_vision",
            evidence=field_of_vision,
        )
    return build_finding(
        "field_of_vision",
        "needs_review",
        "Field-of-vision status could not be determined.",
        field="field_of_vision",
        evidence=field_of_vision,
    )
