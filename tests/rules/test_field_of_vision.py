from cola_label_verification.models import (
    BeverageTypeClassification,
    FieldCandidate,
    FieldExtraction,
    LabelInfo,
)
from cola_label_verification.rules.field_of_vision import (
    _apply_field_of_vision,
    _field_of_vision_metadata,
    field_of_vision_check,
)
from cola_label_verification.rules.models import RuleContext


def _candidate(*, bbox: list[float], image_index: int) -> FieldCandidate:
    return FieldCandidate(
        value="value",
        confidence=0.9,
        evidence="evidence",
        normalized={"bbox": bbox, "image_index": image_index},
    )


def _base_candidates(*, image_index: int = 0) -> dict[str, FieldCandidate | None]:
    return {
        "brand_name": _candidate(bbox=[100, 100, 300, 200], image_index=image_index),
        "class_type": _candidate(bbox=[120, 220, 300, 280], image_index=image_index),
        "alcohol_content": _candidate(
            bbox=[140, 300, 280, 340],
            image_index=image_index,
        ),
    }


def _beverage_type(value: str | None) -> BeverageTypeClassification | None:
    if value is None:
        return None
    return BeverageTypeClassification(
        beverage_type=value,
        confidence=0.9,
        evidence={"source": "test"},
    )


def _context(label_info: LabelInfo) -> RuleContext:
    return RuleContext(label_info=label_info, application_fields=None)


def test_field_of_vision_metadata_passes_when_span_is_narrow() -> None:
    candidates = _base_candidates()
    meta = _field_of_vision_metadata(candidates, [(1000, 800)])
    assert meta is not None
    assert meta["status"] == "pass"
    assert meta["image_index"] == 0
    assert meta["span_ratio"] == 0.2
    assert meta["bbox_union"] == [100.0, 100.0, 300.0, 340.0]


def test_field_of_vision_metadata_needs_review_for_wide_span() -> None:
    candidates = {
        "brand_name": _candidate(bbox=[50, 100, 350, 200], image_index=0),
        "class_type": _candidate(bbox=[400, 220, 700, 280], image_index=0),
        "alcohol_content": _candidate(bbox=[720, 300, 900, 340], image_index=0),
    }
    meta = _field_of_vision_metadata(candidates, [(1000, 800)])
    assert meta is not None
    assert meta["status"] == "needs_review"
    assert meta["span_ratio"] == 0.85


def test_field_of_vision_metadata_needs_review_for_multiple_images() -> None:
    candidates = _base_candidates()
    candidates["class_type"] = _candidate(bbox=[10, 10, 20, 20], image_index=1)
    meta = _field_of_vision_metadata(candidates, [(1000, 800), (1000, 800)])
    assert meta is not None
    assert meta["status"] == "needs_review"
    assert meta["reason"] == "multiple_images"


def test_field_of_vision_metadata_unknown_when_fields_missing() -> None:
    candidates = _base_candidates()
    candidates["class_type"] = None
    meta = _field_of_vision_metadata(candidates, [(1000, 800)])
    assert meta is not None
    assert meta["status"] == "unknown"
    assert meta["reason"] == "missing_fields"


def test_field_of_vision_metadata_unknown_when_bbox_missing() -> None:
    candidates = _base_candidates()
    candidates["brand_name"] = FieldCandidate(
        value="Brand",
        confidence=0.9,
        evidence="Brand",
        normalized={"bbox": [10, 20, 30], "image_index": 0},
    )
    meta = _field_of_vision_metadata(candidates, [(1000, 800)])
    assert meta is not None
    assert meta["status"] == "unknown"
    assert meta["reason"] == "missing_bbox"


def test_field_of_vision_metadata_unknown_when_image_index_invalid() -> None:
    candidates = _base_candidates()
    candidates["brand_name"] = _candidate(bbox=[10, 20, 30, 40], image_index=-1)
    meta = _field_of_vision_metadata(candidates, [(1000, 800)])
    assert meta is not None
    assert meta["status"] == "unknown"
    assert meta["reason"] == "missing_image_index"


def test_field_of_vision_metadata_unknown_when_image_index_out_of_range() -> None:
    candidates = _base_candidates(image_index=2)
    meta = _field_of_vision_metadata(candidates, [(1000, 800)])
    assert meta is not None
    assert meta["status"] == "unknown"
    assert meta["reason"] == "image_index_out_of_range"


def test_field_of_vision_metadata_unknown_when_image_size_invalid() -> None:
    candidates = _base_candidates()
    meta = _field_of_vision_metadata(candidates, [(0, 800)])
    assert meta is not None
    assert meta["status"] == "unknown"
    assert meta["reason"] == "invalid_image_size"


def test_apply_field_of_vision_attaches_metadata_to_key_fields() -> None:
    label_info = LabelInfo(
        brand_name=FieldExtraction(value="Brand", normalized={"source": "qwen"}),
        class_type=FieldExtraction(value="Vodka"),
        alcohol_content=FieldExtraction(value="40%"),
        net_contents=FieldExtraction(value="750 ml", normalized={"keep": True}),
    )
    candidates = _base_candidates()
    updated = _apply_field_of_vision(label_info, candidates, [(1000, 800)])
    field_of_vision = updated.brand_name.normalized["field_of_vision"]
    assert updated.brand_name.normalized["source"] == "qwen"
    assert isinstance(field_of_vision, dict)
    assert updated.class_type.normalized["field_of_vision"] == field_of_vision
    assert updated.alcohol_content.normalized["field_of_vision"] == field_of_vision
    assert updated.net_contents.normalized == {"keep": True}
    assert label_info.brand_name.normalized == {"source": "qwen"}


def test_field_of_vision_check_not_evaluated_without_beverage_type() -> None:
    label_info = LabelInfo()
    finding = field_of_vision_check(_context(label_info))
    assert finding.status == "not_evaluated"
    assert finding.severity == "info"


def test_field_of_vision_check_not_applicable_for_wine() -> None:
    label_info = LabelInfo(beverage_type=_beverage_type("wine"))
    finding = field_of_vision_check(_context(label_info))
    assert finding.status == "not_applicable"
    assert finding.severity == "info"


def test_field_of_vision_check_needs_review_when_metadata_missing() -> None:
    label_info = LabelInfo(beverage_type=_beverage_type("distilled_spirits"))
    finding = field_of_vision_check(_context(label_info))
    assert finding.status == "needs_review"
    assert finding.field == "field_of_vision"


def test_field_of_vision_check_passes_with_pass_status() -> None:
    label_info = LabelInfo(
        beverage_type=_beverage_type("distilled_spirits"),
        brand_name=FieldExtraction(
            normalized={"field_of_vision": {"status": "pass", "span_ratio": 0.2}}
        ),
    )
    finding = field_of_vision_check(_context(label_info))
    assert finding.status == "pass"
    assert finding.severity == "info"
    assert finding.evidence == {"status": "pass", "span_ratio": 0.2}


def test_field_of_vision_check_needs_review_for_needs_review_status() -> None:
    label_info = LabelInfo(
        beverage_type=_beverage_type("distilled_spirits"),
        brand_name=FieldExtraction(
            normalized={"field_of_vision": {"status": "needs_review"}}
        ),
    )
    finding = field_of_vision_check(_context(label_info))
    assert finding.status == "needs_review"
    assert finding.evidence == {"status": "needs_review"}


def test_field_of_vision_check_needs_review_for_unknown_status() -> None:
    label_info = LabelInfo(
        beverage_type=_beverage_type("distilled_spirits"),
        brand_name=FieldExtraction(
            normalized={"field_of_vision": {"status": "unknown"}}
        ),
    )
    finding = field_of_vision_check(_context(label_info))
    assert finding.status == "needs_review"
    assert finding.message == "Field-of-vision status could not be determined."
