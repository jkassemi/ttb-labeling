from cola_label_verification.ocr import FieldCandidate, _field_of_vision_metadata


def _candidate(value: str, *, bbox: list[float], image_index: int) -> FieldCandidate:
    return FieldCandidate(
        value=value,
        confidence=0.9,
        evidence=value,
        normalized={"bbox": bbox, "image_index": image_index},
    )


def test_field_of_vision_pass() -> None:
    candidates = {
        "brand_name": _candidate("Brand", bbox=[100, 100, 300, 200], image_index=0),
        "class_type": _candidate("Vodka", bbox=[120, 220, 320, 280], image_index=0),
        "alcohol_content": _candidate(
            "40% Alc./Vol.", bbox=[140, 300, 280, 340], image_index=0
        ),
    }
    meta = _field_of_vision_metadata(candidates, [(1000, 800)])
    assert meta is not None
    assert meta["status"] == "pass"


def test_field_of_vision_needs_review_for_multiple_images() -> None:
    candidates = {
        "brand_name": _candidate("Brand", bbox=[100, 100, 300, 200], image_index=0),
        "class_type": _candidate("Vodka", bbox=[120, 220, 320, 280], image_index=1),
        "alcohol_content": _candidate(
            "40% Alc./Vol.", bbox=[140, 300, 280, 340], image_index=1
        ),
    }
    meta = _field_of_vision_metadata(candidates, [(1000, 800), (1000, 800)])
    assert meta is not None
    assert meta["status"] == "needs_review"
