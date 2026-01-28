from dataclasses import dataclass


@dataclass(frozen=True)
class OcrLine:
    """OCR line with optional confidence."""

    text: str
    confidence: float | None


@dataclass(frozen=True)
class OcrSpan:
    """OCR text span with bounding box metadata."""

    text: str
    confidence: float | None
    bbox: tuple[float, float, float, float]
    image_index: int


@dataclass(frozen=True)
class OcrOptions:
    """Runtime options passed to the OCR backend."""

    use_doc_orientation_classify: bool
    use_doc_unwarping: bool
    use_textline_orientation: bool


DEFAULT_OCR_OPTIONS = OcrOptions(
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True,
)
