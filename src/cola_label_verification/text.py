import re


def normalize_text(text: str) -> str:
    """Collapse whitespace for OCR-derived strings.

    Used in OCR extraction to standardize line/span text before storing it on
    `OcrLine`/`OcrSpan` objects, so downstream rules compare consistent text
    even when the OCR output contains irregular spacing.
    """
    return " ".join(text.split())


def normalize_for_match(text: str) -> str:
    """Normalize text for loose matching and deduping.

    Used to dedupe OCR lines (by stripping punctuation/whitespace and casing)
    and to align warning header spans with the extracted warning text, so
    matching is resilient to formatting differences in the label.
    """
    return re.sub(r"[^A-Z0-9]+", "", text.upper())
