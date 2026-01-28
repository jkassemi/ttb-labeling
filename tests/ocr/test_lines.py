from collections.abc import Sequence

from PIL import Image

from cola_label_verification.ocr.lines import (
    _dedupe_lines,
    _extract_structure_lines_and_spans,
    _extract_text_lines,
    _extract_text_lines_and_spans,
    _lines_from_text,
    _polygon_to_bbox,
    _spans_from_structure_json,
)
from cola_label_verification.ocr.types import OcrLine, OcrOptions
from cola_label_verification.text import normalize_for_match


class DummyStructureClient:
    """Minimal stub for the structure OCR client's predict interface."""

    def __init__(self, results: Sequence[object]) -> None:
        self._results = list(results)
        self.calls = 0

    def predict(self, array: object) -> Sequence[object]:
        self.calls += 1
        return self._results


class DummyOcrClient:
    """Minimal stub for text OCR predict calls."""

    def __init__(self, results: Sequence[object]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, object]] = []

    def predict(self, array: object, **kwargs: object) -> Sequence[object]:
        self.calls.append(dict(kwargs))
        return self._results


class ResultWithJsonMethod:
    """Result wrapper that exposes a json() method."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class ResultWithJsonAttr:
    """Result wrapper that exposes a json attribute."""

    def __init__(self, payload: object) -> None:
        self.json = payload


def test_polygon_to_bbox_accepts_multiple_shapes() -> None:
    poly_points = [(0, 1), (2, 1), (2, 3), (0, 3)]
    assert _polygon_to_bbox(poly_points) == (0.0, 1.0, 2.0, 3.0)

    flat = [0, 0, 2, 0, 2, 2, 0, 2]
    assert _polygon_to_bbox(flat) == (0.0, 0.0, 2.0, 2.0)

    rect = [5, 5, 1, 2]
    assert _polygon_to_bbox(rect) == (1.0, 2.0, 5.0, 5.0)

    assert _polygon_to_bbox("bad") is None

    class FakePoly:
        def tolist(self) -> list[tuple[int, int]]:
            return poly_points

    assert _polygon_to_bbox(FakePoly()) == (0.0, 1.0, 2.0, 3.0)


def test_spans_from_structure_json_extracts_valid_entries() -> None:
    payload = {
        "payload": {
            "rec_texts": [" Hello  ", "   ", "World"],
            "rec_polys": [
                [0, 0, 5, 0, 5, 5, 0, 5],
                [0, 0, 1, 0, 1, 1, 0, 1],
                "bad",
            ],
            "rec_scores": ["0.9", "not-a-number", 0.7],
        }
    }

    spans = _spans_from_structure_json(payload, image_index=3)

    assert len(spans) == 1
    span = spans[0]
    assert span.text == "Hello"
    assert span.confidence == 0.9
    assert span.bbox == (0.0, 0.0, 5.0, 5.0)
    assert span.image_index == 3


def test_extract_structure_lines_and_spans_handles_json_payloads() -> None:
    image = Image.new("L", (4, 4), 0)
    results = [
        {
            "rec_texts": [" Foo "],
            "rec_polys": [[(0, 0), (2, 0), (2, 2), (0, 2)]],
            "rec_scores": [0.4],
        },
        ResultWithJsonMethod(
            {"rec_texts": ["Bar"], "rec_polys": [[0, 0, 1, 0, 1, 1, 0, 1]]}
        ),
        ResultWithJsonAttr(
            {"rec_texts": ["   "], "rec_polys": [[0, 0, 1, 0, 1, 1, 0, 1]]}
        ),
        ResultWithJsonAttr("not-a-dict"),
    ]
    client = DummyStructureClient(results)

    lines, spans = _extract_structure_lines_and_spans([image], client)

    assert client.calls == 1
    assert [line.text for line in lines] == ["Foo", "Bar"]
    assert [span.text for span in spans] == ["Foo", "Bar"]
    assert spans[0].confidence == 0.4
    assert spans[1].confidence is None
    assert all(span.image_index == 0 for span in spans)


def test_extract_text_lines_and_spans_mixed_page_shapes() -> None:
    page_list = [
        ([(0, 0), (10, 0), (10, 10), (0, 10)], [" Foo  ", 0.8]),
        ("not-a-polygon", ["Bar", "0.4"]),
        ([0, 0, 5, 0, 5, 5, 0, 5], ["   ", 0.9]),
        ("ignored", "not-a-list"),
    ]
    page_dict = {
        "rec_texts": ["Baz", " Quux "],
        "rec_scores": ["0.5", "bad"],
        "rec_polys": [[1, 1, 4, 1, 4, 4, 1, 4], "bad"],
    }
    client = DummyOcrClient([page_list, page_dict])
    options = OcrOptions(
        use_doc_orientation_classify=False,
        use_doc_unwarping=True,
        use_textline_orientation=False,
    )
    image = Image.new("L", (12, 12), 0)

    lines, spans = _extract_text_lines_and_spans([image], client, options)

    assert client.calls == [
        {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": True,
            "use_textline_orientation": False,
        }
    ]
    assert [(line.text, line.confidence) for line in lines] == [
        ("Foo", 0.8),
        ("Bar", 0.4),
        ("Baz", 0.5),
        ("Quux", None),
    ]
    assert [(span.text, span.bbox) for span in spans] == [
        ("Foo", (0.0, 0.0, 10.0, 10.0)),
        ("Baz", (1.0, 1.0, 4.0, 4.0)),
    ]
    assert all(span.image_index == 0 for span in spans)


def test_extract_text_lines_returns_lines_only() -> None:
    page_list = [([(0, 0), (1, 0), (1, 1), (0, 1)], ["Solo", 0.7])]
    client = DummyOcrClient([page_list])
    options = OcrOptions(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
    )
    image = Image.new("RGB", (2, 2), 0)

    lines = _extract_text_lines([image], client, options)

    assert [(line.text, line.confidence) for line in lines] == [("Solo", 0.7)]


def test_lines_from_text_strips_and_ignores_blanks() -> None:
    text = " Foo \n\nBar\n   \n Baz  "

    lines = _lines_from_text(text)

    assert [line.text for line in lines] == ["Foo", "Bar", "Baz"]
    assert all(line.confidence is None for line in lines)


def test_dedupe_lines_prefers_best_confidence() -> None:
    lines = [
        OcrLine(text="Foo!", confidence=0.1),
        OcrLine(text="foo", confidence=0.3),
        OcrLine(text="FOO", confidence=0.3),
        OcrLine(text="!!!", confidence=0.9),
        OcrLine(text="Bar", confidence=None),
    ]

    deduped = _dedupe_lines(lines)
    deduped_by_key = {normalize_for_match(line.text): line for line in deduped}

    assert set(deduped_by_key) == {"FOO", "BAR"}
    assert deduped_by_key["FOO"].text == "FOO"
    assert deduped_by_key["FOO"].confidence == 0.3
    assert deduped_by_key["BAR"].text == "Bar"
