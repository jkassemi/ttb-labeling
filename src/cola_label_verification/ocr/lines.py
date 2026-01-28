from collections.abc import Iterable, Mapping, Sequence

from PIL import Image

from cola_label_verification.ocr.image_variants import _iter_image_variants
from cola_label_verification.ocr.types import OcrLine, OcrOptions, OcrSpan
from cola_label_verification.text import normalize_for_match, normalize_text


def _predict_array(client: object, image: Image.Image) -> Sequence[object]:
    import numpy as np

    if image.mode != "RGB":
        image = image.convert("RGB")
    return client.predict(np.array(image))


def _select_polys(
    texts: Sequence[object],
    *candidates: Sequence[object],
) -> Sequence[object]:
    for candidate in candidates:
        if isinstance(candidate, Sequence) and len(candidate) == len(texts):
            return candidate
    return []


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return []


def _result_to_json(result: object) -> dict[str, object] | None:
    if isinstance(result, dict):
        return result
    json_attr = getattr(result, "json", None)
    if callable(json_attr):
        payload = json_attr()
    else:
        payload = json_attr
    if isinstance(payload, dict):
        return payload
    return None


def _polygon_to_bbox(poly: object) -> tuple[float, float, float, float] | None:
    if isinstance(poly, (str, bytes)):
        return None

    tolist = getattr(poly, "tolist", None)
    if callable(tolist):
        poly = tolist()

    if not isinstance(poly, Sequence):
        return None

    if len(poly) == 4:
        points: list[Sequence[object]] = []
        for point in poly:
            if isinstance(point, (str, bytes)):
                points = []
                break
            point_tolist = getattr(point, "tolist", None)
            if callable(point_tolist):
                point = point_tolist()
            if not isinstance(point, Sequence):
                points = []
                break
            points.append(point)
        if points:
            xs = [float(point[0]) for point in points if len(point) >= 2]
            ys = [float(point[1]) for point in points if len(point) >= 2]
            if not xs or not ys:
                return None
            return min(xs), min(ys), max(xs), max(ys)

    try:
        values = [float(value) for value in poly]
    except (TypeError, ValueError):
        return None
    if len(values) == 8:
        xs = [values[index] for index in range(0, 8, 2)]
        ys = [values[index] for index in range(1, 8, 2)]
        return min(xs), min(ys), max(xs), max(ys)
    if len(values) == 4:
        x0, y0, x1, y1 = values
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    return None


def _iter_ocr_groups(
    data: object,
) -> Iterable[tuple[list[str], list[float | None], list[object]]]:
    if isinstance(data, dict):
        texts = data.get("rec_texts")
        polys = _select_polys(
            texts or [],
            _as_sequence(data.get("rec_polys")),
            _as_sequence(data.get("rec_boxes")),
            _as_sequence(data.get("dt_polys")),
            _as_sequence(data.get("dt_boxes")),
        )
        scores = data.get("rec_scores")
        if (
            isinstance(texts, list)
            and isinstance(polys, list)
            and len(texts) == len(polys)
        ):
            score_list: list[float | None] = []
            if isinstance(scores, list) and len(scores) == len(texts):
                for value in scores:
                    try:
                        score_list.append(float(value))
                    except (TypeError, ValueError):
                        score_list.append(None)
            else:
                score_list = [None] * len(texts)
            return [(texts, score_list, polys)]
        groups: list[tuple[list[str], list[float | None], list[object]]] = []
        for value in data.values():
            groups.extend(_iter_ocr_groups(value))
        return groups
    if isinstance(data, list):
        groups: list[tuple[list[str], list[float | None], list[object]]] = []
        for item in data:
            groups.extend(_iter_ocr_groups(item))
        return groups
    return []


def _spans_from_structure_json(
    data: Mapping[str, object],
    *,
    image_index: int,
) -> list[OcrSpan]:
    spans: list[OcrSpan] = []
    groups = _iter_ocr_groups(data)
    for texts, scores, polys in groups:
        for text, score, poly in zip(texts, scores, polys, strict=False):
            bbox = _polygon_to_bbox(poly)
            if bbox is None:
                continue
            value = normalize_text(str(text))
            if not value:
                continue
            spans.append(
                OcrSpan(
                    text=value,
                    confidence=score,
                    bbox=bbox,
                    image_index=image_index,
                )
            )
    return spans


def _extract_structure_lines_and_spans(
    images: Sequence[Image.Image],
    structure_client: object,
) -> tuple[list[OcrLine], list[OcrSpan]]:
    lines: list[OcrLine] = []
    spans: list[OcrSpan] = []
    for index, image in enumerate(images):
        results = _predict_array(structure_client, image)
        for result in results:
            payload = _result_to_json(result)
            if not payload:
                continue
            new_spans = _spans_from_structure_json(payload, image_index=index)
            spans.extend(new_spans)
            lines.extend(
                [
                    OcrLine(text=span.text, confidence=span.confidence)
                    for span in new_spans
                ]
            )
    return lines, spans


def _lines_from_text(text: str) -> list[OcrLine]:
    lines: list[OcrLine] = []
    for raw_line in text.splitlines():
        cleaned = raw_line.strip()
        if cleaned:
            lines.append(OcrLine(text=cleaned, confidence=None))
    return lines


def _extract_text_lines_and_spans(
    images: Sequence[Image.Image],
    ocr_client: object,
    options: OcrOptions,
    *,
    enhance_images: bool = False,
    geometry_safe: bool = False,
) -> tuple[list[OcrLine], list[OcrSpan]]:
    import numpy as np

    lines: list[OcrLine] = []
    spans: list[OcrSpan] = []
    for image_index, image in enumerate(images):
        for variant in _iter_image_variants(
            image,
            enhance=enhance_images,
            geometry_safe=geometry_safe,
        ):
            if variant.mode != "RGB":
                variant = variant.convert("RGB")
            result = ocr_client.predict(
                np.array(variant),
                use_doc_orientation_classify=options.use_doc_orientation_classify,
                use_doc_unwarping=options.use_doc_unwarping,
                use_textline_orientation=options.use_textline_orientation,
            )
            for page in result:
                if isinstance(page, list):
                    for item in page:
                        if not isinstance(item, (list, tuple)) or len(item) < 2:
                            continue
                        bbox = _polygon_to_bbox(item[0])
                        text_info = item[1]
                        if not isinstance(text_info, (list, tuple)) or not text_info:
                            continue
                        normalized = normalize_text(str(text_info[0]))
                        score = None
                        if len(text_info) > 1:
                            try:
                                score = float(text_info[1])
                            except (TypeError, ValueError):
                                score = None
                        if normalized:
                            lines.append(OcrLine(text=normalized, confidence=score))
                        if bbox is not None and normalized:
                            spans.append(
                                OcrSpan(
                                    text=normalized,
                                    confidence=score,
                                    bbox=bbox,
                                    image_index=image_index,
                                )
                            )
                    continue
                raw_texts = page.get("rec_texts", [])
                raw_scores = page.get("rec_scores", [])
                raw_polys = _select_polys(
                    raw_texts if isinstance(raw_texts, Sequence) else [],
                    _as_sequence(page.get("rec_polys")),
                    _as_sequence(page.get("rec_boxes")),
                    _as_sequence(page.get("dt_polys")),
                    _as_sequence(page.get("dt_boxes")),
                )
                if not isinstance(raw_texts, Sequence):
                    continue
                polys: Sequence[object] = (
                    raw_polys if isinstance(raw_polys, Sequence) else []
                )
                scores: Sequence[object] = (
                    raw_scores if isinstance(raw_scores, Sequence) else []
                )
                for index, line in enumerate(raw_texts):
                    normalized = normalize_text(str(line))
                    if normalized:
                        score = None
                        if index < len(scores):
                            raw_score = scores[index]
                            if isinstance(raw_score, (int, float, str)):
                                try:
                                    score = float(raw_score)
                                except ValueError:
                                    score = None
                        lines.append(OcrLine(text=normalized, confidence=score))
                    if index < len(polys):
                        bbox = _polygon_to_bbox(polys[index])
                        if bbox is not None and normalized:
                            spans.append(
                                OcrSpan(
                                    text=normalized,
                                    confidence=score,
                                    bbox=bbox,
                                    image_index=image_index,
                                )
                            )
    return lines, spans


def _extract_text_lines(
    images: Sequence[Image.Image],
    ocr_client: object,
    options: OcrOptions,
    *,
    enhance_images: bool = False,
    geometry_safe: bool = False,
) -> list[OcrLine]:
    lines, _ = _extract_text_lines_and_spans(
        images,
        ocr_client,
        options,
        enhance_images=enhance_images,
        geometry_safe=geometry_safe,
    )
    return lines


def _dedupe_lines(lines: Sequence[OcrLine]) -> list[OcrLine]:
    deduped: dict[str, OcrLine] = {}
    for line in lines:
        key = normalize_for_match(line.text)
        if not key:
            continue
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = line
            continue
        existing_score = existing.confidence or 0.0
        incoming_score = line.confidence or 0.0
        if incoming_score >= existing_score:
            deduped[key] = line
    return list(deduped.values())
