import json
import logging
import re
import types
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol, cast

import torch
import transformers
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    ProcessorMixin,
)

from cola_label_verification import taxonomy
from cola_label_verification.models import QwenExtractionResult, QwenFieldValue

logger = logging.getLogger(__name__)

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_MAX_TOKENS = 512
QWEN_TEMPERATURE = 0.0
QWEN_TOP_P: float | None = None
AutoModelForVision2Seq: type[PreTrainedModel] | None = getattr(
    transformers, "AutoModelForVision2Seq", None
)
AutoModelForImageTextToText: type[PreTrainedModel] | None = getattr(
    transformers, "AutoModelForImageTextToText", None
)


class QwenProcessor(Protocol):
    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Image.Image],
        return_tensors: str,
    ) -> Mapping[str, torch.Tensor]: ...

    def batch_decode(
        self,
        sequences: Sequence[Sequence[int]] | torch.Tensor,
        *,
        skip_special_tokens: bool,
    ) -> list[str]: ...


class QwenModel(Protocol):
    def generate(self, **kwargs: object) -> torch.Tensor: ...


QwenComponents = tuple[PreTrainedModel, ProcessorMixin, types.ModuleType]

_QWEN_FIELDS = (
    "brand_name",
    "class_type",
    "beverage_type",
    "percentage_of_foreign_wine",
    "alcohol_content",
    "net_contents",
    "name_and_address",
    "warning_text",
    "grape_varietals",
    "appellation_of_origin",
    "country_of_origin",
    "sulfite_declaration",
    "statement_of_composition",
    "commodity_statement_neutral_spirits",
    "commodity_statement_distilled_from",
    "statement_of_age",
    "treatment_with_wood",
    "coloring_materials",
    "fd_and_c_yellow_5",
    "carmine",
    "state_of_distillation",
)
_QWEN_NUMERIC_FIELDS = (
    "alcohol_content",
    "net_contents",
    "percentage_of_foreign_wine",
    "commodity_statement_neutral_spirits",
    "statement_of_age",
)


def _qwen_prompt() -> str:
    """Build the strict JSON-only prompt for Qwen extraction.

    Used by `extract_qwen_field_values` to instruct the model on required keys,
    normalization rules, and class/type options.
    """
    field_list = ", ".join(_QWEN_FIELDS)
    numeric_list = ", ".join(_QWEN_NUMERIC_FIELDS)
    class_options = _qwen_class_type_options()
    class_list = ", ".join(class_options)
    return (
        "You are extracting label text from a single alcohol beverage label. "
        "Return text exactly as it appears on the label (no paraphrasing). "
        "Prefer the most complete instance of each field and include units. "
        "If a field is missing, return null. "
        "Use pattern matching to locate fields (e.g., ABV/proof, mL/oz, government "
        "warning, and origin phrases). "
        "Response format: return ONLY a JSON object (no markdown, no extra text). "
        f"Keys: {field_list}. "
        "Each key maps to either null or an object with keys: text, value, unit. "
        "text must be the exact label text (no paraphrasing). "
        "value must be a number or null; unit must be a short unit string or null. "
        f"Numeric fields: {numeric_list}. "
        "For non-numeric fields, set value and unit to null. "
        "For class_type, set text to the exact label text and set value to one of: "
        f"{class_list}. Set unit to null. "
        "For beverage_type, set text to the exact label text supporting the "
        "category and set value to one of: distilled_spirits, wine. Set unit to null. "
        "If no class option matches, use null. "
        "If beverage type is unclear, use null. "
        "Do not add extra keys."
    )


def _qwen_class_type_options() -> tuple[str, ...]:
    """Collect unique class/type keywords from taxonomy for prompt options."""
    seen: set[str] = set()
    options: list[str] = []
    for keyword in taxonomy.CLASS_KEYWORDS:
        normalized = keyword.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        options.append(keyword)
    return tuple(options)


def _is_vl_config(config: object) -> bool:
    """Detect whether a transformers config represents a vision-language model."""
    class_name = config.__class__.__name__.lower()
    if "vl" in class_name:
        return True
    model_type = getattr(config, "model_type", "")
    return isinstance(model_type, str) and "vl" in model_type


def _select_model_class(
    config: object,
    auto_vl: type[PreTrainedModel] | None,
    auto_image_text: type[PreTrainedModel] | None,
    auto_causal: type[PreTrainedModel],
) -> type[PreTrainedModel]:
    """Choose the correct model class for the given config.

    Used by `preload_qwen_model` to select a VL model when supported,
    falling back to a causal LM class when appropriate.
    """
    is_vl = _is_vl_config(config)
    if is_vl:
        if auto_vl is None:
            if auto_image_text is None:
                raise RuntimeError(
                    "Transformers does not provide a vision-language auto model. "
                    "Install a transformers version that supports Qwen VL."
                )
            return auto_image_text
        return auto_vl
    return auto_causal


def _select_processor_class(
    config: object,
    auto_processor,
):
    """Choose the processor class for the given config.

    Used by `_load_qwen_processor` to prefer Qwen-specific processor classes.
    """
    class_map = {
        "Qwen2_5_VLConfig": "Qwen2_5_VLProcessor",
        "Qwen2VLConfig": "Qwen2VLProcessor",
        "Qwen3VLConfig": "Qwen3VLProcessor",
        "Qwen3VLMoeConfig": "Qwen3VLProcessor",
    }
    class_name = config.__class__.__name__
    if class_name in class_map:
        processor_class = getattr(transformers, class_map[class_name], None)
        if processor_class is not None:
            return processor_class
    return auto_processor


def _build_null_video_processor():
    """Build a video-processor stub for image-only usage.

    Used by `_load_qwen_processor` when a VL processor requires a video
    processor but the app does not support video inputs.
    """
    class _NullVideoProcessor(transformers.BaseVideoProcessor):
        def __init__(self) -> None:
            self.model_input_names = []
            self.merge_size = 1
            self.temporal_patch_size = 1

        def __call__(self, *args, **kwargs):
            raise RuntimeError("Video processing is not supported for Qwen extraction.")

        def get_number_of_video_patches(self, *args, **kwargs):
            raise RuntimeError("Video processing is not supported for Qwen extraction.")

    return _NullVideoProcessor()


def _resolve_local_model_path(model_id: str) -> str | None:
    """Resolve a local Hugging Face cache directory for the model if present.

    Used by `preload_qwen_model` to avoid network access when the model is
    already cached.
    """
    config_path = hf_hub_download(model_id, "config.json", local_files_only=True)
    return str(Path(config_path).parent)


def _load_qwen_processor(
    config: object,
    auto_processor,
    auto_tokenizer,
    auto_image_processor,
    model_source: str,
    local_files_only: bool,
):
    """Load the processor (and image processor/tokenizer when needed).

    Used by `preload_qwen_model` to construct the appropriate processor for
    Qwen VL or causal models.
    """
    processor_class = _select_processor_class(
        config,
        auto_processor,
    )
    if _is_vl_config(config) and processor_class.__name__.endswith("VLProcessor"):
        image_processor = auto_image_processor.from_pretrained(
            model_source,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        tokenizer = auto_tokenizer.from_pretrained(
            model_source,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        chat_template = getattr(tokenizer, "chat_template", None)
        return processor_class(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=_build_null_video_processor(),
            chat_template=chat_template,
        )
    return processor_class.from_pretrained(
        model_source,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )


@lru_cache(maxsize=1)
def preload_qwen_model() -> QwenComponents:
    """Load and cache the Qwen model, processor, and torch module.

    Used by the Gradio app at startup and `extract_qwen_field_values` during
    extraction to avoid repeated initialization.

    Returns:
        Tuple of (model, processor, torch module).
    """
    model_id = QWEN_MODEL_ID
    model_source = _resolve_local_model_path(model_id) or model_id
    local_files_only = model_source != model_id
    config = AutoConfig.from_pretrained(
        model_source,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model_class = _select_model_class(
        config,
        AutoModelForVision2Seq,
        AutoModelForImageTextToText,
        AutoModelForCausalLM,
    )
    device_map = "auto"
    model = model_class.from_pretrained(
        model_source,
        torch_dtype="auto",
        device_map=device_map,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    processor = _load_qwen_processor(
        config,
        AutoProcessor,
        AutoTokenizer,
        AutoImageProcessor,
        model_source,
        local_files_only,
    )
    return model, processor, torch


def _model_device(model, torch_module):
    """Determine the device for a loaded model."""
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch_module.device("cpu")


def _build_messages(
    images: Sequence[Image.Image],
    prompt: str,
) -> list[dict[str, object]]:
    """Build chat messages with images and the prompt.

    Used by `extract_qwen_field_values` before applying the chat template.
    """
    content: list[dict[str, object]] = []
    for image in images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _extract_json(text: str) -> Mapping[str, object]:
    """Extract the last valid JSON object from a model response.

    Used by `extract_qwen_field_values` to parse Qwen output that may include
    extra text or fenced JSON.
    """
    fenced_matches = re.findall(r"```json\\s*(\\{.*?\\})\\s*```", text, re.DOTALL)
    if fenced_matches:
        for snippet in reversed(fenced_matches):
            try:
                payload = json.loads(snippet)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                return payload

    decoder = json.JSONDecoder()
    index = 0
    parsed_objects: list[Mapping[str, object]] = []
    while True:
        index = text.find("{", index)
        if index == -1:
            break
        try:
            payload, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(payload, Mapping):
            parsed_objects.append(payload)
        index += max(end, 1)

    if parsed_objects:
        return parsed_objects[-1]

    logger.error("Qwen response did not contain JSON. Raw response: %r", text)
    raise RuntimeError("Qwen response did not contain JSON.")


def _normalize_value(value: object) -> str | None:
    """Normalize model values into a clean string, if possible.

    Used by `_extract_text_value` and `_normalize_beverage_type_value` to handle
    mixed payload shapes from the model.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        text_value = value.get("text")
        if isinstance(text_value, str):
            cleaned = text_value.strip()
            if cleaned:
                return cleaned
        numeric_value = value.get("value")
        unit_value = value.get("unit")
        if numeric_value is None:
            return None
        if unit_value:
            return f"{numeric_value} {unit_value}".strip() or None
        return str(numeric_value).strip() or None
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return " ".join(parts) if parts else None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned.lower() in {"null", "none", "n/a"}:
            return None
        return cleaned
    return str(value).strip() or None


def _coerce_float(value: object) -> float | None:
    """Coerce numeric-looking inputs into a float when possible.

    Used by `_parse_qwen_field_value` for numeric field extraction.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            cleaned = value.strip()
            if "," in cleaned and "." not in cleaned:
                parts = cleaned.split(",")
                if (
                    len(parts) == 2
                    and parts[0].isdigit()
                    and parts[1].isdigit()
                    and 1 <= len(parts[1]) <= 2
                ):
                    cleaned = f"{parts[0]}.{parts[1]}"
                else:
                    cleaned = cleaned.replace(",", "")
            else:
                cleaned = cleaned.replace(",", "")
            return float(cleaned)
        except ValueError:
            return None
    return None


def _extract_text_value(value: object) -> str | None:
    """Extract a text string from a model value payload.

    Used by `_parse_qwen_field_value` to prioritize explicit text fields.
    """
    if isinstance(value, Mapping):
        text_value = value.get("text")
        if isinstance(text_value, str):
            cleaned = text_value.strip()
            if cleaned:
                return cleaned
    return _normalize_value(value)


def _parse_qwen_field_value(
    raw_value: object,
    *,
    numeric: bool,
) -> QwenFieldValue | None:
    """Parse a Qwen field payload into a normalized value structure.

    Used by `extract_qwen_field_values` for each extracted field.
    """
    text_value = _extract_text_value(raw_value)
    numeric_value: float | None = None
    unit_value: str | None = None
    if numeric:
        source_value = raw_value
        if isinstance(raw_value, Mapping):
            source_value = raw_value.get("value")
            unit_candidate = raw_value.get("unit")
            if isinstance(unit_candidate, str):
                unit_value = unit_candidate.strip() or None
        numeric_value = _coerce_float(source_value)
    if text_value is None and numeric_value is None and unit_value is None:
        return None
    return QwenFieldValue(
        text=text_value,
        numeric_value=numeric_value,
        unit=unit_value,
    )


def _normalize_beverage_type_value(value: object) -> str | None:
    """Normalize the beverage type field value from model output."""
    if isinstance(value, Mapping):
        raw_value = value.get("value")
        if raw_value is not None:
            return _normalize_value(raw_value)
    return _normalize_value(value)


def extract_qwen_field_values(images: Sequence[Image.Image]) -> QwenExtractionResult:
    """Extract structured label fields from images via Qwen.

    Called by the OCR pipeline to augment Paddle OCR output with model-extracted
    label fields.
    """
    model, processor, torch = preload_qwen_model()
    typed_model = cast(QwenModel, model)
    typed_processor = cast(QwenProcessor, processor)
    prompt = _qwen_prompt()
    messages = _build_messages(images, prompt)
    prompt_text = typed_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = cast(
        dict[str, torch.Tensor],
        typed_processor(
            text=[prompt_text],
            images=list(images),
            return_tensors="pt",
        ),
    )
    device = _model_device(model, torch)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generation_args: dict[str, object] = {
        "max_new_tokens": QWEN_MAX_TOKENS,
    }
    if QWEN_TEMPERATURE > 0:
        generation_args.update(
            {
                "do_sample": True,
                "temperature": QWEN_TEMPERATURE,
            }
        )
        if QWEN_TOP_P is not None:
            generation_args["top_p"] = QWEN_TOP_P
    output_ids = typed_model.generate(**inputs, **generation_args)
    decoded = typed_processor.batch_decode(output_ids, skip_special_tokens=True)
    if not decoded:
        raise RuntimeError("Qwen response was empty.")
    payload = _extract_json(decoded[0])
    fields: dict[str, QwenFieldValue | None] = {}
    beverage_type: str | None = None
    for field in _QWEN_FIELDS:
        raw_value = payload.get(field)
        if field == "beverage_type":
            beverage_type = _normalize_beverage_type_value(raw_value)
            continue
        fields[field] = _parse_qwen_field_value(
            raw_value,
            numeric=field in _QWEN_NUMERIC_FIELDS,
        )
    return QwenExtractionResult(fields=fields, beverage_type=beverage_type)
