import json
import types

import pytest
from PIL import Image

from cola_label_verification import vlm


class DummyTensor:
    def __init__(self) -> None:
        self.device = None

    def to(self, device):
        self.device = device
        return self


class DummyProcessor:
    def __init__(self, decoded: list[str]) -> None:
        self.decoded = decoded
        self.chat_template_args: dict[str, object] | None = None
        self.call_args: dict[str, object] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.chat_template_args = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        return "PROMPT"

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Image.Image],
        return_tensors: str,
    ) -> dict[str, DummyTensor]:
        self.call_args = {
            "text": text,
            "images": images,
            "return_tensors": return_tensors,
        }
        return {
            "input_ids": DummyTensor(),
            "pixel_values": DummyTensor(),
        }

    def batch_decode(
        self,
        sequences,
        *,
        skip_special_tokens: bool,
    ) -> list[str]:
        return list(self.decoded)


class DummyModel:
    def __init__(self) -> None:
        self.device = "cpu"
        self.generate_args: dict[str, object] | None = None

    def generate(self, **kwargs: object):
        self.generate_args = kwargs
        return [[1, 2, 3]]


class DummyImageProcessor:
    def __init__(
        self,
        source: str,
        local_files_only: bool,
        trust_remote_code: bool,
    ) -> None:
        self.source = source
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code

    @classmethod
    def from_pretrained(
        cls,
        source: str,
        *,
        local_files_only: bool,
        trust_remote_code: bool,
    ) -> "DummyImageProcessor":
        return cls(source, local_files_only, trust_remote_code)


class DummyTokenizer:
    def __init__(
        self,
        source: str,
        local_files_only: bool,
        trust_remote_code: bool,
    ) -> None:
        self.source = source
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.chat_template = "template"

    @classmethod
    def from_pretrained(
        cls,
        source: str,
        *,
        local_files_only: bool,
        trust_remote_code: bool,
    ) -> "DummyTokenizer":
        return cls(source, local_files_only, trust_remote_code)


class DummyVLProcessor:
    def __init__(
        self,
        *,
        image_processor: DummyImageProcessor,
        tokenizer: DummyTokenizer,
        video_processor,
        chat_template: str | None,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.chat_template = chat_template


class DummyAutoProcessor:
    last_call: dict[str, object] | None = None

    @classmethod
    def from_pretrained(
        cls,
        source: str,
        *,
        local_files_only: bool,
        trust_remote_code: bool,
    ) -> str:
        cls.last_call = {
            "source": source,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        return "processor"


class DummyTorch:
    def device(self, name: str) -> str:
        return f"device:{name}"


def _sample_image() -> Image.Image:
    return Image.new("RGB", (2, 2), color="white")


def test_extract_json_prefers_last_fenced_payload() -> None:
    text = """
    leading
    ```json
    {"a": 1}
    ```
    middle
    ```json
    {"b": 2}
    ```
    trailing
    """
    assert vlm._extract_json(text) == {"b": 2}


def test_extract_json_uses_last_plain_payload() -> None:
    text = "prefix {\"a\": 1} middle {\"b\": 2}"
    assert vlm._extract_json(text) == {"b": 2}


def test_extract_json_raises_without_payload() -> None:
    with pytest.raises(RuntimeError, match="did not contain JSON"):
        vlm._extract_json("no json here")


def test_normalize_value_handles_structures() -> None:
    assert vlm._normalize_value({"text": "  Label  "}) == "Label"
    assert vlm._normalize_value({"value": 12.5, "unit": "%"}) == "12.5 %"
    assert vlm._normalize_value(["A", " ", 7]) == "A 7"
    assert vlm._normalize_value("n/a") is None
    assert vlm._normalize_value("  ") is None


def test_coerce_float_handles_commas_and_invalid() -> None:
    assert vlm._coerce_float("1,234") == 1234.0
    assert vlm._coerce_float("1,25") == 1.25
    assert vlm._coerce_float("1,234.5") == 1234.5
    assert vlm._coerce_float("bad") is None


def test_parse_qwen_field_value_numeric() -> None:
    raw_value = {"text": "12.5% ABV", "value": "12.5", "unit": "%"}
    parsed = vlm._parse_qwen_field_value(raw_value, numeric=True)
    assert parsed is not None
    assert parsed.text == "12.5% ABV"
    assert parsed.numeric_value == 12.5
    assert parsed.unit == "%"


def test_model_device_prefers_explicit_device() -> None:
    class Model:
        device = "cuda:0"

        def parameters(self):
            raise AssertionError("parameters should not be used")

    assert vlm._model_device(Model(), DummyTorch()) == "cuda:0"


def test_model_device_falls_back_to_parameters_and_cpu() -> None:
    class Param:
        def __init__(self, device: str) -> None:
            self.device = device

    class ModelWithParams:
        def parameters(self):
            return iter([Param("cpu")])

    class EmptyModel:
        def parameters(self):
            return iter(())

    assert vlm._model_device(ModelWithParams(), DummyTorch()) == "cpu"
    assert vlm._model_device(EmptyModel(), DummyTorch()) == "device:cpu"


def test_select_model_class_handles_vl_and_non_vl() -> None:
    class DummyVLConfig:
        pass

    class DummyConfig:
        model_type = "text"

    class AutoVL:
        pass

    class AutoCausal:
        pass

    class AutoImageText:
        pass

    assert (
        vlm._select_model_class(
            DummyVLConfig(),
            AutoVL,
            None,
            AutoCausal,
        )
        is AutoVL
    )
    assert (
        vlm._select_model_class(
            DummyConfig(),
            AutoVL,
            None,
            AutoCausal,
        )
        is AutoCausal
    )
    assert (
        vlm._select_model_class(
            DummyVLConfig(),
            None,
            AutoImageText,
            AutoCausal,
        )
        is AutoImageText
    )
    with pytest.raises(RuntimeError, match="vision-language auto model"):
        vlm._select_model_class(DummyVLConfig(), None, None, AutoCausal)


def test_select_processor_class_prefers_qwen_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Qwen2VLConfig:
        pass

    class Qwen2VLProcessor:
        pass

    monkeypatch.setattr(
        vlm.transformers,
        "Qwen2VLProcessor",
        Qwen2VLProcessor,
        raising=False,
    )
    auto_processor = object()
    assert (
        vlm._select_processor_class(Qwen2VLConfig(), auto_processor)
        is Qwen2VLProcessor
    )


def test_load_qwen_processor_vl_path() -> None:
    class DummyVLConfig:
        pass

    processor = vlm._load_qwen_processor(
        DummyVLConfig(),
        DummyVLProcessor,
        DummyTokenizer,
        DummyImageProcessor,
        "local-model",
        True,
    )
    assert isinstance(processor, DummyVLProcessor)
    assert processor.image_processor.source == "local-model"
    assert processor.tokenizer.source == "local-model"
    assert processor.chat_template == "template"
    with pytest.raises(RuntimeError, match="Video processing is not supported"):
        processor.video_processor.get_number_of_video_patches()


def test_load_qwen_processor_non_vl_uses_auto_processor() -> None:
    class DummyConfig:
        pass

    processor = vlm._load_qwen_processor(
        DummyConfig(),
        DummyAutoProcessor,
        DummyTokenizer,
        DummyImageProcessor,
        "remote-model",
        False,
    )
    assert processor == "processor"
    assert DummyAutoProcessor.last_call == {
        "source": "remote-model",
        "local_files_only": False,
        "trust_remote_code": True,
    }


def test_extract_qwen_field_values_parses_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "brand_name": {"text": "Acme Spirits"},
        "class_type": {"text": "Vodka"},
        "beverage_type": {"value": "wine"},
        "alcohol_content": {
            "text": "12.5% ABV",
            "value": "12.5",
            "unit": "%",
        },
        "net_contents": {"text": "750 mL", "value": "750", "unit": "mL"},
        "percentage_of_foreign_wine": {"value": "1,25", "unit": "%"},
    }
    decoded = [json.dumps(payload)]
    model = DummyModel()
    processor = DummyProcessor(decoded)
    torch_module = types.SimpleNamespace()
    monkeypatch.setattr(
        vlm,
        "preload_qwen_model",
        lambda: (model, processor, torch_module),
    )

    result = vlm.extract_qwen_field_values([_sample_image()])

    assert result.beverage_type == "wine"
    assert result.fields["brand_name"] is not None
    assert result.fields["brand_name"].text == "Acme Spirits"
    assert result.fields["alcohol_content"].numeric_value == 12.5
    assert result.fields["alcohol_content"].unit == "%"
    assert result.fields["net_contents"].numeric_value == 750.0
    assert result.fields["percentage_of_foreign_wine"].numeric_value == 1.25
    assert result.fields["warning_text"] is None
    expected_fields = set(vlm._QWEN_FIELDS) - {"beverage_type"}
    assert set(result.fields.keys()) == expected_fields
    assert processor.chat_template_args == {
        "messages": processor.chat_template_args["messages"],
        "tokenize": False,
        "add_generation_prompt": True,
    }
    assert model.generate_args is not None
    assert model.generate_args.get("max_new_tokens") == vlm.QWEN_MAX_TOKENS
    assert "do_sample" not in model.generate_args


def test_extract_qwen_field_values_raises_on_empty_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DummyModel()
    processor = DummyProcessor([])
    torch_module = types.SimpleNamespace()
    monkeypatch.setattr(
        vlm,
        "preload_qwen_model",
        lambda: (model, processor, torch_module),
    )
    with pytest.raises(RuntimeError, match="response was empty"):
        vlm.extract_qwen_field_values([_sample_image()])
