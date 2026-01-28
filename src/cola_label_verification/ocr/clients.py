from collections.abc import Mapping, Sequence
from functools import lru_cache
from threading import Lock

import paddle
from paddleocr import PaddleOCR, PPStructureV3


class ThreadSafeOcrClient:
    """Serialize OCR calls for backends that are not thread-safe."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self._lock = Lock()

    def predict(
        self,
        image: object,
        *,
        use_doc_orientation_classify: bool,
        use_doc_unwarping: bool,
        use_textline_orientation: bool,
    ) -> Sequence[Mapping[str, object]]:
        with self._lock:
            return self._inner.predict(
                image,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping,
                use_textline_orientation=use_textline_orientation,
            )


class ThreadSafeStructureClient:
    """Serialize layout calls for backends that are not thread-safe."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self._lock = Lock()

    def predict(self, input: object) -> Sequence[object]:
        with self._lock:
            return self._inner.predict(input)


def _choose_device() -> str:
    if paddle.is_compiled_with_cuda():
        if paddle.device.cuda.device_count() > 0:
            return "gpu"
    return "cpu"


def _create_paddle_ocr_client(device: str) -> PaddleOCR:
    return PaddleOCR(
        lang="en",
        device=device,
        enable_mkldnn=device != "cpu",
        enable_cinn=False,
    )


def _create_structure_client(device: str) -> PPStructureV3:
    return PPStructureV3(
        device=device,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
    )


@lru_cache(maxsize=1)
def _get_default_ocr_client() -> ThreadSafeOcrClient:
    device = _choose_device()
    return ThreadSafeOcrClient(_create_paddle_ocr_client(device))


def _reset_default_ocr_client_cache() -> None:
    _get_default_ocr_client.cache_clear()


@lru_cache(maxsize=1)
def _get_default_structure_client() -> ThreadSafeStructureClient:
    device = _choose_device()
    return ThreadSafeStructureClient(_create_structure_client(device))


def _reset_default_structure_client_cache() -> None:
    _get_default_structure_client.cache_clear()
