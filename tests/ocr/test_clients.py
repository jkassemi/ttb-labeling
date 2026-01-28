import threading
import time
from typing import Any

import pytest

from cola_label_verification.ocr import clients as ocr_clients


class _RecordingOcrBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[object, bool, bool, bool]] = []

    def predict(
        self,
        image: object,
        *,
        use_doc_orientation_classify: bool,
        use_doc_unwarping: bool,
        use_textline_orientation: bool,
    ) -> list[dict[str, object]]:
        self.calls.append(
            (
                image,
                use_doc_orientation_classify,
                use_doc_unwarping,
                use_textline_orientation,
            )
        )
        return [{"ok": True}]


class _BlockingOcrBackend:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.started = threading.Event()
        self.release = threading.Event()

    def predict(
        self,
        image: object,
        *,
        use_doc_orientation_classify: bool,
        use_doc_unwarping: bool,
        use_textline_orientation: bool,
    ) -> list[dict[str, object]]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if not self.started.is_set():
            self.started.set()
            self.release.wait(timeout=1.0)
        self.active -= 1
        return [{"image": image}]


class _FlakyOcrBackend:
    def __init__(self) -> None:
        self.calls = 0

    def predict(
        self,
        image: object,
        *,
        use_doc_orientation_classify: bool,
        use_doc_unwarping: bool,
        use_textline_orientation: bool,
    ) -> list[dict[str, object]]:
        self.calls += 1
        if self.calls == 1:
            raise ValueError("backend failure")
        return [{"image": image}]


class _RecordingStructureBackend:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def predict(self, input: object) -> list[object]:
        self.calls.append(input)
        return [input]


class _DummyCuda:
    def __init__(self, count: int) -> None:
        self._count = count

    def device_count(self) -> int:
        return self._count


class _DummyDevice:
    def __init__(self, count: int) -> None:
        self.cuda = _DummyCuda(count)


class _DummyPaddle:
    def __init__(self, compiled: bool, count: int) -> None:
        self._compiled = compiled
        self.device = _DummyDevice(count)

    def is_compiled_with_cuda(self) -> bool:
        return self._compiled


class _CapturingFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return {"kwargs": kwargs}


def _threadsafe_predict(
    client: ocr_clients.ThreadSafeOcrClient,
    event: threading.Event,
    results: list[object],
) -> None:
    event.set()
    results.append(
        client.predict(
            "image",
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
    )


def test_thread_safe_ocr_client_passes_through_calls() -> None:
    backend = _RecordingOcrBackend()
    client = ocr_clients.ThreadSafeOcrClient(backend)

    result = client.predict(
        "image",
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )

    assert result == [{"ok": True}]
    assert backend.calls == [("image", True, False, True)]


def test_thread_safe_ocr_client_serializes_calls() -> None:
    backend = _BlockingOcrBackend()
    client = ocr_clients.ThreadSafeOcrClient(backend)

    thread2_ready = threading.Event()

    thread1 = threading.Thread(
        target=client.predict,
        args=("image",),
        kwargs={
            "use_doc_orientation_classify": True,
            "use_doc_unwarping": True,
            "use_textline_orientation": False,
        },
    )

    results: list[object] = []
    thread2 = threading.Thread(
        target=_threadsafe_predict,
        args=(client, thread2_ready, results),
    )

    thread1.start()
    assert backend.started.wait(timeout=1.0)

    thread2.start()
    assert thread2_ready.wait(timeout=1.0)

    time.sleep(0.05)

    assert backend.max_active == 1

    backend.release.set()

    thread1.join(timeout=1.0)
    thread2.join(timeout=1.0)

    assert not thread1.is_alive()
    assert not thread2.is_alive()
    assert results == [[{"image": "image"}]]


def test_thread_safe_ocr_client_releases_lock_on_failure() -> None:
    backend = _FlakyOcrBackend()
    client = ocr_clients.ThreadSafeOcrClient(backend)

    with pytest.raises(ValueError, match="backend failure"):
        client.predict(
            "image",
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
        )

    done = threading.Event()
    results: list[object] = []

    def _invoke() -> None:
        try:
            results.append(
                client.predict(
                    "image",
                    use_doc_orientation_classify=True,
                    use_doc_unwarping=True,
                    use_textline_orientation=True,
                )
            )
        finally:
            done.set()

    thread = threading.Thread(target=_invoke, daemon=True)
    thread.start()

    assert done.wait(timeout=1.0)
    assert results == [[{"image": "image"}]]


def test_thread_safe_structure_client_passes_through_calls() -> None:
    backend = _RecordingStructureBackend()
    client = ocr_clients.ThreadSafeStructureClient(backend)

    result = client.predict({"layout": "payload"})

    assert result == [{"layout": "payload"}]
    assert backend.calls == [{"layout": "payload"}]


@pytest.mark.parametrize(
    ("compiled", "count", "expected"),
    [
        (True, 1, "gpu"),
        (True, 0, "cpu"),
        (False, 3, "cpu"),
    ],
)
def test_choose_device(
    compiled: bool,
    count: int,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _DummyPaddle(compiled, count)
    monkeypatch.setattr(ocr_clients, "paddle", dummy)

    assert ocr_clients._choose_device() == expected


@pytest.mark.parametrize(
    ("device", "expected_mkldnn"),
    [
        ("cpu", False),
        ("gpu", True),
    ],
)
def test_create_paddle_ocr_client_sets_flags(
    device: str,
    expected_mkldnn: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _CapturingFactory()
    monkeypatch.setattr(ocr_clients, "PaddleOCR", factory)

    client = ocr_clients._create_paddle_ocr_client(device)

    assert client == {"kwargs": factory.calls[0]}
    assert factory.calls == [
        {
            "lang": "en",
            "device": device,
            "enable_mkldnn": expected_mkldnn,
            "enable_cinn": False,
        }
    ]


def test_create_structure_client_sets_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _CapturingFactory()
    monkeypatch.setattr(ocr_clients, "PPStructureV3", factory)

    client = ocr_clients._create_structure_client("gpu")

    assert client == {"kwargs": factory.calls[0]}
    assert factory.calls == [
        {
            "device": "gpu",
            "use_doc_orientation_classify": True,
            "use_doc_unwarping": True,
            "use_textline_orientation": True,
        }
    ]


def test_default_ocr_client_caches_and_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    ocr_clients._reset_default_ocr_client_cache()

    created: list[object] = []

    def _fake_choose_device() -> str:
        return "cpu"

    def _fake_create(device: str) -> object:
        created.append({"device": device, "id": len(created)})
        return created[-1]

    monkeypatch.setattr(ocr_clients, "_choose_device", _fake_choose_device)
    monkeypatch.setattr(ocr_clients, "_create_paddle_ocr_client", _fake_create)

    first = ocr_clients._get_default_ocr_client()
    second = ocr_clients._get_default_ocr_client()

    assert first is second
    assert isinstance(first, ocr_clients.ThreadSafeOcrClient)
    assert first._inner is created[0]

    ocr_clients._reset_default_ocr_client_cache()

    third = ocr_clients._get_default_ocr_client()

    assert third is not first
    assert third._inner is created[1]

    ocr_clients._reset_default_ocr_client_cache()


def test_default_structure_client_caches_and_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    ocr_clients._reset_default_structure_client_cache()

    created: list[object] = []

    def _fake_choose_device() -> str:
        return "gpu"

    def _fake_create(device: str) -> object:
        created.append({"device": device, "id": len(created)})
        return created[-1]

    monkeypatch.setattr(ocr_clients, "_choose_device", _fake_choose_device)
    monkeypatch.setattr(ocr_clients, "_create_structure_client", _fake_create)

    first = ocr_clients._get_default_structure_client()
    second = ocr_clients._get_default_structure_client()

    assert first is second
    assert isinstance(first, ocr_clients.ThreadSafeStructureClient)
    assert first._inner is created[0]

    ocr_clients._reset_default_structure_client_cache()

    third = ocr_clients._get_default_structure_client()

    assert third is not first
    assert third._inner is created[1]

    ocr_clients._reset_default_structure_client_cache()
