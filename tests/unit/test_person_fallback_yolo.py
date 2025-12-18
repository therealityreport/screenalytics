from __future__ import annotations

import builtins
import sys
import types

import numpy as np

from tools import episode_run


class _FakeMPS:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeCUDA:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _install_fake_torch(monkeypatch, *, mps_available: bool, cuda_available: bool = False) -> None:
    fake_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=_FakeMPS(mps_available)),
        cuda=_FakeCUDA(cuda_available),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_person_fallback_coreml_device_is_never_passed_to_torch(monkeypatch):
    monkeypatch.setattr(episode_run, "PERSON_FALLBACK_ENABLED", True)
    _install_fake_torch(monkeypatch, mps_available=True)
    detector = episode_run._build_person_fallback_detector("coreml")
    assert detector is not None
    assert detector.device != "coreml"
    assert detector.device == "mps"


def test_person_fallback_records_disabled_reason_when_ultralytics_missing(monkeypatch):
    monkeypatch.setattr(episode_run, "PERSON_FALLBACK_ENABLED", True)

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "ultralytics":
            raise ModuleNotFoundError("No module named 'ultralytics'", name="ultralytics")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    detector = episode_run._build_person_fallback_detector("cpu")
    assert detector is not None
    out = detector.detect_persons(np.zeros((32, 32, 3), dtype=np.uint8))
    assert out == []
    assert detector.load_status == "error"
    assert detector.load_error == "ultralytics_missing"

