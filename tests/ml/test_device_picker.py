from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "episode_run.py"
MODULE_NAME = "episode_run_device_test"
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = module
spec.loader.exec_module(module)  # type: ignore[arg-type]


def test_pick_device_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        import torch  # type: ignore

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    except Exception:  # pragma: no cover - torch missing
        pass

    assert module.pick_device("cpu") == "cpu"
    assert module.pick_device("auto") == "cpu"
