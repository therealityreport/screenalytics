from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("numpy")

from tools import episode_run


def test_init_retinaface_missing_weights_fails(monkeypatch):
    fake_model_zoo = types.SimpleNamespace(get_model=lambda name: None)
    fake_insightface = types.SimpleNamespace(model_zoo=fake_model_zoo)
    monkeypatch.setitem(sys.modules, "insightface", fake_insightface)
    monkeypatch.setitem(sys.modules, "insightface.model_zoo", fake_model_zoo)

    with pytest.raises(RuntimeError) as excinfo:
        episode_run._init_retinaface("retinaface_unit_test_missing", "cpu")

    assert "RetinaFace weights" in str(excinfo.value)
