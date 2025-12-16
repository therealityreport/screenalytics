from __future__ import annotations

import numpy as np
import pytest

from tools import episode_run


class _DummyPytorchBackend:
    def __init__(self, *args, **kwargs) -> None:
        self._ready = False
        self._resolved_device = "cpu"

    def ensure_ready(self) -> None:
        self._ready = True

    @property
    def resolved_device(self) -> str:
        return self._resolved_device

    def encode(self, crops):
        self.ensure_ready()
        return np.zeros((len(crops), 512), dtype=np.float32)


class _FailingTensorRTBackend:
    def __init__(self, *args, **kwargs) -> None:
        self._resolved_device = "tensorrt"

    def ensure_ready(self) -> None:
        raise ImportError("TensorRT not installed")

    @property
    def resolved_device(self) -> str:
        return self._resolved_device

    def encode(self, crops):
        raise ImportError("TensorRT not installed")


def test_get_embedding_backend_falls_back_to_pytorch_when_tensorrt_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(episode_run, "TensorRTEmbeddingBackend", _FailingTensorRTBackend)
    monkeypatch.setattr(episode_run, "ArcFaceEmbedder", _DummyPytorchBackend)

    embedder = episode_run.get_embedding_backend(
        backend_type="tensorrt",
        device="cpu",
        tensorrt_config="config/pipeline/arcface_tensorrt.yaml",
        allow_cpu_fallback=True,
    )

    embedder.ensure_ready()
    assert embedder.resolved_device == "cpu"
    out = embedder.encode([np.zeros((112, 112, 3), dtype=np.uint8)])
    assert out.shape == (1, 512)


def test_get_embedding_backend_raises_when_fallback_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(episode_run, "TensorRTEmbeddingBackend", _FailingTensorRTBackend)

    embedder = episode_run.get_embedding_backend(
        backend_type="tensorrt",
        device="cpu",
        tensorrt_config="config/pipeline/arcface_tensorrt.yaml",
        allow_cpu_fallback=False,
    )

    with pytest.raises(ImportError):
        embedder.ensure_ready()

