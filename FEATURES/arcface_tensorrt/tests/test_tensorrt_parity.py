from __future__ import annotations

import os
from pathlib import Path

import pytest


RUN_TRT_TESTS = os.environ.get("RUN_TRT_TESTS") == "1"
pytestmark = pytest.mark.skipif(not RUN_TRT_TESTS, reason="set RUN_TRT_TESTS=1 to run TensorRT parity tests")


@pytest.mark.slow
def test_tensorrt_parity_against_pytorch_reference() -> None:
    """Parity smoke: compares TensorRT vs PyTorch ArcFace embeddings.

    This is intentionally opt-in because it requires:
    - NVIDIA GPU + TensorRT + PyCUDA installed
    - an existing engine file (set ARCFACE_TRT_ENGINE_PATH)
    - InsightFace models available for the reference backend
    """
    engine_path_raw = os.environ.get("ARCFACE_TRT_ENGINE_PATH")
    if not engine_path_raw:
        pytest.skip("set ARCFACE_TRT_ENGINE_PATH to a built .plan engine to run parity")

    try:
        import tensorrt  # noqa: F401
        import pycuda.driver  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"TensorRT/PyCUDA not installed: {exc}")

    engine_path = Path(engine_path_raw)
    if not engine_path.exists():
        pytest.skip(f"engine not found: {engine_path}")

    from FEATURES.arcface_tensorrt.src.embedding_compare import compare_backends

    result = compare_backends(
        n_samples=8,
        tensorrt_engine_path=engine_path,
        min_cosine_sim=0.995,
        max_l2_dist=0.1,
        batch_size=8,
    )
    if not result.passed:
        pytest.fail(result.failure_reason or "TensorRT parity failed")

