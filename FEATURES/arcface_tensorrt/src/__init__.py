"""
ArcFace TensorRT Embedding Sandbox

TensorRT-accelerated ArcFace embeddings for high-throughput face recognition.
Compares against PyTorch reference implementation for validation.

Usage:
    # Build TensorRT engine from ONNX
    python -m FEATURES.arcface_tensorrt --mode build

    # Compare TensorRT vs PyTorch embeddings
    python -m FEATURES.arcface_tensorrt --mode compare --n-samples 100

    # Run inference only
    python -m FEATURES.arcface_tensorrt --mode inference --input crops/

Artifacts:
    - engines/{model_name}-{sm_arch}.plan - TensorRT engine file
    - comparison_results.json - Backend comparison metrics
"""

from .tensorrt_builder import (
    build_or_load_engine,
    TensorRTConfig,
    get_sm_arch,
)
from .tensorrt_inference import (
    TensorRTArcFace,
    run_tensorrt_embeddings,
)
from .embedding_compare import (
    compare_backends,
    ComparisonResult,
)

__all__ = [
    # Builder
    "build_or_load_engine",
    "TensorRTConfig",
    "get_sm_arch",
    # Inference
    "TensorRTArcFace",
    "run_tensorrt_embeddings",
    # Comparison
    "compare_backends",
    "ComparisonResult",
]
