"""
Embedding Comparison: TensorRT vs PyTorch.

Compares embeddings from TensorRT and PyTorch implementations
to validate equivalence and measure speedup.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing TensorRT vs PyTorch embeddings."""

    n_samples: int = 0

    # Cosine similarity statistics
    cosine_sim_mean: float = 0.0
    cosine_sim_std: float = 0.0
    cosine_sim_min: float = 0.0
    cosine_sim_max: float = 0.0

    # L2 distance statistics
    l2_dist_mean: float = 0.0
    l2_dist_std: float = 0.0
    l2_dist_max: float = 0.0

    # Timing
    pytorch_time_ms: float = 0.0
    tensorrt_time_ms: float = 0.0
    speedup: float = 0.0

    # Per-sample data (optional)
    cosine_similarities: List[float] = field(default_factory=list)

    # Validation
    passed: bool = False
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "cosine_similarity": {
                "mean": round(self.cosine_sim_mean, 6),
                "std": round(self.cosine_sim_std, 6),
                "min": round(self.cosine_sim_min, 6),
                "max": round(self.cosine_sim_max, 6),
            },
            "l2_distance": {
                "mean": round(self.l2_dist_mean, 6),
                "std": round(self.l2_dist_std, 6),
                "max": round(self.l2_dist_max, 6),
            },
            "timing": {
                "pytorch_ms": round(self.pytorch_time_ms, 2),
                "tensorrt_ms": round(self.tensorrt_time_ms, 2),
                "speedup": round(self.speedup, 2),
            },
            "validation": {
                "passed": self.passed,
                "failure_reason": self.failure_reason,
            },
        }


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between two embedding arrays.

    Args:
        emb1: First embeddings (N, D)
        emb2: Second embeddings (N, D)

    Returns:
        Per-sample cosine similarities (N,)
    """
    # Normalize
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)

    # Dot product
    return np.sum(emb1_norm * emb2_norm, axis=1)


def compute_l2_distance(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """
    Compute L2 distance between two embedding arrays.

    Args:
        emb1: First embeddings (N, D)
        emb2: Second embeddings (N, D)

    Returns:
        Per-sample L2 distances (N,)
    """
    return np.linalg.norm(emb1 - emb2, axis=1)


def generate_synthetic_faces(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic face images for testing.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        Synthetic face images (N, 112, 112, 3) uint8 BGR
    """
    np.random.seed(seed)

    # Create synthetic face-like images
    images = np.zeros((n_samples, 112, 112, 3), dtype=np.uint8)

    for i in range(n_samples):
        # Base skin tone (varies per sample)
        base_color = np.array([140 + np.random.randint(-20, 20)] * 3)

        # Fill with base
        images[i] = base_color

        # Add simple face features
        # Eyes (darker regions)
        images[i, 35:45, 25:45] = base_color - 50
        images[i, 35:45, 67:87] = base_color - 50

        # Nose (slight highlight)
        images[i, 50:70, 50:62] = base_color + 10

        # Mouth (darker)
        images[i, 75:85, 40:72] = base_color - 30

        # Add some noise
        noise = np.random.randint(-10, 10, images[i].shape)
        images[i] = np.clip(images[i].astype(np.int32) + noise, 0, 255).astype(np.uint8)

    return images


def compare_backends(
    images: Optional[np.ndarray] = None,
    n_samples: int = 100,
    tensorrt_engine_path: Optional[Path] = None,
    min_cosine_sim: float = 0.995,
    max_l2_dist: float = 0.1,
    batch_size: int = 32,
) -> ComparisonResult:
    """
    Compare TensorRT vs PyTorch ArcFace embeddings.

    Args:
        images: Face images (N, 112, 112, 3) or None to use synthetic
        n_samples: Number of synthetic samples if images not provided
        tensorrt_engine_path: Path to TensorRT engine
        min_cosine_sim: Minimum acceptable mean cosine similarity
        max_l2_dist: Maximum acceptable mean L2 distance
        batch_size: Processing batch size

    Returns:
        ComparisonResult with statistics
    """
    result = ComparisonResult()

    # Generate or use provided images
    if images is None:
        logger.info(f"Generating {n_samples} synthetic face images")
        images = generate_synthetic_faces(n_samples)
    else:
        n_samples = len(images)

    result.n_samples = n_samples

    # Get PyTorch embeddings
    logger.info("Computing PyTorch reference embeddings...")
    try:
        from .tensorrt_inference import get_pytorch_arcface_embeddings

        start = time.perf_counter()
        pytorch_embeddings = get_pytorch_arcface_embeddings(images)
        result.pytorch_time_ms = (time.perf_counter() - start) * 1000

        if pytorch_embeddings is None:
            result.failure_reason = "Failed to get PyTorch embeddings"
            return result

    except Exception as e:
        result.failure_reason = f"PyTorch embedding error: {e}"
        return result

    # Get TensorRT embeddings
    logger.info("Computing TensorRT embeddings...")
    try:
        from .tensorrt_inference import TensorRTArcFace, run_tensorrt_embeddings

        if tensorrt_engine_path:
            engine = TensorRTArcFace(engine_path=tensorrt_engine_path)
        else:
            engine = TensorRTArcFace()

        start = time.perf_counter()
        tensorrt_embeddings = run_tensorrt_embeddings(engine, images, batch_size)
        result.tensorrt_time_ms = (time.perf_counter() - start) * 1000

    except Exception as e:
        result.failure_reason = f"TensorRT embedding error: {e}"
        return result

    # Compute comparison metrics
    logger.info("Computing comparison metrics...")

    cosine_sims = compute_cosine_similarity(pytorch_embeddings, tensorrt_embeddings)
    l2_dists = compute_l2_distance(pytorch_embeddings, tensorrt_embeddings)

    result.cosine_sim_mean = float(np.mean(cosine_sims))
    result.cosine_sim_std = float(np.std(cosine_sims))
    result.cosine_sim_min = float(np.min(cosine_sims))
    result.cosine_sim_max = float(np.max(cosine_sims))

    result.l2_dist_mean = float(np.mean(l2_dists))
    result.l2_dist_std = float(np.std(l2_dists))
    result.l2_dist_max = float(np.max(l2_dists))

    result.cosine_similarities = cosine_sims.tolist()

    # Compute speedup
    if result.pytorch_time_ms > 0:
        result.speedup = result.pytorch_time_ms / result.tensorrt_time_ms

    # Validation
    if result.cosine_sim_mean < min_cosine_sim:
        result.failure_reason = (
            f"Cosine similarity too low: {result.cosine_sim_mean:.6f} < {min_cosine_sim}"
        )
    elif result.l2_dist_mean > max_l2_dist:
        result.failure_reason = (
            f"L2 distance too high: {result.l2_dist_mean:.6f} > {max_l2_dist}"
        )
    else:
        result.passed = True

    logger.info(f"Comparison complete:")
    logger.info(f"  Cosine similarity: {result.cosine_sim_mean:.6f} (std: {result.cosine_sim_std:.6f})")
    logger.info(f"  L2 distance: {result.l2_dist_mean:.6f}")
    logger.info(f"  Speedup: {result.speedup:.2f}x")
    logger.info(f"  Validation: {'PASSED' if result.passed else 'FAILED'}")

    if result.failure_reason:
        logger.warning(f"  Failure reason: {result.failure_reason}")

    return result


def run_benchmark(
    engine_path: Path,
    n_iterations: int = 100,
    batch_sizes: List[int] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Benchmark TensorRT inference at various batch sizes.

    Args:
        engine_path: Path to TensorRT engine
        n_iterations: Iterations per batch size
        batch_sizes: List of batch sizes to test

    Returns:
        Dict mapping batch_size to (mean_ms, std_ms)
    """
    from .tensorrt_inference import TensorRTArcFace

    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]

    engine = TensorRTArcFace(engine_path=engine_path)
    results = {}

    for bs in batch_sizes:
        logger.info(f"Benchmarking batch size {bs}...")

        # Generate test data
        images = generate_synthetic_faces(bs)

        # Warmup
        for _ in range(3):
            engine.embed(images)

        # Measure
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            engine.embed(images)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        mean_ms = float(np.mean(times))
        std_ms = float(np.std(times))

        results[bs] = (mean_ms, std_ms)
        logger.info(f"  Batch {bs}: {mean_ms:.2f} ms (std: {std_ms:.2f} ms)")

    return results
