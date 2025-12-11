"""
Test fixtures for ArcFace TensorRT sandbox.

Provides synthetic face images and mock data for testing.
"""

import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest


def create_synthetic_face_batch(
    n_samples: int = 10,
    height: int = 112,
    width: int = 112,
    seed: int = 42,
) -> np.ndarray:
    """
    Create synthetic face images for testing.

    Args:
        n_samples: Number of samples
        height: Image height (default 112 for ArcFace)
        width: Image width (default 112 for ArcFace)
        seed: Random seed

    Returns:
        BGR images (N, H, W, 3) uint8
    """
    np.random.seed(seed)

    images = np.zeros((n_samples, height, width, 3), dtype=np.uint8)

    for i in range(n_samples):
        # Random skin tone
        base_r = 150 + np.random.randint(-30, 30)
        base_g = 120 + np.random.randint(-30, 30)
        base_b = 100 + np.random.randint(-30, 30)

        # Fill background
        images[i, :, :, 0] = base_b  # B
        images[i, :, :, 1] = base_g  # G
        images[i, :, :, 2] = base_r  # R

        # Add elliptical face region
        cy, cx = height // 2, width // 2
        for y in range(height):
            for x in range(width):
                # Ellipse equation
                ry, rx = height * 0.45, width * 0.35
                if ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 < 1:
                    images[i, y, x] = [base_b + 20, base_g + 20, base_r + 20]

        # Add eye regions
        eye_y = cy - height // 6
        left_eye_x = cx - width // 5
        right_eye_x = cx + width // 5
        eye_size = width // 12

        for ey in range(eye_y - eye_size, eye_y + eye_size):
            for ex in range(left_eye_x - eye_size, left_eye_x + eye_size):
                if 0 <= ey < height and 0 <= ex < width:
                    images[i, ey, ex] = [40, 40, 40]  # Dark

            for ex in range(right_eye_x - eye_size, right_eye_x + eye_size):
                if 0 <= ey < height and 0 <= ex < width:
                    images[i, ey, ex] = [40, 40, 40]

        # Add noise
        noise = np.random.randint(-5, 5, images[i].shape)
        images[i] = np.clip(images[i].astype(np.int32) + noise, 0, 255).astype(np.uint8)

    return images


def create_mock_embeddings(n_samples: int, dim: int = 512, seed: int = 42) -> np.ndarray:
    """
    Create mock embeddings for testing comparison logic.

    Args:
        n_samples: Number of embeddings
        dim: Embedding dimension
        seed: Random seed

    Returns:
        L2-normalized embeddings (N, dim)
    """
    np.random.seed(seed)
    embeddings = np.random.randn(n_samples, dim).astype(np.float32)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    return embeddings


def create_similar_embeddings(
    reference: np.ndarray,
    noise_scale: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """
    Create embeddings similar to reference with controlled noise.

    Args:
        reference: Reference embeddings (N, dim)
        noise_scale: Scale of additive noise
        seed: Random seed

    Returns:
        Similar embeddings (N, dim)
    """
    np.random.seed(seed)
    noise = np.random.randn(*reference.shape).astype(np.float32) * noise_scale
    similar = reference + noise

    # Re-normalize
    norms = np.linalg.norm(similar, axis=1, keepdims=True)
    similar = similar / (norms + 1e-8)

    return similar


# Pytest fixtures

@pytest.fixture
def synthetic_faces() -> np.ndarray:
    """Fixture providing synthetic face images."""
    return create_synthetic_face_batch(n_samples=10)


@pytest.fixture
def synthetic_faces_large() -> np.ndarray:
    """Fixture providing larger batch of synthetic faces."""
    return create_synthetic_face_batch(n_samples=50)


@pytest.fixture
def mock_embeddings() -> np.ndarray:
    """Fixture providing mock embeddings."""
    return create_mock_embeddings(n_samples=10)


@pytest.fixture
def similar_embedding_pair() -> Tuple[np.ndarray, np.ndarray]:
    """Fixture providing two similar embedding sets."""
    ref = create_mock_embeddings(n_samples=10)
    similar = create_similar_embeddings(ref, noise_scale=0.005)
    return ref, similar


@pytest.fixture
def temp_engine_dir(tmp_path) -> Path:
    """Fixture providing temporary directory for engines."""
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    return engine_dir
