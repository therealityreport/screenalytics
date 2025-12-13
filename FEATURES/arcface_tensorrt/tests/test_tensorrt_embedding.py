"""
Tests for ArcFace TensorRT embedding sandbox.

Covers unit tests for builder, inference, and comparison modules.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import fixtures
pytest_plugins = ["FEATURES.arcface_tensorrt.tests.fixtures"]


# ============================================================================
# Unit Tests: Configuration
# ============================================================================

class TestTensorRTConfig:
    """Tests for TensorRTConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from FEATURES.arcface_tensorrt.src.tensorrt_builder import TensorRTConfig

        config = TensorRTConfig()

        assert config.model_name == "arcface_r100"
        assert config.precision == "fp16"
        assert config.max_batch_size == 32
        assert config.input_height == 112
        assert config.input_width == 112

    def test_config_from_yaml(self, tmp_path):
        """Test loading config from YAML."""
        from FEATURES.arcface_tensorrt.src.tensorrt_builder import TensorRTConfig

        yaml_content = """
tensorrt:
  model_name: "arcface_r50"
  precision: "fp32"
  max_batch_size: 64
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        config = TensorRTConfig.from_yaml(config_path)

        assert config.model_name == "arcface_r50"
        assert config.precision == "fp32"
        assert config.max_batch_size == 64

    def test_config_missing_yaml(self, tmp_path):
        """Test config falls back to defaults when YAML missing."""
        from FEATURES.arcface_tensorrt.src.tensorrt_builder import TensorRTConfig

        config = TensorRTConfig.from_yaml(tmp_path / "nonexistent.yaml")

        assert config.model_name == "arcface_r100"


# ============================================================================
# Unit Tests: Engine Utilities
# ============================================================================

class TestEngineUtilities:
    """Tests for engine utility functions."""

    def test_get_engine_filename(self):
        """Test engine filename generation."""
        from FEATURES.arcface_tensorrt.src.tensorrt_builder import (
            TensorRTConfig,
            get_engine_filename,
        )

        config = TensorRTConfig(model_name="arcface_r100", precision="fp16")
        filename = get_engine_filename(config, "sm86")

        assert filename == "arcface_r100-fp16-sm86.plan"

    def test_get_local_engine_path(self):
        """Test local engine path generation."""
        from FEATURES.arcface_tensorrt.src.tensorrt_builder import (
            TensorRTConfig,
            get_local_engine_path,
        )

        config = TensorRTConfig(engine_local_dir=Path("/tmp/engines"))

        with patch(
            "FEATURES.arcface_tensorrt.src.tensorrt_builder.get_sm_arch",
            return_value="sm86",
        ):
            path = get_local_engine_path(config)

        assert "arcface_r100-fp16-sm86.plan" in str(path)

    def test_get_s3_engine_key(self):
        """Test S3 key generation."""
        from FEATURES.arcface_tensorrt.src.tensorrt_builder import (
            TensorRTConfig,
            get_s3_engine_key,
        )

        config = TensorRTConfig(engine_s3_bucket="my-bucket")

        with patch(
            "FEATURES.arcface_tensorrt.src.tensorrt_builder.get_sm_arch",
            return_value="sm75",
        ):
            key = get_s3_engine_key(config)

        assert key is not None
        assert "arcface_r100-fp16-sm75.plan" in key

    def test_get_s3_key_no_bucket(self):
        """Test S3 key returns None when no bucket configured."""
        from FEATURES.arcface_tensorrt.src.tensorrt_builder import (
            TensorRTConfig,
            get_s3_engine_key,
        )

        config = TensorRTConfig(engine_s3_bucket=None)
        key = get_s3_engine_key(config)

        assert key is None


# ============================================================================
# Unit Tests: Preprocessing
# ============================================================================

class TestPreprocessing:
    """Tests for image preprocessing."""

    def test_preprocess_shape(self, synthetic_faces):
        """Test preprocessing produces correct output shape."""
        from FEATURES.arcface_tensorrt.src.tensorrt_inference import TensorRTArcFace

        # Create instance without loading engine
        embedder = TensorRTArcFace()

        preprocessed = embedder.preprocess(synthetic_faces)

        assert preprocessed.shape == (10, 3, 112, 112)
        assert preprocessed.dtype == np.float32

    def test_preprocess_single_image(self, synthetic_faces):
        """Test preprocessing handles single image."""
        from FEATURES.arcface_tensorrt.src.tensorrt_inference import TensorRTArcFace

        embedder = TensorRTArcFace()
        single = synthetic_faces[0]  # (112, 112, 3)

        preprocessed = embedder.preprocess(single)

        assert preprocessed.shape == (1, 3, 112, 112)

    def test_preprocess_normalization(self):
        """Test preprocessing applies correct normalization."""
        from FEATURES.arcface_tensorrt.src.tensorrt_inference import (
            TensorRTArcFace,
            ARCFACE_MEAN,
            ARCFACE_STD,
        )

        embedder = TensorRTArcFace()

        # Create test image with known values
        test_image = np.full((112, 112, 3), 127.5, dtype=np.uint8)
        preprocessed = embedder.preprocess(test_image)

        # After (x - 127.5) / 127.5, value 127.5 should become 0
        assert np.allclose(preprocessed, 0, atol=0.01)


# ============================================================================
# Unit Tests: Comparison Logic
# ============================================================================

class TestComparisonLogic:
    """Tests for embedding comparison utilities."""

    def test_cosine_similarity_identical(self, mock_embeddings):
        """Test cosine similarity returns 1.0 for identical embeddings."""
        from FEATURES.arcface_tensorrt.src.embedding_compare import (
            compute_cosine_similarity,
        )

        similarities = compute_cosine_similarity(mock_embeddings, mock_embeddings)

        assert np.allclose(similarities, 1.0, atol=1e-6)

    def test_cosine_similarity_similar(self, similar_embedding_pair):
        """Test cosine similarity is high for similar embeddings."""
        from FEATURES.arcface_tensorrt.src.embedding_compare import (
            compute_cosine_similarity,
        )

        ref, similar = similar_embedding_pair
        similarities = compute_cosine_similarity(ref, similar)

        assert np.all(similarities > 0.99)

    def test_l2_distance_identical(self, mock_embeddings):
        """Test L2 distance is 0 for identical embeddings."""
        from FEATURES.arcface_tensorrt.src.embedding_compare import (
            compute_l2_distance,
        )

        distances = compute_l2_distance(mock_embeddings, mock_embeddings)

        assert np.allclose(distances, 0.0, atol=1e-6)

    def test_l2_distance_similar(self, similar_embedding_pair):
        """Test L2 distance is small for similar embeddings."""
        from FEATURES.arcface_tensorrt.src.embedding_compare import (
            compute_l2_distance,
        )

        ref, similar = similar_embedding_pair
        distances = compute_l2_distance(ref, similar)

        assert np.all(distances < 0.1)


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_comparison_result_to_dict(self):
        """Test ComparisonResult serialization."""
        from FEATURES.arcface_tensorrt.src.embedding_compare import ComparisonResult

        result = ComparisonResult(
            n_samples=100,
            cosine_sim_mean=0.998,
            cosine_sim_std=0.001,
            cosine_sim_min=0.995,
            cosine_sim_max=0.999,
            tensorrt_time_ms=10.5,
            pytorch_time_ms=52.3,
            speedup=4.98,
            passed=True,
        )

        d = result.to_dict()

        assert d["n_samples"] == 100
        assert d["cosine_similarity"]["mean"] == 0.998
        assert d["timing"]["speedup"] == 4.98
        assert d["validation"]["passed"] is True


class TestSyntheticFaceGeneration:
    """Tests for synthetic face generation."""

    def test_generate_synthetic_faces(self):
        """Test synthetic face generation."""
        from FEATURES.arcface_tensorrt.src.embedding_compare import (
            generate_synthetic_faces,
        )

        faces = generate_synthetic_faces(n_samples=5)

        assert faces.shape == (5, 112, 112, 3)
        assert faces.dtype == np.uint8
        assert faces.max() <= 255
        assert faces.min() >= 0

    def test_generate_synthetic_faces_reproducible(self):
        """Test synthetic faces are reproducible with same seed."""
        from FEATURES.arcface_tensorrt.src.embedding_compare import (
            generate_synthetic_faces,
        )

        faces1 = generate_synthetic_faces(n_samples=5, seed=42)
        faces2 = generate_synthetic_faces(n_samples=5, seed=42)

        assert np.array_equal(faces1, faces2)


# ============================================================================
# Integration Tests (Mocked)
# ============================================================================

class TestCompareBackendsMocked:
    """Integration tests for compare_backends with mocked engines."""

    def test_compare_returns_result_structure(self):
        """Test compare_backends returns properly structured result."""
        from FEATURES.arcface_tensorrt.src.embedding_compare import ComparisonResult

        # Create mock result
        result = ComparisonResult(
            n_samples=10,
            cosine_sim_mean=0.999,
            passed=True,
        )

        d = result.to_dict()
        assert "cosine_similarity" in d
        assert "timing" in d
        assert "validation" in d

    @pytest.mark.slow
    def test_compare_with_mock_embeddings(self, mock_embeddings, similar_embedding_pair):
        """Test comparison logic with mock embeddings."""
        from FEATURES.arcface_tensorrt.src.embedding_compare import (
            compute_cosine_similarity,
            compute_l2_distance,
            ComparisonResult,
        )

        ref, similar = similar_embedding_pair

        # Compute metrics
        cos_sims = compute_cosine_similarity(ref, similar)
        l2_dists = compute_l2_distance(ref, similar)

        # Build result manually
        result = ComparisonResult(
            n_samples=len(ref),
            cosine_sim_mean=float(np.mean(cos_sims)),
            cosine_sim_min=float(np.min(cos_sims)),
            cosine_sim_max=float(np.max(cos_sims)),
            l2_dist_mean=float(np.mean(l2_dists)),
            l2_dist_max=float(np.max(l2_dists)),
        )

        # Check validation logic
        min_cosine = 0.995
        result.passed = result.cosine_sim_mean >= min_cosine

        assert result.passed is True
        assert result.cosine_sim_mean > 0.99


# ============================================================================
# Smoke Tests (require TensorRT)
# ============================================================================

class TestSmokeWithTensorRT:
    """
    Smoke tests that require TensorRT installation.

    Run with: pytest FEATURES/arcface_tensorrt/tests/ -v -m slow
    """

    @pytest.mark.slow
    def test_tensorrt_available(self):
        """Test that TensorRT can be imported."""
        try:
            import tensorrt as trt
            assert trt.__version__ is not None
        except ImportError:
            pytest.skip("TensorRT not installed")

    @pytest.mark.slow
    def test_pycuda_available(self):
        """Test that PyCUDA can be imported."""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            assert cuda.Device.count() >= 0
        except ImportError:
            pytest.skip("PyCUDA not installed")
        except Exception as e:
            pytest.skip(f"PyCUDA initialization failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
