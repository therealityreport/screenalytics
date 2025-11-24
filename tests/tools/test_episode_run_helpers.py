"""
Unit tests for helper functions in tools/episode_run.py

Tests cover:
- _embedding_is_valid: Embedding validation with shape, dtype, NaN/inf, and norm checks
- _analyze_pose_expression: Pose/expression extraction (currently returns None values)
"""

import numpy as np
import pytest


def test_embedding_is_valid_with_valid_512d_embedding():
    """Test that a valid normalized 512D embedding passes all checks."""
    from tools.episode_run import _embedding_is_valid
    
    # Create a valid normalized embedding
    embedding = np.random.randn(512).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)  # Normalize to unit norm
    
    assert _embedding_is_valid(embedding, expected_dim=512) is True


def test_embedding_is_valid_with_none():
    """Test that None embedding is rejected."""
    from tools.episode_run import _embedding_is_valid
    
    assert _embedding_is_valid(None) is False


def test_embedding_is_valid_with_wrong_type():
    """Test that non-numpy array is rejected."""
    from tools.episode_run import _embedding_is_valid
    
    # List instead of numpy array
    embedding = [0.1] * 512
    assert _embedding_is_valid(embedding) is False


def test_embedding_is_valid_with_wrong_ndim():
    """Test that 2D array is rejected (expecting 1D)."""
    from tools.episode_run import _embedding_is_valid
    
    # 2D array instead of 1D
    embedding = np.random.randn(512, 1).astype(np.float32)
    assert _embedding_is_valid(embedding) is False


def test_embedding_is_valid_with_wrong_dimension():
    """Test that wrong dimension (256 instead of 512) is rejected."""
    from tools.episode_run import _embedding_is_valid
    
    # 256D instead of 512D
    embedding = np.random.randn(256).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    assert _embedding_is_valid(embedding, expected_dim=512) is False


def test_embedding_is_valid_with_nan_values():
    """Test that embedding containing NaN is rejected."""
    from tools.episode_run import _embedding_is_valid
    
    embedding = np.random.randn(512).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    embedding[100] = np.nan  # Inject NaN
    
    assert _embedding_is_valid(embedding) is False


def test_embedding_is_valid_with_inf_values():
    """Test that embedding containing inf is rejected."""
    from tools.episode_run import _embedding_is_valid
    
    embedding = np.random.randn(512).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    embedding[100] = np.inf  # Inject inf
    
    assert _embedding_is_valid(embedding) is False


def test_embedding_is_valid_with_bad_norm_too_low():
    """Test that embedding with L2 norm < 0.9 is rejected."""
    from tools.episode_run import _embedding_is_valid
    
    # Create embedding with very small norm
    embedding = np.random.randn(512).astype(np.float32) * 0.01
    assert np.linalg.norm(embedding) < 0.9
    
    assert _embedding_is_valid(embedding) is False


def test_embedding_is_valid_with_bad_norm_too_high():
    """Test that embedding with L2 norm > 1.1 is rejected."""
    from tools.episode_run import _embedding_is_valid
    
    # Create embedding with large norm
    embedding = np.random.randn(512).astype(np.float32) * 10
    assert np.linalg.norm(embedding) > 1.1
    
    assert _embedding_is_valid(embedding) is False


def test_embedding_is_valid_with_custom_dimension():
    """Test that custom expected_dim parameter works correctly."""
    from tools.episode_run import _embedding_is_valid
    
    # Create a valid normalized 128D embedding
    embedding = np.random.randn(128).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    
    assert _embedding_is_valid(embedding, expected_dim=128) is True
    assert _embedding_is_valid(embedding, expected_dim=512) is False  # Wrong dim


def test_embedding_is_valid_with_zero_vector():
    """Test that zero vector is rejected (norm would be 0)."""
    from tools.episode_run import _embedding_is_valid
    
    embedding = np.zeros(512, dtype=np.float32)
    assert _embedding_is_valid(embedding) is False


def test_analyze_pose_expression_returns_none_tuple():
    """Test that _analyze_pose_expression returns (None, None, None)."""
    from tools.episode_run import _analyze_pose_expression
    
    # Test with dummy landmarks (currently unused)
    landmarks = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    
    yaw, pitch, expression = _analyze_pose_expression(landmarks)
    
    assert yaw is None
    assert pitch is None
    assert expression is None


def test_analyze_pose_expression_with_none_landmarks():
    """Test that _analyze_pose_expression handles None landmarks gracefully."""
    from tools.episode_run import _analyze_pose_expression
    
    # Should still return (None, None, None) even with None input
    yaw, pitch, expression = _analyze_pose_expression(None)
    
    assert yaw is None
    assert pitch is None
    assert expression is None


def test_analyze_pose_expression_with_various_inputs():
    """Test that _analyze_pose_expression always returns None regardless of input."""
    from tools.episode_run import _analyze_pose_expression
    
    # Test with different input types
    test_inputs = [
        None,
        [],
        np.array([]),
        np.array([[0, 0], [1, 1]]),
        {"landmark1": [0, 0], "landmark2": [1, 1]},
        "dummy_landmarks",
        12345,
    ]
    
    for landmarks in test_inputs:
        yaw, pitch, expression = _analyze_pose_expression(landmarks)
        assert yaw is None, f"Failed for input: {landmarks}"
        assert pitch is None, f"Failed for input: {landmarks}"
        assert expression is None, f"Failed for input: {landmarks}"


def test_parse_track_id_with_integer():
    """Test parsing track_id from integer."""
    from tools.episode_run import _parse_track_id
    
    assert _parse_track_id(42) == 42
    assert _parse_track_id(0) == 0
    assert _parse_track_id(12345) == 12345


def test_parse_track_id_with_track_string():
    """Test parsing track_id from 'track-XXXXX' string format."""
    from tools.episode_run import _parse_track_id
    
    assert _parse_track_id("track-00001") == 1
    assert _parse_track_id("track-00042") == 42
    assert _parse_track_id("track-12345") == 12345


def test_parse_track_id_with_numeric_string():
    """Test parsing track_id from plain numeric string."""
    from tools.episode_run import _parse_track_id
    
    assert _parse_track_id("42") == 42
    assert _parse_track_id("0") == 0
    assert _parse_track_id("12345") == 12345


def test_parse_track_id_with_invalid_format():
    """Test that invalid track_id formats raise ValueError."""
    from tools.episode_run import _parse_track_id
    import pytest
    
    with pytest.raises(ValueError, match="Invalid track ID"):
        _parse_track_id("invalid")
    
    with pytest.raises(ValueError, match="Cannot parse track_id"):
        _parse_track_id(None)
    
    with pytest.raises(ValueError, match="Cannot parse track_id"):
        _parse_track_id([])
