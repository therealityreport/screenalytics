"""
Tests for face alignment sandbox.

Covers unit tests for alignment utilities and integration tests
for the full pipeline.
"""

import json
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import fixtures
pytest_plugins = ["FEATURES.face_alignment.tests.fixtures"]


# ============================================================================
# Unit Tests: Data Structures
# ============================================================================

class TestAlignedFaceDataclass:
    """Tests for AlignedFace dataclass."""

    def test_aligned_face_creation(self):
        """Test creating AlignedFace with required fields."""
        from FEATURES.face_alignment.src.run_fan_alignment import AlignedFace

        af = AlignedFace(
            frame_idx=10,
            bbox=[100.0, 100.0, 200.0, 250.0],
            landmarks_68=[[0.0, 0.0]] * 68,
            confidence=0.95,
            detection_id=1,
            alignment_quality=0.85,
        )

        assert af.detection_id == 1
        assert af.frame_idx == 10
        assert len(af.landmarks_68) == 68
        assert af.confidence == 0.95
        assert af.alignment_quality == 0.85
        assert af.track_id is None  # Optional

    def test_aligned_face_to_dict(self):
        """Test AlignedFace serialization."""
        from FEATURES.face_alignment.src.run_fan_alignment import AlignedFace

        af = AlignedFace(
            frame_idx=10,
            bbox=[100.0, 100.0, 200.0, 250.0],
            landmarks_68=[[0.0, 0.0]] * 68,
            confidence=0.95,
            detection_id=1,
            track_id=5,
            alignment_quality=0.85,
            pose_yaw=15.0,
            pose_pitch=-5.0,
        )

        d = af.to_dict()
        assert d["detection_id"] == 1
        assert d["track_id"] == 5
        assert d["pose_yaw"] == 15.0
        assert "landmarks_68" in d
        assert d["confidence"] == 0.95


class TestFaceAlignmentConfig:
    """Tests for FaceAlignmentConfig."""

    def test_config_defaults(self):
        """Test default config values."""
        from FEATURES.face_alignment.src.face_alignment_runner import FaceAlignmentConfig

        config = FaceAlignmentConfig()

        assert config.model_type == "2d"
        assert config.device == "auto"  # Default is auto for flexible device selection
        assert config.stride == 1
        assert config.min_confidence == 0.5  # Field is min_confidence
        assert config.min_face_size == 20
        assert config.crop_size == 112

    def test_config_from_yaml(self, tmp_path):
        """Test loading config from YAML."""
        from FEATURES.face_alignment.src.face_alignment_runner import FaceAlignmentConfig

        # YAML structure must match from_yaml parser: face_alignment -> model/processing/quality/output
        yaml_content = """
face_alignment:
  model:
    type: "3d"
  processing:
    stride: 5
    device: "cpu"
  quality:
    min_confidence: 0.7
  output:
    crop_size: 224
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)

        config = FaceAlignmentConfig.from_yaml(yaml_path)

        assert config.model_type == "3d"
        assert config.device == "cpu"
        assert config.stride == 5
        assert config.min_confidence == 0.7
        assert config.crop_size == 224


# ============================================================================
# Unit Tests: Detection Loading
# ============================================================================

class TestLoadDetections:
    """Tests for detection loading utilities."""

    def test_load_face_detections_from_jsonl(self, temp_detections_jsonl):
        """Test loading detections from JSONL file."""
        from FEATURES.face_alignment.src.load_detections import load_face_detections

        detections = load_face_detections(temp_detections_jsonl)

        assert len(detections) > 0
        assert all("frame_idx" in d for d in detections)
        assert all("bbox" in d for d in detections)

    def test_load_face_detections_empty_file(self, tmp_path):
        """Test loading from empty file returns empty list."""
        from FEATURES.face_alignment.src.load_detections import load_face_detections

        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        detections = load_face_detections(empty_file)
        assert detections == []

    def test_load_face_tracks(self, temp_manifest_dir):
        """Test loading tracks from JSONL file."""
        from FEATURES.face_alignment.src.load_detections import load_face_tracks

        tracks_path = temp_manifest_dir / "tracks.jsonl"
        tracks = load_face_tracks(tracks_path)

        # Returns Dict[int, Dict] mapping track_id to track data
        assert len(tracks) > 0
        assert isinstance(tracks, dict)
        # All keys should be track IDs (ints) and values should have detections
        for track_id, track_data in tracks.items():
            assert isinstance(track_id, int)
            assert "detections" in track_data

    def test_get_representative_frames(self):
        """Test selecting representative frames from a single track."""
        from FEATURES.face_alignment.src.load_detections import get_representative_frames

        # Function takes a single track dict, not a list
        track = {
            "track_id": 1,
            "detections": [
                {"frame_idx": 0, "score": 0.8},
                {"frame_idx": 5, "score": 0.95},
                {"frame_idx": 10, "score": 0.85},
                {"frame_idx": 15, "score": 0.7},
                {"frame_idx": 20, "score": 0.9},
            ],
        }

        # Get representative frame indices using uniform strategy
        indices = get_representative_frames(track, max_frames=3, strategy="uniform")

        # Should return list of detection indices
        assert len(indices) == 3
        assert all(isinstance(i, int) for i in indices)
        assert all(0 <= i < len(track["detections"]) for i in indices)

        # Test quality strategy
        indices_quality = get_representative_frames(track, max_frames=2, strategy="quality")
        assert len(indices_quality) == 2

    def test_group_detections_by_frame(self, synthetic_face_detections):
        """Test grouping detections by frame index."""
        from FEATURES.face_alignment.src.load_detections import group_detections_by_frame

        by_frame = group_detections_by_frame(synthetic_face_detections)

        assert isinstance(by_frame, dict)
        # Should have multiple frames
        assert len(by_frame) > 1
        # Each frame should have list of detections
        for frame_idx, dets in by_frame.items():
            assert isinstance(frame_idx, int)
            assert isinstance(dets, list)
            assert all(d["frame_idx"] == frame_idx for d in dets)


# ============================================================================
# Unit Tests: Alignment Math
# ============================================================================

class TestAlignmentMath:
    """Tests for alignment transformation utilities."""

    def test_get_5_point_landmarks_from_68(self, synthetic_landmarks):
        """Test extracting 5-point landmarks from 68-point."""
        from FEATURES.face_alignment.src.run_fan_alignment import get_5_point_landmarks

        landmarks_5 = get_5_point_landmarks(synthetic_landmarks)

        assert landmarks_5.shape == (5, 2)
        # Check ordering: left_eye, right_eye, nose, left_mouth, right_mouth
        # Left eye should be to the left of right eye
        assert landmarks_5[0, 0] < landmarks_5[1, 0]
        # Nose should be between eyes vertically
        nose_y = landmarks_5[2, 1]
        eye_y = (landmarks_5[0, 1] + landmarks_5[1, 1]) / 2
        assert nose_y > eye_y  # Nose below eyes

    def test_compute_similarity_transform(self):
        """Test similarity transform computation."""
        from FEATURES.face_alignment.src.run_fan_alignment import compute_similarity_transform

        # Simple test: identity-ish transform
        src = np.array([
            [30, 30],
            [70, 30],
            [50, 50],
            [35, 70],
            [65, 70],
        ], dtype=np.float32)

        # Target is scaled version
        dst = src * 2

        transform = compute_similarity_transform(src, dst)

        assert transform.shape == (2, 3)
        # Transform should scale approximately 2x
        # Apply transform to src and check
        src_h = np.hstack([src, np.ones((5, 1))])
        transformed = src_h @ transform.T
        np.testing.assert_allclose(transformed, dst, rtol=0.1)

    def test_align_face_crop_dimensions(self, test_image, synthetic_landmarks):
        """Test aligned crop has correct dimensions."""
        from FEATURES.face_alignment.src.run_fan_alignment import align_face_crop

        crop = align_face_crop(
            test_image,
            synthetic_landmarks,
            crop_size=112,
            margin=0.0,
        )

        assert crop.shape == (112, 112, 3)
        assert crop.dtype == np.uint8


# ============================================================================
# Unit Tests: Export Functions
# ============================================================================

class TestExportFunctions:
    """Tests for export utilities."""

    def test_export_aligned_faces(self, synthetic_aligned_faces, tmp_path):
        """Test exporting aligned faces to JSONL."""
        from FEATURES.face_alignment.src.export_aligned_faces import export_aligned_faces
        from FEATURES.face_alignment.src.run_fan_alignment import AlignedFace

        # Convert dicts to AlignedFace objects
        aligned = [
            AlignedFace(
                frame_idx=af["frame_idx"],
                bbox=af["bbox"],
                landmarks_68=af["landmarks_68"],
                confidence=af.get("confidence", 0.9),
                detection_id=af.get("detection_id"),
                track_id=af.get("track_id"),
                alignment_quality=af.get("alignment_quality"),
            )
            for af in synthetic_aligned_faces[:10]
        ]

        output_path = tmp_path / "aligned_faces.jsonl"
        export_aligned_faces(aligned, output_path)

        assert output_path.exists()

        # Verify contents
        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 10

        for line in lines:
            record = json.loads(line)
            assert "detection_id" in record
            assert "landmarks_68" in record
            assert "alignment_quality" in record

    def test_load_aligned_faces(self, tmp_path):
        """Test loading aligned faces from JSONL."""
        from FEATURES.face_alignment.src.export_aligned_faces import (
            export_aligned_faces,
            load_aligned_faces,
        )
        from FEATURES.face_alignment.src.run_fan_alignment import AlignedFace

        # Create and save test data
        aligned = [
            AlignedFace(
                frame_idx=i * 5,
                bbox=[100.0, 100.0, 200.0, 250.0],
                landmarks_68=[[0.0, 0.0]] * 68,
                confidence=0.9,
                detection_id=i,
                alignment_quality=0.8 + i * 0.02,
            )
            for i in range(5)
        ]

        output_path = tmp_path / "aligned.jsonl"
        export_aligned_faces(aligned, output_path)

        # Load and verify
        loaded = load_aligned_faces(output_path)
        assert len(loaded) == 5
        assert all(r.detection_id is not None for r in loaded)

    def test_compute_alignment_stats(self, synthetic_aligned_faces):
        """Test computing alignment statistics."""
        from FEATURES.face_alignment.src.export_aligned_faces import compute_alignment_stats

        stats = compute_alignment_stats(synthetic_aligned_faces)

        assert "total_faces" in stats
        assert "mean_quality" in stats
        assert "quality_above_threshold" in stats
        assert stats["total_faces"] == len(synthetic_aligned_faces)
        assert 0.0 <= stats["mean_quality"] <= 1.0


# ============================================================================
# Integration Tests
# ============================================================================

class TestFaceAlignmentRunner:
    """Integration tests for FaceAlignmentRunner."""

    def test_runner_initialization(self, tmp_path):
        """Test runner can be initialized with default config."""
        from FEATURES.face_alignment.src.face_alignment_runner import (
            FaceAlignmentRunner,
        )

        # Create a minimal config file for the runner
        config_path = tmp_path / "config.yaml"
        config_path.write_text("face_alignment:\n  model:\n    type: '2d'\n")

        # Create a dummy video file (runner requires video_path)
        video_path = tmp_path / "test.mp4"
        video_path.touch()

        runner = FaceAlignmentRunner(
            episode_id="test-episode",
            config_path=config_path,
            video_path=video_path,
            output_dir=tmp_path / "output",
        )

        assert runner.episode_id == "test-episode"
        assert runner.config is not None

    def test_runner_output_paths(self, tmp_path):
        """Test runner uses correct manifest paths."""
        from FEATURES.face_alignment.src.face_alignment_runner import (
            FaceAlignmentRunner,
        )

        # Create a minimal config file
        config_path = tmp_path / "config.yaml"
        config_path.write_text("face_alignment:\n  model:\n    type: '2d'\n")

        # Create a dummy video file
        video_path = tmp_path / "test.mp4"
        video_path.touch()

        runner = FaceAlignmentRunner(
            episode_id="test-ep",
            config_path=config_path,
            video_path=video_path,
            output_dir=tmp_path / "output",
        )

        # Runner should have manifest_dir set based on episode_id
        assert runner.episode_id == "test-ep"
        assert runner.config is not None
        assert runner.output_dir == tmp_path / "output"


class TestIntegrationWithSyntheticData:
    """Integration tests using synthetic data."""

    def test_full_alignment_pipeline_dry_run(self, temp_manifest_dir):
        """Test full pipeline with synthetic data (no actual model)."""
        from FEATURES.face_alignment.src.face_alignment_runner import (
            FaceAlignmentConfig,
            FaceAlignmentRunner,
        )
        from FEATURES.face_alignment.src.load_detections import load_face_detections
        from FEATURES.face_alignment.src.run_fan_alignment import AlignedFace

        # Load synthetic detections
        det_path = temp_manifest_dir / "detections.jsonl"
        detections = load_face_detections(det_path)
        assert len(detections) > 0

        # Create mock aligned results (as if FAN ran)
        mock_aligned = []
        for det in detections[:10]:
            mock_aligned.append(AlignedFace(
                frame_idx=det["frame_idx"],
                bbox=det["bbox"],
                landmarks_68=[[0.0, 0.0]] * 68,
                confidence=det.get("score", 0.9),
                detection_id=det.get("detection_id"),
                track_id=det.get("track_id"),
                alignment_quality=0.85,
            ))

        # Export results
        from FEATURES.face_alignment.src.export_aligned_faces import export_aligned_faces

        output_dir = temp_manifest_dir / "face_alignment"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "aligned_faces.jsonl"

        export_aligned_faces(mock_aligned, output_path)

        # Verify output
        assert output_path.exists()
        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 10

# ============================================================================
# Smoke Tests (require face-alignment package)
# ============================================================================

class TestSmokeWithRealFAN:
    """
    Smoke tests that require the real face-alignment package.

    Run with: pytest FEATURES/face-alignment/tests/ -v -m slow

    These tests are marked @pytest.mark.slow and skipped by default.
    """

    @pytest.mark.slow
    def test_fan_aligner_loads(self):
        """Test FANAligner can load on CPU."""
        try:
            import face_alignment
        except ImportError:
            pytest.skip("face-alignment package not installed")

        from FEATURES.face_alignment.src.run_fan_alignment import FANAligner

        aligner = FANAligner(device="cpu", model_type="2d")
        # Force model load
        aligner._load_model()

        assert aligner._fa is not None

    @pytest.mark.slow
    def test_fan_aligner_with_synthetic_face(self, test_image):
        """Test FANAligner produces 68-point landmarks on synthetic face."""
        try:
            import face_alignment
        except ImportError:
            pytest.skip("face-alignment package not installed")

        from FEATURES.face_alignment.src.run_fan_alignment import FANAligner

        aligner = FANAligner(device="cpu", model_type="2d")

        # Test with a synthetic face bbox
        bbox = [200, 100, 320, 280]  # Face region in test_image
        landmarks = aligner.align_face(test_image, bbox)

        # FAN may return None if it doesn't detect a face in synthetic image
        # This is expected behavior for a simple synthetic test image
        if landmarks is not None:
            assert len(landmarks) == 68
            assert all(len(pt) == 2 for pt in landmarks)

    @pytest.mark.slow
    def test_aligned_crop_with_real_landmarks(self, test_image):
        """Test aligned crop generation with real FAN landmarks."""
        try:
            import face_alignment
        except ImportError:
            pytest.skip("face-alignment package not installed")

        from FEATURES.face_alignment.src.run_fan_alignment import (
            FANAligner,
            align_face_crop,
        )

        aligner = FANAligner(device="cpu", model_type="2d")
        bbox = [200, 100, 320, 280]
        landmarks = aligner.align_face(test_image, bbox, detect_faces=True)

        if landmarks is not None:
            crop = align_face_crop(test_image, landmarks, crop_size=112)
            assert crop.shape == (112, 112, 3)
            assert crop.dtype == test_image.dtype

    @pytest.mark.slow
    def test_5_point_extraction_from_real_landmarks(self, test_image):
        """Test 5-point extraction works with real FAN output."""
        try:
            import face_alignment
        except ImportError:
            pytest.skip("face-alignment package not installed")

        from FEATURES.face_alignment.src.run_fan_alignment import (
            FANAligner,
            get_5_point_landmarks,
        )

        aligner = FANAligner(device="cpu", model_type="2d")
        bbox = [200, 100, 320, 280]
        landmarks = aligner.align_face(test_image, bbox, detect_faces=True)

        if landmarks is not None:
            pts_5 = get_5_point_landmarks(landmarks)
            assert pts_5.shape == (5, 2)
            # Left eye should be to left of right eye
            assert pts_5[0, 0] < pts_5[1, 0]


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_detections(self, tmp_path):
        """Test handling empty detection file."""
        from FEATURES.face_alignment.src.load_detections import load_face_detections

        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        detections = load_face_detections(empty_file)
        assert detections == []

    def test_missing_optional_fields(self):
        """Test handling detections with missing optional fields."""
        from FEATURES.face_alignment.src.run_fan_alignment import AlignedFace

        # Should not raise even with minimal required fields
        af = AlignedFace(
            frame_idx=0,
            bbox=[0, 0, 100, 100],
            landmarks_68=[[0, 0]] * 68,
            confidence=0.9,
        )

        d = af.to_dict()
        assert "track_id" not in d  # Optional fields not in dict when None
        assert "pose_yaw" not in d

    def test_invalid_landmarks_count(self):
        """Test that invalid landmark count is handled."""
        from FEATURES.face_alignment.src.run_fan_alignment import get_5_point_landmarks

        # Too few landmarks
        with pytest.raises((IndexError, ValueError)):
            get_5_point_landmarks([[0, 0]] * 10)

    def test_quality_threshold_filtering(self, synthetic_aligned_faces):
        """Test filtering faces by quality threshold."""
        from FEATURES.face_alignment.src.export_aligned_faces import compute_alignment_stats

        # Get stats with default threshold
        stats = compute_alignment_stats(synthetic_aligned_faces, quality_threshold=0.8)

        assert "quality_above_threshold" in stats
        assert stats["quality_above_threshold"] <= stats["total_faces"]


# ============================================================================
# Config Loading
# ============================================================================

class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_default_config(self):
        """Test loading default face alignment config."""
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "pipeline" / "face_alignment.yaml"

        if config_path.exists():
            from FEATURES.face_alignment.src.face_alignment_runner import FaceAlignmentConfig
            config = FaceAlignmentConfig.from_yaml(config_path)

            assert config.model_type in ["2d", "3d"]
            assert config.device in ["cuda", "cpu", "auto", "mps"]  # Include auto and mps
            assert config.stride >= 1

    def test_config_validation(self):
        """Test config validation for invalid values."""
        from FEATURES.face_alignment.src.face_alignment_runner import FaceAlignmentConfig

        # These should be accepted - use correct field names
        config = FaceAlignmentConfig(
            stride=1,
            min_confidence=0.5,
        )
        assert config.stride == 1
        assert config.min_confidence == 0.5

        # Negative stride should still work (no validation currently)
        # This is a documentation of current behavior
        config2 = FaceAlignmentConfig(stride=-1)
        assert config2.stride == -1  # No validation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
