"""
Integration tests for face alignment in the main pipeline.

Tests the pipeline integration module and configuration loading.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestFaceAlignmentConfig:
    """Tests for face alignment configuration loading."""

    def test_load_default_config(self):
        """Test loading config with defaults."""
        from FEATURES.face_alignment.src.pipeline_integration import (
            load_face_alignment_config,
        )

        config = load_face_alignment_config(Path("/nonexistent/path"))

        assert "face_alignment" in config
        assert config["face_alignment"]["enabled"] is False  # Default disabled
        assert config["face_alignment"]["model"]["type"] == "2d"

    def test_load_config_from_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        from FEATURES.face_alignment.src.pipeline_integration import (
            load_face_alignment_config,
        )

        yaml_content = """
face_alignment:
  enabled: true
  model:
    type: "2d"
    device: "cpu"
  processing:
    stride: 5
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml_content)

        config = load_face_alignment_config(config_path)

        assert config["face_alignment"]["enabled"] is True
        assert config["face_alignment"]["processing"]["stride"] == 5

    def test_is_face_alignment_enabled(self):
        """Test enabled check."""
        from FEATURES.face_alignment.src.pipeline_integration import (
            is_face_alignment_enabled,
        )

        assert is_face_alignment_enabled({"face_alignment": {"enabled": True}}) is True
        assert is_face_alignment_enabled({"face_alignment": {"enabled": False}}) is False
        assert is_face_alignment_enabled({}) is False


class TestPipelineIntegration:
    """Tests for pipeline stage integration."""

    def test_run_stage_disabled(self, tmp_path):
        """Test stage skips when disabled."""
        from FEATURES.face_alignment.src.pipeline_integration import (
            run_face_alignment_stage,
        )

        config = {"face_alignment": {"enabled": False}}

        success, output_path = run_face_alignment_stage(
            episode_id="test-ep",
            manifest_dir=tmp_path,
            video_path=tmp_path / "video.mp4",
            config=config,
        )

        assert success is True
        assert output_path is None

    def test_run_stage_skip_existing(self, tmp_path):
        """Test stage skips when output exists."""
        from FEATURES.face_alignment.src.pipeline_integration import (
            run_face_alignment_stage,
        )

        # Create existing output
        output_dir = tmp_path / "face_alignment"
        output_dir.mkdir(parents=True)
        output_path = output_dir / "aligned_faces.jsonl"
        output_path.write_text('{"test": true}\n')

        config = {"face_alignment": {"enabled": True}}

        success, result_path = run_face_alignment_stage(
            episode_id="test-ep",
            manifest_dir=tmp_path,
            video_path=tmp_path / "video.mp4",
            config=config,
            skip_existing=True,
        )

        assert success is True
        assert result_path == output_path


class TestAlignedCropLoading:
    """Tests for loading aligned crops for embeddings."""

    def test_load_aligned_crops(self, tmp_path):
        """Test loading aligned crops by track."""
        from FEATURES.face_alignment.src.pipeline_integration import (
            load_aligned_crops_for_embedding,
        )

        # Create test data
        aligned_dir = tmp_path / "face_alignment"
        aligned_dir.mkdir(parents=True)

        test_data = [
            {"frame_idx": 0, "track_id": 1, "bbox": [0, 0, 100, 100]},
            {"frame_idx": 5, "track_id": 1, "bbox": [10, 10, 110, 110]},
            {"frame_idx": 10, "track_id": 2, "bbox": [200, 200, 300, 300]},
        ]

        with open(aligned_dir / "aligned_faces.jsonl", "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Load all tracks
        by_track = load_aligned_crops_for_embedding(tmp_path)

        assert len(by_track) == 2
        assert 1 in by_track
        assert 2 in by_track
        assert len(by_track[1]) == 2
        assert len(by_track[2]) == 1

    def test_load_aligned_crops_filtered(self, tmp_path):
        """Test loading aligned crops with track filter."""
        from FEATURES.face_alignment.src.pipeline_integration import (
            load_aligned_crops_for_embedding,
        )

        # Create test data
        aligned_dir = tmp_path / "face_alignment"
        aligned_dir.mkdir(parents=True)

        test_data = [
            {"frame_idx": 0, "track_id": 1, "bbox": [0, 0, 100, 100]},
            {"frame_idx": 10, "track_id": 2, "bbox": [200, 200, 300, 300]},
        ]

        with open(aligned_dir / "aligned_faces.jsonl", "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Load only track 1
        by_track = load_aligned_crops_for_embedding(tmp_path, track_ids=[1])

        assert len(by_track) == 1
        assert 1 in by_track
        assert 2 not in by_track

    def test_load_aligned_crops_missing_file(self, tmp_path):
        """Test loading when file doesn't exist."""
        from FEATURES.face_alignment.src.pipeline_integration import (
            load_aligned_crops_for_embedding,
        )

        by_track = load_aligned_crops_for_embedding(tmp_path)
        assert by_track == {}


class TestGetAlignedCropPath:
    """Tests for getting aligned crop image paths."""

    def test_get_crop_path_exists(self, tmp_path):
        """Test getting crop path when it exists."""
        from FEATURES.face_alignment.src.pipeline_integration import (
            get_aligned_crop_path,
        )

        # Create test crop
        crops_dir = tmp_path / "face_alignment" / "aligned_crops"
        crops_dir.mkdir(parents=True)

        crop_path = crops_dir / "frame_000010_track_5.jpg"
        crop_path.write_text("fake image")

        result = get_aligned_crop_path(tmp_path, frame_idx=10, track_id=5)

        assert result == crop_path

    def test_get_crop_path_not_found(self, tmp_path):
        """Test getting crop path when it doesn't exist."""
        from FEATURES.face_alignment.src.pipeline_integration import (
            get_aligned_crop_path,
        )

        result = get_aligned_crop_path(tmp_path, frame_idx=10, track_id=5)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
