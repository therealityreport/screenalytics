"""Unit tests for run artifact S3 persistence.

Tests cover:
- Artifact store validation with fail-loud behavior
- S3 key generation for run artifacts and exports
- Run artifact sync with mocked S3
- Export upload with mocked S3
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestValidateArtifactStore:
    """Tests for validate_artifact_store function."""

    def test_local_backend_is_always_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that local backend is always valid."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_artifact_store import validate_artifact_store

        is_valid, error, config = validate_artifact_store()

        assert is_valid is True
        assert error is None
        assert config.backend == "local"
        assert config.s3_enabled is False

    def test_s3_backend_without_bucket_is_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that S3 backend without bucket is invalid."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.delenv("SCREENALYTICS_S3_BUCKET", raising=False)
        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_artifact_store import validate_artifact_store

        is_valid, error, config = validate_artifact_store()

        assert is_valid is False
        assert error is not None
        assert "bucket" in error.lower()
        assert config.s3_enabled is True

    def test_fail_loud_raises_on_misconfiguration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that fail_loud=True raises RuntimeError on misconfiguration."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.delenv("SCREENALYTICS_S3_BUCKET", raising=False)
        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_artifact_store import validate_artifact_store

        with pytest.raises(RuntimeError) as exc_info:
            validate_artifact_store(fail_loud=True)

        assert "configuration invalid" in str(exc_info.value).lower()

    def test_s3_backend_with_bucket_is_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that S3 backend with bucket configured is valid."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.setenv("SCREENALYTICS_S3_BUCKET", "test-bucket")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_artifact_store import validate_artifact_store

        is_valid, error, config = validate_artifact_store()

        assert is_valid is True
        assert error is None
        assert config.backend == "s3"
        assert config.bucket == "test-bucket"
        assert config.s3_enabled is True


class TestRunLayoutS3Keys:
    """Tests for S3 key generation in run_layout."""

    def test_run_s3_prefix_format(self) -> None:
        """Test S3 prefix generation format."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from py_screenalytics.run_layout import run_s3_prefix

        prefix = run_s3_prefix("rhoslc-s06e11", "abc123")
        assert prefix == "runs/rhoslc/s06/e11/abc123/"

    def test_run_artifact_s3_key_format(self) -> None:
        """Test artifact S3 key generation."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from py_screenalytics.run_layout import run_artifact_s3_key

        key = run_artifact_s3_key("rhoslc-s06e11", "abc123", "tracks.jsonl")
        assert key == "runs/rhoslc/s06/e11/abc123/tracks.jsonl"

    def test_run_export_s3_key_format(self) -> None:
        """Test export S3 key generation."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from py_screenalytics.run_layout import run_export_s3_key

        key = run_export_s3_key("rhoslc-s06e11", "abc123", "debug_report.pdf")
        assert key == "runs/rhoslc/s06/e11/abc123/exports/debug_report.pdf"

    def test_artifact_and_export_share_run_prefix(self) -> None:
        """Test that artifacts + exports share the same run prefix for parsed ep_ids."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from py_screenalytics.run_layout import run_s3_prefix, run_artifact_s3_key, run_export_s3_key

        prefix = run_s3_prefix("rhoslc-s06e11", "abc123")
        assert run_artifact_s3_key("rhoslc-s06e11", "abc123", "tracks.jsonl").startswith(prefix)
        assert run_export_s3_key("rhoslc-s06e11", "abc123", "debug_report.pdf").startswith(prefix)

    def test_parse_ep_id_standard(self) -> None:
        """Test episode ID parsing for standard format."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from py_screenalytics.run_layout import parse_episode_routing

        routing = parse_episode_routing("rhoslc-s06e11")
        assert routing is not None
        assert routing.show == "rhoslc"
        assert routing.season == 6
        assert routing.episode == 11

    def test_parse_ep_id_fallback(self) -> None:
        """Test episode ID parsing fallback for non-standard format."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from py_screenalytics.run_layout import parse_episode_routing

        routing = parse_episode_routing("custom-episode-id")
        assert routing is None

    def test_get_run_s3_layout_canonical(self) -> None:
        """Test canonical layout when ep_id is parseable."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from py_screenalytics.run_layout import get_run_s3_layout

        layout = get_run_s3_layout("rhoslc-s06e11", "abc123")
        assert layout.s3_layout == "canonical"
        assert layout.write_prefix == "runs/rhoslc/s06/e11/abc123/"
        assert layout.canonical_prefix == "runs/rhoslc/s06/e11/abc123/"
        assert layout.legacy_prefix == "runs/rhoslc-s06e11/abc123/"

    def test_get_run_s3_layout_legacy(self) -> None:
        """Test legacy fallback when ep_id is not parseable."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from py_screenalytics.run_layout import get_run_s3_layout

        layout = get_run_s3_layout("custom-episode-id", "abc123")
        assert layout.s3_layout == "legacy"
        assert layout.canonical_prefix is None
        assert layout.write_prefix == "runs/custom-episode-id/abc123/"
        assert layout.legacy_prefix == "runs/custom-episode-id/abc123/"

    def test_run_artifact_keys_for_read_prefers_canonical(self) -> None:
        """Test canonical-first read order for run artifacts."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from py_screenalytics.run_layout import run_artifact_s3_keys_for_read

        keys = run_artifact_s3_keys_for_read("rhoslc-s06e11", "abc123", "tracks.jsonl")
        assert keys == [
            "runs/rhoslc/s06/e11/abc123/tracks.jsonl",
            "runs/rhoslc-s06e11/abc123/tracks.jsonl",
        ]

    def test_run_export_keys_for_read_prefers_canonical(self) -> None:
        """Test canonical-first read order for exports."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from py_screenalytics.run_layout import run_export_s3_keys_for_read

        keys = run_export_s3_keys_for_read("rhoslc-s06e11", "abc123", "debug_report.pdf")
        assert keys == [
            "runs/rhoslc/s06/e11/abc123/exports/debug_report.pdf",
            "runs/rhoslc-s06e11/abc123/exports/debug_report.pdf",
        ]

    def test_run_keys_for_read_unparseable_ep_id_is_legacy_only(self) -> None:
        """Test read key lists for unparseable ep_ids."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from py_screenalytics.run_layout import run_artifact_s3_keys_for_read, run_export_s3_keys_for_read

        assert run_artifact_s3_keys_for_read("custom-episode-id", "abc123", "tracks.jsonl") == [
            "runs/custom-episode-id/abc123/tracks.jsonl"
        ]
        assert run_export_s3_keys_for_read("custom-episode-id", "abc123", "debug_report.pdf") == [
            "runs/custom-episode-id/abc123/exports/debug_report.pdf"
        ]

    def test_run_artifact_allowlist_includes_core_face_and_body_outputs(self) -> None:
        """Lock in the minimum run-scoped bundle required for run health + PDF exports."""
        import sys

        sys.path.insert(0, str(PROJECT_ROOT))

        from py_screenalytics.run_layout import RUN_ARTIFACT_ALLOWLIST

        required = {
            # Face detect/track/embed/cluster core
            "detections.jsonl",
            "tracks.jsonl",
            "faces.jsonl",
            "identities.json",
            "track_metrics.json",
            "track_reps.jsonl",
            "cluster_centroids.json",
            # Phase markers (used by run diagnostics)
            "detect_track.json",
            "faces_embed.json",
            "cluster.json",
            "body_tracking.json",
            "body_tracking_fusion.json",
            # Face alignment output (embed gating + diagnostics)
            "face_alignment/aligned_faces.jsonl",
            # Body tracking outputs required for fusion + screentime
            "body_tracking/body_detections.jsonl",
            "body_tracking/body_tracks.jsonl",
            "body_tracking/body_embeddings.npy",
            "body_tracking/body_embeddings_meta.json",
            "body_tracking/track_fusion.json",
            "body_tracking/screentime_comparison.json",
        }

        missing = required - set(RUN_ARTIFACT_ALLOWLIST)
        assert not missing, f"RUN_ARTIFACT_ALLOWLIST missing required entries: {sorted(missing)}"


class TestSyncRunArtifactsToS3:
    """Tests for sync_run_artifacts_to_s3 function."""

    def test_sync_skipped_for_local_backend(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that sync is no-op for local backend."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_artifact_store import sync_run_artifacts_to_s3

        result = sync_run_artifacts_to_s3("test-ep", "test-run")

        assert result.success is True
        assert result.backend_type == "local"
        assert result.uploaded_count == 0
        assert result.s3_prefix is None

    def test_sync_returns_error_for_misconfigured_s3(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that sync returns error for misconfigured S3."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.delenv("SCREENALYTICS_S3_BUCKET", raising=False)
        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
        monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_artifact_store import sync_run_artifacts_to_s3

        result = sync_run_artifacts_to_s3("test-ep", "test-run", fail_on_error=False)

        assert result.success is False
        assert len(result.errors) > 0

    def test_sync_raises_on_fail_on_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that sync raises when fail_on_error=True and S3 is misconfigured."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.delenv("SCREENALYTICS_S3_BUCKET", raising=False)
        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
        monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_artifact_store import sync_run_artifacts_to_s3

        with pytest.raises(RuntimeError):
            sync_run_artifacts_to_s3("test-ep", "test-run", fail_on_error=True)


class TestUploadExportToS3:
    """Tests for upload_export_to_s3 function."""

    def test_upload_skipped_for_local_backend(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that export upload is skipped for local backend."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_artifact_store import upload_export_to_s3

        result = upload_export_to_s3(
            ep_id="test-ep",
            run_id="test-run",
            file_bytes=b"test content",
            filename="debug_report.pdf",
        )

        assert result.success is True
        assert result.backend_type == "local"
        assert result.uploaded_count == 0

    def test_upload_returns_error_for_misconfigured_s3(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that export upload returns error for misconfigured S3."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.delenv("SCREENALYTICS_S3_BUCKET", raising=False)
        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
        monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_artifact_store import upload_export_to_s3

        result = upload_export_to_s3(
            ep_id="test-ep",
            run_id="test-run",
            file_bytes=b"test content",
            filename="debug_report.pdf",
            fail_on_error=False,
        )

        assert result.success is False
        assert len(result.errors) > 0


class TestListRunArtifacts:
    """Tests for list_run_artifacts function."""

    def test_list_returns_existing_artifacts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that list_run_artifacts returns existing allowlisted artifacts."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        from py_screenalytics import run_layout

        ep_id = "test-ep-list"
        run_id = "test-run-list"

        # Create run directory with some artifacts
        run_root = run_layout.run_root(ep_id, run_id)
        run_root.mkdir(parents=True, exist_ok=True)

        # Create allowlisted artifacts
        (run_root / "tracks.jsonl").write_text('{"track_id": 1}\n')
        (run_root / "faces.jsonl").write_text('{"face_id": 1}\n')
        (run_root / "identities.json").write_text('{}')

        # Create non-allowlisted artifact
        (run_root / "random.txt").write_text("ignored")

        artifacts = run_layout.list_run_artifacts(ep_id, run_id)

        # Should find 3 allowlisted artifacts
        filenames = [p.name for p, _ in artifacts]
        assert "tracks.jsonl" in filenames
        assert "faces.jsonl" in filenames
        assert "identities.json" in filenames
        assert "random.txt" not in filenames
        assert len(artifacts) == 3

    def test_list_returns_empty_for_missing_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that list_run_artifacts returns empty for non-existent run."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        from py_screenalytics import run_layout

        artifacts = run_layout.list_run_artifacts("nonexistent-ep", "nonexistent-run")
        assert artifacts == []


class TestArtifactStoreDisplay:
    """Tests for get_artifact_store_display function."""

    def test_display_local_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test display string for local backend."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_artifact_store import get_artifact_store_display

        display = get_artifact_store_display()
        assert display == "Local filesystem"

    def test_display_s3_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test display string for S3 backend."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.setenv("SCREENALYTICS_S3_BUCKET", "test-bucket")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_artifact_store import get_artifact_store_display

        display = get_artifact_store_display()
        assert "S3" in display
        assert "test-bucket" in display
        assert "us-west-2" in display


class TestArtifactStoreStatus:
    """Tests for get_artifact_store_status function."""

    def test_status_includes_all_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that status dict includes all expected fields."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_artifact_store import get_artifact_store_status

        status = get_artifact_store_status()

        assert "is_valid" in status
        assert "error" in status
        assert "config" in status
        assert "display" in status
        assert status["is_valid"] is True
        assert status["config"]["backend"] == "local"
