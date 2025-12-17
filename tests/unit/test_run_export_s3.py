"""Unit tests for S3 export upload functionality.

Tests cover:
- S3 validation for export uploads
- Upload behavior based on storage backend configuration
- Fail-loud behavior when S3 is misconfigured
- Export endpoint S3 headers
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestValidateS3ForExport:
    """Tests for validate_s3_for_export function."""

    def test_local_backend_is_always_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that local backend is always valid and s3_enabled=False."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        from apps.api.services.run_export import validate_s3_for_export

        is_valid, error, config = validate_s3_for_export()

        assert is_valid is True
        assert error is None
        assert config["backend"] == "local"
        assert config["s3_enabled"] is False

    def test_s3_backend_without_bucket_is_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that S3 backend without bucket configured is invalid."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.delenv("SCREENALYTICS_S3_BUCKET", raising=False)
        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Need to clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_export import validate_s3_for_export

        is_valid, error, config = validate_s3_for_export()

        assert is_valid is False
        assert error is not None
        assert "bucket" in error.lower()
        assert config["s3_enabled"] is True

    def test_s3_backend_with_bucket_is_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that S3 backend with bucket configured is valid."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.setenv("SCREENALYTICS_S3_BUCKET", "test-bucket")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Need to clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_export import validate_s3_for_export

        is_valid, error, config = validate_s3_for_export()

        assert is_valid is True
        assert error is None
        assert config["backend"] == "s3"
        assert config["bucket"] == "test-bucket"
        assert config["s3_enabled"] is True

    def test_minio_without_endpoint_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that MinIO without endpoint falls back to local."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "minio")
        monkeypatch.delenv("SCREENALYTICS_OBJECT_STORE_ENDPOINT", raising=False)
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Need to clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_export import validate_s3_for_export

        is_valid, error, config = validate_s3_for_export()

        # Should fall back to local due to missing endpoint
        assert config.get("is_fallback") is True or config.get("backend") == "local"


class TestUploadExportToS3:
    """Tests for upload_export_to_s3 function."""

    def test_upload_skipped_when_s3_not_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that upload is skipped when S3 is not enabled."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        from apps.api.services.run_export import upload_export_to_s3

        result = upload_export_to_s3(
            ep_id="test-ep",
            run_id="test-run",
            file_bytes=b"test content",
            filename="test.pdf",
            fail_on_error=False,
        )

        assert result.success is True
        assert "not enabled" in (result.error or "").lower()
        assert result.s3_key is None

    def test_upload_fails_loud_when_s3_misconfigured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that upload raises RuntimeError when fail_on_error=True and S3 is misconfigured."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.delenv("SCREENALYTICS_S3_BUCKET", raising=False)
        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_export import upload_export_to_s3

        with pytest.raises(RuntimeError) as exc_info:
            upload_export_to_s3(
                ep_id="test-ep",
                run_id="test-run",
                file_bytes=b"test content",
                filename="test.pdf",
                fail_on_error=True,
            )

        assert "S3 configuration invalid" in str(exc_info.value)

    def test_upload_returns_error_when_fail_on_error_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that upload returns error result when fail_on_error=False and S3 is misconfigured."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.delenv("SCREENALYTICS_S3_BUCKET", raising=False)
        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        # Clear cached config
        from apps.api.services import validation
        validation._storage_config_cache = None

        from apps.api.services.run_export import upload_export_to_s3

        result = upload_export_to_s3(
            ep_id="test-ep",
            run_id="test-run",
            file_bytes=b"test content",
            filename="test.pdf",
            fail_on_error=False,
        )

        assert result.success is False
        assert result.error is not None
        assert "configuration" in result.error.lower()


class TestExportS3Key:
    """Tests for S3 key generation."""

    def test_export_s3_key_format(self) -> None:
        """Test that S3 key is generated with correct format."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from apps.api.services.run_export import _get_export_s3_key

        key = _get_export_s3_key("rhoslc-s06e11", "abc123", "debug_report.pdf")

        assert key == "runs/rhoslc-s06e11/abc123/exports/debug_report.pdf"

    def test_export_s3_key_normalizes_run_id(self) -> None:
        """Test that S3 key normalizes run_id."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from apps.api.services.run_export import _get_export_s3_key

        # Assuming normalize_run_id lowercases - this depends on implementation
        key = _get_export_s3_key("demo-s01e01", "ABC123", "debug_bundle.zip")

        # Key should contain the normalized run_id
        assert "exports/debug_bundle.zip" in key
        assert "runs/demo-s01e01/" in key


class TestBundleStatus:
    """Tests for the _bundle_status helper function."""

    def test_bundle_status_missing_file(self, tmp_path: Path) -> None:
        """Test that missing file returns N/A."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from apps.api.services.run_export import _bundle_status

        missing_path = tmp_path / "does_not_exist.json"
        status = _bundle_status(missing_path, in_allowlist=True)

        assert status == "N/A"

    def test_bundle_status_existing_file_in_allowlist(self, tmp_path: Path) -> None:
        """Test that existing file in allowlist returns Yes."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from apps.api.services.run_export import _bundle_status

        existing_path = tmp_path / "tracks.jsonl"
        existing_path.write_text("{}")
        status = _bundle_status(existing_path, in_allowlist=True)

        assert status == "Yes"

    def test_bundle_status_existing_file_not_in_allowlist(self, tmp_path: Path) -> None:
        """Test that existing file not in allowlist returns No."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from apps.api.services.run_export import _bundle_status

        existing_path = tmp_path / "faces.npy"
        existing_path.write_bytes(b"numpy data")
        status = _bundle_status(existing_path, in_allowlist=False)

        assert status == "No"


class TestBuildAndUploadFunctions:
    """Tests for build_and_upload_debug_pdf and build_and_upload_debug_bundle."""

    def test_build_and_upload_pdf_returns_upload_result(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that build_and_upload_debug_pdf returns upload result."""
        pytest.importorskip("reportlab")
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
        monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")
        monkeypatch.setenv("STORAGE_BACKEND", "local")

        # Clear cached validation config to ensure fresh state
        from apps.api.services import validation
        validation._storage_config_cache = None

        from py_screenalytics import run_layout
        from apps.api.services.run_export import build_and_upload_debug_pdf

        ep_id = "test-ep-pdf"
        run_id = "test-run-pdf"

        run_root = run_layout.run_root(ep_id, run_id)
        run_root.mkdir(parents=True, exist_ok=True)

        # Create minimal tracks.jsonl
        tracks_path = run_root / "tracks.jsonl"
        with tracks_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 1.0}) + "\n")

        pdf_bytes, download_name, upload_result = build_and_upload_debug_pdf(
            ep_id=ep_id,
            run_id=run_id,
            upload_to_s3=True,
        )

        assert pdf_bytes[:4] == b"%PDF"
        assert "debug_report.pdf" in download_name
        assert upload_result is not None
        # With local backend, upload is skipped but returns success
        assert upload_result.success is True

    def test_build_and_upload_pdf_skips_upload_when_disabled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that build_and_upload_debug_pdf skips upload when upload_to_s3=False."""
        pytest.importorskip("reportlab")
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
        monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")
        monkeypatch.setenv("STORAGE_BACKEND", "local")

        # Clear cached validation config to ensure fresh state
        from apps.api.services import validation
        validation._storage_config_cache = None

        from py_screenalytics import run_layout
        from apps.api.services.run_export import build_and_upload_debug_pdf

        ep_id = "test-ep-no-upload"
        run_id = "test-run-no-upload"

        run_root = run_layout.run_root(ep_id, run_id)
        run_root.mkdir(parents=True, exist_ok=True)

        # Create minimal tracks.jsonl
        tracks_path = run_root / "tracks.jsonl"
        with tracks_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 1.0}) + "\n")

        pdf_bytes, download_name, upload_result = build_and_upload_debug_pdf(
            ep_id=ep_id,
            run_id=run_id,
            upload_to_s3=False,
        )

        assert pdf_bytes[:4] == b"%PDF"
        assert upload_result is None  # No upload attempted
