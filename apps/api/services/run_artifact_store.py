"""Run artifact S3 persistence service.

This module provides functions for syncing run-scoped artifacts to S3 and
validating artifact store configuration with fail-loud behavior.

Key Features:
- sync_run_artifacts_to_s3(): Upload run artifacts to S3
- validate_artifact_store(): Check S3 configuration with fail-loud option
- ArtifactSyncResult: Detailed sync result for observability

S3 Key Structure:
    runs/{show}/s{ss}/e{ee}/{run_id}/
        tracks.jsonl
        faces.jsonl
        detections.jsonl
        identities.json
        ...
        exports/
            debug_report.pdf
            debug_bundle.zip
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def _is_s3_required() -> bool:
    """Check if S3 is required via environment variable.

    Returns True if STORAGE_REQUIRE_S3 or EXPORT_REQUIRE_S3 is set to a truthy value.
    In this mode, S3 misconfiguration should fail requests loudly rather than
    falling back to local storage silently.
    """
    for var in ("STORAGE_REQUIRE_S3", "EXPORT_REQUIRE_S3"):
        val = os.environ.get(var, "").strip().lower()
        if val in ("1", "true", "yes", "on"):
            return True
    return False


def _should_delete_local_after_sync() -> bool:
    """Check if local files should be deleted after successful S3 sync.

    Returns True if STORAGE_DELETE_LOCAL_AFTER_SYNC is set to a truthy value.
    When enabled, local copies of run artifacts are deleted after successful
    S3 upload, making S3 the primary storage location.
    """
    val = os.environ.get("STORAGE_DELETE_LOCAL_AFTER_SYNC", "").strip().lower()
    return val in ("1", "true", "yes", "on")


@dataclass
class ArtifactSyncResult:
    """Result of a run artifact sync operation."""

    success: bool
    """True if all uploads succeeded (or no uploads needed)."""

    total_artifacts: int = 0
    """Total number of artifacts found to sync."""

    uploaded_count: int = 0
    """Number of artifacts successfully uploaded."""

    skipped_count: int = 0
    """Number of artifacts skipped (already exist in S3)."""

    failed_count: int = 0
    """Number of artifacts that failed to upload."""

    bytes_uploaded: int = 0
    """Total bytes uploaded to S3."""

    sync_time_ms: float = 0.0
    """Time taken for sync in milliseconds."""

    s3_prefix: str | None = None
    """S3 prefix where artifacts were uploaded."""

    s3_bucket: str | None = None
    """S3 bucket name."""

    backend_type: str = "unknown"
    """Storage backend type (local, s3, hybrid)."""

    errors: list[str] = field(default_factory=list)
    """List of error messages for failed uploads."""

    uploaded_files: list[str] = field(default_factory=list)
    """List of S3 keys that were uploaded."""

    deleted_count: int = 0
    """Number of local files deleted after successful upload."""

    deleted_files: list[str] = field(default_factory=list)
    """List of local files that were deleted."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "total_artifacts": self.total_artifacts,
            "uploaded_count": self.uploaded_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "bytes_uploaded": self.bytes_uploaded,
            "sync_time_ms": self.sync_time_ms,
            "s3_prefix": self.s3_prefix,
            "s3_bucket": self.s3_bucket,
            "backend_type": self.backend_type,
            "errors": self.errors,
            "uploaded_files": self.uploaded_files,
            "deleted_count": self.deleted_count,
            "deleted_files": self.deleted_files,
        }


@dataclass
class ArtifactStoreConfig:
    """Artifact store configuration summary."""

    backend: str
    """Storage backend type (local, s3, minio, hybrid)."""

    s3_enabled: bool
    """True if S3 uploads are enabled."""

    bucket: str | None
    """S3 bucket name (if S3 enabled)."""

    region: str | None
    """AWS region (if S3 enabled)."""

    endpoint: str | None
    """Custom S3 endpoint (for MinIO)."""

    has_credentials: bool
    """True if AWS credentials are configured."""

    is_fallback: bool
    """True if using fallback configuration due to errors."""

    errors: list[str] = field(default_factory=list)
    """Configuration errors."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "s3_enabled": self.s3_enabled,
            "bucket": self.bucket,
            "region": self.region,
            "endpoint": self.endpoint,
            "has_credentials": self.has_credentials,
            "is_fallback": self.is_fallback,
            "errors": self.errors,
        }


def validate_artifact_store(*, fail_loud: bool = False) -> tuple[bool, str | None, ArtifactStoreConfig]:
    """Validate artifact store configuration.

    Args:
        fail_loud: If True, raise RuntimeError when S3 is enabled but misconfigured.

    Returns:
        Tuple of (is_valid, error_message, config).
        - is_valid: True if configuration is valid (or local backend)
        - error_message: Human-readable error if invalid
        - config: ArtifactStoreConfig with full configuration details

    Raises:
        RuntimeError: If fail_loud=True and S3 is misconfigured.
    """
    from apps.api.services.validation import validate_storage_backend_config

    storage_config = validate_storage_backend_config()

    backend = storage_config.backend
    s3_enabled = backend in ("s3", "minio", "hybrid")

    config = ArtifactStoreConfig(
        backend=backend,
        s3_enabled=s3_enabled,
        bucket=storage_config.bucket,
        region=storage_config.region,
        endpoint=storage_config.s3_endpoint,
        has_credentials=storage_config.has_credentials,
        is_fallback=storage_config.is_fallback,
        errors=storage_config.errors.copy(),
    )

    # For local backend, always valid
    if backend == "local":
        return True, None, config

    # For S3 backends, check bucket is configured
    if s3_enabled and not storage_config.bucket:
        error_msg = (
            f"STORAGE_BACKEND={backend} but S3 bucket is not configured. "
            "Set SCREENALYTICS_S3_BUCKET or AWS_S3_BUCKET in environment."
        )
        config.errors.append(error_msg)

        if fail_loud:
            LOGGER.error("[artifact-store] %s", error_msg)
            raise RuntimeError(f"Artifact store configuration invalid: {error_msg}")

        return False, error_msg, config

    # For MinIO, check endpoint is configured
    if backend == "minio" and not storage_config.s3_endpoint:
        error_msg = (
            "STORAGE_BACKEND=minio but endpoint is not configured. "
            "Set SCREENALYTICS_OBJECT_STORE_ENDPOINT in environment."
        )
        config.errors.append(error_msg)

        if fail_loud:
            LOGGER.error("[artifact-store] %s", error_msg)
            raise RuntimeError(f"Artifact store configuration invalid: {error_msg}")

        return False, error_msg, config

    # Check if we fell back due to configuration errors
    if storage_config.is_fallback:
        error_msg = storage_config.fallback_reason or "Configuration fallback applied"

        if fail_loud:
            LOGGER.error("[artifact-store] %s", error_msg)
            raise RuntimeError(f"Artifact store configuration invalid: {error_msg}")

        return False, error_msg, config

    return True, None, config


def sync_run_artifacts_to_s3(
    ep_id: str,
    run_id: str,
    *,
    fail_on_error: bool | None = None,
    skip_existing: bool = True,
    delete_local_after_sync: bool | None = None,
) -> ArtifactSyncResult:
    """Sync run-scoped artifacts to S3.

    Uploads all run artifacts from the local run directory to S3 using the
    configured storage backend.

    Args:
        ep_id: Episode ID (e.g., 'rhoslc-s06e11').
        run_id: Run ID.
        fail_on_error: If True, raise RuntimeError on S3 errors.
            If None (default), uses STORAGE_REQUIRE_S3 env var.
        skip_existing: If True, skip files that already exist in S3.
        delete_local_after_sync: If True, delete local files after successful S3 upload.
            If None (default), uses STORAGE_DELETE_LOCAL_AFTER_SYNC env var.
            This makes S3 the primary storage location.

    Returns:
        ArtifactSyncResult with sync details.

    Raises:
        RuntimeError: If fail_on_error=True and sync fails.
    """
    # Default fail_on_error based on env var if not explicitly set
    if fail_on_error is None:
        fail_on_error = _is_s3_required()
    # Default delete_local_after_sync based on env var if not explicitly set
    if delete_local_after_sync is None:
        delete_local_after_sync = _should_delete_local_after_sync()
    from py_screenalytics import run_layout

    start_time = time.perf_counter()

    # Validate configuration
    is_valid, error_msg, config = validate_artifact_store(fail_loud=False)

    result = ArtifactSyncResult(
        success=True,
        backend_type=config.backend,
        s3_bucket=config.bucket,
    )

    # If local backend, nothing to sync
    if config.backend == "local":
        result.s3_prefix = None
        result.sync_time_ms = (time.perf_counter() - start_time) * 1000
        LOGGER.debug("[artifact-store] Local backend - no S3 sync needed")
        return result

    # If S3 is misconfigured, handle error
    if not is_valid:
        result.success = False
        result.errors.append(error_msg or "S3 configuration invalid")

        if fail_on_error:
            LOGGER.error("[artifact-store] S3 sync failed: %s", error_msg)
            raise RuntimeError(f"S3 sync failed: {error_msg}")

        result.sync_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    # Get list of artifacts to sync
    artifacts = run_layout.list_run_artifacts(ep_id, run_id)
    result.total_artifacts = len(artifacts)
    result.s3_prefix = run_layout.run_s3_prefix(ep_id, run_id)

    if not artifacts:
        LOGGER.info("[artifact-store] No artifacts to sync for %s/%s", ep_id, run_id)
        result.sync_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    # Get storage service for uploads
    try:
        from apps.api.services.storage import StorageService

        storage = StorageService()
    except Exception as exc:
        result.success = False
        result.errors.append(f"Failed to initialize storage service: {exc}")

        if fail_on_error:
            raise RuntimeError(f"S3 sync failed: {exc}") from exc

        result.sync_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    # Upload each artifact
    for local_path, s3_key in artifacts:
        try:
            # Check if already exists
            if skip_existing and storage.object_exists(s3_key):
                result.skipped_count += 1
                LOGGER.debug("[artifact-store] Skipping %s (already exists)", s3_key)
                continue

            # Read and upload
            data = local_path.read_bytes()
            content_type = _infer_content_type(local_path.name)

            if storage.upload_bytes(data, s3_key, content_type=content_type):
                result.uploaded_count += 1
                result.bytes_uploaded += len(data)
                result.uploaded_files.append(s3_key)
                LOGGER.debug("[artifact-store] Uploaded %s (%d bytes)", s3_key, len(data))
            else:
                result.failed_count += 1
                result.errors.append(f"Upload failed for {s3_key}")

        except Exception as exc:
            result.failed_count += 1
            result.errors.append(f"Error uploading {local_path.name}: {exc}")
            LOGGER.warning("[artifact-store] Failed to upload %s: %s", s3_key, exc)

    result.success = result.failed_count == 0
    result.sync_time_ms = (time.perf_counter() - start_time) * 1000

    if result.success:
        LOGGER.info(
            "[artifact-store] Synced %d artifacts to s3://%s/%s (%d bytes in %.1fms)",
            result.uploaded_count,
            config.bucket,
            result.s3_prefix,
            result.bytes_uploaded,
            result.sync_time_ms,
        )

        # Delete local files after successful sync if requested
        if delete_local_after_sync and (result.uploaded_count > 0 or result.skipped_count > 0):
            LOGGER.info("[artifact-store] Deleting local copies after successful S3 sync...")
            for local_path, s3_key in artifacts:
                try:
                    if local_path.exists():
                        local_path.unlink()
                        result.deleted_count += 1
                        result.deleted_files.append(str(local_path))
                        LOGGER.debug("[artifact-store] Deleted local file: %s", local_path)
                except Exception as del_exc:
                    LOGGER.warning("[artifact-store] Failed to delete %s: %s", local_path, del_exc)

            # Clean up empty directories
            run_root = run_layout.run_root(ep_id, run_id)
            for subdir in ["body_tracking", "face_alignment"]:
                subdir_path = run_root / subdir
                if subdir_path.exists() and subdir_path.is_dir():
                    try:
                        # Only remove if empty
                        if not any(subdir_path.iterdir()):
                            subdir_path.rmdir()
                            LOGGER.debug("[artifact-store] Removed empty directory: %s", subdir_path)
                    except Exception:
                        pass  # Directory not empty or permission issue

            LOGGER.info(
                "[artifact-store] Deleted %d local files after S3 sync",
                result.deleted_count,
            )
    else:
        LOGGER.warning(
            "[artifact-store] Sync completed with %d failures: %s",
            result.failed_count,
            result.errors,
        )
        if fail_on_error:
            raise RuntimeError(f"S3 sync failed: {result.failed_count} uploads failed")

    return result


def upload_export_to_s3(
    ep_id: str,
    run_id: str,
    file_bytes: bytes,
    filename: str,
    *,
    fail_on_error: bool | None = None,
) -> ArtifactSyncResult:
    """Upload an export file (PDF/ZIP) to S3.

    Args:
        ep_id: Episode ID.
        run_id: Run ID.
        file_bytes: Export file content.
        filename: Export filename (e.g., 'debug_report.pdf').
        fail_on_error: If True, raise RuntimeError on failure.
            If None (default), uses EXPORT_REQUIRE_S3 env var.

    Returns:
        ArtifactSyncResult with upload details.
    """
    # Default fail_on_error based on env var if not explicitly set
    if fail_on_error is None:
        fail_on_error = _is_s3_required()

    from py_screenalytics import run_layout

    start_time = time.perf_counter()

    # Validate configuration
    is_valid, error_msg, config = validate_artifact_store(fail_loud=False)

    result = ArtifactSyncResult(
        success=True,
        total_artifacts=1,
        backend_type=config.backend,
        s3_bucket=config.bucket,
    )

    # If local backend, nothing to upload
    if config.backend == "local":
        result.s3_prefix = None
        result.sync_time_ms = (time.perf_counter() - start_time) * 1000
        LOGGER.debug("[artifact-store] Local backend - export not uploaded to S3")
        return result

    # If S3 is misconfigured, handle error
    if not is_valid:
        result.success = False
        result.errors.append(error_msg or "S3 configuration invalid")

        if fail_on_error:
            raise RuntimeError(f"Export upload failed: {error_msg}")

        result.sync_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    # Generate S3 key for export
    s3_key = run_layout.run_export_s3_key(ep_id, run_id, filename)
    result.s3_prefix = run_layout.run_s3_prefix(ep_id, run_id) + "exports/"

    try:
        from apps.api.services.storage import StorageService

        storage = StorageService()
        content_type = _infer_content_type(filename)

        if storage.upload_bytes(file_bytes, s3_key, content_type=content_type):
            result.uploaded_count = 1
            result.bytes_uploaded = len(file_bytes)
            result.uploaded_files.append(s3_key)
            LOGGER.info(
                "[artifact-store] Uploaded export %s to s3://%s/%s (%d bytes)",
                filename,
                config.bucket,
                s3_key,
                len(file_bytes),
            )
        else:
            result.success = False
            result.failed_count = 1
            result.errors.append(f"Upload failed for {s3_key}")

    except Exception as exc:
        result.success = False
        result.failed_count = 1
        result.errors.append(f"Error uploading {filename}: {exc}")

        if fail_on_error:
            raise RuntimeError(f"Export upload failed: {exc}") from exc

    result.sync_time_ms = (time.perf_counter() - start_time) * 1000
    return result


def _infer_content_type(filename: str) -> str:
    """Infer content type from filename extension."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return {
        "json": "application/json",
        "jsonl": "application/x-ndjson",
        "pdf": "application/pdf",
        "zip": "application/zip",
        "npy": "application/octet-stream",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
    }.get(ext, "application/octet-stream")


def get_artifact_store_display() -> str:
    """Get human-readable artifact store configuration for UI display.

    Returns:
        String like 'S3 (bucket: screenalytics, region: us-east-1)' or 'Local filesystem'.
    """
    _, _, config = validate_artifact_store(fail_loud=False)

    if config.backend == "local":
        return "Local filesystem"

    if config.backend in ("s3", "hybrid"):
        bucket = config.bucket or "not configured"
        region = config.region or "us-east-1"
        return f"S3 (bucket: {bucket}, region: {region})"

    if config.backend == "minio":
        bucket = config.bucket or "not configured"
        endpoint = config.endpoint or "not configured"
        return f"MinIO (bucket: {bucket}, endpoint: {endpoint})"

    return f"Unknown ({config.backend})"


def get_artifact_store_status() -> dict[str, Any]:
    """Get artifact store status for API/UI.

    Returns:
        Dict with backend type, S3 config, and any errors.
    """
    is_valid, error_msg, config = validate_artifact_store(fail_loud=False)
    return {
        "is_valid": is_valid,
        "error": error_msg,
        "config": config.to_dict(),
        "display": get_artifact_store_display(),
    }


def write_export_index(
    ep_id: str,
    run_id: str,
    *,
    export_type: str,
    export_key: str | None = None,
    export_bytes: int = 0,
    upload_result: ArtifactSyncResult | None = None,
    artifact_sync_result: ArtifactSyncResult | None = None,
) -> Path:
    """Write export_index.json marker file for later lookup.

    This persists export metadata so users can find S3 keys and sync status
    without re-running exports.

    Args:
        ep_id: Episode ID.
        run_id: Run ID.
        export_type: Type of export ('pdf' or 'zip').
        export_key: S3 key for the export (if uploaded).
        export_bytes: Size of export in bytes.
        upload_result: Result from upload_export_to_s3 (if uploaded).
        artifact_sync_result: Result from sync_run_artifacts_to_s3 (if synced).

    Returns:
        Path to the written export_index.json file.
    """
    import json
    import subprocess
    from datetime import datetime, timezone
    from py_screenalytics import run_layout

    run_root = run_layout.run_root(ep_id, run_id)
    exports_dir = run_root / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    # Get git SHA
    git_sha: str | None = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_sha = result.stdout.strip()
    except Exception:
        pass

    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    index_data: dict[str, Any] = {
        "ep_id": ep_id,
        "run_id": run_id,
        "generated_at": now_iso,
        "git_sha": git_sha,
        "export_type": export_type,
        "export_bytes": export_bytes,
    }

    if export_key:
        index_data["export_s3_key"] = export_key

    if upload_result:
        index_data["export_upload"] = {
            "attempted": True,
            "success": upload_result.success,
            "s3_key": upload_result.uploaded_files[0] if upload_result.uploaded_files else None,
            "bytes_uploaded": upload_result.bytes_uploaded,
            "sync_time_ms": upload_result.sync_time_ms,
            "errors": upload_result.errors,
        }
    else:
        index_data["export_upload"] = {"attempted": False}

    if artifact_sync_result:
        index_data["artifact_sync"] = {
            "attempted": True,
            "success": artifact_sync_result.success,
            "s3_prefix": artifact_sync_result.s3_prefix,
            "s3_bucket": artifact_sync_result.s3_bucket,
            "total_artifacts": artifact_sync_result.total_artifacts,
            "uploaded_count": artifact_sync_result.uploaded_count,
            "skipped_count": artifact_sync_result.skipped_count,
            "failed_count": artifact_sync_result.failed_count,
            "bytes_uploaded": artifact_sync_result.bytes_uploaded,
            "sync_time_ms": artifact_sync_result.sync_time_ms,
            "errors": artifact_sync_result.errors,
        }
    else:
        index_data["artifact_sync"] = {"attempted": False}

    index_path = exports_dir / "export_index.json"
    index_path.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
    LOGGER.info("[artifact-store] Wrote export index to %s", index_path)

    return index_path


def read_export_index(ep_id: str, run_id: str) -> dict[str, Any] | None:
    """Read export_index.json marker file.

    Args:
        ep_id: Episode ID.
        run_id: Run ID.

    Returns:
        Parsed export index data, or None if not found.
    """
    import json
    from py_screenalytics import run_layout

    run_root = run_layout.run_root(ep_id, run_id)
    index_path = run_root / "exports" / "export_index.json"

    if not index_path.exists():
        return None

    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("[artifact-store] Failed to read export index: %s", exc)
        return None
