"""Storage backend abstraction for frames, crops, thumbnails, manifests, and locks.

This module provides a unified interface for storage operations that can be backed by:
- LocalStorageBackend: Local filesystem only
- S3StorageBackend: S3/MinIO only (no local writes)
- HybridStorageBackend: Local filesystem with S3 sync

The backend is selected via the STORAGE_BACKEND environment variable:
- "local": LocalStorageBackend
- "s3" or "minio": S3StorageBackend
- "hybrid": HybridStorageBackend (writes locally, syncs to S3)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Protocol, Tuple

from py_screenalytics.artifacts import get_path

LOGGER = logging.getLogger(__name__)

# Default settings
DEFAULT_BUCKET = "screenalytics"
ARTIFACT_ROOT = "artifacts"
LOCK_STALE_SECONDS = 3600  # 1 hour


@dataclass
class ArtifactInfo:
    """Metadata about a stored artifact."""

    key: str
    size: int
    last_modified: datetime | None = None
    content_type: str | None = None
    etag: str | None = None


@dataclass
class LockInfo:
    """Information about a held lock."""

    operation: str
    holder_pid: int
    holder_host: str
    acquired_at: datetime
    job_id: str | None = None

    def is_stale(self, max_age_seconds: int = LOCK_STALE_SECONDS) -> bool:
        """Check if lock is stale based on age."""
        age = (datetime.utcnow() - self.acquired_at).total_seconds()
        return age > max_age_seconds


@dataclass
class UploadResult:
    """Result of an upload operation."""

    success: bool
    key: str | None = None
    error: str | None = None
    bytes_uploaded: int = 0


@dataclass
class SyncStatus:
    """Status of S3 sync operation."""

    total_files: int = 0
    uploaded_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    bytes_uploaded: int = 0
    errors: List[str] = field(default_factory=list)
    in_progress: bool = False
    completed: bool = False


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    All storage backends must implement these methods to provide a unified
    interface for artifact storage operations.
    """

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Return the backend type identifier ('local', 's3', 'hybrid')."""
        ...

    @property
    @abstractmethod
    def supports_presigned_urls(self) -> bool:
        """Whether this backend supports presigned URLs for direct access."""
        ...

    # -------------------------------------------------------------------------
    # Frame/Crop Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def write_frame(
        self,
        ep_id: str,
        frame_idx: int,
        data: bytes,
        *,
        content_type: str = "image/jpeg",
    ) -> UploadResult:
        """Write a frame image."""
        ...

    @abstractmethod
    def write_crop(
        self,
        ep_id: str,
        track_id: int,
        frame_idx: int,
        data: bytes,
        *,
        content_type: str = "image/jpeg",
    ) -> UploadResult:
        """Write a track crop image."""
        ...

    @abstractmethod
    def get_frame_url(self, ep_id: str, frame_idx: int) -> str | None:
        """Get URL for accessing a frame (presigned URL or file path)."""
        ...

    @abstractmethod
    def get_crop_url(self, ep_id: str, track_id: int, frame_idx: int) -> str | None:
        """Get URL for accessing a crop (presigned URL or file path)."""
        ...

    # -------------------------------------------------------------------------
    # Thumbnail Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def write_thumbnail(
        self,
        ep_id: str,
        entity_type: str,  # "track" or "identity"
        entity_id: str,
        data: bytes,
        *,
        content_type: str = "image/jpeg",
    ) -> UploadResult:
        """Write a thumbnail image."""
        ...

    @abstractmethod
    def get_thumbnail_url(
        self,
        ep_id: str,
        entity_type: str,
        entity_id: str,
    ) -> str | None:
        """Get URL for accessing a thumbnail."""
        ...

    # -------------------------------------------------------------------------
    # Manifest/Progress Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def write_manifest(
        self,
        ep_id: str,
        filename: str,
        data: bytes | str,
    ) -> UploadResult:
        """Write a manifest file (JSON, JSONL, etc.)."""
        ...

    @abstractmethod
    def read_manifest(self, ep_id: str, filename: str) -> bytes | None:
        """Read a manifest file."""
        ...

    @abstractmethod
    def write_progress(self, ep_id: str, progress: Dict[str, Any]) -> bool:
        """Write progress.json atomically."""
        ...

    @abstractmethod
    def read_progress(self, ep_id: str) -> Dict[str, Any] | None:
        """Read current progress.json."""
        ...

    # -------------------------------------------------------------------------
    # Lock Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def acquire_lock(
        self,
        ep_id: str,
        operation: str,
        job_id: str | None = None,
    ) -> Tuple[bool, str | None]:
        """Acquire a lock for an operation. Returns (success, error_message)."""
        ...

    @abstractmethod
    def release_lock(self, ep_id: str, operation: str) -> bool:
        """Release a held lock."""
        ...

    @abstractmethod
    def check_lock(self, ep_id: str, operation: str) -> LockInfo | None:
        """Check if a lock is held and by whom."""
        ...

    @abstractmethod
    def force_release_lock(self, ep_id: str, operation: str) -> bool:
        """Force release a lock (admin operation)."""
        ...

    # -------------------------------------------------------------------------
    # Listing/Cleanup Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def list_artifacts(
        self,
        ep_id: str,
        artifact_type: str,  # "frames", "crops", "thumbs", "manifests"
        *,
        prefix: str | None = None,
        limit: int = 1000,
    ) -> List[ArtifactInfo]:
        """List artifacts of a given type."""
        ...

    @abstractmethod
    def delete_artifacts(
        self,
        ep_id: str,
        artifact_type: str,
        *,
        prefix: str | None = None,
    ) -> int:
        """Delete artifacts. Returns count of deleted items."""
        ...

    @abstractmethod
    def get_storage_usage(self, ep_id: str) -> Dict[str, int]:
        """Get storage usage by artifact type in bytes."""
        ...

    # -------------------------------------------------------------------------
    # S3 Sync Operations (for hybrid mode)
    # -------------------------------------------------------------------------

    def sync_to_s3(
        self,
        ep_id: str,
        *,
        artifact_types: List[str] | None = None,
        progress_callback: Callable[[SyncStatus], None] | None = None,
    ) -> SyncStatus:
        """Sync local artifacts to S3. No-op for non-hybrid backends."""
        return SyncStatus(completed=True)

    def get_sync_status(self, ep_id: str) -> SyncStatus | None:
        """Get current S3 sync status. Returns None if not syncing."""
        return None


class LocalStorageBackend(StorageBackend):
    """Storage backend using local filesystem only."""

    def __init__(self) -> None:
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    @property
    def backend_type(self) -> str:
        return "local"

    @property
    def supports_presigned_urls(self) -> bool:
        return False

    def _get_frames_dir(self, ep_id: str) -> Path:
        return get_path(ep_id, "frames_root") / "frames"

    def _get_crops_dir(self, ep_id: str) -> Path:
        return get_path(ep_id, "frames_root") / "crops"

    def _get_thumbs_dir(self, ep_id: str, entity_type: str) -> Path:
        base = get_path(ep_id, "frames_root") / "thumbs"
        if entity_type == "track":
            return base / "tracks"
        elif entity_type == "identity":
            return base / "identities"
        return base

    def _get_manifests_dir(self, ep_id: str) -> Path:
        return get_path(ep_id, "detections").parent

    def _get_lock_path(self, ep_id: str, operation: str) -> Path:
        manifests_dir = self._get_manifests_dir(ep_id)
        manifests_dir.mkdir(parents=True, exist_ok=True)
        return manifests_dir / f".lock_{operation}"

    def _get_thread_lock(self, key: str) -> threading.Lock:
        with self._global_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def write_frame(
        self,
        ep_id: str,
        frame_idx: int,
        data: bytes,
        *,
        content_type: str = "image/jpeg",
    ) -> UploadResult:
        frames_dir = self._get_frames_dir(ep_id)
        frames_dir.mkdir(parents=True, exist_ok=True)
        path = frames_dir / f"frame_{frame_idx:06d}.jpg"
        try:
            path.write_bytes(data)
            return UploadResult(success=True, key=str(path), bytes_uploaded=len(data))
        except Exception as exc:
            LOGGER.warning("Failed to write frame %s: %s", path, exc)
            return UploadResult(success=False, error=str(exc))

    def write_crop(
        self,
        ep_id: str,
        track_id: int,
        frame_idx: int,
        data: bytes,
        *,
        content_type: str = "image/jpeg",
    ) -> UploadResult:
        crops_dir = self._get_crops_dir(ep_id)
        track_dir = crops_dir / f"track_{track_id:04d}"
        track_dir.mkdir(parents=True, exist_ok=True)
        path = track_dir / f"frame_{frame_idx:06d}.jpg"
        try:
            path.write_bytes(data)
            return UploadResult(success=True, key=str(path), bytes_uploaded=len(data))
        except Exception as exc:
            LOGGER.warning("Failed to write crop %s: %s", path, exc)
            return UploadResult(success=False, error=str(exc))

    def get_frame_url(self, ep_id: str, frame_idx: int) -> str | None:
        path = self._get_frames_dir(ep_id) / f"frame_{frame_idx:06d}.jpg"
        if path.exists():
            return str(path.resolve())
        return None

    def get_crop_url(self, ep_id: str, track_id: int, frame_idx: int) -> str | None:
        path = self._get_crops_dir(ep_id) / f"track_{track_id:04d}" / f"frame_{frame_idx:06d}.jpg"
        if path.exists():
            return str(path.resolve())
        return None

    def write_thumbnail(
        self,
        ep_id: str,
        entity_type: str,
        entity_id: str,
        data: bytes,
        *,
        content_type: str = "image/jpeg",
    ) -> UploadResult:
        thumbs_dir = self._get_thumbs_dir(ep_id, entity_type)
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        # Sanitize entity_id for filename
        safe_id = entity_id.replace("/", "_").replace("\\", "_")
        path = thumbs_dir / f"{safe_id}.jpg"
        try:
            path.write_bytes(data)
            return UploadResult(success=True, key=str(path), bytes_uploaded=len(data))
        except Exception as exc:
            LOGGER.warning("Failed to write thumbnail %s: %s", path, exc)
            return UploadResult(success=False, error=str(exc))

    def get_thumbnail_url(
        self,
        ep_id: str,
        entity_type: str,
        entity_id: str,
    ) -> str | None:
        thumbs_dir = self._get_thumbs_dir(ep_id, entity_type)
        safe_id = entity_id.replace("/", "_").replace("\\", "_")
        path = thumbs_dir / f"{safe_id}.jpg"
        if path.exists():
            return str(path.resolve())
        return None

    def write_manifest(
        self,
        ep_id: str,
        filename: str,
        data: bytes | str,
    ) -> UploadResult:
        manifests_dir = self._get_manifests_dir(ep_id)
        manifests_dir.mkdir(parents=True, exist_ok=True)
        path = manifests_dir / filename
        try:
            if isinstance(data, str):
                path.write_text(data, encoding="utf-8")
                size = len(data.encode("utf-8"))
            else:
                path.write_bytes(data)
                size = len(data)
            return UploadResult(success=True, key=str(path), bytes_uploaded=size)
        except Exception as exc:
            LOGGER.warning("Failed to write manifest %s: %s", path, exc)
            return UploadResult(success=False, error=str(exc))

    def read_manifest(self, ep_id: str, filename: str) -> bytes | None:
        path = self._get_manifests_dir(ep_id) / filename
        if not path.exists():
            return None
        try:
            return path.read_bytes()
        except Exception as exc:
            LOGGER.warning("Failed to read manifest %s: %s", path, exc)
            return None

    def write_progress(self, ep_id: str, progress: Dict[str, Any]) -> bool:
        manifests_dir = self._get_manifests_dir(ep_id)
        manifests_dir.mkdir(parents=True, exist_ok=True)
        path = manifests_dir / "progress.json"
        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(progress, sort_keys=True), encoding="utf-8")
            tmp_path.replace(path)
            return True
        except Exception as exc:
            LOGGER.warning("Failed to write progress %s: %s", path, exc)
            return False

    def read_progress(self, ep_id: str) -> Dict[str, Any] | None:
        path = self._get_manifests_dir(ep_id) / "progress.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Failed to read progress %s: %s", path, exc)
            return None

    def acquire_lock(
        self,
        ep_id: str,
        operation: str,
        job_id: str | None = None,
    ) -> Tuple[bool, str | None]:
        lock_path = self._get_lock_path(ep_id, operation)
        thread_lock = self._get_thread_lock(f"{ep_id}:{operation}")

        with thread_lock:
            # Check existing lock
            if lock_path.exists():
                try:
                    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
                    acquired_at = datetime.fromisoformat(lock_data.get("acquired_at", ""))
                    lock_info = LockInfo(
                        operation=operation,
                        holder_pid=lock_data.get("pid", 0),
                        holder_host=lock_data.get("hostname", ""),
                        acquired_at=acquired_at,
                        job_id=lock_data.get("job_id"),
                    )
                    if not lock_info.is_stale():
                        return False, f"Lock held by PID {lock_info.holder_pid} on {lock_info.holder_host}"
                    LOGGER.warning("Removing stale lock for %s:%s", ep_id, operation)
                except Exception:
                    pass  # Corrupt lock file, will be overwritten

            # Write new lock
            lock_data = {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "acquired_at": datetime.utcnow().isoformat(),
                "job_id": job_id,
            }
            try:
                lock_path.write_text(json.dumps(lock_data), encoding="utf-8")
                return True, None
            except Exception as exc:
                return False, str(exc)

    def release_lock(self, ep_id: str, operation: str) -> bool:
        lock_path = self._get_lock_path(ep_id, operation)
        thread_lock = self._get_thread_lock(f"{ep_id}:{operation}")

        with thread_lock:
            if not lock_path.exists():
                return True
            try:
                # Verify we own the lock
                lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
                if lock_data.get("pid") == os.getpid() and lock_data.get("hostname") == socket.gethostname():
                    lock_path.unlink()
                    return True
                LOGGER.warning("Cannot release lock not owned by this process")
                return False
            except Exception as exc:
                LOGGER.warning("Failed to release lock %s: %s", lock_path, exc)
                return False

    def check_lock(self, ep_id: str, operation: str) -> LockInfo | None:
        lock_path = self._get_lock_path(ep_id, operation)
        if not lock_path.exists():
            return None
        try:
            lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
            return LockInfo(
                operation=operation,
                holder_pid=lock_data.get("pid", 0),
                holder_host=lock_data.get("hostname", ""),
                acquired_at=datetime.fromisoformat(lock_data.get("acquired_at", "")),
                job_id=lock_data.get("job_id"),
            )
        except Exception:
            return None

    def force_release_lock(self, ep_id: str, operation: str) -> bool:
        lock_path = self._get_lock_path(ep_id, operation)
        thread_lock = self._get_thread_lock(f"{ep_id}:{operation}")

        with thread_lock:
            if not lock_path.exists():
                return True
            try:
                lock_path.unlink()
                return True
            except Exception as exc:
                LOGGER.warning("Failed to force release lock %s: %s", lock_path, exc)
                return False

    def list_artifacts(
        self,
        ep_id: str,
        artifact_type: str,
        *,
        prefix: str | None = None,
        limit: int = 1000,
    ) -> List[ArtifactInfo]:
        if artifact_type == "frames":
            base_dir = self._get_frames_dir(ep_id)
        elif artifact_type == "crops":
            base_dir = self._get_crops_dir(ep_id)
        elif artifact_type == "thumbs":
            base_dir = get_path(ep_id, "frames_root") / "thumbs"
        elif artifact_type == "manifests":
            base_dir = self._get_manifests_dir(ep_id)
        else:
            return []

        if not base_dir.exists():
            return []

        results: List[ArtifactInfo] = []
        pattern = f"{prefix}*" if prefix else "*"
        for path in base_dir.rglob(pattern):
            if not path.is_file():
                continue
            if len(results) >= limit:
                break
            try:
                stat = path.stat()
                results.append(
                    ArtifactInfo(
                        key=str(path.relative_to(base_dir)),
                        size=stat.st_size,
                        last_modified=datetime.fromtimestamp(stat.st_mtime),
                    )
                )
            except Exception:
                continue
        return results

    def delete_artifacts(
        self,
        ep_id: str,
        artifact_type: str,
        *,
        prefix: str | None = None,
    ) -> int:
        if artifact_type == "frames":
            base_dir = self._get_frames_dir(ep_id)
        elif artifact_type == "crops":
            base_dir = self._get_crops_dir(ep_id)
        elif artifact_type == "thumbs":
            base_dir = get_path(ep_id, "frames_root") / "thumbs"
        elif artifact_type == "manifests":
            base_dir = self._get_manifests_dir(ep_id)
        else:
            return 0

        if not base_dir.exists():
            return 0

        deleted = 0
        if prefix:
            pattern = f"{prefix}*"
            for path in base_dir.rglob(pattern):
                if path.is_file():
                    try:
                        path.unlink()
                        deleted += 1
                    except Exception:
                        pass
        else:
            # Delete entire directory
            try:
                shutil.rmtree(base_dir)
                deleted = -1  # Indicate full directory deletion
            except Exception:
                pass
        return deleted

    def get_storage_usage(self, ep_id: str) -> Dict[str, int]:
        usage: Dict[str, int] = {}
        for artifact_type in ["frames", "crops", "thumbs", "manifests"]:
            artifacts = self.list_artifacts(ep_id, artifact_type, limit=100000)
            usage[artifact_type] = sum(a.size for a in artifacts)
        return usage


class S3StorageBackend(StorageBackend):
    """Storage backend using S3/MinIO only (no local writes)."""

    def __init__(self, storage_service=None) -> None:
        """Initialize with optional StorageService instance.

        If not provided, will import and create one on first use.
        """
        self._storage_service = storage_service
        self._init_lock = threading.Lock()

    def _get_storage(self):
        """Lazy initialization of StorageService."""
        if self._storage_service is None:
            with self._init_lock:
                if self._storage_service is None:
                    from apps.api.services.storage import StorageService

                    self._storage_service = StorageService()
        return self._storage_service

    def _get_ep_ctx(self, ep_id: str):
        """Get EpisodeContext for an episode."""
        from apps.api.services.storage import episode_context_from_id

        return episode_context_from_id(ep_id)

    def _get_prefixes(self, ep_id: str) -> Dict[str, str]:
        """Get artifact prefixes for an episode."""
        from apps.api.services.storage import artifact_prefixes

        return artifact_prefixes(self._get_ep_ctx(ep_id))

    @property
    def backend_type(self) -> str:
        return "s3"

    @property
    def supports_presigned_urls(self) -> bool:
        return True

    def write_frame(
        self,
        ep_id: str,
        frame_idx: int,
        data: bytes,
        *,
        content_type: str = "image/jpeg",
    ) -> UploadResult:
        storage = self._get_storage()
        prefixes = self._get_prefixes(ep_id)
        key = f"{prefixes['frames']}frame_{frame_idx:06d}.jpg"
        try:
            if storage.upload_bytes(data, key, content_type=content_type):
                return UploadResult(success=True, key=key, bytes_uploaded=len(data))
            return UploadResult(success=False, error="Upload failed")
        except Exception as exc:
            return UploadResult(success=False, error=str(exc))

    def write_crop(
        self,
        ep_id: str,
        track_id: int,
        frame_idx: int,
        data: bytes,
        *,
        content_type: str = "image/jpeg",
    ) -> UploadResult:
        storage = self._get_storage()
        prefixes = self._get_prefixes(ep_id)
        key = f"{prefixes['crops']}track_{track_id:04d}/frame_{frame_idx:06d}.jpg"
        try:
            if storage.upload_bytes(data, key, content_type=content_type):
                return UploadResult(success=True, key=key, bytes_uploaded=len(data))
            return UploadResult(success=False, error="Upload failed")
        except Exception as exc:
            return UploadResult(success=False, error=str(exc))

    def get_frame_url(self, ep_id: str, frame_idx: int) -> str | None:
        storage = self._get_storage()
        prefixes = self._get_prefixes(ep_id)
        key = f"{prefixes['frames']}frame_{frame_idx:06d}.jpg"
        return storage.presign_get(key)

    def get_crop_url(self, ep_id: str, track_id: int, frame_idx: int) -> str | None:
        storage = self._get_storage()
        prefixes = self._get_prefixes(ep_id)
        key = f"{prefixes['crops']}track_{track_id:04d}/frame_{frame_idx:06d}.jpg"
        return storage.presign_get(key)

    def write_thumbnail(
        self,
        ep_id: str,
        entity_type: str,
        entity_id: str,
        data: bytes,
        *,
        content_type: str = "image/jpeg",
    ) -> UploadResult:
        storage = self._get_storage()
        prefixes = self._get_prefixes(ep_id)
        if entity_type == "track":
            key = f"{prefixes['thumbs_tracks']}{entity_id}.jpg"
        else:
            key = f"{prefixes['thumbs_identities']}{entity_id}.jpg"
        try:
            if storage.upload_bytes(data, key, content_type=content_type):
                return UploadResult(success=True, key=key, bytes_uploaded=len(data))
            return UploadResult(success=False, error="Upload failed")
        except Exception as exc:
            return UploadResult(success=False, error=str(exc))

    def get_thumbnail_url(
        self,
        ep_id: str,
        entity_type: str,
        entity_id: str,
    ) -> str | None:
        storage = self._get_storage()
        prefixes = self._get_prefixes(ep_id)
        if entity_type == "track":
            key = f"{prefixes['thumbs_tracks']}{entity_id}.jpg"
        else:
            key = f"{prefixes['thumbs_identities']}{entity_id}.jpg"
        return storage.presign_get(key)

    def write_manifest(
        self,
        ep_id: str,
        filename: str,
        data: bytes | str,
    ) -> UploadResult:
        storage = self._get_storage()
        prefixes = self._get_prefixes(ep_id)
        key = f"{prefixes['manifests']}{filename}"
        if isinstance(data, str):
            data = data.encode("utf-8")
        try:
            content_type = "application/json" if filename.endswith((".json", ".jsonl")) else "application/octet-stream"
            if storage.upload_bytes(data, key, content_type=content_type):
                return UploadResult(success=True, key=key, bytes_uploaded=len(data))
            return UploadResult(success=False, error="Upload failed")
        except Exception as exc:
            return UploadResult(success=False, error=str(exc))

    def read_manifest(self, ep_id: str, filename: str) -> bytes | None:
        storage = self._get_storage()
        prefixes = self._get_prefixes(ep_id)
        key = f"{prefixes['manifests']}{filename}"
        return storage.download_bytes(key)

    def write_progress(self, ep_id: str, progress: Dict[str, Any]) -> bool:
        result = self.write_manifest(ep_id, "progress.json", json.dumps(progress, sort_keys=True))
        return result.success

    def read_progress(self, ep_id: str) -> Dict[str, Any] | None:
        data = self.read_manifest(ep_id, "progress.json")
        if data is None:
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except Exception:
            return None

    def acquire_lock(
        self,
        ep_id: str,
        operation: str,
        job_id: str | None = None,
    ) -> Tuple[bool, str | None]:
        # S3 locks use a key-based approach with conditional writes
        # For now, delegate to Redis-based locking in the tasks module
        from apps.api.tasks import _acquire_lock

        success = _acquire_lock(ep_id, operation, job_id or "unknown")
        if success:
            return True, None
        return False, "Lock already held"

    def release_lock(self, ep_id: str, operation: str) -> bool:
        from apps.api.tasks import _release_lock

        _release_lock(ep_id, operation, "unknown")
        return True

    def check_lock(self, ep_id: str, operation: str) -> LockInfo | None:
        from apps.api.tasks import check_active_job

        active = check_active_job(ep_id, operation)
        if active and active.get("is_active"):
            return LockInfo(
                operation=operation,
                holder_pid=0,
                holder_host="redis",
                acquired_at=datetime.utcnow(),
                job_id=active.get("job_id"),
            )
        return None

    def force_release_lock(self, ep_id: str, operation: str) -> bool:
        from apps.api.tasks import _force_release_lock

        _force_release_lock(ep_id, operation)
        return True

    def list_artifacts(
        self,
        ep_id: str,
        artifact_type: str,
        *,
        prefix: str | None = None,
        limit: int = 1000,
    ) -> List[ArtifactInfo]:
        storage = self._get_storage()
        prefixes = self._get_prefixes(ep_id)

        type_map = {
            "frames": "frames",
            "crops": "crops",
            "thumbs": "thumbs_tracks",  # List tracks by default
            "manifests": "manifests",
        }
        prefix_key = type_map.get(artifact_type)
        if not prefix_key:
            return []

        base_prefix = prefixes.get(prefix_key, "")
        if prefix:
            base_prefix = f"{base_prefix}{prefix}"

        keys = storage.list_objects(base_prefix, max_items=limit)
        return [
            ArtifactInfo(
                key=k[len(prefixes.get(prefix_key, "")) :] if k.startswith(prefixes.get(prefix_key, "")) else k,
                size=0,  # S3 list doesn't include size by default
            )
            for k in keys
        ]

    def delete_artifacts(
        self,
        ep_id: str,
        artifact_type: str,
        *,
        prefix: str | None = None,
    ) -> int:
        storage = self._get_storage()
        prefixes = self._get_prefixes(ep_id)

        type_map = {
            "frames": "frames",
            "crops": "crops",
            "thumbs": "thumbs_tracks",
            "manifests": "manifests",
        }
        prefix_key = type_map.get(artifact_type)
        if not prefix_key:
            return 0

        base_prefix = prefixes.get(prefix_key, "")
        if prefix:
            base_prefix = f"{base_prefix}{prefix}"

        return storage.delete_prefix(base_prefix)

    def get_storage_usage(self, ep_id: str) -> Dict[str, int]:
        # S3 doesn't provide easy size aggregation, return zeros
        return {"frames": 0, "crops": 0, "thumbs": 0, "manifests": 0}


class HybridStorageBackend(StorageBackend):
    """Storage backend that writes locally and syncs to S3.

    This backend:
    - Writes all artifacts to local filesystem first
    - Optionally syncs to S3 in the background or on-demand
    - Serves reads from local if available, falls back to S3
    """

    def __init__(self, storage_service=None) -> None:
        self._local = LocalStorageBackend()
        self._s3 = S3StorageBackend(storage_service)
        self._sync_status: Dict[str, SyncStatus] = {}
        self._sync_lock = threading.Lock()

    @property
    def backend_type(self) -> str:
        return "hybrid"

    @property
    def supports_presigned_urls(self) -> bool:
        return True

    def write_frame(
        self,
        ep_id: str,
        frame_idx: int,
        data: bytes,
        *,
        content_type: str = "image/jpeg",
    ) -> UploadResult:
        # Write locally first
        return self._local.write_frame(ep_id, frame_idx, data, content_type=content_type)

    def write_crop(
        self,
        ep_id: str,
        track_id: int,
        frame_idx: int,
        data: bytes,
        *,
        content_type: str = "image/jpeg",
    ) -> UploadResult:
        return self._local.write_crop(ep_id, track_id, frame_idx, data, content_type=content_type)

    def get_frame_url(self, ep_id: str, frame_idx: int) -> str | None:
        # Try S3 first for presigned URLs, fall back to local
        url = self._s3.get_frame_url(ep_id, frame_idx)
        if url:
            return url
        return self._local.get_frame_url(ep_id, frame_idx)

    def get_crop_url(self, ep_id: str, track_id: int, frame_idx: int) -> str | None:
        url = self._s3.get_crop_url(ep_id, track_id, frame_idx)
        if url:
            return url
        return self._local.get_crop_url(ep_id, track_id, frame_idx)

    def write_thumbnail(
        self,
        ep_id: str,
        entity_type: str,
        entity_id: str,
        data: bytes,
        *,
        content_type: str = "image/jpeg",
    ) -> UploadResult:
        return self._local.write_thumbnail(ep_id, entity_type, entity_id, data, content_type=content_type)

    def get_thumbnail_url(
        self,
        ep_id: str,
        entity_type: str,
        entity_id: str,
    ) -> str | None:
        url = self._s3.get_thumbnail_url(ep_id, entity_type, entity_id)
        if url:
            return url
        return self._local.get_thumbnail_url(ep_id, entity_type, entity_id)

    def write_manifest(
        self,
        ep_id: str,
        filename: str,
        data: bytes | str,
    ) -> UploadResult:
        return self._local.write_manifest(ep_id, filename, data)

    def read_manifest(self, ep_id: str, filename: str) -> bytes | None:
        # Try local first, then S3
        result = self._local.read_manifest(ep_id, filename)
        if result is not None:
            return result
        return self._s3.read_manifest(ep_id, filename)

    def write_progress(self, ep_id: str, progress: Dict[str, Any]) -> bool:
        return self._local.write_progress(ep_id, progress)

    def read_progress(self, ep_id: str) -> Dict[str, Any] | None:
        result = self._local.read_progress(ep_id)
        if result is not None:
            return result
        return self._s3.read_progress(ep_id)

    def acquire_lock(
        self,
        ep_id: str,
        operation: str,
        job_id: str | None = None,
    ) -> Tuple[bool, str | None]:
        # Use local lock for hybrid mode
        return self._local.acquire_lock(ep_id, operation, job_id)

    def release_lock(self, ep_id: str, operation: str) -> bool:
        return self._local.release_lock(ep_id, operation)

    def check_lock(self, ep_id: str, operation: str) -> LockInfo | None:
        return self._local.check_lock(ep_id, operation)

    def force_release_lock(self, ep_id: str, operation: str) -> bool:
        return self._local.force_release_lock(ep_id, operation)

    def list_artifacts(
        self,
        ep_id: str,
        artifact_type: str,
        *,
        prefix: str | None = None,
        limit: int = 1000,
    ) -> List[ArtifactInfo]:
        # List from local
        return self._local.list_artifacts(ep_id, artifact_type, prefix=prefix, limit=limit)

    def delete_artifacts(
        self,
        ep_id: str,
        artifact_type: str,
        *,
        prefix: str | None = None,
    ) -> int:
        # Delete from both local and S3
        local_count = self._local.delete_artifacts(ep_id, artifact_type, prefix=prefix)
        s3_count = self._s3.delete_artifacts(ep_id, artifact_type, prefix=prefix)
        return max(local_count, s3_count)

    def get_storage_usage(self, ep_id: str) -> Dict[str, int]:
        return self._local.get_storage_usage(ep_id)

    def sync_to_s3(
        self,
        ep_id: str,
        *,
        artifact_types: List[str] | None = None,
        progress_callback: Callable[[SyncStatus], None] | None = None,
    ) -> SyncStatus:
        """Sync local artifacts to S3."""
        from apps.api.services.storage import StorageService, artifact_prefixes, episode_context_from_id

        status = SyncStatus(in_progress=True)

        with self._sync_lock:
            self._sync_status[ep_id] = status

        try:
            storage = self._s3._get_storage()
            ep_ctx = episode_context_from_id(ep_id)
            prefixes = artifact_prefixes(ep_ctx)

            types_to_sync = artifact_types or ["frames", "crops", "thumbs", "manifests"]

            for artifact_type in types_to_sync:
                local_artifacts = self._local.list_artifacts(ep_id, artifact_type, limit=100000)
                status.total_files += len(local_artifacts)

                for artifact in local_artifacts:
                    if progress_callback:
                        progress_callback(status)

                    # Determine S3 key based on artifact type
                    if artifact_type == "frames":
                        s3_prefix = prefixes["frames"]
                    elif artifact_type == "crops":
                        s3_prefix = prefixes["crops"]
                    elif artifact_type == "thumbs":
                        s3_prefix = prefixes["thumbs_tracks"]
                    else:
                        s3_prefix = prefixes["manifests"]

                    s3_key = f"{s3_prefix}{artifact.key}"

                    # Check if already exists in S3
                    if storage.object_exists(s3_key):
                        status.skipped_files += 1
                        continue

                    # Read local file and upload
                    if artifact_type == "frames":
                        local_path = self._local._get_frames_dir(ep_id) / artifact.key
                    elif artifact_type == "crops":
                        local_path = self._local._get_crops_dir(ep_id) / artifact.key
                    elif artifact_type == "thumbs":
                        local_path = get_path(ep_id, "frames_root") / "thumbs" / artifact.key
                    else:
                        local_path = self._local._get_manifests_dir(ep_id) / artifact.key

                    if not local_path.exists():
                        status.skipped_files += 1
                        continue

                    try:
                        data = local_path.read_bytes()
                        if storage.upload_bytes(data, s3_key):
                            status.uploaded_files += 1
                            status.bytes_uploaded += len(data)
                        else:
                            status.failed_files += 1
                            status.errors.append(f"Failed to upload {artifact.key}")
                    except Exception as exc:
                        status.failed_files += 1
                        status.errors.append(f"Error uploading {artifact.key}: {exc}")

            status.in_progress = False
            status.completed = True

        except Exception as exc:
            status.in_progress = False
            status.errors.append(f"Sync failed: {exc}")
            LOGGER.exception("S3 sync failed for %s", ep_id)

        with self._sync_lock:
            self._sync_status[ep_id] = status

        return status

    def get_sync_status(self, ep_id: str) -> SyncStatus | None:
        with self._sync_lock:
            return self._sync_status.get(ep_id)


# Factory function to get the appropriate backend
_backend_instance: StorageBackend | None = None
_backend_lock = threading.Lock()
_backend_config_result: Any = None  # StorageConfigResult


def get_storage_backend(force_type: str | None = None) -> StorageBackend:
    """Get the configured storage backend instance.

    Uses validated configuration from apps.api.services.validation module.
    Invalid or misconfigured backends are clearly logged and may fallback to local.

    Args:
        force_type: Override the STORAGE_BACKEND env var ("local", "s3", "hybrid")

    Returns:
        StorageBackend instance based on validated configuration
    """
    global _backend_instance, _backend_config_result

    # Use validation module for config validation (A16: STORAGE_BACKEND check)
    try:
        from apps.api.services.validation import validate_storage_backend_config

        config_result = validate_storage_backend_config()
        _backend_config_result = config_result

        if config_result.is_fallback:
            LOGGER.warning(
                "[storage-backend] Using fallback configuration: %s (original: %s)",
                config_result.backend,
                config_result.original_backend,
            )
        elif config_result.warnings:
            for warning in config_result.warnings:
                LOGGER.warning("[storage-backend] %s", warning)

        backend_type = force_type or config_result.backend

    except ImportError:
        # Fallback if validation module not available
        backend_type = force_type or os.environ.get("STORAGE_BACKEND", "local").lower()

    # Return cached instance if type matches
    if _backend_instance is not None:
        if _backend_instance.backend_type == backend_type:
            return _backend_instance

    with _backend_lock:
        # Double-check after acquiring lock
        if _backend_instance is not None and _backend_instance.backend_type == backend_type:
            return _backend_instance

        if backend_type == "local":
            _backend_instance = LocalStorageBackend()
        elif backend_type in ("s3", "minio"):
            _backend_instance = S3StorageBackend()
        elif backend_type == "hybrid":
            _backend_instance = HybridStorageBackend()
        else:
            LOGGER.error(
                "[storage-backend] Unknown STORAGE_BACKEND '%s', defaulting to local. "
                "Valid options: local, s3, minio, hybrid",
                backend_type,
            )
            _backend_instance = LocalStorageBackend()

        LOGGER.info(
            "[storage-backend] Initialized %s backend",
            _backend_instance.backend_type,
        )

        return _backend_instance


def get_storage_backend_status() -> Dict[str, Any]:
    """Get current storage backend status including validation results.

    Returns config validation result and current backend info for UI display.
    """
    global _backend_config_result

    # Ensure validation has run
    try:
        from apps.api.services.validation import validate_storage_backend_config

        config_result = validate_storage_backend_config()
    except ImportError:
        config_result = None

    backend = get_storage_backend()

    return {
        "backend_type": backend.backend_type,
        "supports_presigned_urls": backend.supports_presigned_urls,
        "validation": config_result.to_dict() if config_result else None,
    }


def reset_storage_backend() -> None:
    """Reset the cached backend instance. Useful for testing."""
    global _backend_instance
    with _backend_lock:
        _backend_instance = None


__all__ = [
    "ArtifactInfo",
    "HybridStorageBackend",
    "LocalStorageBackend",
    "LockInfo",
    "S3StorageBackend",
    "StorageBackend",
    "SyncStatus",
    "UploadResult",
    "get_storage_backend",
    "reset_storage_backend",
]
