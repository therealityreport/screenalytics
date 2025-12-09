"""Enhanced lock management with crash recovery and stale detection.

This module provides:
- B23: Improved lock cleanup on crash
- B24: Crash recovery / resume support
- Enhanced lock handling with explicit owner identity
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

LOGGER = logging.getLogger(__name__)

# Lock configuration
LOCK_STALE_SECONDS_DEFAULT = 3600  # 1 hour - fallback if liveness check fails
LOCK_STALE_SECONDS_QUICK = 300  # 5 minutes - for quick operations
LOCK_HEARTBEAT_INTERVAL = 60  # Heartbeat every minute
LOCK_PID_CHECK_ENABLED = True  # Enable PID-based liveness checks


@dataclass
class LockMetadata:
    """Enhanced lock metadata with owner identity."""

    operation: str
    holder_pid: int
    holder_host: str
    acquired_at: datetime
    job_id: Optional[str] = None
    heartbeat_at: Optional[datetime] = None
    expected_duration_sec: Optional[int] = None
    lock_version: int = 2  # Version for forward compatibility

    def is_stale(
        self,
        max_age_seconds: int = LOCK_STALE_SECONDS_DEFAULT,
        use_heartbeat: bool = True,
    ) -> bool:
        """Check if lock is stale based on age or heartbeat."""
        now = datetime.utcnow()

        # If heartbeat is available and recent, lock is not stale
        if use_heartbeat and self.heartbeat_at:
            heartbeat_age = (now - self.heartbeat_at).total_seconds()
            if heartbeat_age < LOCK_HEARTBEAT_INTERVAL * 3:  # 3 missed heartbeats
                return False

        # Check acquisition time
        age = (now - self.acquired_at).total_seconds()
        return age > max_age_seconds

    def is_owner_alive(self) -> bool:
        """Check if the lock owner process is still running.

        Returns True if:
        - Owner is on a different host (can't verify)
        - Owner process is still running on this host
        """
        current_host = socket.gethostname()

        # Can't verify remote processes
        if self.holder_host != current_host:
            return True  # Assume alive, rely on heartbeat/timeout

        # Check if PID exists on this host
        if not LOCK_PID_CHECK_ENABLED:
            return True

        try:
            # Check if process exists (cross-platform)
            os.kill(self.holder_pid, 0)
            return True
        except OSError:
            return False
        except Exception:
            return True  # Assume alive on error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "pid": self.holder_pid,
            "hostname": self.holder_host,
            "acquired_at": self.acquired_at.isoformat(),
            "heartbeat_at": self.heartbeat_at.isoformat() if self.heartbeat_at else None,
            "job_id": self.job_id,
            "expected_duration_sec": self.expected_duration_sec,
            "lock_version": self.lock_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], operation: str) -> "LockMetadata":
        acquired_at_str = data.get("acquired_at", "")
        try:
            acquired_at = datetime.fromisoformat(acquired_at_str)
        except (ValueError, TypeError):
            acquired_at = datetime.utcnow()

        heartbeat_at = None
        heartbeat_str = data.get("heartbeat_at")
        if heartbeat_str:
            try:
                heartbeat_at = datetime.fromisoformat(heartbeat_str)
            except (ValueError, TypeError):
                pass

        return cls(
            operation=operation,
            holder_pid=data.get("pid", 0),
            holder_host=data.get("hostname", ""),
            acquired_at=acquired_at,
            heartbeat_at=heartbeat_at,
            job_id=data.get("job_id"),
            expected_duration_sec=data.get("expected_duration_sec"),
            lock_version=data.get("lock_version", 1),
        )


@dataclass
class LockAcquisitionResult:
    """Result of a lock acquisition attempt."""

    success: bool
    lock_path: Optional[str] = None
    error: Optional[str] = None
    existing_lock: Optional[LockMetadata] = None
    was_stolen: bool = False
    steal_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "lock_path": self.lock_path,
            "error": self.error,
            "existing_lock": self.existing_lock.to_dict() if self.existing_lock else None,
            "was_stolen": self.was_stolen,
            "steal_reason": self.steal_reason,
        }


class EnhancedLockManager:
    """Enhanced lock manager with PID-based liveness checks and heartbeats.

    This implements requirements:
    - B23: Improve lock cleanup on crash
    - B24: Support for crash recovery / resume
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        """Initialize lock manager.

        Args:
            base_dir: Base directory for lock files. Defaults to DATA_ROOT/locks.
        """
        if base_dir is None:
            data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
            base_dir = data_root / "locks"
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

        self._thread_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._active_locks: Dict[str, LockMetadata] = {}
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()

    def _get_lock_path(self, ep_id: str, operation: str) -> Path:
        """Get the lock file path for an episode/operation."""
        ep_dir = self._base_dir / ep_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        return ep_dir / f".lock_{operation}"

    def _get_thread_lock(self, key: str) -> threading.Lock:
        """Get or create a thread lock for the given key."""
        with self._global_lock:
            if key not in self._thread_locks:
                self._thread_locks[key] = threading.Lock()
            return self._thread_locks[key]

    def acquire_lock(
        self,
        ep_id: str,
        operation: str,
        job_id: Optional[str] = None,
        expected_duration_sec: Optional[int] = None,
        allow_steal: bool = True,
        steal_if_owner_dead: bool = True,
    ) -> LockAcquisitionResult:
        """Acquire a lock for an operation.

        Args:
            ep_id: Episode ID
            operation: Operation name (e.g., "detect_track", "audio_pipeline")
            job_id: Optional job ID for tracking
            expected_duration_sec: Expected operation duration for smarter timeout
            allow_steal: Allow stealing stale locks
            steal_if_owner_dead: Steal lock if owner PID is dead

        Returns:
            LockAcquisitionResult with success status and details
        """
        lock_path = self._get_lock_path(ep_id, operation)
        thread_lock = self._get_thread_lock(f"{ep_id}:{operation}")
        lock_key = f"{ep_id}:{operation}"

        with thread_lock:
            # Check for existing lock
            existing_lock = self._read_lock_file(lock_path, operation)

            if existing_lock:
                # Determine if we can steal the lock
                can_steal = False
                steal_reason = None

                # Check if owner process is dead
                if steal_if_owner_dead and not existing_lock.is_owner_alive():
                    can_steal = True
                    steal_reason = f"Owner process (PID {existing_lock.holder_pid}) is no longer running"

                # Check if lock is stale by time
                if not can_steal and allow_steal:
                    max_age = existing_lock.expected_duration_sec or LOCK_STALE_SECONDS_DEFAULT
                    # Add 50% buffer to expected duration
                    max_age = int(max_age * 1.5)
                    if existing_lock.is_stale(max_age_seconds=max_age):
                        can_steal = True
                        steal_reason = f"Lock is stale (acquired {existing_lock.acquired_at.isoformat()})"

                if can_steal:
                    LOGGER.warning(
                        "[lock] Stealing stale lock for %s:%s - %s (previous holder: PID %d on %s)",
                        ep_id,
                        operation,
                        steal_reason,
                        existing_lock.holder_pid,
                        existing_lock.holder_host,
                    )
                else:
                    # Cannot acquire - lock is held
                    return LockAcquisitionResult(
                        success=False,
                        error=f"Lock held by PID {existing_lock.holder_pid} on {existing_lock.holder_host}",
                        existing_lock=existing_lock,
                    )

            # Write new lock
            new_lock = LockMetadata(
                operation=operation,
                holder_pid=os.getpid(),
                holder_host=socket.gethostname(),
                acquired_at=datetime.utcnow(),
                heartbeat_at=datetime.utcnow(),
                job_id=job_id,
                expected_duration_sec=expected_duration_sec,
            )

            try:
                self._write_lock_file(lock_path, new_lock)
                self._active_locks[lock_key] = new_lock

                # Start heartbeat thread if not running
                self._ensure_heartbeat_thread()

                was_stolen = existing_lock is not None
                return LockAcquisitionResult(
                    success=True,
                    lock_path=str(lock_path),
                    was_stolen=was_stolen,
                    steal_reason=steal_reason if was_stolen else None,
                )

            except Exception as exc:
                LOGGER.exception("[lock] Failed to write lock file %s", lock_path)
                return LockAcquisitionResult(
                    success=False,
                    error=str(exc),
                )

    def release_lock(self, ep_id: str, operation: str) -> bool:
        """Release a held lock.

        Only releases if the current process owns the lock.

        Returns:
            True if lock was released, False otherwise
        """
        lock_path = self._get_lock_path(ep_id, operation)
        thread_lock = self._get_thread_lock(f"{ep_id}:{operation}")
        lock_key = f"{ep_id}:{operation}"

        with thread_lock:
            if not lock_path.exists():
                self._active_locks.pop(lock_key, None)
                return True

            try:
                existing = self._read_lock_file(lock_path, operation)
                if existing is None:
                    self._active_locks.pop(lock_key, None)
                    return True

                # Verify we own the lock
                current_pid = os.getpid()
                current_host = socket.gethostname()

                if existing.holder_pid != current_pid or existing.holder_host != current_host:
                    LOGGER.warning(
                        "[lock] Cannot release lock for %s:%s - owned by PID %d on %s, "
                        "we are PID %d on %s",
                        ep_id,
                        operation,
                        existing.holder_pid,
                        existing.holder_host,
                        current_pid,
                        current_host,
                    )
                    return False

                lock_path.unlink()
                self._active_locks.pop(lock_key, None)
                LOGGER.debug("[lock] Released lock for %s:%s", ep_id, operation)
                return True

            except Exception as exc:
                LOGGER.exception("[lock] Failed to release lock %s", lock_path)
                return False

    def force_release_lock(self, ep_id: str, operation: str) -> bool:
        """Force release a lock regardless of owner.

        Use with caution - should only be called by admin operations.
        """
        lock_path = self._get_lock_path(ep_id, operation)
        thread_lock = self._get_thread_lock(f"{ep_id}:{operation}")
        lock_key = f"{ep_id}:{operation}"

        with thread_lock:
            if not lock_path.exists():
                self._active_locks.pop(lock_key, None)
                return True

            try:
                # Log who owned it before force release
                existing = self._read_lock_file(lock_path, operation)
                if existing:
                    LOGGER.warning(
                        "[lock] Force releasing lock for %s:%s - was owned by PID %d on %s",
                        ep_id,
                        operation,
                        existing.holder_pid,
                        existing.holder_host,
                    )

                lock_path.unlink()
                self._active_locks.pop(lock_key, None)
                return True

            except Exception as exc:
                LOGGER.exception("[lock] Failed to force release lock %s", lock_path)
                return False

    def check_lock(self, ep_id: str, operation: str) -> Optional[LockMetadata]:
        """Check if a lock is held and return its metadata."""
        lock_path = self._get_lock_path(ep_id, operation)
        return self._read_lock_file(lock_path, operation)

    def update_heartbeat(self, ep_id: str, operation: str) -> bool:
        """Update the heartbeat timestamp for an active lock.

        Should be called periodically by long-running jobs.
        """
        lock_path = self._get_lock_path(ep_id, operation)
        thread_lock = self._get_thread_lock(f"{ep_id}:{operation}")

        with thread_lock:
            existing = self._read_lock_file(lock_path, operation)
            if existing is None:
                return False

            # Verify we own the lock
            if existing.holder_pid != os.getpid() or existing.holder_host != socket.gethostname():
                return False

            # Update heartbeat
            existing.heartbeat_at = datetime.utcnow()
            self._write_lock_file(lock_path, existing)
            return True

    def list_locks(self, ep_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all locks, optionally filtered by episode."""
        locks = []

        if ep_id:
            ep_dir = self._base_dir / ep_id
            if ep_dir.exists():
                for lock_file in ep_dir.glob(".lock_*"):
                    operation = lock_file.name[6:]  # Remove ".lock_" prefix
                    metadata = self._read_lock_file(lock_file, operation)
                    if metadata:
                        locks.append({
                            "ep_id": ep_id,
                            "operation": operation,
                            "path": str(lock_file),
                            **metadata.to_dict(),
                        })
        else:
            # List all episode directories
            for ep_dir in self._base_dir.iterdir():
                if ep_dir.is_dir():
                    for lock_file in ep_dir.glob(".lock_*"):
                        operation = lock_file.name[6:]
                        metadata = self._read_lock_file(lock_file, operation)
                        if metadata:
                            locks.append({
                                "ep_id": ep_dir.name,
                                "operation": operation,
                                "path": str(lock_file),
                                **metadata.to_dict(),
                            })

        return locks

    def cleanup_stale_locks(
        self,
        max_age_seconds: int = LOCK_STALE_SECONDS_DEFAULT,
        ep_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Clean up stale locks.

        Returns list of cleaned up locks with their metadata.
        """
        cleaned = []
        locks = self.list_locks(ep_id)

        for lock_info in locks:
            lock_ep_id = lock_info["ep_id"]
            operation = lock_info["operation"]
            lock_path = Path(lock_info["path"])

            metadata = self._read_lock_file(lock_path, operation)
            if metadata is None:
                continue

            should_clean = False
            reason = None

            # Check if owner is dead
            if not metadata.is_owner_alive():
                should_clean = True
                reason = f"Owner process (PID {metadata.holder_pid}) is dead"

            # Check if stale by time
            elif metadata.is_stale(max_age_seconds=max_age_seconds):
                should_clean = True
                reason = f"Lock is stale (age > {max_age_seconds}s)"

            if should_clean:
                LOGGER.info(
                    "[lock-cleanup] Removing stale lock %s:%s - %s",
                    lock_ep_id,
                    operation,
                    reason,
                )
                try:
                    lock_path.unlink()
                    cleaned.append({
                        **lock_info,
                        "cleanup_reason": reason,
                    })
                except Exception as exc:
                    LOGGER.warning(
                        "[lock-cleanup] Failed to remove %s: %s",
                        lock_path,
                        exc,
                    )

        return cleaned

    def _read_lock_file(self, path: Path, operation: str) -> Optional[LockMetadata]:
        """Read and parse a lock file."""
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return LockMetadata.from_dict(data, operation)
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.warning("[lock] Failed to read lock file %s: %s", path, exc)
            return None

    def _write_lock_file(self, path: Path, metadata: LockMetadata) -> None:
        """Write lock metadata to file."""
        path.write_text(json.dumps(metadata.to_dict()), encoding="utf-8")

    def _ensure_heartbeat_thread(self) -> None:
        """Start heartbeat thread if not already running."""
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return

        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="lock-heartbeat",
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        """Background thread to update heartbeats for active locks."""
        while not self._heartbeat_stop.wait(timeout=LOCK_HEARTBEAT_INTERVAL):
            try:
                # Update heartbeats for all locks we own
                for lock_key in list(self._active_locks.keys()):
                    parts = lock_key.split(":", 1)
                    if len(parts) == 2:
                        ep_id, operation = parts
                        self.update_heartbeat(ep_id, operation)
            except Exception as exc:
                LOGGER.warning("[lock-heartbeat] Error updating heartbeats: %s", exc)

    def shutdown(self) -> None:
        """Stop the heartbeat thread and release all locks owned by this process."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)

        # Release all our locks
        for lock_key in list(self._active_locks.keys()):
            parts = lock_key.split(":", 1)
            if len(parts) == 2:
                ep_id, operation = parts
                self.release_lock(ep_id, operation)


# =============================================================================
# Checkpoint/Resume Support (B24)
# =============================================================================


@dataclass
class CheckpointData:
    """Checkpoint data for crash recovery."""

    ep_id: str
    operation: str
    stage: str
    created_at: datetime
    config_hash: str
    progress: Dict[str, Any] = field(default_factory=dict)
    completed_ranges: List[Tuple[int, int]] = field(default_factory=list)  # (start, end) frame ranges
    partial_outputs: Dict[str, str] = field(default_factory=dict)  # output_type -> path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ep_id": self.ep_id,
            "operation": self.operation,
            "stage": self.stage,
            "created_at": self.created_at.isoformat(),
            "config_hash": self.config_hash,
            "progress": self.progress,
            "completed_ranges": self.completed_ranges,
            "partial_outputs": self.partial_outputs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointData":
        created_at_str = data.get("created_at", "")
        try:
            created_at = datetime.fromisoformat(created_at_str)
        except (ValueError, TypeError):
            created_at = datetime.utcnow()

        return cls(
            ep_id=data.get("ep_id", ""),
            operation=data.get("operation", ""),
            stage=data.get("stage", ""),
            created_at=created_at,
            config_hash=data.get("config_hash", ""),
            progress=data.get("progress", {}),
            completed_ranges=[tuple(r) for r in data.get("completed_ranges", [])],
            partial_outputs=data.get("partial_outputs", {}),
        )


class CheckpointManager:
    """Manage checkpoints for crash recovery and job resumption.

    This implements requirement B24: No crash recovery / resume.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        if base_dir is None:
            data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
            base_dir = data_root / "checkpoints"
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, ep_id: str, operation: str) -> Path:
        return self._base_dir / ep_id / f"checkpoint_{operation}.json"

    def save_checkpoint(
        self,
        ep_id: str,
        operation: str,
        stage: str,
        config_hash: str,
        progress: Dict[str, Any],
        completed_ranges: Optional[List[Tuple[int, int]]] = None,
        partial_outputs: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Save a checkpoint for crash recovery.

        Args:
            ep_id: Episode ID
            operation: Operation name (e.g., "detect_track")
            stage: Current stage within operation
            config_hash: Hash of job configuration for validation
            progress: Progress data (frames done, etc.)
            completed_ranges: List of (start, end) frame ranges completed
            partial_outputs: Map of output type to file path

        Returns:
            True if checkpoint saved successfully
        """
        checkpoint = CheckpointData(
            ep_id=ep_id,
            operation=operation,
            stage=stage,
            created_at=datetime.utcnow(),
            config_hash=config_hash,
            progress=progress,
            completed_ranges=completed_ranges or [],
            partial_outputs=partial_outputs or {},
        )

        path = self._checkpoint_path(ep_id, operation)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Write atomically
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(checkpoint.to_dict(), indent=2), encoding="utf-8")
            tmp_path.replace(path)
            LOGGER.debug(
                "[checkpoint] Saved checkpoint for %s:%s at stage %s",
                ep_id,
                operation,
                stage,
            )
            return True
        except Exception as exc:
            LOGGER.warning("[checkpoint] Failed to save checkpoint: %s", exc)
            return False

    def load_checkpoint(self, ep_id: str, operation: str) -> Optional[CheckpointData]:
        """Load existing checkpoint if available."""
        path = self._checkpoint_path(ep_id, operation)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CheckpointData.from_dict(data)
        except Exception as exc:
            LOGGER.warning("[checkpoint] Failed to load checkpoint %s: %s", path, exc)
            return None

    def validate_checkpoint(
        self,
        checkpoint: CheckpointData,
        config_hash: str,
        max_age_hours: int = 24,
    ) -> Tuple[bool, Optional[str]]:
        """Validate if a checkpoint is usable for resumption.

        Args:
            checkpoint: The checkpoint to validate
            config_hash: Current job's config hash (must match)
            max_age_hours: Maximum age of checkpoint to consider valid

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Check config hash matches
        if checkpoint.config_hash != config_hash:
            return False, "Configuration has changed since checkpoint"

        # Check age
        age = datetime.utcnow() - checkpoint.created_at
        if age > timedelta(hours=max_age_hours):
            return False, f"Checkpoint too old ({age.total_seconds() / 3600:.1f} hours)"

        # Validate partial outputs still exist
        for output_type, output_path in checkpoint.partial_outputs.items():
            if not Path(output_path).exists():
                return False, f"Partial output missing: {output_type}"

        return True, None

    def delete_checkpoint(self, ep_id: str, operation: str) -> bool:
        """Delete a checkpoint after successful completion."""
        path = self._checkpoint_path(ep_id, operation)
        if not path.exists():
            return True

        try:
            path.unlink()
            LOGGER.debug("[checkpoint] Deleted checkpoint for %s:%s", ep_id, operation)
            return True
        except Exception as exc:
            LOGGER.warning("[checkpoint] Failed to delete checkpoint: %s", exc)
            return False

    def get_resume_point(
        self,
        checkpoint: CheckpointData,
    ) -> Tuple[int, str]:
        """Determine where to resume from based on checkpoint.

        Returns:
            Tuple of (start_frame, stage)
        """
        # Find the first incomplete frame
        if not checkpoint.completed_ranges:
            return 0, checkpoint.stage

        # Sort ranges and find first gap
        sorted_ranges = sorted(checkpoint.completed_ranges)
        last_end = 0
        for start, end in sorted_ranges:
            if start > last_end:
                # Gap found - resume from here
                return last_end, checkpoint.stage
            last_end = max(last_end, end)

        # All ranges consecutive - resume from end
        return last_end, checkpoint.stage


# =============================================================================
# Module-level instances and exports
# =============================================================================

# Global lock manager instance
_lock_manager: Optional[EnhancedLockManager] = None
_checkpoint_manager: Optional[CheckpointManager] = None


def get_lock_manager() -> EnhancedLockManager:
    """Get the global lock manager instance."""
    global _lock_manager
    if _lock_manager is None:
        _lock_manager = EnhancedLockManager()
    return _lock_manager


def get_checkpoint_manager() -> CheckpointManager:
    """Get the global checkpoint manager instance."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager()
    return _checkpoint_manager


__all__ = [
    # Lock management
    "LockMetadata",
    "LockAcquisitionResult",
    "EnhancedLockManager",
    "get_lock_manager",
    # Checkpointing
    "CheckpointData",
    "CheckpointManager",
    "get_checkpoint_manager",
    # Constants
    "LOCK_STALE_SECONDS_DEFAULT",
    "LOCK_HEARTBEAT_INTERVAL",
]
