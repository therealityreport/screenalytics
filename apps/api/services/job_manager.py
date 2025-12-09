"""Job management with dry run, cancellation, priority, and resource limits.

This module provides:
- E35: Dry run mode
- E36: Job cancellation cleanup
- E37: Job queue priority
- E38: Resource limits
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import multiprocessing

LOGGER = logging.getLogger(__name__)


# =============================================================================
# E35: Dry Run Mode
# =============================================================================


@dataclass
class DryRunEstimate:
    """Estimate from a dry run."""

    # Frame estimates
    total_frames: int
    sampled_frames: int
    effective_fps: float

    # Storage estimates (bytes)
    frames_storage: int
    crops_storage: int
    thumbs_storage: int
    manifests_storage: int
    total_storage: int

    # Time estimates (seconds)
    estimated_runtime_sec: float
    estimated_runtime_str: str

    # Resource estimates
    estimated_faces: int
    estimated_tracks: int
    estimated_s3_ops: int

    # Config summary
    config_summary: Dict[str, Any] = field(default_factory=dict)

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames": {
                "total": self.total_frames,
                "sampled": self.sampled_frames,
                "effective_fps": round(self.effective_fps, 2),
            },
            "storage_bytes": {
                "frames": self.frames_storage,
                "crops": self.crops_storage,
                "thumbs": self.thumbs_storage,
                "manifests": self.manifests_storage,
                "total": self.total_storage,
            },
            "storage_human": {
                "frames": _format_bytes(self.frames_storage),
                "crops": _format_bytes(self.crops_storage),
                "thumbs": _format_bytes(self.thumbs_storage),
                "manifests": _format_bytes(self.manifests_storage),
                "total": _format_bytes(self.total_storage),
            },
            "runtime": {
                "estimated_seconds": round(self.estimated_runtime_sec, 1),
                "estimated_human": self.estimated_runtime_str,
            },
            "estimates": {
                "faces": self.estimated_faces,
                "tracks": self.estimated_tracks,
                "s3_operations": self.estimated_s3_ops,
            },
            "config": self.config_summary,
            "warnings": self.warnings,
        }


def _format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


# Size estimates (bytes)
FRAME_JPEG_SIZE = 220_000  # ~220KB per full frame
CROP_JPEG_SIZE = 40_000  # ~40KB per crop
THUMB_JPEG_SIZE = 20_000  # ~20KB per thumbnail
MANIFEST_ENTRY_SIZE = 500  # ~500 bytes per JSON line

# Performance estimates (frames per second by device)
DEVICE_PERFORMANCE: Dict[str, float] = {
    "cpu": 10.0,  # Slowest
    "mps": 25.0,  # Apple Silicon
    "coreml": 35.0,  # CoreML optimized
    "cuda": 50.0,  # NVIDIA GPU
}


def dry_run_detect_track(
    ep_id: str,
    duration_sec: float,
    video_fps: float,
    config: Dict[str, Any],
) -> DryRunEstimate:
    """Perform a dry run to estimate job output without executing.

    This implements requirement E35: No dry run mode.

    Args:
        ep_id: Episode ID
        duration_sec: Video duration in seconds
        video_fps: Video frames per second
        config: Job configuration

    Returns:
        DryRunEstimate with all projections
    """
    warnings = []

    # Extract config values
    stride = config.get("frame_stride", config.get("stride", 6))
    save_frames = config.get("save_frames", False)
    save_crops = config.get("save_crops", True)
    device = config.get("device", "auto")
    jpeg_quality = config.get("jpeg_quality", 72)

    # Normalize device for performance lookup
    device_key = device.lower()
    if device_key == "auto":
        # Assume moderate performance
        device_key = "mps" if os.uname().machine.startswith("arm") else "cpu"

    # Calculate frame counts
    total_frames = int(duration_sec * video_fps)
    sampled_frames = max(1, total_frames // stride)
    effective_fps = video_fps / stride

    if effective_fps < 0.5:
        warnings.append(
            f"Very sparse sampling ({effective_fps:.2f} FPS effective) - may miss faces"
        )
    elif effective_fps > 30:
        warnings.append(
            f"Dense sampling ({effective_fps:.1f} FPS effective) - consider increasing stride"
        )

    # Estimate faces and tracks
    # Rough heuristics based on typical episode content
    avg_faces_per_frame = 1.5  # Average faces visible per sampled frame
    estimated_faces = int(sampled_frames * avg_faces_per_frame)

    # Assume tracks form from ~10% of total detections
    # (same person across frames consolidates into tracks)
    track_consolidation = max(1, sampled_frames // 30)  # Rough frames per track
    estimated_tracks = max(1, estimated_faces // track_consolidation)

    # Storage estimates
    frames_storage = sampled_frames * FRAME_JPEG_SIZE if save_frames else 0
    crops_storage = estimated_faces * CROP_JPEG_SIZE if save_crops else 0
    thumbs_storage = estimated_tracks * THUMB_JPEG_SIZE
    manifests_storage = (estimated_faces + estimated_tracks + sampled_frames) * MANIFEST_ENTRY_SIZE
    total_storage = frames_storage + crops_storage + thumbs_storage + manifests_storage

    # Adjust for JPEG quality
    quality_factor = jpeg_quality / 72.0  # 72 is baseline
    frames_storage = int(frames_storage * quality_factor)
    crops_storage = int(crops_storage * quality_factor)
    thumbs_storage = int(thumbs_storage * quality_factor)
    total_storage = int(total_storage * quality_factor)

    # Time estimate
    base_fps = DEVICE_PERFORMANCE.get(device_key, 10.0)
    estimated_runtime_sec = sampled_frames / base_fps

    # Add overhead for embedding, clustering, etc.
    estimated_runtime_sec *= 1.3  # 30% overhead

    # S3 operations estimate
    estimated_s3_ops = 0
    storage_backend = os.environ.get("STORAGE_BACKEND", "local")
    if storage_backend in ("s3", "minio", "hybrid"):
        estimated_s3_ops = (
            (sampled_frames if save_frames else 0)
            + (estimated_faces if save_crops else 0)
            + estimated_tracks  # Thumbnails
            + 10  # Manifests and metadata
        )

    # Config summary
    config_summary = {
        "stride": stride,
        "save_frames": save_frames,
        "save_crops": save_crops,
        "device": device,
        "jpeg_quality": jpeg_quality,
        "storage_backend": storage_backend,
    }

    return DryRunEstimate(
        total_frames=total_frames,
        sampled_frames=sampled_frames,
        effective_fps=effective_fps,
        frames_storage=frames_storage,
        crops_storage=crops_storage,
        thumbs_storage=thumbs_storage,
        manifests_storage=manifests_storage,
        total_storage=total_storage,
        estimated_runtime_sec=estimated_runtime_sec,
        estimated_runtime_str=_format_duration(estimated_runtime_sec),
        estimated_faces=estimated_faces,
        estimated_tracks=estimated_tracks,
        estimated_s3_ops=estimated_s3_ops,
        config_summary=config_summary,
        warnings=warnings,
    )


# =============================================================================
# E36: Job Cancellation Cleanup
# =============================================================================


class JobStatus(Enum):
    """Job status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobArtifactRecord:
    """Record of artifacts created by a job."""

    ep_id: str
    job_id: str
    artifact_type: str  # "frames", "crops", "thumbs", "manifests"
    paths: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ep_id": self.ep_id,
            "job_id": self.job_id,
            "artifact_type": self.artifact_type,
            "paths": self.paths,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class CancellationResult:
    """Result of job cancellation."""

    success: bool
    job_id: str
    status: JobStatus
    artifacts_removed: int = 0
    artifacts_marked_invalid: int = 0
    locks_released: bool = False
    progress_cleared: bool = False
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "job_id": self.job_id,
            "status": self.status.value,
            "artifacts_removed": self.artifacts_removed,
            "artifacts_marked_invalid": self.artifacts_marked_invalid,
            "locks_released": self.locks_released,
            "progress_cleared": self.progress_cleared,
            "errors": self.errors,
        }


class JobArtifactTracker:
    """Track artifacts created by jobs for cleanup on cancellation.

    This implements requirement E36: No job cancellation cleanup.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        if base_dir is None:
            data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
            base_dir = data_root / "job_artifacts"
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _artifact_file(self, ep_id: str, job_id: str) -> Path:
        return self._base_dir / ep_id / f"artifacts_{job_id}.json"

    def record_artifact(
        self,
        ep_id: str,
        job_id: str,
        artifact_type: str,
        path: str,
    ) -> None:
        """Record that a job created an artifact."""
        with self._lock:
            artifact_file = self._artifact_file(ep_id, job_id)
            artifact_file.parent.mkdir(parents=True, exist_ok=True)

            records: Dict[str, List[str]] = {}
            if artifact_file.exists():
                try:
                    records = json.loads(artifact_file.read_text())
                except json.JSONDecodeError:
                    pass

            if artifact_type not in records:
                records[artifact_type] = []
            if path not in records[artifact_type]:
                records[artifact_type].append(path)

            artifact_file.write_text(json.dumps(records, indent=2))

    def record_artifacts_batch(
        self,
        ep_id: str,
        job_id: str,
        artifact_type: str,
        paths: List[str],
    ) -> None:
        """Record multiple artifacts at once."""
        with self._lock:
            artifact_file = self._artifact_file(ep_id, job_id)
            artifact_file.parent.mkdir(parents=True, exist_ok=True)

            records: Dict[str, List[str]] = {}
            if artifact_file.exists():
                try:
                    records = json.loads(artifact_file.read_text())
                except json.JSONDecodeError:
                    pass

            if artifact_type not in records:
                records[artifact_type] = []
            for path in paths:
                if path not in records[artifact_type]:
                    records[artifact_type].append(path)

            artifact_file.write_text(json.dumps(records, indent=2))

    def get_job_artifacts(self, ep_id: str, job_id: str) -> Dict[str, List[str]]:
        """Get all artifacts for a job."""
        artifact_file = self._artifact_file(ep_id, job_id)
        if not artifact_file.exists():
            return {}
        try:
            return json.loads(artifact_file.read_text())
        except json.JSONDecodeError:
            return {}

    def cleanup_job_artifacts(
        self,
        ep_id: str,
        job_id: str,
        remove_files: bool = True,
        mark_invalid: bool = True,
    ) -> Tuple[int, int]:
        """Clean up artifacts from a cancelled job.

        Args:
            ep_id: Episode ID
            job_id: Job ID
            remove_files: If True, delete artifact files
            mark_invalid: If True, mark artifacts as invalid in manifests

        Returns:
            Tuple of (files_removed, files_marked_invalid)
        """
        artifacts = self.get_job_artifacts(ep_id, job_id)
        removed = 0
        marked = 0

        for artifact_type, paths in artifacts.items():
            for path_str in paths:
                path = Path(path_str)
                if remove_files and path.exists():
                    try:
                        if path.is_dir():
                            import shutil
                            shutil.rmtree(path)
                        else:
                            path.unlink()
                        removed += 1
                    except Exception as exc:
                        LOGGER.warning(
                            "[job-cleanup] Failed to remove %s: %s",
                            path,
                            exc,
                        )
                elif mark_invalid:
                    # Mark as invalid by creating .invalid marker
                    invalid_marker = Path(str(path) + ".invalid")
                    try:
                        invalid_marker.write_text(
                            json.dumps({
                                "reason": "job_cancelled",
                                "job_id": job_id,
                                "marked_at": datetime.utcnow().isoformat(),
                            })
                        )
                        marked += 1
                    except Exception as exc:
                        LOGGER.warning(
                            "[job-cleanup] Failed to mark %s invalid: %s",
                            path,
                            exc,
                        )

        # Remove the artifact record file
        artifact_file = self._artifact_file(ep_id, job_id)
        if artifact_file.exists():
            try:
                artifact_file.unlink()
            except Exception as exc:
                LOGGER.debug("[job-cleanup] Failed to remove artifact file %s: %s", artifact_file, exc)

        LOGGER.info(
            "[job-cleanup] Cleaned up job %s: removed=%d, marked_invalid=%d",
            job_id,
            removed,
            marked,
        )

        return removed, marked

    def clear_job_record(self, ep_id: str, job_id: str) -> None:
        """Clear artifact record for a completed job."""
        artifact_file = self._artifact_file(ep_id, job_id)
        if artifact_file.exists():
            try:
                artifact_file.unlink()
            except Exception as exc:
                LOGGER.debug("[job-cleanup] Failed to clear job record %s: %s", artifact_file, exc)


def cancel_job(
    ep_id: str,
    job_id: str,
    operation: str,
    cleanup_artifacts: bool = True,
    mark_invalid: bool = True,
) -> CancellationResult:
    """Cancel a running job and clean up.

    This implements requirement E36: No job cancellation cleanup.

    Args:
        ep_id: Episode ID
        job_id: Job ID
        operation: Operation name (e.g., "detect_track")
        cleanup_artifacts: If True, remove partial artifacts
        mark_invalid: If True, mark remaining artifacts as invalid

    Returns:
        CancellationResult with cleanup details
    """
    result = CancellationResult(
        success=False,
        job_id=job_id,
        status=JobStatus.CANCELLED,
    )

    errors = []

    try:
        # Release locks
        from apps.api.services.locks import get_lock_manager

        lock_manager = get_lock_manager()
        if lock_manager.force_release_lock(ep_id, operation):
            result.locks_released = True
        else:
            errors.append("Failed to release lock")

    except Exception as exc:
        errors.append(f"Lock release error: {exc}")

    try:
        # Clean up progress file
        data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
        progress_path = data_root / "manifests" / ep_id / "progress.json"
        if progress_path.exists():
            # Update progress to show cancelled
            try:
                progress = json.loads(progress_path.read_text())
                progress["status"] = "cancelled"
                progress["cancelled_at"] = datetime.utcnow().isoformat()
                progress_path.write_text(json.dumps(progress))
                result.progress_cleared = True
            except Exception as exc:
                errors.append(f"Progress update error: {exc}")

    except Exception as exc:
        errors.append(f"Progress cleanup error: {exc}")

    try:
        # Clean up artifacts
        if cleanup_artifacts or mark_invalid:
            tracker = JobArtifactTracker()
            removed, marked = tracker.cleanup_job_artifacts(
                ep_id,
                job_id,
                remove_files=cleanup_artifacts,
                mark_invalid=mark_invalid,
            )
            result.artifacts_removed = removed
            result.artifacts_marked_invalid = marked

    except Exception as exc:
        errors.append(f"Artifact cleanup error: {exc}")

    try:
        # Delete checkpoint if exists
        from apps.api.services.locks import get_checkpoint_manager

        checkpoint_manager = get_checkpoint_manager()
        checkpoint_manager.delete_checkpoint(ep_id, operation)

    except Exception as exc:
        errors.append(f"Checkpoint cleanup error: {exc}")

    result.errors = errors
    result.success = len(errors) == 0 or (result.locks_released and result.progress_cleared)

    return result


# =============================================================================
# E37: Job Queue Priority
# =============================================================================


class JobPriority(Enum):
    """Job priority levels."""

    LOW = 1
    NORMAL = 5
    HIGH = 10
    URGENT = 20


@dataclass
class QueuedJob:
    """A job in the queue."""

    job_id: str
    ep_id: str
    operation: str
    priority: JobPriority
    config: Dict[str, Any]
    submitted_at: datetime
    started_at: Optional[datetime] = None
    status: JobStatus = JobStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "ep_id": self.ep_id,
            "operation": self.operation,
            "priority": self.priority.name,
            "priority_value": self.priority.value,
            "config": self.config,
            "submitted_at": self.submitted_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "status": self.status.value,
        }


class PriorityJobQueue:
    """Priority-based job queue.

    This implements requirement E37: No job queue priority.

    Jobs are dequeued in priority order (highest first),
    with FIFO ordering within the same priority.
    """

    def __init__(self) -> None:
        self._queue: List[QueuedJob] = []
        self._lock = threading.Lock()
        self._job_index: Dict[str, QueuedJob] = {}

    def enqueue(
        self,
        job_id: str,
        ep_id: str,
        operation: str,
        config: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
    ) -> QueuedJob:
        """Add a job to the queue."""
        job = QueuedJob(
            job_id=job_id,
            ep_id=ep_id,
            operation=operation,
            priority=priority,
            config=config,
            submitted_at=datetime.utcnow(),
        )

        with self._lock:
            self._queue.append(job)
            self._job_index[job_id] = job
            # Sort by priority (descending) then by submission time (ascending)
            self._queue.sort(
                key=lambda j: (-j.priority.value, j.submitted_at)
            )

        LOGGER.info(
            "[job-queue] Enqueued job %s (priority=%s, position=%d)",
            job_id,
            priority.name,
            self._queue.index(job) + 1,
        )

        return job

    def dequeue(self) -> Optional[QueuedJob]:
        """Get the next job to run (highest priority, oldest first)."""
        with self._lock:
            for job in self._queue:
                if job.status == JobStatus.PENDING:
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.utcnow()
                    return job
        return None

    def peek(self) -> Optional[QueuedJob]:
        """Peek at the next job without removing it."""
        with self._lock:
            for job in self._queue:
                if job.status == JobStatus.PENDING:
                    return job
        return None

    def get_job(self, job_id: str) -> Optional[QueuedJob]:
        """Get a specific job by ID."""
        return self._job_index.get(job_id)

    def update_priority(self, job_id: str, priority: JobPriority) -> bool:
        """Update the priority of a pending job."""
        with self._lock:
            job = self._job_index.get(job_id)
            if job is None or job.status != JobStatus.PENDING:
                return False

            job.priority = priority
            # Re-sort queue
            self._queue.sort(
                key=lambda j: (-j.priority.value, j.submitted_at)
            )

        LOGGER.info(
            "[job-queue] Updated priority for %s to %s",
            job_id,
            priority.name,
        )
        return True

    def complete_job(self, job_id: str, status: JobStatus = JobStatus.COMPLETED) -> bool:
        """Mark a job as completed and remove from queue."""
        with self._lock:
            job = self._job_index.get(job_id)
            if job is None:
                return False

            job.status = status
            self._queue = [j for j in self._queue if j.job_id != job_id]
            del self._job_index[job_id]

        return True

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        ep_id: Optional[str] = None,
    ) -> List[QueuedJob]:
        """List jobs, optionally filtered by status or episode."""
        with self._lock:
            jobs = list(self._queue)

        if status:
            jobs = [j for j in jobs if j.status == status]
        if ep_id:
            jobs = [j for j in jobs if j.ep_id == ep_id]

        return jobs

    def queue_length(self) -> int:
        """Get number of pending jobs."""
        with self._lock:
            return sum(1 for j in self._queue if j.status == JobStatus.PENDING)

    def position(self, job_id: str) -> Optional[int]:
        """Get queue position of a job (1-indexed)."""
        with self._lock:
            pending = [j for j in self._queue if j.status == JobStatus.PENDING]
            for idx, job in enumerate(pending):
                if job.job_id == job_id:
                    return idx + 1
        return None


# =============================================================================
# E38: Resource Limits
# =============================================================================


@dataclass
class ResourceLimits:
    """Resource limits for job execution."""

    max_concurrent_jobs: int = 1
    max_cpu_threads: int = 4
    max_memory_mb: int = 8192
    max_gpu_memory_mb: Optional[int] = None
    gpu_devices: List[int] = field(default_factory=list)  # Empty = all available

    @classmethod
    def from_env(cls) -> "ResourceLimits":
        """Load limits from environment variables."""
        cpu_count = multiprocessing.cpu_count()

        return cls(
            max_concurrent_jobs=int(os.environ.get("SCREENALYTICS_MAX_CONCURRENT_JOBS", "1")),
            max_cpu_threads=int(os.environ.get("SCREENALYTICS_MAX_CPU_THREADS", str(min(cpu_count, 4)))),
            max_memory_mb=int(os.environ.get("SCREENALYTICS_MAX_MEMORY_MB", "8192")),
            max_gpu_memory_mb=int(os.environ.get("SCREENALYTICS_MAX_GPU_MB", "0")) or None,
            gpu_devices=[
                int(x) for x in os.environ.get("SCREANALYTICS_GPU_DEVICES", "").split(",")
                if x.strip().isdigit()
            ],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "max_cpu_threads": self.max_cpu_threads,
            "max_memory_mb": self.max_memory_mb,
            "max_gpu_memory_mb": self.max_gpu_memory_mb,
            "gpu_devices": self.gpu_devices,
        }


@dataclass
class ResourceUsage:
    """Current resource usage."""

    active_jobs: int
    active_cpu_threads: int
    memory_used_mb: int
    gpu_memory_used_mb: Optional[int]
    jobs: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_jobs": self.active_jobs,
            "active_cpu_threads": self.active_cpu_threads,
            "memory_used_mb": self.memory_used_mb,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "jobs": self.jobs,
        }


class ResourceManager:
    """Manage resource allocation for jobs.

    This implements requirement E38: No resource limits.
    """

    def __init__(self, limits: Optional[ResourceLimits] = None) -> None:
        self._limits = limits or ResourceLimits.from_env()
        self._lock = threading.Lock()
        self._active_jobs: Dict[str, Dict[str, Any]] = {}  # job_id -> allocation

    @property
    def limits(self) -> ResourceLimits:
        return self._limits

    def update_limits(self, limits: ResourceLimits) -> None:
        """Update resource limits."""
        with self._lock:
            self._limits = limits

    def can_start_job(
        self,
        cpu_threads: int = 1,
        memory_mb: int = 0,
        gpu_memory_mb: int = 0,
    ) -> Tuple[bool, Optional[str]]:
        """Check if a job can start given resource requirements.

        Returns:
            Tuple of (can_start, reason_if_not)
        """
        with self._lock:
            # Check concurrent job limit
            if len(self._active_jobs) >= self._limits.max_concurrent_jobs:
                return False, f"Max concurrent jobs reached ({self._limits.max_concurrent_jobs})"

            # Check CPU threads
            total_threads = sum(j.get("cpu_threads", 0) for j in self._active_jobs.values())
            if total_threads + cpu_threads > self._limits.max_cpu_threads:
                return False, f"CPU thread limit reached ({total_threads}/{self._limits.max_cpu_threads})"

            # Check memory
            total_memory = sum(j.get("memory_mb", 0) for j in self._active_jobs.values())
            if self._limits.max_memory_mb and total_memory + memory_mb > self._limits.max_memory_mb:
                return False, f"Memory limit reached ({total_memory}/{self._limits.max_memory_mb} MB)"

            # Check GPU memory
            if gpu_memory_mb > 0 and self._limits.max_gpu_memory_mb:
                total_gpu = sum(j.get("gpu_memory_mb", 0) for j in self._active_jobs.values())
                if total_gpu + gpu_memory_mb > self._limits.max_gpu_memory_mb:
                    return False, f"GPU memory limit reached ({total_gpu}/{self._limits.max_gpu_memory_mb} MB)"

        return True, None

    def allocate(
        self,
        job_id: str,
        cpu_threads: int = 1,
        memory_mb: int = 0,
        gpu_memory_mb: int = 0,
    ) -> Tuple[bool, Optional[str]]:
        """Allocate resources for a job.

        Returns:
            Tuple of (success, error_message)
        """
        can_start, reason = self.can_start_job(cpu_threads, memory_mb, gpu_memory_mb)
        if not can_start:
            return False, reason

        with self._lock:
            self._active_jobs[job_id] = {
                "cpu_threads": cpu_threads,
                "memory_mb": memory_mb,
                "gpu_memory_mb": gpu_memory_mb,
                "started_at": datetime.utcnow().isoformat(),
            }

        LOGGER.info(
            "[resources] Allocated for job %s: cpu=%d, mem=%dMB, gpu=%dMB",
            job_id,
            cpu_threads,
            memory_mb,
            gpu_memory_mb,
        )
        return True, None

    def release(self, job_id: str) -> bool:
        """Release resources for a completed job."""
        with self._lock:
            if job_id in self._active_jobs:
                del self._active_jobs[job_id]
                LOGGER.info("[resources] Released resources for job %s", job_id)
                return True
        return False

    def get_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        with self._lock:
            jobs_list = [
                {"job_id": jid, **alloc}
                for jid, alloc in self._active_jobs.items()
            ]

            return ResourceUsage(
                active_jobs=len(self._active_jobs),
                active_cpu_threads=sum(j.get("cpu_threads", 0) for j in self._active_jobs.values()),
                memory_used_mb=sum(j.get("memory_mb", 0) for j in self._active_jobs.values()),
                gpu_memory_used_mb=sum(j.get("gpu_memory_mb", 0) for j in self._active_jobs.values()) or None,
                jobs=jobs_list,
            )

    def get_available(self) -> Dict[str, Any]:
        """Get available resources."""
        usage = self.get_usage()
        return {
            "jobs_available": max(0, self._limits.max_concurrent_jobs - usage.active_jobs),
            "cpu_threads_available": max(0, self._limits.max_cpu_threads - usage.active_cpu_threads),
            "memory_mb_available": max(0, self._limits.max_memory_mb - usage.memory_used_mb),
            "gpu_memory_mb_available": max(
                0,
                (self._limits.max_gpu_memory_mb or 0) - (usage.gpu_memory_used_mb or 0)
            ) if self._limits.max_gpu_memory_mb else None,
        }


# =============================================================================
# Module-level instances and exports
# =============================================================================

# Global instances
_job_queue: Optional[PriorityJobQueue] = None
_resource_manager: Optional[ResourceManager] = None
_artifact_tracker: Optional[JobArtifactTracker] = None


def get_job_queue() -> PriorityJobQueue:
    """Get global job queue."""
    global _job_queue
    if _job_queue is None:
        _job_queue = PriorityJobQueue()
    return _job_queue


def get_resource_manager() -> ResourceManager:
    """Get global resource manager."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


def get_artifact_tracker() -> JobArtifactTracker:
    """Get global artifact tracker."""
    global _artifact_tracker
    if _artifact_tracker is None:
        _artifact_tracker = JobArtifactTracker()
    return _artifact_tracker


__all__ = [
    # Dry run
    "DryRunEstimate",
    "dry_run_detect_track",
    # Cancellation
    "JobStatus",
    "JobArtifactRecord",
    "CancellationResult",
    "JobArtifactTracker",
    "cancel_job",
    "get_artifact_tracker",
    # Priority queue
    "JobPriority",
    "QueuedJob",
    "PriorityJobQueue",
    "get_job_queue",
    # Resource limits
    "ResourceLimits",
    "ResourceUsage",
    "ResourceManager",
    "get_resource_manager",
]
