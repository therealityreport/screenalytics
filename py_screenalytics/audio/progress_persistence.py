"""
Progress Persistence for Audio Pipeline.

Saves job progress to file (and optionally Redis) so the UI can show progress
even after page refresh. Supports both file-based persistence and Redis for
real-time updates in distributed environments.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class StepProgress:
    """Progress data for a single pipeline step."""
    name: str
    status: str = "pending"  # pending, running, completed, failed, skipped
    progress: float = 0.0    # 0.0 to 1.0
    message: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


@dataclass
class JobProgress:
    """Overall job progress state."""
    job_id: str
    ep_id: str
    job_type: str = "audio_pipeline"
    status: str = "pending"  # pending, running, completed, failed
    overall_progress: float = 0.0
    current_step: Optional[str] = None
    steps: Dict[str, StepProgress] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "ep_id": self.ep_id,
            "job_type": self.job_type,
            "status": self.status,
            "overall_progress": self.overall_progress,
            "current_step": self.current_step,
            "steps": {name: asdict(step) for name, step in self.steps.items()},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobProgress":
        """Create from dictionary."""
        steps = {
            name: StepProgress(**step_data)
            for name, step_data in data.get("steps", {}).items()
        }
        return cls(
            job_id=data["job_id"],
            ep_id=data["ep_id"],
            job_type=data.get("job_type", "audio_pipeline"),
            status=data.get("status", "pending"),
            overall_progress=data.get("overall_progress", 0.0),
            current_step=data.get("current_step"),
            steps=steps,
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            error=data.get("error"),
        )


class ProgressPersistence:
    """
    Manages progress persistence to file and optionally Redis.

    Usage:
        progress = ProgressPersistence(
            job_id="job_123",
            ep_id="show_s01_e01",
            progress_dir=Path("data/progress"),
        )

        progress.start_job()
        progress.start_step("extract")
        progress.update_step("extract", 0.5, "Extracting audio...")
        progress.complete_step("extract")
        progress.complete_job()
    """

    # Default step weights for audio pipeline
    DEFAULT_STEP_WEIGHTS = {
        "extract": 5,
        "separate": 20,
        "enhance": 15,
        "diarize": 20,
        "voices": 10,
        "transcribe": 15,
        "fuse": 5,
        "export": 5,
        "qc": 5,
    }

    def __init__(
        self,
        job_id: str,
        ep_id: str,
        progress_dir: Optional[Path] = None,
        redis_client: Optional[Any] = None,
        redis_ttl_seconds: int = 3600,
        step_weights: Optional[Dict[str, int]] = None,
    ):
        self.job_id = job_id
        self.ep_id = ep_id
        self.progress_dir = progress_dir or Path("data/progress")
        self.progress_file = self.progress_dir / f"{job_id}.json"
        self.redis_client = redis_client
        self.redis_ttl = redis_ttl_seconds
        self.redis_key = f"job_progress:{job_id}"
        self.step_weights = step_weights or self.DEFAULT_STEP_WEIGHTS
        self._progress: Optional[JobProgress] = None

    def _ensure_dir(self) -> None:
        """Ensure progress directory exists."""
        self.progress_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> JobProgress:
        """Load progress from file or create new."""
        if self._progress is not None:
            return self._progress

        if self.progress_file.exists():
            try:
                with self.progress_file.open("r") as f:
                    data = json.load(f)
                self._progress = JobProgress.from_dict(data)
                return self._progress
            except Exception as e:
                LOGGER.warning(f"Failed to load progress: {e}")

        # Create new progress
        self._progress = JobProgress(
            job_id=self.job_id,
            ep_id=self.ep_id,
        )

        # Initialize steps
        for step_name in self.step_weights:
            self._progress.steps[step_name] = StepProgress(name=step_name)

        return self._progress

    def _save(self) -> None:
        """Save progress to file and Redis."""
        if self._progress is None:
            return

        self._progress.updated_at = datetime.utcnow().isoformat()

        # Save to file
        self._ensure_dir()
        with self.progress_file.open("w") as f:
            json.dump(self._progress.to_dict(), f, indent=2)

        # Save to Redis if available
        if self.redis_client:
            try:
                self.redis_client.setex(
                    self.redis_key,
                    self.redis_ttl,
                    json.dumps(self._progress.to_dict()),
                )
            except Exception as e:
                LOGGER.debug(f"Failed to save progress to Redis: {e}")

    def _calculate_overall_progress(self) -> float:
        """Calculate overall progress based on step weights."""
        progress = self._load()
        total_weight = sum(self.step_weights.values())
        completed_weight = 0.0

        for step_name, weight in self.step_weights.items():
            step = progress.steps.get(step_name)
            if step:
                if step.status == "completed":
                    completed_weight += weight
                elif step.status == "running":
                    completed_weight += weight * step.progress
                elif step.status == "skipped":
                    completed_weight += weight  # Count skipped as complete

        return completed_weight / total_weight if total_weight > 0 else 0.0

    def start_job(self) -> None:
        """Mark job as started."""
        progress = self._load()
        progress.status = "running"
        progress.created_at = datetime.utcnow().isoformat()
        self._save()
        LOGGER.info(f"Job {self.job_id} started for episode {self.ep_id}")

    def start_step(self, step_name: str, message: str = "") -> None:
        """Mark a step as started."""
        progress = self._load()

        if step_name not in progress.steps:
            progress.steps[step_name] = StepProgress(name=step_name)

        step = progress.steps[step_name]
        step.status = "running"
        step.progress = 0.0
        step.message = message
        step.started_at = datetime.utcnow().isoformat()
        step.completed_at = None
        step.error = None

        progress.current_step = step_name
        progress.overall_progress = self._calculate_overall_progress()

        self._save()
        LOGGER.debug(f"Step '{step_name}' started")

    def update_step(
        self,
        step_name: str,
        progress_value: float,
        message: str = "",
    ) -> None:
        """Update progress for a step."""
        progress = self._load()

        if step_name not in progress.steps:
            progress.steps[step_name] = StepProgress(name=step_name, status="running")

        step = progress.steps[step_name]
        step.progress = min(1.0, max(0.0, progress_value))
        if message:
            step.message = message

        progress.overall_progress = self._calculate_overall_progress()

        self._save()
        LOGGER.debug(f"Step '{step_name}' progress: {progress_value*100:.0f}%")

    def complete_step(
        self,
        step_name: str,
        message: str = "Completed",
    ) -> None:
        """Mark a step as completed."""
        progress = self._load()

        if step_name not in progress.steps:
            progress.steps[step_name] = StepProgress(name=step_name)

        step = progress.steps[step_name]
        step.status = "completed"
        step.progress = 1.0
        step.message = message
        step.completed_at = datetime.utcnow().isoformat()

        progress.overall_progress = self._calculate_overall_progress()

        self._save()
        LOGGER.debug(f"Step '{step_name}' completed")

    def skip_step(self, step_name: str, message: str = "Skipped") -> None:
        """Mark a step as skipped."""
        progress = self._load()

        if step_name not in progress.steps:
            progress.steps[step_name] = StepProgress(name=step_name)

        step = progress.steps[step_name]
        step.status = "skipped"
        step.progress = 1.0
        step.message = message
        step.completed_at = datetime.utcnow().isoformat()

        progress.overall_progress = self._calculate_overall_progress()

        self._save()
        LOGGER.debug(f"Step '{step_name}' skipped")

    def fail_step(self, step_name: str, error: str) -> None:
        """Mark a step as failed."""
        progress = self._load()

        if step_name not in progress.steps:
            progress.steps[step_name] = StepProgress(name=step_name)

        step = progress.steps[step_name]
        step.status = "failed"
        step.error = error
        step.completed_at = datetime.utcnow().isoformat()

        # Don't update overall progress for failed steps

        self._save()
        LOGGER.warning(f"Step '{step_name}' failed: {error}")

    def complete_job(self, message: str = "Pipeline completed successfully") -> None:
        """Mark job as completed."""
        progress = self._load()
        progress.status = "completed"
        progress.overall_progress = 1.0
        progress.current_step = None

        self._save()
        LOGGER.info(f"Job {self.job_id} completed")

    def fail_job(self, error: str) -> None:
        """Mark job as failed."""
        progress = self._load()
        progress.status = "failed"
        progress.error = error

        self._save()
        LOGGER.error(f"Job {self.job_id} failed: {error}")

    def get_progress(self) -> JobProgress:
        """Get current job progress."""
        return self._load()

    def get_progress_dict(self) -> Dict[str, Any]:
        """Get current job progress as dictionary."""
        return self._load().to_dict()

    def delete(self) -> None:
        """Delete progress file and Redis key."""
        if self.progress_file.exists():
            self.progress_file.unlink()

        if self.redis_client:
            try:
                self.redis_client.delete(self.redis_key)
            except Exception as e:
                LOGGER.debug(f"Failed to delete Redis key: {e}")

        self._progress = None


def get_redis_client() -> Optional[Any]:
    """
    Get Redis client if available.

    Returns:
        Redis client or None if not available
    """
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        return None

    try:
        import redis
        return redis.from_url(redis_url)
    except ImportError:
        LOGGER.debug("redis package not installed, using file-only persistence")
        return None
    except Exception as e:
        LOGGER.debug(f"Failed to connect to Redis: {e}")
        return None


def load_job_progress(
    job_id: str,
    progress_dir: Optional[Path] = None,
) -> Optional[JobProgress]:
    """
    Load job progress from file.

    Args:
        job_id: Job identifier
        progress_dir: Directory containing progress files

    Returns:
        JobProgress or None if not found
    """
    progress_dir = progress_dir or Path("data/progress")
    progress_file = progress_dir / f"{job_id}.json"

    if not progress_file.exists():
        return None

    try:
        with progress_file.open("r") as f:
            data = json.load(f)
        return JobProgress.from_dict(data)
    except Exception as e:
        LOGGER.warning(f"Failed to load progress for {job_id}: {e}")
        return None


def list_recent_jobs(
    progress_dir: Optional[Path] = None,
    limit: int = 20,
    status_filter: Optional[str] = None,
) -> List[JobProgress]:
    """
    List recent job progress files.

    Args:
        progress_dir: Directory containing progress files
        limit: Maximum number of jobs to return
        status_filter: Optional status to filter by

    Returns:
        List of JobProgress objects, most recent first
    """
    progress_dir = progress_dir or Path("data/progress")

    if not progress_dir.exists():
        return []

    jobs = []
    for progress_file in sorted(
        progress_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        if len(jobs) >= limit:
            break

        try:
            with progress_file.open("r") as f:
                data = json.load(f)
            job = JobProgress.from_dict(data)

            if status_filter is None or job.status == status_filter:
                jobs.append(job)
        except Exception as e:
            LOGGER.debug(f"Failed to load {progress_file}: {e}")

    return jobs
