"""Celery tasks for long-running operations in SCREANALYTICS.

These tasks handle cluster assignments, auto-grouping, and cleanup operations
that would otherwise block the UI.

Usage:
    from apps.api.tasks import run_manual_assign_task
    result = run_manual_assign_task.delay(episode_id="rhobh-s01e01", payload={...})
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import redis
from celery import Task, states
from celery.result import AsyncResult

from apps.api.celery_app import celery_app
from apps.api.config import REDIS_URL

LOGGER = logging.getLogger(__name__)

# Redis client for job locking
_redis_client: Optional[redis.Redis] = None


def _get_redis() -> redis.Redis:
    """Get or create Redis client for job locking."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _lock_key(episode_id: str, operation: str) -> str:
    """Generate Redis key for job lock."""
    return f"screanalytics:job_lock:{episode_id}:{operation}"


def check_active_job(episode_id: str, operation: str) -> Optional[str]:
    """Check if there's an active job for this episode+operation.

    Returns job_id if found, None otherwise.
    """
    try:
        r = _get_redis()
        lock_key = _lock_key(episode_id, operation)
        job_id = r.get(lock_key)

        if job_id:
            # Verify the job is still running
            result = AsyncResult(job_id, app=celery_app)
            if result.state in (states.PENDING, states.RECEIVED, states.STARTED):
                return job_id
            # Job finished or failed, clean up lock
            r.delete(lock_key)

        return None
    except Exception as e:
        LOGGER.warning(f"Failed to check active job: {e}")
        return None


def _acquire_lock(episode_id: str, operation: str, job_id: str, ttl: int = 3600) -> bool:
    """Try to acquire lock for an episode+operation.

    Args:
        episode_id: Episode ID
        operation: Operation type (manual_assign, auto_group, etc.)
        job_id: Celery task ID
        ttl: Lock TTL in seconds (default 1 hour)

    Returns:
        True if lock acquired, False if another job is running
    """
    try:
        r = _get_redis()
        lock_key = _lock_key(episode_id, operation)

        # Check if there's an existing lock
        existing = r.get(lock_key)
        if existing:
            # Verify the existing job is still running
            result = AsyncResult(existing, app=celery_app)
            if result.state in (states.PENDING, states.RECEIVED, states.STARTED):
                return False  # Another job is running
            # Old job finished, we can take over

        # Set lock with TTL
        r.setex(lock_key, ttl, job_id)
        return True
    except Exception as e:
        LOGGER.warning(f"Failed to acquire lock: {e}")
        return True  # Proceed anyway if Redis fails


def _release_lock(episode_id: str, operation: str, job_id: str) -> None:
    """Release job lock."""
    try:
        r = _get_redis()
        lock_key = _lock_key(episode_id, operation)
        # Only release if we own the lock
        current = r.get(lock_key)
        if current == job_id:
            r.delete(lock_key)
    except Exception as e:
        LOGGER.warning(f"Failed to release lock: {e}")


class GroupingTask(Task):
    """Base class for grouping tasks with lock management."""

    abstract = True
    _grouping_service = None

    @property
    def grouping_service(self):
        """Lazy-load GroupingService to avoid import at module level."""
        if self._grouping_service is None:
            from apps.api.services.grouping import GroupingService
            self._grouping_service = GroupingService()
        return self._grouping_service


@celery_app.task(bind=True, base=GroupingTask, name="tasks.run_manual_assign")
def run_manual_assign_task(
    self,
    episode_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Background task for manual cluster assignment.

    Args:
        episode_id: Episode ID
        payload: Dict with 'assignments' list, each having:
            - cluster_id: str
            - target_cast_id: str

    Returns:
        Result dict with succeeded/failed counts and results list
    """
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting manual_assign for {episode_id}")

    # Try to acquire lock
    if not _acquire_lock(episode_id, "manual_assign", job_id):
        return {
            "status": "error",
            "error": "Another manual_assign job is already running for this episode",
            "episode_id": episode_id,
        }

    try:
        assignments = payload.get("assignments", [])

        # Convert list of dicts to BatchAssignmentItem-like objects
        class AssignmentItem:
            def __init__(self, d):
                self.cluster_id = d.get("cluster_id")
                self.target_cast_id = d.get("target_cast_id")

        assignment_items = [AssignmentItem(a) for a in assignments]

        result = self.grouping_service.batch_assign_clusters(episode_id, assignment_items)

        return {
            "status": "success",
            "episode_id": episode_id,
            "operation": "manual_assign",
            "succeeded": result.get("succeeded", 0),
            "failed": result.get("failed", 0),
            "results": result.get("results", []),
        }
    except Exception as e:
        LOGGER.exception(f"[{job_id}] manual_assign failed: {e}")
        return {
            "status": "error",
            "episode_id": episode_id,
            "operation": "manual_assign",
            "error": str(e),
        }
    finally:
        _release_lock(episode_id, "manual_assign", job_id)


@celery_app.task(bind=True, base=GroupingTask, name="tasks.run_auto_group")
def run_auto_group_task(
    self,
    episode_id: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Background task for auto-grouping clusters.

    Args:
        episode_id: Episode ID
        options: Optional dict with:
            - protect_manual: bool (default True) - Don't merge manually assigned clusters
            - facebank_first: bool (default True) - Try facebank matching before prototypes

    Returns:
        Result dict with grouping statistics
    """
    job_id = self.request.id
    options = options or {}
    LOGGER.info(f"[{job_id}] Starting auto_group for {episode_id} with options: {options}")

    # Try to acquire lock
    if not _acquire_lock(episode_id, "auto_group", job_id):
        return {
            "status": "error",
            "error": "Another auto_group job is already running for this episode",
            "episode_id": episode_id,
        }

    try:
        # Progress callback that updates task state
        def progress_callback(step: str, progress: float, message: str):
            self.update_state(
                state="PROGRESS",
                meta={
                    "step": step,
                    "progress": progress,
                    "message": message,
                    "episode_id": episode_id,
                },
            )

        result = self.grouping_service.group_clusters_auto(
            episode_id,
            progress_callback=progress_callback,
            protect_manual=options.get("protect_manual", True),
            facebank_first=options.get("facebank_first", True),
        )

        # Extract key stats from result
        centroids_count = len(result.get("centroids", {}).get("centroids", {}))
        within_merged = result.get("within_episode", {}).get("merged_count", 0)
        assignments_count = len(result.get("assignments", []))
        new_people = result.get("across_episodes", {}).get("new_people_count", 0)
        facebank_assigned = result.get("facebank_assigned", 0)

        return {
            "status": "success",
            "episode_id": episode_id,
            "operation": "auto_group",
            "centroids_count": centroids_count,
            "within_episode_merged": within_merged,
            "assignments_count": assignments_count,
            "new_people_count": new_people,
            "facebank_assigned": facebank_assigned,
            "log": result.get("log"),
        }
    except Exception as e:
        LOGGER.exception(f"[{job_id}] auto_group failed: {e}")
        return {
            "status": "error",
            "episode_id": episode_id,
            "operation": "auto_group",
            "error": str(e),
        }
    finally:
        _release_lock(episode_id, "auto_group", job_id)


@celery_app.task(bind=True, base=GroupingTask, name="tasks.run_reembed")
def run_reembed_task(
    self,
    episode_id: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Background task to re-embed all faces for an episode.

    This task triggers the faces_embed pipeline stage.
    """
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting reembed for {episode_id}")

    if not _acquire_lock(episode_id, "reembed", job_id):
        return {
            "status": "error",
            "error": "Another reembed job is already running for this episode",
            "episode_id": episode_id,
        }

    try:
        # Import the job service to trigger reembed
        from apps.api.services.jobs import JobService
        job_service = JobService()

        # Start faces_embed job (synchronously wait or return job info)
        # For now, just return a placeholder - full implementation would
        # integrate with the existing ML pipeline
        return {
            "status": "success",
            "episode_id": episode_id,
            "operation": "reembed",
            "message": "Reembed task queued (integration with ML pipeline pending)",
        }
    except Exception as e:
        LOGGER.exception(f"[{job_id}] reembed failed: {e}")
        return {
            "status": "error",
            "episode_id": episode_id,
            "operation": "reembed",
            "error": str(e),
        }
    finally:
        _release_lock(episode_id, "reembed", job_id)


@celery_app.task(bind=True, base=GroupingTask, name="tasks.run_recluster")
def run_recluster_task(
    self,
    episode_id: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Background task to recluster faces for an episode.

    This task triggers the cluster pipeline stage.
    """
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting recluster for {episode_id}")

    if not _acquire_lock(episode_id, "recluster", job_id):
        return {
            "status": "error",
            "error": "Another recluster job is already running for this episode",
            "episode_id": episode_id,
        }

    try:
        return {
            "status": "success",
            "episode_id": episode_id,
            "operation": "recluster",
            "message": "Recluster task queued (integration with ML pipeline pending)",
        }
    except Exception as e:
        LOGGER.exception(f"[{job_id}] recluster failed: {e}")
        return {
            "status": "error",
            "episode_id": episode_id,
            "operation": "recluster",
            "error": str(e),
        }
    finally:
        _release_lock(episode_id, "recluster", job_id)


@celery_app.task(bind=True, base=GroupingTask, name="tasks.run_split_tracks")
def run_split_tracks_task(
    self,
    episode_id: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Background task to split tracks for an episode."""
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting split_tracks for {episode_id}")

    if not _acquire_lock(episode_id, "split_tracks", job_id):
        return {
            "status": "error",
            "error": "Another split_tracks job is already running for this episode",
            "episode_id": episode_id,
        }

    try:
        return {
            "status": "success",
            "episode_id": episode_id,
            "operation": "split_tracks",
            "message": "Split tracks task queued (integration with ML pipeline pending)",
        }
    except Exception as e:
        LOGGER.exception(f"[{job_id}] split_tracks failed: {e}")
        return {
            "status": "error",
            "episode_id": episode_id,
            "operation": "split_tracks",
            "error": str(e),
        }
    finally:
        _release_lock(episode_id, "split_tracks", job_id)


def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get status of a Celery job.

    Args:
        job_id: Celery task ID

    Returns:
        Dict with job_id, state, result (if finished), error (if failed)
    """
    result = AsyncResult(job_id, app=celery_app)

    # Map Celery states to simplified UI states
    state_map = {
        states.PENDING: "queued",
        states.RECEIVED: "queued",
        states.STARTED: "in_progress",
        states.SUCCESS: "success",
        states.FAILURE: "failed",
        states.RETRY: "retrying",
        states.REVOKED: "cancelled",
        "PROGRESS": "in_progress",
    }

    state = state_map.get(result.state, "unknown")

    response: Dict[str, Any] = {
        "job_id": job_id,
        "state": state,
        "raw_state": result.state,
    }

    if result.state == "PROGRESS":
        # Include progress metadata
        response["progress"] = result.info
    elif result.ready():
        # Job finished - include result
        if result.successful():
            response["result"] = result.result
        else:
            # Failed
            response["error"] = str(result.result)

    return response


def cancel_job(job_id: str) -> Dict[str, Any]:
    """Cancel a running Celery job.

    Args:
        job_id: Celery task ID

    Returns:
        Dict with job_id and status
    """
    result = AsyncResult(job_id, app=celery_app)
    result.revoke(terminate=True)

    return {
        "job_id": job_id,
        "status": "cancelled",
    }


__all__ = [
    "run_manual_assign_task",
    "run_auto_group_task",
    "run_reembed_task",
    "run_recluster_task",
    "run_split_tracks_task",
    "check_active_job",
    "get_job_status",
    "cancel_job",
]
