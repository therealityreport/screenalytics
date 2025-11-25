"""Celery job status endpoints for background operations.

This router handles status polling and cancellation of Celery-based background jobs
like cluster assignments and auto-grouping.

Separate from the ML pipeline jobs router (apps/api/routers/jobs.py) which handles
detect/track/embed jobs via subprocess.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from celery.result import AsyncResult

from apps.api.celery_app import celery_app
from apps.api.tasks import get_job_status, cancel_job

router = APIRouter(prefix="/celery_jobs", tags=["celery_jobs"])


def _map_celery_state(state: str) -> str:
    """Map Celery states to simplified UI states."""
    mapping = {
        "PENDING": "queued",
        "RECEIVED": "queued",
        "STARTED": "in_progress",
        "PROGRESS": "in_progress",
        "SUCCESS": "success",
        "FAILURE": "failed",
        "RETRY": "retrying",
        "REVOKED": "cancelled",
    }
    return mapping.get(state, "unknown")


@router.get("/{job_id}")
async def get_celery_job_status(job_id: str):
    """Get status of a Celery background job.

    Returns:
        - job_id: The job ID
        - state: Simplified state (queued, in_progress, success, failed, cancelled)
        - raw_state: Original Celery state
        - result: Job result if completed (success or failure)
        - progress: Progress metadata if job is running
    """
    result = AsyncResult(job_id, app=celery_app)

    # Check for truly unknown jobs
    # PENDING can mean either "waiting to run" or "unknown job ID"
    # We check if the job exists in the result backend
    if result.state == "PENDING":
        # Try to get the result to see if the job exists
        # If it doesn't exist, this will return None for result.result
        # and the state will remain PENDING
        # We'll treat this as "queued" since we can't distinguish
        pass

    response = {
        "job_id": job_id,
        "state": _map_celery_state(result.state),
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
            # Failed - include error
            response["error"] = str(result.result) if result.result else "Unknown error"

    return response


@router.post("/{job_id}/cancel")
async def cancel_celery_job(job_id: str):
    """Cancel a running Celery background job.

    Note: This sends a termination signal. Long-running operations may take
    a moment to actually stop.
    """
    result = AsyncResult(job_id, app=celery_app)

    # Check if the job is still running
    if result.ready():
        return {
            "job_id": job_id,
            "status": "already_finished",
            "state": _map_celery_state(result.state),
        }

    # Revoke the task
    result.revoke(terminate=True)

    return {
        "job_id": job_id,
        "status": "cancelled",
    }


@router.get("")
async def list_active_celery_jobs():
    """List currently active Celery jobs.

    Note: This only shows jobs known to the current worker.
    Completed jobs are available via their individual job_id.
    """
    # Get active tasks from Celery
    inspect = celery_app.control.inspect()

    # This might return None if no workers are connected
    active = inspect.active() or {}
    scheduled = inspect.scheduled() or {}
    reserved = inspect.reserved() or {}

    jobs = []

    # Collect all active jobs
    for worker, tasks in active.items():
        for task in tasks:
            jobs.append({
                "job_id": task.get("id"),
                "name": task.get("name"),
                "state": "in_progress",
                "worker": worker,
            })

    # Collect scheduled jobs
    for worker, tasks in scheduled.items():
        for task in tasks:
            request = task.get("request", {})
            jobs.append({
                "job_id": request.get("id"),
                "name": request.get("name"),
                "state": "scheduled",
                "worker": worker,
            })

    # Collect reserved (queued) jobs
    for worker, tasks in reserved.items():
        for task in tasks:
            jobs.append({
                "job_id": task.get("id"),
                "name": task.get("name"),
                "state": "queued",
                "worker": worker,
            })

    return {
        "jobs": jobs,
        "count": len(jobs),
    }


__all__ = ["router"]
