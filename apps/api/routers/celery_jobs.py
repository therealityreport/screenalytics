"""Celery job status endpoints for background operations.

This router handles status polling and cancellation of Celery-based background jobs
including ML pipeline operations (detect/track, faces harvest, cluster) and
grouping operations (manual assign, auto-group).

The Celery-based pipeline jobs provide true async execution via Redis queue,
compared to the subprocess-based jobs in apps/api/routers/jobs.py.
"""

from __future__ import annotations

from typing import Optional, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from celery.result import AsyncResult
from pydantic import BaseModel, Field

from apps.api.celery_app import celery_app
from apps.api.tasks import (
    get_job_status,
    cancel_job,
    check_active_job,
    run_detect_track_task,
    run_faces_embed_task,
    run_cluster_task,
)

router = APIRouter(prefix="/celery_jobs", tags=["celery_jobs"])


# =============================================================================
# Request Models
# =============================================================================

DEVICE_LITERAL = Literal["auto", "cpu", "mps", "coreml", "metal", "apple", "cuda"]


class DetectTrackCeleryRequest(BaseModel):
    """Request model for Celery-based detect/track job."""
    ep_id: str = Field(..., description="Episode identifier")
    stride: int = Field(6, description="Frame stride for detection sampling")
    fps: Optional[float] = Field(None, description="Optional target FPS for sampling")
    device: DEVICE_LITERAL = Field("auto", description="Execution device")
    detector: str = Field("retinaface", description="Face detector backend")
    tracker: str = Field("bytetrack", description="Tracker backend")
    save_frames: bool = Field(False, description="Save sampled frames")
    save_crops: bool = Field(False, description="Save face crops")
    jpeg_quality: int = Field(72, ge=1, le=100, description="JPEG quality")
    det_thresh: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Detection threshold")
    max_gap: Optional[int] = Field(30, ge=1, description="Max frame gap before new track")
    scene_detector: Optional[str] = Field(None, description="Scene detector backend")
    scene_threshold: Optional[float] = Field(None, description="Scene cut threshold")
    scene_min_len: Optional[int] = Field(None, ge=1, description="Minimum frames between scene cuts")
    scene_warmup_dets: Optional[int] = Field(None, ge=0, description="Forced detections after each cut")
    track_high_thresh: Optional[float] = Field(None, ge=0.0, le=1.0, description="ByteTrack track_high_thresh override")
    new_track_thresh: Optional[float] = Field(None, ge=0.0, le=1.0, description="ByteTrack new_track_thresh override")
    track_buffer: Optional[int] = Field(None, ge=1, description="ByteTrack base track_buffer before stride scaling")
    min_box_area: Optional[float] = Field(None, ge=0.0, description="ByteTrack min_box_area override")
    cpu_threads: Optional[int] = Field(None, ge=1, le=16, description="CPU thread cap for detect/track run")
    profile: Optional[str] = Field(None, description="Performance profile")


class FacesEmbedCeleryRequest(BaseModel):
    """Request model for Celery-based faces embed (harvest) job."""
    ep_id: str = Field(..., description="Episode identifier")
    device: DEVICE_LITERAL = Field("auto", description="Execution device")
    save_frames: bool = Field(False, description="Save sampled frames")
    save_crops: bool = Field(True, description="Save face crops")
    jpeg_quality: int = Field(72, ge=1, le=100, description="JPEG quality")
    min_frames_between_crops: int = Field(32, ge=1, description="Min frames between crops")
    thumb_size: int = Field(256, ge=64, le=512, description="Thumbnail size")
    profile: Optional[str] = Field(None, description="Performance profile")


class ClusterCeleryRequest(BaseModel):
    """Request model for Celery-based cluster job."""
    ep_id: str = Field(..., description="Episode identifier")
    device: DEVICE_LITERAL = Field("auto", description="Execution device")
    cluster_thresh: float = Field(0.7, ge=0.2, le=0.99, description="Clustering threshold")
    min_cluster_size: int = Field(2, ge=1, description="Minimum cluster size")
    min_identity_sim: Optional[float] = Field(0.5, ge=0.0, le=0.99, description="Min identity similarity")
    profile: Optional[str] = Field(None, description="Performance profile")


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

    # In eager/unit-test mode, a PENDING task id usually means "unknown"
    if celery_app.conf.task_always_eager and result.state == "PENDING":
        raise HTTPException(status_code=404, detail="Job not found")

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


# =============================================================================
# Pipeline Job Endpoints (Celery-based)
# =============================================================================


@router.post("/detect_track")
async def start_detect_track_celery(req: DetectTrackCeleryRequest):
    """Start a Celery-based detect/track job.

    Returns 202 Accepted with job_id for status polling.

    Check for active jobs before starting to prevent duplicate runs.
    """
    # Check for existing active job
    active_job = check_active_job(req.ep_id, "detect_track")
    if active_job:
        return JSONResponse(
            status_code=409,
            content={
                "job_id": active_job,
                "ep_id": req.ep_id,
                "state": "already_running",
                "message": f"Detect/track job {active_job} is already running for this episode",
            },
        )

    # Build options dict (profile not passed - episode_run.py doesn't accept it)
    options = {
        "stride": req.stride,
        "fps": req.fps,
        "device": req.device,
        "detector": req.detector,
        "tracker": req.tracker,
        "save_frames": req.save_frames,
        "save_crops": req.save_crops,
        "jpeg_quality": req.jpeg_quality,
        "det_thresh": req.det_thresh,
        "max_gap": req.max_gap,
        "scene_detector": req.scene_detector,
        "scene_threshold": req.scene_threshold,
        "scene_min_len": req.scene_min_len,
        "scene_warmup_dets": req.scene_warmup_dets,
        "track_high_thresh": req.track_high_thresh,
        "new_track_thresh": req.new_track_thresh,
        "track_buffer": req.track_buffer,
        "min_box_area": req.min_box_area,
        "cpu_threads": req.cpu_threads,
    }

    # Start Celery task
    result = run_detect_track_task.delay(req.ep_id, options)

    return {
        "job_id": result.id,
        "ep_id": req.ep_id,
        "state": "queued",
        "operation": "detect_track",
        "message": "Detect/track job queued via Celery",
    }


@router.post("/faces_embed")
async def start_faces_embed_celery(req: FacesEmbedCeleryRequest):
    """Start a Celery-based faces embed (harvest) job.

    Returns 202 Accepted with job_id for status polling.
    """
    # Check for existing active job
    active_job = check_active_job(req.ep_id, "faces_embed")
    if active_job:
        return JSONResponse(
            status_code=409,
            content={
                "job_id": active_job,
                "ep_id": req.ep_id,
                "state": "already_running",
                "message": f"Faces embed job {active_job} is already running for this episode",
            },
        )

    # Build options dict (profile not passed - episode_run.py doesn't accept it)
    options = {
        "device": req.device,
        "save_frames": req.save_frames,
        "save_crops": req.save_crops,
        "jpeg_quality": req.jpeg_quality,
        "min_frames_between_crops": req.min_frames_between_crops,
        "thumb_size": req.thumb_size,
    }

    # Start Celery task
    result = run_faces_embed_task.delay(req.ep_id, options)

    return {
        "job_id": result.id,
        "ep_id": req.ep_id,
        "state": "queued",
        "operation": "faces_embed",
        "message": "Faces embed job queued via Celery",
    }


@router.post("/cluster")
async def start_cluster_celery(req: ClusterCeleryRequest):
    """Start a Celery-based clustering job.

    Returns 202 Accepted with job_id for status polling.
    """
    # Check for existing active job
    active_job = check_active_job(req.ep_id, "cluster")
    if active_job:
        return JSONResponse(
            status_code=409,
            content={
                "job_id": active_job,
                "ep_id": req.ep_id,
                "state": "already_running",
                "message": f"Cluster job {active_job} is already running for this episode",
            },
        )

    # Build options dict (profile not passed - episode_run.py doesn't accept it)
    options = {
        "device": req.device,
        "cluster_thresh": req.cluster_thresh,
        "min_cluster_size": req.min_cluster_size,
        "min_identity_sim": req.min_identity_sim,
    }

    # Start Celery task
    result = run_cluster_task.delay(req.ep_id, options)

    return {
        "job_id": result.id,
        "ep_id": req.ep_id,
        "state": "queued",
        "operation": "cluster",
        "message": "Cluster job queued via Celery",
    }


__all__ = ["router"]
