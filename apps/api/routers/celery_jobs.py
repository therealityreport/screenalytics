"""Celery job status endpoints for background operations.

This router handles status polling and cancellation of Celery-based background jobs
including ML pipeline operations (detect/track, faces harvest, cluster) and
grouping operations (manual assign, auto-group).

The Celery-based pipeline jobs provide true async execution via Redis queue,
compared to the subprocess-based jobs in apps/api/routers/jobs.py.

THERMAL SAFETY
==============
The ML pipeline endpoints enforce CPU-safe defaults to prevent laptop overheating:

1. Profile-based defaults:
   - low_power:   stride=12, fps=15, cpu_threads=2 (default for CPU/CoreML/MPS)
   - balanced:    stride=6,  fps=24, cpu_threads=4 (default for CUDA)
   - performance: stride=4,  fps=30, cpu_threads=8 (auto-downgraded on non-CUDA)

2. Device-aware profile resolution:
   - CPU/CoreML/MPS/Metal → low_power profile by default
   - CUDA → balanced profile by default
   - "performance" profile rejected on non-CUDA devices (auto-downgraded to balanced)

3. CPU thread limits:
   - Always applied via environment variables: OMP_NUM_THREADS, MKL_NUM_THREADS,
     OPENBLAS_NUM_THREADS, VECLIB_MAXIMUM_THREADS, NUMEXPR_NUM_THREADS, etc.
   - Default: 2 threads for laptops (SCREENALYTICS_MAX_CPU_THREADS env var)
   - Combined with worker concurrency=2, worst case = 4 total threads

4. Configuration warnings:
   - Low stride (<= 2) on CPU devices: high CPU load warning
   - save_frames + save_crops with low stride: resource-intensive warning

Override defaults by setting SCREENALYTICS_MAX_CPU_THREADS environment variable.
"""

from __future__ import annotations

import logging
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, Literal, Any, Dict

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

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/celery_jobs", tags=["celery_jobs"])


# =============================================================================
# Performance Profiles for CPU-Safe Execution
# =============================================================================
# These profiles control thermal behavior on laptops. The key insight is that
# Celery workers run in separate processes that can peg all CPU cores if not
# constrained. Default to conservative settings for laptop-friendly runs.

PROFILE_DEFAULTS = {
    "low_power": {
        "stride": 12,      # Every 12th frame - fast, fewer false positives
        "fps": 15.0,
        "save_frames": False,
        "save_crops": True,
        "cpu_threads": 2,  # Critical for laptop thermal control
    },
    "balanced": {
        "stride": 6,       # Every 6th frame - good balance
        "fps": 24.0,
        "save_frames": False,
        "save_crops": True,
        "cpu_threads": 4,
    },
    "performance": {
        "stride": 4,       # Every 4th frame - thorough
        "fps": 30.0,
        "save_frames": False,
        "save_crops": True,
        "cpu_threads": 8,
    },
}

# Device-to-profile mapping: laptop-friendly devices get low_power by default
DEVICE_DEFAULT_PROFILE = {
    "coreml": "low_power",
    "mps": "low_power",
    "cpu": "low_power",
    "metal": "low_power",
    "apple": "low_power",
    "cuda": "balanced",    # CUDA GPUs can handle more
    "auto": "low_power",   # Be conservative by default
}

# CPU-only devices where "performance" profile should be rejected/downgraded
CPU_BOUND_DEVICES = {"cpu", "coreml", "mps", "metal", "apple", "auto"}

# Maximum safe CPU threads for laptops (can be overridden via env var)
DEFAULT_LAPTOP_CPU_THREADS = int(os.environ.get("SCREENALYTICS_MAX_CPU_THREADS", "2"))


def _resolve_profile(device: str, profile: str | None) -> tuple[str, str | None]:
    """Resolve the effective profile for a device, applying safety guardrails.

    Returns:
        (resolved_profile, warning_message or None)
    """
    device_lower = (device or "auto").strip().lower()

    # Determine default profile based on device
    default_profile = DEVICE_DEFAULT_PROFILE.get(device_lower, "low_power")

    if profile is None:
        return default_profile, None

    profile_lower = profile.strip().lower()

    # Validate profile name
    if profile_lower not in PROFILE_DEFAULTS:
        return default_profile, f"Unknown profile '{profile}', using {default_profile}"

    # Guardrail: performance profile on CPU-bound devices is dangerous
    if profile_lower == "performance" and device_lower in CPU_BOUND_DEVICES:
        warning = (
            f"Profile 'performance' is not recommended for {device} devices (causes overheating). "
            f"Auto-downgrading to 'balanced'. Use CUDA for performance profile."
        )
        LOGGER.warning(warning)
        return "balanced", warning

    return profile_lower, None


def _apply_profile_defaults(
    options: dict,
    profile: str,
    device: str,
) -> tuple[dict, list[str]]:
    """Apply profile-based defaults to options, with explicit override precedence.

    Precedence (highest to lowest):
        1. Explicit options from request
        2. Profile defaults
        3. Built-in safe defaults

    Returns:
        (modified_options, list of warnings)
    """
    warnings: list[str] = []
    profile_settings = PROFILE_DEFAULTS.get(profile, PROFILE_DEFAULTS["low_power"])
    device_lower = (device or "auto").strip().lower()

    # Apply profile defaults only where options are None or not explicitly set
    # Note: stride has a default in Pydantic, so we check if it matches the schema default

    # CPU threads: ALWAYS apply a limit for thermal safety
    if options.get("cpu_threads") is None:
        # Use profile default, but cap at laptop-safe maximum
        profile_threads = profile_settings.get("cpu_threads", 2)
        safe_threads = min(profile_threads, DEFAULT_LAPTOP_CPU_THREADS)
        options["cpu_threads"] = safe_threads
        LOGGER.info(f"Applying CPU thread limit: {safe_threads} (profile={profile})")

    # FPS: apply profile default if not set
    if options.get("fps") is None:
        options["fps"] = profile_settings.get("fps")

    # Guardrail: dangerous configuration detection
    is_cpu_bound = device_lower in CPU_BOUND_DEVICES
    stride = options.get("stride", 6)
    save_frames = options.get("save_frames", False)
    save_crops = options.get("save_crops", False)

    # Warn about dangerous combinations
    if is_cpu_bound and stride <= 2:
        warnings.append(
            f"stride={stride} on {device} will cause high CPU load. Consider stride >= 6."
        )

    if is_cpu_bound and save_frames and save_crops and stride <= 4:
        warnings.append(
            "save_frames=True + save_crops=True with low stride on CPU is very resource-intensive."
        )

    return options, warnings


# =============================================================================
# Request Models
# =============================================================================

DEVICE_LITERAL = Literal["auto", "cpu", "mps", "coreml", "metal", "apple", "cuda"]
EXECUTION_MODE_LITERAL = Literal["redis", "local"]


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
    execution_mode: Optional[EXECUTION_MODE_LITERAL] = Field(
        "redis",
        description="Execution mode: 'redis' enqueues job via Celery, 'local' runs synchronously in-process"
    )


class FacesEmbedCeleryRequest(BaseModel):
    """Request model for Celery-based faces embed (harvest) job."""
    ep_id: str = Field(..., description="Episode identifier")
    device: DEVICE_LITERAL = Field("auto", description="Execution device")
    save_frames: bool = Field(False, description="Save sampled frames")
    save_crops: bool = Field(True, description="Save face crops")
    jpeg_quality: int = Field(72, ge=1, le=100, description="JPEG quality")
    min_frames_between_crops: int = Field(32, ge=1, description="Min frames between crops")
    thumb_size: int = Field(256, ge=64, le=512, description="Thumbnail size")
    cpu_threads: Optional[int] = Field(None, ge=1, le=16, description="CPU thread cap")
    profile: Optional[str] = Field(None, description="Performance profile")
    execution_mode: Optional[EXECUTION_MODE_LITERAL] = Field(
        "redis",
        description="Execution mode: 'redis' enqueues job via Celery, 'local' runs synchronously in-process"
    )


class ClusterCeleryRequest(BaseModel):
    """Request model for Celery-based cluster job."""
    ep_id: str = Field(..., description="Episode identifier")
    device: DEVICE_LITERAL = Field("auto", description="Execution device")
    cluster_thresh: float = Field(0.7, ge=0.2, le=0.99, description="Clustering threshold")
    min_cluster_size: int = Field(2, ge=1, description="Minimum cluster size")
    min_identity_sim: Optional[float] = Field(0.5, ge=0.0, le=0.99, description="Min identity similarity")
    cpu_threads: Optional[int] = Field(None, ge=1, le=16, description="CPU thread cap")
    profile: Optional[str] = Field(None, description="Performance profile")
    execution_mode: Optional[EXECUTION_MODE_LITERAL] = Field(
        "redis",
        description="Execution mode: 'redis' enqueues job via Celery, 'local' runs synchronously in-process"
    )


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
# Local Execution Helper (for execution_mode="local")
# =============================================================================


def _find_project_root() -> Path:
    """Find the SCREENALYTICS project root directory."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parents[3]


def _run_local_subprocess(
    command: list[str],
    episode_id: str,
    operation: str,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a pipeline command synchronously as subprocess.

    This is used for local execution mode to run the pipeline directly
    instead of queuing via Celery. Uses the same subprocess approach as
    the Celery worker for consistency.

    Args:
        command: Command list to execute
        episode_id: Episode ID for logging
        operation: Operation name (detect_track, faces_embed, cluster)
        options: Job options for logging and env setup

    Returns:
        Result dict with status, output, and progress data
    """
    project_root = _find_project_root()

    LOGGER.info(f"[{episode_id}] Starting local {operation}")
    LOGGER.info(f"[{episode_id}] Command: {' '.join(command)}")

    # Set up environment with CPU thread limits
    env = os.environ.copy()
    cpu_threads = options.get("cpu_threads")
    if cpu_threads:
        max_allowed = os.cpu_count() or 8
        threads = max(1, min(int(cpu_threads), max_allowed))
        env.update({
            "SCREENALYTICS_MAX_CPU_THREADS": str(threads),
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "VECLIB_MAXIMUM_THREADS": str(threads),
            "NUMEXPR_NUM_THREADS": str(threads),
            "ORT_INTRA_OP_NUM_THREADS": str(threads),
            "ORT_INTER_OP_NUM_THREADS": "1",
        })

    try:
        # Run subprocess synchronously
        result = subprocess.run(
            command,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else f"Exit code {result.returncode}"
            LOGGER.error(f"[{episode_id}] local {operation} failed: {error_msg}")
            return {
                "status": "error",
                "ep_id": episode_id,
                "operation": operation,
                "execution_mode": "local",
                "error": error_msg,
                "return_code": result.returncode,
            }

        LOGGER.info(f"[{episode_id}] local {operation} completed successfully")
        return {
            "status": "completed",
            "ep_id": episode_id,
            "operation": operation,
            "execution_mode": "local",
            "return_code": 0,
            "stdout": result.stdout[:2000] if result.stdout else None,
        }

    except subprocess.TimeoutExpired:
        LOGGER.error(f"[{episode_id}] local {operation} timed out after 1 hour")
        return {
            "status": "error",
            "ep_id": episode_id,
            "operation": operation,
            "execution_mode": "local",
            "error": "Job timed out after 1 hour",
        }
    except Exception as e:
        LOGGER.exception(f"[{episode_id}] local {operation} raised exception: {e}")
        return {
            "status": "error",
            "ep_id": episode_id,
            "operation": operation,
            "execution_mode": "local",
            "error": str(e),
        }


def _build_detect_track_command(
    episode_id: str,
    options: Dict[str, Any],
    project_root: Path,
) -> list[str]:
    """Build command for detect_track pipeline."""
    from py_screenalytics.artifacts import get_path

    video_path = get_path(episode_id, "video")
    manifests_dir = get_path(episode_id, "detections").parent
    progress_file = manifests_dir / "progress.json"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(project_root / "tools" / "episode_run.py"),
        "--ep-id", episode_id,
        "--video", str(video_path),
        "--stride", str(options.get("stride", 6)),
        "--device", options.get("device", "auto"),
        "--progress-file", str(progress_file),
    ]

    fps_value = options.get("fps")
    if fps_value is not None and fps_value > 0:
        command += ["--fps", str(fps_value)]
    if options.get("detector"):
        command += ["--detector", options["detector"]]
    if options.get("tracker"):
        command += ["--tracker", options["tracker"]]
    if options.get("save_frames"):
        command.append("--save-frames")
    if options.get("save_crops"):
        command.append("--save-crops")
    if options.get("jpeg_quality"):
        command += ["--jpeg-quality", str(options["jpeg_quality"])]
    if options.get("det_thresh") is not None:
        command += ["--det-thresh", str(options["det_thresh"])]
    if options.get("max_gap"):
        command += ["--max-gap", str(options["max_gap"])]
    if options.get("scene_detector"):
        command += ["--scene-detector", options["scene_detector"]]
    if options.get("scene_threshold") is not None:
        command += ["--scene-threshold", str(options["scene_threshold"])]
    if options.get("scene_min_len") is not None:
        command += ["--scene-min-len", str(options["scene_min_len"])]
    if options.get("scene_warmup_dets") is not None:
        command += ["--scene-warmup-dets", str(options["scene_warmup_dets"])]
    if options.get("track_high_thresh") is not None:
        command += ["--track-high-thresh", str(options["track_high_thresh"])]
    if options.get("new_track_thresh") is not None:
        command += ["--new-track-thresh", str(options["new_track_thresh"])]
    if options.get("track_buffer") is not None:
        command += ["--track-buffer", str(options["track_buffer"])]
    if options.get("min_box_area") is not None:
        command += ["--min-box-area", str(options["min_box_area"])]

    return command


def _build_faces_embed_command(
    episode_id: str,
    options: Dict[str, Any],
    project_root: Path,
) -> list[str]:
    """Build command for faces_embed pipeline."""
    from py_screenalytics.artifacts import get_path

    manifests_dir = get_path(episode_id, "detections").parent
    progress_file = manifests_dir / "progress.json"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(project_root / "tools" / "episode_run.py"),
        "--ep-id", episode_id,
        "--faces-embed",
        "--device", options.get("device", "auto"),
        "--progress-file", str(progress_file),
    ]

    if options.get("save_frames"):
        command.append("--save-frames")
    if options.get("save_crops"):
        command.append("--save-crops")
    if options.get("jpeg_quality"):
        command += ["--jpeg-quality", str(options["jpeg_quality"])]
    if options.get("min_frames_between_crops"):
        command += ["--min-frames-between-crops", str(options["min_frames_between_crops"])]
    if options.get("thumb_size"):
        command += ["--thumb-size", str(options["thumb_size"])]

    return command


def _build_cluster_command(
    episode_id: str,
    options: Dict[str, Any],
    project_root: Path,
) -> list[str]:
    """Build command for cluster pipeline."""
    from py_screenalytics.artifacts import get_path

    manifests_dir = get_path(episode_id, "detections").parent
    progress_file = manifests_dir / "progress.json"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(project_root / "tools" / "episode_run.py"),
        "--ep-id", episode_id,
        "--cluster",
        "--device", options.get("device", "auto"),
        "--progress-file", str(progress_file),
    ]

    if options.get("cluster_thresh") is not None:
        command += ["--cluster-thresh", str(options["cluster_thresh"])]
    if options.get("min_cluster_size"):
        command += ["--min-cluster-size", str(options["min_cluster_size"])]
    if options.get("min_identity_sim") is not None:
        command += ["--min-identity-sim", str(options["min_identity_sim"])]

    return command


# =============================================================================
# Pipeline Job Endpoints (Celery-based)
# =============================================================================


@router.post("/detect_track")
async def start_detect_track_celery(req: DetectTrackCeleryRequest):
    """Start a detect/track job.

    Execution Mode:
        - execution_mode="redis" (default): Enqueues job via Celery, returns 202 with job_id
        - execution_mode="local": Runs job synchronously in-process, returns result when done

    Thermal Safety:
        - Resolves profile based on device (low_power for CPU/CoreML/MPS)
        - Auto-downgrades "performance" profile on non-CUDA devices
        - Always applies CPU thread limits (default: 2 for laptops)

    Check for active jobs before starting to prevent duplicate runs.
    """
    execution_mode = req.execution_mode or "redis"

    # Check for existing active job (only for redis mode, local mode handles its own locking)
    if execution_mode == "redis":
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

    # Resolve profile based on device (with safety guardrails)
    resolved_profile, profile_warning = _resolve_profile(req.device, req.profile)
    LOGGER.info(f"[{req.ep_id}] Profile: {req.profile} -> {resolved_profile} (device={req.device})")

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

    # Apply profile-based defaults (CPU thread limits, fps, etc.)
    options, config_warnings = _apply_profile_defaults(options, resolved_profile, req.device)

    # Collect all warnings
    all_warnings = []
    if profile_warning:
        all_warnings.append(profile_warning)
    all_warnings.extend(config_warnings)

    LOGGER.info(
        f"[{req.ep_id}] detect_track options: stride={options.get('stride')}, "
        f"cpu_threads={options.get('cpu_threads')}, device={req.device}, profile={resolved_profile}, "
        f"execution_mode={execution_mode}"
    )

    # Handle local execution mode
    if execution_mode == "local":
        project_root = _find_project_root()
        command = _build_detect_track_command(req.ep_id, options, project_root)
        result = _run_local_subprocess(command, req.ep_id, "detect_track", options)

        if all_warnings:
            result["warnings"] = all_warnings
        result["profile"] = resolved_profile

        if result.get("status") == "error":
            return JSONResponse(status_code=500, content=result)
        return result

    # Redis/Celery mode (default)
    result = run_detect_track_task.delay(req.ep_id, options)

    response = {
        "job_id": result.id,
        "ep_id": req.ep_id,
        "state": "queued",
        "operation": "detect_track",
        "execution_mode": "redis",
        "profile": resolved_profile,
        "cpu_threads": options.get("cpu_threads"),
        "message": "Detect/track job queued via Celery",
    }

    if all_warnings:
        response["warnings"] = all_warnings

    return response


@router.post("/faces_embed")
async def start_faces_embed_celery(req: FacesEmbedCeleryRequest):
    """Start a faces embed (harvest) job.

    Execution Mode:
        - execution_mode="redis" (default): Enqueues job via Celery, returns 202 with job_id
        - execution_mode="local": Runs job synchronously in-process, returns result when done

    Thermal Safety:
        - Applies CPU thread limits for laptop-friendly operation
    """
    execution_mode = req.execution_mode or "redis"

    # Check for existing active job (only for redis mode)
    if execution_mode == "redis":
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

    # Resolve profile based on device
    resolved_profile, profile_warning = _resolve_profile(req.device, req.profile)
    LOGGER.info(f"[{req.ep_id}] faces_embed Profile: {req.profile} -> {resolved_profile} (device={req.device})")

    # Build options dict (profile not passed - episode_run.py doesn't accept it)
    options = {
        "device": req.device,
        "save_frames": req.save_frames,
        "save_crops": req.save_crops,
        "jpeg_quality": req.jpeg_quality,
        "min_frames_between_crops": req.min_frames_between_crops,
        "thumb_size": req.thumb_size,
        "cpu_threads": req.cpu_threads,
    }

    # Apply profile-based CPU thread defaults
    options, config_warnings = _apply_profile_defaults(options, resolved_profile, req.device)

    LOGGER.info(
        f"[{req.ep_id}] faces_embed options: cpu_threads={options.get('cpu_threads')}, "
        f"device={req.device}, profile={resolved_profile}, execution_mode={execution_mode}"
    )

    # Collect all warnings
    all_warnings = []
    if profile_warning:
        all_warnings.append(profile_warning)
    all_warnings.extend(config_warnings)

    # Handle local execution mode
    if execution_mode == "local":
        project_root = _find_project_root()
        command = _build_faces_embed_command(req.ep_id, options, project_root)
        result = _run_local_subprocess(command, req.ep_id, "faces_embed", options)

        if all_warnings:
            result["warnings"] = all_warnings
        result["profile"] = resolved_profile

        if result.get("status") == "error":
            return JSONResponse(status_code=500, content=result)
        return result

    # Redis/Celery mode (default)
    result = run_faces_embed_task.delay(req.ep_id, options)

    response = {
        "job_id": result.id,
        "ep_id": req.ep_id,
        "state": "queued",
        "operation": "faces_embed",
        "execution_mode": "redis",
        "profile": resolved_profile,
        "cpu_threads": options.get("cpu_threads"),
        "message": "Faces embed job queued via Celery",
    }

    if all_warnings:
        response["warnings"] = all_warnings

    return response


@router.post("/cluster")
async def start_cluster_celery(req: ClusterCeleryRequest):
    """Start a clustering job.

    Execution Mode:
        - execution_mode="redis" (default): Enqueues job via Celery, returns 202 with job_id
        - execution_mode="local": Runs job synchronously in-process, returns result when done

    Thermal Safety:
        - Applies CPU thread limits for laptop-friendly operation
    """
    execution_mode = req.execution_mode or "redis"

    # Check for existing active job (only for redis mode)
    if execution_mode == "redis":
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

    # Resolve profile based on device
    resolved_profile, profile_warning = _resolve_profile(req.device, req.profile)
    LOGGER.info(f"[{req.ep_id}] cluster Profile: {req.profile} -> {resolved_profile} (device={req.device})")

    # Build options dict (profile not passed - episode_run.py doesn't accept it)
    options = {
        "device": req.device,
        "cluster_thresh": req.cluster_thresh,
        "min_cluster_size": req.min_cluster_size,
        "min_identity_sim": req.min_identity_sim,
        "cpu_threads": req.cpu_threads,
    }

    # Apply profile-based CPU thread defaults
    options, config_warnings = _apply_profile_defaults(options, resolved_profile, req.device)

    LOGGER.info(
        f"[{req.ep_id}] cluster options: cpu_threads={options.get('cpu_threads')}, "
        f"device={req.device}, profile={resolved_profile}, execution_mode={execution_mode}"
    )

    # Collect all warnings
    all_warnings = []
    if profile_warning:
        all_warnings.append(profile_warning)
    all_warnings.extend(config_warnings)

    # Handle local execution mode
    if execution_mode == "local":
        project_root = _find_project_root()
        command = _build_cluster_command(req.ep_id, options, project_root)
        result = _run_local_subprocess(command, req.ep_id, "cluster", options)

        if all_warnings:
            result["warnings"] = all_warnings
        result["profile"] = resolved_profile

        if result.get("status") == "error":
            return JSONResponse(status_code=500, content=result)
        return result

    # Redis/Celery mode (default)
    result = run_cluster_task.delay(req.ep_id, options)

    response = {
        "job_id": result.id,
        "ep_id": req.ep_id,
        "state": "queued",
        "operation": "cluster",
        "execution_mode": "redis",
        "profile": resolved_profile,
        "cpu_threads": options.get("cpu_threads"),
        "message": "Cluster job queued via Celery",
    }

    if all_warnings:
        response["warnings"] = all_warnings

    return response


__all__ = ["router"]
