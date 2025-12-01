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

import asyncio
import io
import json
import logging
import math
import os
import platform
import shutil
import signal
import sys
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Literal, Any, Dict, List, Generator

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
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
from apps.api.services.log_formatter import LogFormatter, format_config_block, format_completion_summary
from celery.result import AsyncResult

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/celery_jobs", tags=["celery_jobs"])


# =============================================================================
# Local Mode CPU Limiting - Thermal Safety
# =============================================================================
# Default CPU limit for local mode: 200% (2 cores worth) for thermal safety
_LOCAL_CPULIMIT_PERCENT = int(os.environ.get("SCREENALYTICS_LOCAL_CPULIMIT_PERCENT", "200"))

try:
    import psutil  # type: ignore
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


def _maybe_wrap_with_cpulimit_local(command: list[str], profile: str = "balanced") -> tuple[list[str], bool]:
    """Wrap command with thermal limiting for local mode safety.

    On macOS: Uses taskpolicy with QoS clamps (works with Metal/CoreML threads)
    On Linux: Uses cpulimit (SIGSTOP/SIGCONT based)

    Note: cpulimit does NOT work on macOS with CoreML/Metal because those
    frameworks use GPU threads that ignore SIGSTOP/SIGCONT signals.

    Args:
        command: The command to wrap
        profile: Thermal profile - "low_power" uses background QoS, others use utility

    Returns:
        tuple of (wrapped_command, thermal_limit_applied)
    """
    if _LOCAL_CPULIMIT_PERCENT <= 0:
        return command, False

    # macOS: use taskpolicy with QoS clamp (works with CoreML/Metal threads)
    if platform.system() == "Darwin":
        # background = lowest priority, minimal thermal impact (for low_power)
        # utility = reduced priority, thermally managed (for balanced/performance)
        qos_clamp = "background" if profile == "low_power" else "utility"
        taskpolicy = shutil.which("taskpolicy")
        if taskpolicy:
            LOGGER.info(
                "[LOCAL MODE] Using taskpolicy -c %s for thermal safety (macOS native)",
                qos_clamp,
            )
            return [taskpolicy, "-c", qos_clamp, "--", *command], True
        else:
            LOGGER.warning(
                "taskpolicy not found on macOS. This is unexpected - taskpolicy should "
                "be available on all modern macOS versions. Falling back to no thermal limit."
            )
            return command, False

    # Linux: use cpulimit (SIGSTOP/SIGCONT based - works for CPU-bound processes)
    binary = shutil.which("cpulimit")
    if not binary:
        LOGGER.warning(
            "cpulimit binary not found for local mode. "
            "CPU usage will NOT be capped at %d%%. "
            "Install cpulimit (apt install cpulimit) "
            "or use psutil CPU affinity fallback.",
            _LOCAL_CPULIMIT_PERCENT,
        )
        return command, False

    LOGGER.info(
        "[LOCAL MODE] Wrapping command with cpulimit at %d%% for thermal safety",
        _LOCAL_CPULIMIT_PERCENT,
    )
    return [binary, "-l", str(_LOCAL_CPULIMIT_PERCENT), "-i", "--", *command], True


def _apply_cpu_affinity_fallback_local(pid: int, limit_percent: int) -> bool:
    """Apply CPU affinity as fallback when cpulimit is not available.

    Uses psutil to restrict the process to a subset of available CPUs
    proportional to the requested CPU limit percentage.

    Returns:
        True if affinity was successfully applied, False otherwise.
    """
    if not _PSUTIL_AVAILABLE:
        LOGGER.debug("psutil not available; CPU affinity fallback skipped for pid %d", pid)
        return False

    try:
        proc = psutil.Process(pid)
        cpu_count = psutil.cpu_count()
        if cpu_count is None or cpu_count <= 1:
            return False

        # Calculate how many CPUs to use based on limit percentage
        # e.g., 200% with 8 cores -> use min(2, 8) = 2 cores
        cores_to_use = max(1, min(cpu_count, int(math.ceil(limit_percent / 100.0))))

        # Set affinity to first N cores
        affinity_list = list(range(cores_to_use))
        proc.cpu_affinity(affinity_list)
        LOGGER.info(
            "[LOCAL MODE] Applied CPU affinity fallback for pid %d: using %d of %d cores (limit=%d%%)",
            pid, cores_to_use, cpu_count, limit_percent
        )
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError) as exc:
        LOGGER.debug("CPU affinity fallback failed for pid %d: %s", pid, exc)
        return False


# =============================================================================
# Local Job Tracking - Prevents duplicate jobs and enables cancellation
# =============================================================================
import threading

_running_local_jobs: Dict[str, Dict[str, Any]] = {}  # ep_id::operation -> job info
_running_local_jobs_lock = threading.Lock()


def _get_running_local_job(ep_id: str, operation: str) -> Dict[str, Any] | None:
    """Check if a local job is already running for this episode/operation."""
    key = f"{ep_id}::{operation}"
    with _running_local_jobs_lock:
        return _running_local_jobs.get(key)


def _register_local_job(ep_id: str, operation: str, pid: int, job_id: str | None = None) -> None:
    """Register a running local job."""
    key = f"{ep_id}::{operation}"
    with _running_local_jobs_lock:
        _running_local_jobs[key] = {
            "ep_id": ep_id,
            "operation": operation,
            "pid": pid,
            "job_id": job_id or f"local-{ep_id}-{operation}",
            "started_at": __import__("time").time(),
        }
    LOGGER.info(f"[{ep_id}] Registered local {operation} job (PID {pid}, job_id={job_id})")


def _unregister_local_job(ep_id: str, operation: str) -> None:
    """Unregister a local job when it completes or is cancelled.

    Also releases the file-based lock to allow future jobs to run.
    """
    key = f"{ep_id}::{operation}"
    with _running_local_jobs_lock:
        if key in _running_local_jobs:
            del _running_local_jobs[key]
            LOGGER.info(f"[{ep_id}] Unregistered local {operation} job")

    # Also release file-based lock (defined later in this module)
    # Use try/except since _release_job_lock may not be defined during module load
    try:
        _release_job_lock(ep_id, operation)
    except NameError:
        pass  # Lock functions not yet defined during module initialization


def _list_local_jobs(ep_id: str | None = None) -> List[Dict[str, Any]]:
    """List all running local jobs, optionally filtered by episode.

    Also cleans up stale entries for processes that are no longer running.
    """
    import psutil

    result = []
    stale_keys = []

    with _running_local_jobs_lock:
        for key, job_info in list(_running_local_jobs.items()):
            job_ep_id = job_info.get("ep_id")
            pid = job_info.get("pid")

            # Filter by episode if specified
            if ep_id and job_ep_id != ep_id:
                continue

            # Check if process is still running
            is_running = False
            try:
                proc = psutil.Process(pid)
                is_running = proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                is_running = False

            if not is_running:
                stale_keys.append(key)
                continue

            result.append({
                "job_id": job_info.get("job_id", f"local-{job_ep_id}-{job_info.get('operation')}"),
                "ep_id": job_ep_id,
                "operation": job_info.get("operation"),
                "pid": pid,
                "started_at": job_info.get("started_at"),
                "state": "running",
                "source": "local",
            })

        # Clean up stale entries
        for key in stale_keys:
            del _running_local_jobs[key]
            LOGGER.info(f"Cleaned up stale local job: {key}")

    return result


def _kill_process_tree_by_pid(pid: int) -> tuple[bool, str]:
    """Kill a process (and its children) by PID using the process group."""
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            os.waitpid(-pgid, os.WNOHANG)
        except Exception:
            pass
        return True, "terminated"
    except ProcessLookupError:
        return False, "not_found"
    except Exception as exc:  # Defensive fallback if process groups fail
        try:
            import psutil  # type: ignore
            proc = psutil.Process(pid)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                proc.kill()
            return True, "terminated"
        except Exception as inner_exc:
            return False, f"error:{inner_exc}"


def _is_local_job_running(ep_id: str, operation: str) -> bool:
    """Check if a local job is currently running for this episode/operation."""
    jobs = _list_local_jobs(ep_id)
    return any(j.get("operation") == operation for j in jobs)


def _detect_running_episode_processes() -> List[Dict[str, Any]]:
    """Detect running episode_run.py processes by scanning the process list.

    This catches processes that weren't registered (e.g., after API restart).
    """
    import psutil
    import re

    result = []
    ep_id_pattern = re.compile(r"--ep-id\s+(\S+)")

    try:
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                cmdline_str = " ".join(cmdline) if cmdline else ""

                # Check if this is an episode_run.py process
                if "episode_run.py" not in cmdline_str:
                    continue

                # Extract ep_id from command line
                match = ep_id_pattern.search(cmdline_str)
                if not match:
                    continue

                ep_id = match.group(1)
                pid = proc.info["pid"]
                create_time = proc.info.get("create_time", 0)

                # Determine operation type
                if "--faces-embed" in cmdline_str:
                    operation = "faces_embed"
                elif "--cluster" in cmdline_str:
                    operation = "cluster"
                else:
                    operation = "detect_track"

                result.append({
                    "job_id": f"orphan-{pid}",
                    "ep_id": ep_id,
                    "operation": operation,
                    "pid": pid,
                    "started_at": create_time,
                    "state": "running",
                    "source": "detected",  # Not registered, detected by scanning
                })

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

    except Exception as e:
        LOGGER.warning(f"Failed to scan for episode_run processes: {e}")

    return result


def get_all_running_jobs(ep_id: str | None = None) -> List[Dict[str, Any]]:
    """Get all running jobs (registered + detected orphan processes)."""
    # Get registered local jobs
    registered = _list_local_jobs(ep_id)
    registered_pids = {j.get("pid") for j in registered}

    # Detect any orphan processes not in our registry
    detected = _detect_running_episode_processes()

    # Filter by ep_id if specified and exclude already-registered PIDs
    result = list(registered)
    for job in detected:
        if job.get("pid") in registered_pids:
            continue
        if ep_id and job.get("ep_id") != ep_id:
            continue
        result.append(job)

    return result


# =============================================================================
# File-Based Job Locking - Prevents duplicate jobs across API restarts
# =============================================================================
# Lock files are stored at: data/manifests/{episode_id}/.lock_{operation}
# Each lock contains: {"pid": N, "started_at": "...", "hostname": "..."}


def _get_lock_path(ep_id: str, operation: str) -> Path:
    """Get the path for a job lock file."""
    from py_screenalytics.artifacts import get_path

    manifests_dir = get_path(ep_id, "detections").parent
    manifests_dir.mkdir(parents=True, exist_ok=True)
    return manifests_dir / f".lock_{operation}"


def _is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    import psutil

    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def _acquire_job_lock(ep_id: str, operation: str) -> tuple[bool, str | None]:
    """Acquire file-based lock for a job.

    Returns:
        (success, error_message): True if lock acquired, False with message if already locked
    """
    import socket
    from datetime import datetime

    lock_path = _get_lock_path(ep_id, operation)

    # Check if lock already exists
    if lock_path.exists():
        try:
            lock_data = json.loads(lock_path.read_text())
            existing_pid = lock_data.get("pid")
            existing_host = lock_data.get("hostname", "unknown")

            # Check if the locking process is still running (on same host)
            if existing_pid and socket.gethostname() == existing_host:
                if _is_process_running(existing_pid):
                    started = lock_data.get("started_at", "unknown")
                    return False, f"Job already running (PID {existing_pid}, started {started})"
                # Stale lock from dead process - safe to remove
                LOGGER.info(f"[{ep_id}] Removing stale lock for {operation} (PID {existing_pid} no longer running)")
            else:
                # Different host or can't verify - check lock age
                started_str = lock_data.get("started_at")
                if started_str:
                    try:
                        started = datetime.fromisoformat(started_str.replace("Z", "+00:00"))
                        age_hours = (datetime.now(started.tzinfo) - started).total_seconds() / 3600
                        if age_hours < 4:  # Assume job could still be running if < 4 hours old
                            return False, f"Job may be running on {existing_host} (started {started_str})"
                    except (ValueError, TypeError):
                        pass
                # Old lock or can't parse - assume stale
                LOGGER.info(f"[{ep_id}] Removing potentially stale lock for {operation}")

            lock_path.unlink(missing_ok=True)
        except (json.JSONDecodeError, IOError) as e:
            LOGGER.warning(f"[{ep_id}] Could not read lock file, removing: {e}")
            lock_path.unlink(missing_ok=True)

    # Create the lock file
    try:
        lock_data = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": datetime.utcnow().isoformat() + "Z",
            "ep_id": ep_id,
            "operation": operation,
        }
        lock_path.write_text(json.dumps(lock_data, indent=2))
        LOGGER.info(f"[{ep_id}] Acquired lock for {operation}")
        return True, None
    except IOError as e:
        return False, f"Failed to create lock: {e}"


def _release_job_lock(ep_id: str, operation: str) -> None:
    """Release file-based lock for a job."""
    lock_path = _get_lock_path(ep_id, operation)
    try:
        lock_path.unlink(missing_ok=True)
        LOGGER.info(f"[{ep_id}] Released lock for {operation}")
    except IOError as e:
        LOGGER.warning(f"[{ep_id}] Failed to release lock for {operation}: {e}")


def _check_job_lock(ep_id: str, operation: str) -> Dict[str, Any] | None:
    """Check if a lock exists and return lock info if active.

    Returns:
        Lock info dict if active lock exists, None otherwise
    """
    import socket

    lock_path = _get_lock_path(ep_id, operation)
    if not lock_path.exists():
        return None

    try:
        lock_data = json.loads(lock_path.read_text())
        existing_pid = lock_data.get("pid")
        existing_host = lock_data.get("hostname", "unknown")

        # If same host, check if process is running
        if socket.gethostname() == existing_host:
            if existing_pid and _is_process_running(existing_pid):
                return lock_data
            # Process dead, lock is stale
            return None

        # Different host - can't verify, return lock data (caller decides)
        return lock_data
    except (json.JSONDecodeError, IOError):
        return None


# =============================================================================
# Per-Episode Log Storage - Persist logs for UI display after completion
# =============================================================================
# Logs are stored at: data/manifests/{episode_id}/logs/{operation}_latest.json
# Each log file contains: {"logs": [...], "status": "completed"|"error", "elapsed_seconds": N, ...}


def _get_log_storage_path(episode_id: str, operation: str) -> Path:
    """Get the path for storing/retrieving operation logs."""
    from py_screenalytics.artifacts import get_path

    manifests_dir = get_path(episode_id, "detections").parent
    logs_dir = manifests_dir / "logs"
    return logs_dir / f"{operation}_latest.json"


def save_operation_logs(
    episode_id: str,
    operation: str,
    formatted_logs: List[str],
    status: str,
    elapsed_seconds: float,
    raw_logs: List[str] | None = None,
    extra: Dict[str, Any] | None = None,
) -> bool:
    """Save operation logs to persistent storage.

    Stores both formatted (human-readable) and raw logs for debugging.

    Args:
        episode_id: Episode identifier
        operation: Operation name (detect_track, faces_embed, cluster)
        formatted_logs: List of formatted/cleaned log lines
        status: Final status (completed, error, cancelled, timeout)
        elapsed_seconds: Total runtime in seconds
        raw_logs: Optional list of raw unprocessed log lines (for debugging)
        extra: Additional metadata to store

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        log_path = _get_log_storage_path(episode_id, operation)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "episode_id": episode_id,
            "operation": operation,
            "status": status,
            "logs": formatted_logs,  # Primary formatted logs
            "raw_logs": raw_logs or formatted_logs,  # Raw logs for debugging
            "elapsed_seconds": elapsed_seconds,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **(extra or {}),
        }

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        LOGGER.info(f"[{episode_id}] Saved {len(formatted_logs)} formatted + {len(raw_logs or [])} raw log lines for {operation}")
        return True

    except Exception as e:
        LOGGER.warning(f"[{episode_id}] Failed to save logs for {operation}: {e}")
        return False


def load_operation_logs(episode_id: str, operation: str) -> Dict[str, Any] | None:
    """Load operation logs from persistent storage.

    Returns:
        Dict with logs, status, elapsed_seconds, etc. or None if not found
    """
    try:
        log_path = _get_log_storage_path(episode_id, operation)
        if not log_path.exists():
            return None

        with open(log_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        LOGGER.warning(f"[{episode_id}] Failed to load logs for {operation}: {e}")
        return None


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
# Default raised from 2→3 for better local mode performance while remaining thermal-safe
DEFAULT_LAPTOP_CPU_THREADS = int(os.environ.get("SCREENALYTICS_MAX_CPU_THREADS", "3"))


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
    allow_cpu_fallback: bool = Field(
        False,
        description="Allow falling back to CPU if requested accelerator (coreml/cuda) is unavailable. "
                    "If False (default), fails fast with an error when accelerator is unavailable."
    )
    execution_mode: Optional[EXECUTION_MODE_LITERAL] = Field(
        "redis",
        description="Execution mode: 'redis' enqueues job via Celery, 'local' runs synchronously in-process"
    )


class FacesEmbedCeleryRequest(BaseModel):
    """Request model for Celery-based faces embed (harvest) job."""
    ep_id: str = Field(..., description="Episode identifier")
    device: DEVICE_LITERAL = Field("auto", description="Execution device")
    save_frames: bool = Field(False, description="Save sampled frames")
    save_crops: bool = Field(False, description="Save face crops (enable explicitly to avoid storage bloat)")
    jpeg_quality: int = Field(72, ge=1, le=100, description="JPEG quality")
    min_frames_between_crops: int = Field(32, ge=1, description="Min frames between crops")
    thumb_size: int = Field(256, ge=64, le=512, description="Thumbnail size")
    cpu_threads: Optional[int] = Field(None, ge=1, le=16, description="CPU thread cap")
    profile: Optional[str] = Field(None, description="Performance profile")
    allow_cpu_fallback: bool = Field(
        False,
        description="Allow falling back to CPU if requested accelerator is unavailable."
    )
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
    allow_cpu_fallback: bool = Field(
        False,
        description="Allow falling back to CPU if requested accelerator is unavailable."
    )
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


@router.get("/local")
async def list_local_jobs(ep_id: str | None = None):
    """List running local jobs (not Celery).

    These are subprocess jobs running on the local machine, detected either
    from our registry or by scanning for episode_run.py processes.

    Args:
        ep_id: Optional filter by episode ID
    """
    jobs = get_all_running_jobs(ep_id)
    return {
        "jobs": jobs,
        "count": len(jobs),
    }


@router.get("/logs/{ep_id}/{operation}")
async def get_operation_logs(ep_id: str, operation: str):
    """Get the most recent logs for an operation.

    This endpoint returns the last saved logs for a given episode and operation.
    Logs are persisted when local mode jobs complete (success, error, or cancelled).

    Args:
        ep_id: Episode identifier
        operation: Operation name (detect_track, faces_embed, cluster)

    Returns:
        - status: "completed" | "error" | "cancelled" | "timeout" | "none"
        - logs: List of log lines (empty if status is "none")
        - elapsed_seconds: Runtime in seconds
        - updated_at: ISO timestamp of when logs were saved
    """
    valid_operations = {"detect_track", "faces_embed", "cluster"}
    if operation not in valid_operations:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid operation '{operation}'. Must be one of: {', '.join(valid_operations)}"
        )

    data = load_operation_logs(ep_id, operation)

    if data is None:
        return {
            "status": "none",
            "logs": [],
            "elapsed_seconds": 0,
            "updated_at": None,
            "episode_id": ep_id,
            "operation": operation,
        }

    return data


@router.get("/stream/{job_id}")
async def stream_celery_job(job_id: str, ep_id: str | None = None):
    """Stream progress for a Celery job as NDJSON (for queue mode UI).

    Args:
        job_id: Celery task ID
        ep_id: Optional episode ID for audio pipeline jobs (enables progress file polling)
    """

    def _progress_line(state: str, progress: dict | None, status: str | None = None) -> str:
        progress = progress or {}
        pct = progress.get("progress", 0)
        return json.dumps({
            "type": "progress",
            "state": state,
            "progress": pct,
            "message": progress.get("message", ""),
            "step": progress.get("step") or progress.get("phase") or "",
            "step_name": progress.get("step_name") or progress.get("step") or progress.get("phase") or "",
            "step_order": progress.get("step_order", 0),
            "total_steps": progress.get("total_steps", 0),
            "timestamp": time.time(),
            "status": status or state,
        }) + "\n"

    def _summary_line(status: str, error: str | None, elapsed: float) -> str:
        payload = {
            "type": "summary",
            "status": status,
            "elapsed_seconds": elapsed,
        }
        if error:
            payload["error"] = error
        return json.dumps(payload) + "\n"

    def _read_progress_file(ep_id: str) -> dict | None:
        """Read audio_progress.json for the episode if it exists."""
        if not ep_id:
            return None
        data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
        progress_file = data_root / "manifests" / ep_id / "audio_progress.json"
        if progress_file.exists():
            try:
                return json.loads(progress_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return None

    def _gen():
        result = AsyncResult(job_id, app=celery_app)
        start = time.time()
        poll_interval = 2.0
        max_runtime = 7200.0

        # If job is unknown, return 404 via summary
        if result.state == "PENDING" and not result.backend.get_task_meta(job_id):
            yield _summary_line("not_found", "Job not found", 0.0)
            return

        while True:
            state = result.state
            info = result.info if isinstance(result.info, dict) else {}

            # For chain tasks, also check progress file (written by individual tasks)
            progress = _read_progress_file(ep_id) if ep_id else None
            if not progress:
                progress = info.get("progress") if isinstance(info, dict) else {}

            # Emit progress line
            yield _progress_line(state.lower(), progress)

            if result.ready():
                elapsed = time.time() - start
                if result.successful():
                    yield _summary_line("success", None, elapsed)
                elif result.failed():
                    err = ""
                    if isinstance(info, dict):
                        err = str(info.get("exc_message") or info.get("error") or info)
                    else:
                        err = str(info)
                    yield _summary_line("error", err, elapsed)
                elif state == "REVOKED":
                    yield _summary_line("cancelled", None, elapsed)
                else:
                    yield _summary_line(state.lower(), None, elapsed)
                break

            if time.time() - start > max_runtime:
                yield _summary_line("timeout", "Timed out waiting for job", max_runtime)
                break

            time.sleep(poll_interval)

    headers = {
        "Cache-Control": "no-cache",
        "X-Content-Type-Options": "nosniff",
    }
    return StreamingResponse(_gen(), media_type="application/x-ndjson", headers=headers)


@router.get("/{job_id}")
async def get_celery_job_status(job_id: str):
    """Get status of a Celery background job.

    This endpoint returns status of jobs submitted via Redis/Celery mode.
    Local mode jobs are synchronous and do not use this endpoint.

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
        "execution_mode": "redis",
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
    """Cancel a running Celery or local background job.

    Supports:
    - Celery jobs: Sends revoke signal
    - Local jobs (orphan-{pid}, local-{ep_id}-{operation}): Kills process by PID

    Note: This sends a termination signal. Long-running operations may take
    a moment to actually stop.
    """
    import psutil

    # Handle local/orphan jobs by PID
    if job_id.startswith("orphan-"):
        # Extract PID from orphan-{pid} format
        try:
            pid = int(job_id.replace("orphan-", ""))
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    proc.kill()
                LOGGER.info(f"Killed orphan process PID {pid}")
                return {"job_id": job_id, "status": "cancelled", "pid": pid}
            except psutil.NoSuchProcess:
                return {"job_id": job_id, "status": "already_finished", "message": "Process not found"}
            except psutil.AccessDenied:
                return {"job_id": job_id, "status": "error", "message": "Access denied killing process"}
        except ValueError:
            return {"job_id": job_id, "status": "error", "message": "Invalid orphan job ID format"}

    if job_id.startswith("local-"):
        # Extract info from local-{ep_id}-{operation} format
        parts = job_id.replace("local-", "").rsplit("-", 1)
        if len(parts) >= 2:
            ep_id, operation = parts[0], parts[1]
            # Find job in registry
            jobs = _list_local_jobs(ep_id)
            for job in jobs:
                if job.get("operation") == operation:
                    pid = job.get("pid")
                    if pid:
                        success, reason = _kill_process_tree_by_pid(pid)
                        _unregister_local_job(ep_id, operation)

                        # Clear progress file for audio pipeline to stop UI auto-refresh
                        if operation == "audio_pipeline":
                            try:
                                progress_path = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")) / "manifests" / ep_id / "audio_progress.json"
                                progress_path.unlink(missing_ok=True)
                            except Exception:
                                pass

                        if success:
                            LOGGER.info(f"Killed local job PID {pid} ({ep_id}::{operation})")
                            return {"job_id": job_id, "status": "cancelled", "pid": pid}
                        if reason == "not_found":
                            return {"job_id": job_id, "status": "already_finished", "message": "Process not found"}
                        return {"job_id": job_id, "status": "error", "message": reason}
            return {"job_id": job_id, "status": "not_found", "message": "Local job not found in registry"}
        return {"job_id": job_id, "status": "error", "message": "Invalid local job ID format"}

    # Standard Celery job cancellation
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


def _extract_job_metadata(task_info: Dict) -> Dict[str, Any]:
    """Extract ep_id and operation from Celery task info.

    Celery task info includes args/kwargs that contain our job parameters.
    This extracts the episode ID and operation type for matching.
    """
    audio_stage_order = {
        "ingest": 1,
        "separate": 2,
        "enhance": 3,
        "diarize": 4,
        "transcribe": 5,
        "voices": 6,
        "align": 7,
        "export": 8,
        "qc": 9,
        "pipeline": 10,
    }
    result: Dict[str, Any] = {}

    # Try to get ep_id from args (first positional argument)
    args = task_info.get("args") or []
    if args and len(args) > 0:
        result["ep_id"] = args[0]

    # Try to get ep_id from kwargs if not in args
    kwargs = task_info.get("kwargs") or {}
    if "ep_id" in kwargs:
        result["ep_id"] = kwargs["ep_id"]
    elif "episode_id" in kwargs:
        result["ep_id"] = kwargs["episode_id"]

    # Infer operation from task name
    task_name = task_info.get("name", "")
    task_lower = task_name.lower()
    if "detect_track" in task_lower:
        result["operation"] = "detect_track"
    elif "faces_embed" in task_lower or "faces_harvest" in task_lower:
        result["operation"] = "faces_embed"
    elif "cluster" in task_lower:
        result["operation"] = "cluster"
    elif task_lower.startswith("audio."):
        result["operation"] = "audio_pipeline"
        # Capture the specific audio stage for richer progress display
        stage_name = task_lower.split(".", 1)[-1]
        result["stage"] = stage_name
        result["stage_order"] = audio_stage_order.get(stage_name, 0)
    else:
        # Try to extract from task name pattern like "tasks.run_detect_track_task"
        parts = task_name.split(".")
        if parts:
            last_part = parts[-1].replace("run_", "").replace("_task", "")
            result["operation"] = last_part

    return result


@router.post("/kill_all_local")
async def kill_all_local_jobs(ep_id: str | None = None):
    """Kill all local/orphan jobs, optionally filtered by episode.

    This is useful for cleaning up stale processes that weren't properly
    terminated (e.g., after a crash or page refresh).

    Args:
        ep_id: Optional episode ID to filter jobs. If not provided, kills all local jobs.

    Returns:
        List of killed job IDs and their status.
    """
    import psutil

    jobs = get_all_running_jobs(ep_id)
    results = []

    for job in jobs:
        job_id = job.get("job_id", "")
        pid = job.get("pid")
        source = job.get("source", "")
        operation = job.get("operation", "")
        job_ep_id = job.get("ep_id", "")

        if not pid:
            continue

        # Only kill local/detected jobs, not Celery jobs
        if source not in ("local", "detected"):
            continue

        try:
            proc = psutil.Process(pid)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                proc.kill()

            # Unregister from local job registry if registered
            if source == "local" and job_ep_id and operation:
                _unregister_local_job(job_ep_id, operation)

            LOGGER.info(f"Killed local job PID {pid} ({job_ep_id}::{operation})")
            results.append({"job_id": job_id, "pid": pid, "status": "killed"})
        except psutil.NoSuchProcess:
            if source == "local" and job_ep_id and operation:
                _unregister_local_job(job_ep_id, operation)
            results.append({"job_id": job_id, "pid": pid, "status": "already_dead"})
        except psutil.AccessDenied:
            results.append({"job_id": job_id, "pid": pid, "status": "access_denied"})
        except Exception as e:
            results.append({"job_id": job_id, "pid": pid, "status": "error", "message": str(e)})

    return {
        "killed_count": len([r for r in results if r.get("status") == "killed"]),
        "already_dead_count": len([r for r in results if r.get("status") == "already_dead"]),
        "results": results,
    }


@router.get("")
async def list_active_celery_jobs():
    """List currently active Celery jobs.

    Note: This only shows jobs known to the current worker.
    Completed jobs are available via their individual job_id.

    Each job now includes ep_id and operation for matching after page reload.
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
            metadata = _extract_job_metadata(task)
            jobs.append({
                "job_id": task.get("id"),
                "name": task.get("name"),
                "state": "in_progress",
                "worker": worker,
                "ep_id": metadata.get("ep_id"),
                "operation": metadata.get("operation"),
                "stage": metadata.get("stage"),
                "stage_order": metadata.get("stage_order"),
                "source": "celery",
            })

    # Collect scheduled jobs
    for worker, tasks in scheduled.items():
        for task in tasks:
            request = task.get("request", {})
            metadata = _extract_job_metadata(request)
            jobs.append({
                "job_id": request.get("id"),
                "name": request.get("name"),
                "state": "scheduled",
                "worker": worker,
                "ep_id": metadata.get("ep_id"),
                "operation": metadata.get("operation"),
                "stage": metadata.get("stage"),
                "stage_order": metadata.get("stage_order"),
                "source": "celery",
            })

    # Collect reserved (queued) jobs
    for worker, tasks in reserved.items():
        for task in tasks:
            metadata = _extract_job_metadata(task)
            jobs.append({
                "job_id": task.get("id"),
                "name": task.get("name"),
                "state": "queued",
                "worker": worker,
                "ep_id": metadata.get("ep_id"),
                "operation": metadata.get("operation"),
                "stage": metadata.get("stage"),
                "stage_order": metadata.get("stage_order"),
                "source": "celery",
            })

    # Also include local running jobs (registered + detected orphan processes)
    local_jobs = get_all_running_jobs()
    for local_job in local_jobs:
        jobs.append({
            "job_id": local_job.get("job_id"),
            "name": f"local_{local_job.get('operation', 'unknown')}",
            "state": local_job.get("state", "running"),
            "worker": f"local (PID {local_job.get('pid')})",
            "ep_id": local_job.get("ep_id"),
            "operation": local_job.get("operation"),
            "source": local_job.get("source", "local"),
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


def _start_local_subprocess(
    command: list[str],
    episode_id: str,
    operation: str,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Start a pipeline subprocess in the background (non-blocking).

    This starts the subprocess and returns immediately with job info.
    The UI should poll for progress using /episodes/{ep_id}/progress.

    Args:
        command: Command list to execute
        episode_id: Episode ID for logging
        operation: Operation name (detect_track, faces_embed, cluster)
        options: Job options for logging and env setup

    Returns:
        Dict with job_id, status, etc.
    """
    import uuid

    # Check for existing running job
    existing_job = _get_running_local_job(episode_id, operation)
    if existing_job:
        existing_pid = existing_job.get("pid")
        job_id = existing_job.get("job_id", f"local-{episode_id}-{operation}")
        LOGGER.info(f"[{episode_id}] {operation} already running (PID {existing_pid})")
        return {
            "job_id": job_id,
            "ep_id": episode_id,
            "operation": operation,
            "execution_mode": "local",
            "state": "already_running",
            "pid": existing_pid,
            "message": f"Job already running (PID {existing_pid})",
        }

    project_root = _find_project_root()
    job_id = f"local-{uuid.uuid4().hex[:12]}"

    # Build context string for logging
    device = options.get("device", "auto")
    stride = options.get("stride", 6)

    LOGGER.info(f"[{episode_id}] Starting local {operation} (device={device}, stride={stride})")
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
        LOGGER.info(f"[{episode_id}] CPU threads limited to {threads}")
    else:
        # CRITICAL: Always set CPU limits to prevent thermal issues
        default_threads = 2
        env.update({
            "SCREENALYTICS_MAX_CPU_THREADS": str(default_threads),
            "OMP_NUM_THREADS": str(default_threads),
            "MKL_NUM_THREADS": str(default_threads),
            "OPENBLAS_NUM_THREADS": str(default_threads),
            "VECLIB_MAXIMUM_THREADS": str(default_threads),
            "NUMEXPR_NUM_THREADS": str(default_threads),
            "ORT_INTRA_OP_NUM_THREADS": str(default_threads),
            "ORT_INTER_OP_NUM_THREADS": "1",
        })
        LOGGER.info(f"[{episode_id}] CPU threads limited to {default_threads} (default)")

    try:
        # Start subprocess in new process group (non-blocking)
        process = subprocess.Popen(
            command,
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,  # Create new process group
        )

        # Register the job for tracking
        _register_local_job(episode_id, operation, process.pid, job_id)
        LOGGER.info(f"[{episode_id}] Process started (PID {process.pid}, job_id={job_id})")

        return {
            "job_id": job_id,
            "ep_id": episode_id,
            "operation": operation,
            "execution_mode": "local",
            "state": "started",
            "pid": process.pid,
            "message": f"Job started (PID {process.pid}). Poll /episodes/{episode_id}/progress for updates.",
        }

    except Exception as e:
        LOGGER.exception(f"[{episode_id}] Failed to start local {operation}: {e}")
        return {
            "job_id": job_id,
            "ep_id": episode_id,
            "operation": operation,
            "execution_mode": "local",
            "state": "error",
            "error": str(e),
            "message": f"Failed to start: {e}",
        }


def _run_local_subprocess_blocking(
    command: list[str],
    episode_id: str,
    operation: str,
    options: Dict[str, Any],
    timeout: int = 3600,
) -> Dict[str, Any]:
    """Run a pipeline command synchronously, blocking until completion.

    This is the correct implementation for local mode - runs the subprocess
    in the foreground, tied to the HTTP request lifecycle. If the request
    is cancelled, the subprocess is terminated.

    Args:
        command: Command list to execute
        episode_id: Episode ID for logging
        operation: Operation name (detect_track, faces_embed, cluster)
        options: Job options for logging and env setup
        timeout: Maximum time in seconds (default 3600 = 1 hour)

    Returns:
        Result dict with status, logs, and summary data
    """
    import signal
    import time

    # Check for existing running job
    existing_job = _get_running_local_job(episode_id, operation)
    if existing_job:
        existing_pid = existing_job.get("pid")
        LOGGER.warning(f"[{episode_id}] {operation} already running (PID {existing_pid})")
        return {
            "status": "error",
            "ep_id": episode_id,
            "operation": operation,
            "execution_mode": "local",
            "error": f"A {operation} job is already running for this episode (PID {existing_pid}). "
                     "Wait for it to complete or kill it manually.",
            "logs": [f"Existing job PID: {existing_pid}"],
            "elapsed_seconds": 0,
        }

    project_root = _find_project_root()
    logs: List[str] = []

    # Build context string for logging
    device = options.get("device", "auto")
    stride = options.get("stride", 6)
    profile = options.get("profile", "default")
    cpu_threads = options.get("cpu_threads", 2)

    logs.append(f"[LOCAL MODE] Starting {operation}")
    logs.append(f"  Device: {device}, Profile: {profile}")
    logs.append(f"  Stride: {stride}, CPU threads: {cpu_threads}")
    logs.append(f"  This runs synchronously - page refresh will cancel the job.")
    LOGGER.info(f"[{episode_id}] Starting local {operation} (device={device}, stride={stride}, threads={cpu_threads})")
    LOGGER.info(f"[{episode_id}] Command: {' '.join(command)}")

    # Set up environment with CPU thread limits
    # For local mode, we enforce a hard cap of 2 threads for thermal safety on laptops
    env = os.environ.copy()
    LOCAL_THREADS_CAP = int(os.environ.get("SCREENALYTICS_LOCAL_MAX_THREADS", "4"))
    default_local_threads = int(os.environ.get("SCREENALYTICS_LOCAL_DEFAULT_THREADS", "2"))
    local_max_threads = min(int(cpu_threads or default_local_threads), LOCAL_THREADS_CAP)
    env.update({
        "SCREENALYTICS_MAX_CPU_THREADS": str(local_max_threads),
        "OMP_NUM_THREADS": str(local_max_threads),
        "MKL_NUM_THREADS": str(local_max_threads),
        "OPENBLAS_NUM_THREADS": str(local_max_threads),
        "VECLIB_MAXIMUM_THREADS": str(local_max_threads),
        "NUMEXPR_NUM_THREADS": str(local_max_threads),
        "ORT_INTRA_OP_NUM_THREADS": str(local_max_threads),
        "ORT_INTER_OP_NUM_THREADS": "1",
        # Enable local mode instrumentation for verbose phase-level logging
        "LOCAL_MODE_INSTRUMENTATION": "1",
    })
    logs.append(f"Local mode: CPU threads capped at {local_max_threads} for thermal safety")
    LOGGER.info(f"[{episode_id}] Local mode CPU threads capped at {local_max_threads}")

    # Apply thermal limiting wrapper for thermal safety
    # On macOS: uses taskpolicy with QoS clamp (works with CoreML/Metal)
    # On Linux: uses cpulimit (SIGSTOP/SIGCONT based)
    effective_command, thermal_limit_applied = _maybe_wrap_with_cpulimit_local(command, profile)
    if thermal_limit_applied:
        if platform.system() == "Darwin":
            qos = "background" if profile == "low_power" else "utility"
            logs.append(f"Local mode: Using taskpolicy -c {qos} for thermal safety (macOS)")
        else:
            logs.append(f"Local mode: Using cpulimit at {_LOCAL_CPULIMIT_PERCENT}% for thermal safety")

    start_time = time.time()
    process: subprocess.Popen | None = None

    def _kill_process_tree(proc: subprocess.Popen) -> None:
        """Kill process and all its children using process group."""
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait(timeout=5)
        except (ProcessLookupError, OSError) as e:
            LOGGER.debug(f"Process already dead: {e}")
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass

    try:
        # Start subprocess in a new process group so we can kill all children
        # BUT NOT with start_new_session=True so it stays attached to this request
        process = subprocess.Popen(
            effective_command,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,  # Create new process group for clean termination
        )

        # Apply CPU affinity fallback if thermal limiting wasn't available
        if not thermal_limit_applied and _LOCAL_CPULIMIT_PERCENT > 0:
            if _apply_cpu_affinity_fallback_local(process.pid, _LOCAL_CPULIMIT_PERCENT):
                logs.append(f"Local mode: Using CPU affinity fallback (thermal limiter not available)")

        # Register the job for tracking/duplicate prevention
        _register_local_job(episode_id, operation, process.pid, job_id=None)
        logs.append(f"Process started (PID {process.pid})")

        # Wait for process to complete with timeout
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(process)
            _unregister_local_job(episode_id, operation)
            elapsed = time.time() - start_time
            logs.append(f"TIMEOUT: Job timed out after {timeout}s")
            LOGGER.error(f"[{episode_id}] local {operation} timed out after {timeout}s")
            return {
                "status": "error",
                "ep_id": episode_id,
                "operation": operation,
                "execution_mode": "local",
                "error": f"Job timed out after {timeout} seconds",
                "logs": logs,
                "elapsed_seconds": elapsed,
            }

        elapsed = time.time() - start_time
        _unregister_local_job(episode_id, operation)

        # Collect output
        if stdout:
            for line in stdout.strip().split("\n"):
                if line.strip():
                    logs.append(line.strip())

        if process.returncode != 0:
            error_msg = stderr.strip() if stderr else f"Exit code {process.returncode}"
            logs.append(f"ERROR: {error_msg}")
            LOGGER.error(f"[{episode_id}] local {operation} failed: {error_msg}")
            return {
                "status": "error",
                "ep_id": episode_id,
                "operation": operation,
                "execution_mode": "local",
                "error": error_msg,
                "return_code": process.returncode,
                "logs": logs,
                "elapsed_seconds": elapsed,
            }

        # Format elapsed time nicely
        if elapsed >= 60:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            elapsed_str = f"{mins}m {secs}s"
        else:
            elapsed_str = f"{elapsed:.1f}s"
        logs.append(f"[LOCAL MODE] {operation} completed successfully in {elapsed_str}")
        LOGGER.info(f"[{episode_id}] local {operation} completed successfully in {elapsed_str}")
        return {
            "status": "completed",
            "ep_id": episode_id,
            "operation": operation,
            "execution_mode": "local",
            "return_code": 0,
            "logs": logs,
            "elapsed_seconds": elapsed,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        if process and process.poll() is None:
            _kill_process_tree(process)
        _unregister_local_job(episode_id, operation)
        error_msg = str(e)
        logs.append(f"EXCEPTION: {error_msg}")
        LOGGER.exception(f"[{episode_id}] local {operation} raised exception: {e}")
        return {
            "status": "error",
            "ep_id": episode_id,
            "operation": operation,
            "execution_mode": "local",
            "error": error_msg,
            "logs": logs,
            "elapsed_seconds": elapsed,
        }


async def _run_local_subprocess_async(
    command: list[str],
    episode_id: str,
    operation: str,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a pipeline command as subprocess, tied to the HTTP request lifecycle.

    DEPRECATED: Use _run_local_subprocess_blocking() instead for synchronous local mode.

    This is used for local execution mode. The subprocess is killed if the
    HTTP request is cancelled (e.g., page refresh). This ensures local mode
    is truly synchronous and tied to the browser session.

    IMPORTANT: Uses process group to ensure all child processes are killed
    when the parent is terminated.

    Args:
        command: Command list to execute
        episode_id: Episode ID for logging
        operation: Operation name (detect_track, faces_embed, cluster)
        options: Job options for logging and env setup

    Returns:
        Result dict with status, logs, and summary data
    """
    import signal
    import time

    # Check for existing running job
    existing_job = _get_running_local_job(episode_id, operation)
    if existing_job:
        existing_pid = existing_job.get("pid")
        LOGGER.warning(f"[{episode_id}] {operation} already running (PID {existing_pid})")
        return {
            "status": "error",
            "ep_id": episode_id,
            "operation": operation,
            "execution_mode": "local",
            "error": f"A {operation} job is already running for this episode (PID {existing_pid}). "
                     "Wait for it to complete or kill it manually.",
            "logs": [f"Existing job PID: {existing_pid}"],
            "elapsed_seconds": 0,
        }

    project_root = _find_project_root()
    logs: List[str] = []

    # Build context string for logging
    device = options.get("device", "auto")
    stride = options.get("stride", 6)
    profile = options.get("profile", "default")

    logs.append(f"[LOCAL MODE] Starting {operation}")
    logs.append(f"  Device: {device}, Stride: {stride}")
    logs.append(f"  This runs synchronously - page refresh will cancel the job.")
    LOGGER.info(f"[{episode_id}] Starting local {operation} (device={device}, stride={stride})")
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
        logs.append(f"CPU threads limited to {threads}")
    else:
        # CRITICAL: Always set CPU limits to prevent thermal issues
        default_threads = 2
        env.update({
            "SCREENALYTICS_MAX_CPU_THREADS": str(default_threads),
            "OMP_NUM_THREADS": str(default_threads),
            "MKL_NUM_THREADS": str(default_threads),
            "OPENBLAS_NUM_THREADS": str(default_threads),
            "VECLIB_MAXIMUM_THREADS": str(default_threads),
            "NUMEXPR_NUM_THREADS": str(default_threads),
            "ORT_INTRA_OP_NUM_THREADS": str(default_threads),
            "ORT_INTER_OP_NUM_THREADS": "1",
        })
        logs.append(f"CPU threads limited to {default_threads} (default)")

    start_time = time.time()
    process: subprocess.Popen | None = None

    def _kill_process_tree(proc: subprocess.Popen) -> None:
        """Kill process and all its children using process group."""
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait(timeout=5)
        except (ProcessLookupError, OSError) as e:
            LOGGER.debug(f"Process already dead: {e}")
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass

    try:
        # Use Popen with start_new_session=True to create a new process group
        # This allows us to kill all child processes when cancelling
        process = subprocess.Popen(
            command,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,  # Create new process group
        )

        # Register the job
        _register_local_job(episode_id, operation, process.pid)
        logs.append(f"Process started (PID {process.pid})")

        # Wait for process in a way that allows cancellation
        # Poll every 0.5 seconds to check if we should cancel
        timeout_seconds = 3600  # 1 hour
        elapsed = 0.0
        while process.poll() is None:
            await asyncio.sleep(0.5)
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                _kill_process_tree(process)
                _unregister_local_job(episode_id, operation)
                logs.append(f"TIMEOUT: Job timed out after 1 hour")
                LOGGER.error(f"[{episode_id}] local {operation} timed out after 1 hour")
                return {
                    "status": "error",
                    "ep_id": episode_id,
                    "operation": operation,
                    "execution_mode": "local",
                    "error": "Job timed out after 1 hour",
                    "logs": logs,
                    "elapsed_seconds": elapsed,
                }

        elapsed = time.time() - start_time
        _unregister_local_job(episode_id, operation)

        # Collect output
        stdout, stderr = process.communicate(timeout=10)
        if stdout:
            for line in stdout.strip().split("\n"):
                if line.strip():
                    logs.append(line.strip())

        if process.returncode != 0:
            error_msg = stderr.strip() if stderr else f"Exit code {process.returncode}"
            logs.append(f"ERROR: {error_msg}")
            LOGGER.error(f"[{episode_id}] local {operation} failed: {error_msg}")
            return {
                "status": "error",
                "ep_id": episode_id,
                "operation": operation,
                "execution_mode": "local",
                "error": error_msg,
                "return_code": process.returncode,
                "logs": logs,
                "elapsed_seconds": elapsed,
            }

        # Format elapsed time nicely
        if elapsed >= 60:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            elapsed_str = f"{mins}m {secs}s"
        else:
            elapsed_str = f"{elapsed:.1f}s"
        logs.append(f"[LOCAL MODE] {operation} completed successfully in {elapsed_str}")
        LOGGER.info(f"[{episode_id}] local {operation} completed successfully in {elapsed_str}")
        return {
            "status": "completed",
            "ep_id": episode_id,
            "operation": operation,
            "execution_mode": "local",
            "return_code": 0,
            "logs": logs,
            "elapsed_seconds": elapsed,
        }

    except asyncio.CancelledError:
        # Request was cancelled (e.g., page refresh) - kill the subprocess tree
        elapsed = time.time() - start_time
        if process and process.poll() is None:
            LOGGER.warning(f"[{episode_id}] Request cancelled, killing {operation} process tree (PID {process.pid})")
            _kill_process_tree(process)
            logs.append(f"CANCELLED: Request cancelled after {elapsed:.1f}s, process tree killed")
        _unregister_local_job(episode_id, operation)
        raise  # Re-raise to propagate cancellation

    except Exception as e:
        elapsed = time.time() - start_time
        if process and process.poll() is None:
            _kill_process_tree(process)
        _unregister_local_job(episode_id, operation)
        error_msg = str(e)
        logs.append(f"EXCEPTION: {error_msg}")
        LOGGER.exception(f"[{episode_id}] local {operation} raised exception: {e}")
        return {
            "status": "error",
            "ep_id": episode_id,
            "operation": operation,
            "execution_mode": "local",
            "error": error_msg,
            "logs": logs,
            "elapsed_seconds": elapsed,
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
    if options.get("allow_cpu_fallback"):
        command.append("--allow-cpu-fallback")

    # Suppress noisy ONNX/model warnings by default for cleaner logs
    if not options.get("verbose"):
        command.append("--quiet")

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
    # Use --sample-every-n-frames instead of deprecated --min-frames-between-crops
    if options.get("min_frames_between_crops"):
        command += ["--sample-every-n-frames", str(options["min_frames_between_crops"])]
    if options.get("thumb_size"):
        command += ["--thumb-size", str(options["thumb_size"])]
    if options.get("allow_cpu_fallback"):
        command.append("--allow-cpu-fallback")

    # Suppress noisy ONNX/model warnings by default for cleaner logs
    if not options.get("verbose"):
        command.append("--quiet")

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
    if options.get("allow_cpu_fallback"):
        command.append("--allow-cpu-fallback")

    # Suppress noisy ONNX/model warnings by default for cleaner logs
    if not options.get("verbose"):
        command.append("--quiet")

    return command


# =============================================================================
# Streaming Local Subprocess - Live log output for Local mode
# =============================================================================


def _stream_local_subprocess(
    command: list[str],
    episode_id: str,
    operation: str,
    options: Dict[str, Any],
    timeout: int = 3600,
) -> Generator[str, None, None]:
    """Run a pipeline command and yield log lines as newline-delimited JSON.

    This generator streams log lines as they are produced by the subprocess,
    enabling live updates in the UI. Uses LogFormatter to clean up raw output.

    Each yielded line is a JSON object:
    - {"type": "log", "line": "...", "raw": "..."} for log lines
    - {"type": "summary", "status": "completed"|"error", ...} at the end

    Args:
        command: Command list to execute
        episode_id: Episode ID for logging
        operation: Operation name (detect_track, faces_embed, cluster)
        options: Job options for logging and env setup
        timeout: Maximum time in seconds (default 3600 = 1 hour)

    Yields:
        Newline-delimited JSON strings
    """
    # Check for existing running job (in-memory registry)
    existing_job = _get_running_local_job(episode_id, operation)
    if existing_job:
        existing_pid = existing_job.get("pid")
        yield json.dumps({
            "type": "error",
            "message": f"A {operation} job is already running for this episode (PID {existing_pid})"
        }) + "\n"
        return

    # Check file-based lock (survives API restart)
    lock_info = _check_job_lock(episode_id, operation)
    if lock_info:
        yield json.dumps({
            "type": "error",
            "message": f"A {operation} job may already be running (lock: PID {lock_info.get('pid')}, host {lock_info.get('hostname')})"
        }) + "\n"
        return

    # Acquire file-based lock
    lock_acquired, lock_error = _acquire_job_lock(episode_id, operation)
    if not lock_acquired:
        yield json.dumps({
            "type": "error",
            "message": f"Could not start {operation}: {lock_error}"
        }) + "\n"
        return

    project_root = _find_project_root()

    # Extract config values
    device = options.get("device", "auto")
    stride = options.get("stride", 6)
    profile = options.get("profile", "low_power")
    cpu_threads = options.get("cpu_threads", 2)

    # Initialize the log formatter
    formatter = LogFormatter(episode_id, operation)

    # Track both formatted and raw lines
    formatted_logs: List[str] = []
    raw_logs: List[str] = []

    def _emit_formatted(line: str, raw_line: str | None = None) -> str:
        """Helper to emit a formatted log line and track it."""
        formatted_logs.append(line)
        if raw_line is not None:
            raw_logs.append(raw_line)
        return json.dumps({"type": "log", "line": line}) + "\n"

    def _emit_raw_only(raw_line: str) -> None:
        """Track a raw line that was suppressed from formatted output."""
        raw_logs.append(raw_line)

    # Set up environment with CPU thread limits
    env = os.environ.copy()
    LOCAL_THREADS_CAP = int(os.environ.get("SCREENALYTICS_LOCAL_MAX_THREADS", "4"))
    default_local_threads = int(os.environ.get("SCREENALYTICS_LOCAL_DEFAULT_THREADS", "2"))
    local_max_threads = min(int(cpu_threads or default_local_threads), LOCAL_THREADS_CAP)
    env.update({
        "SCREENALYTICS_MAX_CPU_THREADS": str(local_max_threads),
        "OMP_NUM_THREADS": str(local_max_threads),
        "MKL_NUM_THREADS": str(local_max_threads),
        "OPENBLAS_NUM_THREADS": str(local_max_threads),
        "VECLIB_MAXIMUM_THREADS": str(local_max_threads),
        "NUMEXPR_NUM_THREADS": str(local_max_threads),
        "ORT_INTRA_OP_NUM_THREADS": str(local_max_threads),
        "ORT_INTER_OP_NUM_THREADS": "1",
        "LOCAL_MODE_INSTRUMENTATION": "1",
        # Force unbuffered Python output for real-time streaming
        "PYTHONUNBUFFERED": "1",
    })

    # Apply thermal limiting wrapper for thermal safety
    # On macOS: uses taskpolicy with QoS clamp (works with CoreML/Metal)
    # On Linux: uses cpulimit (SIGSTOP/SIGCONT based)
    effective_command, thermal_limit_applied = _maybe_wrap_with_cpulimit_local(command, profile)
    thermal_limit_info = None
    if thermal_limit_applied:
        if platform.system() == "Darwin":
            qos = "background" if profile == "low_power" else "utility"
            thermal_limit_info = f"taskpolicy -c {qos}"
        else:
            thermal_limit_info = f"cpulimit {_LOCAL_CPULIMIT_PERCENT}%"

    # Try to get video metadata for frame count
    total_frames = None
    fps = None
    try:
        from py_screenalytics.artifacts import get_path
        video_path = get_path(episode_id, "video")
        if video_path.exists():
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or None
                cap.release()
    except Exception:
        pass  # Video metadata is optional

    # Emit canonical config block
    config_block = format_config_block(
        operation=operation,
        episode_id=episode_id,
        device=device,
        profile=profile,
        cpu_threads=local_max_threads,
        stride=stride if operation == "detect_track" else None,
        total_frames=total_frames,
        fps=fps,
        thermal_limit_info=thermal_limit_info,
    )
    for line in config_block.split("\n"):
        yield _emit_formatted(line, line)

    LOGGER.info(f"[{episode_id}] Starting streaming local {operation}")
    LOGGER.info(f"[{episode_id}] Command: {' '.join(command)}")

    start_time = time.time()
    process: subprocess.Popen | None = None

    def _kill_process_tree(proc: subprocess.Popen) -> None:
        """Kill process and all its children using process group."""
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait(timeout=5)
        except (ProcessLookupError, OSError) as e:
            LOGGER.debug(f"Process already dead: {e}")
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass

    def _save_logs_always(status: str, elapsed: float, extra: Dict[str, Any] | None = None) -> None:
        """Always persist logs, even on crash/cancel. Called from finally block."""
        # Add formatter finalization message if any
        final_msg = formatter.finalize()
        if final_msg:
            for line in final_msg.split("\n"):
                formatted_logs.append(line)

        save_operation_logs(
            episode_id,
            operation,
            formatted_logs,
            status,
            elapsed,
            raw_logs=raw_logs,
            extra=extra,
        )

    def _clean_noise(text: str) -> str:
        if not text:
            return text
        if "call apply_model on this" in text.lower():
            return "Demucs failed to load the MDX model (library noise suppressed)"
        return text

    try:
        # Start subprocess with line-buffered output for streaming
        process = subprocess.Popen(
            effective_command,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line-buffered
            env=env,
            start_new_session=True,
        )

        # Apply CPU affinity fallback if thermal limiting wasn't available
        if not thermal_limit_applied and _LOCAL_CPULIMIT_PERCENT > 0:
            if _apply_cpu_affinity_fallback_local(process.pid, _LOCAL_CPULIMIT_PERCENT):
                affinity_msg = "[INFO] Using CPU affinity fallback (thermal limiter not available)"
                yield _emit_formatted(affinity_msg, affinity_msg)

        # Register the job
        _register_local_job(episode_id, operation, process.pid, job_id=None)
        pid_msg = f"[INFO] Process started (PID {process.pid})"
        yield _emit_formatted(pid_msg, pid_msg)

        # Stream stdout lines as they arrive, applying formatting
        assert process.stdout is not None
        for raw_line in iter(process.stdout.readline, ""):
            if not raw_line:
                break

            # Check for timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                _kill_process_tree(process)
                _unregister_local_job(episode_id, operation)
                timeout_msg = f"[ERROR] Job timed out after {timeout}s"
                yield _emit_formatted(timeout_msg, raw_line)

                # Save logs in finally block
                _save_logs_always("timeout", elapsed)

                yield json.dumps({
                    "type": "summary",
                    "status": "timeout",
                    "elapsed_seconds": elapsed,
                    "error": f"Job timed out after {timeout} seconds",
                }) + "\n"
                return

            stripped = raw_line.rstrip()
            if not stripped:
                continue

            # Suppress demucs BagOfModels noise that leaks through demucs stdout
            if "call apply_model on this" in stripped.lower():
                continue

            # Check if this is a JSON progress line - send as separate message type
            if stripped.startswith('{') and '"phase"' in stripped:
                try:
                    progress_data = json.loads(stripped)
                    if isinstance(progress_data, dict) and "phase" in progress_data:
                        # Audio pipeline uses different progress format with more fields
                        if operation == "audio_pipeline":
                            # Pass through audio pipeline progress with all fields intact
                            yield json.dumps({
                                "type": "audio_progress",
                                "phase": progress_data.get("phase", ""),
                                "progress": progress_data.get("progress", 0),
                                "message": progress_data.get("message", ""),
                                "step_name": progress_data.get("step_name", ""),
                                "step_progress": progress_data.get("step_progress", 0),
                                "step_order": progress_data.get("step_order", 0),
                                "total_steps": progress_data.get("total_steps", 9),
                                "voice_clusters": progress_data.get("voice_clusters"),
                                "labeled_voices": progress_data.get("labeled_voices"),
                                "unlabeled_voices": progress_data.get("unlabeled_voices"),
                            }) + "\n"
                        else:
                            # Standard progress for detect_track, faces_embed, cluster
                            yield json.dumps({
                                "type": "progress",
                                "frames_done": progress_data.get("frames_done", 0),
                                "frames_total": progress_data.get("frames_total", 0),
                                "phase": progress_data.get("phase", ""),
                                "fps_infer": progress_data.get("fps_infer"),
                                "secs_done": progress_data.get("secs_done", 0),
                            }) + "\n"
                        # Also format and show in logs
                        formatted_line = formatter.format_line(stripped)
                        if formatted_line:
                            yield _emit_formatted(formatted_line, stripped)
                        else:
                            _emit_raw_only(stripped)
                        continue
                except json.JSONDecodeError:
                    pass  # Not valid JSON, process normally

            # Apply formatting
            formatted_line = formatter.format_line(stripped)
            if formatted_line:
                # Line was formatted and should be shown
                yield _emit_formatted(formatted_line, stripped)
            else:
                # Line was suppressed - still track raw
                _emit_raw_only(stripped)

        # Wait for process to complete
        process.wait()
        elapsed = time.time() - start_time
        _unregister_local_job(episode_id, operation)

        # Handle result
        if process.returncode != 0:
            error_msg = f"Process exited with code {process.returncode}"
            yield _emit_formatted(f"[ERROR] {error_msg}", f"ERROR: {error_msg}")
            LOGGER.error(f"[{episode_id}] local {operation} failed: {error_msg}")

            # Save logs
            _save_logs_always("error", elapsed, {"return_code": process.returncode})

            yield json.dumps({
                "type": "summary",
                "status": "error",
                "elapsed_seconds": elapsed,
                "return_code": process.returncode,
                "error": error_msg,
            }) + "\n"
            return

        # Success - emit completion summary
        success_msg = format_completion_summary(operation, "completed", elapsed)
        yield _emit_formatted(success_msg, success_msg)
        LOGGER.info(f"[{episode_id}] local {operation} completed successfully")

        # Save logs
        _save_logs_always("completed", elapsed)

        yield json.dumps({
            "type": "summary",
            "status": "completed",
            "elapsed_seconds": elapsed,
            "return_code": 0,
        }) + "\n"

    except GeneratorExit:
        # Client disconnected (page refresh, navigation, etc.) - kill the subprocess
        elapsed = time.time() - start_time
        if process and process.poll() is None:
            LOGGER.warning(f"[{episode_id}] Client disconnected, killing {operation} process tree (PID {process.pid})")
            _kill_process_tree(process)
            cancel_msg = f"[CANCELLED] Client disconnected after {elapsed:.1f}s, process killed"
            formatted_logs.append(cancel_msg)
            raw_logs.append(cancel_msg)
        _unregister_local_job(episode_id, operation)

        # ALWAYS save logs, even on cancel
        _save_logs_always("cancelled", elapsed)
        raise

    except Exception as e:
        elapsed = time.time() - start_time
        if process and process.poll() is None:
            _kill_process_tree(process)
        _unregister_local_job(episode_id, operation)

        error_msg = _clean_noise(str(e))
        exception_line = f"[EXCEPTION] {error_msg}"
        formatted_logs.append(exception_line)
        raw_logs.append(exception_line)
        yield json.dumps({"type": "log", "line": exception_line}) + "\n"
        LOGGER.exception(f"[{episode_id}] local {operation} raised exception: {e}")

        # ALWAYS save logs, even on exception
        _save_logs_always("error", elapsed, {"exception": error_msg})

        yield json.dumps({
            "type": "summary",
            "status": "error",
            "elapsed_seconds": elapsed,
            "error": error_msg,
        }) + "\n"


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
        "allow_cpu_fallback": req.allow_cpu_fallback,
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

    # Handle local execution mode - streaming subprocess with live log output
    if execution_mode == "local":
        # Add profile to options for logging
        options["profile"] = resolved_profile

        project_root = _find_project_root()
        command = _build_detect_track_command(req.ep_id, options, project_root)

        # Return streaming response for live log updates
        return StreamingResponse(
            _stream_local_subprocess(command, req.ep_id, "detect_track", options),
            media_type="application/x-ndjson",
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
            },
        )

    # Redis/Celery mode (default) - enqueues job and returns immediately
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
        "allow_cpu_fallback": req.allow_cpu_fallback,
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

    # Handle local execution mode - streaming subprocess with live log output
    if execution_mode == "local":
        # Add profile to options for logging
        options["profile"] = resolved_profile

        project_root = _find_project_root()
        command = _build_faces_embed_command(req.ep_id, options, project_root)

        # Return streaming response for live log updates
        return StreamingResponse(
            _stream_local_subprocess(command, req.ep_id, "faces_embed", options),
            media_type="application/x-ndjson",
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
            },
        )

    # Redis/Celery mode (default) - enqueues job and returns immediately
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
        "allow_cpu_fallback": req.allow_cpu_fallback,
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

    # Handle local execution mode - streaming subprocess with live log output
    if execution_mode == "local":
        # Add profile to options for logging
        options["profile"] = resolved_profile

        project_root = _find_project_root()
        command = _build_cluster_command(req.ep_id, options, project_root)

        # Return streaming response for live log updates
        return StreamingResponse(
            _stream_local_subprocess(command, req.ep_id, "cluster", options),
            media_type="application/x-ndjson",
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
            },
        )

    # Redis/Celery mode (default) - enqueues job and returns immediately
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


# =============================================================================
# Enhancement #1: Parallel Job Execution Endpoints
# =============================================================================


class ParallelJobRequest(BaseModel):
    """Request model for running parallel jobs across multiple episodes."""
    episode_ids: List[str] = Field(..., description="List of episode IDs to process")
    operation: str = Field(..., description="Operation to run: auto_group, refresh_similarity, manual_assign")
    options: Optional[Dict[str, Any]] = Field(None, description="Options to pass to each task")


@router.post("/parallel")
async def start_parallel_jobs(req: ParallelJobRequest):
    """Start the same operation on multiple episodes in parallel.

    Uses Celery group to execute jobs concurrently across episodes.
    Returns a group_id that can be used to track overall progress.

    Args:
        episode_ids: List of episode IDs to process
        operation: Operation to run (auto_group, refresh_similarity, manual_assign)
        options: Options to pass to each task

    Returns:
        - group_id: ID to track the parallel job group
        - job_ids: Map of episode_id to individual job_id
        - status: "queued" or "error"
    """
    from apps.api.tasks import run_parallel_jobs

    result = run_parallel_jobs(
        episode_ids=req.episode_ids,
        operation=req.operation,
        options=req.options,
    )

    if result.get("status") == "error":
        return JSONResponse(
            status_code=400,
            content=result,
        )

    return result


@router.get("/parallel/{group_id}")
async def get_parallel_job_status(group_id: str):
    """Get status of a parallel job group.

    Returns overall progress and per-episode results.

    Args:
        group_id: The group ID returned from /parallel endpoint

    Returns:
        - status: overall status (in_progress, success, partial_success, failed)
        - progress: fraction complete (0.0 to 1.0)
        - completed: number of completed jobs
        - failed: number of failed jobs
        - results: per-episode job status
    """
    from apps.api.tasks import get_parallel_job_status as get_status

    return get_status(group_id)


# =============================================================================
# Enhancement #9: Job History Persistence Endpoints
# =============================================================================


@router.get("/history")
async def get_job_history_endpoint(
    user_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
):
    """Get job history for a user.

    Returns list of recent jobs with status and timing information.

    Args:
        user_id: User identifier (defaults to "anonymous")
        limit: Maximum number of records (default 20)
        offset: Number of records to skip (for pagination)

    Returns:
        List of job records, newest first
    """
    from apps.api.tasks import get_job_history

    records = get_job_history(user_id, limit, offset)
    return {
        "jobs": records,
        "count": len(records),
        "limit": limit,
        "offset": offset,
    }


@router.get("/active")
async def get_active_jobs_endpoint(user_id: Optional[str] = None):
    """Get all active (queued/in_progress) jobs for a user.

    This is useful for the "My Jobs" panel to show what's currently running.

    Args:
        user_id: User identifier (defaults to "anonymous")

    Returns:
        List of active job records with current status
    """
    from apps.api.tasks import get_active_jobs_for_user

    jobs = get_active_jobs_for_user(user_id)
    return {
        "jobs": jobs,
        "count": len(jobs),
    }


__all__ = ["router"]
