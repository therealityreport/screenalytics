from __future__ import annotations

import base64
import html
import json
import logging
import math
import mimetypes
import numbers
import os
import platform
import re
import signal
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import urlparse

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import streamlit as st

# `streamlit` is an optional dependency in some CI/test contexts (e.g. unit tests that
# load helpers to validate non-UI utilities). Keep imports defensive so importing this
# module doesn't require Streamlit to be fully installed.
try:  # pragma: no cover
    import streamlit.components.v1 as components
except Exception:  # pragma: no cover
    components = None  # type: ignore[assignment]

# Streamlit 1.44+ reorganized internal modules - use try/except for compatibility.
# In non-Streamlit test environments, fall back to minimal stubs.
try:
    # Streamlit 1.44+
    from streamlit.runtime.scriptrunner import (
        add_script_run_ctx,
        get_script_run_ctx,
        RerunData,
    )
except Exception:  # pragma: no cover
    try:  # pragma: no cover
        # Streamlit < 1.44
        from streamlit.runtime.scriptrunner.script_run_context import (
            add_script_run_ctx,
            get_script_run_ctx,
        )
        from streamlit.runtime.scriptrunner.script_requests import RerunData
    except Exception:  # pragma: no cover
        def get_script_run_ctx():  # type: ignore[no-redef]
            return None

        def add_script_run_ctx(*_args, **_kwargs):  # type: ignore[no-redef]
            return None

        class RerunData:  # type: ignore[no-redef]
            def __init__(self, *args, **kwargs) -> None:
                self.query_string = kwargs.get("query_string")
                self.page_script_hash = kwargs.get("page_script_hash")

DEFAULT_TITLE = "SCREENALYTICS"
DATA_ROOT = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
DEFAULT_STRIDE = 6  # Every 6th frame - balances accuracy vs false positives
DEFAULT_DETECTOR = "retinaface"
DEFAULT_TRACKER = "bytetrack"

_TZ_EST = ZoneInfo("America/New_York")

# Platform-aware device defaults
# On macOS Apple Silicon: prefer CoreML for quiet, thermal-safe operation
# On other platforms: use "auto" which will pick CUDA > CPU
_IS_MACOS_APPLE_SILICON = (
    platform.system().lower() == "darwin" and
    platform.machine().lower().startswith(("arm", "aarch64"))
)
DEFAULT_DEVICE = "coreml" if _IS_MACOS_APPLE_SILICON else "auto"
DEFAULT_DEVICE_LABEL = "CoreML" if _IS_MACOS_APPLE_SILICON else "Auto"

DEFAULT_DET_THRESH = 0.65  # Raised from 0.5 to reduce false positive face detections
DEFAULT_MAX_GAP = 60
DEFAULT_CLUSTER_SIMILARITY = float(os.environ.get("SCREENALYTICS_CLUSTER_SIM", "0.58"))
_LOCAL_MEDIA_CACHE_SIZE = 256

LOGGER = logging.getLogger(__name__)
DIAG = os.getenv("DIAG_LOG", "0") == "1"


def _diag(tag: str, **kw) -> None:
    """Diagnostic logger enabled via DIAG_LOG=1."""
    if DIAG:
        LOGGER.info("[DIAG:%s] %s", tag, json.dumps(kw, ensure_ascii=False))


def _now_iso() -> str:
    """UTC timestamp in ISO-8601 with 'Z' suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


TRACK_HIGH_THRESH_DEFAULT = _env_float(
    "SCREENALYTICS_TRACK_HIGH_THRESH",
    _env_float("BYTE_TRACK_HIGH_THRESH", 0.55),  # Raised from 0.45 - stricter track continuation
)
TRACK_NEW_THRESH_DEFAULT = _env_float(
    "SCREENALYTICS_NEW_TRACK_THRESH",
    _env_float("BYTE_TRACK_NEW_TRACK_THRESH", 0.75),  # Raised from 0.60 - much stricter new track creation
)
TRACK_BUFFER_BASE_DEFAULT = max(
    _env_int("SCREENALYTICS_TRACK_BUFFER", _env_int("BYTE_TRACK_BUFFER", 30)),
    1,
)
MIN_BOX_AREA_DEFAULT = max(
    _env_float("SCREENALYTICS_MIN_BOX_AREA", _env_float("BYTE_TRACK_MIN_BOX_AREA", 100.0)),  # Raised from 20 - filter tiny false detections
    0.0,
)


# Thumbnail constants
THUMB_W, THUMB_H = 200, 250
# Use absolute path for placeholder to work regardless of working directory
_PLACEHOLDER = str(Path(__file__).parent / "assets" / "placeholder_face.svg")
_THUMB_CACHE_STATE_KEY = "_thumb_async_cache"
_THUMB_JOB_STATE_KEY = "_thumb_async_jobs"
_MAX_ASYNC_THUMB_WORKERS = 8

LABEL = {
    DEFAULT_DETECTOR: "RetinaFace (recommended)",
    DEFAULT_TRACKER: "ByteTrack (default)",
    "strongsort": "StrongSORT (ReID)",
}
DEVICE_LABELS = ["Auto", "CPU", "MPS", "CoreML", "CUDA"]
DEVICE_VALUE_MAP = {"Auto": "auto", "CPU": "cpu", "MPS": "mps", "CoreML": "coreml", "CUDA": "cuda"}
DEVICE_VALUE_TO_LABEL = {value.lower(): label for label, value in DEVICE_VALUE_MAP.items()}
DEVICE_VALUE_TO_LABEL["metal"] = "CoreML"
DEVICE_VALUE_TO_LABEL["apple"] = "CoreML"
DEVICE_VALUE_TO_LABEL.setdefault("coreml", "CoreML")
DEVICE_VALUE_TO_LABEL.setdefault("metal", "CoreML")
DEVICE_VALUE_TO_LABEL.setdefault("apple", "CoreML")
DETECTOR_OPTIONS = [
    ("RetinaFace (recommended)", DEFAULT_DETECTOR),
]
DETECTOR_LABELS = [label for label, _ in DETECTOR_OPTIONS]
DETECTOR_VALUE_MAP = {label: value for label, value in DETECTOR_OPTIONS}
DETECTOR_LABEL_MAP = {value: label for label, value in DETECTOR_OPTIONS}
FACE_ONLY_DETECTORS = {"retinaface"}
TRACKER_OPTIONS = [
    ("ByteTrack (default)", DEFAULT_TRACKER),
    ("StrongSORT (ReID)", "strongsort"),
]
TRACKER_LABELS = [label for label, _ in TRACKER_OPTIONS]
TRACKER_VALUE_MAP = {label: value for label, value in TRACKER_OPTIONS}
TRACKER_LABEL_MAP = {value: label for label, value in TRACKER_OPTIONS}

# Performance profile settings
# Higher strides = fewer frames = fewer false detections
PROFILE_LABELS = ["Low Power", "Balanced", "Performance"]
PROFILE_VALUE_MAP = {"Low Power": "low_power", "Balanced": "balanced", "Performance": "performance"}
PROFILE_LABEL_MAP = {v: k for k, v in PROFILE_VALUE_MAP.items()}
PROFILE_DEFAULTS = {
    "low_power": {
        "stride": 12,  # Every 12th frame - fast, fewer false positives
        "fps": 15.0,
        "save_frames": False,
        "save_crops": True,
        "cpu_threads": 2,
    },
    "balanced": {
        "stride": 6,  # Every 6th frame - good balance
        "fps": 24.0,
        "save_frames": False,
        "save_crops": True,
        "cpu_threads": 4,
    },
    "performance": {
        "stride": 4,  # Every 4th frame - thorough
        "fps": 30.0,
        "save_frames": False,
        "save_crops": True,
        "cpu_threads": 8,
    },
}
# Default profile per device type
# On macOS Apple Silicon: low_power profile for thermal safety
# On other platforms: balanced for better throughput
DEVICE_DEFAULT_PROFILE = {
    "coreml": "low_power",  # CoreML: use higher stride to reduce false detections
    "mps": "low_power",
    "cpu": "low_power",
    "cuda": "balanced",  # CUDA can handle more frames
    "auto": "low_power" if _IS_MACOS_APPLE_SILICON else "balanced",
}

SUPPORTED_PIPELINE_COMBOS = {
    "harvest": {(DEFAULT_DETECTOR, DEFAULT_TRACKER)},
    "cluster": {(DEFAULT_DETECTOR, DEFAULT_TRACKER)},
}
SCENE_DETECTOR_OPTIONS = [
    ("PySceneDetect (recommended)", "pyscenedetect"),
    ("HSV histogram (fallback)", "internal"),
    ("Disabled", "off"),
]
SCENE_DETECTOR_LABELS = [label for label, _ in SCENE_DETECTOR_OPTIONS]
SCENE_DETECTOR_VALUE_MAP = {label: value for label, value in SCENE_DETECTOR_OPTIONS}
SCENE_DETECTOR_LABEL_MAP = {value: label for label, value in SCENE_DETECTOR_OPTIONS}
# Default to pyscenedetect (recommended for reality TV with many cuts)
_SCENE_DETECTOR_ENV = os.environ.get("SCENE_DETECTOR", "pyscenedetect").strip().lower()
SCENE_DETECTOR_DEFAULT = _SCENE_DETECTOR_ENV if _SCENE_DETECTOR_ENV in SCENE_DETECTOR_LABEL_MAP else "pyscenedetect"
_EP_ID_REGEX = re.compile(r"^(?P<show>.+)-s(?P<season>\d{2})e(?P<episode>\d{2})$", re.IGNORECASE)
_CUSTOM_SHOWS_SESSION_KEY = "_custom_show_registry"

# =============================================================================
# Execution Mode (Local vs Redis/Celery)
# =============================================================================
# This controls whether jobs run synchronously in-process ("local") or are
# queued via Redis/Celery ("redis"). The mode is stored per-episode in session
# state and persisted to localStorage for the browser.
#
# IMPORTANT: We prefer Redis when available because:
# - Background processing doesn't block the UI
# - Better progress tracking and cancellation
# - Worker concurrency is properly managed
#
# Local mode is a fallback for laptops without Redis, with explicit warnings
# about thermal impact for long/heavy episodes.

EXECUTION_MODE_SESSION_KEY = "_execution_mode"
EXECUTION_MODE_OPTIONS = [
    ("Redis/Celery (queued)", "redis"),
    ("Local (CPU/CoreML on this machine)", "local"),
]
EXECUTION_MODE_LABELS = [label for label, _ in EXECUTION_MODE_OPTIONS]
EXECUTION_MODE_VALUE_MAP = {label: value for label, value in EXECUTION_MODE_OPTIONS}
EXECUTION_MODE_LABEL_MAP = {value: label for label, value in EXECUTION_MODE_OPTIONS}

# Cache for Redis availability check
_REDIS_AVAILABLE_CACHE: Dict[str, Any] = {"checked": False, "available": False, "last_check": 0}
_REDIS_CHECK_INTERVAL_SEC = 60  # Re-check every 60 seconds


def _check_redis_available() -> bool:
    """Check if Redis/Celery is available and healthy.

    Returns True if we can connect to the Celery workers via the API.
    Caches result to avoid repeated checks.
    """
    import time
    now = time.time()

    # Use cached result if recent
    if _REDIS_AVAILABLE_CACHE["checked"] and (now - _REDIS_AVAILABLE_CACHE["last_check"]) < _REDIS_CHECK_INTERVAL_SEC:
        return _REDIS_AVAILABLE_CACHE["available"]

    try:
        base = st.session_state.get("api_base")
        if not base:
            base = os.environ.get("API_BASE", "http://127.0.0.1:8000")
        resp = requests.get(f"{base}/celery_jobs", timeout=3)
        resp.raise_for_status()
        # If we got a response, Redis is available
        _REDIS_AVAILABLE_CACHE["available"] = True
    except requests.RequestException:
        _REDIS_AVAILABLE_CACHE["available"] = False

    _REDIS_AVAILABLE_CACHE["checked"] = True
    _REDIS_AVAILABLE_CACHE["last_check"] = now
    return _REDIS_AVAILABLE_CACHE["available"]


def _get_execution_mode_default() -> str:
    """Determine the default execution mode.

    - If Redis is available and healthy: default to 'redis'
    - Otherwise: default to 'local' with warnings
    """
    if _check_redis_available():
        return "redis"
    return "local"


def _execution_mode_key(ep_id: str) -> str:
    """Return session state key for execution mode per episode."""
    return f"{EXECUTION_MODE_SESSION_KEY}::{ep_id}"


def get_execution_mode(ep_id: str) -> str:
    """Get the current execution mode for an episode.

    If no mode is explicitly set, returns:
    - "redis" if Redis is available
    - "local" otherwise
    """
    if not ep_id:
        return _get_execution_mode_default()
    key = _execution_mode_key(ep_id)
    mode = st.session_state.get(key)
    if mode in ("redis", "local"):
        return mode
    return _get_execution_mode_default()


def set_execution_mode(ep_id: str, mode: str) -> None:
    """Set the execution mode for an episode.

    Args:
        ep_id: Episode identifier
        mode: "redis" or "local"
    """
    if not ep_id:
        return
    if mode not in ("redis", "local"):
        mode = _get_execution_mode_default()
    key = _execution_mode_key(ep_id)
    st.session_state[key] = mode


def execution_mode_label(mode: str) -> str:
    """Convert execution mode value to display label."""
    return EXECUTION_MODE_LABEL_MAP.get(mode, EXECUTION_MODE_LABELS[0])


def execution_mode_index(mode: str) -> int:
    """Get the index of an execution mode in the options list."""
    try:
        values = [v for _, v in EXECUTION_MODE_OPTIONS]
        return values.index(mode)
    except ValueError:
        return 0


def is_redis_available() -> bool:
    """Check if Redis/Celery backend is available."""
    return _check_redis_available()


def render_execution_mode_selector(
    ep_id: str,
    key_suffix: str = "",
    *,
    frame_count: int | None = None,
    duration_seconds: float | None = None,
) -> str:
    """Render an execution mode dropdown and return the current mode.

    This component allows switching between local (direct) and redis (queued)
    execution modes. The selection is stored in session state per episode.

    Args:
        ep_id: Episode identifier
        key_suffix: Optional suffix for the selectbox key to avoid conflicts
        frame_count: Optional total frame count for heavy episode warnings
        duration_seconds: Optional video duration for heavy episode warnings

    Returns:
        Current execution mode ("redis" or "local")
    """
    if not ep_id:
        return _get_execution_mode_default()

    redis_available = is_redis_available()
    current_mode = get_execution_mode(ep_id)
    current_index = execution_mode_index(current_mode)

    widget_key = f"execution_mode_selector::{ep_id}::{key_suffix}"

    # Customize help text based on Redis availability
    if redis_available:
        help_text = "Redis/Celery queues jobs for background workers. Local runs on this machine synchronously."
    else:
        help_text = (
            "⚠️ Redis/Celery not available. "
            "Local runs on this machine (may be slow/hot for long episodes). "
            "Start Redis to enable background processing."
        )

    selected_label = st.selectbox(
        "Execution Mode",
        EXECUTION_MODE_LABELS,
        index=current_index,
        key=widget_key,
        help=help_text,
    )

    selected_mode = EXECUTION_MODE_VALUE_MAP.get(selected_label, _get_execution_mode_default())

    # Update state if changed
    if selected_mode != current_mode:
        set_execution_mode(ep_id, selected_mode)

    # Show warning for Local mode on heavy episodes
    if selected_mode == "local":
        is_heavy = False
        warning_parts = []

        if frame_count and frame_count > 5000:
            is_heavy = True
            warning_parts.append(f"{frame_count:,} frames")

        if duration_seconds and duration_seconds > 600:  # > 10 minutes
            is_heavy = True
            mins = int(duration_seconds // 60)
            warning_parts.append(f"{mins}+ min video")

        if is_heavy:
            warning_msg = f"⚠️ Local mode on heavy episode ({', '.join(warning_parts)}) may cause thermal throttling."
            if redis_available:
                warning_msg += " Consider using Redis/Celery for background processing."
            else:
                warning_msg += " Consider using low_power profile or high stride."
            st.warning(warning_msg)

    return selected_mode


def is_local_mode(ep_id: str) -> bool:
    """Check if the current execution mode is 'local' for an episode."""
    return get_execution_mode(ep_id) == "local"


def is_redis_mode(ep_id: str) -> bool:
    """Check if the current execution mode is 'redis' for an episode."""
    return get_execution_mode(ep_id) == "redis"


# =============================================================================
# Local Mode Log Persistence and Hydration
# =============================================================================
# These helpers fetch persisted logs from the backend and cache them in session
# state so they can be displayed on page load without re-running jobs.

_LOCAL_LOGS_SESSION_KEY = "_local_logs_cache"
_REMOTE_LOG_OPERATIONS = {"detect_track", "faces_embed", "cluster"}
_ALL_LOG_OPERATIONS = _REMOTE_LOG_OPERATIONS | {"audio_pipeline"}


def _get_logs_cache() -> Dict[str, Dict[str, Any]]:
    """Get the logs cache dict from session state."""
    return st.session_state.setdefault(_LOCAL_LOGS_SESSION_KEY, {})


def _logs_cache_key(ep_id: str, operation: str) -> str:
    """Generate a cache key for an episode/operation."""
    return f"{ep_id}::{operation}"


def fetch_operation_logs(ep_id: str, operation: str) -> Dict[str, Any] | None:
    """Fetch persisted logs for an operation from the backend API.

    Args:
        ep_id: Episode identifier
        operation: One of 'detect_track', 'faces_embed', 'cluster'

    Returns:
        Dict with logs, status, elapsed_seconds, etc. or None if not available
    """
    if operation not in _REMOTE_LOG_OPERATIONS:
        LOGGER.debug("Skip remote log fetch for unsupported operation: %s", operation)
        return None

    try:
        base = st.session_state.get("api_base")
        if not base:
            base = os.environ.get("API_BASE", "http://127.0.0.1:8000")
        url = f"{base}/celery_jobs/logs/{ep_id}/{operation}?include_history=true&limit=25"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "none":
            return None
        return data
    except requests.RequestException as exc:
        LOGGER.debug(f"Failed to fetch logs for {ep_id}/{operation}: {exc}")
        return None


def get_cached_logs(ep_id: str, operation: str) -> Dict[str, Any] | None:
    """Get cached logs from session state for an episode/operation.

    Returns:
        Dict with logs, status, elapsed_seconds, etc. or None if not cached
    """
    cache = _get_logs_cache()
    key = _logs_cache_key(ep_id, operation)
    return cache.get(key)


def cache_logs(
    ep_id: str,
    operation: str,
    logs: List[str],
    status: str,
    elapsed_seconds: float,
    raw_logs: List[str] | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    """Cache logs in session state for an episode/operation.

    Args:
        ep_id: Episode identifier
        operation: Operation name
        logs: List of formatted log lines
        status: Final status (completed, error, cancelled, timeout)
        elapsed_seconds: Total runtime in seconds
        raw_logs: Optional list of raw unprocessed log lines
        extra: Additional metadata to store
    """
    cache = _get_logs_cache()
    key = _logs_cache_key(ep_id, operation)
    cache[key] = {
        "episode_id": ep_id,
        "operation": operation,
        "logs": logs,
        "raw_logs": raw_logs or logs,
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        **(extra or {}),
    }


def _audio_history_path(ep_id: str) -> Path:
    """Return path for persisted audio run history JSONL."""
    manifests = DATA_ROOT / "manifests" / ep_id / "logs"
    return manifests / "audio_runs.jsonl"


def append_audio_run_history(
    ep_id: str,
    operation: str,
    status: str,
    start_ts: float,
    end_ts: float,
    log_lines: List[str],
) -> None:
    """Persist a single audio run entry locally for UI history."""
    try:
        path = _audio_history_path(ep_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        duration = max(0.0, end_ts - start_ts)
        record = {
            "ep_id": ep_id,
            "operation": operation,
            "status": status,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_seconds": duration,
            "log_excerpt": log_lines[-200:],
            "created_at": _now_iso(),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:  # best-effort; never block UI
        LOGGER.debug("Failed to append audio history for %s/%s: %s", ep_id, operation, exc)


def load_audio_run_history(ep_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Load recent audio runs for display."""
    path = _audio_history_path(ep_id)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        # Newest entries are appended last
        records = list(reversed(records))
        return records[:limit]
    except Exception as exc:
        LOGGER.debug("Failed to load audio history for %s: %s", ep_id, exc)
        return []


def format_est(ts: float | str | None) -> str:
    """Format timestamp in America/New_York for UI."""
    if ts is None:
        return ""
    try:
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.astimezone(_TZ_EST).strftime("%Y-%m-%d %I:%M %p ET")
    except Exception:
        return ""


def hydrate_logs_for_episode(ep_id: str, force: bool = False) -> Dict[str, Dict[str, Any]]:
    """Fetch and cache persisted logs for all operations for an episode.

    This should be called on page load to populate the log cache with
    any previously saved logs. Only fetches if not already cached unless
    force=True.

    Args:
        ep_id: Episode identifier
        force: If True, always fetch from backend even if cached

    Returns:
        Dict mapping operation name to log data
    """
    result: Dict[str, Dict[str, Any]] = {}
    cache = _get_logs_cache()

    for operation in _REMOTE_LOG_OPERATIONS:
        key = _logs_cache_key(ep_id, operation)

        # Skip if already cached and not forcing refresh
        if not force and key in cache:
            if cache[key]:
                result[operation] = cache[key]
            continue

        # Fetch from backend
        data = fetch_operation_logs(ep_id, operation)
        if data:
            cache[key] = data
            result[operation] = data
        else:
            # Mark as fetched but empty so we don't keep retrying
            cache[key] = {}

    return result


def render_cached_logs(ep_id: str, operation: str) -> bool:
    """Render cached logs for an operation if available.

    Should be called to display "Last run" logs on page load.

    Args:
        ep_id: Episode identifier
        operation: Operation name

    Returns:
        True if logs were rendered, False if no logs available
    """
    cached = get_cached_logs(ep_id, operation)
    if not cached or not cached.get("logs"):
        return False

    logs = cached.get("logs", [])
    status = cached.get("status", "unknown")
    elapsed = cached.get("elapsed_seconds", 0)
    updated_at = cached.get("updated_at")

    # Format elapsed time
    if elapsed >= 60:
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        elapsed_str = f"{elapsed_min}m {elapsed_sec}s"
    else:
        elapsed_str = f"{elapsed:.1f}s"

    # Status indicator
    if status == "completed":
        status_icon = "completed"
        status_label = "Last run completed"
    elif status == "error":
        status_icon = "error"
        status_label = "Last run failed"
    elif status == "cancelled":
        status_icon = "cancelled"
        status_label = "Last run cancelled"
    elif status == "timeout":
        status_icon = "timeout"
        status_label = "Last run timed out"
    else:
        status_icon = "unknown"
        status_label = f"Last run: {status}"

    # Build header with timestamp if available
    header_parts = [status_label]
    if elapsed > 0:
        header_parts.append(f"({elapsed_str})")
    if updated_at:
        # Parse and format timestamp
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
            header_parts.append(f"at {formatted}")
        except Exception:
            pass

    # Render in an expander
    with st.expander(f"Last run log - {' '.join(header_parts)}", expanded=False):
        st.code("\n".join(logs), language="text")

    return True


def _restart_api_server() -> bool:
    """Kill any existing API server on port 8000 and start a new one.

    Returns True if restart was initiated successfully.
    """
    project_root = Path(__file__).parent.parent.parent  # workspace-ui -> apps -> project root

    # Kill any existing process on port 8000
    try:
        result = subprocess.run(
            ["lsof", "-ti", ":8000"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    LOGGER.info(f"Killed process {pid} on port 8000")
                except (ProcessLookupError, ValueError):
                    pass
            time.sleep(0.5)  # Give OS time to release the port
    except subprocess.TimeoutExpired:
        LOGGER.warning("Timeout checking for processes on port 8000")
    except Exception as e:
        LOGGER.warning(f"Error checking port 8000: {e}")

    # Start the API server in background
    api_script = project_root / "apps" / "api" / "main.py"
    if not api_script.exists():
        LOGGER.error(f"API script not found: {api_script}")
        return False

    try:
        # Start uvicorn in background
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.Popen(
            [
                str(project_root / ".venv" / "bin" / "python"),
                "-m", "uvicorn",
                "apps.api.main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ],
            cwd=str(project_root),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        LOGGER.info("API server restart initiated")
        return True
    except Exception as e:
        LOGGER.error(f"Failed to start API server: {e}")
        return False


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "off", "no"}:
        return False
    return default


def known_shows(include_session: bool = True) -> List[str]:
    """Return a sorted list of known show identifiers from episodes + S3 (plus session state).

    All show codes are normalized to UPPERCASE and deduplicated case-insensitively.
    This ensures 'rhoslc', 'RHOSLC', and 'RhoSlc' all resolve to a single 'RHOSLC' entry.
    """
    # Use uppercase keys for deduplication
    shows: set[str] = set()

    def _remember(show_value: Any) -> None:
        if not show_value or not isinstance(show_value, str):
            return
        # Normalize to uppercase for consistent display and deduplication
        cleaned = show_value.strip().upper()
        if cleaned:
            shows.add(cleaned)

    try:
        episodes_payload = api_get("/episodes")
    except requests.RequestException:
        episodes_payload = {}
    for record in episodes_payload.get("episodes", []):
        show_value = record.get("show_slug")
        if not show_value:
            parsed = parse_ep_id(record.get("ep_id", ""))
            show_value = parsed["show"] if parsed else None
        _remember(show_value)

    try:
        s3_payload = api_get("/episodes/s3_shows")
    except requests.RequestException:
        s3_payload = {}
    for entry in s3_payload.get("shows", []):
        _remember(entry.get("show"))

    try:
        registry_payload = api_get("/shows")
    except requests.RequestException:
        registry_payload = {}
    for entry in registry_payload.get("shows", []):
        _remember(entry.get("show_id"))

    if include_session:
        for entry in st.session_state.get(_CUSTOM_SHOWS_SESSION_KEY, []):
            _remember(entry)

    # Return sorted uppercase codes
    return sorted(shows)


def remember_custom_show(show_id: str) -> None:
    """Persist a show identifier in session state so dropdowns include it immediately.

    The show_id is normalized to UPPERCASE for consistency.
    Case-insensitive deduplication ensures 'rhoslc' and 'RHOSLC' don't create duplicates.
    """
    if not show_id or not isinstance(show_id, str):
        return
    # Normalize to uppercase
    cleaned = show_id.strip().upper()
    if not cleaned:
        return
    custom: List[str] = st.session_state.setdefault(_CUSTOM_SHOWS_SESSION_KEY, [])
    # Check case-insensitively to avoid duplicates
    existing_upper = {s.upper() for s in custom}
    if cleaned not in existing_upper:
        custom.append(cleaned)


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        if math.isnan(value) or math.isinf(value):
            return None
        return int(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return int(float(cleaned))
        except ValueError:
            return None
    return None


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def format_count(value: Any) -> str | None:
    numeric = coerce_int(value)
    if numeric is None:
        return None
    return f"{numeric:,}"


SCENE_THRESHOLD_DEFAULT = max(_env_float("SCENE_THRESHOLD", 27.0), 0.0)
SCENE_MIN_LEN_DEFAULT = max(_env_int("SCENE_MIN_LEN", 12), 1)
SCENE_WARMUP_DETS_DEFAULT = max(_env_int("SCENE_WARMUP_DETS", 3), 0)


def describe_error(url: str, exc: requests.RequestException) -> str:
    detail = str(exc)
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            detail = exc.response.text or exc.response.reason or detail
        except Exception:  # pragma: no cover
            detail = str(exc)
    return f"{url} → {detail}"


def _api_base() -> str:
    base = st.session_state.get("api_base")
    if not base:
        raise RuntimeError("init_page() must be called before API access")
    return base


def init_page(title: str = DEFAULT_TITLE) -> Dict[str, str]:
    # Set page config with wide layout - use try/except to handle multi-page navigation
    # where set_page_config may have already been called
    try:
        st.set_page_config(page_title=title, layout="wide")
    except st.errors.StreamlitAPIException:
        pass  # Already configured, ignore

    # Clear render flags at the start of each page run to allow widgets to be rendered fresh
    st.session_state.pop("_episode_selector_rendered", None)

    inject_custom_fonts()

    api_base = st.session_state.get("api_base") or _env("SCREENALYTICS_API_URL", "http://localhost:8000")
    st.session_state["api_base"] = api_base  # Always set (setdefault doesn't overwrite "")
    backend = st.session_state.get("backend") or _env("STORAGE_BACKEND", "s3").lower()
    st.session_state["backend"] = backend
    bucket = st.session_state.get("bucket") or (
        _env("AWS_S3_BUCKET") or _env("SCREENALYTICS_OBJECT_STORE_BUCKET") or ("local" if backend == "local" else "screenalytics")
    )
    st.session_state["bucket"] = bucket

    query_ep_id = st.query_params.get("ep_id", "")
    stored_ep_id = st.session_state.get("ep_id")
    if stored_ep_id is None:
        st.session_state["ep_id"] = query_ep_id
    elif query_ep_id and query_ep_id != stored_ep_id:
        st.session_state["ep_id"] = query_ep_id
    elif stored_ep_id and (not query_ep_id or query_ep_id != stored_ep_id):
        params = st.query_params
        params["ep_id"] = stored_ep_id
        st.query_params = params

    if "device_default_label" not in st.session_state:
        st.session_state["device_default_label"] = _guess_device_label()
    st.session_state.setdefault("detector_choice", DEFAULT_DETECTOR)
    st.session_state.setdefault("tracker_choice", DEFAULT_TRACKER)
    st.session_state.setdefault("scene_detector_choice", SCENE_DETECTOR_DEFAULT)

    sidebar = st.sidebar

    # API status at very top of sidebar
    sidebar.header("API")
    sidebar.code(api_base)
    health_url = f"{api_base}/healthz"
    api_healthy = False
    health_timeout = float(os.environ.get("SCREENALYTICS_HEALTH_TIMEOUT", "1.0"))
    try:
        resp = requests.get(health_url, timeout=health_timeout)
        resp.raise_for_status()
        sidebar.success("/healthz OK")
        api_healthy = True
    except requests.Timeout:
        sidebar.warning(f"/healthz timed out after {health_timeout:.1f}s (continuing)")
    except requests.RequestException as exc:
        sidebar.error(describe_error(health_url, exc))
    sidebar.caption(f"Backend: {backend} | Bucket: {bucket}")
    # Buttons: refresh status or restart API
    btn_col1, btn_col2 = sidebar.columns(2)
    with btn_col1:
        if st.button("🔄 Refresh", key="sidebar_refresh_api", use_container_width=True):
            st.rerun()
    with btn_col2:
        if st.button("⚡ Restart API", key="sidebar_restart_api", use_container_width=True):
            _restart_api_server()
            import time
            time.sleep(2)  # Give API time to start
            st.rerun()
    sidebar.divider()

    # Episode selector
    render_sidebar_episode_selector()

    render_workspace_nav()

    # Render global episode selector in sidebar
    sidebar.divider()
    render_sidebar_episode_selector()

    return {
        "api_base": api_base,
        "backend": backend,
        "bucket": bucket,
        "ep_id": st.session_state.get("ep_id", ""),
    }


def set_ep_id(ep_id: str, rerun: bool = True) -> None:
    if not ep_id:
        return
    current = st.session_state.get("ep_id")
    if current == ep_id:
        params = st.query_params
        params["ep_id"] = ep_id
        st.query_params = params
        return
    st.session_state["ep_id"] = ep_id
    params = st.query_params
    params["ep_id"] = ep_id
    st.query_params = params
    if rerun:
        st.rerun()


def get_ep_id() -> str:
    return st.session_state.get("ep_id", "")


def get_ep_id_from_query_params() -> str:
    """Get ep_id from query params, falling back to session state.

    Returns:
        The ep_id from query params if present, otherwise from session state,
        or empty string if neither is set.
    """
    query_ep_id = st.query_params.get("ep_id", "")
    if query_ep_id:
        return query_ep_id
    return st.session_state.get("ep_id", "")


def render_sidebar_episode_selector() -> str | None:
    """Render a global episode selector in the sidebar.

    Returns the selected ep_id, or None if no episodes exist.

    Behavior:
    - Defaults to most recently uploaded episode when no ep_id is set
    - Persists selection in st.session_state across page changes
    - Falls back to newest episode if previously selected ep_id no longer exists
    - Only renders the widget once per page run to avoid duplicate key errors
    """
    # Guard: only render the selector widget once per page run
    # This prevents DuplicateWidgetID errors if init_page() is called multiple times
    _rendered_key = "_episode_selector_rendered"
    if st.session_state.get(_rendered_key):
        return st.session_state.get("ep_id")
    st.session_state[_rendered_key] = True

    try:
        episodes_payload = api_get("/episodes")
    except requests.RequestException:
        st.sidebar.info("API unavailable")
        return None

    episodes = episodes_payload.get("episodes", [])

    if not episodes:
        st.sidebar.info("No episodes yet — upload one to get started.")
        return None

    # Sort by updated_at descending (most recently worked on first)
    # Fallback to created_at, then ep_id for stable sorting
    def sort_key(ep: Dict[str, Any]) -> tuple:
        updated_at = ep.get("updated_at", "")
        created_at = ep.get("created_at", "")
        return (updated_at, created_at, ep.get("ep_id", ""))

    episodes = sorted(episodes, key=sort_key, reverse=True)

    # Determine current ep_id with lock support
    ep_ids = [ep.get("ep_id") for ep in episodes if ep.get("ep_id")]
    locked = bool(st.session_state.get("ep_locked", True))
    locked_ep_id = st.session_state.get("locked_ep_id", "")
    current_ep_id = st.session_state.get("ep_id", "")

    if locked and locked_ep_id in ep_ids:
        current_ep_id = locked_ep_id
    else:
        # If current ep_id doesn't exist in list, fallback to most recent
        if current_ep_id and current_ep_id not in ep_ids and ep_ids:
            current_ep_id = ep_ids[0]  # Most recent episode
        elif not current_ep_id and ep_ids:
            current_ep_id = ep_ids[0]  # Default to most recent

    # Build labels for selectbox
    def format_label(ep: Dict[str, Any]) -> str:
        ep_id = ep["ep_id"]
        parsed = parse_ep_id(ep_id)
        if parsed:
            show = parsed["show"].upper()
            season = parsed["season"]
            episode = parsed["episode"]
            return f"{show} S{season:02d}E{episode:02d}"
        return ep_id

    labels = {ep["ep_id"]: format_label(ep) for ep in episodes}

    # Determine index for selectbox
    try:
        current_index = ep_ids.index(current_ep_id)
    except ValueError:
        current_index = 0

    # Header + lock toggle
    header_cols = st.sidebar.columns([4, 1])
    header_cols[0].markdown("**Episode**")
    lock_label = "🔒" if locked else "💾"
    lock_help = "Unlock to change episode" if locked else "Lock selection across page reloads"
    toggle = header_cols[1].button(lock_label, key="ep_lock_toggle", help=lock_help)
    if toggle:
        if locked:
            st.session_state["ep_locked"] = False
            st.session_state["locked_ep_id"] = ""
            locked = False
            locked_ep_id = ""
        else:
            st.session_state["ep_locked"] = True
            st.session_state["locked_ep_id"] = current_ep_id
            locked = True
            locked_ep_id = current_ep_id

    # Render selectbox in sidebar
    selected_ep_id = st.sidebar.selectbox(
        f"Episode {'🔒' if locked else '🔓'}",
        options=ep_ids,
        format_func=lambda eid: labels.get(eid, eid),
        index=current_index,
        key="global_episode_selector",
        disabled=locked,
    )

    # Update session state if selection changed (and not locked)
    if not locked and selected_ep_id != current_ep_id:
        set_ep_id(selected_ep_id, rerun=False)
        current_ep_id = selected_ep_id

    # Keep locked ep_id consistent
    if locked:
        st.session_state["locked_ep_id"] = current_ep_id
        set_ep_id(current_ep_id, rerun=False)

    # Cache clear button
    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear Python Cache", help="Clear .pyc files and __pycache__ directories"):
        import shutil
        from pathlib import Path

        try:
            # Get the project root (3 levels up from ui_helpers.py)
            project_root = Path(__file__).resolve().parents[2]

            # Clear __pycache__ directories
            pycache_count = 0
            for pycache_dir in project_root.rglob("__pycache__"):
                shutil.rmtree(pycache_dir, ignore_errors=True)
                pycache_count += 1

            # Clear .pyc files
            pyc_count = 0
            for pyc_file in project_root.rglob("*.pyc"):
                pyc_file.unlink(missing_ok=True)
                pyc_count += 1

            st.sidebar.success(f"✅ Cleared {pycache_count} cache dirs, {pyc_count} .pyc files")
        except Exception as exc:
            st.sidebar.error(f"Cache clear failed: {exc}")

    return selected_ep_id


def api_get(path: str, **kwargs) -> Dict[str, Any]:
    base = st.session_state.get("api_base")
    if not base:
        raise RuntimeError("init_page() must be called before API access")
    timeout = kwargs.pop("timeout", 60)
    resp = requests.get(f"{base}{path}", timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, json: Dict[str, Any] | None = None, **kwargs) -> Dict[str, Any]:
    base = st.session_state.get("api_base")
    if not base:
        raise RuntimeError("init_page() must be called before API access")
    timeout = kwargs.pop("timeout", 60)
    resp = requests.post(f"{base}{path}", json=json or {}, timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp.json()


def api_patch(path: str, json: Dict[str, Any] | None = None, **kwargs) -> Dict[str, Any]:
    """Send a PATCH request to the API."""
    base = st.session_state.get("api_base")
    if not base:
        raise RuntimeError("init_page() must be called before API access")
    timeout = kwargs.pop("timeout", 60)
    resp = requests.patch(f"{base}{path}", json=json or {}, timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp.json()


def api_delete(path: str, **kwargs) -> Dict[str, Any]:
    base = st.session_state.get("api_base")
    if not base:
        raise RuntimeError("init_page() must be called before API access")
    timeout = kwargs.pop("timeout", 60)
    resp = requests.delete(f"{base}{path}", timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp.json()


def api_post_files(path: str, files: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """POST request with file upload.

    Args:
        path: API path
        files: Dict of {name: (filename, file_bytes, content_type)}
        **kwargs: Additional requests kwargs

    Returns:
        JSON response
    """
    base = st.session_state.get("api_base")
    if not base:
        raise RuntimeError("init_page() must be called before API access")
    timeout = kwargs.pop("timeout", 120)
    resp = requests.post(f"{base}{path}", files=files, timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp.json()


# ============================================================================
# Async Job Helpers (Celery/Redis background jobs)
# ============================================================================

ASYNC_JOB_KEY = "_async_job"


def _run_local_job_with_progress(
    base: str,
    endpoint: str,
    payload: Dict[str, Any],
    operation: str,
    episode_id: str,
    progress_endpoint: str,
) -> Optional[Dict[str, Any]]:
    """Run a local-mode job with real-time progress polling.

    Uses a background thread to make the HTTP request while the main thread
    polls for progress updates and displays them in a Streamlit status container.

    Args:
        base: API base URL
        endpoint: API endpoint for the job
        payload: Request payload
        operation: Operation name for display
        episode_id: Episode ID
        progress_endpoint: Endpoint to poll for progress

    Returns:
        Response dict from the API, or None if failed
    """
    import queue

    # Shared state between threads
    result_queue: queue.Queue = queue.Queue()

    def _run_request():
        """Background thread to run the HTTP request."""
        try:
            resp = requests.post(f"{base}{endpoint}", json=payload, timeout=600)
            resp.raise_for_status()
            result_queue.put(("success", resp.json()))
        except requests.RequestException as e:
            result_queue.put(("error", str(e)))
        except Exception as e:
            result_queue.put(("error", str(e)))

    # Start the request in a background thread
    worker = threading.Thread(target=_run_request, daemon=True)
    worker.start()

    # Use st.status for expandable progress display
    with st.status(f"Running {operation} (local mode)...", expanded=True) as status:
        last_step = ""
        last_progress = 0.0
        poll_count = 0
        max_polls = 1200  # 10 minutes at 0.5s intervals

        while worker.is_alive() and poll_count < max_polls:
            poll_count += 1

            # Check if we have a result
            try:
                result_type, result_data = result_queue.get_nowait()
                break
            except queue.Empty:
                pass

            # Poll for progress
            try:
                progress_resp = requests.get(f"{base}{progress_endpoint}", timeout=5)
                if progress_resp.status_code == 200:
                    progress_data = progress_resp.json()
                    entries = progress_data.get("entries", [])

                    if entries:
                        latest = entries[-1]
                        step = latest.get("step", "")
                        progress = latest.get("progress", 0)
                        message = latest.get("message", "")

                        # Update status display
                        if step != last_step or progress != last_progress:
                            pct = int(progress * 100)
                            status.update(label=f"{operation}: {step} ({pct}%)")
                            st.write(f"**{step}**: {message}")
                            last_step = step
                            last_progress = progress

                    # Check if finished
                    if progress_data.get("finished"):
                        if progress_data.get("error"):
                            status.update(label=f"{operation} failed", state="error")
                            st.error(f"Error: {progress_data.get('error')}")
                        break
            except requests.RequestException:
                pass  # Progress polling failed, continue waiting

            time.sleep(0.5)

        # Wait for thread to finish if still running
        worker.join(timeout=5)

        # Get final result
        try:
            result_type, result_data = result_queue.get(timeout=1)
        except queue.Empty:
            status.update(label=f"{operation} timed out", state="error")
            st.error(f"Job timed out after {max_polls * 0.5:.0f} seconds")
            return None

        if result_type == "error":
            status.update(label=f"{operation} failed", state="error")
            st.error(f"Failed: {result_data}")
            return None

        # Success - show detailed results
        data = result_data
        job_status = data.get("status", "")
        result = data.get("result", {})

        if job_status in ("completed", "success"):
            status.update(label=f"✅ {operation} complete!", state="complete", expanded=True)

            # Show summary stats
            if result:
                summary = result.get("summary", result)
                clusters = summary.get("clusters", summary.get("total_clusters", 0))
                identities = summary.get("identities", 0)
                assigned = summary.get("assignments_count", summary.get("assigned_clusters", 0))
                new_people = summary.get("new_people_count", 0)

                cols = st.columns(4)
                cols[0].metric("Clusters", clusters)
                cols[1].metric("Assigned", assigned)
                cols[2].metric("New People", new_people)
                cols[3].metric("Identities", identities)

            # Show detailed log if available
            progress_log = data.get("progress_log", [])
            log_data = result.get("log", {})

            if log_data and log_data.get("steps"):
                with st.expander("📋 Detailed Log", expanded=False):
                    log_lines = []
                    for step_info in log_data.get("steps", []):
                        step_name = step_info.get("step", "")
                        step_status = step_info.get("status", "")
                        duration_ms = step_info.get("duration_ms", 0)
                        icon = "✓" if step_status == "success" else ("⊘" if step_status == "skipped" else "✗")
                        log_lines.append(f"[{icon}] {step_name}: {step_status} ({duration_ms}ms)")
                        for detail in step_info.get("details", []):
                            log_lines.append(f"    • {detail}")
                    st.code("\n".join(log_lines), language=None)
            elif progress_log:
                with st.expander("📋 Progress Log", expanded=False):
                    for entry in progress_log:
                        pct = int(entry.get("progress", 0) * 100)
                        st.write(f"[{pct}%] **{entry.get('step', '')}**: {entry.get('message', '')}")

        elif job_status == "error":
            status.update(label=f"❌ {operation} failed", state="error")
            st.error(f"Error: {data.get('error', 'Unknown error')}")
        else:
            status.update(label=f"✅ {operation} complete", state="complete")

        return data


def submit_async_job(
    endpoint: str,
    payload: Dict[str, Any],
    operation: str,
    episode_id: str,
) -> Optional[Dict[str, Any]]:
    """Submit an async job to the API and store in session state.

    Respects the execution mode setting for the episode:
    - Redis mode: Queues job via Celery, stores job_id for polling
    - Local mode: Runs job synchronously and returns result directly

    Args:
        endpoint: API endpoint (e.g., /episodes/{ep_id}/clusters/group_async)
        payload: Request payload
        operation: Operation name for display (e.g., "Auto Group")
        episode_id: Episode ID

    Returns:
        Response dict if submitted, None if failed or sync fallback occurred
    """
    base = st.session_state.get("api_base")
    if not base:
        st.error("API not initialized")
        return None

    # Get execution mode for this episode and add to payload
    execution_mode = get_execution_mode(episode_id)
    payload = {**payload, "execution_mode": execution_mode}

    # For local mode, use threading + progress polling for real-time updates
    if execution_mode == "local":
        # Determine progress endpoint based on the job type
        progress_endpoint = None
        if "clusters/group" in endpoint:
            progress_endpoint = f"/episodes/{episode_id}/clusters/group/progress"

        if progress_endpoint:
            # Use threading + polling for operations that support progress
            return _run_local_job_with_progress(
                base=base,
                endpoint=endpoint,
                payload=payload,
                operation=operation,
                episode_id=episode_id,
                progress_endpoint=progress_endpoint,
            )
        else:
            # Fallback to simple spinner for operations without progress endpoints
            with st.spinner(f"Running {operation} in local mode..."):
                try:
                    resp = requests.post(f"{base}{endpoint}", json=payload, timeout=600)
                    resp.raise_for_status()
                    data = resp.json()

                    status = data.get("status", "")
                    if status in ("completed", "success"):
                        st.success(f"✅ {operation} completed (local mode)")
                    elif status == "error":
                        st.error(f"❌ {operation} failed: {data.get('error', 'Unknown error')}")
                    else:
                        if data.get("async") is False:
                            st.info(f"{operation} completed (sync fallback)")
                    return data
                except requests.RequestException as e:
                    st.error(f"Failed to run {operation}: {describe_error(f'{base}{endpoint}', e)}")
                    return None

    # Redis mode - check if a job is already running
    existing = st.session_state.get(ASYNC_JOB_KEY)
    if existing:
        st.warning(f"⏳ A job is already running: {existing.get('operation')} ({existing.get('job_id', '')[:8]}...)")
        return None

    try:
        resp = requests.post(f"{base}{endpoint}", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("async", False) and data.get("job_id"):
            # Store job in session state for polling
            st.session_state[ASYNC_JOB_KEY] = {
                "job_id": data["job_id"],
                "operation": operation,
                "episode_id": episode_id,
                "created_at": time.time(),
            }
            st.info(f"⏳ {operation} started... (job {data['job_id'][:8]}...)")
            return data
        else:
            # Synchronous fallback - return result directly
            return data
    except requests.RequestException as e:
        st.error(f"Failed to submit {operation}: {describe_error(f'{base}{endpoint}', e)}")
        return None


def get_active_async_job() -> Optional[Dict[str, Any]]:
    """Get the currently active async job, if any."""
    return st.session_state.get(ASYNC_JOB_KEY)


def clear_async_job() -> None:
    """Clear the active async job and related tracking state from session state."""
    st.session_state.pop(ASYNC_JOB_KEY, None)
    st.session_state.pop("_async_job_state_tracking", None)
    st.session_state.pop("_async_job_poll_failures", None)
    st.session_state.pop("_async_job_retry_count", None)


def poll_async_job() -> Optional[Dict[str, Any]]:
    """Poll the active async job status.

    Returns:
        Job status dict with 'state', 'result', 'progress', etc., or None if no job/error
    """
    job = st.session_state.get(ASYNC_JOB_KEY)
    if not job:
        return None

    base = st.session_state.get("api_base")
    if not base:
        return None

    try:
        resp = requests.get(f"{base}/celery_jobs/{job['job_id']}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def render_async_job_status() -> bool:
    """Render active job status banner if a job is running.

    Returns:
        True if job is still running (caller should schedule refresh),
        False if no job or job is complete
    """
    job = st.session_state.get(ASYNC_JOB_KEY)
    if not job:
        return False

    # Timeouts for stale job detection
    STALE_JOB_TIMEOUT = 15 * 60  # 15 minutes max total job time
    QUEUED_TIMEOUT = 2 * 60  # 2 minutes stuck in queued
    NO_PROGRESS_TIMEOUT = 5 * 60  # 5 minutes without progress update

    created_at = job.get("created_at", 0)
    now = time.time()

    # Overall job timeout
    if now - created_at > STALE_JOB_TIMEOUT:
        st.warning(f"Job timed out after 15 minutes: {job.get('operation', 'unknown')}")
        clear_async_job()
        return False

    # Track consecutive poll failures
    poll_fail_key = "_async_job_poll_failures"
    max_poll_failures = 5  # Clear job after 5 consecutive failures

    status = poll_async_job()
    if not status:
        failures = st.session_state.get(poll_fail_key, 0) + 1
        st.session_state[poll_fail_key] = failures
        if failures >= max_poll_failures:
            st.warning(f"Unable to check job status after {failures} attempts. Clearing stale job.")
            clear_async_job()
            st.session_state.pop(poll_fail_key, None)
            return False
        st.warning(f"Unable to check job status: {job['job_id'][:8]}... (attempt {failures}/{max_poll_failures})")
        return False

    # Reset poll failure counter on success
    st.session_state.pop(poll_fail_key, None)

    state = status.get("state", "unknown")
    operation = job.get("operation", "Operation")
    job_id_short = job["job_id"][:8]

    # Track state transitions and progress to detect stuck jobs
    state_key = "_async_job_state_tracking"
    state_tracking = st.session_state.get(state_key, {})
    prev_state = state_tracking.get("state")
    prev_progress = state_tracking.get("progress_pct", 0)

    # Get current progress percentage
    progress = status.get("progress", {})
    current_progress_pct = progress.get("progress", 0) if progress else 0

    # Detect if state or progress changed
    state_changed = prev_state != state
    progress_changed = current_progress_pct != prev_progress

    if state_changed or progress_changed:
        # Update tracking with current state and time
        st.session_state[state_key] = {
            "state": state,
            "progress_pct": current_progress_pct,
            "last_update": now,
        }
    else:
        # Check for stuck job
        last_update = state_tracking.get("last_update", created_at)

        # Jobs stuck in 'queued' for too long (worker likely not running)
        if state == "queued" and (now - last_update) > QUEUED_TIMEOUT:
            st.warning(
                f"Job stuck in queue for over 2 minutes. Celery worker may not be running. "
                f"Clearing job: {job.get('operation', 'unknown')}"
            )
            clear_async_job()
            st.session_state.pop(state_key, None)
            return False

        # Jobs without progress update for too long
        if state == "in_progress" and (now - last_update) > NO_PROGRESS_TIMEOUT:
            st.warning(
                f"Job has not reported progress for 5 minutes. "
                f"Clearing stale job: {job.get('operation', 'unknown')}"
            )
            clear_async_job()
            st.session_state.pop(state_key, None)
            return False

    # Handle unexpected states
    if state in ("retrying", "unknown"):
        retry_count_key = "_async_job_retry_count"
        retry_count = st.session_state.get(retry_count_key, 0) + 1
        st.session_state[retry_count_key] = retry_count
        if retry_count >= 3:
            st.warning(f"Job in unexpected state '{state}' after {retry_count} checks. Clearing.")
            clear_async_job()
            st.session_state.pop(retry_count_key, None)
            st.session_state.pop(state_key, None)
            return False
        st.info(f"⏳ **{operation}** {state}... ({job_id_short}...)")
        return True

    if state in ("queued", "in_progress"):
        # Show progress with visual progress bar
        progress = status.get("progress", {})
        step = progress.get("step", "")
        message = progress.get("message", "Working...")
        progress_pct = progress.get("progress", 0)  # 0.0-1.0
        current = progress.get("current", 0)
        total = progress.get("total", 0)

        # Build info message
        if step:
            st.info(f"⏳ **{operation}** in progress ({job_id_short}...)\n\n**{step}**: {message}")
        else:
            st.info(f"⏳ **{operation}** in progress ({job_id_short}...)")

        # Show visual progress bar
        if progress_pct > 0:
            st.progress(min(progress_pct, 1.0))
            if current and total:
                st.caption(f"Progress: {current:,} / {total:,} ({progress_pct * 100:.1f}%)")
            else:
                st.caption(f"Progress: {progress_pct * 100:.1f}%")
        elif state == "queued":
            st.caption("Waiting in queue...")
        else:
            st.caption("Starting...")

        return True

    elif state == "success":
        result = status.get("result", {})
        log_data = result.get("log", {})
        summary = result.get("summary", result)  # Fallback to result if no summary

        # Build detailed log lines
        log_lines = []
        has_log = False

        if log_data and log_data.get("steps"):
            has_log = True
            log_lines.append("=" * 50)
            log_lines.append(f"{operation.upper()} - DETAILED LOG")
            log_lines.append("=" * 50)

            for step_info in log_data.get("steps", []):
                step_name = step_info.get("step", "")
                step_status = step_info.get("status", "")
                duration_ms = step_info.get("duration_ms", 0)
                details = step_info.get("details", [])

                status_icon = "✓" if step_status == "success" else ("⊘" if step_status == "skipped" else "✗")
                step_display = step_name.replace("_", " ").title()

                log_lines.append("")
                log_lines.append(f"[{status_icon}] {step_display}")
                log_lines.append(f"    Duration: {duration_ms}ms")

                for detail in details:
                    log_lines.append(f"    • {detail}")

                if step_info.get("error"):
                    log_lines.append(f"    ⚠️ ERROR: {step_info.get('error')}")

            total_duration = log_data.get("total_duration_ms", 0)
            log_lines.append("")
            log_lines.append("=" * 50)
            log_lines.append(f"Total Duration: {total_duration}ms")
            log_lines.append("=" * 50)

        # Format success message based on operation type
        if "auto" in operation.lower() or "group" in operation.lower():
            assigned = summary.get("assignments_count", summary.get("succeeded", 0))
            new_people = summary.get("new_people_count", 0)
            facebank = summary.get("facebank_assigned", 0)
            clusters = summary.get("clusters", 0)
            identities = summary.get("identities", 0)

            msg_parts = [f"✅ **{operation}** complete!"]
            if clusters:
                msg_parts.append(f"\n• **Clusters:** {clusters}")
            if assigned:
                msg_parts.append(f"\n• **Processed:** {assigned}")
            if new_people:
                msg_parts.append(f"\n• **New people:** {new_people}")
            if facebank:
                msg_parts.append(f"\n• **Facebank matches:** {facebank}")
            if identities:
                msg_parts.append(f"\n• **Identities:** {identities}")

            st.success("".join(msg_parts))

        elif "assign" in operation.lower():
            succeeded = summary.get("succeeded", 0)
            failed = summary.get("failed", 0)
            st.success(f"✅ **{operation}** complete!\n\n• **Succeeded:** {succeeded}\n• **Failed:** {failed}")
        elif "cleanup" in operation.lower():
            tracks_before = summary.get("tracks_before", 0)
            tracks_after = summary.get("tracks_after", 0)
            clusters_before = summary.get("clusters_before", 0)
            clusters_after = summary.get("clusters_after", 0)
            st.success(
                f"✅ **{operation}** complete!\n\n"
                f"• **Tracks:** {tracks_before:,} → {tracks_after:,}\n"
                f"• **Clusters:** {clusters_before:,} → {clusters_after:,}"
            )
        else:
            st.success(f"✅ **{operation}** complete!")

        # Show detailed log if available
        if has_log:
            with st.expander("📋 Detailed Log", expanded=False):
                st.code("\n".join(log_lines), language=None)

        clear_async_job()
        return False

    elif state == "failed":
        error = status.get("error", status.get("result", "Unknown error"))
        st.error(f"❌ **{operation}** failed: {error}")
        clear_async_job()
        return False

    elif state == "cancelled":
        st.warning(f"🚫 **{operation}** was cancelled")
        clear_async_job()
        return False

    return False


def fetch_trr_metadata(show_slug: str) -> Dict[str, Any]:
    """Fetch TRR canonical metadata for a show from the Postgres backend.

    Args:
        show_slug: Show identifier (e.g., 'RHOBH', 'RHOSLC')

    Returns:
        Dict containing:
        - show: Show metadata object
        - seasons: List of season objects
        - episodes: List of episode objects
        - cast: List of cast member objects

    Raises:
        RuntimeError: If API base not initialized
        requests.HTTPError: If API calls fail
    """
    # Fetch show metadata first to validate the show exists
    show = api_get(f"/metadata/shows/{show_slug}")

    # Fetch related data
    seasons_resp = api_get(f"/metadata/shows/{show_slug}/seasons")
    episodes_resp = api_get(f"/metadata/shows/{show_slug}/episodes")

    # Cast may not exist yet for all shows; handle 404 gracefully
    cast_rows = []
    try:
        cast_rows = api_get(f"/metadata/shows/{show_slug}/cast")
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code != 404:
            raise

    return {
        "show": show,
        "seasons": seasons_resp.get("seasons", []),
        "episodes": episodes_resp.get("episodes", []),
        "cast": cast_rows,
    }


def _episode_status_payload(ep_id: str, *, run_id: str | None = None) -> Dict[str, Any] | None:
    url = f"{_api_base()}/episodes/{ep_id}/status"
    try:
        run_id_value = run_id.strip() if isinstance(run_id, str) else ""
        params = {"run_id": run_id_value} if run_id_value else None
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    try:
        payload = resp.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def get_episode_status(ep_id: str, *, run_id: str | None = None) -> Dict[str, Any] | None:
    return _episode_status_payload(ep_id, run_id=run_id)


def link_local(path: Path | str) -> str:
    return f"`{path}`"


@lru_cache(maxsize=_LOCAL_MEDIA_CACHE_SIZE)
def _data_url_cache(path_str: str) -> str | None:
    file_path = Path(path_str)
    try:
        data = file_path.read_bytes()
    except OSError:
        return None
    encoded = base64.b64encode(data).decode("ascii")
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type:
        mime_type = "image/jpeg"
    return f"data:{mime_type};base64,{encoded}"


def seed_display_source(seed: Dict[str, Any] | None) -> str | None:
    """Return the preferred image reference for a facebank seed."""
    if not seed:
        return None
    for key in ("display_s3_key", "display_key", "image_s3_key"):
        value = seed.get(key)
        if value:
            return value
    for key in ("orig_s3_key", "orig_uri", "display_url", "image_uri"):
        value = seed.get(key)
        if value:
            return value
    return None


def ensure_media_url(path_or_url: str | Path | None) -> str | None:
    """Return a browser-safe URL for local artifacts, falling back to the original string."""
    if not path_or_url:
        return None
    value = str(path_or_url)
    parsed = urlparse(value)
    scheme = parsed.scheme.lower()
    if scheme in {"http", "https", "data"}:
        return value
    candidate_paths: List[Path] = []
    if scheme == "file":
        candidate_paths.append(Path(parsed.path))
    else:
        candidate_paths.append(Path(value))
    first = candidate_paths[0]
    if not first.is_absolute():
        candidate_paths.append((DATA_ROOT / first).expanduser())
    for candidate in candidate_paths:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved.is_file():
            cached = _data_url_cache(str(resolved))
            if cached:
                return cached
    return value


def human_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f} {units[idx]}"


def ds(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        st.info("No rows yet.")
    else:
        st.dataframe(rows, use_container_width=True)


def device_default_label() -> str:
    return st.session_state.get("device_default_label", "CPU")


def device_label_index(label: str) -> int:
    try:
        return DEVICE_LABELS.index(label)
    except ValueError:
        return 0


def device_label_from_value(value: str | None) -> str:
    if not value:
        return device_default_label()
    normalized = str(value).strip().lower()
    label = DEVICE_VALUE_TO_LABEL.get(normalized)
    if label:
        return label
    if normalized in {"0", "cuda", "cudaexecutionprovider", "gpu"}:
        return "CUDA"
    if normalized in {"coreml", "coremlexecutionprovider", "metal", "apple"}:
        return "CoreML"
    if normalized == "mps":
        return "MPS"
    if normalized == "auto":
        return "Auto"
    return device_default_label()


def detector_default_value() -> str:
    return st.session_state.get("detector_choice", DEFAULT_DETECTOR)


def detector_label_index(value: str | None) -> int:
    key = str(value).lower() if value else DEFAULT_DETECTOR
    label = DETECTOR_LABEL_MAP.get(key)
    if label in DETECTOR_LABELS:
        return DETECTOR_LABELS.index(label)
    return 0


def detector_label_from_value(value: str | None) -> str:
    if not value:
        return "unknown"
    value = value.lower()
    return DETECTOR_LABEL_MAP.get(value, value)


def remember_detector(value: str | None) -> None:
    key = (value or "").lower() if value else ""
    if key in FACE_ONLY_DETECTORS:
        st.session_state["detector_choice"] = key


def tracker_default_value() -> str:
    return st.session_state.get("tracker_choice", DEFAULT_TRACKER)


def tracker_label_index(value: str | None) -> int:
    key = str(value).lower() if value else DEFAULT_TRACKER
    label = TRACKER_LABEL_MAP.get(key)
    if label in TRACKER_LABELS:
        return TRACKER_LABELS.index(label)
    return 0


def tracker_label_from_value(value: str | None) -> str:
    if not value:
        return "unknown"
    return TRACKER_LABEL_MAP.get(value.lower(), value)


def remember_tracker(value: str | None) -> None:
    key = (value or "").lower() if value else ""
    if key in TRACKER_LABEL_MAP:
        st.session_state["tracker_choice"] = key


# ─── Profile helper functions ───────────────────────────────────────────────


def default_profile_for_device(device: str | None) -> str:
    """Return the default performance profile for a given device type."""
    if not device:
        return "balanced"
    normalized = str(device).strip().lower()
    return DEVICE_DEFAULT_PROFILE.get(normalized, "balanced")


def profile_value_from_state(state_value: str | None) -> str:
    """Normalize profile value from session state."""
    if not state_value:
        return "balanced"
    normalized = str(state_value).strip().lower()
    # Handle both value format ("low_power") and label format ("Low Power")
    if normalized in PROFILE_DEFAULTS:
        return normalized
    # Try to map from label
    return PROFILE_VALUE_MAP.get(state_value, "balanced")


def profile_label_index(value: str | None) -> int:
    """Get the index of a profile value in PROFILE_LABELS."""
    if not value:
        return 1  # Default to "Balanced"
    normalized = str(value).strip().lower()
    label = PROFILE_LABEL_MAP.get(normalized)
    if label and label in PROFILE_LABELS:
        return PROFILE_LABELS.index(label)
    return 1  # Default to "Balanced"


def profile_defaults(profile_value: str | None) -> dict:
    """Get default settings for a performance profile."""
    if not profile_value:
        return PROFILE_DEFAULTS.get("balanced", {})
    normalized = str(profile_value).strip().lower()
    return PROFILE_DEFAULTS.get(normalized, PROFILE_DEFAULTS.get("balanced", {}))


def profile_label_from_value(value: str | None) -> str:
    """Convert profile value to display label."""
    if not value:
        return "Balanced"
    normalized = str(value).strip().lower()
    return PROFILE_LABEL_MAP.get(normalized, "Balanced")


def scene_detector_label_index(value: str | None = None) -> int:
    effective = (value or SCENE_DETECTOR_DEFAULT).lower()
    label = SCENE_DETECTOR_LABEL_MAP.get(effective, SCENE_DETECTOR_LABELS[0])
    try:
        return SCENE_DETECTOR_LABELS.index(label)
    except ValueError:
        return 0


def scene_detector_value_from_label(label: str | None) -> str:
    if not label:
        return SCENE_DETECTOR_DEFAULT
    return SCENE_DETECTOR_VALUE_MAP.get(label, SCENE_DETECTOR_DEFAULT)


def scene_detector_label(value: str | None) -> str:
    key = (value or SCENE_DETECTOR_DEFAULT).lower()
    return SCENE_DETECTOR_LABEL_MAP.get(key, key)


def default_detect_track_payload(
    ep_id: str,
    *,
    stride: int | None = None,
    device: str | None = None,
    det_thresh: float | None = None,
) -> Dict[str, Any]:
    """Build default payload for detect/track pipeline.

    IMPORTANT: Defaults are optimized for minimal resource usage:
    - save_frames=False (explicitly enable to save sampled frames)
    - save_crops=False (explicitly enable to save face crops)
    - jpeg_quality=72 (lower quality for smaller files when enabled)

    This ensures default runs are fast and don't generate unnecessary I/O.
    """
    payload: Dict[str, Any] = {
        "ep_id": ep_id,
        "stride": int(stride if stride is not None else DEFAULT_STRIDE),
        "device": (device or DEFAULT_DEVICE).lower(),
        "detector": DEFAULT_DETECTOR,
        "tracker": DEFAULT_TRACKER,
        "det_thresh": float(det_thresh if det_thresh is not None else DEFAULT_DET_THRESH),
        "save_frames": False,  # CHANGED: off by default for minimal I/O
        "save_crops": False,   # CHANGED: off by default for minimal I/O
        "jpeg_quality": 72,    # CHANGED: lower quality (was 85) when enabled
        "scene_detector": SCENE_DETECTOR_DEFAULT,
        "scene_threshold": SCENE_THRESHOLD_DEFAULT,
        "scene_min_len": SCENE_MIN_LEN_DEFAULT,
        "scene_warmup_dets": SCENE_WARMUP_DETS_DEFAULT,
    }
    return payload


def default_cleanup_payload(ep_id: str) -> Dict[str, Any]:
    """Default payload for /jobs/episode_cleanup_async."""
    return {
        "ep_id": ep_id,
        "stride": DEFAULT_STRIDE,
        "fps": 0.0,
        "device": DEFAULT_DEVICE,
        "embed_device": DEFAULT_DEVICE,
        "detector": DEFAULT_DETECTOR,
        "tracker": DEFAULT_TRACKER,
        "max_gap": DEFAULT_MAX_GAP,
        "det_thresh": DEFAULT_DET_THRESH,
        "save_frames": False,
        "save_crops": False,
        "jpeg_quality": 85,
        "scene_detector": SCENE_DETECTOR_DEFAULT,
        "scene_threshold": SCENE_THRESHOLD_DEFAULT,
        "scene_min_len": SCENE_MIN_LEN_DEFAULT,
        "scene_warmup_dets": SCENE_WARMUP_DETS_DEFAULT,
        "cluster_thresh": DEFAULT_CLUSTER_SIMILARITY,
        "min_cluster_size": 2,
        "thumb_size": 256,
        # Focus cleanup on unassigned identities/clusters; avoid full recluster by default.
        "actions": ["split_tracks", "reembed", "group_clusters"],
        "write_back": True,
    }


@lru_cache(maxsize=1)
def _onnx_provider_names() -> set[str]:
    try:
        import onnxruntime as ort  # type: ignore

        return {provider.lower() for provider in ort.get_available_providers()}
    except Exception:  # pragma: no cover - onnxruntime optional
        return set()


def _coreml_provider_present() -> bool:
    return any(name.startswith("coreml") for name in _onnx_provider_names())


def _cuda_provider_present() -> bool:
    return any(name.startswith("cuda") for name in _onnx_provider_names())


@lru_cache(maxsize=1)
def is_apple_silicon() -> bool:
    return platform.system().lower() == "darwin" and platform.machine().lower().startswith(("arm", "aarch64"))


def _guess_device_label() -> str:
    if _cuda_provider_present():
        return "CUDA"
    if _coreml_provider_present():
        return "CoreML"
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():  # pragma: no cover
            return "CUDA"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():  # pragma: no cover
            return "MPS"
    except Exception:  # pragma: no cover
        pass
    return "CPU"


@lru_cache(maxsize=1)
def list_supported_devices() -> list[str]:
    """Return list of device labels supported on the current host.

    This is the single source of truth for which devices are available.
    The UI should use this to filter the device selector options.

    Returns:
        List of device labels (e.g., ["Auto", "CPU", "CUDA"])
    """
    supported = ["Auto", "CPU"]  # Always available

    # Check for CUDA
    if _cuda_provider_present():
        supported.append("CUDA")
    else:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                supported.append("CUDA")
        except Exception:
            pass

    # Check for Apple Silicon / CoreML / MPS
    if is_apple_silicon():
        if _coreml_provider_present():
            supported.append("CoreML")
        # Check for MPS
        try:
            import torch  # type: ignore
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                supported.append("MPS")
        except Exception:
            pass

    return supported


def validate_device(device_label: str) -> tuple[bool, str | None]:
    """Validate that a device is supported on the current host.

    Args:
        device_label: Device label (e.g., "CUDA", "CoreML")

    Returns:
        Tuple of (is_valid, error_message)
    """
    supported = list_supported_devices()
    if device_label in supported:
        return True, None

    # Provide helpful error message
    if device_label == "CUDA":
        return False, "CUDA is not available on this host. No CUDA-enabled GPU detected."
    elif device_label in ("CoreML", "MPS"):
        return False, f"{device_label} is only available on Apple Silicon Macs."
    else:
        return False, f"Unknown device: {device_label}. Supported: {', '.join(supported)}"


def parse_ep_id(ep_id: str) -> Optional[Dict[str, int | str]]:
    match = _EP_ID_REGEX.match(ep_id)
    if not match:
        return None
    show = match.group("show")
    try:
        season = int(match.group("season"))
        episode = int(match.group("episode"))
    except ValueError:
        return None
    return {"show": show, "season": season, "episode": episode}


def _manifest_path(ep_id: str, filename: str) -> Path:
    return DATA_ROOT / "manifests" / ep_id / filename


def tracks_detector_value(ep_id: str) -> str | None:
    path = _manifest_path(ep_id, "tracks.jsonl")
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                detector = payload.get("detector")
                if detector:
                    return str(detector).lower()
    except OSError:
        return None
    return None


def tracks_tracker_value(ep_id: str) -> str | None:
    path = _manifest_path(ep_id, "tracks.jsonl")
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tracker = payload.get("tracker")
                if tracker:
                    return str(tracker).lower()
    except OSError:
        return None
    return None


def detect_tracker_combo(ep_id: str, detect_status: Dict[str, Any] | None = None) -> tuple[str | None, str | None]:
    detector = tracks_detector_value(ep_id)
    tracker = tracks_tracker_value(ep_id)
    if not detector and detect_status:
        detector = detect_status.get("detector")
    if not tracker and detect_status:
        tracker = detect_status.get("tracker")
    det_value = str(detector).lower() if detector else None
    tracker_value = str(tracker).lower() if tracker else None
    return det_value, tracker_value


def pipeline_combo_supported(stage: str, detector: str | None, tracker: str | None) -> bool:
    det_value = str(detector).lower() if detector else None
    tracker_value = str(tracker).lower() if tracker else None
    if not det_value or not tracker_value:
        return False
    combos = SUPPORTED_PIPELINE_COMBOS.get(stage.lower())
    if not combos:
        return True
    return (det_value, tracker_value) in combos


def detector_is_face_only(ep_id: str, detect_status: Dict[str, Any] | None = None) -> bool:
    detector = tracks_detector_value(ep_id)
    if detector:
        return detector.lower() in FACE_ONLY_DETECTORS
    status_detector = None
    if detect_status and isinstance(detect_status, dict):
        status_detector = detect_status.get("detector")
    if status_detector:
        return str(status_detector).lower() in FACE_ONLY_DETECTORS
    manifest_path = _manifest_path(ep_id, "tracks.jsonl")
    if not manifest_path.exists():
        return False
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return True
    except OSError:
        return False
    return False


def tracks_detector_label(ep_id: str) -> str:
    return detector_label_from_value(tracks_detector_value(ep_id))


def tracks_tracker_label(ep_id: str) -> str:
    return tracker_label_from_value(tracks_tracker_value(ep_id))


def try_switch_page(page_path: str) -> None:
    """Switch pages without surfacing the sidebar warning fallback."""
    try:
        st.switch_page(page_path)
    except Exception:
        # Fallback: set query param so sidebar nav picks up the target page
        params = st.query_params
        params["page"] = page_path
        st.query_params = params
        st.rerun()


def inject_custom_fonts() -> None:
    """Inject custom font faces and set defaults for body/headings."""
    font_base = Path(__file__).resolve().parent / "assets" / "fonts"

    # Remote font hosting (Vercel) for easier loading in Streamlit
    remote_base = "https://trr-app.vercel.app/admin/fonts"
    remote_plymouth = f"{remote_base}/PlymouthSerial-ExtraBold.otf"
    remote_rude = f"{remote_base}/RudeSlabCondensedCondensedBold.otf"

    # Local fallbacks (copied into repo)
    local_plymouth = (font_base / "Plymouth Serial" / "PlymouthSerialExtraBold-10035290.otf").as_posix()
    local_rude = (font_base / "Rude Slab Condensed" / "RudeSlabCondensedCondensedBold-930861866.otf").as_posix()

    plymouth_src = remote_plymouth if remote_plymouth else local_plymouth
    rude_src = remote_rude if remote_rude else local_rude

    css = f"""
    <style>
    @font-face {{
        font-family: 'PlymouthSerialExtraBold';
        src: url('{plymouth_src}') format('opentype'),
             url('{local_plymouth}') format('opentype');
        font-weight: 800;
    }}
    @font-face {{
        font-family: 'RudeSlabCondensedBold';
        src: url('{rude_src}') format('opentype'),
             url('{local_rude}') format('opentype');
        font-weight: 700;
    }}
    html, body, div, button, input, textarea {{
        font-family: 'PlymouthSerialExtraBold', 'Inter', 'Helvetica', sans-serif;
        font-weight: 800;
    }}
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'RudeSlabCondensedBold', 'PlymouthSerialExtraBold', 'Inter', 'Helvetica', sans-serif;
        font-weight: 700;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def _nav_to_faces(view: str | None = None) -> None:
    """Helper to jump to Faces Review with a preset view."""
    if view:
        st.session_state["facebank_view"] = view
        st.session_state.pop("selected_person", None)
        st.session_state.pop("selected_identity", None)
        st.session_state.pop("selected_track", None)
    try_switch_page("pages/3_Faces_Review.py")


def render_workspace_nav() -> None:
    """Render grouped navigation with Faces Review sub-pages."""
    with st.sidebar:
        st.markdown("### Workspace")
        if hasattr(st, "page_link"):
            st.page_link("streamlit_app.py", label="Home")
            st.page_link("pages/0_Upload_Video.py", label="Upload Video")
            st.page_link("pages/1_Episodes.py", label="Episodes")
            st.page_link("pages/2_Episode_Detail.py", label="Episode Detail")
            st.page_link("pages/4_Cast.py", label="Cast Management")
            # Faces Review with visible sub-pages (indented, no emojis)
            st.page_link("pages/3_Faces_Review.py", label="Faces Review")
            st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;Cast Members (12)", unsafe_allow_html=True)
            st.page_link("pages/3_Smart_Suggestions.py", label="    Smart Suggestions")
            st.page_link("pages/4_Screentime.py", label="Screentime & Health")
            st.page_link("pages/9_Docs_Dashboard.py", label="Docs Dashboard")
        else:
            st.button("Home", key="nav_home", on_click=lambda: try_switch_page("streamlit_app.py"))
            st.button("Upload Video", key="nav_upload", on_click=lambda: try_switch_page("pages/0_Upload_Video.py"))
            st.button("Episodes", key="nav_episodes", on_click=lambda: try_switch_page("pages/1_Episodes.py"))
            st.button("Episode Detail", key="nav_ep_detail", on_click=lambda: try_switch_page("pages/2_Episode_Detail.py"))
            st.button("Cast Management", key="nav_cast", on_click=lambda: try_switch_page("pages/4_Cast.py"))
            st.button("Faces Review", key="nav_faces_overview", on_click=lambda: try_switch_page("pages/3_Faces_Review.py"))
            st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;Cast Members (12)", unsafe_allow_html=True)
            st.button("    Smart Suggestions", key="nav_suggestions", on_click=lambda: try_switch_page("pages/3_Smart_Suggestions.py"))
            st.button("Screentime & Health", key="nav_screentime", on_click=lambda: try_switch_page("pages/4_Screentime.py"))
            st.button("Docs Dashboard", key="nav_docs_dashboard", on_click=lambda: try_switch_page("pages/9_Docs_Dashboard.py"))


# =============================================================================
# Docs + Feature Coverage UI (read-only)
# =============================================================================

_DOCS_CATALOG_DEFAULT_RELATIVE_PATH = Path("docs") / "_meta" / "docs_catalog.json"


def _repo_root() -> Path:
    # apps/workspace-ui/ui_helpers.py -> apps/workspace-ui -> apps -> repo root
    return Path(__file__).resolve().parents[2]


def load_docs_catalog(catalog_path: str | Path | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Load docs catalog JSON used by docs dashboard + header popovers.

    Returns: (catalog, error). If error is not None, catalog is None.
    """
    repo_root = _repo_root()
    relative_path = Path(catalog_path) if catalog_path is not None else _DOCS_CATALOG_DEFAULT_RELATIVE_PATH
    catalog_file = (repo_root / relative_path).resolve()

    if not catalog_file.exists():
        return None, f"Docs catalog not found at `{relative_path}` (expected in repo root)."

    try:
        raw = catalog_file.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Failed to read docs catalog `{relative_path}`: {exc}"

    if not isinstance(data, dict):
        return None, "Docs catalog must be a JSON object."
    if not isinstance(data.get("docs"), list):
        return None, 'Docs catalog missing required key: "docs" (list).'
    if not isinstance(data.get("features"), dict):
        return None, 'Docs catalog missing required key: "features" (object).'

    return data, None


def _feature_present_in_ui(feature_id: str, catalog: dict[str, Any]) -> bool:
    docs = catalog.get("docs") or []
    if not isinstance(docs, list):
        return False
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if feature_id not in (doc.get("features") or []):
            continue
        surfaces = doc.get("ui_surfaces_expected") or []
        if any(isinstance(surface, str) and surface.startswith("workspace-ui:") for surface in surfaces):
            return True
    return False


def _feature_present_in_code(feature: dict[str, Any]) -> tuple[bool, list[str]]:
    repo_root = _repo_root()
    paths_expected = feature.get("paths_expected") or []
    missing: list[str] = []
    if not isinstance(paths_expected, list):
        return False, ["<invalid paths_expected>"]
    for expected in paths_expected:
        if not isinstance(expected, str) or not expected.strip():
            continue
        resolved = repo_root / expected
        if not resolved.exists():
            missing.append(expected)
    return len(missing) == 0, missing


def render_page_header(page_id: str, page_title: str) -> None:
    """Shared header with Docs/Features dialogs.

    Keep this function light-weight; it should be safe to call on every rerender.
    """
    title_col, features_col, todo_col = st.columns([6, 2, 2], vertical_alignment="center")
    with title_col:
        st.title(page_title)

    def _render_features_body() -> None:
        catalog, error = load_docs_catalog()
        if error:
            st.warning(error)
            st.caption("Merge the docs catalog PR, or add `docs/_meta/docs_catalog.json`.")
            return

        assert catalog is not None
        docs = [d for d in catalog.get("docs", []) if isinstance(d, dict)]
        relevant_docs = [
            d
            for d in docs
            if page_id in (d.get("ui_surfaces_expected") or [])
        ]

        features: set[str] = set()
        models: set[str] = set()
        jobs: set[str] = set()
        for doc in relevant_docs:
            features.update([f for f in (doc.get("features") or []) if isinstance(f, str)])
            models.update([m for m in (doc.get("models") or []) if isinstance(m, str)])
            jobs.update([j for j in (doc.get("jobs") or []) if isinstance(j, str)])

        if not relevant_docs:
            st.info("No docs mapping for this page yet (ui_surfaces_expected). Showing global feature catalog.")
            features = set([f for f in (catalog.get("features") or {}).keys() if isinstance(f, str)])

        st.markdown("### Features")
        feature_catalog = catalog.get("features") or {}
        if not isinstance(feature_catalog, dict):
            feature_catalog = {}

        for feature_id in sorted(features):
            feature = feature_catalog.get(feature_id) if isinstance(feature_catalog.get(feature_id), dict) else None
            if feature is None:
                st.markdown(f"- `{feature_id}` — status: `unknown` (missing from catalog)")
                continue

            title = feature.get("title") or feature_id
            status = feature.get("status") or "unknown"
            present_in_code, missing_paths = _feature_present_in_code(feature)
            present_in_ui = _feature_present_in_ui(feature_id, catalog)

            st.markdown(f"#### {title}")
            st.caption(
                f"status: `{status}` | present in code: `{present_in_code}` | present in UI: `{present_in_ui}`"
            )
            if missing_paths:
                st.warning("Missing paths: " + ", ".join(f"`{p}`" for p in missing_paths))

            phases = feature.get("phases") or {}
            if isinstance(phases, dict) and phases:
                st.markdown("**Phases**")
                for phase, phase_status in phases.items():
                    st.markdown(f"- `{phase}`: `{phase_status}`")

        if models:
            st.markdown("### Models")
            st.code("\n".join(sorted(models)))
        if jobs:
            st.markdown("### Jobs")
            st.code("\n".join(sorted(jobs)))

    def _extract_doc_implementation_hint(path_value: Any, *, max_chars: int = 220) -> str:
        """Best-effort 1-line hint describing what a TODO doc aims to change."""
        if not isinstance(path_value, str):
            return ""
        rel_path = path_value.strip().lstrip("/\\")
        if not rel_path:
            return ""
        repo_root = _repo_root()
        file_path = (repo_root / rel_path).resolve()
        try:
            raw = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""
        # Cap read size to keep UI snappy.
        raw = raw[:50_000]
        lines = [ln.strip() for ln in raw.splitlines()]

        # Skip leading frontmatter/title.
        i = 0
        if i < len(lines) and lines[i].startswith("---"):
            i += 1
            while i < len(lines) and not lines[i].startswith("---"):
                i += 1
            i += 1
        while i < len(lines) and (not lines[i] or lines[i].startswith("#")):
            i += 1

        # Prefer a short bullet list if present.
        bullets: list[str] = []
        for j in range(i, min(i + 80, len(lines))):
            ln = lines[j]
            if not ln:
                if bullets:
                    break
                continue
            if ln.startswith("#"):
                if bullets:
                    break
                continue
            if ln.startswith(("- ", "* ", "+ ")):
                bullets.append(ln[2:].strip())
                if len(bullets) >= 3:
                    break
        hint = " · ".join([b for b in bullets if b]) if bullets else ""

        # Fallback: first paragraph-ish line.
        if not hint:
            for j in range(i, min(i + 40, len(lines))):
                ln = lines[j]
                if not ln or ln.startswith("#"):
                    continue
                hint = ln
                break

        hint = " ".join(hint.split())
        if len(hint) > max_chars:
            hint = hint[: max_chars - 1].rstrip() + "…"
        return hint

    def _render_todo_body() -> None:
        catalog, error = load_docs_catalog()
        if error:
            st.warning(error)
            return

        assert catalog is not None
        docs = [d for d in catalog.get("docs", []) if isinstance(d, dict)]
        todo_statuses = {"in_progress", "draft", "outdated"}
        todo_docs = [
            d for d in docs if d.get("status") in todo_statuses
        ]
        todo_docs.sort(key=lambda d: (str(d.get("status")), str(d.get("title"))))

        if not todo_docs:
            st.info("No in-progress/draft/outdated docs found in catalog.")
            return

        feature_catalog = catalog.get("features") or {}
        if not isinstance(feature_catalog, dict):
            feature_catalog = {}

        def _feature_statuses_for(features: list[str]) -> str:
            parts: list[str] = []
            for fid in features:
                feature = feature_catalog.get(fid) if isinstance(feature_catalog.get(fid), dict) else None
                status = (feature or {}).get("status") if isinstance(feature, dict) else None
                parts.append(f"{fid}:{status or 'unknown'}")
            return ", ".join(parts)

        def _feature_pending_for(features: list[str]) -> str:
            pending: list[str] = []
            for fid in features:
                feature = feature_catalog.get(fid) if isinstance(feature_catalog.get(fid), dict) else None
                items = (feature or {}).get("pending") if isinstance(feature, dict) else None
                if isinstance(items, list):
                    pending.extend([str(it).strip() for it in items if isinstance(it, str) and it.strip()])
            deduped: list[str] = []
            for item in pending:
                if item not in deduped:
                    deduped.append(item)
            return "; ".join(deduped)

        def _paths_expected_for(features: list[str]) -> str:
            paths: list[str] = []
            for fid in features:
                feature = feature_catalog.get(fid) if isinstance(feature_catalog.get(fid), dict) else None
                items = (feature or {}).get("paths_expected") if isinstance(feature, dict) else None
                if isinstance(items, list):
                    paths.extend([str(it).strip() for it in items if isinstance(it, str) and it.strip()])
            deduped: list[str] = []
            for item in paths:
                if item not in deduped:
                    deduped.append(item)
            return ", ".join(deduped)

        rows: list[dict[str, Any]] = []
        for d in todo_docs:
            features_list = [f for f in (d.get("features") or []) if isinstance(f, str)]
            models_list = [m for m in (d.get("models") or []) if isinstance(m, str)]
            jobs_list = [j for j in (d.get("jobs") or []) if isinstance(j, str)]
            surfaces_list = [s for s in (d.get("ui_surfaces_expected") or []) if isinstance(s, str)]
            path_value = d.get("path", "")
            rows.append(
                {
                    "status": d.get("status", ""),
                    "type": d.get("type", ""),
                    "title": d.get("title", ""),
                    "last_updated": d.get("last_updated", ""),
                    "features": ", ".join(features_list),
                    "feature_status": _feature_statuses_for(features_list),
                    "expected_code_paths": _paths_expected_for(features_list),
                    "implementation_hint": _extract_doc_implementation_hint(path_value),
                    "pending": _feature_pending_for(features_list),
                    "models": ", ".join(models_list),
                    "jobs": ", ".join(jobs_list),
                    "ui_surfaces_expected": ", ".join(surfaces_list),
                    "path": path_value,
                }
            )

        st.dataframe(rows, use_container_width=True, hide_index=True)

    dialog = getattr(st, "dialog", None)
    if callable(dialog):

        @dialog("PAGE FEATURES", width="large")
        def _page_features_dialog() -> None:
            st.caption(f"Page: `{page_id}`")
            _render_features_body()

        @dialog("TO-DO", width="large")
        def _todo_dialog() -> None:
            _render_todo_body()

        with features_col:
            if st.button("PAGE FEATURES", key=f"{page_id}::page_features_dialog", use_container_width=True):
                _page_features_dialog()
        with todo_col:
            if st.button("TO-DO", key=f"{page_id}::todo_dialog", use_container_width=True):
                _todo_dialog()
        return

    # Fallback for older Streamlit versions without dialogs.
    with features_col:
        with st.expander("PAGE FEATURES", expanded=False):
            _render_features_body()
    with todo_col:
        with st.expander("TO-DO", expanded=False):
            _render_todo_body()


def format_mmss(seconds: float | int | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def progress_ratio(progress: Dict[str, Any]) -> float:
    frames_total = progress.get("frames_total") or 0
    frames_done = progress.get("frames_done") or 0
    if frames_total and frames_total > 0:
        return max(min(frames_done / frames_total, 1.0), 0.0)
    return 0.0


def eta_seconds(progress: Dict[str, Any]) -> float | None:
    secs_total = progress.get("secs_total")
    secs_done = progress.get("secs_done")
    if secs_total is not None and secs_done is not None:
        remaining = max(secs_total - secs_done, 0.0)
        return remaining
    frames_total = progress.get("frames_total")
    frames_done = progress.get("frames_done")
    fps = progress.get("fps_infer") or progress.get("fps_detected")
    if frames_total and frames_done is not None and fps and fps > 0:
        remaining_frames = max(frames_total - frames_done, 0)
        return remaining_frames / fps if remaining_frames >= 0 else None
    return None


def total_seconds_hint(progress: Dict[str, Any]) -> float | None:
    secs_total = progress.get("secs_total")
    if secs_total is not None:
        return secs_total
    frames_total = progress.get("frames_total")
    fps = progress.get("fps_infer") or progress.get("analyzed_fps") or progress.get("fps_detected")
    if frames_total and fps and fps > 0:
        return frames_total / fps
    return None


def iter_sse_events(
    response: requests.Response,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    event_name = "message"
    data_lines: List[str] = []
    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.strip()
            if not line:
                if data_lines:
                    data_str = "\n".join(data_lines)
                    try:
                        payload = json.loads(data_str)
                    except json.JSONDecodeError:
                        payload = {"raw": data_str}
                    yield event_name or "message", payload  # type: ignore[misc]
                event_name = "message"
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
    finally:
        response.close()


def episode_artifact_prefixes(ep_id: str) -> Dict[str, str] | None:
    parsed = parse_ep_id(ep_id)
    if not parsed:
        return None
    show = parsed["show"]
    season = int(parsed["season"])  # type: ignore[arg-type]
    episode = int(parsed["episode"])  # type: ignore[arg-type]
    return {
        "frames": f"artifacts/frames/{show}/s{season:02d}/e{episode:02d}/frames/",
        "crops": f"artifacts/crops/{show}/s{season:02d}/e{episode:02d}/tracks/",
        "manifests": f"artifacts/manifests/{show}/s{season:02d}/e{episode:02d}/",
    }


def update_progress_display(
    progress: Dict[str, Any],
    *,
    progress_bar,
    status_placeholder,
    detail_placeholder,
    requested_device: str,
    requested_detector: str | None,
    requested_tracker: str | None,
) -> tuple[str, str]:
    ratio = progress_ratio(progress)
    progress_bar.progress(ratio)
    status_line, frames_line = compose_progress_text(
        progress,
        requested_device=requested_device,
        requested_detector=requested_detector,
        requested_tracker=requested_tracker,
    )
    status_placeholder.write(status_line)
    detail_placeholder.caption(frames_line)
    return status_line, frames_line


def compose_progress_text(
    progress: Dict[str, Any],
    *,
    requested_device: str,
    requested_detector: str | None,
    requested_tracker: str | None,
) -> tuple[str, str]:
    video_time = progress.get("video_time")
    video_total = progress.get("video_total")
    phase = progress.get("phase") or "detect"
    device_label = progress.get("device") or requested_device or "--"
    resolved_detector_device = progress.get("resolved_device")
    device_text = device_label
    if resolved_detector_device and resolved_detector_device != device_label:
        device_text = f"{device_label} (detector={resolved_detector_device})"
    raw_detector = progress.get("detector") or requested_detector
    detector_label = detector_label_from_value(raw_detector) if raw_detector else "--"
    raw_tracker = progress.get("tracker") or requested_tracker
    tracker_label = tracker_label_from_value(raw_tracker) if raw_tracker else "--"
    fps_value = progress.get("fps_infer") or progress.get("analyzed_fps") or progress.get("fps_detected")
    fps_text = f"{fps_value:.2f} fps" if fps_value else "--"
    time_prefix = ""
    if video_time is not None and video_total is not None:
        try:
            done_value = float(video_time)
            total_value = float(video_total)
        except (TypeError, ValueError):
            pass
        else:
            done_value = min(done_value, total_value)
            time_prefix = f"{format_mmss(done_value)} / {format_mmss(total_value)} • "
    status_line = (
        f"{time_prefix}"
        f"phase={phase} • detector={detector_label} • tracker={tracker_label} • device={device_text} • fps={fps_text}"
    )
    frames_line = f"Frames {progress.get('frames_done', 0):,} / {progress.get('frames_total') or '?'}"
    return status_line, frames_line


def _is_phase_done(progress: Dict[str, Any]) -> bool:
    phase = str(progress.get("phase", "")).lower()
    if phase == "done":
        return True
    step = str(progress.get("step", "")).lower()
    if not step:
        return False
    if phase.startswith("scene_detect"):
        return False
    if phase == "detect":
        return False
    return step == "done"


def _phase_from_endpoint(endpoint_path: str | None) -> str | None:
    if not endpoint_path:
        return None
    lowered = endpoint_path.lower()
    if "faces_embed" in lowered:
        return "faces_embed"
    if "cluster" in lowered:
        return "cluster"
    return None


def _fetch_async_job_error(ep_id: str, phase: str) -> str | None:
    """
    Fetch the most recent async job record for the given episode and phase
    to extract error details when progress file is missing.
    """
    try:
        resp = requests.get(
            f"{_api_base()}/jobs",
            params={"ep_id": ep_id, "job_type": phase},
            timeout=5,
        )
        resp.raise_for_status()
        jobs = resp.json()
        if not isinstance(jobs, list) or not jobs:
            return None
        # Get most recent job (jobs are typically sorted by creation time)
        job = jobs[0]
        if isinstance(job, dict):
            error = job.get("error")
            if error:
                return f"Job failed: {error}"
            state = job.get("state")
            if state == "failed":
                stderr_log = job.get("stderr_log")
                if stderr_log:
                    return f"Job failed (check stderr log: {stderr_log})"
                return "Job failed during initialization (no error details available)"
    except Exception:
        # Don't fail if we can't fetch job details, just return None
        pass
    return None


def _summary_from_status(ep_id: str, phase: str, *, run_id: str | None = None) -> Dict[str, Any] | None:
    payload = _episode_status_payload(ep_id, run_id=run_id)
    if not payload:
        return None

    # For detect_track, count files directly since status API doesn't include it
    if phase == "detect_track":
        from py_screenalytics.artifacts import get_path
        from py_screenalytics import run_layout

        run_id_value = run_id.strip() if isinstance(run_id, str) else ""
        run_id_norm: str | None = None
        if run_id_value:
            try:
                run_id_norm = run_layout.normalize_run_id(run_id_value)
            except ValueError:
                run_id_norm = None

        if run_id_norm:
            manifests_dir = run_layout.run_root(ep_id, run_id_norm)
            det_path = manifests_dir / "detections.jsonl"
            track_path = manifests_dir / "tracks.jsonl"
        else:
            det_path = get_path(ep_id, "detections")
            track_path = get_path(ep_id, "tracks")
        if det_path.exists() and track_path.exists():
            det_count = sum(1 for line in det_path.open("r", encoding="utf-8") if line.strip())
            track_count = sum(1 for line in track_path.open("r", encoding="utf-8") if line.strip())
            return {
                "stage": "detect_track",
                "detections": det_count,
                "tracks": track_count,
            }
        return None

    phase_block = payload.get(phase)
    if not isinstance(phase_block, dict):
        return None
    status_value = str(phase_block.get("status", "")).lower()
    if status_value != "success":
        return None
    summary: Dict[str, Any] = {"stage": phase}
    stats: Dict[str, Any] = {}
    faces_value = phase_block.get("faces")
    if isinstance(faces_value, int):
        stats.setdefault("faces", faces_value)
        key = "faces" if phase == "faces_embed" else "faces_count"
        summary[key] = faces_value
    identities_value = phase_block.get("identities")
    if isinstance(identities_value, int):
        summary["identities_count"] = identities_value
        stats.setdefault("clusters", identities_value)
    if stats:
        summary["stats"] = stats
    return summary


def _is_complete_summary(summary: Dict[str, Any] | None) -> bool:
    if not isinstance(summary, dict):
        return False
    if isinstance(summary.get("stage"), str) and summary["stage"].strip():
        return True
    numeric_keys = (
        "detections",
        "tracks",
        "faces",
        "faces_count",
        "identities",
        "identities_count",
    )
    for key in numeric_keys:
        if coerce_int(summary.get(key)) is not None:
            return True
    nested = summary.get("summary")
    if isinstance(nested, dict):
        return _is_complete_summary(nested)
    return False


def attempt_sse_run(
    endpoint_path: str,
    payload: Dict[str, Any],
    *,
    update_cb,
) -> tuple[Dict[str, Any] | None, str | None, bool]:
    url = f"{_api_base()}{endpoint_path}"
    headers = {"Accept": "text/event-stream"}
    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=(5, 300))
        response.raise_for_status()
    except requests.RequestException as exc:
        return None, describe_error(url, exc), False

    content_type = (response.headers.get("Content-Type") or "").lower()
    if "text/event-stream" not in content_type:
        try:
            body = response.json()
        except ValueError as exc:  # pragma: no cover - unexpected
            return None, f"Unexpected response from {endpoint_path}: {exc}", False
        summary = body if isinstance(body, dict) else {"raw": body}
        return summary, None, False

    final_summary: Dict[str, Any] | None = None
    try:
        for event_name, event_payload in iter_sse_events(response):
            if not isinstance(event_payload, dict):
                continue
            update_cb(event_payload)
            summary_candidate = event_payload.get("summary")
            if isinstance(summary_candidate, dict) and _is_complete_summary(summary_candidate):
                final_summary = summary_candidate
            phase = str(event_payload.get("phase", "")).lower()
            if event_name == "error" or phase == "error":
                return None, event_payload.get("error") or "Job failed", True
            if event_name == "done" or _is_phase_done(event_payload):
                if final_summary:
                    return final_summary, None, True
                if isinstance(summary_candidate, dict) and _is_complete_summary(summary_candidate):
                    return summary_candidate, None, True
                return None, None, True
    finally:
        response.close()
    if final_summary:
        return final_summary, None, True
    ep_id_value = payload.get("ep_id")
    if isinstance(ep_id_value, str):
        phase_hint = _phase_from_endpoint(endpoint_path) or "detect_track"
        status_summary = _summary_from_status(ep_id_value, phase_hint)
        if status_summary:
            return status_summary, None, True
    return None, None, True


def fallback_poll_progress(
    ep_id: str,
    payload: Dict[str, Any],
    *,
    update_cb,
    status_placeholder,
    job_started: bool,
    async_endpoint: str,
) -> tuple[Dict[str, Any] | None, str | None]:
    if not job_started:
        try:
            requests.post(f"{_api_base()}{async_endpoint}", json=payload, timeout=30).raise_for_status()
        except requests.RequestException as exc:
            return None, describe_error(f"{_api_base()}{async_endpoint}", exc)

    progress_url = f"{_api_base()}/episodes/{ep_id}/progress"
    phase_hint = _phase_from_endpoint(async_endpoint) or "detect_track"
    last_progress: Dict[str, Any] | None = None
    poll_attempts = 0
    max_poll_attempts = 60  # 30 seconds (60 attempts * 0.5s sleep)
    while True:
        try:
            resp = requests.get(progress_url, timeout=5)
            if resp.status_code == 404:
                poll_attempts += 1
                if poll_attempts > max_poll_attempts:
                    # Timeout: progress file never appeared, job likely failed during init
                    # Fetch job record to surface actual error
                    job_error = _fetch_async_job_error(ep_id, phase_hint)
                    return (
                        None,
                        job_error
                        or f"Job initialization timed out after {max_poll_attempts * 0.5:.0f}s (progress file never created)",
                    )
                status_placeholder.info("initializing...")
                time.sleep(0.5)
                continue
            resp.raise_for_status()
        except requests.RequestException as exc:
            return None, describe_error(progress_url, exc)
        payload_body = resp.json()
        progress = payload_body.get("progress") or payload_body
        if not isinstance(progress, dict):
            time.sleep(0.5)
            continue
        update_cb(progress)
        last_progress = progress
        phase = str(progress.get("phase", "")).lower()
        if phase == "error":
            return None, progress.get("error") or "Job failed"
        if _is_phase_done(progress):
            summary_block = progress.get("summary")
            if isinstance(summary_block, dict):
                return summary_block, None
            status_placeholder.info("Async job finished without summary; using latest status metrics …")
            status_summary = _summary_from_status(ep_id, phase_hint)
            if status_summary:
                return status_summary, None
            if last_progress:
                return last_progress, None
            return progress, None
        time.sleep(0.5)


def normalize_summary(ep_id: str, raw: Dict[str, Any] | None) -> Dict[str, Any]:
    summary = raw or {}
    if "summary" in summary and isinstance(summary["summary"], dict):
        summary = summary["summary"]
    result_block = summary.get("result") if isinstance(summary.get("result"), dict) else None
    artifacts = summary.setdefault("artifacts", {})
    local = artifacts.setdefault("local", {})
    manifests_dir = DATA_ROOT / "manifests" / ep_id
    local.setdefault("detections", str(manifests_dir / "detections.jsonl"))
    local.setdefault("tracks", str(manifests_dir / "tracks.jsonl"))
    local.setdefault("faces", str(manifests_dir / "faces.jsonl"))
    local.setdefault("identities", str(manifests_dir / "identities.json"))
    counts_candidates = [summary]
    if result_block:
        counts_candidates.append(result_block)
        counts = result_block.get("counts") if isinstance(result_block.get("counts"), dict) else None
        if counts:
            counts_candidates.append(counts)
    for key in ("detections", "tracks", "faces", "identities"):
        if key in summary and summary[key] is not None:
            continue
        value = None
        for candidate in counts_candidates:
            if key in candidate and candidate.get(key) is not None:
                value = candidate.get(key)
                break
            count_key = f"{key}_count"
            if candidate.get(count_key) is not None:
                value = candidate.get(count_key)
                break
        if value is not None:
            summary[key] = value
    # Final local-manifest fallback for counts
    # If a job returns a summary "stage" without counts, infer from artifacts.
    try:
        if summary.get("detections") is None and local.get("detections"):
            det_path = Path(str(local["detections"]))
            if det_path.exists():
                with det_path.open("r", encoding="utf-8") as fh:
                    summary["detections"] = sum(1 for line in fh if line.strip())
        if summary.get("tracks") is None and local.get("tracks"):
            trk_path = Path(str(local["tracks"]))
            if trk_path.exists():
                with trk_path.open("r", encoding="utf-8") as fh:
                    summary["tracks"] = sum(1 for line in fh if line.strip())
        # Fallback for faces count (critical for auto-run faces→cluster transition)
        if summary.get("faces") is None and local.get("faces"):
            faces_path = Path(str(local["faces"]))
            if faces_path.exists():
                with faces_path.open("r", encoding="utf-8") as fh:
                    summary["faces"] = sum(1 for line in fh if line.strip())
        # Fallback for identities count (for cluster phase completion)
        if summary.get("identities") is None and local.get("identities"):
            identities_path = Path(str(local["identities"]))
            if identities_path.exists():
                try:
                    payload = json.loads(identities_path.read_text(encoding="utf-8"))
                    identities_list = payload.get("identities") if isinstance(payload, dict) else None
                    if isinstance(identities_list, list):
                        summary["identities"] = len(identities_list)
                except (json.JSONDecodeError, KeyError):
                    pass
    except OSError:
        pass
    return summary


def scene_cuts_badge_text(summary: Dict[str, Any] | None) -> str | None:
    scene_block: Dict[str, Any] | None = None
    if isinstance(summary, dict):
        scene_field = summary.get("scene_cuts")
        if isinstance(scene_field, dict):
            scene_block = scene_field
        else:
            scene_block = summary if "count" in summary and "scene_cuts" not in summary else None
    if not scene_block:
        return None
    count = scene_block.get("count")
    detector_name: str | None = None
    detector_value = scene_block.get("detector")
    if isinstance(detector_value, str):
        if detector_value == "off":
            detector_name = "disabled"
        else:
            detector_name = scene_detector_label(detector_value)
    if isinstance(count, int):
        prefix = f"Scene cuts: {count:,}"
        if detector_name:
            if detector_name == "disabled":
                return "Scene cuts: disabled"
            return f"{prefix} via {detector_name}"
        return prefix
    return None


def letterbox_thumb_url(url: str | None, size: int = 256) -> str | None:
    """Placeholder hook for thumbnail sizing (S3 URLs already return square crops)."""

    return url


def identity_card(title: str, subtitle: str, image_url: str | None, extra=None):
    card = st.container(border=True)
    with card:
        if image_url:
            st.image(image_url, use_container_width=True)
        st.markdown(f"**{title}**")
        if subtitle:
            st.caption(subtitle)
        if extra:
            extra()
    return card


def track_card(title: str, caption: str, image_url: str | None, extra=None):
    card = st.container(border=True)
    with card:
        if image_url:
            st.image(image_url, use_container_width=True)
        st.markdown(f"**{title}**")
        st.caption(caption)
        if extra:
            extra()
    return card


def frame_card(title: str, image_url: str | None, extra=None):
    card = st.container(border=True)
    with card:
        if image_url:
            st.image(image_url, use_container_width=True)
        st.caption(title)
        if extra:
            extra()
    return card


def s3_uri(prefix: str | None, bucket: str | None = None) -> str:
    if not prefix:
        return ""
    bucket_name = bucket or st.session_state.get("bucket") or "screenalytics"
    cleaned = prefix.lstrip("/")
    if not bucket_name:
        return cleaned
    return f"s3://{bucket_name}/{cleaned}"


def run_job_with_progress(
    ep_id: str,
    endpoint_path: str,
    payload: Dict[str, Any],
    *,
    requested_device: str,
    async_endpoint: str | None = None,
    requested_detector: str | None = None,
    requested_tracker: str | None = None,
    use_async_only: bool = False,
):
    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()
    detail_placeholder = st.empty()
    log_expander = st.expander("Detailed log", expanded=False)
    with log_expander:
        log_placeholder = st.empty()
    log_lines: List[str] = []
    max_log_lines = 25
    last_logged_frame: int | None = None
    last_logged_phase: str | None = None
    frame_log_interval = 100
    events_seen = False

    def _mode_context() -> str:
        parts: List[str] = []
        if requested_detector:
            parts.append(f"detector={requested_detector}")
        if requested_tracker:
            parts.append(f"tracker={requested_tracker}")
        if requested_device:
            parts.append(f"device={requested_device}")
        context = ", ".join(parts) if parts else f"device={requested_device}"
        return context

    def _append_log(entry: str) -> None:
        log_lines.append(entry)
        if len(log_lines) > max_log_lines:
            del log_lines[0 : len(log_lines) - max_log_lines]
        log_placeholder.code("\n\n".join(log_lines), language="text")

    dedupe_key = f"last_progress_event::{endpoint_path}"
    run_id_key = f"current_run_id::{endpoint_path}"
    st.session_state.pop(dedupe_key, None)
    phase_hint = _phase_from_endpoint(endpoint_path) or "detect_track"

    _append_log(f"Starting request to {endpoint_path} ({_mode_context()})…")

    def _cb(progress: Dict[str, Any]) -> None:
        nonlocal last_logged_frame, last_logged_phase, events_seen
        if not isinstance(progress, dict):
            return
        events_seen = True

        # Honor run_id: ignore events from stale runs
        event_run_id = progress.get("run_id")
        if event_run_id:
            current_run_id = st.session_state.get(run_id_key)
            if current_run_id and current_run_id != event_run_id:
                # Stale event from a previous run, ignore it
                return
            if not current_run_id:
                # First event from this run, record it
                st.session_state[run_id_key] = event_run_id

        event_key = (
            progress.get("phase"),
            progress.get("frames_done"),
            progress.get("step"),
            event_run_id,
        )
        if st.session_state.get(dedupe_key) == event_key:
            return
        st.session_state[dedupe_key] = event_key

        # Clamp video_time to video_total on UI side as extra safety
        video_time = progress.get("video_time")
        video_total = progress.get("video_total")
        if video_time is not None and video_total is not None and video_time > video_total:
            progress = progress.copy()
            progress["video_time"] = video_total
        if progress.get("detector"):
            remember_detector(progress.get("detector"))
        current_phase = str(progress.get("phase") or "")
        if current_phase.lower() == "log":
            stream_label = progress.get("stream") or "stdout"
            message = progress.get("message")
            if message:
                _append_log(f"[{stream_label}] {message}")
            return
        status_line, frames_line = update_progress_display(
            progress,
            progress_bar=progress_bar,
            status_placeholder=status_placeholder,
            detail_placeholder=detail_placeholder,
            requested_device=requested_device,
            requested_detector=requested_detector,
            requested_tracker=requested_tracker,
        )
        frames_done_val = coerce_int(progress.get("frames_done"))
        log_event = False
        if last_logged_phase is None or current_phase != last_logged_phase:
            log_event = True
        elif frames_done_val is not None:
            previous = last_logged_frame or 0
            if frames_done_val - previous >= frame_log_interval:
                log_event = True
        elif last_logged_frame is None:
            log_event = True
        if log_event:
            _append_log(f"{status_line}\n{frames_line}")
            last_logged_phase = current_phase
            if frames_done_val is not None:
                last_logged_frame = frames_done_val

    summary = None
    error_message = None
    try:
        if use_async_only:
            target_endpoint = async_endpoint or endpoint_path
            _append_log(
                f"Using async endpoint {target_endpoint} immediately ({_mode_context()}); polling for progress…"
            )
            summary, error_message = fallback_poll_progress(
                ep_id,
                payload,
                update_cb=_cb,
                status_placeholder=status_placeholder,
                job_started=False,
                async_endpoint=target_endpoint,
            )
        else:
            summary, error_message, job_started = attempt_sse_run(endpoint_path, payload, update_cb=_cb)
            if not events_seen:
                if error_message:
                    _append_log(f"Request failed before any realtime updates ({_mode_context()}): {error_message}")
                elif summary:
                    _append_log(f"Server returned a synchronous summary before streaming updates ({_mode_context()}).")
                else:
                    fallback_hint = f"; checking async fallback via {async_endpoint}" if async_endpoint else ""
                    _append_log(
                        f"No realtime events received yet from {endpoint_path} ({_mode_context()}){fallback_hint}."
                    )
            if async_endpoint and (summary is None or error_message):
                _append_log(f"Falling back to async endpoint {async_endpoint} for {endpoint_path} ({_mode_context()})…")
                fallback_summary, fallback_error = fallback_poll_progress(
                    ep_id,
                    payload,
                    update_cb=_cb,
                    status_placeholder=status_placeholder,
                    job_started=job_started,
                    async_endpoint=async_endpoint,
                )
                if fallback_summary is not None or fallback_error is not None:
                    summary = fallback_summary
                    error_message = fallback_error
                if fallback_error:
                    _append_log(
                        f"Async endpoint {async_endpoint} reported an error ({_mode_context()}): {fallback_error}"
                    )
                elif fallback_summary:
                    _append_log(f"Async endpoint {async_endpoint} returned a summary ({_mode_context()}).")
        if summary and isinstance(summary, dict):
            remember_detector(summary.get("detector"))
            remember_tracker(summary.get("tracker"))
        if summary is None and not error_message and phase_hint:
            status_summary = _summary_from_status(ep_id, phase_hint)
            if status_summary:
                summary = status_summary
        return summary, error_message
    finally:
        st.session_state.pop(dedupe_key, None)
        st.session_state.pop(run_id_key, None)


# =============================================================================
# Celery Job Polling (for async pipeline jobs via Redis queue)
# =============================================================================


def run_celery_job_with_progress(
    ep_id: str,
    operation: str,
    payload: Dict[str, Any],
    *,
    requested_device: str = "auto",
    requested_detector: str | None = None,
    requested_tracker: str | None = None,
):
    """Run a job via Celery and poll for completion.

    Uses the /celery_jobs/* endpoints for true async execution via Redis.

    Args:
        ep_id: Episode ID
        operation: One of "detect_track", "faces_embed", "cluster"
        payload: Request payload for the job
        requested_device: Device string for context display
        requested_detector: Detector string for context display
        requested_tracker: Tracker string for context display

    Returns:
        Tuple of (summary_dict, error_message)
    """
    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()
    detail_placeholder = st.empty()
    log_expander = st.expander("Detailed log", expanded=False)
    with log_expander:
        log_placeholder = st.empty()
    log_lines: List[str] = []
    max_log_lines = 25

    def _mode_context() -> str:
        parts: List[str] = []
        if requested_detector:
            parts.append(f"detector={requested_detector}")
        if requested_tracker:
            parts.append(f"tracker={requested_tracker}")
        if requested_device:
            parts.append(f"device={requested_device}")
        return ", ".join(parts) if parts else f"device={requested_device}"

    def _append_log(entry: str) -> None:
        log_lines.append(entry)
        if len(log_lines) > max_log_lines:
            del log_lines[0 : len(log_lines) - max_log_lines]
        log_placeholder.code("\n\n".join(log_lines), language="text")

    # Map operation to endpoint
    endpoint_map = {
        "detect_track": "/celery_jobs/detect_track",
        "faces_embed": "/celery_jobs/faces_embed",
        "cluster": "/celery_jobs/cluster",
    }
    endpoint = endpoint_map.get(operation)
    if not endpoint:
        return None, f"Unknown operation: {operation}"

    _append_log(f"Starting {operation} via Celery ({_mode_context()})...")

    # Start the Celery job
    try:
        resp = requests.post(f"{_api_base()}{endpoint}", json=payload, timeout=30)
        resp.raise_for_status()
        start_result = resp.json()
    except requests.RequestException as exc:
        error_msg = describe_error(f"{_api_base()}{endpoint}", exc)
        _append_log(f"Failed to start job: {error_msg}")
        return None, error_msg

    job_id = start_result.get("job_id")
    if not job_id:
        _append_log("Error: No job_id returned from Celery endpoint")
        return None, "No job_id returned"

    # Store job_id in session state for running job detection
    store_celery_job_id(ep_id, operation, job_id)

    if start_result.get("state") == "already_running":
        _append_log(f"Job {job_id} is already running for this episode")
        status_placeholder.info(f"⏳ {operation} job is already running...")
        # Fall through to poll for completion

    _append_log(f"Job {job_id} queued. Polling for progress...")
    status_placeholder.info(f"⏳ {operation} job queued via Celery...")

    # Poll for completion
    poll_url = f"{_api_base()}/celery_jobs/{job_id}"
    poll_interval = 2.0  # seconds
    max_poll_time = 3600  # 1 hour max
    elapsed = 0.0
    queued_since: float | None = None
    last_log_message: str | None = None  # Track last logged message to avoid duplicates

    while elapsed < max_poll_time:
        time.sleep(poll_interval)
        elapsed += poll_interval

        try:
            poll_resp = requests.get(poll_url, timeout=10)
            poll_resp.raise_for_status()
            status_data = poll_resp.json()
        except requests.RequestException as exc:
            _append_log(f"Poll error: {exc}")
            continue

        state = status_data.get("state", "unknown")
        raw_state = status_data.get("raw_state", "")
        progress_info = status_data.get("progress") or {}

        if state == "success":
            progress_bar.progress(1.0)
            result = status_data.get("result", {})
            result_status = result.get("status", "success")
            _append_log(f"Job completed: {result_status}")
            detail_placeholder.caption(f"Completed in {elapsed:.1f}s")
            # Clear job_id from session state - job is done
            clear_celery_job_id(ep_id, operation)

            # Check if the subprocess actually succeeded
            if result_status == "error":
                error_msg = result.get("error", "Subprocess failed")
                status_placeholder.error(f"❌ {operation} failed: {error_msg}")
                _append_log(f"Subprocess error: {error_msg}")
                # Include any stdout/stderr for debugging
                stdout = result.get("stdout")
                if stdout:
                    _append_log(f"Output: {stdout[:500]}")
                return result.get("progress"), error_msg

            status_placeholder.success(f"✅ {operation} completed successfully")

            # Read progress file for detailed results
            progress_data = result.get("progress")
            if progress_data:
                _append_log(f"Progress data: {progress_data.get('phase', 'done')}")
                return progress_data, None

            return result, None

        elif state == "failed":
            progress_bar.progress(1.0)
            error = status_data.get("error", "Unknown error")
            status_placeholder.error(f"❌ {operation} failed: {error}")
            _append_log(f"Job failed: {error}")
            # Clear job_id from session state - job is done
            clear_celery_job_id(ep_id, operation)
            return None, error

        elif state == "cancelled":
            progress_bar.progress(1.0)
            status_placeholder.warning(f"⚠️ {operation} was cancelled")
            _append_log("Job was cancelled")
            # Clear job_id from session state - job is done
            clear_celery_job_id(ep_id, operation)
            return None, "Job cancelled"

        elif state == "in_progress":
            # Update progress display
            progress_pct = progress_info.get("progress", 0.0)
            message = progress_info.get("message", f"Running {operation}...")
            step = progress_info.get("step", operation)

            progress_bar.progress(min(progress_pct, 1.0))
            status_placeholder.info(f"⏳ {step}: {message}")
            # Only log when progress actually changes
            log_msg = f"Progress: {progress_pct*100:.1f}% - {message}"
            if log_msg != last_log_message:
                _append_log(log_msg)
                last_log_message = log_msg

        else:
            # queued, retrying, or unknown
            status_placeholder.info(f"⏳ {operation}: {state} ({raw_state})")
            default_msg = (
                progress_info.get("message")
                or status_data.get("message")
                or f"Starting {operation} pipeline..."
            )
            # Only log when message changes
            log_msg = f"Progress: 0.0% - {default_msg}"
            if log_msg != last_log_message:
                _append_log(log_msg)
                last_log_message = log_msg
            # Detect stuck queued jobs (common when Celery worker or Redis is down)
            if state in {"queued", "unknown"}:
                queued_since = queued_since or time.time()
                if time.time() - queued_since > 15:
                    error_msg = (
                        f"{operation} has been queued for over 15s without starting. "
                        "Check that the Celery worker and Redis are running. "
                        "Run 'scripts/dev_auto.sh' or start Redis/Celery manually."
                    )
                    status_placeholder.error(f"⚠️ {error_msg}")
                    _append_log(error_msg)
                    return None, error_msg

    # Timeout - clear job_id from session state
    clear_celery_job_id(ep_id, operation)
    _append_log(f"Job timed out after {max_poll_time}s")
    return None, f"Job timed out after {max_poll_time}s"


def start_celery_job_async(
    operation: str,
    payload: Dict[str, Any],
) -> tuple[str | None, str | None]:
    """Start a Celery job without waiting for completion.

    Returns (job_id, error_message).
    """
    endpoint_map = {
        "detect_track": "/celery_jobs/detect_track",
        "faces_embed": "/celery_jobs/faces_embed",
        "cluster": "/celery_jobs/cluster",
    }
    endpoint = endpoint_map.get(operation)
    if not endpoint:
        return None, f"Unknown operation: {operation}"

    try:
        resp = requests.post(f"{_api_base()}{endpoint}", json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        return result.get("job_id"), None
    except requests.RequestException as exc:
        return None, describe_error(f"{_api_base()}{endpoint}", exc)


def check_celery_job_status(job_id: str) -> Dict[str, Any] | None:
    """Check the status of a Celery job.

    Returns status dict or None if error.
    """
    try:
        resp = requests.get(f"{_api_base()}/celery_jobs/{job_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def _celery_job_session_key(ep_id: str, job_type: str) -> str:
    """Get session state key for tracking a Celery job."""
    return f"{ep_id}::celery_job::{job_type}"


def store_celery_job_id(ep_id: str, job_type: str, job_id: str) -> None:
    """Store a Celery job_id in session state for tracking."""
    import streamlit as st

    key = _celery_job_session_key(ep_id, job_type)
    st.session_state[key] = job_id


def clear_celery_job_id(ep_id: str, job_type: str) -> None:
    """Clear a stored Celery job_id from session state."""
    import streamlit as st

    key = _celery_job_session_key(ep_id, job_type)
    if key in st.session_state:
        del st.session_state[key]


def get_stored_celery_job_id(ep_id: str, job_type: str) -> str | None:
    """Get a stored Celery job_id from session state."""
    import streamlit as st

    key = _celery_job_session_key(ep_id, job_type)
    return st.session_state.get(key)


# Cache for batch job fetches to avoid repeated API calls on page load
# Suggestion 2: Use longer TTL during active jobs (10s) vs idle (3s)
_running_jobs_cache: Dict[str, Tuple[float, Dict[str, Dict[str, Any] | None]]] = {}
_RUNNING_JOBS_CACHE_TTL_ACTIVE = 8.0  # 8 second TTL during active jobs
_RUNNING_JOBS_CACHE_TTL_IDLE = 2.0  # 2 second TTL when idle


def get_all_running_jobs_for_episode(ep_id: str) -> Dict[str, Dict[str, Any] | None]:
    """Fetch all running jobs for an episode in a single batch.

    This is much faster than calling get_running_job_for_episode 4 times
    since it reuses API responses across job types.

    Returns:
        Dict mapping job_type -> job_info or None
    """
    import time as _time

    cache_key = ep_id
    now = _time.time()

    # Check cache with adaptive TTL (longer during active jobs)
    if cache_key in _running_jobs_cache:
        cached_time, cached_data = _running_jobs_cache[cache_key]
        # Use longer TTL if any job was found in last fetch
        has_active_jobs = any(v is not None for v in cached_data.values())
        ttl = _RUNNING_JOBS_CACHE_TTL_ACTIVE if has_active_jobs else _RUNNING_JOBS_CACHE_TTL_IDLE
        if now - cached_time < ttl:
            return cached_data

    running_states = {"running", "in_progress", "queued", "started", "pending", "scheduled", "retrying"}
    api_timeout = 2  # Reduced timeout for faster response
    job_types = ["detect_track", "faces_embed", "cluster", "audio_pipeline"]
    results: Dict[str, Dict[str, Any] | None] = {jt: None for jt in job_types}

    # Fetch local jobs once (covers all job types)
    local_jobs = []
    try:
        resp = requests.get(
            f"{_api_base()}/celery_jobs/local",
            params={"ep_id": ep_id},
            timeout=api_timeout,
        )
        resp.raise_for_status()
        local_jobs = resp.json().get("jobs", [])
    except requests.RequestException:
        pass

    # Process local jobs
    progress_data = get_episode_progress(ep_id)
    for job in local_jobs:
        op = job.get("operation", "")
        if op in job_types:
            state = str(job.get("state", "")).lower()
            if state in running_states:
                result = {
                    "job_id": job.get("job_id"),
                    "state": state,
                    "started_at": job.get("started_at"),
                    "job_type": op,
                    "source": "celery_local",
                    "pid": job.get("pid"),
                }
                if progress_data:
                    frames_done = progress_data.get("frames_done", 0)
                    frames_total = progress_data.get("frames_total", 1)
                    if frames_total > 0:
                        result["progress_pct"] = (frames_done / frames_total) * 100
                    result["message"] = f"Phase: {progress_data.get('phase', 'unknown')}"
                    result["frames_done"] = frames_done
                    result["frames_total"] = frames_total
                results[op] = result

    # Only check other sources if local didn't find everything
    unfound = [jt for jt in job_types if results[jt] is None]
    if unfound:
        # Fetch Celery jobs once
        try:
            resp = requests.get(f"{_api_base()}/celery_jobs", timeout=api_timeout)
            resp.raise_for_status()
            celery_jobs = resp.json().get("jobs", [])

            for job in celery_jobs:
                job_ep_id = job.get("ep_id")
                op = job.get("operation") or job.get("name", "").replace("local_", "")
                state = str(job.get("state", "")).lower()

                if job_ep_id == ep_id and op in unfound and state in running_states:
                    results[op] = {
                        "job_id": job.get("job_id"),
                        "state": state,
                        "started_at": job.get("started_at"),
                        "job_type": op,
                        "source": "celery_active",
                    }
        except requests.RequestException:
            pass

    # Cache the results
    _running_jobs_cache[cache_key] = (now, results)
    return results


def invalidate_running_jobs_cache(ep_id: str | None = None) -> None:
    """Invalidate the running jobs cache, optionally for a specific episode."""
    if ep_id:
        _running_jobs_cache.pop(ep_id, None)
    else:
        _running_jobs_cache.clear()


def get_running_job_for_episode(ep_id: str, job_type: str) -> Dict[str, Any] | None:
    """Check if there's a running job of a specific type for an episode.

    Args:
        ep_id: Episode identifier
        job_type: One of "detect_track", "faces_embed", "cluster", "audio_pipeline"

    Returns:
        Job info dict with progress if running, None otherwise.
        Dict includes: job_id, state, progress_pct, message, started_at

    Uses batch fetching internally for better performance when called multiple times.
    """
    # Use batch fetch for efficiency
    all_jobs = get_all_running_jobs_for_episode(ep_id)
    result = all_jobs.get(job_type)

    # For audio_pipeline, merge progress data
    if result and job_type == "audio_pipeline":
        audio_progress_data = get_audio_progress(ep_id)
        if audio_progress_data:
            overall = audio_progress_data.get("overall_progress") or audio_progress_data.get("progress")
            if overall is not None:
                pct = overall * 100 if overall <= 1 else overall
                result["progress_pct"] = max(result.get("progress_pct", 0), pct)
            result.setdefault("step_name", audio_progress_data.get("step_name") or audio_progress_data.get("step"))
            if audio_progress_data.get("message"):
                result["message"] = audio_progress_data["message"]

    return result


def get_running_job_for_episode_full(ep_id: str, job_type: str) -> Dict[str, Any] | None:
    """Check if there's a running job of a specific type for an episode.

    Full version that checks all sources including legacy APIs.
    Use get_running_job_for_episode for faster cached lookups.

    Args:
        ep_id: Episode identifier
        job_type: One of "detect_track", "faces_embed", "cluster", "audio_pipeline"

    Returns:
        Job info dict with progress if running, None otherwise.
        Dict includes: job_id, state, progress_pct, message, started_at

    Checks multiple sources in priority order:
    1. /celery_jobs/local (local subprocess jobs)
    2. /celery_jobs (active Celery tasks)
    3. Legacy /jobs API (subprocess jobs)
    4. Session state for stored job_id
    """
    running_states = {"running", "in_progress", "queued", "started", "pending", "scheduled", "retrying"}
    api_timeout = 3  # Reduced from 10s for local API calls
    audio_progress_data = get_audio_progress(ep_id) if job_type == "audio_pipeline" else None

    def _merge_audio_progress(result: Dict[str, Any]) -> Dict[str, Any]:
        """Attach audio progress.json data when available."""
        if job_type != "audio_pipeline":
            return result

        progress_data = audio_progress_data or get_audio_progress(ep_id)
        if not progress_data:
            return result

        overall = progress_data.get("overall_progress") or progress_data.get("progress")
        step_progress = progress_data.get("step_progress")
        if overall is not None:
            pct = overall * 100 if overall <= 1 else overall
            result["progress_pct"] = max(result.get("progress_pct", 0), pct)
        elif step_progress is not None:
            pct = step_progress * 100 if step_progress <= 1 else step_progress
            result["progress_pct"] = max(result.get("progress_pct", 0), pct)

        result.setdefault("step_name", progress_data.get("step_name") or progress_data.get("step"))
        result.setdefault("step_order", progress_data.get("step_order", 0))
        result.setdefault("total_steps", progress_data.get("total_steps", 9))
        if progress_data.get("message"):
            result["message"] = progress_data["message"]
        return result

    def _check_legacy_jobs() -> Dict[str, Any] | None:
        """Check legacy /jobs API."""
        try:
            resp = requests.get(
                f"{_api_base()}/jobs",
                params={"ep_id": ep_id, "job_type": job_type, "limit": 1},
                timeout=api_timeout,
            )
            resp.raise_for_status()
            jobs = resp.json().get("jobs", [])

            for job in jobs:
                state = str(job.get("state", "")).lower()
                if state in running_states:
                    job_id = job.get("job_id")
                    result = {
                        "job_id": job_id,
                        "state": state,
                        "started_at": job.get("started_at"),
                        "job_type": job_type,
                        "source": "legacy_jobs",
                    }

                    # Try to get progress from Celery if it's a Celery job
                    if job_id:
                        celery_status = check_celery_job_status(job_id)
                        if celery_status:
                            progress = celery_status.get("progress", {})
                            result["progress_pct"] = progress.get("progress", 0) * 100
                            result["message"] = progress.get("message", "")

                    # Try to get progress from progress.json file
                    if "progress_pct" not in result:
                        progress_data = get_episode_progress(ep_id)
                        if progress_data:
                            frames_done = progress_data.get("frames_done", 0)
                            frames_total = progress_data.get("frames_total", 1)
                            if frames_total > 0:
                                result["progress_pct"] = (frames_done / frames_total) * 100
                            result["message"] = f"Phase: {progress_data.get('phase', 'unknown')}"
                            result["frames_done"] = frames_done
                            result["frames_total"] = frames_total

                    return _merge_audio_progress(result)
        except requests.RequestException:
            pass
        return None

    def _check_local_jobs() -> Dict[str, Any] | None:
        """Check /celery_jobs/local for local subprocess jobs."""
        try:
            resp = requests.get(
                f"{_api_base()}/celery_jobs/local",
                params={"ep_id": ep_id},
                timeout=api_timeout,
            )
            resp.raise_for_status()
            local_jobs = resp.json().get("jobs", [])

            for job in local_jobs:
                op = job.get("operation", "")
                if op == job_type:
                    state = str(job.get("state", "")).lower()
                    if state in running_states:
                        result = {
                            "job_id": job.get("job_id"),
                            "state": state,
                            "started_at": job.get("started_at"),
                            "job_type": job_type,
                            "source": "celery_local",
                            "pid": job.get("pid"),
                        }

                        # Get progress from progress.json
                        progress_data = get_episode_progress(ep_id)
                        if progress_data:
                            frames_done = progress_data.get("frames_done", 0)
                            frames_total = progress_data.get("frames_total", 1)
                            if frames_total > 0:
                                result["progress_pct"] = (frames_done / frames_total) * 100
                            result["message"] = f"Phase: {progress_data.get('phase', 'unknown')}"
                            result["frames_done"] = frames_done
                            result["frames_total"] = frames_total

                        return _merge_audio_progress(result)
        except requests.RequestException:
            pass
        return None

    def _check_celery_jobs() -> Dict[str, Any] | None:
        """Check /celery_jobs for active Celery tasks."""
        try:
            resp = requests.get(f"{_api_base()}/celery_jobs", timeout=api_timeout)
            resp.raise_for_status()
            celery_jobs = resp.json().get("jobs", [])

            # Aggregate audio pipeline progress across stages
            if job_type == "audio_pipeline":
                matching = [
                    job for job in celery_jobs
                    if job.get("ep_id") == ep_id
                    and (job.get("operation") == "audio_pipeline")
                    and str(job.get("state", "")).lower() in running_states
                ]
                if matching:
                    # Use the furthest stage by order
                    sorted_jobs = sorted(matching, key=lambda j: j.get("stage_order") or 0, reverse=True)
                    current = sorted_jobs[0]
                    state = str(current.get("state", "")).lower() or "running"
                    stage_name = current.get("stage") or current.get("name", "")
                    stage_order = current.get("stage_order", 0)
                    total_steps = 10  # audio pipeline has 10 coarse stages including export/QC
                    progress_pct = min(100.0, max(stage_order, 1) / total_steps * 100.0)
                    result = {
                        "job_id": current.get("job_id"),
                        "state": state,
                        "job_type": job_type,
                        "source": "celery_active",
                        "step_name": stage_name,
                        "step_order": stage_order,
                        "total_steps": total_steps,
                        "message": (stage_name or "").replace("_", " ").title(),
                        "progress_pct": progress_pct,
                    }
                    return _merge_audio_progress(result)

            for job in celery_jobs:
                job_ep_id = job.get("ep_id")
                op = job.get("operation") or job.get("name", "").replace("local_", "")
                state = str(job.get("state", "")).lower()

                # Match by ep_id and operation
                if job_ep_id == ep_id and op == job_type and state in running_states:
                    result = {
                        "job_id": job.get("job_id"),
                        "state": state,
                        "started_at": job.get("started_at"),
                        "job_type": job_type,
                        "source": "celery_active",
                    }
                    return _merge_audio_progress(result)
        except requests.RequestException:
            pass
        return None

    def _check_session_state() -> Dict[str, Any] | None:
        """Check session state for stored job_id."""
        stored_job_id = get_stored_celery_job_id(ep_id, job_type)
        if stored_job_id:
            celery_status = check_celery_job_status(stored_job_id)
            if celery_status:
                state = str(celery_status.get("state", "")).lower()
                if state in running_states:
                    progress = celery_status.get("progress", {})
                    result = {
                        "job_id": stored_job_id,
                        "state": state,
                        "job_type": job_type,
                        "source": "session_state",
                        "progress_pct": progress.get("progress", 0) * 100,
                        "message": progress.get("message", ""),
                    }
                    return result
                else:
                    # Job is no longer running, clear from session state
                    clear_celery_job_id(ep_id, job_type)
        return None

    # Check sources in priority order (most likely first for faster response)
    # Local jobs are most common when using local execution mode
    checkers = [
        _check_local_jobs,   # Most likely for local mode
        _check_celery_jobs,  # Active Celery tasks
        _check_legacy_jobs,  # Legacy /jobs API
        _check_session_state,  # Session state fallback
    ]

    for checker in checkers:
        try:
            result = checker()
            if result:
                return _merge_audio_progress(result)
        except Exception:
            pass  # Continue checking other sources

    # NOTE: We previously had a fallback that trusted progress files even without job registry,
    # but this caused stale progress bars after crashes/restarts. Progress files are only used
    # to ENHANCE job info from actual registries, not as a standalone source of truth.
    # If no actual job is found in any registry, don't show a progress bar.

    return None


def get_episode_progress(ep_id: str) -> Dict[str, Any] | None:
    """Read progress.json for an episode to get current job progress.

    Returns progress dict or None if not found/error.
    """
    try:
        progress_path = DATA_ROOT / "manifests" / ep_id / "progress.json"
        if not progress_path.exists():
            return None

        with open(progress_path, "r") as f:
            import json
            return json.load(f)
    except Exception:
        return None


def get_audio_progress(ep_id: str) -> Dict[str, Any] | None:
    """Read audio_progress.json for an episode to show audio pipeline progress.

    Returns progress data only if there's actually a running job - prevents
    showing stale progress from crashed/cancelled jobs.

    NOTE: This function is now only used to ENHANCE progress info from actual
    job registries. It does NOT standalone determine if a job is running.
    The status field must be checked by callers.
    """
    try:
        progress_path = DATA_ROOT / "manifests" / ep_id / "audio_progress.json"
        if not progress_path.exists():
            return None
        with open(progress_path, "r") as f:
            import json
            data = json.load(f)

        # Always ignore completed/errored/stale entries - these are terminal states
        overall = data.get("overall_progress") or data.get("progress")
        status = str(data.get("status", "")).lower()
        if overall is not None and overall >= 1:
            return None
        if status in {"succeeded", "completed", "complete", "error", "cancelled", "timeout", "failed", "stale"}:
            return None

        # IMPORTANT: Progress files can become stale if jobs crash or are interrupted.
        # ALWAYS verify with job registries before trusting progress data.
        # Check file age and verify with Celery/local job tracker
        timestamp = data.get("timestamp")
        if timestamp is not None:
            import time
            age_seconds = time.time() - timestamp

            # For ANY file with running status, verify there's actually a job
            # (previously only checked for files older than 60s, but that's too permissive)
            try:
                # Check both Celery and local job registries
                has_active_job = False

                # Check Celery jobs
                try:
                    resp = requests.get(f"{_api_base()}/celery_jobs", timeout=2)
                    if resp.ok:
                        jobs = resp.json().get("jobs", [])
                        has_active_job = any(
                            j.get("ep_id") == ep_id and
                            (j.get("operation") == "audio_pipeline" or
                             str(j.get("name", "")).startswith("audio."))
                            for j in jobs
                        )
                except requests.RequestException:
                    pass

                # Check local/subprocess jobs if Celery didn't find anything
                if not has_active_job:
                    try:
                        resp = requests.get(f"{_api_base()}/celery_jobs/local", timeout=2)
                        if resp.ok:
                            jobs = resp.json().get("jobs", [])
                            has_active_job = any(
                                j.get("ep_id") == ep_id and
                                (j.get("operation") == "audio_pipeline" or
                                 j.get("job_type") == "audio_pipeline")
                                for j in jobs
                            )
                    except requests.RequestException:
                        pass

                if not has_active_job:
                    # No active job found in any registry - mark file as stale
                    # Only mark stale if file is older than 10 seconds (give new jobs time to register)
                    if age_seconds > 10:
                        data["status"] = "stale"
                        data["message"] = f"Marked stale: no active job found (age: {age_seconds:.0f}s)"
                        progress_path.write_text(json.dumps(data), encoding="utf-8")
                    return None

            except Exception:
                # If we can't verify, use conservative timeout
                if age_seconds > 120:  # 2 minutes without verification = stale
                    return None

        return data
    except Exception:
        return None


# =============================================================================
# Audio Pipeline Phase Status Checks
# =============================================================================


def _get_audio_paths(ep_id: str) -> Dict[str, Path]:
    """Get standard audio artifact paths for an episode."""
    manifests_dir = DATA_ROOT / "manifests" / ep_id
    return {
        "original": manifests_dir / "episode_original.wav",
        "vocals": manifests_dir / "episode_vocals.wav",
        "vocals_enhanced": manifests_dir / "episode_vocals_enhanced.wav",
        "diarization": manifests_dir / "audio_diarization.jsonl",
        # Legacy paths for backward compatibility
        "diarization_pyannote": manifests_dir / "audio_diarization_pyannote.jsonl",
        "diarization_gpt4o": manifests_dir / "audio_diarization_gpt4o.jsonl",
        "diarization_comparison": manifests_dir / "audio_diarization_comparison.json",
        "asr_raw": manifests_dir / "audio_asr_raw.jsonl",
        "voice_clusters": manifests_dir / "audio_voice_clusters.json",
        "voice_mapping": manifests_dir / "audio_voice_mapping.json",
        "transcript_jsonl": manifests_dir / "episode_transcript.jsonl",
        "transcript_vtt": manifests_dir / "episode_transcript.vtt",
        "qc": manifests_dir / "audio_qc.json",
    }


def check_audio_files_exist(ep_id: str) -> Dict[str, Any]:
    """Check if audio files have been created (Phase 1).

    Returns dict with:
        - exists: bool - True if essential audio files exist
        - original: bool - True if original audio exists
        - vocals: bool - True if vocals (separated) exists
        - vocals_enhanced: bool - True if enhanced vocals exists
        - status: str - "complete", "partial", or "missing"
    """
    paths = _get_audio_paths(ep_id)
    original = paths["original"].exists()
    vocals = paths["vocals"].exists()
    vocals_enhanced = paths["vocals_enhanced"].exists()

    # Consider complete if we have at least vocals (enhanced is optional)
    if vocals:
        status = "complete" if vocals_enhanced else "partial"
    else:
        status = "missing"

    return {
        "exists": vocals,  # Minimum requirement
        "original": original,
        "vocals": vocals,
        "vocals_enhanced": vocals_enhanced,
        "status": status,
    }


def check_diarization_complete(ep_id: str) -> Dict[str, Any]:
    """Check if diarization + transcription has been run (Phase 2).

    Returns dict with:
        - complete: bool - True if diarization and ASR exist
        - diarization: bool - True if diarization exists (NeMo MSDD)
        - diarization_pyannote: bool - (Legacy) True if old pyannote diarization exists
        - diarization_gpt4o: bool - (Legacy) True if old GPT-4o diarization exists
        - asr: bool - True if ASR raw transcript exists
        - voice_clusters: bool - True if initial voice clusters exist
        - segment_count: int - Number of diarization segments
        - cluster_count: int - Number of voice clusters
        - status: str - "complete", "partial", or "not_run"
    """
    paths = _get_audio_paths(ep_id)

    diarization = paths["diarization"].exists()
    # Legacy paths for backward compatibility
    diarization_pyannote = paths["diarization_pyannote"].exists()
    diarization_gpt4o = paths["diarization_gpt4o"].exists()
    asr = paths["asr_raw"].exists()
    voice_clusters = paths["voice_clusters"].exists()

    # Count segments and clusters
    segment_count = 0
    cluster_count = 0

    if diarization:
        try:
            with open(paths["diarization"], "r") as f:
                segment_count = sum(1 for line in f if line.strip())
        except Exception:
            pass

    if voice_clusters:
        try:
            with open(paths["voice_clusters"], "r") as f:
                data = json.load(f)
                cluster_count = len(data.get("clusters", []))
        except Exception:
            pass

    # Determine status
    if diarization and asr:
        status = "complete"
    elif diarization or asr:
        status = "partial"
    else:
        status = "not_run"

    return {
        "complete": diarization and asr,
        "diarization": diarization,
        "diarization_pyannote": diarization_pyannote,  # Legacy
        "diarization_gpt4o": diarization_gpt4o,  # Legacy
        "asr": asr,
        "voice_clusters": voice_clusters,
        "segment_count": segment_count,
        "cluster_count": cluster_count,
        "status": status,
    }


def check_voice_assignments_complete(ep_id: str) -> Dict[str, Any]:
    """Check if voice assignments have been made (Phase 3 - Manual Review).

    Returns dict with:
        - complete: bool - True if any voices have been assigned names
        - voice_mapping_exists: bool - True if voice mapping file exists
        - labeled_count: int - Number of labeled voice clusters
        - unlabeled_count: int - Number of unlabeled voice clusters
        - total_count: int - Total voice clusters
        - status: str - "complete", "partial", or "not_started"
    """
    paths = _get_audio_paths(ep_id)

    if not paths["voice_mapping"].exists():
        return {
            "complete": False,
            "voice_mapping_exists": False,
            "labeled_count": 0,
            "unlabeled_count": 0,
            "total_count": 0,
            "status": "not_started",
        }

    try:
        with open(paths["voice_mapping"], "r") as f:
            mapping = json.load(f)

        # Count labeled vs unlabeled
        entries = mapping if isinstance(mapping, list) else mapping.get("mappings", [])
        labeled = sum(1 for m in entries if m.get("is_labeled", False))
        unlabeled = len(entries) - labeled

        # Consider complete if at least some voices are labeled
        if labeled > 0:
            status = "complete" if unlabeled == 0 else "partial"
        else:
            status = "not_started"

        return {
            "complete": labeled > 0,
            "voice_mapping_exists": True,
            "labeled_count": labeled,
            "unlabeled_count": unlabeled,
            "total_count": len(entries),
            "status": status,
        }
    except Exception:
        return {
            "complete": False,
            "voice_mapping_exists": True,
            "labeled_count": 0,
            "unlabeled_count": 0,
            "total_count": 0,
            "status": "error",
        }


def check_transcript_finalized(ep_id: str) -> Dict[str, Any]:
    """Check if transcript has been finalized (Phase 4).

    Returns dict with:
        - complete: bool - True if transcript and QC exist
        - transcript_jsonl: bool - True if JSONL transcript exists
        - transcript_vtt: bool - True if VTT subtitle exists
        - qc: bool - True if QC report exists
        - qc_status: str - QC status (ok, warn, needs_review, failed, unknown)
        - transcript_row_count: int - Number of transcript rows
        - status: str - "complete", "partial", or "not_run"
    """
    paths = _get_audio_paths(ep_id)

    transcript_jsonl = paths["transcript_jsonl"].exists()
    transcript_vtt = paths["transcript_vtt"].exists()
    qc = paths["qc"].exists()

    # Get QC status and transcript row count
    qc_status = "unknown"
    transcript_row_count = 0

    if qc:
        try:
            with open(paths["qc"], "r") as f:
                qc_data = json.load(f)
                qc_status = qc_data.get("status", "unknown")
        except Exception:
            pass

    if transcript_jsonl:
        try:
            with open(paths["transcript_jsonl"], "r") as f:
                transcript_row_count = sum(1 for line in f if line.strip())
        except Exception:
            pass

    # Determine status
    if transcript_jsonl and qc:
        status = "complete"
    elif transcript_jsonl or qc:
        status = "partial"
    else:
        status = "not_run"

    return {
        "complete": transcript_jsonl and qc,
        "transcript_jsonl": transcript_jsonl,
        "transcript_vtt": transcript_vtt,
        "qc": qc,
        "qc_status": qc_status,
        "transcript_row_count": transcript_row_count,
        "status": status,
    }


def get_audio_pipeline_status(ep_id: str) -> Dict[str, Any]:
    """Get complete audio pipeline status for all phases.

    Returns dict with status of all phases and recommended next action.
    """
    audio_files = check_audio_files_exist(ep_id)
    diarization = check_diarization_complete(ep_id)
    voice_assignments = check_voice_assignments_complete(ep_id)
    transcript = check_transcript_finalized(ep_id)

    # Determine next action
    if not audio_files["exists"]:
        next_action = "create_audio_files"
        next_action_label = "Create Audio Files"
    elif not diarization["complete"]:
        next_action = "run_diarization"
        next_action_label = "Run Diarization + Transcription"
    elif not voice_assignments["complete"]:
        next_action = "review_voices"
        next_action_label = "Review & Assign Voices"
    elif not transcript["complete"]:
        next_action = "finalize_transcript"
        next_action_label = "Finalize Transcript"
    else:
        next_action = "done"
        next_action_label = "Audio Pipeline Complete"

    return {
        "audio_files": audio_files,
        "diarization": diarization,
        "voice_assignments": voice_assignments,
        "transcript": transcript,
        "next_action": next_action,
        "next_action_label": next_action_label,
    }


def cancel_running_job(job_id: str) -> tuple[bool, str]:
    """Cancel a running Celery job.

    Returns (success, message).
    """
    try:
        resp = requests.post(f"{_api_base()}/celery_jobs/{job_id}/cancel", timeout=10)
        resp.raise_for_status()
        result = resp.json()
        status = result.get("status", "unknown")
        if status in ("cancelled", "already_finished"):
            return True, f"Job {job_id[:8]}... {status}"
        return False, f"Unexpected status: {status}"
    except requests.RequestException as exc:
        return False, describe_error(f"Cancel job {job_id}", exc)


def load_operation_logs(ep_id: str, operation: str) -> Dict[str, Any] | None:
    """Load the most recent logs for an operation from the API.

    This fetches persisted logs that were saved when a local mode job completed.
    Used to display previous run logs on page load.

    Args:
        ep_id: Episode identifier
        operation: Operation name (detect_track, faces_embed, cluster, audio_pipeline)

    Returns:
        Dict with logs, status, elapsed_seconds, etc. or None if no logs exist
    """
    if operation not in _REMOTE_LOG_OPERATIONS:
        return None

    try:
        resp = requests.get(
            f"{_api_base()}/celery_jobs/logs/{ep_id}/{operation}",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        # Return None if no logs exist
        if data.get("status") == "none":
            return None

        return data

    except requests.RequestException as exc:
        logging.debug(f"Failed to load logs for {ep_id}/{operation}: {exc}")
        return None


def render_previous_logs(
    ep_id: str,
    operation: str,
    *,
    expanded: bool = False,
    show_if_none: bool = False,
) -> bool:
    """Render the most recent logs for an operation in a Streamlit expander.

    Uses cached logs from session state first (populated by hydrate_logs_for_episode
    on page load), falling back to API call if not cached.

    Displays formatted (human-friendly) logs only - clean and copy/paste-able.

    Args:
        ep_id: Episode identifier
        operation: Operation name (detect_track, faces_embed, cluster)
        expanded: Whether to expand the log expander by default
        show_if_none: Whether to show a placeholder when no logs exist

    Returns:
        True if logs were found and rendered, False otherwise
    """
    # Try cached logs first (from hydrate_logs_for_episode on page load)
    data = get_cached_logs(ep_id, operation)

    # Fall back to API call if not cached
    if not data or not data.get("logs"):
        data = load_operation_logs(ep_id, operation)

    if data is None:
        if show_if_none:
            with st.expander("Previous run logs", expanded=False):
                st.caption("No previous logs available for this operation.")
        return False

    history = data.get("history") or []
    runs = history if history else [data]

    status = data.get("status", "unknown")
    formatted_logs = data.get("logs", [])
    elapsed_seconds = data.get("elapsed_seconds", 0)
    updated_at = data.get("updated_at", "")

    # Format elapsed time
    if elapsed_seconds >= 60:
        elapsed_min = int(elapsed_seconds // 60)
        elapsed_sec = int(elapsed_seconds % 60)
        elapsed_str = f"{elapsed_min}m {elapsed_sec}s"
    else:
        elapsed_str = f"{elapsed_seconds:.1f}s"

    # Format status icon
    status_icons = {
        "completed": "✅",
        "error": "❌",
        "cancelled": "⚠️",
        "timeout": "⏱️",
    }
    icon = status_icons.get(status, "ℹ️")

    # Format timestamp for display
    timestamp_str = ""
    if updated_at:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            timestamp_str = dt.strftime(" (%Y-%m-%d %H:%M)")
        except (ValueError, TypeError):
            pass

    expander_label = f"{icon} Previous {operation} run ({status}, {elapsed_str}){timestamp_str}"

    with st.expander(expander_label, expanded=expanded):
        # Run selector when history available
        selected_run = runs[0]
        if history:
            options = []
            for run in runs:
                ts = run.get("updated_at")
                ts_label = format_est(ts) or ts or "unknown time"
                options.append(f"{ts_label} · {run.get('status', 'unknown')} · {run.get('elapsed_seconds', 0):.1f}s")
            idx = st.selectbox(
                "Choose run",
                options=range(len(options)),
                format_func=lambda i: options[i],
                index=0,
                key=f"{ep_id}::{operation}::log_history_select",
            )
            selected_run = runs[idx]

        selected_logs = selected_run.get("logs", [])
        if not selected_logs:
            st.caption("No log lines recorded.")
            return True

        st.code("\n".join(selected_logs), language="text")

    return True


# =============================================================================
# Execution Mode Job Helpers
# =============================================================================


def run_pipeline_job_with_mode(
    ep_id: str,
    operation: str,
    payload: Dict[str, Any],
    *,
    requested_device: str = "auto",
    requested_detector: str | None = None,
    requested_tracker: str | None = None,
    timeout: int = 3600,
):
    """Run a pipeline job respecting the current execution mode for the episode.

    This function handles both local and redis execution modes:
    - Local mode: Runs synchronously - blocks until complete, no job ID, no polling
    - Redis mode: Queues via Celery and polls for completion

    Args:
        ep_id: Episode identifier
        operation: One of "detect_track", "faces_embed", "cluster"
        payload: Request payload for the job (execution_mode will be added)
        requested_device: Device string for context display
        requested_detector: Detector string for context display
        requested_tracker: Tracker string for context display
        timeout: Timeout in seconds for local mode (default 3600 = 1 hour)

    Returns:
        Tuple of (summary_dict, error_message)
    """
    execution_mode = get_execution_mode(ep_id)

    def _ensure_local_completion_signals(ep_id: str, operation: str, summary: Dict[str, Any]) -> None:
        """Best-effort: write run marker + progress.json when local job completes.

        Local mode sometimes finishes without updating run markers; auto-run relies on
        marker/progress mtimes to advance to the next phase. We defensively write them.
        """
        status = str(summary.get("status") or "").lower()
        if status not in {"completed", "success"}:
            return

        LOGGER.info("[LOCAL MODE] Writing completion signals for %s/%s", ep_id, operation)

        manifests_dir = DATA_ROOT / "manifests" / ep_id
        run_dir = manifests_dir / "runs"
        now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

        def _count_lines(path: Path) -> int | None:
            try:
                if path.exists():
                    with path.open("r", encoding="utf-8") as fh:
                        return sum(1 for line in fh if line.strip())
            except OSError:
                pass
            return None

        def _write_progress_file(step_name: str) -> None:
            progress_path = manifests_dir / "progress.json"
            payload = {
                "ep_id": ep_id,
                "phase": "done",
                "step": step_name,
                "status": "completed",
                "updated_at": now_iso,
            }
            try:
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                progress_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                # Signal to UI that this phase just finished (used by auto-run fallbacks)
                st.session_state[f"{ep_id}::{step_name}_just_completed"] = True
            except OSError as exc:
                LOGGER.warning("[LOCAL MODE] Failed to write progress.json for %s/%s: %s", ep_id, step_name, exc)

        def _write_run_marker(step_name: str, payload: Dict[str, Any]) -> None:
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                marker_path = run_dir / f"{step_name}.json"
                marker_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except OSError as exc:
                LOGGER.warning("[LOCAL MODE] Failed to write run marker for %s/%s: %s", ep_id, step_name, exc)

        # Build marker payload with fallbacks for counts
        marker_payload: Dict[str, Any] = {
            "phase": operation,
            "status": "success",
            "ep_id": ep_id,
            "finished_at": now_iso,
            "started_at": summary.get("started_at") or summary.get("started") or summary.get("start_time"),
            "device": summary.get("device") or summary.get("requested_device"),
            "requested_device": summary.get("requested_device"),
            "resolved_device": summary.get("resolved_device"),
            "profile": summary.get("profile"),
            "version": summary.get("version"),
        }

        manifests_counts = {
            "detect_track": {
                "detections": _count_lines(manifests_dir / "detections.jsonl"),
                "tracks": _count_lines(manifests_dir / "tracks.jsonl"),
            },
            "faces_embed": {
                "faces": _count_lines(manifests_dir / "faces.jsonl"),
            },
            "cluster": {
                "identities": None,
            },
        }
        # Identities count (json structure)
        if operation == "cluster":
            identities_path = manifests_dir / "identities.json"
            try:
                if identities_path.exists():
                    data = json.loads(identities_path.read_text(encoding="utf-8"))
                    ids_list = data.get("identities") if isinstance(data, dict) else None
                    if isinstance(ids_list, list):
                        manifests_counts["cluster"]["identities"] = len(ids_list)
            except (OSError, json.JSONDecodeError):
                pass

        # Merge summary counts with manifest fallbacks
        if operation == "detect_track":
            marker_payload["detections"] = summary.get("detections") or manifests_counts["detect_track"]["detections"]
            marker_payload["tracks"] = summary.get("tracks") or manifests_counts["detect_track"]["tracks"]
            marker_payload["detector"] = summary.get("detector")
            marker_payload["tracker"] = summary.get("tracker")
            marker_payload["stride"] = summary.get("stride")
        elif operation == "faces_embed":
            marker_payload["faces"] = summary.get("faces") or manifests_counts["faces_embed"]["faces"]
            marker_payload["device"] = summary.get("device") or summary.get("requested_device")
        elif operation == "cluster":
            marker_payload["identities"] = summary.get("identities") or manifests_counts["cluster"]["identities"]
            marker_payload["faces"] = summary.get("faces") or manifests_counts["faces_embed"]["faces"]
            marker_payload["cluster_thresh"] = summary.get("cluster_thresh")
            marker_payload["min_cluster_size"] = summary.get("min_cluster_size")

        _write_run_marker(operation, marker_payload)
        _write_progress_file(operation)

    # Add execution_mode to payload
    payload = {**payload, "execution_mode": execution_mode}

    # Map operation to endpoint
    endpoint_map = {
        "detect_track": "/celery_jobs/detect_track",
        "faces_embed": "/celery_jobs/faces_embed",
        "cluster": "/celery_jobs/cluster",
    }
    endpoint = endpoint_map.get(operation)
    if not endpoint:
        return None, f"Unknown operation: {operation}"

    if execution_mode == "local":
        # Local mode: streaming response with live log updates
        # The backend streams log lines as newline-delimited JSON
        status_placeholder = st.empty()

        # Progress bar section - shows % complete and frame counts
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0.0)
            progress_text = st.empty()
            progress_text.caption("Starting...")

        log_expander = st.expander("Detailed log", expanded=True)
        with log_expander:
            log_placeholder = st.empty()

        # Build context string for display
        context_parts = [f"device={requested_device}"]
        if requested_detector:
            context_parts.append(f"detector={requested_detector}")
        if requested_tracker:
            context_parts.append(f"tracker={requested_tracker}")
        context_str = ", ".join(context_parts)

        status_placeholder.info(f"⏳ [LOCAL MODE] Running {operation} ({context_str})...")
        log_lines: List[str] = []
        log_placeholder.code(
            f"[LOCAL MODE] Starting {operation} ({context_str})...\n"
            "Waiting for live logs...",
            language="text",
        )

        # Track progress info for display
        current_progress = {"frames_done": 0, "frames_total": 0, "phase": "starting", "pct": 0.0}

        try:
            # Streaming request - reads lines as they arrive
            with requests.post(
                f"{_api_base()}{endpoint}",
                json=payload,
                stream=True,
                timeout=None,  # No timeout - stream can run for hours
            ) as resp:
                resp.raise_for_status()

                summary: Dict[str, Any] = {}

                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue

                    try:
                        msg = json.loads(raw_line)
                    except json.JSONDecodeError:
                        # Treat non-JSON lines as plain log lines
                        log_lines.append(raw_line)
                        log_placeholder.code("\n".join(log_lines), language="text")
                        continue

                    msg_type = msg.get("type", "")

                    if msg_type == "log":
                        line = msg.get("line", "")
                        log_lines.append(line)
                        log_placeholder.code("\n".join(log_lines), language="text")

                    elif msg_type == "progress":
                        # Update progress bar with real-time data
                        frames_done = msg.get("frames_done", 0)
                        frames_total = msg.get("frames_total", 1)
                        phase = msg.get("phase", "processing")
                        fps_infer = msg.get("fps_infer")
                        secs_done = msg.get("secs_done", 0)

                        if frames_total > 0:
                            pct = min((frames_done / frames_total), 1.0)
                            progress_bar.progress(pct)

                            # Build progress text with details
                            parts = [f"**{pct*100:.1f}%** - {frames_done:,} / {frames_total:,} frames"]
                            if fps_infer and fps_infer > 0:
                                parts.append(f"({fps_infer:.1f} fps)")
                            if secs_done > 0:
                                elapsed_min = int(secs_done // 60)
                                elapsed_sec = int(secs_done % 60)
                                if elapsed_min > 0:
                                    parts.append(f"- {elapsed_min}m {elapsed_sec}s elapsed")
                                else:
                                    parts.append(f"- {elapsed_sec}s elapsed")

                                # Estimate remaining time
                                if pct > 0.01 and pct < 1.0:
                                    eta_secs = (secs_done / pct) * (1 - pct)
                                    if eta_secs > 60:
                                        eta_min = int(eta_secs // 60)
                                        eta_sec = int(eta_secs % 60)
                                        parts.append(f"(~{eta_min}m {eta_sec}s remaining)")
                                    else:
                                        parts.append(f"(~{int(eta_secs)}s remaining)")

                            progress_text.caption(" ".join(parts))
                        else:
                            progress_text.caption(f"Phase: {phase}")

                        # Store current progress
                        current_progress.update({
                            "frames_done": frames_done,
                            "frames_total": frames_total,
                            "phase": phase,
                            "pct": pct if frames_total > 0 else 0,
                        })

                    elif msg_type == "error":
                        # Initial error (e.g., job already running)
                        error_msg = msg.get("message", "Unknown error")
                        log_lines.append(f"ERROR: {error_msg}")
                        log_placeholder.code("\n".join(log_lines), language="text")
                        status_placeholder.error(f"❌ [LOCAL MODE] {operation} failed: {error_msg}")
                        return {"status": "error", "error": error_msg, "logs": log_lines}, error_msg

                    elif msg_type == "summary":
                        summary = msg
                        LOGGER.info("[LOCAL MODE] Received summary for %s: status=%s", operation, msg.get("status"))
                        break

                # Check if we received a valid summary
                # FIX 4: Check for empty dict explicitly - {} is falsy but may slip through
                if not summary or summary == {} or not summary.get("status"):
                    # Stream ended without summary - check if manifests exist
                    LOGGER.warning("[LOCAL MODE] No/empty summary for %s/%s - using fallback", ep_id, operation)
                    summary = {"status": "completed", "elapsed_seconds": 0, "fallback": True}

                # Process summary
                status = summary.get("status", "unknown")
                elapsed_seconds = summary.get("elapsed_seconds", 0)
                error_msg = summary.get("error")

                # Format elapsed time
                if elapsed_seconds >= 60:
                    elapsed_min = int(elapsed_seconds // 60)
                    elapsed_sec = int(elapsed_seconds % 60)
                    elapsed_str = f"{elapsed_min}m {elapsed_sec}s"
                else:
                    elapsed_str = f"{elapsed_seconds:.1f}s"

                result = {
                    "status": status,
                    "logs": log_lines,
                    "elapsed_seconds": elapsed_seconds,
                    **summary,
                }

                # Cache logs in session state for display on page reload
                cache_logs(ep_id, operation, log_lines, status, elapsed_seconds)

                if status == "completed":
                    # Show 100% completion on progress bar
                    progress_bar.progress(1.0)
                    progress_text.caption(f"**100%** - Completed in {elapsed_str}")
                    status_placeholder.success(f"✅ [LOCAL MODE] {operation} completed in {elapsed_str}")
                    _ensure_local_completion_signals(ep_id, operation, result)

                    # CRITICAL: Set completion flags in session state BEFORE returning
                    # This ensures phase advancement even if Streamlit interrupts the return
                    # (Streamlit may restart script on widget updates)
                    st.session_state[f"{ep_id}::{operation}_just_completed"] = True
                    st.session_state[f"{ep_id}::{operation}_completed_at"] = time.time()
                    st.session_state[f"{ep_id}::{operation}_summary"] = result
                    LOGGER.info(
                        "[LOCAL MODE] Set completion flags for %s/%s: just_completed=True, completed_at=%s",
                        ep_id, operation, st.session_state[f"{ep_id}::{operation}_completed_at"]
                    )

                    return result, None
                elif status == "error":
                    progress_text.caption(f"❌ Failed after {elapsed_str}")
                    status_placeholder.error(f"❌ [LOCAL MODE] {operation} failed: {error_msg}")
                    return result, error_msg
                elif status == "timeout":
                    progress_text.caption(f"⏱️ Timed out after {elapsed_str}")
                    status_placeholder.error(f"❌ [LOCAL MODE] {operation} timed out after {elapsed_str}")
                    return result, f"Timed out after {elapsed_str}"
                else:
                    # Unknown status
                    status_placeholder.warning(f"⚠️ [LOCAL MODE] {operation} finished with status: {status}")
                    return result, None

        except requests.exceptions.Timeout:
            error_msg = f"Request timed out after {timeout}s"
            status_placeholder.error(f"❌ {error_msg}")
            return None, error_msg

        except requests.RequestException as exc:
            error_msg = describe_error(f"{_api_base()}{endpoint}", exc)
            status_placeholder.error(f"❌ {error_msg}")
            return None, error_msg

    else:
        # Redis mode - use existing Celery job flow with polling
        return run_celery_job_with_progress(
            ep_id,
            operation,
            payload,
            requested_device=requested_device,
            requested_detector=requested_detector,
            requested_tracker=requested_tracker,
        )


def run_episode_pipeline_job(
    ep_id: str,
    *,
    device: str = "auto",
    stride: int = 1,
    det_thresh: float = 0.65,
    cluster_thresh: float = 0.75,
    save_crops: bool = True,
    save_frames: bool = False,
    reuse_detections: bool = False,
    reuse_embeddings: bool = False,
    profile: str | None = None,
    poll_interval: float = 2.0,
    timeout: int = 7200,
) -> tuple[Dict[str, Any] | None, str | None]:
    """Run the full episode processing pipeline via the job API.

    This function uses the new /jobs/episode-run endpoint which runs the
    engine (detect_track -> faces_embed -> cluster) as a single job.

    Args:
        ep_id: Episode identifier (e.g., 'rhobh-s05e14')
        device: Compute device ('auto', 'cpu', 'cuda', 'coreml')
        stride: Frame stride for detection (higher = faster but less accurate)
        det_thresh: Face detection confidence threshold
        cluster_thresh: Clustering distance threshold
        save_crops: Whether to save face crops
        save_frames: Whether to save full frames
        reuse_detections: Skip detect_track if artifacts exist
        reuse_embeddings: Skip faces_embed if artifacts exist
        profile: Optional preset profile ('fast', 'balanced', 'quality')
        poll_interval: Seconds between status polls
        timeout: Maximum seconds to wait for completion

    Returns:
        Tuple of (result dict with summary, error message or None)
    """
    endpoint = "/jobs/episode-run"
    payload = {
        "ep_id": ep_id,
        "device": device,
        "stride": stride,
        "det_thresh": det_thresh,
        "cluster_thresh": cluster_thresh,
        "save_crops": save_crops,
        "save_frames": save_frames,
        "reuse_detections": reuse_detections,
        "reuse_embeddings": reuse_embeddings,
    }
    if profile:
        payload["profile"] = profile

    # UI elements for progress display
    status_placeholder = st.empty()
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        progress_text.caption("Starting episode pipeline...")

    # Build context string for display
    context_parts = [f"device={device}"]
    if stride > 1:
        context_parts.append(f"stride={stride}")
    if profile:
        context_parts.append(f"profile={profile}")
    context_str = ", ".join(context_parts)

    status_placeholder.info(f"⏳ Starting episode pipeline ({context_str})...")

    try:
        # Start the job
        resp = requests.post(
            f"{_api_base()}{endpoint}",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        job_info = resp.json()
        job_id = job_info.get("job_id")

        if not job_id:
            error_msg = "No job_id returned from API"
            status_placeholder.error(f"❌ {error_msg}")
            return None, error_msg

        status_placeholder.info(f"⏳ Job {job_id[:8]}... running ({context_str})")

        # Poll for completion
        start_time = time.time()
        stage_idx = 0
        stages = ["detect_track", "faces_embed", "cluster"]

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                error_msg = f"Job timed out after {timeout}s"
                status_placeholder.error(f"❌ {error_msg}")
                return {"job_id": job_id, "status": "timeout"}, error_msg

            # Get job status
            status_resp = requests.get(
                f"{_api_base()}/jobs/{job_id}",
                timeout=10,
            )
            status_resp.raise_for_status()
            job_status = status_resp.json()

            state = job_status.get("state", "unknown")

            if state == "running":
                # Update progress based on estimated stage
                # Rough estimate: detect_track=60%, faces_embed=30%, cluster=10%
                pct = min(elapsed / (timeout * 0.5), 0.95)  # Cap at 95% until done
                progress_bar.progress(pct)

                elapsed_min = int(elapsed // 60)
                elapsed_sec = int(elapsed % 60)
                if elapsed_min > 0:
                    elapsed_str = f"{elapsed_min}m {elapsed_sec}s"
                else:
                    elapsed_str = f"{elapsed_sec}s"
                progress_text.caption(f"Running... {elapsed_str} elapsed")

            elif state == "succeeded":
                progress_bar.progress(1.0)
                summary = job_status.get("summary", {})
                runtime = summary.get("runtime_sec", elapsed)

                if runtime >= 60:
                    runtime_min = int(runtime // 60)
                    runtime_sec = int(runtime % 60)
                    runtime_str = f"{runtime_min}m {runtime_sec}s"
                else:
                    runtime_str = f"{runtime:.1f}s"

                identities = summary.get("identities_count", 0)
                tracks = summary.get("tracks_count", 0)
                faces = summary.get("faces_count", 0)

                progress_text.caption(
                    f"**100%** - Completed in {runtime_str} "
                    f"({tracks} tracks, {faces} faces, {identities} identities)"
                )
                status_placeholder.success(
                    f"✅ Episode pipeline completed in {runtime_str} - "
                    f"Found {identities} identities from {tracks} tracks"
                )
                return {"job_id": job_id, "status": "succeeded", **summary}, None

            elif state == "failed":
                error_msg = job_status.get("error", "Unknown error")
                progress_text.caption(f"❌ Failed after {elapsed:.0f}s")
                status_placeholder.error(f"❌ Episode pipeline failed: {error_msg}")
                return {"job_id": job_id, "status": "failed", "error": error_msg}, error_msg

            else:
                # Unknown state - keep polling
                progress_text.caption(f"Status: {state}")

            time.sleep(poll_interval)

    except requests.RequestException as exc:
        error_msg = describe_error(f"{_api_base()}{endpoint}", exc)
        status_placeholder.error(f"❌ {error_msg}")
        return None, error_msg


def run_async_job_with_mode(
    ep_id: str,
    endpoint: str,
    payload: Dict[str, Any],
    operation: str,
) -> Dict[str, Any] | None:
    """Run an async job respecting the current execution mode for the episode.

    This function handles async jobs (like refresh_similarity, batch_assign, group_async):
    - Local mode: Runs synchronously and returns result directly
    - Redis mode: Queues via Celery and returns job info for polling

    Args:
        ep_id: Episode identifier
        endpoint: API endpoint to call (e.g., /episodes/{ep_id}/refresh_similarity_async)
        payload: Request payload for the job (execution_mode will be added)
        operation: Operation name for display purposes

    Returns:
        Response dict from the API, or None if failed
    """
    execution_mode = get_execution_mode(ep_id)

    # Add execution_mode to payload
    payload = {**payload, "execution_mode": execution_mode}

    if execution_mode == "local":
        # Run synchronously with loading spinner
        with st.spinner(f"Running {operation} in local mode..."):
            try:
                resp = requests.post(
                    f"{_api_base()}{endpoint}",
                    json=payload,
                    timeout=600,  # 10 minute timeout for local mode
                )
                resp.raise_for_status()
                result = resp.json()

                status = result.get("status", "unknown")
                if status in ("completed", "success"):
                    st.success(f"✅ {operation} completed successfully (local mode)")
                elif status == "error":
                    st.error(f"❌ {operation} failed: {result.get('error', 'Unknown error')}")
                else:
                    st.info(f"{operation} returned: {status}")

                return result

            except requests.RequestException as exc:
                st.error(f"❌ {operation} failed: {describe_error(f'{_api_base()}{endpoint}', exc)}")
                return None
    else:
        # Redis mode - use existing async job submission
        return submit_async_job(
            endpoint=endpoint,
            payload=payload,
            operation=operation,
            episode_id=ep_id,
        )


def run_audio_pipeline_with_streaming(
    ep_id: str,
    overwrite: bool = False,
    asr_provider: str = "openai_whisper",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> tuple[Dict[str, Any] | None, str | None]:
    """Run audio pipeline with real-time streaming logs and progress.

    This function handles the audio pipeline in local mode with streaming output:
    - Creates UI placeholders for status, progress bar, and logs
    - Makes a streaming POST request to the API
    - Updates progress and logs in real-time as data arrives
    - Returns final result when complete

    Args:
        ep_id: Episode identifier
        overwrite: Whether to overwrite existing artifacts
        asr_provider: ASR provider to use (openai_whisper or gemini)
        min_speakers: Minimum expected speakers (hint for diarization)
        max_speakers: Maximum expected speakers (hint for diarization)

    Returns:
        Tuple of (result dict, error message or None)
    """
    endpoint = "/jobs/episode_audio_pipeline"
    payload = {
        "ep_id": ep_id,
        "overwrite": overwrite,
        "asr_provider": asr_provider,
        "run_mode": "local",
    }
    if min_speakers is not None:
        payload["min_speakers"] = min_speakers
    if max_speakers is not None:
        payload["max_speakers"] = max_speakers

    # Create UI placeholders
    status_placeholder = st.empty()

    # Progress bar section
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        progress_text.caption("Starting audio pipeline...")

    # Use a container with header instead of expander to avoid nesting issues
    st.markdown("**Detailed Log:**")
    log_placeholder = st.empty()

    # Audio pipeline step names for display
    step_names = {
        "init": "Initializing",
        "extract": "Extracting Audio",
        "separate": "Separating Vocals (MDX-Extra)",
        "enhance": "Enhancing Audio (Resemble)",
        "diarize": "Speaker Diarization (Pyannote)",
        "voices": "Voice Clustering",
        "transcribe": "Transcription (ASR)",
        "fuse": "Fusing Transcript",
        "export": "Exporting Artifacts",
        "qc": "Quality Control",
        "s3_sync": "S3 Sync",
        "complete": "Complete",
        "error": "Error",
    }

    status_placeholder.info(f"⏳ [LOCAL MODE] Running audio pipeline for {ep_id}...")
    log_lines: List[str] = []
    log_placeholder.code(
        f"[LOCAL MODE] Starting audio pipeline for {ep_id}...\n"
        f"ASR Provider: {asr_provider}\n"
        f"Overwrite: {overwrite}\n"
        "Waiting for live logs...",
        language="text",
    )

    start_time = time.time()

    try:
        def _clean_msg(text: str) -> str:
            """Remove demucs BagOfModels noise from user-facing messages."""
            if not text:
                return text
            lowered = text.lower()
            if "call apply_model on this" in lowered:
                return "Demucs/MDX model failed to load (library noise suppressed)"
            return text

        # Streaming request - reads lines as they arrive
        with requests.post(
            f"{_api_base()}{endpoint}",
            json=payload,
            stream=True,
            timeout=None,  # No timeout - audio pipeline can take a long time
        ) as resp:
            resp.raise_for_status()

            final_result: Dict[str, Any] = {}

            # Noise patterns to filter from logs
            noise_patterns = [
                "Call apply_model on this",  # Demucs BagOfModels repr
                "BagOfModels(",  # Demucs model repr
                "<demucs.",  # Demucs internal repr
            ]

            def _is_noise(line: str) -> bool:
                """Check if a log line is noise that should be filtered."""
                for pattern in noise_patterns:
                    if pattern.lower() in line.lower():
                        return True
                return False

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                try:
                    msg = json.loads(raw_line)
                except json.JSONDecodeError:
                    # Treat non-JSON lines as plain log lines (filter noise)
                    if not _is_noise(raw_line):
                        log_lines.append(raw_line)
                        log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                    continue

                # Handle different message types from _stream_local_subprocess
                msg_type = msg.get("type", "")

                # Handle log messages (from LogFormatter)
                if msg_type == "log":
                    line = msg.get("line", "")
                    if line and not _is_noise(line):
                        log_lines.append(line)
                        log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                    continue

                # Handle audio_progress messages (new format for audio pipeline)
                if msg_type == "audio_progress":
                    phase = msg.get("phase", "")
                    progress = msg.get("progress", 0)
                    message = _clean_msg(msg.get("message", ""))
                    step_name = msg.get("step_name", "")
                    step_progress = msg.get("step_progress", 0)
                    step_order = msg.get("step_order", 0)
                    total_steps = msg.get("total_steps", 9)
                else:
                    # Legacy: direct phase/progress format (backward compatibility)
                    phase = msg.get("phase", "")
                    progress = msg.get("progress", 0)
                    message = _clean_msg(msg.get("message", ""))
                    step_name = msg.get("step_name", "")
                    step_progress = msg.get("step_progress", 0)
                    step_order = msg.get("step_order", 0)
                    total_steps = msg.get("total_steps", 9)

                # Skip if no phase (not a progress message)
                if not phase and msg_type not in ("audio_progress", "summary"):
                    continue

                # Get human-readable step name
                display_name = step_names.get(phase, step_name or phase.replace("_", " ").title())

                # Calculate elapsed time
                elapsed_secs = time.time() - start_time
                elapsed_min = int(elapsed_secs // 60)
                elapsed_sec = int(elapsed_secs % 60)
                elapsed_str = f"{elapsed_min}m {elapsed_sec}s" if elapsed_min > 0 else f"{elapsed_sec}s"

                # Handle summary message (completion/error from _stream_local_subprocess)
                if msg_type == "summary":
                    status = msg.get("status", "")
                    if status == "completed":
                        progress_bar.progress(1.0)
                        completion_msg = f"Audio pipeline completed in {elapsed_str}"
                        log_lines.append(f"✅ {completion_msg}")
                        log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                        progress_text.caption(f"**100%** - {completion_msg}")
                        status_placeholder.success(f"✅ [LOCAL MODE] {completion_msg}")
                        result = {"status": "succeeded", "logs": log_lines, "elapsed_seconds": elapsed_secs}
                        cache_logs(ep_id, "audio_pipeline", log_lines, "completed", elapsed_secs)
                        append_audio_run_history(ep_id, "audio_pipeline", "succeeded", start_time, time.time(), log_lines)
                        return result, None
                    elif status == "error":
                        error_msg = msg.get("error", "Unknown error")
                        log_lines.append(f"ERROR: {error_msg}")
                        log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                        status_placeholder.error(f"❌ [LOCAL MODE] Audio pipeline failed: {error_msg}")
                        progress_text.caption(f"❌ Failed after {elapsed_str}")
                        result = {"status": "error", "error": error_msg, "logs": log_lines, "elapsed_seconds": elapsed_secs}
                        cache_logs(ep_id, "audio_pipeline", log_lines, "error", elapsed_secs)
                        append_audio_run_history(ep_id, "audio_pipeline", "error", start_time, time.time(), log_lines)
                        return result, error_msg
                    continue

                # Handle error phase
                if phase == "error":
                    error_msg = message or "Unknown error"
                    error_msg = _clean_msg(error_msg)
                    log_lines.append(f"ERROR: {error_msg}")
                    log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                    status_placeholder.error(f"❌ [LOCAL MODE] Audio pipeline failed: {error_msg}")
                    progress_text.caption(f"❌ Failed after {elapsed_str}")

                    result = {
                        "status": "error",
                        "error": error_msg,
                        "logs": log_lines,
                        "elapsed_seconds": elapsed_secs,
                    }
                    cache_logs(ep_id, "audio_pipeline", log_lines, "error", elapsed_secs)
                    append_audio_run_history(ep_id, "audio_pipeline", "error", start_time, time.time(), log_lines)
                    return result, error_msg

                # Handle completion
                if phase == "complete":
                    progress_bar.progress(1.0)
                    voice_clusters = msg.get("voice_clusters", 0)
                    labeled_voices = msg.get("labeled_voices", 0)
                    unlabeled_voices = msg.get("unlabeled_voices", 0)

                    completion_msg = f"Audio pipeline completed in {elapsed_str}"
                    if voice_clusters > 0:
                        completion_msg += f" - {voice_clusters} voice clusters"

                    log_lines.append(f"✅ {completion_msg}")
                    log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                    progress_text.caption(f"**100%** - {completion_msg}")
                    status_placeholder.success(f"✅ [LOCAL MODE] {completion_msg}")

                    result = {
                        "status": "succeeded",
                        "logs": log_lines,
                        "elapsed_seconds": elapsed_secs,
                        "voice_clusters": voice_clusters,
                        "labeled_voices": labeled_voices,
                        "unlabeled_voices": unlabeled_voices,
                    }
                    cache_logs(ep_id, "audio_pipeline", log_lines, "completed", elapsed_secs)
                    append_audio_run_history(ep_id, "audio_pipeline", "succeeded", start_time, time.time(), log_lines)
                    return result, None

                # Regular progress update
                progress_bar.progress(min(progress, 1.0))

                # Build progress text with step info
                pct = progress * 100
                parts = [f"**{pct:.1f}%**"]

                if step_order > 0 and total_steps > 0:
                    parts.append(f"Step {step_order}/{total_steps}:")

                parts.append(display_name)

                if step_progress > 0 and step_progress < 1:
                    parts.append(f"({step_progress*100:.0f}%)")

                parts.append(f"- {elapsed_str} elapsed")

                # Estimate remaining time
                if progress > 0.05 and progress < 1.0:
                    eta_secs = (elapsed_secs / progress) * (1 - progress)
                    if eta_secs > 60:
                        eta_min = int(eta_secs // 60)
                        eta_sec = int(eta_secs % 60)
                        parts.append(f"(~{eta_min}m {eta_sec}s remaining)")
                    else:
                        parts.append(f"(~{int(eta_secs)}s remaining)")

                progress_text.caption(" ".join(parts))

                # Add log line
                if message:
                    log_line = f"[{elapsed_str}] [{display_name}] {_clean_msg(message)}"
                else:
                    log_line = f"[{elapsed_str}] [{display_name}] Progress: {pct:.1f}%"

                log_lines.append(log_line)
                log_placeholder.code("\n".join(log_lines[-100:]), language="text")

            # If we get here without a complete/error, something went wrong
            elapsed_secs = time.time() - start_time
            error_msg = "Audio pipeline stream ended unexpectedly"
            log_lines.append(f"WARNING: {error_msg}")
            log_placeholder.code("\n".join(log_lines[-100:]), language="text")
            status_placeholder.warning(f"⚠️ [LOCAL MODE] {error_msg}")

            cache_logs(ep_id, "audio_pipeline", log_lines, "unknown", elapsed_secs)
            append_audio_run_history(ep_id, "audio_pipeline", "unknown", start_time, time.time(), log_lines)
            return {"status": "unknown", "logs": log_lines, "elapsed_seconds": elapsed_secs}, error_msg
    except requests.RequestException as exc:
        elapsed_secs = time.time() - start_time
        error_msg = describe_error(f"{_api_base()}{endpoint}", exc)
        log_lines.append(f"ERROR: {error_msg}")
        log_placeholder.code("\n".join(log_lines[-100:]), language="text")
        status_placeholder.error(f"❌ {error_msg}")
        cache_logs(ep_id, "audio_pipeline", log_lines, "error", elapsed_secs)
        append_audio_run_history(ep_id, "audio_pipeline", "error", start_time, time.time(), log_lines)
        return None, error_msg

    return {"status": "unknown", "logs": log_lines, "elapsed_seconds": elapsed_secs}, error_msg


def run_audio_pipeline_with_celery_streaming(
    ep_id: str,
    overwrite: bool = False,
    asr_provider: str = "openai_whisper",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> tuple[Dict[str, Any] | None, str | None]:
    """Queue audio pipeline via Celery and stream progress/logs without page reruns."""
    # Start job
    payload = {
        "ep_id": ep_id,
        "overwrite": overwrite,
        "asr_provider": asr_provider,
        "run_mode": "queue",
    }
    if min_speakers is not None:
        payload["min_speakers"] = min_speakers
    if max_speakers is not None:
        payload["max_speakers"] = max_speakers

    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    progress_text = st.empty()
    progress_text.caption("Queueing audio pipeline...")
    st.markdown("**Detailed Log:**")
    log_placeholder = st.empty()
    log_lines: List[str] = []
    pending_states = {"pending", "queued", "scheduled", "received"}
    fallback_to_local = False

    def _clean_msg(text: str) -> str:
        if not text:
            return text
        if "call apply_model on this" in text.lower():
            return "Demucs/MDX model noise suppressed"
        return text

    try:
        start_resp = requests.post(f"{_api_base()}/jobs/episode_audio_pipeline", json=payload, timeout=30)
        start_resp.raise_for_status()
        start_data = start_resp.json()
        job_id = start_data.get("job_id")
        if not job_id:
            status_placeholder.error(f"❌ Failed to start job: {start_data}")
            return None, "Failed to start job"
    except requests.RequestException as exc:
        err = describe_error(f"{_api_base()}/jobs/episode_audio_pipeline", exc)
        status_placeholder.error(f"❌ {err}")
        return None, err

    status_placeholder.info(f"⏳ [QUEUE MODE] Audio pipeline queued: {job_id}")
    log_lines.append(f"[QUEUE] Job queued: {job_id}")
    log_placeholder.code("\n".join(log_lines[-200:]), language="text")

    # Stream progress
    def _cancel_job(job_id: str) -> tuple[bool, str]:
        try:
            resp = requests.post(f"{_api_base()}/celery_jobs/{job_id}/cancel", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("status") in {"cancelled", "already_finished"}, data.get("status", "unknown")
        except requests.RequestException as exc:
            return False, describe_error(f"{_api_base()}/celery_jobs/{job_id}/cancel", exc)

    try:
        start_time = time.time()
        last_progress_ts = start_time

        # Pass ep_id to enable progress file polling for chain tasks
        with requests.get(
            f"{_api_base()}/celery_jobs/stream/{job_id}",
            params={"ep_id": ep_id},
            stream=True,
            timeout=None,
        ) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                try:
                    msg = json.loads(raw_line)
                except json.JSONDecodeError:
                    log_lines.append(raw_line)
                    log_placeholder.code("\n".join(log_lines[-200:]), language="text")
                    continue

                mtype = msg.get("type", "")
                if mtype == "progress":
                    progress = msg.get("progress", 0)
                    pct = progress * 100 if progress <= 1 else progress
                    progress_bar.progress(min(progress, 1.0))
                    step_name = msg.get("step_name") or msg.get("step") or "Running"
                    message = _clean_msg(msg.get("message", ""))
                    step_order = msg.get("step_order", 0)
                    total_steps = msg.get("total_steps", 9)
                    state = str(msg.get("state", "")).lower()
                    elapsed = int(time.time() - start_time)
                    elapsed_str = f"{elapsed//60}m {elapsed%60}s" if elapsed >= 60 else f"{elapsed}s"

                    if progress > 0:
                        last_progress_ts = time.time()

                    # If still queued/pending with zero progress for too long, fallback to local
                    if (state in pending_states and progress <= 0 and not fallback_to_local
                            and time.time() - start_time > 12):
                        fallback_to_local = True
                        warn_msg = "Queue is stuck >12s with 0% progress. Cancelling and running locally..."
                        log_lines.append(warn_msg)
                        log_placeholder.code("\n".join(log_lines[-200:]), language="text")
                        status_placeholder.warning(warn_msg)
                        _cancel_job(job_id)
                        return run_audio_pipeline_with_streaming(
                            ep_id=ep_id,
                            overwrite=overwrite,
                            asr_provider=asr_provider,
                        )

                    parts = [f"**{pct:.1f}%**"]
                    if step_order and total_steps:
                        parts.append(f"Step {step_order}/{total_steps}:")
                    parts.append(step_name.replace("_", " ").title())
                    if message:
                        parts.append(f"- {message}")
                    parts.append(f"({elapsed_str})")
                    progress_text.caption(" ".join(parts))

                    if message:
                        log_lines.append(f"[{elapsed_str}] [{step_name}] {message}")
                    else:
                        log_lines.append(f"[{elapsed_str}] [{step_name}] {pct:.1f}%")
                    log_placeholder.code("\n".join(log_lines[-200:]), language="text")

                elif mtype == "log":
                    line = _clean_msg(msg.get("line", ""))
                    if line:
                        log_lines.append(line)
                        log_placeholder.code("\n".join(log_lines[-200:]), language="text")

                elif mtype == "summary":
                    status = msg.get("status", "unknown")
                    elapsed = msg.get("elapsed_seconds", time.time() - start_time)
                    elapsed_str = f"{int(elapsed)//60}m {int(elapsed)%60}s" if elapsed >= 60 else f"{elapsed:.1f}s"
                    if status == "success" or status == "succeeded":
                        progress_bar.progress(1.0)
                        progress_text.caption(f"**100%** - Completed in {elapsed_str}")
                        status_placeholder.success(f"✅ [QUEUE MODE] Audio pipeline completed in {elapsed_str}")
                        log_lines.append(f"✅ Completed in {elapsed_str}")
                        log_placeholder.code("\n".join(log_lines[-200:]), language="text")
                        cache_logs(ep_id, "audio_pipeline", log_lines, "completed", elapsed)
                        append_audio_run_history(ep_id, "audio_pipeline", "succeeded", start_time, time.time(), log_lines)
                        return {"status": "succeeded", "elapsed_seconds": elapsed}, None
                    else:
                        error_msg = _clean_msg(msg.get("error", "Unknown error"))
                        progress_text.caption(f"❌ {error_msg}")
                        status_placeholder.error(f"❌ [QUEUE MODE] Audio pipeline failed: {error_msg}")
                        log_lines.append(f"ERROR: {error_msg}")
                        log_placeholder.code("\n".join(log_lines[-200:]), language="text")
                        cache_logs(ep_id, "audio_pipeline", log_lines, status, elapsed, extra={"error": error_msg})
                        append_audio_run_history(ep_id, "audio_pipeline", status, start_time, time.time(), log_lines)
                        return {"status": status, "error": error_msg, "elapsed_seconds": elapsed}, error_msg

                else:
                    # Unknown type, just log raw
                    log_lines.append(json.dumps(msg))
                    log_placeholder.code("\n".join(log_lines[-200:]), language="text")

            # Stream ended unexpectedly
            elapsed = time.time() - start_time
            warn_msg = "Audio pipeline stream ended unexpectedly"
            status_placeholder.warning(warn_msg)
            log_lines.append(f"WARNING: {warn_msg}")
            log_placeholder.code("\n".join(log_lines[-200:]), language="text")
            cache_logs(ep_id, "audio_pipeline", log_lines, "unknown", elapsed)
            append_audio_run_history(ep_id, "audio_pipeline", "unknown", start_time, time.time(), log_lines)
            return {"status": "unknown", "logs": log_lines, "elapsed_seconds": elapsed}, warn_msg

    except requests.RequestException as exc:
        err = describe_error(f"{_api_base()}/celery_jobs/stream/{job_id}", exc)
        status_placeholder.error(f"❌ {err}")
        log_lines.append(f"ERROR: {err}")
        log_placeholder.code("\n".join(log_lines[-200:]), language="text")
        append_audio_run_history(ep_id, "audio_pipeline", "error", start_time, time.time(), log_lines)
        return None, err


def run_incremental_with_streaming(
    ep_id: str,
    operation: str,
    endpoint: str,
    payload: Dict[str, Any],
    operation_display_name: str = None,
) -> tuple[Dict[str, Any] | None, str | None]:
    """Run an incremental audio operation with real-time streaming logs and progress.

    This function handles incremental operations (diarize_only, transcribe_only, voices_only)
    in local mode with streaming output:
    - Creates UI placeholders for status, progress bar, and logs
    - Makes a streaming POST request to the API
    - Updates progress and logs in real-time as data arrives
    - Returns final result when complete

    Args:
        ep_id: Episode identifier
        operation: Operation type (diarize_only, transcribe_only, voices_only)
        endpoint: API endpoint to call
        payload: Request payload dict
        operation_display_name: Human-readable name for display (auto-generated if None)

    Returns:
        Tuple of (result dict, error message or None)
    """
    display_name = operation_display_name or operation.replace("_", " ").title()

    # Create UI placeholders
    status_placeholder = st.empty()

    # Progress bar section
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        progress_text.caption(f"Starting {display_name}...")

    # Log section
    st.markdown("**Detailed Log:**")
    log_placeholder = st.empty()

    status_placeholder.info(f"⏳ [LOCAL MODE] Running {display_name} for {ep_id}...")
    log_lines: List[str] = []
    log_placeholder.code(
        f"[LOCAL MODE] Starting {display_name} for {ep_id}...\n"
        "Waiting for live logs...",
        language="text",
    )

    start_time = time.time()

    try:
        # Streaming request - reads lines as they arrive
        with requests.post(
            f"{_api_base()}{endpoint}",
            json=payload,
            stream=True,
            timeout=None,  # No timeout - operations can take a long time
        ) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                try:
                    msg = json.loads(raw_line)
                except json.JSONDecodeError:
                    # Treat non-JSON lines as plain log lines
                    log_lines.append(raw_line)
                    log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                    continue

                msg_type = msg.get("type", "")

                # Handle log messages
                if msg_type == "log":
                    line = msg.get("line", "")
                    if line:
                        log_lines.append(line)
                        log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                    continue

                # Handle error type
                if msg_type == "error":
                    error_msg = msg.get("message", "Unknown error")
                    elapsed_secs = time.time() - start_time
                    elapsed_str = f"{int(elapsed_secs)//60}m {int(elapsed_secs)%60}s" if elapsed_secs >= 60 else f"{elapsed_secs:.1f}s"
                    log_lines.append(f"ERROR: {error_msg}")
                    log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                    status_placeholder.error(f"❌ [LOCAL MODE] {display_name} failed: {error_msg}")
                    progress_text.caption(f"❌ Failed after {elapsed_str}")
                    append_audio_run_history(ep_id, operation, "error", start_time, time.time(), log_lines)
                    return {"status": "error", "error": error_msg, "logs": log_lines}, error_msg

                # Handle summary message (completion/error)
                if msg_type == "summary":
                    status = msg.get("status", "")
                    elapsed_secs = time.time() - start_time
                    elapsed_str = f"{int(elapsed_secs)//60}m {int(elapsed_secs)%60}s" if elapsed_secs >= 60 else f"{elapsed_secs:.1f}s"

                    if status in ("completed", "success", "succeeded"):
                        progress_bar.progress(1.0)
                        completion_msg = f"{display_name} completed in {elapsed_str}"
                        log_lines.append(f"✅ {completion_msg}")
                        log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                        progress_text.caption(f"**100%** - {completion_msg}")
                        status_placeholder.success(f"✅ [LOCAL MODE] {completion_msg}")
                        append_audio_run_history(ep_id, operation, "succeeded", start_time, time.time(), log_lines)
                        return {"status": "succeeded", "logs": log_lines, "elapsed_seconds": elapsed_secs}, None
                    else:
                        error_msg = msg.get("error", "Unknown error")
                        log_lines.append(f"ERROR: {error_msg}")
                        log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                        status_placeholder.error(f"❌ [LOCAL MODE] {display_name} failed: {error_msg}")
                        progress_text.caption(f"❌ Failed after {elapsed_str}")
                        append_audio_run_history(ep_id, operation, "error", start_time, time.time(), log_lines)
                        return {"status": "error", "error": error_msg, "logs": log_lines}, error_msg

                # Handle audio_progress messages (from diarize_only, transcribe_only, etc.)
                if msg_type == "audio_progress":
                    phase = msg.get("phase", "")
                    progress = msg.get("progress", 0)
                    message = msg.get("message", "")
                    step_name = msg.get("step_name", phase)

                    # Handle completion
                    if phase == "complete":
                        elapsed_secs = time.time() - start_time
                        elapsed_str = f"{int(elapsed_secs)//60}m {int(elapsed_secs)%60}s" if elapsed_secs >= 60 else f"{elapsed_secs:.1f}s"
                        progress_bar.progress(1.0)
                        completion_msg = message or f"{display_name} completed"
                        log_lines.append(f"✅ {completion_msg}")
                        log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                        progress_text.caption(f"**100%** - {completion_msg} ({elapsed_str})")
                        status_placeholder.success(f"✅ [LOCAL MODE] {completion_msg}")
                        append_audio_run_history(ep_id, operation, "succeeded", start_time, time.time(), log_lines)
                        return {"status": "succeeded", "logs": log_lines, "elapsed_seconds": elapsed_secs, **msg}, None

                    # Handle error
                    if phase == "error":
                        elapsed_secs = time.time() - start_time
                        elapsed_str = f"{int(elapsed_secs)//60}m {int(elapsed_secs)%60}s" if elapsed_secs >= 60 else f"{elapsed_secs:.1f}s"
                        error_msg = message or "Unknown error"
                        log_lines.append(f"ERROR: {error_msg}")
                        log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                        status_placeholder.error(f"❌ [LOCAL MODE] {display_name} failed: {error_msg}")
                        progress_text.caption(f"❌ Failed after {elapsed_str}")
                        append_audio_run_history(ep_id, operation, "error", start_time, time.time(), log_lines)
                        return {"status": "error", "error": error_msg, "logs": log_lines}, error_msg

                    # Update progress
                    if message:
                        log_lines.append(f"[{step_name}] {message}")
                        log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                    if progress > 0:
                        pct = progress * 100 if progress <= 1 else progress
                        progress_bar.progress(min(progress, 0.99))
                        elapsed_secs = time.time() - start_time
                        elapsed_str = f"{int(elapsed_secs)//60}m {int(elapsed_secs)%60}s" if elapsed_secs >= 60 else f"{elapsed_secs:.1f}s"
                        progress_text.caption(f"**{pct:.1f}%** - {step_name or phase}: {message or 'Working...'} ({elapsed_str})")
                    continue

                # Handle legacy progress messages (phase-based from audio_pipeline_run.py)
                phase = msg.get("phase", "")
                progress = msg.get("progress", 0)
                message = msg.get("message", "")
                step_name = msg.get("step_name", phase)

                if phase == "error":
                    error_msg = message or "Unknown error"
                    elapsed_secs = time.time() - start_time
                    elapsed_str = f"{int(elapsed_secs)//60}m {int(elapsed_secs)%60}s" if elapsed_secs >= 60 else f"{elapsed_secs:.1f}s"
                    log_lines.append(f"ERROR: {error_msg}")
                    log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                    status_placeholder.error(f"❌ [LOCAL MODE] {display_name} failed: {error_msg}")
                    progress_text.caption(f"❌ Failed after {elapsed_str}")
                    append_audio_run_history(ep_id, operation, "error", start_time, time.time(), log_lines)
                    return {"status": "error", "error": error_msg, "logs": log_lines}, error_msg

                if phase == "complete":
                    elapsed_secs = time.time() - start_time
                    elapsed_str = f"{int(elapsed_secs)//60}m {int(elapsed_secs)%60}s" if elapsed_secs >= 60 else f"{elapsed_secs:.1f}s"
                    progress_bar.progress(1.0)
                    completion_msg = message or f"{display_name} completed"
                    log_lines.append(f"✅ {completion_msg}")
                    log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                    progress_text.caption(f"**100%** - {completion_msg} ({elapsed_str})")
                    status_placeholder.success(f"✅ [LOCAL MODE] {completion_msg}")
                    append_audio_run_history(ep_id, operation, "succeeded", start_time, time.time(), log_lines)
                    return {"status": "succeeded", "logs": log_lines, "elapsed_seconds": elapsed_secs, **msg}, None

                # Update progress bar and text
                if phase:
                    pct = progress * 100 if progress <= 1 else progress
                    progress_bar.progress(min(progress, 0.99))

                    elapsed_secs = time.time() - start_time
                    elapsed_str = f"{int(elapsed_secs)//60}m {int(elapsed_secs)%60}s" if elapsed_secs >= 60 else f"{elapsed_secs:.1f}s"

                    progress_text.caption(f"**{pct:.1f}%** - {step_name or phase}: {message or 'Working...'} ({elapsed_str})")
                    if message:
                        log_lines.append(f"[{elapsed_str}] {message}")
                        log_placeholder.code("\n".join(log_lines[-100:]), language="text")

            # Stream ended unexpectedly (no completion message)
            elapsed_secs = time.time() - start_time
            elapsed_str = f"{int(elapsed_secs)//60}m {int(elapsed_secs)%60}s" if elapsed_secs >= 60 else f"{elapsed_secs:.1f}s"

            # Check if it was actually successful (sometimes stream just ends)
            if progress_bar._delta_generator._cursor.index > 0:  # Progress was made
                log_lines.append(f"Stream ended after {elapsed_str}")
                log_placeholder.code("\n".join(log_lines[-100:]), language="text")
                status_placeholder.warning(f"⚠️ [LOCAL MODE] {display_name} stream ended. Check results manually.")
                append_audio_run_history(ep_id, operation, "unknown", start_time, time.time(), log_lines)
                return {"status": "unknown", "logs": log_lines, "elapsed_seconds": elapsed_secs}, None

            warn_msg = f"{display_name} stream ended unexpectedly"
            status_placeholder.warning(warn_msg)
            log_lines.append(f"WARNING: {warn_msg}")
            log_placeholder.code("\n".join(log_lines[-100:]), language="text")
            append_audio_run_history(ep_id, operation, "unknown", start_time, time.time(), log_lines)
            return {"status": "unknown", "logs": log_lines, "elapsed_seconds": elapsed_secs}, warn_msg

    except requests.RequestException as exc:
        elapsed_secs = time.time() - start_time
        err = describe_error(f"{_api_base()}{endpoint}", exc)
        status_placeholder.error(f"❌ {err}")
        log_lines.append(f"ERROR: {err}")
        log_placeholder.code("\n".join(log_lines[-100:]), language="text")
        append_audio_run_history(ep_id, operation, "error", start_time, time.time(), log_lines)
        return None, err


def track_skeleton_html(num_frames: int = 12, thumb_width: int = 120) -> str:
    """Generate skeleton HTML for track detail loading state.

    Shows animated placeholder frames in a carousel-style layout while track data is being loaded.
    Uses the same 4:5 aspect ratio as the actual carousel.
    """
    thumb_height = int(thumb_width * 5 / 4)  # 4:5 aspect ratio
    frames = "".join([
        f'<div class="skeleton-frame" style="width:{thumb_width}px;height:{thumb_height}px;"></div>'
        for _ in range(num_frames)
    ])
    return f'''
    <style>
        .skeleton-carousel {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 16px 0;
        }}
        .skeleton-arrow {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: rgba(0,0,0,0.1);
            flex-shrink: 0;
        }}
        .skeleton-track {{
            display: flex;
            gap: 8px;
            overflow: hidden;
            flex: 1;
        }}
        .skeleton-frame {{
            flex-shrink: 0;
            border-radius: 6px;
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: skeleton-shimmer 1.5s infinite;
        }}
        @keyframes skeleton-shimmer {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
    </style>
    <div class="skeleton-carousel">
        <div class="skeleton-arrow"></div>
        <div class="skeleton-track">{frames}</div>
        <div class="skeleton-arrow"></div>
    </div>
    '''


def track_carousel_html(
    track_id: int,
    frames: List[Dict[str, Any]],
    *,
    thumb_width: int = 120,
    visible_count: int = 8,
) -> str:
    """Generate a carousel-style frame viewer with navigation arrows.

    Args:
        track_id: Track ID for unique element IDs
        frames: List of frame dicts with 'crop_url' or 'url' and 'frame_idx'
        thumb_width: Width of each thumbnail in pixels
        visible_count: Number of frames visible at once

    Returns:
        HTML string for the carousel with CSS and JavaScript
    """
    if not frames:
        return '<div class="track-carousel empty"><span>No frames available</span></div>'

    carousel_id = f"carousel_{track_id}"
    thumb_height = int(thumb_width * 5 / 4)  # 4:5 aspect ratio

    # Build frame thumbnails
    frame_items = []
    for i, frame in enumerate(frames):
        url = frame.get("crop_url") or frame.get("url") or frame.get("thumbnail_url")
        frame_idx = frame.get("frame_idx", i)
        similarity = frame.get("similarity")

        if url:
            src = html.escape(str(url))
            alt = html.escape(f"Frame {frame_idx}")

            # Add similarity badge if available
            badge = ""
            if similarity is not None:
                pct = int(similarity * 100)
                badge = f'<span class="frame-badge">{pct}%</span>'

            frame_items.append(f'''
                <div class="carousel-frame" data-idx="{frame_idx}">
                    <img src="{src}" alt="{alt}" loading="lazy" />
                    {badge}
                    <span class="frame-label">#{frame_idx}</span>
                </div>
            ''')

    frames_html = "".join(frame_items)
    total_frames = len(frame_items)

    return f'''
    <style>
        .track-carousel {{
            position: relative;
            width: 100%;
            margin: 16px 0;
        }}
        .track-carousel.empty {{
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100px;
            color: #666;
            background: #f5f5f5;
            border-radius: 8px;
        }}
        .carousel-container {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .carousel-arrow {{
            width: 36px;
            height: 36px;
            border: none;
            border-radius: 50%;
            background: rgba(0,0,0,0.6);
            color: white;
            font-size: 18px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            transition: background 0.2s, transform 0.1s;
            z-index: 10;
        }}
        .carousel-arrow:hover {{
            background: rgba(0,0,0,0.8);
            transform: scale(1.1);
        }}
        .carousel-arrow:disabled {{
            background: rgba(0,0,0,0.2);
            cursor: not-allowed;
            transform: none;
        }}
        .carousel-track {{
            display: flex;
            gap: 8px;
            overflow-x: auto;
            scroll-behavior: smooth;
            scrollbar-width: none;
            -ms-overflow-style: none;
            padding: 4px 0;
            flex: 1;
        }}
        .carousel-track::-webkit-scrollbar {{
            display: none;
        }}
        .carousel-frame {{
            position: relative;
            flex-shrink: 0;
            width: {thumb_width}px;
            height: {thumb_height}px;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            background: #f0f0f0;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .carousel-frame:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }}
        .carousel-frame img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        .carousel-frame .frame-badge {{
            position: absolute;
            top: 4px;
            right: 4px;
            background: rgba(33, 150, 243, 0.9);
            color: white;
            font-size: 10px;
            font-weight: bold;
            padding: 2px 5px;
            border-radius: 3px;
        }}
        .carousel-frame .frame-label {{
            position: absolute;
            bottom: 4px;
            left: 4px;
            background: rgba(0,0,0,0.6);
            color: white;
            font-size: 10px;
            padding: 2px 5px;
            border-radius: 3px;
        }}
        .carousel-info {{
            text-align: center;
            font-size: 12px;
            color: #666;
            margin-top: 8px;
        }}
    </style>

    <div class="track-carousel" id="{carousel_id}">
        <div class="carousel-container">
            <button class="carousel-arrow carousel-prev" onclick="scrollCarousel('{carousel_id}', -1)">◀</button>
            <div class="carousel-track">
                {frames_html}
            </div>
            <button class="carousel-arrow carousel-next" onclick="scrollCarousel('{carousel_id}', 1)">▶</button>
        </div>
        <div class="carousel-info">{total_frames} frames · Scroll or use arrows to navigate</div>
    </div>

    <script>
        function scrollCarousel(carouselId, direction) {{
            const carousel = document.getElementById(carouselId);
            const track = carousel.querySelector('.carousel-track');
            const frameWidth = {thumb_width} + 8; // width + gap
            const scrollAmount = frameWidth * 4; // scroll 4 frames at a time
            track.scrollBy({{ left: direction * scrollAmount, behavior: 'smooth' }});
        }}
    </script>
    '''


def cluster_track_row_carousel_html(
    cluster_id: str,
    track_id: int,
    frames: List[Dict[str, Any]],
    track_similarity: float | None = None,
    track_faces: int = 0,
    *,
    thumb_width: int = 100,
    max_visible: int = 12,
) -> str:
    """Generate a compact carousel row for a single track within a cluster view.

    Shows up to max_visible frames with arrows to scroll if more frames exist.
    Uses 4:5 aspect ratio thumbnails.

    Args:
        cluster_id: Cluster ID for unique element IDs
        track_id: Track ID for display and unique keys
        frames: List of frame dicts with 'crop_url' or 'url' and 'frame_idx'
        track_similarity: Track similarity score (0-1) for badge display
        track_faces: Total number of faces in track
        thumb_width: Width of each thumbnail in pixels
        max_visible: Maximum number of frames visible at once

    Returns:
        HTML string for the track row with carousel navigation
    """
    carousel_id = f"cluster_{cluster_id}_track_{track_id}"
    thumb_height = int(thumb_width * 5 / 4)  # 4:5 aspect ratio

    # Build similarity badge
    sim_badge = ""
    if track_similarity is not None:
        pct = int(track_similarity * 100)
        color = "#4CAF50" if pct >= 75 else "#FF9800" if pct >= 60 else "#F44336"
        sim_badge = f'<span style="background:{color};color:white;padding:2px 6px;border-radius:4px;font-size:11px;font-weight:bold;">{pct}%</span>'

    # Build frame thumbnails
    frame_items = []
    for i, frame in enumerate(frames):
        url = frame.get("crop_url") or frame.get("url") or frame.get("thumbnail_url") or frame.get("rep_thumb_url")
        frame_idx = frame.get("frame_idx", i)
        similarity = frame.get("similarity")

        if url:
            src = html.escape(str(url))
            alt_text = html.escape(f"Track {track_id} frame {frame_idx}")

            # Similarity badge for individual frame
            badge = ""
            if similarity is not None:
                fpct = int(similarity * 100)
                badge = f'<span class="ctr-frame-badge">{fpct}%</span>'

            frame_items.append(f'''
                <div class="ctr-frame">
                    <img src="{src}" alt="{alt_text}" loading="lazy" />
                    {badge}
                </div>
            ''')

    if not frame_items:
        # No frames - show placeholder
        return f'''
        <div class="ctr-row" id="{carousel_id}">
            <div class="ctr-header">
                <span class="ctr-track-label">Track {track_id}</span>
                {sim_badge}
                <span class="ctr-face-count">{track_faces} faces</span>
            </div>
            <div class="ctr-empty">No frames available</div>
        </div>
        '''

    frames_html = "".join(frame_items)
    total_frames = len(frame_items)
    show_arrows = total_frames > max_visible

    # Arrow buttons (only if needed)
    prev_btn = f'<button class="ctr-arrow ctr-prev" onclick="scrollClusterTrack(\'{carousel_id}\', -1)">◀</button>' if show_arrows else ''
    next_btn = f'<button class="ctr-arrow ctr-next" onclick="scrollClusterTrack(\'{carousel_id}\', 1)">▶</button>' if show_arrows else ''

    return f'''
    <div class="ctr-row" id="{carousel_id}">
        <div class="ctr-header">
            <span class="ctr-track-label">Track {track_id}</span>
            {sim_badge}
            <span class="ctr-face-count">{track_faces} faces</span>
        </div>
        <div class="ctr-carousel">
            {prev_btn}
            <div class="ctr-track">
                {frames_html}
            </div>
            {next_btn}
        </div>
    </div>
    '''


def cluster_track_rows_css() -> str:
    """Return CSS for cluster track row carousels. Include once per page."""
    return '''
    <style>
        .ctr-row {
            margin-bottom: 16px;
            padding: 12px;
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
        }
        .ctr-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        .ctr-track-label {
            font-weight: bold;
            font-size: 14px;
        }
        .ctr-face-count {
            color: #888;
            font-size: 12px;
            margin-left: auto;
        }
        .ctr-carousel {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .ctr-arrow {
            width: 32px;
            height: 32px;
            border: none;
            border-radius: 50%;
            background: rgba(100,100,100,0.6);
            color: white;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            transition: background 0.2s;
        }
        .ctr-arrow:hover {
            background: rgba(100,100,100,0.9);
        }
        .ctr-track {
            display: flex;
            gap: 8px;
            overflow-x: auto;
            scroll-behavior: smooth;
            scrollbar-width: none;
            -ms-overflow-style: none;
            padding: 4px 0;
            flex: 1;
        }
        .ctr-track::-webkit-scrollbar {
            display: none;
        }
        .ctr-frame {
            position: relative;
            flex-shrink: 0;
            width: 100px;
            height: 125px;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            background: #333;
        }
        .ctr-frame img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .ctr-frame-badge {
            position: absolute;
            top: 3px;
            right: 3px;
            background: rgba(33, 150, 243, 0.85);
            color: white;
            font-size: 9px;
            font-weight: bold;
            padding: 1px 4px;
            border-radius: 3px;
        }
        .ctr-empty {
            color: #666;
            font-style: italic;
            padding: 20px;
            text-align: center;
        }
    </style>
    <script>
        function scrollClusterTrack(rowId, direction) {
            const row = document.getElementById(rowId);
            if (!row) return;
            const track = row.querySelector('.ctr-track');
            if (!track) return;
            const frameWidth = 100 + 8; // width + gap
            const scrollAmount = frameWidth * 4; // scroll 4 frames at a time
            track.scrollBy({ left: direction * scrollAmount, behavior: 'smooth' });
        }
    </script>
    '''


def track_row_html(track_id: int, items: List[Dict[str, Any]], thumb_width: int = 200) -> str:
    if not items:
        return '<div class="track-grid empty">' "<span>No frames available for this track yet.</span>" "</div>"
    thumbs: List[str] = []
    for item in items:
        url = item.get("url")
        if not url:
            continue
        frame_idx = item.get("frame_idx", "?")
        alt_text = html.escape(f"Track {track_id} frame {frame_idx}")
        src = html.escape(str(url))
        thumbs.append(f'<img class="thumb" loading="lazy" src="{src}" alt="{alt_text}" />')
    if not thumbs:
        return '<div class="track-grid empty">' "<span>No frames available for this track yet.</span>" "</div>"
    thumbs_html = "".join(thumbs)
    return f"""
    <style>
      .track-grid {{
        display: grid;
        grid-template-columns: repeat(5, {thumb_width}px);
        gap: 12px;
        margin: 16px 0;
        padding: 12px;
        border: 1px solid #eee;
        border-radius: 8px;
        background: #fafafa;
      }}
      .track-grid.empty {{
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100px;
        color: #666;
      }}
      .track-grid .thumb {{
        width: {thumb_width}px;
        aspect-ratio: 4 / 5;
        object-fit: fill;
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
      }}
      .track-grid .thumb:hover {{
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        cursor: pointer;
      }}
    </style>
    <div class="track-grid">{thumbs_html}</div>
    """


def render_track_row(track_id: int, items: List[Dict[str, Any]], thumb_width: int = 200) -> None:
    html_block = track_row_html(track_id, items, thumb_width=thumb_width)
    # Calculate height: rows of thumbs (200px * 5/4 for 4:5 aspect) + gaps + padding
    thumb_height = int(thumb_width * 5 / 4)  # 4:5 aspect ratio means height = width * 5/4
    num_rows = (len(items) + 4) // 5  # Round up to get number of rows
    row_height = max(num_rows * thumb_height + (num_rows - 1) * 12 + 50, 150)  # thumbs + gaps + padding
    components.html(html_block, height=row_height, scrolling=False)


def inject_thumb_css() -> None:
    """Inject CSS for fixed 200×250 thumbnail frames."""
    st.markdown(
        f"""
    <style>
      .thumb {{
        width:{THUMB_W}px;
        height:{THUMB_H}px;
        border-radius:8px;
        background:#111;
        overflow:hidden;
        display:flex;
        align-items:center;
        justify-content:center;
      }}
      .thumb.thumb-hidden {{
        display:none !important;
      }}
      .thumb img {{
        width:100%;
        height:100%;
        object-fit:cover;
      }}
      .thumb-strip {{
        display:flex;
        gap:12px;
        flex-wrap:nowrap;
        align-items:center;
      }}
      .thumb-cell {{
        flex:0 0 {THUMB_W}px;
      }}
      .thumb-skeleton {{
        background:linear-gradient(90deg,#222 25%,#2b2b2b 37%,#222 63%);
        background-size:400% 100%;
        animation:pulse 1.2s ease-in-out infinite;
      }}
      @keyframes pulse {{
        0%{{background-position:100% 0}}
        100%{{background-position:-100% 0}}
      }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def inject_log_container_css() -> None:
    """Inject CSS to limit log container height and make scrollable.

    Prevents logs from extending the page indefinitely by adding max-height
    with overflow-y scroll to code blocks used for log display.
    """
    st.markdown(
        """
    <style>
        /* Limit log height and make scrollable */
        [data-testid="stCodeBlock"] {
            max-height: 400px;
            overflow-y: auto !important;
        }
        /* Ensure the pre inside also scrolls properly */
        [data-testid="stCodeBlock"] pre {
            max-height: 380px;
            overflow-y: auto !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def _thumb_async_cache() -> Dict[str, Dict[str, Any]]:
    cache = st.session_state.setdefault(_THUMB_CACHE_STATE_KEY, {})
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[_THUMB_CACHE_STATE_KEY] = cache
    return cache


def _thumb_job_state() -> Dict[str, str]:
    jobs = st.session_state.setdefault(_THUMB_JOB_STATE_KEY, {})
    if not isinstance(jobs, dict):
        jobs = {}
        st.session_state[_THUMB_JOB_STATE_KEY] = jobs
    return jobs


def thumb_is_pending(src: str | None) -> bool:
    if not src:
        return False
    entry = _thumb_async_cache().get(src)
    return bool(entry and entry.get("status") == "pending")


def _store_thumb_result(src: str, url: str | None, *, error: str | None = None) -> None:
    cache = _thumb_async_cache()
    cache[src] = {
        "status": "ready" if url else "error",
        "url": url,
        "error": error,
        "updated": time.time(),
    }
    jobs = _thumb_job_state()
    jobs.pop(src, None)


def _blocking_thumb_fetch(src: str, api_base: str, ttl: int = 3600) -> str | None:
    params = {"key": src, "ttl": ttl}
    inferred_mime = _infer_mime(src)
    response = requests.get(f"{api_base}/files/presign", params=params, timeout=3)
    response.raise_for_status()
    data = response.json()
    presigned_url = data.get("url")
    resolved_mime = data.get("content_type") or inferred_mime
    if not presigned_url:
        return None
    if presigned_url.startswith("https://"):
        return presigned_url
    img_response = requests.get(presigned_url, timeout=5)
    img_response.raise_for_status()
    encoded = base64.b64encode(img_response.content).decode("ascii")
    content_type = img_response.headers.get("content-type") or resolved_mime or "image/jpeg"
    return f"data:{content_type};base64,{encoded}"


def _thumb_worker_runner(src: str, api_base: str, ttl: int) -> None:
    ctx = get_script_run_ctx()
    if ctx is None:
        return
    error: str | None = None
    url: str | None = None
    try:
        url = _blocking_thumb_fetch(src, api_base, ttl=ttl)
    except requests.RequestException as exc:  # pragma: no cover - network errors
        error = str(exc)
        _diag("UI_RESOLVE_FAIL", src=src, reason="s3_presign_error", error=error)
    except Exception as exc:  # pragma: no cover - defensive guard
        error = str(exc)
        _diag("UI_RESOLVE_FAIL", src=src, reason="thumb_worker_error", error=error)
    _store_thumb_result(src, url, error=error)
    if ctx.script_requests:
        ctx.script_requests.request_rerun(
            RerunData(query_string=ctx.query_string, page_script_hash=ctx.page_script_hash)
        )


def _start_thumb_worker(src: str, api_base: str, ttl: int = 3600) -> None:
    jobs = _thumb_job_state()
    status = jobs.get(src)
    if status == "running":
        return
    running_now = sum(1 for value in jobs.values() if value == "running")
    if running_now >= _MAX_ASYNC_THUMB_WORKERS:
        jobs[src] = "queued"
        return
    ctx = get_script_run_ctx()
    if ctx is None or ctx.script_requests is None:
        err: str | None = None
        try:
            url = _blocking_thumb_fetch(src, api_base, ttl=ttl)
        except Exception as exc:  # pragma: no cover
            err = str(exc)
            _diag("UI_RESOLVE_FAIL", src=src, reason="s3_presign_error", error=err)
            url = None
        _store_thumb_result(src, url, error=err or (None if url else "presign_failed"))
        return
    jobs[src] = "running"
    worker = threading.Thread(target=_thumb_worker_runner, args=(src, api_base, ttl), daemon=True)
    add_script_run_ctx(worker, ctx)
    worker.start()


def _resolve_thumb_async(src: str, api_base: str) -> str | None:
    cache = _thumb_async_cache()
    entry = cache.get(src)
    if entry:
        status = entry.get("status")
        if status == "ready":
            return entry.get("url")
        if status == "error":
            cache.pop(src, None)
            entry = None
    if not entry:
        cache[src] = {"status": "pending", "url": None, "error": None, "updated": time.time()}
    _start_thumb_worker(src, api_base)
    return None


def resolve_thumb(src: str | None) -> str | None:
    """Resolve thumbnail source to a browser-safe URL or None for placeholder.

    Resolution order:
    1. Already a data URL → return as-is
    2. HTTPS URL (S3 presigned) → return as-is (CORS-safe)
    3. HTTP localhost API URL → fetch and convert to data URL
    4. Local file path exists → convert to data URL
    5. S3 key (artifacts/**) → presign via API asynchronously
    6. None → return None for placeholder
    """
    if not src:
        return None

    if isinstance(src, str) and src.startswith("data:"):
        return src
    if isinstance(src, str) and src.startswith("https://"):
        return src

    if isinstance(src, str) and src.startswith("http://localhost:"):
        try:
            response = requests.get(src, timeout=2)
            if response.ok and response.content:
                encoded = base64.b64encode(response.content).decode("ascii")
                content_type = response.headers.get("content-type") or _infer_mime(src) or "image/jpeg"
                return f"data:{content_type};base64,{encoded}"
        except Exception as exc:
            _diag("UI_RESOLVE_FAIL", src=src, reason="localhost_fetch_error", error=str(exc))

    try:
        path = Path(src)
        if path.exists() and path.is_file():
            converted = _data_url_cache(str(path))
            if converted:
                return converted
    except (OSError, ValueError) as exc:
        _diag("UI_RESOLVE_FAIL", src=src, reason="local_file_error", error=str(exc))

    if isinstance(src, str) and (
        src.startswith("artifacts/")
        or src.startswith("raw/")
        or ("/" in src and not src.startswith("/") and not src.startswith("http"))
    ):
        api_base = st.session_state.get("api_base") or _api_base()
        return _resolve_thumb_async(src, api_base)

    _diag("UI_RESOLVE_FAIL", src=src, reason="all_methods_failed")
    return None


@lru_cache(maxsize=1)
def _placeholder_thumb_url() -> str:
    return resolve_thumb(_PLACEHOLDER) or ensure_media_url(_PLACEHOLDER) or _PLACEHOLDER


def _infer_mime(name: str | None) -> str | None:
    if not name:
        return None
    lowered = name.lower()
    if lowered.endswith(".png"):
        return "image/png"
    if lowered.endswith(".webp"):
        return "image/webp"
    if lowered.endswith(".avif"):
        return "image/avif"
    if lowered.endswith(".gif"):
        return "image/gif"
    if lowered.endswith(".bmp"):
        return "image/bmp"
    if lowered.endswith(".tif") or lowered.endswith(".tiff"):
        return "image/tiff"
    return "image/jpeg"


def thumb_html(src: str | None, alt: str = "thumb", *, hide_if_missing: bool = False) -> str:
    """Generate HTML for a fixed 200×250 thumbnail frame.

    Args:
        src: Image source URL or path, or None for placeholder
        alt: Alt text for the image
        hide_if_missing: When True, do not render a placeholder and hide the frame if the
            image fails to load (404, revoked presign, etc.).

    Returns:
        HTML string for the thumbnail or an empty string when hiding missing images
    """
    placeholder = _placeholder_thumb_url()
    img = resolve_thumb(src)
    pending = thumb_is_pending(src)
    if not img:
        if hide_if_missing:
            return ""
        img = placeholder

    escaped_alt = html.escape(alt)
    if hide_if_missing:
        onerror_handler = "this.closest('.thumb').classList.add('thumb-hidden');"
    else:
        escaped_placeholder = placeholder.replace("'", "\\'")
        onerror_handler = f"this.onerror=null;this.src='{escaped_placeholder}';"
    wrapper_class = "thumb"
    if pending and img == placeholder:
        wrapper_class = "thumb thumb-skeleton"

    return (
        f'<div class="{wrapper_class}">'
        f'<img src="{img}" alt="{escaped_alt}" loading="lazy" decoding="async" onerror="{onerror_handler}"/>'
        "</div>"
    )
