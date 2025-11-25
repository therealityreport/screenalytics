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
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import types

try:
    import streamlit as st
    import streamlit.components.v1 as components
    from streamlit.runtime.scriptrunner.script_run_context import (
        add_script_run_ctx,
        get_script_run_ctx,
    )
    from streamlit.runtime.scriptrunner.script_requests import RerunData
except Exception:
    class _ContextStub:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _StubSidebar:
        def __getattr__(self, name):  # noqa: ANN001
            def _noop(*args, **kwargs):
                return None

            return _noop

    class _StubStreamlit:
        def __init__(self) -> None:
            self.session_state: dict = {}
            self.query_params: dict = {}
            self.sidebar = _StubSidebar()

        def __getattr__(self, name):  # noqa: ANN001
            def _noop(*args, **kwargs):
                return None

            return _noop

        def expander(self, *args, **kwargs):
            return _ContextStub()

        def columns(self, n=1):
            count = max(int(n), 0)
            return tuple(_ContextStub() for _ in range(count))

    class _ComponentsStub:
        def __getattr__(self, name):  # noqa: ANN001
            def _noop(*args, **kwargs):
                return None

            return _noop

    st = _StubStreamlit()  # type: ignore[assignment]
    components = _ComponentsStub()

    def add_script_run_ctx(*args, **kwargs):  # type: ignore[override]
        return None

    def get_script_run_ctx(*args, **kwargs):  # type: ignore[override]
        return None

    class RerunData:  # type: ignore[override]
        pass

DEFAULT_TITLE = "SCREENALYTICS"
DATA_ROOT = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
DEFAULT_STRIDE = 6
DEFAULT_DETECTOR = "retinaface"
DEFAULT_TRACKER = "bytetrack"
DEFAULT_DEVICE = "coreml"
DEFAULT_DEVICE_LABEL = "CoreML"
DEFAULT_DET_THRESH = 0.65
DEFAULT_MAX_GAP = 60
DEFAULT_CLUSTER_SIMILARITY = float(os.environ.get("SCREENALYTICS_CLUSTER_SIM", "0.7"))
_LOCAL_MEDIA_CACHE_SIZE = 256

LOGGER = logging.getLogger(__name__)
DIAG = os.getenv("DIAG_LOG", "0") == "1"


def _diag(tag: str, **kw) -> None:
    """Diagnostic logger enabled via DIAG_LOG=1."""
    if DIAG:
        LOGGER.info("[DIAG:%s] %s", tag, json.dumps(kw, ensure_ascii=False))


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
    _env_float("BYTE_TRACK_HIGH_THRESH", 0.45),
)
TRACK_NEW_THRESH_DEFAULT = _env_float(
    "SCREENALYTICS_NEW_TRACK_THRESH",
    _env_float("BYTE_TRACK_NEW_TRACK_THRESH", 0.70),
)
TRACK_BUFFER_BASE_DEFAULT = max(
    _env_int("SCREENALYTICS_TRACK_BUFFER", _env_int("BYTE_TRACK_BUFFER", 30)),
    1,
)
MIN_BOX_AREA_DEFAULT = max(
    _env_float("SCREENALYTICS_MIN_BOX_AREA", _env_float("BYTE_TRACK_MIN_BOX_AREA", 20.0)),
    0.0,
)


# Thumbnail constants
THUMB_W, THUMB_H = 200, 250
_PLACEHOLDER = "apps/workspace-ui/assets/placeholder_face.svg"
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
SUPPORTED_PIPELINE_COMBOS = {
    "harvest": {(DEFAULT_DETECTOR, DEFAULT_TRACKER), (DEFAULT_DETECTOR, "strongsort")},
    "cluster": {(DEFAULT_DETECTOR, DEFAULT_TRACKER), (DEFAULT_DETECTOR, "strongsort")},
}
SCENE_DETECTOR_OPTIONS = [
    ("PySceneDetect (recommended)", "pyscenedetect"),
    ("HSV histogram (fallback)", "internal"),
    ("Disabled", "off"),
]
SCENE_DETECTOR_LABELS = [label for label, _ in SCENE_DETECTOR_OPTIONS]
SCENE_DETECTOR_VALUE_MAP = {label: value for label, value in SCENE_DETECTOR_OPTIONS}
SCENE_DETECTOR_LABEL_MAP = {value: label for label, value in SCENE_DETECTOR_OPTIONS}
_SCENE_DETECTOR_ENV = os.environ.get("SCENE_DETECTOR", "pyscenedetect").strip().lower()
SCENE_DETECTOR_DEFAULT = _SCENE_DETECTOR_ENV if _SCENE_DETECTOR_ENV in SCENE_DETECTOR_LABEL_MAP else "pyscenedetect"
PERF_PROFILE_PATH = Path(__file__).resolve().parents[2] / "config" / "pipeline" / "performance_profiles.yaml"
PROFILE_OPTIONS = [
    ("Low Power (recommended on laptops)", "low_power"),
    ("Balanced", "balanced"),
    ("High Accuracy (GPU)", "high_accuracy"),
]
PROFILE_LABELS = [label for label, _ in PROFILE_OPTIONS]
PROFILE_VALUE_MAP = {label: value for label, value in PROFILE_OPTIONS}
PROFILE_VALUE_TO_LABEL = {value: label for label, value in PROFILE_OPTIONS}
PROFILE_VALUE_TO_LABEL["fast_cpu"] = PROFILE_VALUE_TO_LABEL.get("low_power", "Low Power")
_EP_ID_REGEX = re.compile(r"^(?P<show>.+)-s(?P<season>\d{2})e(?P<episode>\d{2})$", re.IGNORECASE)
_CUSTOM_SHOWS_SESSION_KEY = "_custom_show_registry"


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


def _profile_alias(name: str | None) -> str | None:
    if not name:
        return None
    normalized = name.strip().lower()
    if normalized == "fast_cpu":
        return "low_power"
    return normalized


@lru_cache(maxsize=1)
def performance_profiles() -> Dict[str, Any]:
    if not PERF_PROFILE_PATH.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        LOGGER.warning("pyyaml not installed; skipping performance profile load")
        return {}
    try:
        with PERF_PROFILE_PATH.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # pragma: no cover - file/io specific
        LOGGER.warning("Failed to load performance profiles: %s", exc)
        return {}


def profile_defaults(profile_value: str | None) -> Dict[str, Any]:
    profile_key = _profile_alias(profile_value)
    if not profile_key:
        return {}
    profiles = performance_profiles()
    profile_cfg = profiles.get(profile_key, {}) if isinstance(profiles, dict) else {}
    if not isinstance(profile_cfg, dict):
        return {}
    return {
        "stride": profile_cfg.get("frame_stride"),
        "fps": profile_cfg.get("detection_fps_limit") or profile_cfg.get("max_fps"),
        "save_frames": profile_cfg.get("save_frames"),
        "save_crops": profile_cfg.get("save_crops"),
        "cpu_threads": profile_cfg.get("cpu_threads"),
    }


def profile_label_index(profile_value: str | None) -> int:
    normalized = _profile_alias(profile_value) or "balanced"
    label = PROFILE_VALUE_TO_LABEL.get(normalized, PROFILE_LABELS[1])
    try:
        return PROFILE_LABELS.index(label)
    except ValueError:
        return 0


def profile_label_from_value(profile_value: str | None) -> str:
    normalized = _profile_alias(profile_value)
    if normalized and normalized in PROFILE_VALUE_TO_LABEL:
        return PROFILE_VALUE_TO_LABEL[normalized]
    return PROFILE_LABELS[1]


def profile_value_from_state(value: str | None) -> str | None:
    """Interpret a profile value stored in session (label or key)."""
    if not value:
        return None
    if value in PROFILE_VALUE_TO_LABEL:
        return _profile_alias(value)
    return PROFILE_VALUE_MAP.get(value) or _profile_alias(value)


def default_profile_for_device(device_value: str | None) -> str:
    # Default to balanced everywhere; users can opt into low_power/high_accuracy explicitly.
    return "balanced"


def known_shows(include_session: bool = True) -> List[str]:
    """Return a sorted list of known show identifiers from episodes + S3 (plus session state)."""
    # Use case-insensitive dedupe so RHOSLC == rhoslc in dropdowns
    shows: dict[str, str] = {}

    def _remember(show_value: Any) -> None:
        if not show_value or not isinstance(show_value, str):
            return
        cleaned = show_value.strip()
        if not cleaned:
            return
        lowered = cleaned.lower()
        existing = shows.get(lowered)
        # Prefer the first seen value; if we later see a lowercase variant, adopt it
        if existing is None or existing != existing.lower():
            shows[lowered] = cleaned

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

    # Return values sorted case-insensitively but without duplicates by case
    return sorted(shows.values(), key=lambda value: value.lower())


def remember_custom_show(show_id: str) -> None:
    """Persist a show identifier in session state so dropdowns include it immediately."""
    cleaned = (show_id or "").strip()
    if not cleaned:
        return
    custom: List[str] = st.session_state.setdefault(_CUSTOM_SHOWS_SESSION_KEY, [])
    if cleaned not in custom:
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


# ============================================================================
# Unified Error Display Component
# ============================================================================

def show_error(
    message: str,
    *,
    context: str | None = None,
    severity: str = "error",
    show_timestamp: bool = False,
    details: str | None = None,
) -> None:
    """Display a consistent error message in the UI.

    Args:
        message: The main error message
        context: Optional context (e.g., "while loading cast data")
        severity: One of "error", "warning", "info"
        show_timestamp: Whether to include a timestamp
        details: Optional technical details (shown in expandable section)
    """
    # Build the message
    parts = []
    if show_timestamp:
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        parts.append(f"[{timestamp}]")
    if context:
        parts.append(f"{context}:")
    parts.append(message)
    full_message = " ".join(parts)

    # Display with appropriate severity
    if severity == "warning":
        st.warning(full_message)
    elif severity == "info":
        st.info(full_message)
    else:
        st.error(full_message)

    # Show technical details if provided
    if details:
        with st.expander("Technical details", expanded=False):
            st.code(details, language="text")


def show_api_error(
    exc: Exception,
    *,
    context: str | None = None,
    url: str | None = None,
) -> None:
    """Display an API error with consistent formatting.

    Args:
        exc: The exception that occurred
        context: Optional context (e.g., "loading cast members")
        url: Optional URL that was being accessed
    """
    if isinstance(exc, requests.HTTPError) and hasattr(exc, "response"):
        resp = exc.response
        status = resp.status_code if resp else "unknown"
        if status >= 500:
            message = f"Server error ({status}). Please try again later."
        elif status == 404:
            message = "Resource not found."
        elif status == 422:
            message = "Invalid request data."
        elif status >= 400:
            message = f"Request failed ({status})."
        else:
            message = str(exc)
    elif isinstance(exc, requests.Timeout):
        message = "Request timed out. Please try again."
    elif isinstance(exc, requests.ConnectionError):
        message = "Could not connect to server. Check your network connection."
    elif isinstance(exc, requests.RequestException):
        message = f"Network error: {exc}"
    else:
        message = str(exc)

    details = None
    if url:
        details = f"URL: {url}\n"
    if hasattr(exc, "__traceback__"):
        import traceback
        details = (details or "") + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    show_error(message, context=context, details=details)


def _api_base() -> str:
    base = st.session_state.get("api_base")
    if not base:
        raise RuntimeError("init_page() must be called before API access")
    return base


_PAGE_CONFIG_KEY = "_page_config_set"


def init_page(title: str = DEFAULT_TITLE) -> Dict[str, str]:
    if not st.session_state.get(_PAGE_CONFIG_KEY):
        st.set_page_config(page_title=title, layout="wide")
        st.session_state[_PAGE_CONFIG_KEY] = True
    api_base = st.session_state.get("api_base") or _env("SCREENALYTICS_API_URL", "http://localhost:8000")
    st.session_state.setdefault("api_base", api_base)
    backend = st.session_state.get("backend") or _env("STORAGE_BACKEND", "s3").lower()
    st.session_state.setdefault("backend", backend)
    bucket = st.session_state.get("bucket") or (
        _env("AWS_S3_BUCKET")
        or _env("SCREENALYTICS_OBJECT_STORE_BUCKET")
        or ("local" if backend == "local" else "screenalytics")
    )
    st.session_state.setdefault("bucket", bucket)

    query_params = st.query_params
    query_ep_id = query_params.get("ep_id", "")
    if query_ep_id and not st.session_state.get("_ep_id_query_origin"):
        st.session_state["_ep_id_query_origin"] = "external"
    stored_ep_id = st.session_state.get("ep_id")
    if stored_ep_id is None:
        st.session_state["ep_id"] = query_ep_id
    elif query_ep_id and query_ep_id != stored_ep_id:
        st.session_state["ep_id"] = query_ep_id
    elif stored_ep_id and (not query_ep_id or query_ep_id != stored_ep_id):
        params = st.query_params
        params["ep_id"] = stored_ep_id
        st.query_params = params
        st.session_state.setdefault("_ep_id_query_origin", "app")

    if "device_default_label" not in st.session_state:
        st.session_state["device_default_label"] = _guess_device_label()
    st.session_state.setdefault("detector_choice", DEFAULT_DETECTOR)
    st.session_state.setdefault("tracker_choice", DEFAULT_TRACKER)
    st.session_state.setdefault("scene_detector_choice", SCENE_DETECTOR_DEFAULT)

    sidebar = st.sidebar
    sidebar.header("API")
    sidebar.code(api_base)
    health_url = f"{api_base}/healthz"
    try:
        resp = requests.get(health_url, timeout=5)
        resp.raise_for_status()
        sidebar.success("/healthz OK")
    except requests.RequestException as exc:
        sidebar.error(describe_error(health_url, exc))
    sidebar.caption(f"Backend: {backend} | Bucket: {bucket}")

    # Render global episode selector in sidebar
    sidebar.divider()
    render_sidebar_episode_selector()

    return {
        "api_base": api_base,
        "backend": backend,
        "bucket": bucket,
        "ep_id": st.session_state.get("ep_id", ""),
    }


def set_ep_id(ep_id: str, rerun: bool = True, update_query_params: bool = True, origin: str | None = None) -> None:
    if not ep_id:
        return
    resolved_origin = origin or st.session_state.get("_ep_id_query_origin")
    if update_query_params and not resolved_origin:
        resolved_origin = "app"
    current = st.session_state.get("ep_id")
    if current == ep_id:
        if update_query_params:
            params = st.query_params
            if params.get("ep_id") != ep_id:
                params["ep_id"] = ep_id
                st.query_params = params
            if resolved_origin:
                st.session_state["_ep_id_query_origin"] = resolved_origin
        elif resolved_origin:
            st.session_state["_ep_id_query_origin"] = resolved_origin
        return
    st.session_state["ep_id"] = ep_id
    if update_query_params:
        params = st.query_params
        if params.get("ep_id") != ep_id:
            params["ep_id"] = ep_id
            st.query_params = params
        if resolved_origin:
            st.session_state["_ep_id_query_origin"] = resolved_origin
    elif resolved_origin:
        st.session_state["_ep_id_query_origin"] = resolved_origin
    if rerun:
        st.rerun()


def get_ep_id() -> str:
    return st.session_state.get("ep_id", "")


def get_ep_id_from_query_params(*, allow_app_injected: bool = True) -> str:
    """Return ep_id from query params, optionally ignoring app-injected values."""
    if "ep_id" not in st.query_params:
        return ""
    ep_value = st.query_params.get("ep_id") or ""
    if not ep_value:
        return ""
    origin = st.session_state.get("_ep_id_query_origin")
    if not allow_app_injected and origin == "app":
        return ""
    return str(ep_value)


def render_sidebar_episode_selector() -> str | None:
    """Render a global episode selector in the sidebar.

    Returns the selected ep_id, or None if no episodes exist.

    Behavior:
    - Defaults to most recently uploaded episode when no ep_id is set
    - Persists selection in st.session_state across page changes
    - Falls back to newest episode if previously selected ep_id no longer exists
    """
    try:
        episodes_payload = api_get("/episodes")
    except requests.RequestException:
        st.sidebar.info("API unavailable")
        return None

    episodes = episodes_payload.get("episodes", [])

    if not episodes:
        st.sidebar.info("No episodes yet — upload one to get started.")
        return None

    # Sort by created_at descending (most recent first)
    # Fallback to ep_id sorting if created_at is missing
    def sort_key(ep: Dict[str, Any]) -> tuple:
        created_at = ep.get("created_at", "")
        return (created_at, ep.get("ep_id", ""))

    episodes = sorted(episodes, key=sort_key, reverse=True)

    # Determine current ep_id
    current_ep_id = st.session_state.get("ep_id", "")
    ep_ids = [ep["ep_id"] for ep in episodes]

    # If current ep_id doesn't exist in list, fallback to most recent
    if current_ep_id and current_ep_id not in ep_ids:
        current_ep_id = ep_ids[0]  # Most recent episode
    elif not current_ep_id:
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

    # Render selectbox in sidebar
    selected_ep_id = st.sidebar.selectbox(
        "Episode",
        options=ep_ids,
        format_func=lambda eid: labels.get(eid, eid),
        index=current_index,
        key="global_episode_selector",
    )

    # Update session state if selection changed
    if selected_ep_id != current_ep_id:
        set_ep_id(selected_ep_id, rerun=False)

    # Explicit load button to apply selection and rerun the app
    if st.sidebar.button("Load Episode", key="load_episode_button"):
        set_ep_id(selected_ep_id, rerun=True, origin="sidebar_button")

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
    return resp.json() if resp.content else {}


def api_delete(path: str, **kwargs) -> Dict[str, Any]:
    base = st.session_state.get("api_base")
    if not base:
        raise RuntimeError("init_page() must be called before API access")
    timeout = kwargs.pop("timeout", 60)
    resp = requests.delete(f"{base}{path}", timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


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
        if exc.response.status_code != 404:
            raise

    return {
        "show": show,
        "seasons": seasons_resp.get("seasons", []),
        "episodes": episodes_resp.get("episodes", []),
        "cast": cast_rows,
    }


def _episode_status_payload(ep_id: str) -> Dict[str, Any] | None:
    url = f"{_api_base()}/episodes/{ep_id}/status"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    try:
        payload = resp.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def get_episode_status(ep_id: str) -> Dict[str, Any] | None:
    return _episode_status_payload(ep_id)


def get_track_images(ep_id: str, track_id: int) -> List[str]:
    """Return canonical track images (crops preferred, legacy thumbs fallback)."""

    try:
        payload = api_get(f"/episodes/{ep_id}/tracks/{int(track_id)}")
    except requests.RequestException:
        return []
    frames = payload.get("frames") or []
    urls: List[str] = []
    for frame in frames:
        candidate = frame.get("media_url") or frame.get("thumbnail_url")
        resolved = ensure_media_url(candidate)
        if resolved:
            urls.append(resolved)
    if not urls:
        resolved = ensure_media_url(payload.get("media_url") or payload.get("thumbnail_url"))
        if resolved:
            urls.append(resolved)
    return urls


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
    device_value = (device or DEFAULT_DEVICE).lower()
    profile_value = default_profile_for_device(device_value)
    profile_cfg = profile_defaults(profile_value)
    stride_default = profile_cfg.get("stride") if profile_cfg else None
    save_frames_default = profile_cfg.get("save_frames") if profile_cfg else None
    save_crops_default = profile_cfg.get("save_crops") if profile_cfg else None
    cpu_threads_default = profile_cfg.get("cpu_threads") if profile_cfg else None
    fps_default = profile_cfg.get("fps") if profile_cfg else None

    payload: Dict[str, Any] = {
        "ep_id": ep_id,
        "stride": int(stride if stride is not None else (stride_default or DEFAULT_STRIDE)),
        "device": device_value,
        "profile": profile_value,
        "detector": DEFAULT_DETECTOR,
        "tracker": DEFAULT_TRACKER,
        "det_thresh": float(det_thresh if det_thresh is not None else DEFAULT_DET_THRESH),
        "save_frames": bool(save_frames_default if save_frames_default is not None else True),
        "save_crops": bool(save_crops_default if save_crops_default is not None else True),
        "jpeg_quality": 85,
        "scene_detector": SCENE_DETECTOR_DEFAULT,
        "scene_threshold": SCENE_THRESHOLD_DEFAULT,
        "scene_min_len": SCENE_MIN_LEN_DEFAULT,
        "scene_warmup_dets": SCENE_WARMUP_DETS_DEFAULT,
    }
    if cpu_threads_default is not None:
        try:
            payload["cpu_threads"] = int(cpu_threads_default)
        except (TypeError, ValueError):
            pass
    if fps_default is not None:
        try:
            payload["fps"] = float(fps_default)
        except (TypeError, ValueError):
            pass
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
        "actions": ["split_tracks", "reembed", "recluster", "group_clusters"],
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
    try:
        st.switch_page(page_path)
        return
    except Exception:
        # Try common fallback: prepend pages/ if missing
        alt = f"pages/{page_path}" if not page_path.startswith("pages/") else page_path.replace("pages/", "", 1)
        try:
            st.switch_page(alt)
            return
        except Exception:
            st.info("Use the sidebar navigation to open the target page.")


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
        raw_payload = resp.json()
        jobs_block = raw_payload.get("jobs") if isinstance(raw_payload, dict) else raw_payload
        jobs = jobs_block if isinstance(jobs_block, list) else None
        if not jobs:
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


def _summary_from_status(ep_id: str, phase: str) -> Dict[str, Any] | None:
    payload = _episode_status_payload(ep_id)
    if not payload:
        return None

    # For detect_track, count files directly since status API doesn't include it
    if phase == "detect_track":
        from py_screenalytics.artifacts import get_path

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
    events_seen = False
    progress_events_seen = False
    try:
        for event_name, event_payload in iter_sse_events(response):
            if not isinstance(event_payload, dict):
                continue
            update_cb(event_payload)
            events_seen = True
            phase = str(event_payload.get("phase", "")).lower()
            if event_name != "log" and phase != "log":
                progress_events_seen = True
            summary_candidate = event_payload.get("summary")
            if isinstance(summary_candidate, dict) and _is_complete_summary(summary_candidate):
                final_summary = summary_candidate
            if event_name == "error" or phase == "error":
                return None, event_payload.get("error") or "Job failed", progress_events_seen
            if event_name == "done" or _is_phase_done(event_payload):
                if final_summary:
                    return final_summary, None, progress_events_seen
                if isinstance(summary_candidate, dict) and _is_complete_summary(summary_candidate):
                    return summary_candidate, None, progress_events_seen
                return None, None, progress_events_seen
    finally:
        response.close()
    if final_summary:
        return final_summary, None, progress_events_seen
    ep_id_value = payload.get("ep_id")
    if isinstance(ep_id_value, str):
        phase_hint = _phase_from_endpoint(endpoint_path) or "detect_track"
        status_summary = _summary_from_status(ep_id_value, phase_hint)
        if status_summary:
            return status_summary, None, progress_events_seen
    return None, None, progress_events_seen


def fallback_poll_progress(
    ep_id: str,
    payload: Dict[str, Any],
    *,
    update_cb,
    status_placeholder,
    job_started: bool,
    async_endpoint: str,
    force_async_start: bool = False,
    log_fn=None,
) -> tuple[Dict[str, Any] | None, str | None]:
    job_id: str | None = None

    def _read_stderr_tail(path: str | None, max_lines: int = 20) -> str | None:
        if not path:
            return None
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()
        except OSError:
            return None
        if not lines:
            return None
        return "".join(lines[-max_lines:]).strip()

    should_start_async = force_async_start or not job_started
    if should_start_async:
        try:
            resp = requests.post(f"{_api_base()}{async_endpoint}", json=payload, timeout=30)
            resp.raise_for_status()
            try:
                body = resp.json()
                if isinstance(body, dict):
                    job_id = body.get("job_id")
                    if log_fn and job_id:
                        log_fn(f"Async job id: {job_id} (via {async_endpoint})")
            except ValueError:
                job_id = None
        except requests.RequestException as exc:
            return None, describe_error(f"{_api_base()}{async_endpoint}", exc)
        if not job_id:
            return None, f"Async endpoint {async_endpoint} did not return a job_id"

    progress_url = f"{_api_base()}/episodes/{ep_id}/progress"
    job_progress_url = f"{_api_base()}/jobs/{job_id}/progress" if job_id else None
    job_detail_url = f"{_api_base()}/jobs/{job_id}" if job_id else None
    phase_hint = _phase_from_endpoint(async_endpoint) or "detect_track"
    last_progress: Dict[str, Any] | None = None
    poll_attempts = 0
    max_poll_attempts = 60  # 30 seconds (60 attempts * 0.5s sleep)
    while True:
        try:
            if job_progress_url:
                resp = requests.get(job_progress_url, timeout=5)
            else:
                resp = requests.get(progress_url, timeout=5)
            if resp.status_code == 404:
                poll_attempts += 1
                if poll_attempts > max_poll_attempts:
                    job_error = _fetch_async_job_error(ep_id, phase_hint) if not job_progress_url else None
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
            return None, describe_error(job_progress_url or progress_url, exc)
        payload_body = resp.json()
        progress = payload_body.get("progress") if isinstance(payload_body, dict) else None
        if progress is None and isinstance(payload_body, dict):
            progress = payload_body
        if isinstance(progress, dict) and progress:
            update_cb(progress)
            last_progress = progress
            phase = str(progress.get("phase", "")).lower()
            if phase == "error":
                return None, progress.get("error") or "Job failed"
            if _is_phase_done(progress):
                summary_block = progress.get("summary")
                if isinstance(summary_block, dict):
                    return summary_block, None
        state = str(payload_body.get("state", "")).lower() if isinstance(payload_body, dict) else ""
        if job_progress_url and state in {"succeeded", "failed", "canceled"}:
            if state != "succeeded":
                error_msg = payload_body.get("error") or f"Job {state}"
                if job_detail_url:
                    try:
                        job_detail = requests.get(job_detail_url, timeout=5).json()
                    except Exception:
                        job_detail = None
                    if isinstance(job_detail, dict):
                        stderr_tail = _read_stderr_tail(job_detail.get("stderr_log"))
                        if stderr_tail:
                            error_msg = f"{error_msg}\n\nstderr tail:\n{stderr_tail}"
                return None, error_msg
            summary_payload: Dict[str, Any] | None = None
            if isinstance(payload_body.get("track_metrics"), dict):
                summary_payload = {"track_metrics": payload_body["track_metrics"]}
            if not summary_payload and isinstance(progress, dict):
                summary_candidate = progress.get("summary") if isinstance(progress.get("summary"), dict) else progress
                if _is_complete_summary(summary_candidate):
                    summary_payload = summary_candidate
            if not summary_payload and job_detail_url:
                try:
                    job_detail = requests.get(job_detail_url, timeout=5).json()
                    if isinstance(job_detail, dict):
                        detail_summary = job_detail.get("summary") if isinstance(job_detail.get("summary"), dict) else None
                        if detail_summary:
                            summary_payload = detail_summary
                        elif isinstance(job_detail.get("track_metrics"), dict):
                            summary_payload = {"track_metrics": job_detail["track_metrics"]}
                        elif isinstance(progress, dict):
                            summary_payload = progress
                except Exception:
                    summary_payload = summary_payload or None
            return summary_payload or progress or payload_body, None
        if not job_progress_url and isinstance(progress, dict) and _is_phase_done(progress):
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
    # Final local-manifest fallback for detect/track counts
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
                log_fn=_append_log,
            )
        else:
            summary, error_message, progress_seen = attempt_sse_run(endpoint_path, payload, update_cb=_cb)
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
                force_async_start = bool(error_message) or not progress_seen
                fallback_summary, fallback_error = fallback_poll_progress(
                    ep_id,
                    payload,
                    update_cb=_cb,
                    status_placeholder=status_placeholder,
                    job_started=progress_seen,
                    async_endpoint=async_endpoint,
                    force_async_start=force_async_start,
                    log_fn=_append_log,
                )
                summary = fallback_summary if fallback_summary is not None else summary
                error_message = fallback_error or error_message
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
        if summary is None and not error_message:
            error_message = f"No progress or summary returned from {endpoint_path}; check API logs."
        return summary, error_message
    finally:
        st.session_state.pop(dedupe_key, None)
        st.session_state.pop(run_id_key, None)


def track_row_html(
    track_id: int,
    items: List[Dict[str, Any]],
    thumb_height: int = 200,
    thumb_width: int | None = None,
) -> str:
    if not items:
        return f'<div class="track-rail track-rail-{track_id} empty">' "<span>No frames available for this track yet.</span>" "</div>"
    thumbs: List[str] = []
    width_px = thumb_width if thumb_width is not None else thumb_height
    for item in items:
        url = item.get("url")
        if not url:
            continue
        frame_idx = item.get("frame_idx", "?")
        alt_text = html.escape(f"Track {track_id} frame {frame_idx}")
        src = html.escape(str(url))
        thumbs.append(f'<img class="thumb" loading="lazy" src="{src}" alt="{alt_text}" />')
    if not thumbs:
        return f'<div class="track-rail track-rail-{track_id} empty">' "<span>No frames available for this track yet.</span>" "</div>"
    thumbs_html = "".join(thumbs)
    return f"""
    <style>
      .track-grid {{
        display: grid;
        grid-template-columns: repeat(5, {width_px}px);
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
        width: {width_px}px;
        height: {thumb_height}px;
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
    <div class="track-rail track-rail-{track_id}"><div class="track-grid">{thumbs_html}</div></div>
    """


def render_track_row(track_id: int, items: List[Dict[str, Any]], thumb_width: int = 200) -> None:
    thumb_height = int(thumb_width * 5 / 4)
    html_block = track_row_html(track_id, items, thumb_height=thumb_height, thumb_width=thumb_width)
    # Calculate height: rows of thumbs (height) + gaps + padding
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
    raise_status = getattr(response, "raise_for_status", None)
    if callable(raise_status):
        raise_status()
    try:
        data = response.json()
    except (ValueError, TypeError) as exc:
        _diag("UI_RESOLVE_FAIL", src=src, reason="json_parse_error", error=str(exc))
        return None
    presigned_url = data.get("url") if isinstance(data, dict) else None
    resolved_mime = (data.get("content_type") if isinstance(data, dict) else None) or inferred_mime
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
        # Context lost - store error result to prevent stuck "pending" state
        _store_thumb_result(src, None, error="context_lost")
        _diag("UI_RESOLVE_FAIL", src=src, reason="context_lost", error="script context unavailable")
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
    """Queue async thumbnail fetch and return cached URL if ready, else None.

    Returns:
        Presigned URL if cached and ready, None if pending or error (placeholder should be shown).
    """
    cache = _thumb_async_cache()
    entry = cache.get(src)
    if entry:
        status = entry.get("status")
        if status == "ready":
            return entry.get("url")
        if status == "error":
            _diag("UI_THUMB_ASYNC", src=src, status="error_retry", error=entry.get("error"))
            cache.pop(src, None)
            entry = None
        elif status == "pending":
            # Already queued, waiting for worker to complete
            return None
    if not entry:
        cache[src] = {"status": "pending", "url": None, "error": None, "updated": time.time()}
        _diag("UI_THUMB_ASYNC", src=src, status="queued")
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
        presigned = _blocking_thumb_fetch(src, api_base)
        if presigned:
            return presigned
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
