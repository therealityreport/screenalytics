"""Centralized Session State Manager for Streamlit UI.

Provides safe, typed access to session state with namespaced keys to prevent
collisions and KeyError crashes.
"""
from __future__ import annotations

from typing import Any, TypeVar, overload

try:
    import streamlit as st
except ImportError:
    st = None  # Allow importing for tests without streamlit


T = TypeVar("T")


# ============================================================================
# Session State Key Definitions
# ============================================================================

class SessionKeys:
    """Namespaced session state keys to prevent collisions."""

    # Cast page keys
    CAST_SHOW_SELECT = "cast_show_select"
    CAST_ACTIVE_SHOW = "cast_active_show"
    CAST_EDIT_ID = "cast_edit_id"
    CAST_EDIT_NAME = "cast_edit_name"
    CAST_EDIT_FULL_NAME = "cast_edit_full_name"
    CAST_SHOW_ADD_FORM = "cast_show_add_form"
    CAST_SHOW_UPLOAD_FORM = "cast_show_upload_form"
    NEW_CAST_CREATED = "new_cast_created"
    NEW_CAST_NAME_STORED = "new_cast_name_stored"
    SELECTED_CAST_ID = "selected_cast_id"

    # Episode keys
    EP_ID = "ep_id"
    PREV_EP_ID = "prev_ep_id"

    # Faces review keys
    FILTER_CAST_ID = "filter_cast_id"
    FILTER_CAST_NAME = "filter_cast_name"

    # API/Config keys
    API_BASE = "api_base"

    # Cache keys
    CAST_CAROUSEL_CACHE = "cast_carousel_cache"
    CAST_PEOPLE_CACHE = "cast_carousel_people_cache"
    TRACK_MEDIA_CACHE = "track_media_cache"
    THUMB_RESULT_CACHE = "_thumb_result_cache"
    THUMB_JOB_STATE = "_thumb_job_state"


# ============================================================================
# Session State Access Functions
# ============================================================================

def _get_session_state():
    """Get the Streamlit session state, or None if not available."""
    if st is None:
        return None
    try:
        return st.session_state
    except RuntimeError:
        # Not in a Streamlit context
        return None


@overload
def get(key: str) -> Any: ...

@overload
def get(key: str, default: T) -> T: ...

def get(key: str, default: Any = None) -> Any:
    """Safely get a value from session state.

    Args:
        key: The session state key
        default: Default value if key doesn't exist

    Returns:
        The value or default
    """
    state = _get_session_state()
    if state is None:
        return default
    return state.get(key, default)


def set(key: str, value: Any) -> None:
    """Set a value in session state.

    Args:
        key: The session state key
        value: The value to set
    """
    state = _get_session_state()
    if state is not None:
        state[key] = value


def delete(key: str) -> Any:
    """Safely delete a key from session state.

    Args:
        key: The session state key

    Returns:
        The deleted value, or None if key didn't exist
    """
    state = _get_session_state()
    if state is None:
        return None
    return state.pop(key, None)


def exists(key: str) -> bool:
    """Check if a key exists in session state.

    Args:
        key: The session state key

    Returns:
        True if key exists
    """
    state = _get_session_state()
    if state is None:
        return False
    return key in state


def setdefault(key: str, default: T) -> T:
    """Get a value, setting it to default if not exists.

    Args:
        key: The session state key
        default: Default value to set if key doesn't exist

    Returns:
        The existing or newly set value
    """
    state = _get_session_state()
    if state is None:
        return default
    return state.setdefault(key, default)


def require(key: str, error_msg: str | None = None) -> Any:
    """Get a value, raising error if it doesn't exist.

    Args:
        key: The session state key
        error_msg: Optional custom error message

    Returns:
        The value

    Raises:
        KeyError: If key doesn't exist
    """
    state = _get_session_state()
    if state is None:
        raise RuntimeError("Session state not available")
    if key not in state:
        raise KeyError(error_msg or f"Required session state key missing: {key}")
    return state[key]


def clear_prefix(prefix: str) -> int:
    """Clear all keys starting with a prefix.

    Args:
        prefix: The key prefix to match

    Returns:
        Number of keys deleted
    """
    state = _get_session_state()
    if state is None:
        return 0
    keys_to_delete = [k for k in state if k.startswith(prefix)]
    for key in keys_to_delete:
        state.pop(key, None)
    return len(keys_to_delete)


def get_int(key: str, default: int = 0) -> int:
    """Get an integer value from session state.

    Args:
        key: The session state key
        default: Default value if key doesn't exist or is invalid

    Returns:
        The integer value or default
    """
    value = get(key)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_str(key: str, default: str = "") -> str:
    """Get a string value from session state.

    Args:
        key: The session state key
        default: Default value if key doesn't exist

    Returns:
        The string value or default
    """
    value = get(key)
    if value is None:
        return default
    return str(value)


def get_bool(key: str, default: bool = False) -> bool:
    """Get a boolean value from session state.

    Args:
        key: The session state key
        default: Default value if key doesn't exist

    Returns:
        The boolean value or default
    """
    value = get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


# ============================================================================
# Celery Job State Management (Phase 2)
# ============================================================================

import time
from typing import Dict, Optional, Callable
import requests

# Job state session keys
ACTIVE_JOB_KEY = "celery_active_job"


def get_active_job() -> Optional[Dict[str, Any]]:
    """Get the currently active background job from session state.

    Returns:
        Dict with job_id, operation, episode_id, created_at, or None if no active job
    """
    return get(ACTIVE_JOB_KEY)


def set_active_job(job_id: str, operation: str, episode_id: str) -> None:
    """Store an active job in session state.

    Args:
        job_id: Celery task ID
        operation: Operation type (manual_assign, auto_group, etc.)
        episode_id: Episode ID the job is running for
    """
    set(ACTIVE_JOB_KEY, {
        "job_id": job_id,
        "operation": operation,
        "episode_id": episode_id,
        "created_at": time.time(),
    })


def clear_active_job() -> None:
    """Clear the active job from session state."""
    delete(ACTIVE_JOB_KEY)


def poll_job_status(api_base: str, job_id: str) -> Optional[Dict[str, Any]]:
    """Poll job status from the Celery jobs API.

    Args:
        api_base: API base URL (e.g., http://localhost:8000)
        job_id: Celery task ID

    Returns:
        Dict with job_id, state, result, etc., or None if request failed
    """
    try:
        resp = requests.get(f"{api_base}/celery_jobs/{job_id}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def render_job_status(api_base: str) -> bool:
    """Render active job status in Streamlit UI if one exists.

    This should be called at the top of the page after loading episode data.

    Args:
        api_base: API base URL

    Returns:
        True if a job is still running (page should auto-refresh),
        False if no job or job is complete
    """
    if st is None:
        return False

    job = get_active_job()
    if not job:
        return False

    status = poll_job_status(api_base, job["job_id"])
    if not status:
        st.warning(f"Unable to check job status: {job['job_id']}")
        return False

    state = status.get("state", "unknown")
    operation = job.get("operation", "operation")
    job_id_short = job["job_id"][:8]

    if state in ("queued", "in_progress"):
        # Show progress info
        progress = status.get("progress", {})
        step = progress.get("step", "")
        message = progress.get("message", "Working...")

        if progress:
            st.info(f"⏳ {operation} in progress ({job_id_short}...)\n\n**{step}**: {message}")
        else:
            st.info(f"⏳ {operation} in progress ({job_id_short}...)")

        # Signal that page should auto-refresh
        return True

    elif state == "success":
        result = status.get("result", {})
        succeeded = result.get("succeeded", 0)
        failed = result.get("failed", 0)

        if operation == "manual_assign":
            st.success(f"✅ Assignment complete! {succeeded} succeeded, {failed} failed")
        elif operation == "auto_group":
            assigned = result.get("assignments_count", 0)
            new_people = result.get("new_people_count", 0)
            st.success(f"✅ Auto-grouping complete! {assigned} clusters assigned, {new_people} new people created")
        else:
            st.success(f"✅ {operation} complete!")

        # Clear job and trigger cache invalidation
        clear_active_job()
        return False

    elif state == "failed":
        error = status.get("error", status.get("result", "Unknown error"))
        st.error(f"❌ {operation} failed: {error}")
        clear_active_job()
        return False

    elif state == "cancelled":
        st.warning(f"🚫 {operation} was cancelled")
        clear_active_job()
        return False

    return False


def submit_async_job(
    api_base: str,
    endpoint: str,
    payload: Dict[str, Any],
    operation: str,
    episode_id: str,
) -> Optional[str]:
    """Submit an async job to the API and store in session state.

    Args:
        api_base: API base URL
        endpoint: API endpoint (e.g., /episodes/{ep_id}/clusters/batch_assign_async)
        payload: Request payload
        operation: Operation name for display
        episode_id: Episode ID

    Returns:
        Job ID if submitted successfully, None otherwise
    """
    if st is None:
        return None

    # Check if a job is already running
    existing = get_active_job()
    if existing:
        st.warning(f"A job is already running: {existing.get('operation')} ({existing.get('job_id', '')[:8]}...)")
        return None

    try:
        resp = requests.post(
            f"{api_base}{endpoint}",
            json=payload,
            timeout=10,
        )
        if resp.status_code in (200, 202):
            data = resp.json()
            job_id = data.get("job_id")
            if job_id and data.get("async", True):
                # Store job in session state
                set_active_job(job_id, operation, episode_id)
                return job_id
            elif not data.get("async"):
                # Synchronous fallback - no job to track
                return None
        else:
            error = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            st.error(f"Failed to submit job: {error}")
            return None
    except Exception as e:
        st.error(f"Failed to submit job: {e}")
        return None


# Convenience function for auto-refresh polling
def should_auto_refresh() -> bool:
    """Check if page should auto-refresh due to running job.

    Call this after render_job_status() to determine if st.rerun() should be scheduled.
    """
    job = get_active_job()
    return job is not None
