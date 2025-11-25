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
