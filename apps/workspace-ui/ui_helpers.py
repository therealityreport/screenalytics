from __future__ import annotations

import base64
import html
import json
import logging
import math
import mimetypes
import numbers
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import streamlit as st
import streamlit.components.v1 as components

DEFAULT_TITLE = "SCREENALYTICS"
DATA_ROOT = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
DEFAULT_STRIDE = 3
DEFAULT_DETECTOR = "retinaface"
DEFAULT_TRACKER = "bytetrack"
DEFAULT_DEVICE = "auto"
DEFAULT_DEVICE_LABEL = "Auto"
DEFAULT_DET_THRESH = 0.5
DEFAULT_MAX_GAP = 30
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
    _env_float("BYTE_TRACK_HIGH_THRESH", 0.5),
)
TRACK_NEW_THRESH_DEFAULT = _env_float(
    "SCREENALYTICS_NEW_TRACK_THRESH",
    _env_float("BYTE_TRACK_NEW_TRACK_THRESH", 0.5),
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

LABEL = {
    DEFAULT_DETECTOR: "RetinaFace (recommended)",
    DEFAULT_TRACKER: "ByteTrack (default)",
    "strongsort": "StrongSORT (ReID)",
}
DEVICE_LABELS = ["Auto", "CPU", "MPS", "CUDA"]
DEVICE_VALUE_MAP = {"Auto": "auto", "CPU": "cpu", "MPS": "mps", "CUDA": "cuda"}
DEVICE_VALUE_TO_LABEL = {value.lower(): label for label, value in DEVICE_VALUE_MAP.items()}
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
SCENE_DETECTOR_OPTIONS = [
    ("PySceneDetect (recommended)", "pyscenedetect"),
    ("HSV histogram (fallback)", "internal"),
    ("Disabled", "off"),
]
SCENE_DETECTOR_LABELS = [label for label, _ in SCENE_DETECTOR_OPTIONS]
SCENE_DETECTOR_VALUE_MAP = {label: value for label, value in SCENE_DETECTOR_OPTIONS}
SCENE_DETECTOR_LABEL_MAP = {value: label for label, value in SCENE_DETECTOR_OPTIONS}
_SCENE_DETECTOR_ENV = os.environ.get("SCENE_DETECTOR", "pyscenedetect").strip().lower()
SCENE_DETECTOR_DEFAULT = (
    _SCENE_DETECTOR_ENV if _SCENE_DETECTOR_ENV in SCENE_DETECTOR_LABEL_MAP else "pyscenedetect"
)
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


def known_shows(include_session: bool = True) -> List[str]:
    """Return a sorted list of known show identifiers from episodes + S3 (plus session state)."""
    shows: set[str] = set()

    def _remember(show_value: Any) -> None:
        if not show_value or not isinstance(show_value, str):
            return
        cleaned = show_value.strip()
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

    return sorted(shows, key=lambda value: value.lower())


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


def _api_base() -> str:
    base = st.session_state.get("api_base")
    if not base:
        raise RuntimeError("init_page() must be called before API access")
    return base


def init_page(title: str = DEFAULT_TITLE) -> Dict[str, str]:
    st.set_page_config(page_title=title, layout="wide")
    api_base = st.session_state.get("api_base") or _env("SCREENALYTICS_API_URL", "http://localhost:8000")
    st.session_state.setdefault("api_base", api_base)
    backend = st.session_state.get("backend") or _env("STORAGE_BACKEND", "local").lower()
    st.session_state.setdefault("backend", backend)
    bucket = st.session_state.get("bucket") or (
        _env("AWS_S3_BUCKET")
        or _env("SCREENALYTICS_OBJECT_STORE_BUCKET")
        or ("local" if backend == "local" else "")
    )
    st.session_state.setdefault("bucket", bucket)

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

    # Cache clear button
    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear Python Cache", help="Clear .pyc files and __pycache__ directories"):
        import subprocess
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


def api_delete(path: str, **kwargs) -> Dict[str, Any]:
    base = st.session_state.get("api_base")
    if not base:
        raise RuntimeError("init_page() must be called before API access")
    timeout = kwargs.pop("timeout", 60)
    resp = requests.delete(f"{base}{path}", timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp.json()


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
    label = DEVICE_VALUE_TO_LABEL.get(value.lower())
    if label:
        return label
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
    payload: Dict[str, Any] = {
        "ep_id": ep_id,
        "stride": int(stride if stride is not None else DEFAULT_STRIDE),
        "device": (device or DEFAULT_DEVICE).lower(),
        "detector": DEFAULT_DETECTOR,
        "tracker": DEFAULT_TRACKER,
        "det_thresh": float(det_thresh if det_thresh is not None else DEFAULT_DET_THRESH),
        "save_frames": True,
        "save_crops": True,
        "jpeg_quality": 85,
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
        "actions": ["split_tracks", "reembed", "recluster", "group_clusters"],
        "write_back": True,
    }


def _guess_device_label() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():  # pragma: no cover
            return "Auto"
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


def detector_is_face_only(ep_id: str) -> bool:
    detector = tracks_detector_value(ep_id)
    if detector is None:
        return False
    return detector.lower() in FACE_ONLY_DETECTORS


def tracks_detector_label(ep_id: str) -> str:
    return detector_label_from_value(tracks_detector_value(ep_id))


def tracks_tracker_label(ep_id: str) -> str:
    return tracker_label_from_value(tracks_tracker_value(ep_id))


def try_switch_page(page_path: str) -> None:
    try:
        st.switch_page(page_path)
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


def iter_sse_events(response: requests.Response) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
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
                    return None, job_error or f"Job initialization timed out after {max_poll_attempts * 0.5:.0f}s (progress file never created)"
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
            st.image(image_url, use_column_width=True)
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
            st.image(image_url, use_column_width=True)
        st.markdown(f"**{title}**")
        st.caption(caption)
        if extra:
            extra()
    return card


def frame_card(title: str, image_url: str | None, extra=None):
    card = st.container(border=True)
    with card:
        if image_url:
            st.image(image_url, use_column_width=True)
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
                    _append_log(
                        f"Request failed before any realtime updates ({_mode_context()}): {error_message}"
                    )
                elif summary:
                    _append_log(
                        f"Server returned a synchronous summary before streaming updates ({_mode_context()})."
                    )
                else:
                    fallback_hint = (
                        f"; checking async fallback via {async_endpoint}"
                        if async_endpoint
                        else ""
                    )
                    _append_log(
                        f"No realtime events received yet from {endpoint_path} ({_mode_context()}){fallback_hint}."
                    )
            if async_endpoint and (summary is None or error_message):
                _append_log(
                    f"Falling back to async endpoint {async_endpoint} for {endpoint_path} ({_mode_context()})…"
                )
                fallback_summary, fallback_error = fallback_poll_progress(
                    ep_id,
                    payload,
                    update_cb=_cb,
                    status_placeholder=status_placeholder,
                    job_started=job_started,
                    async_endpoint=async_endpoint,
                )
                if fallback_summary is not None or fallback_error is None:
                    summary = fallback_summary
                    error_message = fallback_error
                if fallback_error:
                    _append_log(
                        f"Async endpoint {async_endpoint} reported an error ({_mode_context()}): {fallback_error}"
                    )
                elif fallback_summary:
                    _append_log(
                        f"Async endpoint {async_endpoint} returned a summary ({_mode_context()})."
                    )
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


def track_row_html(track_id: int, items: List[Dict[str, Any]], thumb_width: int = 200) -> str:
    if not items:
        return (
            "<div class=\"track-grid empty\">"
            "<span>No frames available for this track yet.</span>"
            "</div>"
        )
    thumbs: List[str] = []
    for item in items:
        url = item.get("url")
        if not url:
            continue
        frame_idx = item.get("frame_idx", "?")
        alt_text = html.escape(f"Track {track_id} frame {frame_idx}")
        src = html.escape(str(url))
        thumbs.append(
            f'<img class="thumb" loading="lazy" src="{src}" alt="{alt_text}" />'
        )
    if not thumbs:
        return (
            "<div class=\"track-grid empty\">"
            "<span>No frames available for this track yet.</span>"
            "</div>"
        )
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
    st.markdown(f"""
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
    """, unsafe_allow_html=True)


def resolve_thumb(src: str | None) -> str | None:
    """Resolve thumbnail source to a browser-safe URL or None for placeholder.

    Resolution order:
    1. Already a data URL → return as-is
    2. HTTPS URL (S3 presigned) → return as-is (CORS-safe)
    3. HTTP localhost API URL → fetch and convert to data URL
    4. Local file path exists → convert to data URL
    5. S3 key (artifacts/**) → presign via API → fetch and convert to data URL
    6. None → return None for placeholder
    """
    if not src:
        return None
    
    # Already a data URL? Return as-is
    if isinstance(src, str) and src.startswith("data:"):
        return src
    
    # HTTPS URLs (S3 presigned) can be loaded directly - return as-is
    if isinstance(src, str) and src.startswith("https://"):
        return src
    
    # HTTP localhost API URLs need to be fetched and converted to data URLs
    # (Streamlit at :8501 can't load images from :8000 without CORS)
    if isinstance(src, str) and src.startswith("http://localhost:"):
        try:
            response = requests.get(src, timeout=2)
            if response.ok and response.content:
                encoded = base64.b64encode(response.content).decode("ascii")
                content_type = response.headers.get("content-type") or _infer_mime(src) or "image/jpeg"
                return f"data:{content_type};base64,{encoded}"
        except Exception as exc:
            _diag("UI_RESOLVE_FAIL", src=src, reason="localhost_fetch_error", error=str(exc))
    
    # Try as local file path
    try:
        path = Path(src)
        if path.exists() and path.is_file():
            converted = _data_url_cache(str(path))
            if converted:
                return converted
    except (OSError, ValueError) as exc:
        _diag("UI_RESOLVE_FAIL", src=src, reason="local_file_error", error=str(exc))
    
    # Try as S3 key - call presign endpoint and then fetch
    if isinstance(src, str) and (
        src.startswith("artifacts/") or 
        src.startswith("raw/") or
        ("/" in src and not src.startswith("/") and not src.startswith("http"))
    ):
        try:
            api_base = st.session_state.get("api_base") or "http://localhost:8000"
            params = {"key": src, "ttl": 3600}
            inferred_mime = _infer_mime(src)
            response = requests.get(f"{api_base}/files/presign", params=params, timeout=2)
            if response.ok:
                data = response.json()
                presigned_url = data.get("url")
                resolved_mime = data.get("content_type") or inferred_mime
                if presigned_url:
                    # For HTTPS presigned URLs, return directly
                    if presigned_url.startswith("https://"):
                        return presigned_url
                    # For HTTP URLs, fetch and convert to data URL
                    img_response = requests.get(presigned_url, timeout=5)
                    if img_response.ok and img_response.content:
                        encoded = base64.b64encode(img_response.content).decode("ascii")
                        content_type = img_response.headers.get("content-type") or resolved_mime or "image/jpeg"
                        return f"data:{content_type};base64,{encoded}"
        except Exception as exc:
            _diag("UI_RESOLVE_FAIL", src=src, reason="s3_presign_error", error=str(exc))

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

    return (
        '<div class="thumb">'
        f'<img src="{img}" alt="{escaped_alt}" loading="lazy" decoding="async" onerror="{onerror_handler}"/>'
        "</div>"
    )
