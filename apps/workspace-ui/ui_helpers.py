from __future__ import annotations

import os
import re
import json
import time
import html
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

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
LABEL = {
    DEFAULT_DETECTOR: "RetinaFace (recommended)",
    "yolov8face": "YOLOv8-face (alt)",
    DEFAULT_TRACKER: "ByteTrack (default)",
    "strongsort": "StrongSORT (ReID)",
}
DEVICE_LABELS = ["Auto", "CPU", "MPS", "CUDA"]
DEVICE_VALUE_MAP = {"Auto": "auto", "CPU": "cpu", "MPS": "mps", "CUDA": "cuda"}
DETECTOR_OPTIONS = [
    ("RetinaFace (recommended)", DEFAULT_DETECTOR),
    ("YOLOv8-face (alt)", "yolov8face"),
]
DETECTOR_LABELS = [label for label, _ in DETECTOR_OPTIONS]
DETECTOR_VALUE_MAP = {label: value for label, value in DETECTOR_OPTIONS}
DETECTOR_LABEL_MAP = {value: label for label, value in DETECTOR_OPTIONS}
FACE_ONLY_DETECTORS = {"retinaface", "yolov8face"}
TRACKER_OPTIONS = [
    ("ByteTrack (default)", DEFAULT_TRACKER),
    ("StrongSORT (ReID)", "strongsort"),
]
TRACKER_LABELS = [label for label, _ in TRACKER_OPTIONS]
TRACKER_VALUE_MAP = {label: value for label, value in TRACKER_OPTIONS}
TRACKER_LABEL_MAP = {value: label for label, value in TRACKER_OPTIONS}
_EP_ID_REGEX = re.compile(r"^(?P<show>.+)-s(?P<season>\d{2})e(?P<episode>\d{2})$", re.IGNORECASE)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


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


def link_local(path: Path | str) -> str:
    return f"`{path}`"


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


def default_detect_track_payload(
    ep_id: str,
    *,
    stub: bool = False,
    stride: int | None = None,
    device: str | None = None,
    det_thresh: float | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ep_id": ep_id,
        "stub": bool(stub),
        "stride": int(stride if stride is not None else DEFAULT_STRIDE),
        "device": (device or DEFAULT_DEVICE).lower(),
        "detector": DEFAULT_DETECTOR,
        "tracker": DEFAULT_TRACKER,
        "det_thresh": float(det_thresh if det_thresh is not None else DEFAULT_DET_THRESH),
    }
    return payload


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
        path = _manifest_path(ep_id, "tracks.jsonl")
        return not path.exists()
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
) -> None:
    ratio = progress_ratio(progress)
    progress_bar.progress(ratio)
    secs_total = total_seconds_hint(progress)
    secs_done = progress.get("secs_done") or progress.get("elapsed_sec")
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
    status_placeholder.write(
        f"{format_mmss(secs_done)} / {format_mmss(secs_total)} • phase={phase} • detector={detector_label} • tracker={tracker_label} • device={device_text} • fps={fps_text}"
    )
    detail_placeholder.caption(
        f"Frames {progress.get('frames_done', 0):,} / {progress.get('frames_total') or '?'}"
    )


def attempt_sse_run(
    endpoint_path: str,
    payload: Dict[str, Any],
    *,
    update_cb,
) -> tuple[Dict[str, Any] | None, str | None, bool]:
    url = f"{_api_base()}{endpoint_path}"
    headers = {"Accept": "text/event-stream"}
    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=(5, 60))
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
            if isinstance(event_payload.get("summary"), dict):
                final_summary = event_payload["summary"]
            phase = str(event_payload.get("phase", "")).lower()
            if event_name == "error" or phase == "error":
                return None, event_payload.get("error") or "Job failed", True
            if event_name == "done" or phase == "done":
                return final_summary or event_payload.get("summary"), None, True
    finally:
        response.close()
    return final_summary, None, True


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
    while True:
        try:
            resp = requests.get(progress_url, timeout=5)
            if resp.status_code == 404:
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
        phase = str(progress.get("phase", "")).lower()
        if phase == "error":
            return None, progress.get("error") or "Job failed"
        if phase == "done" and isinstance(progress.get("summary"), dict):
            return progress["summary"], None
        time.sleep(0.5)


def normalize_summary(ep_id: str, raw: Dict[str, Any] | None) -> Dict[str, Any]:
    summary = raw or {}
    if "summary" in summary and isinstance(summary["summary"], dict):
        summary = summary["summary"]
    artifacts = summary.setdefault("artifacts", {})
    local = artifacts.setdefault("local", {})
    manifests_dir = DATA_ROOT / "manifests" / ep_id
    local.setdefault("detections", str(manifests_dir / "detections.jsonl"))
    local.setdefault("tracks", str(manifests_dir / "tracks.jsonl"))
    local.setdefault("faces", str(manifests_dir / "faces.jsonl"))
    local.setdefault("identities", str(manifests_dir / "identities.json"))
    if "detections" not in summary and summary.get("detections_count") is not None:
        summary["detections"] = summary.get("detections_count")
    if "tracks" not in summary and summary.get("tracks_count") is not None:
        summary["tracks"] = summary.get("tracks_count")
    if "faces" not in summary and summary.get("faces_count") is not None:
        summary["faces"] = summary.get("faces_count")
    if "identities" not in summary and summary.get("identities_count") is not None:
        summary["identities"] = summary.get("identities_count")
    return summary


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
    async_endpoint: str | None,
    requested_detector: str | None = None,
    requested_tracker: str | None = None,
):
    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()
    detail_placeholder = st.empty()

    def _cb(progress: Dict[str, Any]) -> None:
        if progress.get("detector"):
            remember_detector(progress.get("detector"))
        update_progress_display(
            progress,
            progress_bar=progress_bar,
            status_placeholder=status_placeholder,
            detail_placeholder=detail_placeholder,
            requested_device=requested_device,
            requested_detector=requested_detector,
            requested_tracker=requested_tracker,
        )

    summary, error_message, job_started = attempt_sse_run(endpoint_path, payload, update_cb=_cb)
    if (error_message or summary is None) and not error_message and async_endpoint:
        summary, error_message = fallback_poll_progress(
            ep_id,
            payload,
            update_cb=_cb,
            status_placeholder=status_placeholder,
            job_started=job_started,
            async_endpoint=async_endpoint,
        )
    if summary and isinstance(summary, dict):
        remember_detector(summary.get("detector"))
        remember_tracker(summary.get("tracker"))
    return summary, error_message


def track_row_html(track_id: int, items: List[Dict[str, Any]], thumb_height: int = 200) -> str:
    rail_id = f"rail-{track_id}"
    if not items:
        return (
            "<div class=\"track-row empty\">"
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
            "<div class=\"track-row empty\">"
            "<span>No frames available for this track yet.</span>"
            "</div>"
        )
    thumb_css = f"height:{thumb_height}px;"
    thumbs_html = "".join(thumbs)
    return f"""
    <style>
      .track-row {{ display:flex; align-items:center; gap:8px; margin:8px 0; }}
      .track-row button {{ border:1px solid #d0d7de; border-radius:4px; background:#fff; cursor:pointer; }}
      .track-row button:focus {{ outline:2px solid #0b5fff; }}
      .track-row .rail {{ overflow-x:auto; scroll-behavior:smooth; white-space:nowrap; border:1px solid #eee; border-radius:6px; padding:6px; flex:1; }}
      .track-row .thumb {{ {thumb_css} aspect-ratio:4 / 5; object-fit:cover; display:inline-block; margin-right:6px; }}
    </style>
    <div class="track-row">
      <button class="arrow" onclick="document.getElementById('{rail_id}').scrollBy({{left:-400, behavior:'smooth'}})">&larr;</button>
      <div id="{rail_id}" class="rail">{thumbs_html}</div>
      <button class="arrow" onclick="document.getElementById('{rail_id}').scrollBy({{left:400, behavior:'smooth'}})">&rarr;</button>
    </div>
    """


def render_track_row(track_id: int, items: List[Dict[str, Any]], thumb_height: int = 200) -> None:
    html_block = track_row_html(track_id, items, thumb_height=thumb_height)
    row_height = thumb_height + 80
    components.html(html_block, height=row_height, scrolling=False)
