from __future__ import annotations
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_screenalytics.artifacts import ensure_dirs, get_path

API_BASE_URL = os.environ.get("SCREENALYTICS_API_URL", "http://localhost:8000")
STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "local").lower()
STORAGE_BUCKET = (
    os.environ.get("AWS_S3_BUCKET")
    or os.environ.get("SCREENALYTICS_OBJECT_STORE_BUCKET")
    or ("local" if STORAGE_BACKEND == "local" else "")
)
DEFAULT_STRIDE = 5


def _describe_error(url: str, exc: requests.RequestException) -> str:
    detail = str(exc)
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            detail = exc.response.text or exc.response.reason or detail
        except Exception:  # pragma: no cover - best effort guard
            detail = str(exc)
    return f"{url} → {detail}"


def _check_api_health(base_url: str) -> tuple[bool, str | None]:
    url = f"{base_url}/healthz"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return True, None
    except requests.RequestException as exc:
        return False, _describe_error(url, exc)


def _post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _upload_file(url: str, data: bytes, headers: Dict[str, str] | None = None) -> None:
    resp = requests.put(url, data=data, headers=headers, timeout=120)
    resp.raise_for_status()


def _mirror_local(ep_id: str, data: bytes, local_path: str) -> Path:
    ensure_dirs(ep_id)
    dest = Path(local_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        handle.write(data)
    return dest


def _artifact_paths(ep_id: str) -> Dict[str, Path]:
    return {
        "video": get_path(ep_id, "video"),
        "detections": get_path(ep_id, "detections"),
        "tracks": get_path(ep_id, "tracks"),
    }


def _upload_target_hint(backend: str, bucket: str) -> str:
    ep_placeholder = "<ep_id>"
    if backend == "local":
        data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
        local_path = data_root / "videos" / ep_placeholder / "episode.mp4"
        return f"{local_path}"

    prefix = os.environ.get("AWS_S3_PREFIX", "raw/").strip("/")
    prefix_part = f"{prefix}/" if prefix else ""
    scheme = "s3"
    return f"{scheme}://{bucket}/{prefix_part}videos/{ep_placeholder}/episode.mp4"


st.set_page_config(page_title="Screenalytics Upload", page_icon="📺", layout="centered")
st.title("Screenalytics Upload")
st.caption("Create an episode, push footage to object storage, and kick the detect/track pipeline.")

st.sidebar.header("API connection")
st.sidebar.code(API_BASE_URL)
api_ready, api_error = _check_api_health(API_BASE_URL)
if api_ready:
    st.sidebar.success("API reachable")
else:
    st.sidebar.error(f"Health check failed: {api_error}")
sidebar_storage = f"Storage backend: {STORAGE_BACKEND}"
if STORAGE_BUCKET:
    sidebar_storage += f" | Bucket: {STORAGE_BUCKET}"
st.sidebar.write(sidebar_storage)
st.sidebar.caption(f"Uploads land at: {_upload_target_hint(STORAGE_BACKEND, STORAGE_BUCKET or '<bucket>')}")
if not api_ready:
    st.error(
        f"API not reachable at {API_BASE_URL}. Verify it is running and that /healthz is accessible."
    )
    if st.button("Retry health check"):
        st.experimental_rerun()
    st.stop()

with st.form("episode-upload"):
    show_ref = st.text_input("Show", placeholder="rhoslc", help="Slug or ID")
    season_number = st.number_input("Season", min_value=0, max_value=999, value=1, step=1)
    episode_number = st.number_input("Episode #", min_value=0, max_value=999, value=1, step=1)
    title = st.text_input("Title", placeholder="Don't Ice Me Bro", help="Optional episode title")
    include_air_date = st.checkbox("Set air date", value=False)
    air_date_value = st.date_input(
        "Air date",
        value=date.today(),
        disabled=not include_air_date,
        help="Optional premiere date",
    )
    uploaded_file = st.file_uploader("Episode video", type=["mp4"], accept_multiple_files=False)
    use_stub_detect = st.checkbox(
        "Use stub (fast, no ML)", value=False, help="Handy for quick smoke tests without YOLOv8."
    )
    submit = st.form_submit_button("Upload episode")

detect_resp: Dict[str, Any] | None = None
job_error: str | None = None

if submit:
    if not show_ref.strip():
        st.error("Show is required.")
        st.stop()
    if uploaded_file is None:
        st.error("Attach an .mp4 before submitting.")
        st.stop()

    air_date_payload = air_date_value.isoformat() if include_air_date else None
    payload = {
        "show_slug_or_id": show_ref.strip(),
        "season_number": int(season_number),
        "episode_number": int(episode_number),
        "title": title or None,
        "air_date": air_date_payload,
    }

    episodes_path = "/episodes"
    episodes_url = f"{API_BASE_URL}{episodes_path}"
    try:
        create_resp = _post_json(episodes_path, payload)
    except requests.RequestException as exc:
        st.error(f"Episode create failed: {_describe_error(episodes_url, exc)}")
        st.stop()

    ep_id = create_resp["ep_id"]
    st.info(f"Episode `{ep_id}` ready; requesting upload target…")

    presign_path = f"/episodes/{ep_id}/assets"
    presign_url = f"{API_BASE_URL}{presign_path}"
    try:
        presign_resp = _post_json(presign_path, {})
    except requests.RequestException as exc:
        st.error(f"Presign failed: {_describe_error(presign_url, exc)}")
        st.stop()

    raw_bytes = uploaded_file.getbuffer().tobytes()
    upload_url = presign_resp.get("upload_url")
    key = presign_resp.get("key") or presign_resp.get("object_key")
    st.info(
        f"Uploading to s3://{presign_resp['bucket']}/{key}" if upload_url else f"Writing to {presign_resp['local_video_path']}"
    )
    if upload_url:
        try:
            _upload_file(upload_url, raw_bytes, presign_resp.get("headers"))
        except requests.RequestException as exc:
            err = _describe_error(upload_url, exc)
            st.error(f"Upload failed: {err}")
            if "NoSuchBucket" in err:
                st.info("Bucket not found. Run scripts/s3_bootstrap.sh or set S3_AUTO_CREATE=1.")
            st.stop()

    local_video = _mirror_local(ep_id, raw_bytes, presign_resp["local_video_path"])
    st.success(f"Upload to object storage + local mirror complete for `{ep_id}`.")

    jobs_path = "/jobs/detect_track"
    jobs_url = f"{API_BASE_URL}{jobs_path}"
    job_payload = {"ep_id": ep_id, "stub": bool(use_stub_detect), "stride": DEFAULT_STRIDE}
    mode_label = "stub (no ML)" if use_stub_detect else "YOLOv8 + ByteTrack"
    spinner_label = f"Running detect/track ({mode_label})…"
    with st.spinner(spinner_label):
        try:
            detect_resp = _post_json(jobs_path, job_payload)
        except requests.RequestException as exc:
            job_error = _describe_error(jobs_url, exc)
    if detect_resp:
        st.success(
            f"Detect/track ({mode_label}) complete → detections: {detect_resp['detections_count']}, "
            f"tracks: {detect_resp['tracks_count']}"
        )
    elif job_error:
        st.error(f"Detect/track ({mode_label}) failed: {job_error}")

    artifacts = _artifact_paths(ep_id)
    st.subheader("Artifacts")
    st.code(str(local_video), language="bash")
    st.markdown(f"Detections → `{artifacts['detections']}`")
    st.markdown(f"Tracks → `{artifacts['tracks']}`")
    if job_error:
        st.caption("Re-run via POST /jobs/detect_track once the backend issue is resolved.")
