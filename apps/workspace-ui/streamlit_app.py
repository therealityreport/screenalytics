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
DEFAULT_STRIDE = 5


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


st.set_page_config(page_title="Screenalytics Upload", page_icon="📺", layout="centered")
st.title("Screenalytics Upload")
st.caption("Create an episode, push footage to MinIO, and kick the detect/track stub.")

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
    run_detect_track = st.checkbox("Run detect/track (stub)", value=True)
    run_detect_track = st.checkbox("Run detect/track (stub)", value=True)
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

    try:
        create_resp = _post_json("/episodes", payload)
    except requests.HTTPError as exc:
        st.error(f"Episode create failed: {exc.response.text}")
        st.stop()
    except requests.RequestException as exc:
        st.error(f"Episode create failed: {exc}")
        st.stop()

    ep_id = create_resp["ep_id"]
    st.info(f"Episode `{ep_id}` ready; requesting upload target…")

    try:
        presign_resp = _post_json(f"/episodes/{ep_id}/assets", {})
    except requests.HTTPError as exc:
        st.error(f"Presign failed: {exc.response.text}")
        st.stop()
    except requests.RequestException as exc:
        st.error(f"Presign failed: {exc}")
        st.stop()

    raw_bytes = uploaded_file.getbuffer().tobytes()
    try:
        _upload_file(presign_resp["upload_url"], raw_bytes, presign_resp.get("headers"))
    except requests.HTTPError as exc:
        st.error(f"Upload failed: {exc.response.text}")
        st.stop()
    except requests.RequestException as exc:
        st.error(f"Upload failed: {exc}")
        st.stop()

    local_video = _mirror_local(ep_id, raw_bytes, presign_resp["local_video_path"])
    st.success(f"Upload to MinIO + local mirror complete for `{ep_id}`.")

    if run_detect_track:
        st.write("Running detect/track stub…")
        try:
            detect_resp = _post_json(
                "/jobs/detect_track",
                {"ep_id": ep_id, "stub": True, "stride": DEFAULT_STRIDE},
            )
            st.success(
                f"Detect/track complete → detections: {detect_resp['detections_count']}, "
                f"tracks: {detect_resp['tracks_count']}"
            )
        except requests.HTTPError as exc:
            job_error = exc.response.text
            st.error(f"Detect/track failed: {exc.response.text}")
        except requests.RequestException as exc:
            job_error = str(exc)
            st.error(f"Detect/track failed: {exc}")

    artifacts = _artifact_paths(ep_id)
    st.subheader("Artifacts")
    st.code(str(local_video), language="bash")
    st.markdown(f"Detections → `{artifacts['detections']}`")
    st.markdown(f"Tracks → `{artifacts['tracks']}`")
    if not run_detect_track:
        st.caption("Run detect/track later via POST /jobs/detect_track.")
    elif job_error:
        st.caption("Detect/track stub failed; see errors above.")
