"""Upload Video page - upload new episodes and browse existing S3 videos."""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict

import requests

# Ensure AWS credentials path is set for boto3
if "AWS_SHARED_CREDENTIALS_FILE" not in os.environ:
    home = os.path.expanduser("~")
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = f"{home}/.aws/credentials"
if "AWS_CONFIG_FILE" not in os.environ:
    home = os.path.expanduser("~")
    os.environ["AWS_CONFIG_FILE"] = f"{home}/.aws/config"

os.environ.setdefault("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", "5120")

import streamlit as st

# Path setup for imports - page is in apps/workspace-ui/pages/
PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parent.parent  # apps/workspace-ui
PROJECT_ROOT = PAGE_PATH.parents[3]  # SCREENALYTICS root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

from py_screenalytics.artifacts import ensure_dirs, get_path  # noqa: E402

try:  # noqa: E402 - prefer shared helper when available
    from apps.api.services.storage import (
        parse_v2_episode_key as _shared_parse_v2_episode_key,
    )
except ImportError:  # pragma: no cover - UI fallback when API helper missing
    _V2_KEY_RE = re.compile(
        r"raw/videos/(?P<show>[^/]+)/s(?P<season>\d{2})/e(?P<episode>\d{2})/episode\.mp4",
        re.IGNORECASE,
    )

    def _shared_parse_v2_episode_key(key: str) -> Dict[str, object] | None:
        match = _V2_KEY_RE.search(key)
        if not match:
            return None
        show = match.group("show")
        season = int(match.group("season"))
        episode = int(match.group("episode"))
        return {
            "ep_id": f"{show.lower()}-s{season:02d}e{episode:02d}",
            "show": show,
            "show_slug": show,
            "season": season,
            "episode": episode,
            "key_version": "v2",
        }


def parse_v2_episode_key(key: str) -> Dict[str, object] | None:
    return _shared_parse_v2_episode_key(key)


import ui_helpers as helpers  # noqa: E402

# NOTE: Do NOT access st.* here before helpers.init_page()
# Streamlit requires st.set_page_config to be the first Streamlit call.


def _get_video_meta(ep_id: str) -> Dict[str, Any] | None:
    try:
        return helpers.api_get(f"/episodes/{ep_id}/video_meta")
    except requests.RequestException as exc:
        st.warning(helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id}/video_meta", exc))
        return None


ASYNC_JOBS_KEY = "async_jobs"
ADD_SHOW_OPTION = "+ Add new show..."


def _job_state() -> Dict[str, Dict[str, Any]]:
    return st.session_state.setdefault(ASYNC_JOBS_KEY, {})


def _register_async_job(
    job_resp: Dict[str, Any],
    *,
    ep_id: str,
    label: str,
    stride: int,
    fps: float | None,
    device: str,
) -> None:
    jobs = _job_state()
    job_id = job_resp["job_id"]
    artifacts = job_resp.get("artifacts") or {
        "video": str(get_path(ep_id, "video")),
        "detections": str(get_path(ep_id, "detections")),
        "tracks": str(get_path(ep_id, "tracks")),
    }
    jobs[job_id] = {
        "job_id": job_id,
        "ep_id": ep_id,
        "label": label,
        "requested_stride": stride,
        "requested_fps": fps,
        "requested_device": device,
        "artifacts": artifacts,
    }


def _render_job_sections() -> bool:
    jobs = _job_state()
    if not jobs:
        return False
    st.subheader("Detect/track progress")
    any_running = False
    for job_id, meta in list(jobs.items()):
        running = _render_single_job(job_id, meta, jobs)
        any_running = any_running or running
        st.divider()
    return any_running


def _render_single_job(job_id: str, meta: Dict[str, Any], jobs: Dict[str, Dict[str, Any]]) -> bool:
    st.markdown(f"**{meta.get('label', f'Job {job_id}')}**")
    st.caption(f"Job ID: `{job_id}` - Episode `{meta.get('ep_id')}`")
    try:
        progress_resp = helpers.api_get(f"/jobs/{job_id}/progress", timeout=10)
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}/jobs/{job_id}/progress", exc))
        if st.button("Dismiss", key=f"dismiss-{job_id}"):
            jobs.pop(job_id, None)
            st.rerun()
        return False
    progress = progress_resp.get("progress") or {}
    state = progress_resp.get("state", "unknown")
    frames_done = progress.get("frames_done") or 0
    frames_total = progress.get("frames_total") or 0
    ratio = helpers.progress_ratio(progress)
    st.progress(ratio)
    total_seconds = helpers.total_seconds_hint(progress)
    eta = helpers.eta_seconds(progress)
    device_label = progress.get("device") or meta.get("requested_device", "auto")
    fps_value = progress.get("fps_infer") or progress.get("analyzed_fps") or progress.get("fps_detected")
    fps_text = f"{fps_value:.2f} fps" if fps_value else "--"
    elapsed_seconds = progress.get("secs_done") or progress.get("elapsed_sec")
    info_text = (
        f"Elapsed: {helpers.format_mmss(elapsed_seconds)} - "
        f"Total: {helpers.format_mmss(total_seconds)} - "
        f"ETA: {helpers.format_mmss(eta)} - "
        f"Device: {device_label} - FPS: {fps_text}"
    )
    st.caption(info_text)
    st.caption(f"Frames {frames_done:,} / {frames_total or '?'}")

    if state == "running":
        cancel_col, _ = st.columns([1, 3])
        with cancel_col:
            if st.button("Cancel job", key=f"cancel-{job_id}"):
                try:
                    helpers.api_post(f"/jobs/{job_id}/cancel")
                except requests.RequestException as exc:
                    st.error(helpers.describe_error(f"{cfg['api_base']}/jobs/{job_id}/cancel", exc))
                else:
                    st.info("Cancel requested...")
                    st.rerun()
        return True

    detail = {}
    try:
        detail = helpers.api_get(f"/jobs/{job_id}", timeout=10)
    except requests.RequestException as exc:
        st.warning(helpers.describe_error(f"{cfg['api_base']}/jobs/{job_id}", exc))

    summary = detail.get("summary") if detail else None
    error_msg = detail.get("error") if detail else None
    if state == "succeeded":
        counts = []
        if summary:
            normalized = helpers.normalize_summary(meta.get("ep_id", ""), summary)
            det_count = helpers.coerce_int(normalized.get("detections"))
            trk_count = helpers.coerce_int(normalized.get("tracks"))
            if det_count is not None:
                counts.append(f"detections: {det_count:,}")
            if trk_count is not None:
                counts.append(f"tracks: {trk_count:,}")
        msg = "Completed" + (" - " + ", ".join(counts) if counts else "")
        st.success(msg or "Job succeeded")
    elif state == "canceled":
        st.warning("Job canceled.")
    else:
        fallback_err = error_msg or "Job failed without error detail."
        st.error(f"Job failed: {fallback_err}")
    artifacts = meta.get("artifacts") or {}
    artifact_line = " | ".join(
        [
            f"Video -> {helpers.link_local(artifacts.get('video', get_path(meta['ep_id'], 'video')))}",
            f"Detections -> {helpers.link_local(artifacts.get('detections', get_path(meta['ep_id'], 'detections')))}",
            f"Tracks -> {helpers.link_local(artifacts.get('tracks', get_path(meta['ep_id'], 'tracks')))}",
        ]
    )
    st.caption(artifact_line)
    prefixes = helpers.episode_artifact_prefixes(meta.get("ep_id", ""))
    if prefixes:
        st.caption(
            "S3 -> "
            f"Frames {helpers.s3_uri(prefixes['frames'])} | "
            f"Crops {helpers.s3_uri(prefixes['crops'])} | "
            f"Manifests {helpers.s3_uri(prefixes['manifests'])}"
        )
    if st.button("Dismiss result", key=f"dismiss-{job_id}"):
        jobs.pop(job_id, None)
        st.rerun()
    return False


def _launch_default_detect_track(ep_id: str, *, label: str) -> Dict[str, Any] | None:
    payload = helpers.default_detect_track_payload(ep_id)
    payload["stride"] = helpers.DEFAULT_STRIDE
    payload["device"] = helpers.DEFAULT_DEVICE
    endpoint = "/jobs/detect_track_async"
    try:
        job_resp = helpers.api_post(endpoint, payload)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response else None
        if status in {404, 405, 501}:
            try:
                resp = helpers.api_post("/jobs/detect_track", payload)
            except requests.RequestException as sync_exc:
                st.error(helpers.describe_error(f"{cfg['api_base']}/jobs/detect_track", sync_exc))
                return None
            else:
                st.info("Async detect/track not available; ran synchronously.")
                resp.setdefault("ep_id", ep_id)
                return {"sync": resp, "payload": payload}
        st.error(helpers.describe_error(f"{cfg['api_base']}{endpoint}", exc))
        return None
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}{endpoint}", exc))
        return None
    _register_async_job(
        job_resp,
        ep_id=ep_id,
        label=label,
        stride=payload.get("stride", helpers.DEFAULT_STRIDE),
        fps=payload.get("fps"),
        device=payload.get("device", helpers.DEFAULT_DEVICE),
    )
    st.success(f"Job `{job_resp['job_id']}` queued; monitor progress above.")
    job_resp["payload"] = payload
    return {"job": job_resp, "payload": payload}


cfg = helpers.init_page("Screenalytics Upload")

# Handle ep_id query parameter AFTER init_page (set_page_config must be first st call)
# If ?ep_id=<id> is present, enter "replace existing episode" mode
# If no ep_id, clear any lingering state for fresh upload
_replace_mode_ep_id: str | None = None
_query_ep_id = st.query_params.get("ep_id")
if _query_ep_id:
    # Keep ep_id in query params and session state for replace mode
    _replace_mode_ep_id = str(_query_ep_id)
    st.session_state["ep_id"] = _replace_mode_ep_id
    st.session_state["_ep_id_query_origin"] = True
else:
    # Clear lingering episode state for fresh upload
    st.session_state.pop("ep_id", None)
    st.session_state.pop("_ep_id_query_origin", None)
    st.session_state.pop("upload_ep_params_cleaned", None)

st.title("Upload & Run")

# Handle deferred navigation after state flush
if st.session_state.get("navigate_to_detail"):
    st.session_state.pop("navigate_to_detail")
    helpers.try_switch_page("pages/2_Episode_Detail.py")

if "detector" in st.session_state:
    del st.session_state["detector"]
if "tracker" in st.session_state:
    del st.session_state["tracker"]
st.cache_data.clear()

flash_message = st.session_state.pop("upload_flash", None)
if flash_message:
    st.success(flash_message)

# Show replace mode banner if ep_id was provided in query params
if _replace_mode_ep_id:
    st.warning(
        f"**Replace Mode**: Uploading will replace the video for episode `{_replace_mode_ep_id}`. "
        "Existing artifacts will be overwritten."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Cancel Replace", key="cancel_replace_mode"):
            # Remove ep_id from query params and reload
            st.query_params.pop("ep_id", None)
            st.session_state.pop("ep_id", None)
            st.rerun()
    st.divider()

jobs_running = _render_job_sections()


def _upload_file(bucket: str, key: str, file_obj, content_type: str = "video/mp4") -> None:
    """Upload file to S3 using boto3 with progress tracking.

    Args:
        bucket: S3 bucket name
        key: S3 object key
        file_obj: File-like object supporting read()
        content_type: MIME type for the uploaded file (default: video/mp4)
    """
    import boto3.session
    from botocore.exceptions import NoCredentialsError, PartialCredentialsError

    # Get file size for progress tracking
    file_obj.seek(0, 2)  # Seek to end
    file_size = file_obj.tell()
    file_obj.seek(0)  # Reset to beginning

    # Create progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Uploading to S3: 0 MB / {file_size / (1024**2):.1f} MB (0%)")

    # Use boto3 for reliable uploads with progress tracking
    try:
        # Create session that will load credentials from ~/.aws/credentials
        session = boto3.session.Session()

        # Verify credentials are available before attempting upload
        credentials = session.get_credentials()
        if credentials is None:
            raise NoCredentialsError()

        s3_client = session.client(
            "s3",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
    except (NoCredentialsError, PartialCredentialsError) as e:
        raise Exception(
            "AWS credentials not found. Please run 'aws configure' or set "
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
        ) from e

    # Upload in chunks to maintain Streamlit context and show progress
    # Using 10MB chunks for good balance between progress updates and performance
    chunk_size = 10 * 1024 * 1024  # 10 MB
    uploaded_bytes = 0

    # Use multipart upload for large files
    if file_size > chunk_size:
        multipart = s3_client.create_multipart_upload(
            Bucket=bucket,
            Key=key,
            ContentType=content_type,
        )
        upload_id = multipart["UploadId"]

        parts = []
        part_number = 1

        try:
            while True:
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    break

                # Upload this part
                part = s3_client.upload_part(
                    Bucket=bucket,
                    Key=key,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=chunk,
                )

                parts.append({
                    "PartNumber": part_number,
                    "ETag": part["ETag"],
                })

                # Update progress in main thread (where Streamlit context is available)
                uploaded_bytes += len(chunk)
                progress = uploaded_bytes / file_size
                progress_bar.progress(min(progress, 1.0))
                mb_uploaded = uploaded_bytes / (1024**2)
                mb_total = file_size / (1024**2)
                status_text.text(f"Uploading to S3: {mb_uploaded:.1f} MB / {mb_total:.1f} MB ({progress * 100:.1f}%)")

                part_number += 1

            # Complete the multipart upload
            s3_client.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
        except Exception as e:
            # Abort the multipart upload on error
            s3_client.abort_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
            )
            raise
    else:
        # For small files, use simple put_object
        data = file_obj.read()
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        uploaded_bytes = len(data)
        progress_bar.progress(1.0)
        mb_uploaded = uploaded_bytes / (1024**2)
        status_text.text(f"Uploading to S3: {mb_uploaded:.1f} MB / {mb_uploaded:.1f} MB (100%)")

    # Complete the progress bar
    progress_bar.progress(1.0)
    status_text.text(f"Upload complete: {file_size / (1024**2):.1f} MB")
    time.sleep(0.5)  # Brief pause to show completion
    progress_bar.empty()
    status_text.empty()


def _upload_presigned(
    upload_url: str,
    file_obj,
    headers: dict,
    content_type: str = "video/mp4",
    max_retries: int = 3,
) -> None:
    """Upload file using presigned URL (no boto3 required).

    Reads the entire file into memory before uploading to avoid SSL streaming
    issues with OpenSSL 3.x. For very large files (>2GB), consider using
    boto3 multipart upload instead.

    Includes retry logic for transient SSL/network errors.

    Args:
        upload_url: Presigned S3 PUT URL
        file_obj: File-like object supporting read() and seek()
        headers: Headers to include in the PUT request
        content_type: MIME type for the uploaded file (default: video/mp4)
        max_retries: Maximum number of retry attempts for transient errors (default: 3)
    """
    from requests.exceptions import SSLError, ConnectionError, Timeout, ChunkedEncodingError

    # Get file size for progress tracking
    file_obj.seek(0, 2)  # Seek to end
    file_size = file_obj.tell()
    file_obj.seek(0)  # Reset to beginning

    # Create progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Merge content-type and content-length into headers
    request_headers = dict(headers)
    request_headers["Content-Type"] = content_type
    request_headers["Content-Length"] = str(file_size)

    # Read entire file into memory (avoids SSL streaming issues with OpenSSL 3.x)
    status_text.text(f"Reading file into memory: {file_size / (1024**2):.1f} MB...")
    progress_bar.progress(0.1)
    file_data = file_obj.read()
    progress_bar.progress(0.2)
    status_text.text(f"Uploading: 0 MB / {file_size / (1024**2):.1f} MB (0%)")

    # Retry loop for transient network errors
    last_error = None
    for attempt in range(max_retries + 1):
        retry_info = f" (attempt {attempt + 1}/{max_retries + 1})" if attempt > 0 else ""
        status_text.text(
            f"Uploading: {file_size / (1024**2):.1f} MB...{retry_info}"
        )
        progress_bar.progress(0.3)

        try:
            # Upload using requests.put with pre-read data (not streaming)
            # This avoids SSL buffer issues with OpenSSL 3.x streaming uploads
            response = requests.put(
                upload_url,
                data=file_data,
                headers=request_headers,
                timeout=(30, 600),  # 30s connect, 600s read timeout for large files
            )
            response.raise_for_status()
            # Success - break out of retry loop
            break

        except (SSLError, ConnectionError, Timeout, ChunkedEncodingError) as exc:
            last_error = exc
            if attempt < max_retries:
                # Exponential backoff: 2s, 4s, 8s
                wait_time = 2 ** (attempt + 1)
                status_text.text(
                    f"Upload interrupted ({type(exc).__name__}). Retrying in {wait_time}s... "
                    f"(attempt {attempt + 1}/{max_retries + 1})"
                )
                progress_bar.progress(0)
                time.sleep(wait_time)
            else:
                # All retries exhausted
                progress_bar.empty()
                status_text.empty()
                raise Exception(
                    f"Upload failed after {max_retries + 1} attempts. "
                    f"Last error: {type(exc).__name__}: {exc}"
                ) from exc

    # Complete the progress bar
    progress_bar.progress(1.0)
    status_text.text(f"Upload complete: {file_size / (1024**2):.1f} MB")
    time.sleep(0.5)  # Brief pause to show completion
    progress_bar.empty()
    status_text.empty()


def _mirror_local(ep_id: str, file_obj, local_path: str, chunk_size: int = 8 * 1024 * 1024) -> Path | None:
    """Mirror file to local disk using streaming to avoid memory buffering.

    Args:
        ep_id: Episode identifier
        file_obj: File-like object supporting read()
        local_path: Destination path
        chunk_size: Chunk size for streaming (default: 8 MB)

    Returns:
        Path to written file, or None if write failed

    Raises:
        OSError: If directory creation or file write fails (disk full, permissions, etc.)
    """
    try:
        ensure_dirs(ep_id)
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as handle:
            while True:
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
        return dest
    except (OSError, IOError) as exc:
        error_msg = f"Failed to write to {local_path}: {exc}"
        raise OSError(error_msg) from exc


def _rollback_episode_creation(ep_id: str) -> None:
    """Delete episode from store after upload failure.

    Args:
        ep_id: Episode ID to delete
    """
    try:
        helpers.api_delete(f"/episodes/{ep_id}")
        st.info(f"Rolled back episode `{ep_id}` from store.")
    except requests.RequestException as rollback_exc:
        st.warning(f"Failed to roll back episode: {rollback_exc}. You may need to manually delete `{ep_id}`.")


def _navigate_to_detail_with_ep(ep_id: str) -> None:
    """Navigate to Episode Detail page with proper state flush.

    This ensures the ep_id is set in session state and flushed via rerun
    before switching pages, preventing navigation sync issues.

    Args:
        ep_id: Episode ID to set before navigation
    """
    st.session_state["navigate_to_detail"] = True
    helpers.set_ep_id(ep_id, rerun=True)


def _s3_item_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    key = item.get("key") or ""
    parsed_key = parse_v2_episode_key(key) if key else None
    parsed_ep = helpers.parse_ep_id(item.get("ep_id", ""))
    show = parsed_key.get("show_slug") if parsed_key else (parsed_ep or {}).get("show")
    season = parsed_key.get("season") if parsed_key else (parsed_ep or {}).get("season")
    episode = parsed_key.get("episode") if parsed_key else (parsed_ep or {}).get("episode")
    ep_id = parsed_key.get("ep_id") if parsed_key else item.get("ep_id")
    return {
        "ep_id": ep_id,
        "show": show,
        "season": season,
        "episode": episode,
        "key_meta": parsed_key,
    }


new_show_name = ""
add_show_clicked = False
show_choice = None

with st.form("episode-upload"):
    st.subheader("Upload new episode")
    show_options = helpers.known_shows()
    select_options = list(show_options) if show_options else []
    if ADD_SHOW_OPTION not in select_options:
        select_options.append(ADD_SHOW_OPTION)
    show_choice = st.selectbox(
        "Show",
        options=select_options or [ADD_SHOW_OPTION],
        key="upload_show_choice",
        help="Select an existing show or add a new one",
    )
    if show_choice == ADD_SHOW_OPTION:
        st.caption("Add a new show slug/ID to track episodes for.")
        new_show_name = st.text_input(
            "New show slug or ID",
            key="upload_new_show_input",
            placeholder="rhoslc",
            help="This becomes the show slug used for new episodes",
        )
        add_show_clicked = st.form_submit_button("Add show")
    else:
        st.session_state.pop("upload_new_show_input", None)
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
    st.caption("Video will be uploaded to S3. After upload, open Episode Detail to run detect/track.")

    # Audio pipeline options
    st.divider()
    run_audio_after_upload = st.checkbox(
        "Run audio pipeline after upload",
        value=False,
        help="Extract audio, transcribe, and identify speakers automatically",
    )
    audio_asr_provider = st.selectbox(
        "ASR Provider",
        options=["openai_whisper", "gemini"],
        index=0,
        disabled=not run_audio_after_upload,
        help="Whisper is faster and more accurate; Gemini is cheaper for long videos",
    )

    submit = st.form_submit_button("Upload episode", type="primary")

if add_show_clicked:
    pending_show = (st.session_state.get("upload_new_show_input") or new_show_name or "").strip()
    if not pending_show:
        st.error("Enter a new show slug/ID before adding.")
    else:
        # Normalize to uppercase for consistent show codes
        pending_show = pending_show.upper()
        helpers.remember_custom_show(pending_show)
        st.session_state["upload_show_choice"] = pending_show
        st.session_state["upload_new_show_input"] = ""
        st.success(f"Added show `{pending_show}` to the dropdown.")
        st.rerun()

if submit:
    show_ref = show_choice or ""
    if show_ref == ADD_SHOW_OPTION:
        show_ref = (st.session_state.get("upload_new_show_input") or new_show_name or "").strip()
        if show_ref:
            helpers.remember_custom_show(show_ref)
            st.session_state["upload_show_choice"] = show_ref.upper()
    # Normalize show code to uppercase for consistency
    show_ref = show_ref.strip().upper()
    if not show_ref:
        st.error("Show is required.")
        st.stop()
    if uploaded_file is None:
        st.error("Attach an .mp4 before submitting.")
        st.stop()

    # Preflight validation: check file size
    file_size = uploaded_file.size
    MIN_FILE_SIZE = 1 * 1024 * 1024  # 1 MB minimum
    if file_size == 0:
        st.error("Uploaded file is empty (0 bytes). Please select a valid video file.")
        st.stop()
    if file_size < MIN_FILE_SIZE:
        st.warning(
            f"Uploaded file is suspiciously small ({file_size / 1024:.1f} KB). "
            "This may be corrupted or incomplete. Minimum recommended size is 1 MB."
        )
        st.error("Upload cancelled. Please verify your video file.")
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
        create_resp = helpers.api_post("/episodes", payload)
    except requests.RequestException as exc:
        endpoint = f"{cfg['api_base']}/episodes"
        st.error(f"Episode create failed: {helpers.describe_error(endpoint, exc)}")
        st.stop()

    ep_id = create_resp["ep_id"]
    st.info(f"Episode `{ep_id}` created. Requesting upload target...")

    presign_path = f"/episodes/{ep_id}/assets"
    try:
        presign_resp = helpers.api_post(presign_path)
    except requests.RequestException as exc:
        endpoint = f"{cfg['api_base']}{presign_path}"
        st.error(f"Presign failed: {helpers.describe_error(endpoint, exc)}")
        st.stop()

    # Honor the presign contract: check method to determine upload strategy
    upload_method = presign_resp.get("method", "FILE")
    upload_url = presign_resp.get("upload_url")
    upload_headers = presign_resp.get("headers", {})
    bucket = presign_resp.get("bucket")
    key = presign_resp.get("key") or presign_resp.get("object_key")

    if upload_method == "PUT" and bucket and key:
        # Try boto3 multipart upload first (more reliable on macOS with OpenSSL 3.x)
        # Fall back to presigned URL if boto3 credentials are unavailable
        st.info(f"Uploading to s3://{bucket}/{key}...")
        s3_upload_success = False

        try:
            _upload_file(bucket, key, uploaded_file)
            s3_upload_success = True
        except Exception as boto_exc:
            # boto3 failed (likely no credentials), try presigned URL
            st.warning(f"Direct S3 upload failed ({type(boto_exc).__name__}), trying presigned URL...")
            uploaded_file.seek(0)

            if upload_url:
                try:
                    _upload_presigned(upload_url, uploaded_file, upload_headers)
                    s3_upload_success = True
                except Exception as presign_exc:
                    st.error(f"Upload failed: {type(presign_exc).__name__}: {presign_exc}")
                    _rollback_episode_creation(ep_id)
                    st.stop()
            else:
                st.error(f"Upload failed: {type(boto_exc).__name__}: {boto_exc}")
                _rollback_episode_creation(ep_id)
                st.stop()

        # Seek back to beginning for local mirror after S3 upload
        uploaded_file.seek(0)
    elif upload_method == "FILE":
        # Local-only mode - skip remote upload entirely
        st.info(f"Writing to local storage: {presign_resp['local_video_path']}")
    else:
        # Unexpected method - warn but continue with local mirror
        st.warning(f"Unknown upload method '{upload_method}', falling back to local-only")

    try:
        _mirror_local(ep_id, uploaded_file, presign_resp["local_video_path"])
    except OSError as exc:
        st.error(f"Failed to write local copy: {exc}")
        if upload_method == "PUT":
            st.warning("Video uploaded to S3 successfully, but local mirror failed. Check disk space and permissions.")
        _rollback_episode_creation(ep_id)
        st.stop()

    video_meta = _get_video_meta(ep_id)
    detected_fps_value = video_meta.get("fps_detected") if video_meta else None
    if detected_fps_value:
        st.info(f"Detected FPS: {detected_fps_value:.3f}")

    artifacts = {
        "video": get_path(ep_id, "video"),
    }
    if upload_method == "PUT":
        flash_lines = [f"Episode `{ep_id}` uploaded to S3 successfully."]
    else:
        flash_lines = [f"Episode `{ep_id}` saved locally."]
    flash_lines.append(f"Video -> {helpers.link_local(artifacts['video'])}")
    flash_lines.append("Go to Episode Detail to run detect/track processing.")

    # Trigger audio pipeline if requested
    if run_audio_after_upload:
        try:
            audio_payload = {
                "ep_id": ep_id,
                "overwrite": False,
                "asr_provider": audio_asr_provider,
            }
            audio_resp = helpers.api_post("/jobs/episode_audio_pipeline", json=audio_payload)
            audio_job_id = audio_resp.get("job_id")
            if audio_job_id:
                helpers.store_celery_job_id(ep_id, "audio_pipeline", audio_job_id)
                flash_lines.append(f"Audio pipeline started: `{audio_job_id}`")
        except requests.RequestException as exc:
            flash_lines.append(f"Audio pipeline failed to start: {exc}")

    st.session_state["upload_flash"] = "\n".join(flash_lines)
    helpers.set_ep_id(ep_id)

st.button(
    "Open Episode Detail",
    on_click=lambda: helpers.try_switch_page("pages/2_Episode_Detail.py"),
)

s3_loaded = True
st.subheader("Existing Episode (browse S3)")
try:
    s3_payload = helpers.api_get("/episodes/s3_videos")
    s3_items = s3_payload.get("items", [])
    s3_loaded = True
except requests.RequestException as exc:
    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/s3_videos", exc))
    s3_items = []
    s3_loaded = False

if s3_items:
    s3_search = st.text_input("Filter S3 videos", "").strip().lower()

    # Expand filter to search ep_id, show, season, episode, and S3 key
    def _matches_filter(item: Dict[str, Any], search_term: str) -> bool:
        if not search_term:
            return True
        meta = _s3_item_metadata(item)
        search_fields = [
            str(item.get("ep_id", "")),
            str(meta.get("show", "")),
            str(meta.get("season", "")),
            str(meta.get("episode", "")),
            str(item.get("key", "")),
        ]
        return any(search_term in field.lower() for field in search_fields)

    filtered_items = [item for item in s3_items if _matches_filter(item, s3_search)]

    # Reset selectbox selection when filter changes to avoid stale index
    prev_filter = st.session_state.get("s3_prev_filter", "")
    if s3_search != prev_filter:
        st.session_state["s3_prev_filter"] = s3_search
        if "s3_video_select" in st.session_state:
            del st.session_state["s3_video_select"]

    if filtered_items:
        # Sort by show (alphabetically), then season (descending), then episode (descending)
        def _sort_key(item: Dict[str, Any]) -> tuple:
            meta = _s3_item_metadata(item)
            show = meta.get("show") or ""
            season = meta.get("season")
            episode = meta.get("episode")
            # Use negative values for descending sort, fallback to 0 if None
            season_val = -int(season) if season is not None else 0
            episode_val = -int(episode) if episode is not None else 0
            return (show.lower(), season_val, episode_val)

        sorted_items = sorted(filtered_items, key=_sort_key)

        def _format_item(item: Dict[str, Any]) -> str:
            size = item.get("size")
            size_mb = f"{(size or 0) / (1024**2):.1f} MB" if size else "size ?"
            last_mod_raw = item.get("last_modified")
            # Format timestamp: extract date portion (YYYY-MM-DD) from ISO timestamp
            if last_mod_raw and len(last_mod_raw) >= 10:
                last_mod = last_mod_raw[:10]  # Take first 10 chars: YYYY-MM-DD
            else:
                last_mod = last_mod_raw or "unknown"
            return f"{item['ep_id']} - {size_mb} - {last_mod}"

        selected_index = st.selectbox(
            "S3 videos",
            list(range(len(sorted_items))),
            format_func=lambda idx: _format_item(sorted_items[idx]),
            key="s3_video_select",
        )
        selected_item = sorted_items[selected_index]
        selected_meta = _s3_item_metadata(selected_item)
        ep_id_from_s3 = selected_meta.get("ep_id") or selected_item.get("ep_id")
        show_label = selected_meta.get("show")
        season_label = selected_meta.get("season")
        episode_label = selected_meta.get("episode")
        st.write(f"S3 key: `{selected_item['key']}`")
        if show_label and season_label is not None and episode_label is not None:
            st.caption(f"{show_label} - s{int(season_label):02d}e{int(episode_label):02d}")
        tracked = bool(selected_item.get("exists_in_store"))
        st.write(f"Tracked in store: {tracked}")
        if ep_id_from_s3:
            helpers.set_ep_id(ep_id_from_s3, rerun=False)

        prefixes = helpers.episode_artifact_prefixes(ep_id_from_s3) if ep_id_from_s3 else None
        if prefixes:
            st.caption(
                "S3 artifacts -> "
                f"Frames {helpers.s3_uri(prefixes['frames'], cfg.get('bucket'))} | "
                f"Crops {helpers.s3_uri(prefixes['crops'], cfg.get('bucket'))} | "
                f"Manifests {helpers.s3_uri(prefixes['manifests'], cfg.get('bucket'))}"
            )

        detail_data = None
        if ep_id_from_s3 and tracked:
            try:
                detail_data = helpers.api_get(f"/episodes/{ep_id_from_s3}")
            except requests.RequestException as exc:
                st.warning(helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id_from_s3}", exc))
        if detail_data:
            s3_info = detail_data.get("s3", {})
            st.write(f"V2 key: `{s3_info.get('v2_key')}` (exists={s3_info.get('v2_exists')})")
            st.write(f"V1 key: `{s3_info.get('v1_key')}` (exists={s3_info.get('v1_exists')})")
            if not s3_info.get("v2_exists") and s3_info.get("v1_exists"):
                st.warning("Found legacy v1 object; mirror will fall back to v1 but new uploads use the v2 path.")

        if not tracked:
            st.warning("Episode not in local store yet.")
            if st.button("Create episode in store", key=f"create_episode_{ep_id_from_s3}"):
                if not (ep_id_from_s3 and show_label and season_label is not None and episode_label is not None):
                    st.error("Unable to parse S3 key into show/season/episode (v2 keys required).")
                else:
                    payload = {
                        "ep_id": ep_id_from_s3,
                        "show_slug": str(show_label),
                        "season": int(season_label),
                        "episode": int(episode_label),
                    }
                    try:
                        upsert_resp = helpers.api_post("/episodes/upsert_by_id", payload)
                    except requests.RequestException as exc:
                        st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/upsert_by_id", exc))
                    else:
                        st.success(f"Episode `{upsert_resp['ep_id']}` tracked (created={upsert_resp['created']}).")
                        helpers.set_ep_id(upsert_resp["ep_id"])
                        st.rerun()
        elif ep_id_from_s3:
            action_cols = st.columns([1, 1, 1])
            with action_cols[0]:
                st.button(
                    "Open Episode Detail",
                    key=f"open_detail_{ep_id_from_s3}",
                    use_container_width=True,
                    on_click=lambda ep=ep_id_from_s3: _navigate_to_detail_with_ep(ep),
                )
            with action_cols[1]:
                if st.button(
                    "Mirror from S3",
                    key=f"mirror_{ep_id_from_s3}",
                    use_container_width=True,
                ):
                    with st.spinner(f"Mirroring video from S3 for {ep_id_from_s3}..."):
                        try:
                            mirror_resp = helpers.api_post(f"/episodes/{ep_id_from_s3}/mirror")
                        except requests.RequestException as exc:
                            st.error(
                                helpers.describe_error(
                                    f"{cfg['api_base']}/episodes/{ep_id_from_s3}/mirror",
                                    exc,
                                )
                            )
                        else:
                            st.success(
                                f"Mirrored to {helpers.link_local(mirror_resp['local_video_path'])} "
                                f"({helpers.human_size(mirror_resp.get('bytes'))})"
                            )
            with action_cols[2]:
                if st.button(
                    "Queue detect/track (defaults)",
                    key=f"queue_detect_track_{ep_id_from_s3}",
                    use_container_width=True,
                ):
                    with st.spinner("Queueing detect/track (RetinaFace + ByteTrack)..."):
                        result = _launch_default_detect_track(
                            ep_id_from_s3,
                            label=f"S3 detect/track - {ep_id_from_s3}",
                        )
                    if result and result.get("job"):
                        st.rerun()

            artifacts = {
                "video": get_path(ep_id_from_s3, "video"),
                "detections": get_path(ep_id_from_s3, "detections"),
                "tracks": get_path(ep_id_from_s3, "tracks"),
            }
            st.caption(
                f"Video -> {helpers.link_local(artifacts['video'])} | "
                f"Detections -> {helpers.link_local(artifacts['detections'])} | "
                f"Tracks -> {helpers.link_local(artifacts['tracks'])}"
            )
    elif s3_loaded:
        st.warning("No S3 videos match that filter.")
else:
    st.info("No S3 videos found (or API error).")

if jobs_running:
    time.sleep(0.5)
    st.rerun()
