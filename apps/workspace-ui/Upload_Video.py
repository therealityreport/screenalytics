from __future__ import annotations

import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict

import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    EndpointConnectionError,
    ReadTimeoutError,
)
import requests
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime, timedelta

DEBUG_UPLOAD = os.getenv("DEBUG_UPLOAD", "").lower() in {"1", "true", "yes", "on"}

# Global lock for thread-safe session state updates
_session_state_lock = threading.Lock()

# TTL for upload progress keys (1 hour)
UPLOAD_PROGRESS_TTL_SECONDS = 3600

# Ensure AWS credentials path is set for boto3
if "AWS_SHARED_CREDENTIALS_FILE" not in os.environ:
    home = os.path.expanduser("~")
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = f"{home}/.aws/credentials"
if "AWS_CONFIG_FILE" not in os.environ:
    home = os.path.expanduser("~")
    os.environ["AWS_CONFIG_FILE"] = f"{home}/.aws/config"

os.environ.setdefault("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", "10240")

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

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
            "ep_id": f"{show}-s{season:02d}e{episode:02d}",
            "show": show,
            "show_slug": show,
            "season": season,
            "episode": episode,
            "key_version": "v2",
        }


def parse_v2_episode_key(key: str) -> Dict[str, object] | None:
    return _shared_parse_v2_episode_key(key)


import ui_helpers as helpers  # noqa: E402

cfg: Dict[str, Any] = {}


def _get_video_meta(ep_id: str) -> Dict[str, Any] | None:
    try:
        return helpers.api_get(f"/episodes/{ep_id}/video_meta")
    except requests.RequestException as exc:
        st.warning(helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id}/video_meta", exc))
        return None


ASYNC_JOBS_KEY = "async_jobs"
ADD_SHOW_OPTION = "➕ Add new show…"


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
    st.caption(f"Job ID: `{job_id}` · Episode `{meta.get('ep_id')}`")
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
        f"Elapsed: {helpers.format_mmss(elapsed_seconds)} · "
        f"Total: {helpers.format_mmss(total_seconds)} · "
        f"ETA: {helpers.format_mmss(eta)} · "
        f"Device: {device_label} · FPS: {fps_text}"
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
                    st.info("Cancel requested…")
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
        msg = "Completed" + (" · " + ", ".join(counts) if counts else "")
        st.success(msg or "Job succeeded")
    elif state == "canceled":
        st.warning("Job canceled.")
    else:
        fallback_err = error_msg or "Job failed without error detail."
        st.error(f"Job failed: {fallback_err}")
    artifacts = meta.get("artifacts") or {}
    artifact_line = " | ".join(
        [
            f"Video → {helpers.link_local(artifacts.get('video', get_path(meta['ep_id'], 'video')))}",
            f"Detections → {helpers.link_local(artifacts.get('detections', get_path(meta['ep_id'], 'detections')))}",
            f"Tracks → {helpers.link_local(artifacts.get('tracks', get_path(meta['ep_id'], 'tracks')))}",
        ]
    )
    st.caption(artifact_line)
    prefixes = helpers.episode_artifact_prefixes(meta.get("ep_id", ""))
    if prefixes:
        st.caption(
            "S3 → "
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


def _aws_client_config(connect_timeout: int = 5, read_timeout: int = 60, retries: int = 3) -> Config:
    """Return a conservative AWS Config for UI-bound calls."""
    return Config(
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        retries={"max_attempts": retries, "mode": "standard"},
    )


def _truncate_error_message(msg: str, max_length: int = 200) -> str:
    """Truncate error message to prevent UI overflow."""
    if len(msg) <= max_length:
        return msg
    return msg[:max_length] + "... (truncated)"


def _classify_s3_error(exc: Exception) -> tuple[str, str]:
    """Return (error_type, message) for S3/STS/network errors."""
    # Check most specific exceptions first, then more general ones
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        if code in {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}:
            return "credentials", _truncate_error_message(f"{code}: {exc}")
        return (code or "other").lower(), _truncate_error_message(str(exc))
    if isinstance(exc, ReadTimeoutError):
        return "timeout", _truncate_error_message(str(exc))
    if isinstance(exc, EndpointConnectionError):
        return "network", _truncate_error_message(str(exc))
    # Check specific request exceptions before general RequestException
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return "network", _truncate_error_message(str(exc))
    if isinstance(exc, requests.RequestException):
        return "network", _truncate_error_message(str(exc))
    if isinstance(exc, BotoCoreError):
        return "other", _truncate_error_message(str(exc))
    return "other", _truncate_error_message(str(exc))


def _validate_s3_credentials() -> tuple[bool, Exception | None]:
    """Validate AWS credentials quickly using STS.

    Returns:
        Tuple of (is_valid, error). error is None if valid.
    """
    try:
        session = boto3.session.Session()
        credentials = session.get_credentials()

        if credentials is None:
            return False, ValueError("AWS credentials not found. Please run 'aws configure'.")

        config = _aws_client_config(connect_timeout=5, read_timeout=10, retries=2)
        sts_client = session.client(
            "sts",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            config=config,
        )
        sts_client.get_caller_identity()
        return True, None
    except (BotoCoreError, ClientError) as exc:
        return False, exc
    except Exception as exc:  # pragma: no cover - defensive catch for UI surfacing
        return False, exc


def _create_s3_client_with_timeout():
    """Create S3 client with extended timeouts for large file uploads.

    Returns:
        Configured boto3 S3 client
    """
    session = boto3.session.Session()
    credentials = session.get_credentials()

    if credentials is None:
        raise Exception("AWS credentials not found. Please run 'aws configure'.")

    config = _aws_client_config(connect_timeout=5, read_timeout=60, retries=3)

    return session.client(
        "s3",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        config=config
    )


def _upload_file(bucket: str, key: str, file_obj, content_type: str = "video/mp4") -> Dict[str, Any]:
    """Upload file to S3 using boto3 with progress tracking and bounded timeouts."""
    file_obj.seek(0, 2)
    file_size = file_obj.tell()
    file_obj.seek(0)

    result: Dict[str, Any] = {
        "s3_succeeded": False,
        "s3_error_type": None,
        "s3_error_message": None,
    }

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Uploading to S3: 0 MB / {file_size / (1024**2):.1f} MB (0%)")

    try:
        s3_client = _create_s3_client_with_timeout()
    except Exception as exc:
        error_type, error_message = _classify_s3_error(exc)
        result["s3_error_type"] = error_type
        result["s3_error_message"] = error_message
        progress_bar.empty()
        status_text.empty()
        return result

    chunk_size = 50 * 1024 * 1024  # 50 MB
    uploaded_bytes = 0
    # Update progress every 5% or 100MB, whichever is smaller (better for small files)
    progress_update_interval = min(100 * 1024 * 1024, max(1 * 1024 * 1024, file_size // 20))
    last_progress_update = 0
    upload_id = None

    try:
        if file_size > chunk_size:
            multipart = s3_client.create_multipart_upload(
                Bucket=bucket,
                Key=key,
                ContentType=content_type,
            )
            upload_id = multipart["UploadId"]
            parts = []
            part_number = 1

            while True:
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    break

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

                uploaded_bytes += len(chunk)

                if (uploaded_bytes - last_progress_update >= progress_update_interval) or (uploaded_bytes >= file_size):
                    progress = uploaded_bytes / file_size
                    progress_bar.progress(min(progress, 1.0))
                    mb_uploaded = uploaded_bytes / (1024**2)
                    mb_total = file_size / (1024**2)
                    status_text.text(f"Uploading to S3: {mb_uploaded:.1f} MB / {mb_total:.1f} MB ({progress * 100:.1f}%)")
                    last_progress_update = uploaded_bytes

                part_number += 1

            s3_client.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
        else:
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

        progress_bar.progress(1.0)
        status_text.text(f"✅ Upload complete: {file_size / (1024**2):.1f} MB")
        time.sleep(0.5)
        result["s3_succeeded"] = True
        return result
    except Exception as exc:
        error_type, error_message = _classify_s3_error(exc)
        result["s3_error_type"] = error_type
        result["s3_error_message"] = error_message
        if upload_id:
            try:
                s3_client.abort_multipart_upload(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                )
            except Exception as abort_exc:
                logging.getLogger(__name__).warning(
                    f"Failed to abort multipart upload {upload_id}: {abort_exc}. "
                    "Orphaned parts may remain in S3 bucket.",
                )
        progress_bar.empty()
        status_text.empty()
        return result
    finally:
        try:
            status_text.empty()
            progress_bar.empty()
        except Exception:
            pass


def _mirror_local(ep_id: str, file_obj, local_path: str, chunk_size: int = 8 * 1024 * 1024,
                  progress_callback=None) -> Path | None:
    """Mirror file to local disk using streaming to avoid memory buffering.

    Args:
        ep_id: Episode identifier
        file_obj: File-like object supporting read()
        local_path: Destination path
        chunk_size: Chunk size for streaming (default: 8 MB)
        progress_callback: Optional callback(bytes_written, total_bytes) for progress updates

    Returns:
        Path to written file, or None if write failed

    Raises:
        OSError: If directory creation or file write fails (disk full, permissions, etc.)
    """
    try:
        ensure_dirs(ep_id)
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Get total size for progress tracking
        file_obj.seek(0, 2)
        total_size = file_obj.tell()
        file_obj.seek(0)

        bytes_written = 0
        with dest.open("wb") as handle:
            while True:
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(bytes_written, total_size)

        return dest
    except (OSError, IOError) as exc:
        error_msg = f"Failed to write to {local_path}: {exc}"
        raise OSError(error_msg) from exc


def _write_local_copy_with_progress(ep_id: str, file_obj, local_path: str, file_size: int) -> Dict[str, Any]:
    """Write the uploaded file to local_path with UI progress."""
    local_progress_bar = st.progress(0)
    local_status_text = st.empty()

    def local_progress_callback(bytes_written, total_bytes):
        progress = bytes_written / total_bytes if total_bytes > 0 else 0
        local_progress_bar.progress(min(progress, 1.0))
        mb_written = bytes_written / (1024**2)
        mb_total = total_bytes / (1024**2)
        local_status_text.text(f"Writing to disk: {mb_written:.1f} MB / {mb_total:.1f} MB ({progress * 100:.1f}%)")
        time.sleep(0.05)

    try:
        file_obj.seek(0)
    except Exception as seek_exc:
        local_progress_bar.empty()
        local_status_text.empty()
        return {
            "succeeded": False,
            "path": None,
            "error_message": f"Failed to reset file pointer: {seek_exc}",
        }

    try:
        _mirror_local(ep_id, file_obj, local_path, progress_callback=local_progress_callback)
        local_progress_bar.progress(1.0)
        local_status_text.text(f"✅ Saved to local storage: {file_size / (1024**2):.1f} MB")
        time.sleep(0.5)
        return {"succeeded": True, "path": local_path, "error_message": None}
    except OSError as exc:
        return {"succeeded": False, "path": None, "error_message": str(exc)}
    finally:
        local_progress_bar.empty()
        local_status_text.empty()


def _upload_via_presigned_put(upload_url: str, file_obj, headers: dict = None) -> None:
    """Upload file via presigned PUT URL with progress tracking.

    Args:
        upload_url: Presigned PUT URL from API
        file_obj: File-like object supporting read()
        headers: Optional headers to include in PUT request
    """
    # Get file size for progress tracking
    file_obj.seek(0, 2)
    file_size = file_obj.tell()
    file_obj.seek(0)

    # Create progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Uploading via presigned URL: 0 MB / {file_size / (1024**2):.1f} MB (0%)")

    # Prepare headers
    upload_headers = headers.copy() if headers else {}
    if 'Content-Type' not in upload_headers:
        upload_headers['Content-Type'] = 'video/mp4'
    if 'Content-Length' not in upload_headers:
        upload_headers['Content-Length'] = str(file_size)

    # Upload in chunks with progress tracking
    chunk_size = 8 * 1024 * 1024  # 8 MB chunks for streaming
    uploaded_bytes = 0
    last_progress_update = 0
    # Update progress every 5% or 50MB, whichever is smaller
    progress_update_interval = min(50 * 1024 * 1024, max(1 * 1024 * 1024, file_size // 20))

    class ProgressFileReader:
        """Wrapper to track upload progress."""
        def __init__(self, file_obj, callback):
            self.file_obj = file_obj
            self.callback = callback
            self.bytes_read = 0

        def read(self, size=-1):
            chunk = self.file_obj.read(size)
            self.bytes_read += len(chunk)
            self.callback(self.bytes_read, file_size)
            return chunk

    def progress_callback(bytes_uploaded, total_bytes):
        nonlocal last_progress_update
        # Only update UI periodically to avoid overhead
        if (bytes_uploaded - last_progress_update >= progress_update_interval) or (bytes_uploaded >= total_bytes):
            progress = bytes_uploaded / total_bytes if total_bytes > 0 else 0
            progress_bar.progress(min(progress, 1.0))
            mb_uploaded = bytes_uploaded / (1024**2)
            mb_total = total_bytes / (1024**2)
            status_text.text(f"Uploading: {mb_uploaded:.1f} MB / {mb_total:.1f} MB ({progress * 100:.1f}%)")
            last_progress_update = bytes_uploaded
            # Small sleep to yield control
            time.sleep(0.05)

    # Wrap file object with progress tracker
    progress_reader = ProgressFileReader(file_obj, progress_callback)

    try:
        # Send PUT request with extended timeout for large files
        response = requests.put(
            upload_url,
            data=progress_reader,
            headers=upload_headers,
            timeout=(5, 300),  # 5s connect, 5m read to stay bounded
        )
        response.raise_for_status()

        # Complete progress bar
        progress_bar.progress(1.0)
        status_text.text(f"✅ Upload complete: {file_size / (1024**2):.1f} MB")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

    except requests.RequestException as e:
        progress_bar.empty()
        status_text.empty()
        raise Exception(f"Presigned PUT upload failed: {e}")


def _upload_file_to_s3_async(local_path: str, bucket: str, key: str,
                              progress_key: str, content_type: str = "video/mp4") -> None:
    """Upload file to S3 in background thread with progress tracking via session state.

    Args:
        local_path: Path to local file to upload
        bucket: S3 bucket name
        key: S3 object key
        progress_key: Session state key for tracking progress
        content_type: MIME type for the uploaded file
    """
    try:
        s3_client = _create_s3_client_with_timeout()
        file_size = Path(local_path).stat().st_size

        # Use boto3's upload_file with callback for progress
        from boto3.s3.transfer import TransferConfig

        # Configure multipart upload for files > 50MB
        config = TransferConfig(
            multipart_threshold=50 * 1024 * 1024,  # 50 MB
            multipart_chunksize=50 * 1024 * 1024,  # 50 MB parts
            max_concurrency=10,  # Allow parallel part uploads
            use_threads=True
        )

        def progress_callback(bytes_transferred):
            """Update progress in session state with thread safety."""
            with _session_state_lock:
                if progress_key in st.session_state:
                    st.session_state[progress_key]["bytes_uploaded"] = bytes_transferred
                    st.session_state[progress_key]["total_bytes"] = file_size

        # Upload file with progress tracking
        s3_client.upload_file(
            local_path,
            bucket,
            key,
            ExtraArgs={'ContentType': content_type},
            Config=config,
            Callback=progress_callback
        )

        # Mark as complete in session state
        with _session_state_lock:
            if progress_key in st.session_state:
                st.session_state[progress_key]["status"] = "completed"
                st.session_state[progress_key]["error"] = None

    except Exception as e:
        # Record error in session state
        with _session_state_lock:
            if progress_key in st.session_state:
                st.session_state[progress_key]["status"] = "failed"
                st.session_state[progress_key]["error"] = str(e)
        logging.error(f"S3 upload failed: {e}")


def _rollback_episode_creation(ep_id: str, bucket: str = None, key: str = None) -> None:
    """Delete episode from store after upload failure and clean up local and S3 artifacts.

    Args:
        ep_id: Episode ID to delete
        bucket: Optional S3 bucket name for cleanup
        key: Optional S3 object key for cleanup
    """
    cleanup_actions = []

    # 1. Abort any incomplete S3 multipart uploads if bucket/key provided
    if bucket and key:
        try:
            s3_client = _create_s3_client_with_timeout()
            # List all multipart uploads for this key
            uploads_resp = s3_client.list_multipart_uploads(Bucket=bucket, Prefix=key)
            uploads = uploads_resp.get('Uploads', [])

            for upload in uploads:
                if upload.get('Key') == key:
                    upload_id = upload.get('UploadId')
                    try:
                        s3_client.abort_multipart_upload(
                            Bucket=bucket,
                            Key=key,
                            UploadId=upload_id
                        )
                        cleanup_actions.append(f"S3 multipart upload {upload_id[:8]}...")
                    except Exception as abort_exc:
                        logging.warning(f"Failed to abort S3 multipart upload {upload_id}: {abort_exc}")
        except Exception as s3_exc:
            st.caption(f"Could not cleanup S3 multipart uploads: {s3_exc}")

    # 2. Delete episode from database
    try:
        helpers.api_delete(f"/episodes/{ep_id}")
        cleanup_actions.append("database record")
    except requests.RequestException as rollback_exc:
        st.warning(f"Failed to delete episode from database: {rollback_exc}")

    # 2. Clean up local video file if it exists
    try:
        video_path = get_path(ep_id, "video")
        if video_path.exists():
            video_path.unlink()
            cleanup_actions.append(f"video file ({video_path})")
    except Exception as exc:
        st.caption(f"Could not delete local video: {exc}")

    # 3. Clean up local directories (video parent directory)
    try:
        video_dir = get_path(ep_id, "video").parent
        if video_dir.exists() and not any(video_dir.iterdir()):
            video_dir.rmdir()
            cleanup_actions.append(f"video directory")
    except Exception as exc:
        st.caption(f"Could not delete video directory: {exc}")

    # 4. Clean up manifest files if they exist
    try:
        detections_path = get_path(ep_id, "detections")
        if detections_path.exists():
            detections_path.unlink()
            cleanup_actions.append("detections manifest")
        tracks_path = get_path(ep_id, "tracks")
        if tracks_path.exists():
            tracks_path.unlink()
            cleanup_actions.append("tracks manifest")

        # Clean up manifests directory if empty
        manifests_dir = detections_path.parent
        if manifests_dir.exists() and not any(manifests_dir.iterdir()):
            manifests_dir.rmdir()
    except Exception as exc:
        st.caption(f"Could not delete manifests: {exc}")

    if cleanup_actions:
        st.info(f"Rolled back episode `{ep_id}`: cleaned up {', '.join(cleanup_actions)}.")
    else:
        st.warning(f"Failed to roll back episode `{ep_id}`. You may need to manually clean up artifacts.")


def _cleanup_stale_progress_keys() -> int:
    """Remove stale upload progress keys from session state based on TTL.

    Returns:
        Number of keys cleaned up
    """
    now = datetime.now()
    keys_to_remove = []

    for key in list(st.session_state.keys()):
        if key.startswith("s3_upload_"):
            upload_info = st.session_state.get(key, {})
            # Check if entry has a timestamp
            if "timestamp" in upload_info:
                try:
                    timestamp = upload_info["timestamp"]
                    age_seconds = (now - timestamp).total_seconds()
                except Exception:
                    # If timestamp is malformed, drop the key to avoid crashes.
                    keys_to_remove.append(key)
                else:
                    # Remove if older than TTL
                    if age_seconds > UPLOAD_PROGRESS_TTL_SECONDS:
                        keys_to_remove.append(key)
            # Also remove completed/failed uploads after they've been displayed for a while
            elif upload_info.get("status") in {"completed", "failed"}:
                # If no timestamp but completed/failed, assume it's old and clean up
                keys_to_remove.append(key)

    for key in keys_to_remove:
        st.session_state.pop(key, None)
        # Also clean up related future keys
        future_key = key.replace("s3_upload_", "s3_future_")
        st.session_state.pop(future_key, None)

    return len(keys_to_remove)


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


def handle_episode_upload(
    *,
    show_ref: str,
    season_number: int,
    episode_number: int,
    title: str | None,
    include_air_date: bool,
    air_date_value: date,
    uploaded_file,
    existing_ep_id: str | None = None,
) -> Dict[str, Any]:
    """Handle the heavy upload workflow inside a spinner-safe function."""
    # Preflight validation: check file size
    file_size = uploaded_file.size
    MIN_FILE_SIZE = 1 * 1024 * 1024  # 1 MB minimum
    MAX_RECOMMENDED_SIZE = 8 * 1024 * 1024 * 1024  # 8 GB

    if file_size == 0:
        st.error("Uploaded file is empty (0 bytes). Please select a valid video file.")
        return {"status": "error", "reason": "empty_file"}
    if file_size < MIN_FILE_SIZE:
        st.warning(
            f"Uploaded file is suspiciously small ({file_size / 1024:.1f} KB). "
            "This may be corrupted or incomplete. Minimum recommended size is 1 MB."
        )
        st.error("Upload cancelled. Please verify your video file.")
        return {"status": "error", "reason": "too_small"}
    if file_size > MAX_RECOMMENDED_SIZE:
        st.error(
            f"File size ({file_size / (1024**3):.2f} GB) exceeds recommended limit of 8 GB. "
            "Streamlit buffers files in memory, which may cause out-of-memory errors. "
            "For very large files, consider uploading directly to S3 using AWS CLI or direct presigned URL."
        )
        return {"status": "error", "reason": "too_large"}

    created_new_episode = False
    if existing_ep_id:
        ep_id = existing_ep_id
        st.info(f"Replacing video for existing episode `{ep_id}`")
    else:
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
            return {"status": "error", "reason": "create_failed"}

        ep_id = create_resp["ep_id"]
        created_new_episode = True
        st.info(f"Episode `{ep_id}` created. Requesting upload target…")

    presign_path = f"/episodes/{ep_id}/assets"
    try:
        presign_resp = helpers.api_post(presign_path)
    except requests.RequestException as exc:
        endpoint = f"{cfg['api_base']}{presign_path}"
        st.error(f"Presign failed: {helpers.describe_error(endpoint, exc)}")
        return {"status": "error", "reason": "presign_failed", "ep_id": ep_id}

    # Validate presign response structure
    if not isinstance(presign_resp, dict):
        st.error(f"Invalid presign response: expected dict, got {type(presign_resp).__name__}")
        if created_new_episode:
            _rollback_episode_creation(ep_id)
        return {"status": "error", "reason": "invalid_presign", "ep_id": ep_id}

    # Extract upload method and targets from presign response
    upload_method = presign_resp.get("method", "FILE").upper()  # Default to FILE for backward compat
    upload_url = presign_resp.get("upload_url")
    upload_headers = presign_resp.get("headers", {})
    local_video_path = presign_resp.get("local_video_path")
    bucket = presign_resp.get("bucket")
    key = presign_resp.get("key") or presign_resp.get("object_key")

    # Validate required fields based on upload method
    if upload_method == "PUT":
        if not upload_url:
            st.error("Presign response method='PUT' but missing 'upload_url'")
            if created_new_episode:
                _rollback_episode_creation(ep_id, bucket, key)
            return {"status": "error", "reason": "missing_upload_url", "ep_id": ep_id}
        st.info(f"Target: Presigned PUT to {upload_url[:50]}...")
    elif upload_method == "FILE":
        if not local_video_path:
            st.error("Presign response method='FILE' but missing 'local_video_path'")
            if created_new_episode:
                _rollback_episode_creation(ep_id, bucket, key)
            return {"status": "error", "reason": "missing_local_path", "ep_id": ep_id}
        st.info(f"Target: Local file {local_video_path}")
    elif bucket and key:
        # Legacy: Direct S3 upload with credentials
        st.info(f"Target: s3://{bucket}/{key} (direct boto3)")
    else:
        st.error("Invalid presign response: no valid upload method detected")
        if created_new_episode:
            _rollback_episode_creation(ep_id, bucket, key)
        return {"status": "error", "reason": "no_upload_method", "ep_id": ep_id}

    # Credential validation only applies to direct boto3 uploads; presigned/local paths must not block on AWS creds.
    requires_boto3 = upload_method not in {"PUT", "FILE"} and bool(bucket and key)
    if requires_boto3:
        creds_valid, creds_error = _validate_s3_credentials()
        if not creds_valid:
            detail = f"{type(creds_error).__name__}: {creds_error}" if creds_error else "Unknown credential error"
            st.error("Direct S3 upload requires AWS credentials.")
            st.error(f"Credential check failed: {detail}")
            st.info("Configure AWS credentials or switch to a presigned/local upload method.")
            return {"status": "error", "reason": "credentials", "ep_id": ep_id}

    # Route to appropriate upload method based on presign response
    needs_mirror = False  # Track if we need to call mirror endpoint after upload
    cloud_result = {
        "attempted": False,
        "succeeded": False,
        "error_type": None,
        "error_message": None,
        "bucket": bucket,
        "key": key,
        "region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    }
    local_result = {"succeeded": False, "path": None, "error_message": None}
    fallback_local_path = local_video_path or str(get_path(ep_id, "video"))

    if upload_method == "PUT":
        # Method 1: Presigned PUT upload (no AWS credentials needed)
        st.info("📤 Uploading via presigned URL...")
        cloud_result["attempted"] = True

        presigned_seek_error = None
        try:
            uploaded_file.seek(0)
        except Exception as seek_exc:
            presigned_seek_error = seek_exc
        if presigned_seek_error:
            st.error(f"Failed to reset file pointer: {presigned_seek_error}")
            if created_new_episode:
                _rollback_episode_creation(ep_id, bucket, key)
            return {"status": "error", "reason": "seek_failed", "ep_id": ep_id}

        try:
            _upload_via_presigned_put(upload_url, uploaded_file, upload_headers)
            st.success("✅ Upload complete via presigned URL")
            needs_mirror = True  # Need to mirror from remote to local
            cloud_result["succeeded"] = True
        except Exception as exc:
            error_type, error_message = _classify_s3_error(exc)
            cloud_result.update({
                "succeeded": False,
                "error_type": error_type,
                "error_message": error_message,
            })
            st.warning(f"Cloud upload failed via presigned URL: {error_message}")
            if fallback_local_path:
                local_result = _write_local_copy_with_progress(ep_id, uploaded_file, fallback_local_path, file_size)

    elif upload_method == "FILE":
        # Method 2: Direct local file write
        st.info("📦 Saving video to local storage...")
        local_write = _write_local_copy_with_progress(ep_id, uploaded_file, local_video_path, file_size)
        local_result.update(local_write)
        cloud_result["status_label"] = "LOCAL ONLY"
        cloud_result["succeeded"] = True
        cloud_result["attempted"] = False
        cloud_result["error_type"] = "not_attempted"
        cloud_result["error_message"] = "Presign provided a local-only path (no cloud upload attempted)."
        if not local_result["succeeded"]:
            st.error(f"Failed to write local copy: {local_result['error_message']}")
            if created_new_episode:
                _rollback_episode_creation(ep_id, bucket, key)
            return {"status": "error", "reason": "local_write_failed", "ep_id": ep_id}
        else:
            st.success("✅ Video saved to local storage")

    elif bucket and key:
        # Method 3: Legacy direct boto3 upload (requires AWS credentials)
        st.info("☁️ Uploading to S3 via boto3...")
        cloud_result["attempted"] = True

        boto3_seek_error = None
        try:
            uploaded_file.seek(0)
        except Exception as seek_exc:
            boto3_seek_error = seek_exc
        if boto3_seek_error:
            st.error(f"Failed to reset file pointer: {boto3_seek_error}")
            if created_new_episode:
                _rollback_episode_creation(ep_id, bucket, key)
            return {"status": "error", "reason": "seek_failed", "ep_id": ep_id}

        upload_resp = _upload_file(bucket, key, uploaded_file)
        cloud_result["succeeded"] = bool(upload_resp.get("s3_succeeded"))
        cloud_result["error_type"] = upload_resp.get("s3_error_type")
        cloud_result["error_message"] = upload_resp.get("s3_error_message")

        if cloud_result["succeeded"]:
            st.success("✅ Uploaded to S3")
            needs_mirror = True  # Need to mirror from S3 to local
        elif fallback_local_path:
            st.warning("Cloud upload failed; attempting to keep a local-only copy.")
            local_result = _write_local_copy_with_progress(ep_id, uploaded_file, fallback_local_path, file_size)

    # Call mirror endpoint if upload was to remote storage
    if needs_mirror:
        st.info("🔄 Mirroring video to local storage...")
        mirror_endpoint = f"/episodes/{ep_id}/mirror"
        try:
            with st.spinner("Downloading video from remote storage..."):
                # Use extended timeout for large file downloads (5 minutes)
                mirror_resp = helpers.api_post(mirror_endpoint, timeout=300)
            local_path = mirror_resp.get("local_video_path")
            bytes_mirrored = mirror_resp.get("bytes")
            local_result["succeeded"] = True
            local_result["path"] = local_path
            if local_path:
                st.success(f"✅ Mirrored to {local_path} ({helpers.human_size(bytes_mirrored) if bytes_mirrored else 'unknown size'})")
            else:
                st.success("✅ Video mirrored to local storage")
        except requests.RequestException as exc:
            mirror_url = f"{cfg['api_base']}{mirror_endpoint}"
            local_result["error_message"] = helpers.describe_error(mirror_url, exc)
            st.error(f"Mirror failed: {local_result['error_message']}")
            st.warning("Video uploaded successfully but local mirror failed. You may need to manually mirror from Episode Detail.")

    cloud_ok = bool(cloud_result.get("succeeded"))
    local_ok = bool(local_result.get("succeeded"))
    if cloud_ok and local_ok:
        outcome_state = "cloud_and_local_ok"
    elif cloud_ok and not local_ok:
        outcome_state = "cloud_ok_local_failed"
    elif not cloud_ok and local_ok:
        outcome_state = "cloud_failed_local_ok"
    else:
        outcome_state = "both_failed"

    if outcome_state == "cloud_and_local_ok":
        st.success("Upload complete – S3 and local mirror succeeded.")
    elif outcome_state == "cloud_ok_local_failed":
        st.warning("Cloud upload succeeded but local mirror failed. S3 is the source of truth.")
    elif outcome_state == "cloud_failed_local_ok":
        local_path_hint = local_result.get("path") or fallback_local_path
        if cloud_result.get("attempted"):
            st.warning(
                f"Cloud upload failed; local copy is available at {local_path_hint}. "
                "Episode will not be treated as fully uploaded to S3."
            )
        else:
            st.warning(
                f"Cloud upload was not attempted (local-only path). "
                f"Local copy is available at {local_path_hint}."
            )
    else:
        st.error("Upload failed. Cleaning up partial state.")
        if created_new_episode:
            _rollback_episode_creation(ep_id, cloud_result.get("bucket"), cloud_result.get("key"))
        return {"status": outcome_state, "ep_id": ep_id, "cloud_ok": cloud_ok, "local_ok": local_ok}

    cloud_label = cloud_result.get("status_label")
    cloud_status_text = cloud_label or ("OK" if cloud_ok else "FAILED" if cloud_result.get("attempted") else "NOT ATTEMPTED")
    st.caption(
        f"Upload status → Cloud: {cloud_status_text}, "
        f"Local: {'OK' if local_ok else 'FAILED'}"
    )

    if outcome_state in {"cloud_failed_local_ok", "cloud_ok_local_failed"}:
        with st.expander("Details"):
            st.write(f"S3 error type: {cloud_result.get('error_type') or 'n/a'}")
            st.write(f"S3 error message: {cloud_result.get('error_message') or 'n/a'}")
            st.write(f"Bucket: {cloud_result.get('bucket') or 'unknown'}")
            st.write(f"Region: {cloud_result.get('region') or 'unknown'}")
            if local_result.get("error_message"):
                st.write(f"Local mirror error: {local_result['error_message']}")

    if outcome_state == "cloud_and_local_ok":
        video_meta = _get_video_meta(ep_id)
        detected_fps_value = video_meta.get("fps_detected") if video_meta else None
        if detected_fps_value:
            st.info(f"Detected FPS: {detected_fps_value:.3f}")

        artifacts = {
            "video": get_path(ep_id, "video"),
        }
        flash_lines = ["Upload complete – S3 and local mirror succeeded.", f"Episode `{ep_id}`", f"Video → {helpers.link_local(artifacts['video'])}"]
        flash_lines.append("Go to Episode Detail to run detect/track processing.")

        st.session_state["upload_flash"] = "\n".join(flash_lines)
        # Avoid rerun so the post-upload success/progress UI remains visible.
        helpers.set_ep_id(ep_id, rerun=False, update_query_params=False)
    elif outcome_state != "both_failed":
        # Partial success: keep episode for manual remediation but avoid green flash
        helpers.set_ep_id(ep_id, rerun=False, update_query_params=False)

    return {
        "status": outcome_state,
        "ep_id": ep_id,
        "cloud_ok": cloud_ok,
        "local_ok": local_ok,
    }


def main():
    """Main application entry point with top-level error handling."""
    try:
        global cfg
        try:
            cfg = helpers.init_page("Screenalytics Upload")
        except Exception as init_exc:
            st.error("Failed to initialize upload page.")
            st.exception(init_exc)
            logging.exception("init_page failed in Upload_Video")
            st.stop()
        if DEBUG_UPLOAD:
            st.write("DEBUG: layout start")
        st.title("Upload & Run")
        st.caption("Upload page initialized.")
        if DEBUG_UPLOAD:
            st.write("DEBUG: init_page complete")

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
        jobs_running = _render_job_sections()
        if DEBUG_UPLOAD:
            st.write("DEBUG: rendered background jobs/progress")

        # Clean up stale upload progress keys to prevent memory leaks
        cleaned_count = _cleanup_stale_progress_keys()
        if DEBUG_UPLOAD and cleaned_count > 0:
            st.caption(f"Cleaned up {cleaned_count} stale progress entries")

        # Render S3 upload progress for any ongoing background uploads
        s3_uploads_in_progress = False
        with _session_state_lock:
            s3_progress_keys = [k for k in st.session_state.keys() if k.startswith("s3_upload_")]
            progress_snap = {k: st.session_state.get(k, {}) for k in s3_progress_keys}
        if progress_snap:
            st.subheader("Background S3 Uploads")
            for progress_key, upload_info in progress_snap.items():
                ep_id = progress_key.replace("s3_upload_", "")

                status = upload_info.get("status", "unknown")
                bytes_uploaded = upload_info.get("bytes_uploaded", 0)
                total_bytes = upload_info.get("total_bytes", 0)
                error = upload_info.get("error")

                st.markdown(f"**Episode `{ep_id}`**")

                if status == "uploading":
                    s3_uploads_in_progress = True
                    progress = bytes_uploaded / total_bytes if total_bytes > 0 else 0
                    st.progress(min(progress, 1.0))
                    mb_uploaded = bytes_uploaded / (1024**2)
                    mb_total = total_bytes / (1024**2)
                    st.caption(f"Uploading to S3: {mb_uploaded:.1f} MB / {mb_total:.1f} MB ({progress * 100:.1f}%)")
                elif status == "completed":
                    st.success(f"✅ Upload complete: {total_bytes / (1024**2):.1f} MB")
                    if st.button(f"Dismiss", key=f"dismiss_s3_{ep_id}"):
                        with _session_state_lock:
                            st.session_state.pop(progress_key, None)
                            st.session_state.pop(f"s3_future_{ep_id}", None)
                        st.rerun()
                elif status == "failed":
                    st.error(f"❌ Upload failed: {error}")
                    if st.button(f"Dismiss", key=f"dismiss_s3_{ep_id}"):
                        with _session_state_lock:
                            st.session_state.pop(progress_key, None)
                            st.session_state.pop(f"s3_future_{ep_id}", None)
                        st.rerun()

                st.divider()

        # Auto-refresh disabled for stability; manually refresh if needed.
        st.session_state.pop("upload_rerun_count", None)

        ep_id_param = helpers.get_ep_id_from_query_params(allow_app_injected=False)
        if ep_id_param:
            st.session_state["ep_id"] = ep_id_param
            mode = "replace"
        else:
            # Sidebar / new episode path: always start clean.
            st.session_state["ep_id"] = ""
            st.session_state.pop("_ep_id_query_origin", None)
            mode = "create"

        def _render_upload_summary(ep_id: str, *, created: bool, show_ref: str | None = None,
                                   season_number: int | None = None, episode_number: int | None = None) -> None:
            if not ep_id:
                return
            if created:
                season_label = f"{int(season_number):02d}" if season_number is not None else "??"
                episode_label = f"{int(episode_number):02d}" if episode_number is not None else "??"
                show_label = (show_ref or "unknown").strip() or "unknown"
                st.success(
                    f"Created episode `{ep_id}` (Show {show_label} · Season {season_label} Episode {episode_label}) "
                    "and uploaded video."
                )
            else:
                st.success(f"Replaced video for `{ep_id}`.")
            st.button(
                "Open Episode Detail",
                key=f"open_detail_after_upload_{ep_id}",
                on_click=lambda ep=ep_id: _navigate_to_detail_with_ep(ep),
            )

        upload_mode_choice: str
        if mode == "replace":
            st.radio(
                "Upload mode",
                ["New episode", "Existing episode"],
                index=1,
                key="upload_mode_choice_locked",
                disabled=True,
                help="ep_id provided in URL; uploads will replace this episode.",
            )
            upload_mode_choice = "Existing episode"
            st.subheader(f"Replace video for `{ep_id_param}`")
            st.caption("Upload is locked to this episode because ep_id is present in the URL.")
        else:
            st.caption("Mode: New episode (select show/season/episode to create)")
            upload_mode_choice = st.radio(
                "Upload mode",
                ["New episode", "Existing episode"],
                index=0,
                key="upload_mode_choice",
            )

        is_new_episode_mode = upload_mode_choice == "New episode" and mode != "replace"

        if is_new_episode_mode:
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
                uploaded_file = st.file_uploader(
                    "Episode video",
                    type=["mp4"],
                    accept_multiple_files=False,
                    help="Maximum file size: 10 GB. Supports full-length episode uploads.",
                )
                st.caption("Video will be uploaded to S3. After upload, open Episode Detail to run detect/track.")
                submit = st.form_submit_button("Create episode & upload", type="primary")

            if add_show_clicked:
                pending_show = (st.session_state.get("upload_new_show_input") or new_show_name or "").strip()
                if not pending_show:
                    st.error("Enter a new show slug/ID before adding.")
                else:
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
                        st.session_state["upload_show_choice"] = show_ref

                if not show_ref.strip():
                    st.warning("Please pick a show before uploading.")
                elif uploaded_file is None:
                    st.warning("Attach an .mp4 before submitting.")
                else:
                    if DEBUG_UPLOAD:
                        st.write("DEBUG: starting handle_episode_upload (new episode)")
                    with st.spinner("Creating episode and uploading video..."):
                        try:
                            outcome = handle_episode_upload(
                                show_ref=show_ref,
                                season_number=int(season_number),
                                episode_number=int(episode_number),
                                title=title,
                                include_air_date=include_air_date,
                                air_date_value=air_date_value,
                                uploaded_file=uploaded_file,
                                existing_ep_id=None,
                            )
                            if DEBUG_UPLOAD:
                                st.write(f"DEBUG: handle_episode_upload returned {outcome}")
                        except Exception as exc:  # final safety net to avoid blank UI
                            if DEBUG_UPLOAD:
                                st.write("DEBUG: handle_episode_upload raised")
                            st.error("Unexpected error during upload. See details below.")
                            st.exception(exc)
                        else:
                            if outcome and outcome.get("status") == "cloud_and_local_ok":
                                _render_upload_summary(
                                    outcome.get("ep_id", ""),
                                    created=True,
                                    show_ref=show_ref,
                                    season_number=int(season_number),
                                    episode_number=int(episode_number),
                                )
        else:
            existing_episode_error = None
            existing_ep_options: list[Dict[str, Any]] = []
            manual_existing_ep_id = ""
            selected_existing_ep_id = ep_id_param or ""
            episode_choice_label = "Upload to existing episode"
            if mode == "replace":
                episode_choice_label = f"Replace video for `{ep_id_param}`"

            def _format_existing_label(ep_id_val: str) -> str:
                parsed = helpers.parse_ep_id(ep_id_val)
                if parsed:
                    return f"{parsed['show'].upper()} · s{parsed['season']:02d}e{parsed['episode']:02d} ({ep_id_val})"
                return ep_id_val

            with st.form("existing-episode-upload"):
                st.subheader(episode_choice_label)
                if mode != "replace":
                    try:
                        episodes_payload = helpers.api_get("/episodes")
                        existing_ep_options = episodes_payload.get("episodes", [])
                    except requests.RequestException as exc:
                        existing_episode_error = helpers.describe_error(f"{cfg['api_base']}/episodes", exc)
                        existing_ep_options = []

                    ep_ids = [ep["ep_id"] for ep in existing_ep_options if ep.get("ep_id")]
                    selected_existing_ep_id = st.selectbox(
                        "Episode to replace",
                        ep_ids or [""],
                        format_func=lambda eid: _format_existing_label(eid) if eid else "Select an episode",
                        key="upload_existing_ep_select",
                    )
                    manual_existing_ep_id = st.text_input(
                        "Or enter episode ID",
                        key="upload_existing_ep_manual",
                        placeholder="rhoslc-s06e02",
                    )
                uploaded_file = st.file_uploader(
                    "Episode video",
                    type=["mp4"],
                    accept_multiple_files=False,
                    help="Select the replacement .mp4",
                )
                submit = st.form_submit_button("Upload replacement", type="primary")

            chosen_ep_id = (
                ep_id_param
                or (manual_existing_ep_id or "").strip()
                or (selected_existing_ep_id or "").strip()
            )

            if existing_episode_error:
                st.warning(f"Could not load tracked episodes for selection: {existing_episode_error}")

            if submit:
                if not chosen_ep_id:
                    st.warning("Select an episode to replace before uploading.")
                elif uploaded_file is None:
                    st.warning("Attach an .mp4 before submitting.")
                else:
                    if DEBUG_UPLOAD:
                        st.write("DEBUG: starting handle_episode_upload (existing episode)")
                    with st.spinner("Uploading video to S3 and local mirror..."):
                        try:
                            outcome = handle_episode_upload(
                                show_ref="",
                                season_number=0,
                                episode_number=0,
                                title=None,
                                include_air_date=False,
                                air_date_value=date.today(),
                                uploaded_file=uploaded_file,
                                existing_ep_id=chosen_ep_id,
                            )
                            if DEBUG_UPLOAD:
                                st.write(f"DEBUG: handle_episode_upload returned {outcome}")
                        except Exception as exc:  # final safety net to avoid blank UI
                            if DEBUG_UPLOAD:
                                st.write("DEBUG: handle_episode_upload raised")
                            st.error("Unexpected error during upload. See details below.")
                            st.exception(exc)
                        else:
                            if outcome and outcome.get("status") == "cloud_and_local_ok":
                                _render_upload_summary(outcome.get("ep_id", chosen_ep_id), created=False)

        st.button(
            "Open Episode Detail",
            on_click=lambda: _navigate_to_detail_with_ep(st.session_state.get("ep_id", "")),
        )

        if upload_mode_choice != "New episode" or mode == "replace":
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
                        return f"{item['ep_id']} · {size_mb} · {last_mod}"

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
                        st.caption(f"{show_label} · s{int(season_label):02d}e{int(episode_label):02d}")
                    tracked = bool(selected_item.get("exists_in_store"))
                    st.write(f"Tracked in store: {tracked}")
                    if ep_id_from_s3:
                        helpers.set_ep_id(ep_id_from_s3, rerun=False)

                    prefixes = helpers.episode_artifact_prefixes(ep_id_from_s3) if ep_id_from_s3 else None
                    if prefixes:
                        st.caption(
                            "S3 artifacts → "
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
                                with st.spinner("Queueing detect/track (RetinaFace + ByteTrack)…"):
                                    result = _launch_default_detect_track(
                                        ep_id_from_s3,
                                        label=f"S3 detect/track · {ep_id_from_s3}",
                                    )
                                if result and result.get("job"):
                                    st.rerun()

                        artifacts = {
                            "video": get_path(ep_id_from_s3, "video"),
                            "detections": get_path(ep_id_from_s3, "detections"),
                            "tracks": get_path(ep_id_from_s3, "tracks"),
                        }
                        st.caption(
                            f"Video → {helpers.link_local(artifacts['video'])} | "
                            f"Detections → {helpers.link_local(artifacts['detections'])} | "
                            f"Tracks → {helpers.link_local(artifacts['tracks'])}"
                        )
                elif s3_loaded:
                    st.warning("No S3 videos match that filter.")
            else:
                st.info("No S3 videos found (or API error).")

        # Auto-refresh if jobs running with lightweight backoff (capped at 0.5s to avoid UI freeze)
        if jobs_running:
            job_iteration = st.session_state.get("job_rerun_count", 0)
            st.session_state["job_rerun_count"] = job_iteration + 1
            sleep_time = min(0.5, 0.1 * (2 ** (job_iteration // 2)))
            time.sleep(sleep_time)
            st.rerun()
        else:
            # Reset counter when no jobs running
            st.session_state.pop("job_rerun_count", None)

    except Exception as exc:
        # Top-level error handler to prevent blank page crashes
        st.error("⚠️ An unexpected error occurred in the Upload page")
        st.error(f"Error type: {type(exc).__name__}")
        st.error(f"Error message: {str(exc)}")
        with st.expander("Full error details (for debugging)"):
            st.exception(exc)
        st.info("Try refreshing the page. If the problem persists, please report this issue.")
        logging.exception("Unexpected error in Upload page main()")
 
 
if __name__ == "__main__":
    main()
