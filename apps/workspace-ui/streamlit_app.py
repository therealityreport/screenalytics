from __future__ import annotations

import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from py_screenalytics.artifacts import ensure_dirs, get_path  # noqa: E402

try:  # noqa: E402 - prefer shared helper when available
    from apps.api.services.storage import parse_v2_episode_key as _shared_parse_v2_episode_key
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


def _get_video_meta(ep_id: str) -> Dict[str, Any] | None:
    try:
        return helpers.api_get(f"/episodes/{ep_id}/video_meta")
    except requests.RequestException as exc:
        st.warning(helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id}/video_meta", exc))
        return None


ASYNC_JOBS_KEY = "async_jobs"


def _job_state() -> Dict[str, Dict[str, Any]]:
    return st.session_state.setdefault(ASYNC_JOBS_KEY, {})


def _register_async_job(
    job_resp: Dict[str, Any],
    *,
    ep_id: str,
    label: str,
    stride: int,
    fps: float | None,
    stub: bool,
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
        "stub": stub,
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
            det_count = summary.get("detections_count")
            trk_count = summary.get("tracks_count")
            if det_count is not None and trk_count is not None:
                counts.append(f"detections: {det_count:,}")
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
    payload = helpers.default_detect_track_payload(ep_id, stub=False)
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
                st.error(
                    helpers.describe_error(f"{cfg['api_base']}/jobs/detect_track", sync_exc)
                )
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
        stub=payload.get("stub", False),
        device=payload.get("device", helpers.DEFAULT_DEVICE),
    )
    st.success(f"Job `{job_resp['job_id']}` queued; monitor progress above.")
    job_resp["payload"] = payload
    return {"job": job_resp, "payload": payload}

cfg = helpers.init_page("Screenalytics Upload")
st.title("Upload & Run")

if "detector" in st.session_state:
    del st.session_state["detector"]
if "tracker" in st.session_state:
    del st.session_state["tracker"]
st.cache_data.clear()

flash_message = st.session_state.pop("upload_flash", None)
if flash_message:
    st.success(flash_message)
jobs_running = _render_job_sections()


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


with st.form("episode-upload"):
    st.subheader("Upload new episode")
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
    st.caption(
        "Detect/track runs automatically after upload using RetinaFace + ByteTrack (det_thresh=0.5, device=auto)."
    )
    submit = st.form_submit_button("Upload episode")

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
        create_resp = helpers.api_post("/episodes", payload)
    except requests.RequestException as exc:
        endpoint = f"{cfg['api_base']}/episodes"
        st.error(f"Episode create failed: {helpers.describe_error(endpoint, exc)}")
        st.stop()

    ep_id = create_resp["ep_id"]
    st.info(f"Episode `{ep_id}` created. Requesting upload target…")

    presign_path = f"/episodes/{ep_id}/assets"
    try:
        presign_resp = helpers.api_post(presign_path)
    except requests.RequestException as exc:
        endpoint = f"{cfg['api_base']}{presign_path}"
        st.error(f"Presign failed: {helpers.describe_error(endpoint, exc)}")
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
            err = helpers.describe_error(upload_url, exc)
            st.error(f"Upload failed: {err}")
            st.stop()

    _mirror_local(ep_id, raw_bytes, presign_resp["local_video_path"])
    video_meta = _get_video_meta(ep_id)
    detected_fps_value = video_meta.get("fps_detected") if video_meta else None
    if detected_fps_value:
        st.info(f"Detected FPS: {detected_fps_value:.3f}")

    with st.spinner("Queueing detect/track (RetinaFace + ByteTrack)…"):
        job_result = _launch_default_detect_track(ep_id, label=f"Upload detect/track · {ep_id}")
    if job_result is None:
        st.stop()

    artifacts = {
        "video": get_path(ep_id, "video"),
        "detections": get_path(ep_id, "detections"),
        "tracks": get_path(ep_id, "tracks"),
    }
    flash_lines = [f"Episode `{ep_id}` upload complete."]
    flash_lines.append(f"Video → {helpers.link_local(artifacts['video'])}")
    flash_lines.append(f"Detections → {helpers.link_local(artifacts['detections'])}")
    flash_lines.append(f"Tracks → {helpers.link_local(artifacts['tracks'])}")
    if job_result.get("job"):
        job_resp = job_result["job"]
        flash_lines.append(
            f"Detect/track job `{job_resp['job_id']}` queued (RetinaFace + ByteTrack, auto device)."
        )
    elif job_result.get("sync"):
        sync_resp = job_result["sync"]
        dets = sync_resp.get("detections_count")
        trks = sync_resp.get("tracks_count")
        flash_lines.append(
            "Detect/track ran synchronously (RetinaFace + ByteTrack). "
            f"detections={dets if dets is not None else '?'} tracks={trks if trks is not None else '?'}"
        )
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
    filtered_items = [item for item in s3_items if not s3_search or s3_search in item["ep_id"].lower()]
    if filtered_items:
        def _format_item(item: Dict[str, Any]) -> str:
            size = item.get("size")
            size_mb = f"{(size or 0) / (1024**2):.1f} MB" if size else "size ?"
            last_mod = item.get("last_modified") or "unknown"
            return f"{item['ep_id']} · {size_mb} · {last_mod}"

        selected_index = st.selectbox(
            "S3 videos",
            list(range(len(filtered_items))),
            format_func=lambda idx: _format_item(filtered_items[idx]),
            key="s3_video_select",
        )
        selected_item = filtered_items[selected_index]
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
                        st.success(
                            f"Episode `{upsert_resp['ep_id']}` tracked (created={upsert_resp['created']})."
                        )
                        helpers.set_ep_id(upsert_resp["ep_id"])
                        st.rerun()
        elif ep_id_from_s3:
            action_cols = st.columns([1, 1, 1])
            with action_cols[0]:
                st.button(
                    "Open Episode Detail",
                    key=f"open_detail_{ep_id_from_s3}",
                    use_container_width=True,
                    on_click=lambda: helpers.try_switch_page("pages/2_Episode_Detail.py"),
                )
            with action_cols[1]:
                if st.button("Mirror from S3", key=f"mirror_{ep_id_from_s3}", use_container_width=True):
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

if jobs_running:
    time.sleep(0.5)
    st.rerun()
