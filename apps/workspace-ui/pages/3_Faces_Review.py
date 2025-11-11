from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

PROJECT_ROOT = PAGE_PATH.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import ui_helpers as helpers  # noqa: E402
from apps.api.services.storage import StorageService, episode_context_from_id  # noqa: E402

cfg = helpers.init_page("Faces & Tracks")
st.title("Faces & Tracks Review")


@st.cache_resource(show_spinner=False)
def _storage_client() -> StorageService | None:
    try:
        return StorageService()
    except Exception:
        return None


def _manifests_dir(ep_id: str) -> Path:
    return helpers.DATA_ROOT / "manifests" / ep_id


def _faces_path(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "faces.jsonl"


def _identities_path(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "identities.json"


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _load_facebank(ep_id: str) -> Dict[str, Any]:
    path = _identities_path(ep_id)
    if not path.exists():
        return {"ep_id": ep_id, "identities": [], "stats": {"faces": 0, "clusters": 0}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ep_id": ep_id, "identities": [], "stats": {"faces": 0, "clusters": 0}}
    if "identities" not in data:
        data["identities"] = []
    data.setdefault("stats", {"faces": 0, "clusters": len(data.get("identities", []))})
    return data


def _sync_facebank_to_s3(ep_id: str, path: Path) -> None:
    storage = _storage_client()
    if storage is None:
        return
    try:
        ctx = episode_context_from_id(ep_id)
    except ValueError:
        return
    try:
        storage.put_artifact(ctx, "manifests", path, path.name)
    except Exception:
        pass


def _save_facebank(ep_id: str, data: Dict[str, Any]) -> None:
    path = _identities_path(ep_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _sync_facebank_to_s3(ep_id, path)


def _faces_lookup(ep_id: str) -> tuple[Dict[int, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    faces_by_track: Dict[int, List[Dict[str, Any]]] = {}
    faces_index: Dict[str, Dict[str, Any]] = {}
    for row in _iter_jsonl(_faces_path(ep_id)):
        track_id = int(row.get("track_id", -1))
        faces_by_track.setdefault(track_id, []).append(row)
        face_id = row.get("face_id")
        if isinstance(face_id, str):
            faces_index[face_id] = row
    return faces_by_track, faces_index


def _recompute_identity_counts(facebank: Dict[str, Any], faces_by_track: Dict[int, List[Dict[str, Any]]]) -> None:
    for identity in facebank.get("identities", []):
        count = 0
        for track_id in identity.get("track_ids", []):
            count += len(faces_by_track.get(track_id, []))
        identity["count"] = count
    total_faces = sum(identity.get("count", 0) for identity in facebank.get("identities", []))
    facebank.setdefault("stats", {})["faces"] = total_faces
    facebank["stats"]["clusters"] = len(facebank.get("identities", []))


def _find_identity(facebank: Dict[str, Any], identity_id: str) -> Dict[str, Any] | None:
    for identity in facebank.get("identities", []):
        if identity.get("identity_id") == identity_id:
            return identity
    return None


def _merge_identities(facebank: Dict[str, Any], source_id: str, target_id: str) -> bool:
    if source_id == target_id:
        return False
    source = _find_identity(facebank, source_id)
    target = _find_identity(facebank, target_id)
    if not source or not target:
        return False
    target_tracks = set(target.get("track_ids", []))
    for track in source.get("track_ids", []):
        target_tracks.add(track)
    target["track_ids"] = sorted(target_tracks)
    target["samples"] = list({*(target.get("samples", []) or []), *(source.get("samples", []) or [])})[:5]
    facebank["identities"] = [identity for identity in facebank["identities"] if identity.get("identity_id") != source_id]
    return True


def _move_track(facebank: Dict[str, Any], track_id: int, target_id: str) -> bool:
    target = _find_identity(facebank, target_id)
    if not target:
        return False
    moved = False
    for identity in facebank.get("identities", []):
        tracks = identity.get("track_ids", [])
        if track_id in tracks:
            if identity is target:
                return False
            tracks.remove(track_id)
            identity["track_ids"] = tracks
            moved = True
    if moved:
        target_tracks = set(target.get("track_ids", []))
        target_tracks.add(track_id)
        target["track_ids"] = sorted(target_tracks)
    return moved


def _resolve_face_preview(ep_id: str, face: Dict[str, Any]) -> str | Path | None:
    rel_path = face.get("crop_rel_path")
    if isinstance(rel_path, str):
        local_path = helpers.DATA_ROOT / "frames" / ep_id / rel_path
        if local_path.exists():
            return local_path
    s3_key = face.get("crop_s3_key")
    if isinstance(s3_key, str):
        storage = _storage_client()
        if storage:
            url = storage.presign_get(s3_key)
            if url:
                return url
    return None


def _thumbnail_sources(ep_id: str, identity: Dict[str, Any], face_index: Dict[str, Dict[str, Any]]) -> List[tuple[str | Path, str]]:
    sources: List[Dict[str, Any]] = []
    rep = identity.get("rep")
    if isinstance(rep, dict):
        sources.append(rep)
    for face_id in identity.get("samples", []) or []:
        face = face_index.get(face_id)
        if face:
            sources.append(face)
    seen: set[tuple[int | None, int | None]] = set()
    thumbnails: List[tuple[str | Path, str]] = []
    for face in sources:
        key = (face.get("track_id"), face.get("frame_idx"))
        if key in seen:
            continue
        seen.add(key)
        preview = _resolve_face_preview(ep_id, face)
        if preview is None:
            continue
        caption = f"Track {face.get('track_id')}" if face.get("track_id") is not None else "Sample"
        thumbnails.append((preview, caption))
        if len(thumbnails) >= 3:
            break
    return thumbnails


def _require_episode() -> str:
    ep_id = helpers.get_ep_id()
    if ep_id:
        return ep_id
    try:
        payload = helpers.api_get("/episodes")
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}/episodes", exc))
        st.stop()
    episodes = payload.get("episodes", [])
    if not episodes:
        st.info("No episodes yet.")
        st.stop()
    option_lookup = {ep["ep_id"]: ep for ep in episodes}
    selection = st.selectbox(
        "Episode",
        list(option_lookup.keys()),
        format_func=lambda eid: f"{eid} ({option_lookup[eid]['show_slug']})",
    )
    if st.button("Load episode", use_container_width=True):
        helpers.set_ep_id(selection)
    st.stop()


ep_id = _require_episode()
helpers.set_ep_id(ep_id)

faces_by_track, faces_index = _faces_lookup(ep_id)
facebank = _load_facebank(ep_id)
identities = facebank.get("identities", [])

faces_path = _faces_path(ep_id)
identities_path = _identities_path(ep_id)
tracks_path = _manifests_dir(ep_id) / "tracks.jsonl"

default_device_label = helpers.device_default_label()
col_faces_run, col_cluster_run = st.columns(2)
with col_faces_run:
    st.markdown("### Faces Harvest")
    faces_stub = st.checkbox("Use stub (fast)", value=True, key="faces_stub_review")
    faces_device_choice = st.selectbox(
        "Device",
        helpers.DEVICE_LABELS,
        index=helpers.device_label_index(default_device_label),
        key="faces_device_review",
    )
    faces_device_value = helpers.DEVICE_VALUE_MAP[faces_device_choice]
    faces_save_crops = st.checkbox("Save face crops to S3", value=True, key="faces_crops_review")
    faces_jpeg_quality = st.number_input(
        "JPEG quality",
        min_value=50,
        max_value=100,
        value=85,
        step=5,
        key="faces_q_review",
    )
    if st.button("Run Faces Harvest", use_container_width=True, key="faces_run_review"):
        payload = {
            "ep_id": ep_id,
            "stub": bool(faces_stub),
            "device": faces_device_value,
            "save_crops": bool(faces_save_crops),
            "jpeg_quality": int(faces_jpeg_quality),
        }
        with st.spinner("Running faces harvest…"):
            summary, error_message = helpers.run_job_with_progress(
                ep_id,
                "/jobs/faces_embed",
                payload,
                requested_device=faces_device_value,
                async_endpoint="/jobs/faces_embed_async",
            )
        if error_message:
            st.error(error_message)
        else:
            normalized = helpers.normalize_summary(ep_id, summary)
            faces_count = normalized.get("faces")
            st.success(
                "Faces harvest complete" + (
                    f" · faces {faces_count:,}" if isinstance(faces_count, int) else ""
                )
            )
            st.rerun()

with col_cluster_run:
    st.markdown("### Cluster Identities")
    cluster_stub = st.checkbox("Use stub", value=True, key="cluster_stub_review")
    cluster_device_choice = st.selectbox(
        "Device",
        helpers.DEVICE_LABELS,
        index=helpers.device_label_index(default_device_label),
        key="cluster_device_review",
    )
    cluster_device_value = helpers.DEVICE_VALUE_MAP[cluster_device_choice]
    if st.button("Run Cluster", use_container_width=True, key="cluster_run_review"):
        payload = {"ep_id": ep_id, "stub": bool(cluster_stub), "device": cluster_device_value}
        with st.spinner("Clustering faces…"):
            summary, error_message = helpers.run_job_with_progress(
                ep_id,
                "/jobs/cluster",
                payload,
                requested_device=cluster_device_value,
                async_endpoint="/jobs/cluster_async",
            )
        if error_message:
            st.error(error_message)
        else:
            normalized = helpers.normalize_summary(ep_id, summary)
            ids_count = normalized.get("identities")
            st.success(
                "Clustering complete" + (
                    f" · identities {ids_count:,}" if isinstance(ids_count, int) else ""
                )
            )
            st.rerun()

st.subheader("Artifacts")
st.write(f"Faces manifest → {helpers.link_local(faces_path)}")
st.write(f"Identities manifest → {helpers.link_local(identities_path)}")
st.write(f"Tracks manifest → {helpers.link_local(tracks_path)}")

metric_faces = sum(len(items) for items in faces_by_track.values())
metrics = st.columns(3)
metrics[0].metric("Face samples", metric_faces)
metrics[1].metric("Identities", len(identities))
metrics[2].metric("Tracks linked", sum(len(identity.get("track_ids", [])) for identity in identities))

if not identities:
    st.info("No identities yet. Run faces harvest and cluster, then refresh.")
else:
    st.subheader("Manage identities")
    identity_ids = [identity.get("identity_id") for identity in identities if identity.get("identity_id")]

    with st.form("rename_identity_form"):
        rename_target = st.selectbox("Identity", identity_ids, key="rename_identity_select")
        new_label = st.text_input("New label", key="rename_identity_label")
        if st.form_submit_button("Rename"):
            target = _find_identity(facebank, rename_target)
            if target:
                target["label"] = new_label.strip() or target.get("label") or rename_target
                _recompute_identity_counts(facebank, faces_by_track)
                _save_facebank(ep_id, facebank)
                st.success("Identity renamed")
                st.rerun()

    with st.form("merge_identity_form"):
        source_id = st.selectbox("Source", identity_ids, key="merge_source")
        target_id = st.selectbox("Target", identity_ids, key="merge_target")
        if st.form_submit_button("Merge identities"):
            if source_id == target_id:
                st.warning("Choose two different identities")
            elif _merge_identities(facebank, source_id, target_id):
                _recompute_identity_counts(facebank, faces_by_track)
                _save_facebank(ep_id, facebank)
                st.success("Identities merged")
                st.rerun()
            else:
                st.error("Unable to merge identities")

    track_pool = sorted({track for identity in identities for track in identity.get("track_ids", [])})
    if track_pool:
        with st.form("move_track_form"):
            track_choice = st.selectbox("Track ID", track_pool, key="move_track_id")
            dest_identity = st.selectbox("Move to", identity_ids, key="move_track_dest")
            if st.form_submit_button("Move track"):
                if _move_track(facebank, int(track_choice), dest_identity):
                    _recompute_identity_counts(facebank, faces_by_track)
                    _save_facebank(ep_id, facebank)
                    st.success("Track reassigned")
                    st.rerun()
                else:
                    st.error("Track move failed")

    st.subheader("Facebank")
    for identity in identities:
        label = identity.get("label") or identity.get("identity_id")
        tracks = identity.get("track_ids", [])
        count = identity.get("count", 0)
        with st.container():
            st.markdown(f"**{label}** · Tracks: {', '.join(str(t) for t in tracks) or '—'} · Faces: {count}")
            thumbs = _thumbnail_sources(ep_id, identity, faces_index)
            if thumbs:
                cols = st.columns(len(thumbs))
                for col, (uri, caption) in zip(cols, thumbs):
                    with col:
                        st.image(uri, caption=caption, use_column_width=True)
            else:
                st.caption("No thumbnails saved yet. Run faces harvest with crops enabled.")
