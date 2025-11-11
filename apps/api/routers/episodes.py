from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from py_screenalytics.artifacts import ensure_dirs, get_path

from apps.api.services import roster as roster_service
from apps.api.services import identities as identity_service
from apps.api.services.episodes import EpisodeStore
from apps.api.services.storage import (
    StorageService,
    artifact_prefixes,
    delete_local_tree,
    delete_s3_prefix,
    episode_context_from_id,
    v2_artifact_prefixes,
)

router = APIRouter()
EPISODE_STORE = EpisodeStore()
STORAGE = StorageService()
LOGGER = logging.getLogger(__name__)


def _manifests_dir(ep_id: str) -> Path:
    return get_path(ep_id, "detections").parent


def _faces_path(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "faces.jsonl"


def _faces_ops_path(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "faces_ops.jsonl"


def _append_face_ops(ep_id: str, entries: Iterable[Dict[str, Any]]) -> None:
    entries = list(entries)
    if not entries:
        return
    path = _faces_ops_path(ep_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            payload = dict(entry)
            payload.setdefault("ts", timestamp)
            handle.write(json.dumps(payload) + "\n")


def _identities_path(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "identities.json"


def _runs_dir(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "runs"


def _tracks_path(ep_id: str) -> Path:
    return get_path(ep_id, "tracks")


def _thumbs_root(ep_id: str) -> Path:
    return get_path(ep_id, "frames_root") / "thumbs"


def _remove_face_assets(ep_id: str, rows: Iterable[Dict[str, Any]]) -> None:
    frames_root = get_path(ep_id, "frames_root")
    thumbs_root = _thumbs_root(ep_id)
    for row in rows:
        thumb_rel = row.get("thumb_rel_path")
        if isinstance(thumb_rel, str):
            thumb_file = thumbs_root / thumb_rel
            try:
                thumb_file.unlink()
            except FileNotFoundError:
                pass
        crop_rel = row.get("crop_rel_path")
        if isinstance(crop_rel, str):
            crop_file = frames_root / crop_rel
            try:
                crop_file.unlink()
            except FileNotFoundError:
                pass


def _analytics_root(ep_id: str) -> Path:
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
    return data_root / "analytics" / ep_id


def _episode_local_dirs(ep_id: str) -> List[Path]:
    dirs = [
        get_path(ep_id, "video").parent,
        get_path(ep_id, "frames_root"),
        _manifests_dir(ep_id),
        _analytics_root(ep_id),
    ]
    unique: List[Path] = []
    seen = set()
    for path in dirs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _load_run_marker(ep_id: str, phase: str) -> Dict[str, Any] | None:
    marker_path = _runs_dir(ep_id) / f"{phase}.json"
    if not marker_path.exists():
        return None
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _phase_status_from_marker(phase: str, marker: Dict[str, Any]) -> Dict[str, Any]:
    status_value = str(marker.get("status") or "unknown").lower()
    return {
        "phase": phase,
        "status": status_value,
        "faces": _safe_int(marker.get("faces")),
        "identities": _safe_int(marker.get("identities")),
        "started_at": marker.get("started_at"),
        "finished_at": marker.get("finished_at"),
        "version": marker.get("version"),
        "source": "marker",
    }


def _faces_phase_status(ep_id: str) -> Dict[str, Any]:
    marker = _load_run_marker(ep_id, "faces_embed")
    if marker:
        return _phase_status_from_marker("faces_embed", marker)
    faces_path = _faces_path(ep_id)
    faces_count = _count_nonempty_lines(faces_path)
    status_value = "success" if faces_count > 0 else "missing"
    source = "output" if faces_path.exists() else "absent"
    return {
        "phase": "faces_embed",
        "status": status_value,
        "faces": faces_count,
        "identities": None,
        "started_at": None,
        "finished_at": None,
        "version": None,
        "source": source,
    }


def _cluster_phase_status(ep_id: str) -> Dict[str, Any]:
    marker = _load_run_marker(ep_id, "cluster")
    if marker:
        return _phase_status_from_marker("cluster", marker)
    identities_path = _identities_path(ep_id)
    faces_total = 0
    identities_count = 0
    if identities_path.exists():
        try:
            payload = json.loads(identities_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        identities = payload.get("identities") if isinstance(payload, dict) else None
        if isinstance(identities, list):
            identities_count = len(identities)
        stats_block = payload.get("stats") if isinstance(payload, dict) else None
        if isinstance(stats_block, dict):
            faces_total = _safe_int(stats_block.get("faces")) or 0
    status_value = "success" if identities_count > 0 else "missing"
    source = "output" if identities_path.exists() else "absent"
    return {
        "phase": "cluster",
        "status": status_value,
        "faces": faces_total,
        "identities": identities_count,
        "started_at": None,
        "finished_at": None,
        "version": None,
        "source": source,
    }


def _delete_episode_assets(ep_id: str, options) -> Dict[str, Any]:
    record = EPISODE_STORE.get(ep_id)
    if not record:
        raise HTTPException(status_code=404, detail="Episode not found")
    try:
        ep_ctx = episode_context_from_id(ep_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid episode id") from exc
    local_deleted = 0
    delete_local = getattr(options, "delete_local", True)
    if delete_local:
        for path in _episode_local_dirs(ep_id):
            if not path.exists():
                continue
            try:
                delete_local_tree(path)
                local_deleted += 1
            except Exception as exc:  # pragma: no cover - best effort cleanup
                LOGGER.warning("Failed to delete %s: %s", path, exc)
    s3_deleted = 0
    include_s3 = bool(getattr(options, "include_s3", False) or getattr(options, "delete_artifacts", False))
    delete_raw = bool(getattr(options, "delete_raw", False))
    prefixes: Dict[str, str] | None = None
    if include_s3 or delete_raw:
        prefixes = v2_artifact_prefixes(ep_ctx)
    if include_s3 and prefixes:
        for key in ("frames", "crops", "manifests", "analytics", "thumbs_tracks", "thumbs_identities"):
            prefix = prefixes.get(key)
            if prefix:
                s3_deleted += delete_s3_prefix(STORAGE.bucket, prefix, storage=STORAGE)
    if delete_raw and prefixes:
        for key in ("raw_v2", "raw_v1"):
            prefix = prefixes.get(key)
            if prefix:
                s3_deleted += delete_s3_prefix(STORAGE.bucket, prefix, storage=STORAGE)
    removed = EPISODE_STORE.delete(ep_id)
    return {
        "ep_id": ep_id,
        "deleted": {"local_dirs": local_deleted, "s3_objects": s3_deleted},
        "removed_from_store": removed,
    }


def _delete_all_records(options) -> Dict[str, Any]:
    records = EPISODE_STORE.list()
    deleted: List[str] = []
    totals = {"local_dirs": 0, "s3_objects": 0}
    for record in records:
        result = _delete_episode_assets(record.ep_id, options)
        deleted.append(result["ep_id"])
        totals["local_dirs"] += result["deleted"]["local_dirs"]
        totals["s3_objects"] += result["deleted"]["s3_objects"]
    return {"deleted": totals, "episodes": deleted, "count": len(deleted)}


FRAME_IDX_RE = re.compile(r"frame_(\d+)\.jpg$", re.IGNORECASE)
TRACK_LIST_MAX_LIMIT = 500


def _load_faces(ep_id: str, *, include_skipped: bool = True) -> List[Dict[str, Any]]:
    path = _faces_path(ep_id)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
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
                if not include_skipped and obj.get("skip"):
                    continue
                rows.append(obj)
    return rows


def _write_faces(ep_id: str, rows: List[Dict[str, Any]]) -> Path:
    path = _faces_path(ep_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _load_tracks(ep_id: str) -> List[Dict[str, Any]]:
    path = _tracks_path(ep_id)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
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


def _write_tracks(ep_id: str, rows: List[Dict[str, Any]]) -> Path:
    path = _tracks_path(ep_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _load_identities(ep_id: str) -> Dict[str, Any]:
    path = _identities_path(ep_id)
    if not path.exists():
        return {"ep_id": ep_id, "identities": [], "stats": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ep_id": ep_id, "identities": [], "stats": {}}


def _write_identities(ep_id: str, payload: Dict[str, Any]) -> Path:
    path = _identities_path(ep_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _sync_manifests(ep_id: str, *paths: Path) -> None:
    try:
        ep_ctx = episode_context_from_id(ep_id)
    except ValueError:
        return
    for path in paths:
        if path and path.exists():
            try:
                STORAGE.put_artifact(ep_ctx, "manifests", path, path.name)
            except Exception:
                continue


def _identity_lookup(data: Dict[str, Any]) -> Dict[int, str]:
    lookup: Dict[int, str] = {}
    for identity in data.get("identities", []):
        identity_id = identity.get("identity_id")
        if not identity_id:
            continue
        for track_id in identity.get("track_ids", []) or []:
            try:
                lookup[int(track_id)] = identity_id
            except (TypeError, ValueError):
                continue
    return lookup


def _resolve_thumb_url(ep_id: str, rel_path: str | None, s3_key: str | None) -> str | None:
    if s3_key:
        url = STORAGE.presign_get(s3_key)
        if url:
            return url
    if not rel_path:
        return None
    local = _thumbs_root(ep_id) / rel_path
    if local.exists():
        return str(local)
    return None


def _recount_track_faces(ep_id: str) -> None:
    faces = _load_faces(ep_id, include_skipped=False)
    counts: Dict[int, int] = defaultdict(int)
    for face in faces:
        try:
            counts[int(face.get("track_id", -1))] += 1
        except (TypeError, ValueError):
            continue
    track_rows = _load_tracks(ep_id)
    if not track_rows:
        return
    for row in track_rows:
        try:
            track_id = int(row.get("track_id", -1))
        except (TypeError, ValueError):
            track_id = -1
        row["faces_count"] = counts.get(track_id, 0)
    path = _write_tracks(ep_id, track_rows)
    _sync_manifests(ep_id, path)


def _update_identity_stats(ep_id: str, payload: Dict[str, Any]) -> None:
    faces_count = len(_load_faces(ep_id, include_skipped=False))
    payload.setdefault("stats", {})
    payload["stats"]["faces"] = faces_count
    payload["stats"]["clusters"] = len(payload.get("identities", []))


def _frame_idx_from_name(name: str) -> int | None:
    match = FRAME_IDX_RE.search(name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _track_face_rows(ep_id: str, track_id: int) -> Dict[int, Dict[str, Any]]:
    faces = _load_faces(ep_id, include_skipped=False)
    rows: Dict[int, Dict[str, Any]] = {}
    for row in faces:
        try:
            tid = int(row.get("track_id", -1))
            frame_idx = int(row.get("frame_idx", -1))
        except (TypeError, ValueError):
            continue
        if tid != track_id:
            continue
        rows.setdefault(frame_idx, row)
    return rows


def _max_candidate_count(limit: int, offset: int, sample: int) -> int:
    limit = max(1, limit)
    offset = max(0, offset)
    sample = max(1, sample)
    return max(1, (offset + limit) * sample)


def _apply_sampling(entries: List[Dict[str, Any]], sample: int, offset: int, limit: int) -> List[Dict[str, Any]]:
    sample = max(1, sample)
    offset = max(0, offset)
    limit = max(1, limit)
    downsampled = [item for idx, item in enumerate(entries) if idx % sample == 0]
    if offset >= len(downsampled):
        return []
    end = offset + limit if limit else None
    return downsampled[offset:end]


def _require_episode_context(ep_id: str):
    try:
        ctx = episode_context_from_id(ep_id)
    except ValueError as exc:  # pragma: no cover - invalid ids rejected upstream
        raise HTTPException(status_code=400, detail="Invalid episode id") from exc
    return ctx, artifact_prefixes(ctx)


def _media_entry(track_id: int, frame_idx: int, key: str, url: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "track_id": track_id,
        "frame_idx": frame_idx,
        "key": key,
        "url": url,
        "w": meta.get("crop_width") or meta.get("width"),
        "h": meta.get("crop_height") or meta.get("height"),
        "ts": meta.get("ts"),
    }



def _list_track_frame_media(ep_id: str, track_id: int, sample: int, limit: int, offset: int) -> List[Dict[str, Any]]:
    sample = max(1, sample)
    limit = max(1, min(limit, TRACK_LIST_MAX_LIMIT))
    offset = max(0, offset)
    face_rows = _track_face_rows(ep_id, track_id)
    if not face_rows:
        return []
    frame_indices = sorted(face_rows.keys())
    max_candidates = _max_candidate_count(limit, offset, sample)
    frame_indices = frame_indices[:max_candidates]
    entries: List[Dict[str, Any]] = []
    if STORAGE.backend == "local":
        frames_dir = get_path(ep_id, "frames_root") / "frames"
        for idx in frame_indices:
            path = frames_dir / f"frame_{idx:06d}.jpg"
            if not path.exists():
                continue
            path_str = path.as_posix()
            meta = face_rows.get(idx, {})
            entries.append(_media_entry(track_id, idx, path_str, path_str, meta))
    else:
        _, prefixes = _require_episode_context(ep_id)
        base_prefix = prefixes["frames"]
        for idx in frame_indices:
            key = f"{base_prefix}frame_{idx:06d}.jpg"
            url = STORAGE.presign_get(key)
            if not url:
                continue
            meta = face_rows.get(idx, {})
            entries.append(_media_entry(track_id, idx, key, url, meta))
    entries.sort(key=lambda item: item["frame_idx"])
    return _apply_sampling(entries, sample, offset, limit)


class EpisodeCreateRequest(BaseModel):
    show_slug_or_id: str = Field(..., min_length=1, description="Show slug or identifier")
    season_number: int = Field(..., ge=0, le=999, description="Season number")
    episode_number: int = Field(..., ge=0, le=999, description="Episode number within the season")
    title: str | None = Field(None, max_length=200)
    air_date: date | None = None


class EpisodeCreateResponse(BaseModel):
    ep_id: str


class EpisodeSummary(BaseModel):
    ep_id: str
    show_slug: str
    season_number: int
    episode_number: int
    title: str | None
    air_date: str | None


class EpisodeListResponse(BaseModel):
    episodes: List[EpisodeSummary]


class EpisodeUpsert(BaseModel):
    ep_id: str = Field(..., min_length=3, description="Deterministic ep_id (slug-sXXeYY)")
    show_slug: str = Field(..., min_length=1)
    season: int = Field(..., ge=0, le=999)
    episode: int = Field(..., ge=0, le=999)
    title: str | None = Field(None, max_length=200)
    air_date: date | None = None


class FaceMoveRequest(BaseModel):
    from_track_id: int = Field(..., ge=0)
    face_ids: List[str] = Field(..., min_length=1, description="Face identifiers to move")
    target_identity_id: str | None = Field(None, description="Existing identity to receive frames")
    new_identity_name: str | None = Field(None, description="Create a new identity with this name")
    show_id: str | None = Field(None, description="Optional show slug for roster updates")


class TrackFrameMoveRequest(BaseModel):
    frame_ids: List[int] = Field(..., min_length=1, description="Track frame indices to move")
    target_identity_id: str | None = Field(None, description="Existing identity target")
    new_identity_name: str | None = Field(None, description="Optional new identity name")
    show_id: str | None = Field(None, description="Optional show slug override")


class TrackFrameDeleteRequest(BaseModel):
    frame_ids: List[int] = Field(..., min_length=1, description="Track frame indices to delete")
    delete_assets: bool = True


class IdentityRenameRequest(BaseModel):
    label: str | None = Field(None, max_length=120)


class IdentityNameRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    show: str | None = Field(None, description="Optional show slug override")


class IdentityMergeRequest(BaseModel):
    source_id: str
    target_id: str


class TrackMoveRequest(BaseModel):
    target_identity_id: str | None = None


class TrackDeleteRequest(BaseModel):
    delete_faces: bool = True


class FrameDeleteRequest(BaseModel):
    track_id: int
    frame_idx: int
    delete_assets: bool = False




class DeleteEpisodeIn(BaseModel):
    include_s3: bool = False


class DeleteAllIn(BaseModel):
    confirm: str
    include_s3: bool = False


class DeleteEpisodeLegacyIn(BaseModel):
    delete_artifacts: bool = True
    delete_raw: bool = False
    delete_local: bool = True


class PurgeAllLegacyIn(BaseModel):
    confirm: str
    delete_artifacts: bool = True
    delete_raw: bool = False
    delete_local: bool = True


class S3VideoItem(BaseModel):
    bucket: str
    key: str
    ep_id: str
    show: str | None = None
    season: int | None = None
    episode: int | None = None
    size: int | None = None
    last_modified: str | None = None
    etag: str | None = None
    exists_in_store: bool
    key_version: str | None = None


class S3VideosResponse(BaseModel):
    items: List[S3VideoItem]
    count: int


class EpisodeS3Status(BaseModel):
    bucket: str
    v2_key: str | None = None
    v2_exists: bool = False
    v1_key: str
    v1_exists: bool


class EpisodeLocalStatus(BaseModel):
    path: str
    exists: bool


class EpisodeDetailResponse(BaseModel):
    ep_id: str
    show_slug: str
    season_number: int
    episode_number: int
    title: str | None
    air_date: str | None
    s3: EpisodeS3Status
    local: EpisodeLocalStatus


class PhaseStatus(BaseModel):
    phase: str
    status: str
    faces: int | None = None
    identities: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    version: str | None = None
    source: str | None = None


class EpisodeStatusResponse(BaseModel):
    ep_id: str
    faces_embed: PhaseStatus
    cluster: PhaseStatus


class AssetUploadResponse(BaseModel):
    ep_id: str
    method: str
    bucket: str
    key: str
    object_key: str | None = None  # backwards compatibility
    upload_url: str | None
    expires_in: int | None
    headers: Dict[str, str]
    path: str | None = None
    local_video_path: str
    backend: str


class EpisodeMirrorResponse(BaseModel):
    ep_id: str
    local_video_path: str
    bytes: int | None = None
    etag: str | None = None
    used_key_version: str | None = None


class EpisodeVideoMeta(BaseModel):
    ep_id: str
    local_exists: bool
    local_video_path: str
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    duration_sec: float | None = None
    fps_detected: float | None = None


class DeleteEpisodeIn(BaseModel):
    delete_artifacts: bool = True
    delete_raw: bool = False
    delete_local: bool = True


class PurgeAllIn(BaseModel):
    confirm: str
    delete_artifacts: bool = True
    delete_raw: bool = False
    delete_local: bool = True


@router.get("/episodes", response_model=EpisodeListResponse, tags=["episodes"])
def list_episodes() -> EpisodeListResponse:
    records = EPISODE_STORE.list()
    episodes = [
        EpisodeSummary(
            ep_id=record.ep_id,
            show_slug=record.show_ref,
            season_number=record.season_number,
            episode_number=record.episode_number,
            title=record.title,
            air_date=record.air_date,
        )
        for record in records
    ]
    return EpisodeListResponse(episodes=episodes)


@router.get("/episodes/s3_videos", response_model=S3VideosResponse, tags=["episodes"])
def list_s3_videos(q: str | None = Query(None), limit: int = Query(200, ge=1, le=1000)) -> S3VideosResponse:
    raw_items = STORAGE.list_episode_videos_s3(limit=limit)
    items: List[S3VideoItem] = []
    for obj in raw_items:
        ep_id = obj.get("ep_id")
        if not isinstance(ep_id, str):
            continue
        if q and q.lower() not in ep_id.lower():
            continue
        items.append(
            S3VideoItem(
                bucket=obj.get("bucket", STORAGE.bucket),
                key=str(obj.get("key")),
                ep_id=ep_id,
                show=obj.get("show"),
                season=obj.get("season"),
                episode=obj.get("episode"),
                size=obj.get("size"),
                last_modified=str(obj.get("last_modified")) if obj.get("last_modified") else None,
                etag=obj.get("etag"),
                exists_in_store=EPISODE_STORE.exists(ep_id),
                key_version=obj.get("key_version"),
            )
        )
        if len(items) >= limit:
            break
    return S3VideosResponse(items=items, count=len(items))


@router.post("/episodes/{ep_id}/delete")
def delete_episode_new(ep_id: str, body: DeleteEpisodeIn = Body(default=DeleteEpisodeIn())) -> Dict[str, Any]:
    return _delete_episode_assets(ep_id, body)


@router.post("/episodes/delete_all")
def delete_all(body: DeleteAllIn) -> Dict[str, Any]:
    if body.confirm.strip() != "DELETE ALL":
        raise HTTPException(status_code=400, detail="Confirmation text mismatch.")
    delete_opts = DeleteEpisodeIn(include_s3=body.include_s3)
    return _delete_all_records(delete_opts)


@router.delete("/episodes/{ep_id}")
def delete_episode(ep_id: str, body: DeleteEpisodeLegacyIn = Body(default=DeleteEpisodeLegacyIn())) -> Dict[str, Any]:
    return _delete_episode_assets(ep_id, body)


@router.post("/episodes/purge_all")
def purge_all(body: PurgeAllLegacyIn) -> Dict[str, Any]:
    if body.confirm.strip() != "DELETE ALL":
        raise HTTPException(status_code=400, detail="Confirmation text mismatch.")
    delete_opts = DeleteEpisodeLegacyIn(
        delete_artifacts=body.delete_artifacts,
        delete_raw=body.delete_raw,
        delete_local=body.delete_local,
    )
    return _delete_all_records(delete_opts)


@router.get("/episodes/{ep_id}", response_model=EpisodeDetailResponse, tags=["episodes"])
def episode_details(ep_id: str) -> EpisodeDetailResponse:
    record = EPISODE_STORE.get(ep_id)
    if not record:
        raise HTTPException(status_code=404, detail="Episode not found")

    local_path = get_path(ep_id, "video")
    v2_key = STORAGE.video_object_key_v2(record.show_ref, record.season_number, record.episode_number)
    v1_key = STORAGE.video_object_key_v1(ep_id)
    v2_exists = STORAGE.object_exists(v2_key)
    v1_exists = STORAGE.object_exists(v1_key)

    return EpisodeDetailResponse(
        ep_id=record.ep_id,
        show_slug=record.show_ref,
        season_number=record.season_number,
        episode_number=record.episode_number,
        title=record.title,
        air_date=record.air_date,
        s3=EpisodeS3Status(
            bucket=STORAGE.bucket,
            v2_key=v2_key,
            v2_exists=v2_exists,
            v1_key=v1_key,
            v1_exists=v1_exists,
        ),
        local=EpisodeLocalStatus(path=str(local_path), exists=local_path.exists()),
    )


@router.get("/episodes/{ep_id}/progress", tags=["episodes"])
def episode_progress(ep_id: str) -> dict:
    progress_path = get_path(ep_id, "detections").parent / "progress.json"
    if not progress_path.exists():
        raise HTTPException(status_code=404, detail="Progress not available")
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=503, detail="Progress file corrupt") from exc
    return {"ep_id": ep_id, "progress": payload}


@router.get("/episodes/{ep_id}/status", response_model=EpisodeStatusResponse, tags=["episodes"])
def episode_run_status(ep_id: str) -> EpisodeStatusResponse:
    faces_status = PhaseStatus(**_faces_phase_status(ep_id))
    cluster_status = PhaseStatus(**_cluster_phase_status(ep_id))
    return EpisodeStatusResponse(ep_id=ep_id, faces_embed=faces_status, cluster=cluster_status)


@router.post("/episodes", response_model=EpisodeCreateResponse, tags=["episodes"])
def create_episode(payload: EpisodeCreateRequest) -> EpisodeCreateResponse:
    record = EPISODE_STORE.upsert(
        show_ref=payload.show_slug_or_id,
        season_number=payload.season_number,
        episode_number=payload.episode_number,
        title=payload.title,
        air_date=payload.air_date,
    )
    return EpisodeCreateResponse(ep_id=record.ep_id)


@router.post("/episodes/upsert_by_id", tags=["episodes"])
def upsert_by_id(payload: EpisodeUpsert) -> dict:
    try:
        record, created = EPISODE_STORE.upsert_ep_id(
            ep_id=payload.ep_id,
            show_slug=payload.show_slug,
            season=payload.season,
            episode=payload.episode,
            title=payload.title,
            air_date=payload.air_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ep_id": record.ep_id,
        "created": created,
        "show_slug": record.show_ref,
        "season": record.season_number,
        "episode": record.episode_number,
        "title": record.title,
        "air_date": record.air_date,
    }


@router.post("/episodes/{ep_id}/assets", response_model=AssetUploadResponse, tags=["episodes"])
def presign_episode_assets(ep_id: str) -> AssetUploadResponse:
    record = EPISODE_STORE.get(ep_id)
    if not record:
        raise HTTPException(status_code=404, detail="Episode not found")

    ensure_dirs(ep_id)
    v2_key = STORAGE.video_object_key_v2(record.show_ref, record.season_number, record.episode_number)
    presigned = STORAGE.presign_episode_video(ep_id, object_key=v2_key)
    local_video_path = get_path(ep_id, "video")
    path = presigned.path or (str(local_video_path) if presigned.method == "FILE" else None)

    return AssetUploadResponse(
        ep_id=presigned.ep_id,
        method=presigned.method,
        bucket=presigned.bucket,
        key=presigned.object_key,
        object_key=presigned.object_key,
        upload_url=presigned.upload_url,
        expires_in=presigned.expires_in,
        headers=presigned.headers,
        path=path,
        local_video_path=str(local_video_path),
        backend=presigned.backend,
    )


@router.post(
    "/episodes/{ep_id}/mirror",
    response_model=EpisodeMirrorResponse,
    tags=["episodes"],
)
def mirror_episode_video(ep_id: str) -> EpisodeMirrorResponse:
    record = EPISODE_STORE.get(ep_id)
    if not record:
        raise HTTPException(status_code=404, detail="Episode not found")

    ensure_dirs(ep_id)
    try:
        result = STORAGE.ensure_local_mirror(
            ep_id,
            show_ref=record.show_ref,
            season_number=record.season_number,
            episode_number=record.episode_number,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return EpisodeMirrorResponse(
        ep_id=ep_id,
        local_video_path=str(result.get("local_video_path")),
        bytes=result.get("bytes"),
        etag=result.get("etag"),
        used_key_version=result.get("used_key_version"),
    )


@router.post(
    "/episodes/{ep_id}/hydrate",
    response_model=EpisodeMirrorResponse,
    tags=["episodes"],
)
def hydrate_episode_video(ep_id: str) -> EpisodeMirrorResponse:
    return mirror_episode_video(ep_id)


@router.get(
    "/episodes/{ep_id}/video_meta",
    response_model=EpisodeVideoMeta,
    tags=["episodes"],
)
def episode_video_meta(ep_id: str) -> EpisodeVideoMeta:
    video_path = get_path(ep_id, "video")
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Local video not found")

    width: int | None = None
    height: int | None = None
    frames: int | None = None
    fps_detected: float | None = None
    duration_sec: float | None = None

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
            frames_val = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frames = int(frames_val) if frames_val and frames_val > 0 else None
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            fps_detected = float(fps_val) if fps_val and fps_val > 0 else None
            if frames and fps_detected:
                duration_sec = frames / fps_detected
        cap.release()
    except Exception as exc:  # pragma: no cover - best effort
        raise HTTPException(status_code=500, detail=f"Failed to analyze video: {exc}") from exc

    return EpisodeVideoMeta(
        ep_id=ep_id,
        local_exists=True,
        local_video_path=str(video_path),
        width=width,
        height=height,
        frames=frames,
        duration_sec=duration_sec,
        fps_detected=fps_detected,
    )


@router.get("/episodes/{ep_id}/identities")
def list_identities(ep_id: str) -> dict:
    payload = _load_identities(ep_id)
    track_lookup = {int(row.get("track_id", -1)): row for row in _load_tracks(ep_id)}
    identities = []
    for identity in payload.get("identities", []):
        track_ids = []
        for raw_tid in identity.get("track_ids", []) or []:
            try:
                track_ids.append(int(raw_tid))
            except (TypeError, ValueError):
                continue
        faces_total = identity.get("size")
        if faces_total is None:
            faces_total = sum(int(track_lookup.get(tid, {}).get("faces_count", 0)) for tid in track_ids)
        identities.append(
            {
                "identity_id": identity.get("identity_id"),
                "label": identity.get("label"),
                "name": identity.get("name"),
                "track_ids": track_ids,
                "faces": faces_total,
                "rep_thumbnail_url": _resolve_thumb_url(
                    ep_id,
                    identity.get("rep_thumb_rel_path"),
                    identity.get("rep_thumb_s3_key"),
                ),
            }
        )
    try:
        ctx = episode_context_from_id(ep_id)
        show_slug = ctx.show_slug
    except ValueError:
        show_slug = None
    if show_slug:
        for entry in payload.get("identities", []):
            name = entry.get("name")
            if isinstance(name, str) and name.strip():
                try:
                    roster_service.add_if_missing(show_slug, name)
                except ValueError:
                    pass
    return {"identities": identities, "stats": payload.get("stats", {})}


@router.get("/episodes/{ep_id}/cluster_tracks")
def list_cluster_tracks(
    ep_id: str,
    limit_per_cluster: int | None = Query(None, ge=1, description="Optional max tracks per cluster"),
) -> dict:
    try:
        return identity_service.cluster_track_summary(ep_id, limit_per_cluster=limit_per_cluster)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/episodes/{ep_id}/faces_grid")
def faces_grid(ep_id: str, track_id: int | None = Query(None)) -> dict:
    faces = _load_faces(ep_id, include_skipped=False)
    identity_lookup = _identity_lookup(_load_identities(ep_id))
    items: List[dict] = []
    for row in faces:
        try:
            tid = int(row.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        if track_id is not None and tid != track_id:
            continue
        items.append(
            {
                "face_id": row.get("face_id"),
                "track_id": tid,
                "frame_idx": row.get("frame_idx"),
                "ts": row.get("ts"),
                "thumbnail_url": _resolve_thumb_url(ep_id, row.get("thumb_rel_path"), row.get("thumb_s3_key")),
                "identity_id": identity_lookup.get(tid),
            }
        )
    return {"faces": items, "count": len(items)}


@router.get("/episodes/{ep_id}/identities/{identity_id}")
def identity_detail(ep_id: str, identity_id: str) -> dict:
    payload = _load_identities(ep_id)
    identity = next((item for item in payload.get("identities", []) if item.get("identity_id") == identity_id), None)
    if not identity:
        raise HTTPException(status_code=404, detail="Identity not found")
    track_lookup = {int(row.get("track_id", -1)): row for row in _load_tracks(ep_id)}
    tracks_payload = []
    for raw_tid in identity.get("track_ids", []) or []:
        try:
            tid = int(raw_tid)
        except (TypeError, ValueError):
            continue
        track_row = track_lookup.get(tid, {})
        tracks_payload.append(
            {
                "track_id": tid,
                "faces_count": track_row.get("faces_count", 0),
                "thumbnail_url": _resolve_thumb_url(ep_id, track_row.get("thumb_rel_path"), track_row.get("thumb_s3_key")),
            }
        )
    return {
        "identity": {
            "identity_id": identity_id,
            "label": identity.get("label"),
            "name": identity.get("name"),
            "track_ids": identity.get("track_ids", []),
            "rep_thumbnail_url": _resolve_thumb_url(
                ep_id,
                identity.get("rep_thumb_rel_path"),
                identity.get("rep_thumb_s3_key"),
            ),
        },
        "tracks": tracks_payload,
    }


@router.get("/episodes/{ep_id}/tracks/{track_id}")
def track_detail(ep_id: str, track_id: int) -> dict:
    faces = [row for row in _load_faces(ep_id, include_skipped=False) if int(row.get("track_id", -1)) == track_id]
    frames = [
        {
            "face_id": row.get("face_id"),
            "frame_idx": row.get("frame_idx"),
            "ts": row.get("ts"),
            "thumbnail_url": _resolve_thumb_url(ep_id, row.get("thumb_rel_path"), row.get("thumb_s3_key")),
            "skip": row.get("skip"),
        }
        for row in faces
    ]
    track_row = next((row for row in _load_tracks(ep_id) if int(row.get("track_id", -1)) == track_id), None)
    return {
        "track_id": track_id,
        "faces_count": len(frames),
        "thumbnail_url": _resolve_thumb_url(
            ep_id,
            (track_row or {}).get("thumb_rel_path"),
            (track_row or {}).get("thumb_s3_key"),
        ),
        "frames": frames,
    }


@router.get("/episodes/{ep_id}/tracks/{track_id}/crops")
def list_track_crops(
    ep_id: str,
    track_id: int,
    sample: int = Query(5, ge=1, le=100, description="Return every Nth crop"),
    limit: int = Query(200, ge=1, le=TRACK_LIST_MAX_LIMIT),
    start_after: str | None = Query(None, description="Opaque cursor returned by the previous call"),
) -> Dict[str, Any]:
    ctx, _ = _require_episode_context(ep_id)
    payload = STORAGE.list_track_crops(ctx, track_id, sample=sample, max_keys=limit, start_after=start_after)
    face_rows = _track_face_rows(ep_id, track_id)
    items = payload.get("items", []) if isinstance(payload, dict) else []
    for item in items:
        frame_idx = item.get("frame_idx")
        try:
            frame_int = int(frame_idx)
        except (TypeError, ValueError):
            continue
        meta = face_rows.get(frame_int, {})
        if meta:
            if "w" not in item and "crop_width" in meta:
                item["w"] = meta.get("crop_width") or meta.get("width")
            if "h" not in item and "crop_height" in meta:
                item["h"] = meta.get("crop_height") or meta.get("height")
            item.setdefault("ts", meta.get("ts"))
    return payload


@router.get("/episodes/{ep_id}/tracks/{track_id}/frames")
def list_track_frames(
    ep_id: str,
    track_id: int,
    sample: int = Query(5, ge=1, le=100, description="Return every Nth frame"),
    limit: int = Query(200, ge=1, le=TRACK_LIST_MAX_LIMIT),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    return _list_track_frame_media(ep_id, track_id, sample, limit, offset)


@router.post("/episodes/{ep_id}/tracks/{track_id}/frames/move")
def move_track_frames(ep_id: str, track_id: int, body: TrackFrameMoveRequest) -> dict:
    frame_ids = sorted({int(idx) for idx in body.frame_ids or []})
    if not frame_ids:
        raise HTTPException(status_code=400, detail="frame_ids_required")
    face_rows = _track_face_rows(ep_id, track_id)
    if not face_rows:
        raise HTTPException(status_code=404, detail="track_not_found")
    selected_faces: List[str] = []
    ops: List[Dict[str, Any]] = []
    for frame_idx in frame_ids:
        row = face_rows.get(frame_idx)
        if not row:
            raise HTTPException(status_code=404, detail=f"frame_not_found:{frame_idx}")
        face_id = row.get("face_id")
        if not face_id:
            raise HTTPException(status_code=400, detail=f"face_id_missing:{frame_idx}")
        selected_faces.append(str(face_id))
        ops.append({"frame_idx": frame_idx, "face_id": face_id})
    try:
        result = identity_service.move_frames(
            ep_id,
            track_id,
            selected_faces,
            target_identity_id=body.target_identity_id,
            new_identity_name=body.new_identity_name,
            show_id=body.show_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _append_face_ops(
        ep_id,
        [
            {
                "op": "move_frame",
                "frame_idx": entry["frame_idx"],
                "face_id": entry["face_id"],
                "source_track_id": track_id,
                "target_track_id": result.get("new_track_id"),
                "target_identity_id": result.get("target_identity_id") or body.target_identity_id,
            }
            for entry in ops
        ],
    )
    return {
        "moved": len(selected_faces),
        "frame_ids": frame_ids,
        "new_track_id": result.get("new_track_id"),
        "target_identity_id": result.get("target_identity_id"),
        "target_name": result.get("target_name"),
        "clusters": result.get("clusters"),
    }


@router.delete("/episodes/{ep_id}/tracks/{track_id}/frames")
def delete_track_frames(ep_id: str, track_id: int, body: TrackFrameDeleteRequest) -> dict:
    frame_ids = sorted({int(idx) for idx in body.frame_ids or []})
    if not frame_ids:
        raise HTTPException(status_code=400, detail="frame_ids_required")
    faces = _load_faces(ep_id)
    removed: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []
    target_set = set(frame_ids)
    for row in faces:
        try:
            tid = int(row.get("track_id", -1))
        except (TypeError, ValueError):
            tid = -1
        frame_idx = row.get("frame_idx")
        try:
            frame_val = int(frame_idx)
        except (TypeError, ValueError):
            frame_val = None
        if tid == track_id and frame_val in target_set:
            removed.append(row)
        else:
            kept.append(row)
    if not removed:
        raise HTTPException(status_code=404, detail="frames_not_found")
    faces_path = _write_faces(ep_id, kept)
    if body.delete_assets:
        _remove_face_assets(ep_id, removed)
    _append_face_ops(
        ep_id,
        [
            {
                "op": "delete_frame",
                "track_id": track_id,
                "frame_idx": int(row.get("frame_idx", -1)),
                "face_id": row.get("face_id"),
            }
            for row in removed
        ],
    )
    _recount_track_faces(ep_id)
    identities = _load_identities(ep_id)
    _update_identity_stats(ep_id, identities)
    identities_path = _write_identities(ep_id, identities)
    _sync_manifests(ep_id, faces_path, identities_path)
    return {
        "track_id": track_id,
        "deleted": len(removed),
        "remaining": len(kept),
    }


@router.post("/episodes/{ep_id}/identities/{identity_id}/rename")
def rename_identity(ep_id: str, identity_id: str, body: IdentityRenameRequest) -> dict:
    payload = _load_identities(ep_id)
    identity = next((item for item in payload.get("identities", []) if item.get("identity_id") == identity_id), None)
    if not identity:
        raise HTTPException(status_code=404, detail="Identity not found")
    label = (body.label or "").strip()
    identity["label"] = label or None
    _update_identity_stats(ep_id, payload)
    path = _write_identities(ep_id, payload)
    _sync_manifests(ep_id, path)
    return {"identity_id": identity_id, "label": identity["label"]}


@router.post("/episodes/{ep_id}/identities/{identity_id}/name")
def assign_identity_name(ep_id: str, identity_id: str, body: IdentityNameRequest) -> dict:
    try:
        return identity_service.assign_identity_name(ep_id, identity_id, body.name, body.show)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/episodes/{ep_id}/identities/merge")
def merge_identities(ep_id: str, body: IdentityMergeRequest) -> dict:
    payload = _load_identities(ep_id)
    identities = payload.get("identities", [])
    source = next((item for item in identities if item.get("identity_id") == body.source_id), None)
    target = next((item for item in identities if item.get("identity_id") == body.target_id), None)
    if not source or not target:
        raise HTTPException(status_code=404, detail="Source or target identity not found")
    merged_track_ids = set(target.get("track_ids", []) or [])
    for tid in source.get("track_ids", []) or []:
        merged_track_ids.add(tid)
    target["track_ids"] = sorted({int(x) for x in merged_track_ids})
    payload["identities"] = [item for item in identities if item.get("identity_id") != body.source_id]
    _update_identity_stats(ep_id, payload)
    path = _write_identities(ep_id, payload)
    _sync_manifests(ep_id, path)
    return {"target_id": body.target_id, "track_ids": target["track_ids"]}


@router.post("/episodes/{ep_id}/tracks/{track_id}/move")
def move_track(ep_id: str, track_id: int, body: TrackMoveRequest) -> dict:
    payload = _load_identities(ep_id)
    identities = payload.get("identities", [])
    source_identity = None
    target_identity = None
    for identity in identities:
        if body.target_identity_id and identity.get("identity_id") == body.target_identity_id:
            target_identity = identity
        if track_id in identity.get("track_ids", []):
            source_identity = identity
    if body.target_identity_id and target_identity is None:
        raise HTTPException(status_code=404, detail="Target identity not found")
    if source_identity and track_id in source_identity.get("track_ids", []):
        source_identity["track_ids"] = [tid for tid in source_identity["track_ids"] if tid != track_id]
    if target_identity is not None:
        if track_id not in target_identity.get("track_ids", []):
            target_identity.setdefault("track_ids", []).append(track_id)
            target_identity["track_ids"] = sorted(target_identity["track_ids"])
    _update_identity_stats(ep_id, payload)
    path = _write_identities(ep_id, payload)
    _sync_manifests(ep_id, path)
    return {
        "identity_id": body.target_identity_id,
        "track_ids": target_identity["track_ids"] if target_identity else [],
    }


@router.post("/episodes/{ep_id}/faces/move_frames")
def move_faces(ep_id: str, body: FaceMoveRequest) -> dict:
    try:
        return identity_service.move_frames(
            ep_id,
            body.from_track_id,
            body.face_ids,
            target_identity_id=body.target_identity_id,
            new_identity_name=body.new_identity_name,
            show_id=body.show_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc




@router.delete("/episodes/{ep_id}/identities/{identity_id}")
def delete_identity(ep_id: str, identity_id: str) -> dict:
    payload = _load_identities(ep_id)
    identities = payload.get("identities", [])
    before = len(identities)
    payload["identities"] = [item for item in identities if item.get("identity_id") != identity_id]
    if len(payload["identities"]) == before:
        raise HTTPException(status_code=404, detail="Identity not found")
    _update_identity_stats(ep_id, payload)
    path = _write_identities(ep_id, payload)
    _sync_manifests(ep_id, path)
    return {"deleted": identity_id, "remaining": len(payload["identities"])}


@router.delete("/episodes/{ep_id}/tracks/{track_id}")
def delete_track(ep_id: str, track_id: int, payload: TrackDeleteRequest = Body(default=TrackDeleteRequest())) -> dict:
    faces = _load_faces(ep_id)
    if payload.delete_faces:
        faces = [row for row in faces if int(row.get("track_id", -1)) != track_id]
        faces_path = _write_faces(ep_id, faces)
    else:
        faces_path = _faces_path(ep_id)
    track_rows = _load_tracks(ep_id)
    kept_tracks = [row for row in track_rows if int(row.get("track_id", -1)) != track_id]
    if len(kept_tracks) == len(track_rows):
        raise HTTPException(status_code=404, detail="Track not found")
    tracks_path = _write_tracks(ep_id, kept_tracks)
    identities = _load_identities(ep_id)
    for identity in identities.get("identities", []):
        identity["track_ids"] = [tid for tid in identity.get("track_ids", []) if tid != track_id]
    _update_identity_stats(ep_id, identities)
    identities_path = _write_identities(ep_id, identities)
    _recount_track_faces(ep_id)
    _sync_manifests(ep_id, faces_path, tracks_path, identities_path)
    return {"track_id": track_id, "faces_deleted": payload.delete_faces}


@router.delete("/episodes/{ep_id}/frames")
def delete_frame(ep_id: str, payload: FrameDeleteRequest) -> dict:
    faces = _load_faces(ep_id)
    removed_rows = [
        row
        for row in faces
        if int(row.get("track_id", -1)) == payload.track_id and int(row.get("frame_idx", -1)) == payload.frame_idx
    ]
    if not removed_rows:
        raise HTTPException(status_code=404, detail="Face frame not found")
    faces = [row for row in faces if row not in removed_rows]
    faces_path = _write_faces(ep_id, faces)
    if payload.delete_assets:
        frames_root = get_path(ep_id, "frames_root")
        for row in removed_rows:
            thumb_rel = row.get("thumb_rel_path")
            if isinstance(thumb_rel, str):
                thumb_file = _thumbs_root(ep_id) / thumb_rel
                try:
                    thumb_file.unlink()
                except FileNotFoundError:
                    pass
            crop_rel = row.get("crop_rel_path")
            if isinstance(crop_rel, str):
                crop_file = frames_root / crop_rel
                try:
                    crop_file.unlink()
                except FileNotFoundError:
                    pass
    _recount_track_faces(ep_id)
    identities = _load_identities(ep_id)
    _update_identity_stats(ep_id, identities)
    identities_path = _write_identities(ep_id, identities)
    _sync_manifests(ep_id, faces_path, identities_path)
    return {
        "track_id": payload.track_id,
        "frame_idx": payload.frame_idx,
        "removed": len(removed_rows),
        "remaining": len(faces),
    }
