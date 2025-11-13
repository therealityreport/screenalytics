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

DIAG = os.getenv("DIAG_LOG", "0") == "1"


def _diag(tag: str, **kw) -> None:
    """Diagnostic logger enabled via DIAG_LOG=1."""
    if DIAG:
        LOGGER.info("[DIAG:%s] %s", tag, json.dumps(kw, ensure_ascii=False))


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


def _crops_root(ep_id: str) -> Path:
    return get_path(ep_id, "frames_root") / "crops"


def _resolve_crop_url(ep_id: str, rel_path: str | None, s3_key: str | None) -> str | None:
    if s3_key:
        url = STORAGE.presign_get(s3_key)
        if url:
            return url
    if not rel_path:
        return None
    normalized = rel_path.strip()
    if not normalized:
        return None
    frames_root = get_path(ep_id, "frames_root")
    local = frames_root / normalized
    if local.exists():
        return str(local)
    rel_parts = Path(normalized)
    if rel_parts.parts and rel_parts.parts[0] == "crops":
        rel_parts = Path(*rel_parts.parts[1:])
    fallback_root = Path(os.environ.get("SCREENALYTICS_CROPS_FALLBACK_ROOT", "data/crops")).expanduser()
    legacy = fallback_root / ep_id / "tracks" / rel_parts
    if legacy.exists():
        return str(legacy)
    return None


def _resolve_face_media_url(ep_id: str, row: Dict[str, Any] | None) -> str | None:
    if not row:
        return None
    media = _resolve_crop_url(ep_id, row.get("crop_rel_path"), row.get("crop_s3_key"))
    if media:
        return media
    return _resolve_thumb_url(ep_id, row.get("thumb_rel_path"), row.get("thumb_s3_key"))


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
                # Assets cleanup should be idempotent; ignore missing thumbs.
                pass
        crop_rel = row.get("crop_rel_path")
        if isinstance(crop_rel, str):
            crop_file = frames_root / crop_rel
            try:
                crop_file.unlink()
            except FileNotFoundError:
                # Crops may have been purged already by another worker.
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


def _first_face_lookup(ep_id: str) -> Dict[int, Dict[str, Any]]:
    """Return the best candidate face row for each track.

    We prefer non-skipped faces when available, but fall back to skipped rows so
    that every track can expose a preview thumbnail in the UI."""

    faces = _load_faces(ep_id, include_skipped=True)
    lookup: Dict[int, Dict[str, Any]] = {}
    scores: Dict[int, tuple[int, int]] = {}
    for row in faces:
        try:
            tid = int(row.get("track_id", -1))
            frame_idx = int(row.get("frame_idx", 10**9))
        except (TypeError, ValueError):
            continue
        skip_flag = 1 if row.get("skip") else 0
        candidate_score = (skip_flag, frame_idx)
        best_score = scores.get(tid)
        if best_score is None or candidate_score < best_score:
            lookup[tid] = row
            scores[tid] = candidate_score
    return lookup


def _require_episode_context(ep_id: str):
    try:
        ctx = episode_context_from_id(ep_id)
    except ValueError as exc:  # pragma: no cover - invalid ids rejected upstream
        raise HTTPException(status_code=400, detail="Invalid episode id") from exc
    return ctx, artifact_prefixes(ctx)


_FRAME_FILE_RE = re.compile(r"frame_(\d{6})\.jpg$", re.IGNORECASE)


def _parse_frame_idx_from_name(name: str) -> int | None:
    match = _FRAME_FILE_RE.search(name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _track_crop_candidates(ep_id: str, track_id: int) -> List[Path]:
    track_component = f"track_{track_id:04d}"
    frames_root = get_path(ep_id, "frames_root")
    primary = frames_root / "crops" / track_component
    fallback_root = Path(os.environ.get("SCREENALYTICS_CROPS_FALLBACK_ROOT", "data/crops")).expanduser()
    legacy = fallback_root / ep_id / "tracks" / track_component
    candidates: List[Path] = []
    for path in (primary, legacy):
        if path not in candidates:
            candidates.append(path)
    return candidates


def _load_crop_index(path: Path) -> List[Dict[str, Any]]:
    index_path = path / "index.json"
    if not index_path.exists():
        return []
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Failed to parse crop index at %s", index_path)
        return []
    if not isinstance(data, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        frame_idx = entry.get("frame_idx")
        if not isinstance(key, str):
            continue
        try:
            frame_val = int(frame_idx)
        except (TypeError, ValueError):
            frame_val = _parse_frame_idx_from_name(Path(key).name)
        if frame_val is None:
            continue
        normalized.append({"key": key, "frame_idx": frame_val, "ts": entry.get("ts")})
    return normalized


def _discover_crop_entries(ep_id: str, track_id: int) -> List[Dict[str, Any]]:
    track_component = f"track_{track_id:04d}"
    for root in _track_crop_candidates(ep_id, track_id):
        if not root.exists():
            continue
        entries = _load_crop_index(root)
        if not entries:
            for jpeg in sorted(root.glob("frame_*.jpg")):
                idx = _parse_frame_idx_from_name(jpeg.name)
                if idx is None:
                    continue
                entries.append({"key": f"{track_component}/{jpeg.name}", "frame_idx": idx, "ts": None})
        if not entries:
            continue
        normalized: Dict[int, Dict[str, Any]] = {}
        for entry in entries:
            filename = Path(entry["key"]).name
            rel_path = Path("crops") / track_component / filename
            abs_path = root / filename
            normalized[int(entry["frame_idx"])] = {
                "frame_idx": int(entry["frame_idx"]),
                "ts": entry.get("ts"),
                "rel_path": rel_path.as_posix(),
                "abs_path": abs_path if abs_path.exists() else None,
            }
        if normalized:
            return [normalized[idx] for idx in sorted(normalized.keys())]
    return []



def _list_track_frame_media(ep_id: str, track_id: int, sample: int, page: int, page_size: int) -> Dict[str, Any]:
    sample = max(1, sample)
    page = max(1, page)
    page_size = max(1, min(page_size, TRACK_LIST_MAX_LIMIT))
    face_rows = _track_face_rows(ep_id, track_id)
    crops = _discover_crop_entries(ep_id, track_id)
    ctx, prefixes = _require_episode_context(ep_id)
    crops_prefix = (prefixes or {}).get("crops") if prefixes else None

    # Load track centroid for similarity computation
    try:
        from apps.api.services.track_reps import load_track_reps
        import numpy as np
        track_reps = load_track_reps(ep_id)
        track_rep = track_reps.get(f"track_{track_id:04d}", {})
        track_centroid = np.array(track_rep.get("embed", []), dtype=np.float32) if track_rep else None
    except Exception:
        track_centroid = None

    if not crops and not face_rows:
        return {
            "items": [],
            "total": 0,
            "total_frames": 0,
            "page": page,
            "page_size": page_size,
            "sample": sample,
        }
    if not crops:
        frame_indices = sorted(face_rows.keys())
        sampled_indices = frame_indices[::sample]
        total_items = len(sampled_indices)
        start = (page - 1) * page_size
        end = start + page_size
        page_indices = sampled_indices[start:end] if start < total_items else []
        items: List[Dict[str, Any]] = []
        for idx in page_indices:
            meta = face_rows.get(idx, {})
            media_url = _resolve_face_media_url(ep_id, meta)
            fallback = _resolve_thumb_url(ep_id, meta.get("thumb_rel_path"), meta.get("thumb_s3_key"))
            url = media_url or fallback

            # Compute similarity to track centroid
            similarity = None
            if track_centroid is not None and track_centroid.size > 0:
                embedding = meta.get("embedding")
                if embedding:
                    import numpy as np
                    frame_embed = np.array(embedding, dtype=np.float32)
                    if frame_embed.size == track_centroid.size:
                        similarity = float(np.dot(frame_embed, track_centroid))

            items.append(
                {
                    "track_id": track_id,
                    "frame_idx": idx,
                    "ts": meta.get("ts"),
                    "media_url": url,
                    "thumbnail_url": url,
                    "crop_rel_path": meta.get("crop_rel_path"),
                    "crop_s3_key": meta.get("crop_s3_key"),
                    "w": meta.get("crop_width") or meta.get("width"),
                    "h": meta.get("crop_height") or meta.get("height"),
                    "skip": meta.get("skip"),
                    "face_id": meta.get("face_id"),
                    "similarity": similarity,
                }
            )
        return {
            "items": items,
            "total": total_items,
            "total_frames": len(frame_indices),
            "returned": len(items),
            "page": page,
            "page_size": page_size,
            "sample": sample,
        }

    sampled_entries = crops[::sample]
    total_items = len(sampled_entries)
    start = (page - 1) * page_size
    end = start + page_size
    page_entries = sampled_entries[start:end] if start < total_items else []
    items: List[Dict[str, Any]] = []
    for entry in page_entries:
        frame_idx = entry["frame_idx"]
        meta = face_rows.get(frame_idx, {})
        local_path = entry.get("abs_path")
        local_url = str(local_path) if isinstance(local_path, Path) and local_path.exists() else None
        rel_path = entry.get("rel_path")
        s3_key = None
        if crops_prefix and rel_path:
            suffix = rel_path.split("crops/", 1)[-1]
            s3_key = f"{crops_prefix}{suffix}"
        media_url = local_url or _resolve_crop_url(ep_id, rel_path, s3_key if not local_url else None)
        fallback = _resolve_thumb_url(ep_id, meta.get("thumb_rel_path"), meta.get("thumb_s3_key"))
        url = media_url or fallback

        # Compute similarity to track centroid
        similarity = None
        if track_centroid is not None and track_centroid.size > 0:
            embedding = meta.get("embedding") if meta else None
            if embedding:
                import numpy as np
                frame_embed = np.array(embedding, dtype=np.float32)
                if frame_embed.size == track_centroid.size:
                    similarity = float(np.dot(frame_embed, track_centroid))

        items.append(
            {
                "track_id": track_id,
                "frame_idx": frame_idx,
                "ts": meta.get("ts") if meta else entry.get("ts"),
                "media_url": url,
                "thumbnail_url": url,
                "crop_rel_path": rel_path,
                "crop_s3_key": s3_key,
                "w": meta.get("crop_width") or meta.get("width"),
                "h": meta.get("crop_height") or meta.get("height"),
                "skip": meta.get("skip"),
                "face_id": meta.get("face_id"),
                "similarity": similarity,
            }
        )
    return {
        "items": items,
        "total": total_items,
        "total_frames": len(crops),
        "returned": len(items),
        "page": page,
        "page_size": page_size,
        "sample": sample,
    }


def _count_track_crops(ctx, track_id: int) -> int:
    total = 0
    cursor: str | None = None
    while True:
        payload = STORAGE.list_track_crops(ctx, track_id, sample=1, max_keys=500, start_after=cursor)
        items = payload.get("items", []) if isinstance(payload, dict) else []
        total += len(items)
        cursor = payload.get("next_start_after") if isinstance(payload, dict) else None
        if not cursor:
            break
    return total


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


class S3Show(BaseModel):
    show: str
    episode_count: int


class S3ShowsResponse(BaseModel):
    shows: List[S3Show]
    count: int


class S3EpisodeForShow(BaseModel):
    ep_id: str
    season: int
    episode: int
    key: str
    exists_in_store: bool


class S3EpisodesForShowResponse(BaseModel):
    show: str
    episodes: List[S3EpisodeForShow]
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


@router.get("/episodes/s3_shows", response_model=S3ShowsResponse, tags=["episodes"])
def list_s3_shows() -> S3ShowsResponse:
    """List all shows available in S3 with episode counts."""
    raw_items = STORAGE.list_episode_videos_s3(limit=10000)

    # Group by show
    show_episodes: Dict[str, int] = {}
    for obj in raw_items:
        show = obj.get("show")
        if show and isinstance(show, str):
            show_episodes[show] = show_episodes.get(show, 0) + 1

    # Convert to sorted list
    shows = [
        S3Show(show=show, episode_count=count)
        for show, count in sorted(show_episodes.items())
    ]
    return S3ShowsResponse(shows=shows, count=len(shows))


@router.get("/episodes/s3_shows/{show}/episodes", response_model=S3EpisodesForShowResponse, tags=["episodes"])
def list_s3_episodes_for_show(show: str) -> S3EpisodesForShowResponse:
    """List all episodes for a specific show from S3."""
    raw_items = STORAGE.list_episode_videos_s3(limit=10000)

    # Filter by show and collect episodes
    episodes: List[S3EpisodeForShow] = []
    for obj in raw_items:
        obj_show = obj.get("show")
        if obj_show and isinstance(obj_show, str) and obj_show.lower() == show.lower():
            ep_id = obj.get("ep_id")
            season = obj.get("season")
            episode = obj.get("episode")
            if ep_id and isinstance(season, int) and isinstance(episode, int):
                episodes.append(
                    S3EpisodeForShow(
                        ep_id=str(ep_id),
                        season=season,
                        episode=episode,
                        key=str(obj.get("key", "")),
                        exists_in_store=EPISODE_STORE.exists(str(ep_id)),
                    )
                )

    # Sort by season and episode
    episodes.sort(key=lambda x: (x.season, x.episode))
    return S3EpisodesForShowResponse(show=show, episodes=episodes, count=len(episodes))


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
    first_faces = _first_face_lookup(ep_id)
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
        preview_url = None
        if track_ids:
            preview_url = _resolve_face_media_url(ep_id, first_faces.get(track_ids[0]))
        if not preview_url:
            preview_url = _resolve_thumb_url(
                ep_id,
                identity.get("rep_thumb_rel_path"),
                identity.get("rep_thumb_s3_key"),
            )
        identities.append(
            {
                "identity_id": identity.get("identity_id"),
                "label": identity.get("label"),
                "name": identity.get("name"),
                "track_ids": track_ids,
                "faces": faces_total,
                "rep_thumbnail_url": preview_url,
                "rep_media_url": preview_url,
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
                    # Ignore duplicates when multiple workers enqueue roster seeds.
                    pass
    return {"identities": identities, "stats": payload.get("stats", {})}


@router.get("/episodes/{ep_id}/cluster_tracks")
def list_cluster_tracks(
    ep_id: str,
    limit_per_cluster: int | None = Query(None, ge=1, description="Optional max tracks per cluster"),
) -> dict:
    try:
        payload = identity_service.cluster_track_summary(ep_id, limit_per_cluster=limit_per_cluster)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    first_faces = _first_face_lookup(ep_id)
    for cluster in payload.get("clusters", []):
        for track in cluster.get("tracks", []) or []:
            tid = track.get("track_id")
            try:
                tid_int = int(tid)
            except (TypeError, ValueError):
                continue
            media_url = _resolve_face_media_url(ep_id, first_faces.get(tid_int))
            if media_url:
                track["rep_media_url"] = media_url
                track["rep_thumb_url"] = media_url
    return payload


@router.get("/episodes/{ep_id}/clusters/{cluster_id}/track_reps")
def get_cluster_track_reps(ep_id: str, cluster_id: str) -> dict:
    """Get representative frames with similarity scores for all tracks in a cluster."""
    try:
        from apps.api.services.track_reps import build_cluster_track_reps, load_track_reps, load_cluster_centroids

        track_reps = load_track_reps(ep_id)
        cluster_centroids = load_cluster_centroids(ep_id)

        result = build_cluster_track_reps(ep_id, cluster_id, track_reps, cluster_centroids)

        # Resolve crop URLs
        for track in result.get("tracks", []):
            crop_key = track.get("crop_key")
            if crop_key:
                # Use existing _resolve_crop_url helper
                url = _resolve_crop_url(ep_id, crop_key, None)
                track["crop_url"] = url

        return result
    except Exception as exc:
        LOGGER.error(f"Failed to load track reps for cluster {cluster_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to load track representatives: {exc}") from exc


@router.get("/episodes/{ep_id}/people/{person_id}/clusters_summary")
def get_person_clusters_summary(ep_id: str, person_id: str) -> dict:
    """Get clusters summary with track representatives for a person in an episode."""
    try:
        from apps.api.services.track_reps import build_cluster_track_reps, load_track_reps, load_cluster_centroids

        # Parse episode to get show
        ep_ctx = episode_context_from_id(ep_id)
        show_slug = ep_ctx.show_slug.upper()

        # Load people data
        from apps.api.services.people import PeopleService
        people_service = PeopleService()
        person = people_service.get_person(show_slug, person_id)

        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        # Filter cluster IDs for this episode
        cluster_ids = person.get("cluster_ids", []) if isinstance(person, dict) else []
        if not isinstance(cluster_ids, list):
            LOGGER.warning(f"cluster_ids is not a list: {type(cluster_ids)}, value: {cluster_ids}")
            cluster_ids = []

        episode_clusters = [
            cid.split(":", 1)[1] if ":" in cid else cid
            for cid in cluster_ids
            if isinstance(cid, str) and cid.startswith(f"{ep_id}:")
        ]

        if not episode_clusters:
            return {
                "person_id": person_id,
                "clusters": [],
                "total_clusters": 0,
                "total_tracks": 0,
            }

        # Load track reps and centroids once
        track_reps = load_track_reps(ep_id)
        cluster_centroids = load_cluster_centroids(ep_id)

        # Load identities for face counts
        identities_data = _load_identities(ep_id)
        LOGGER.info(f"identities_data type: {type(identities_data)}, keys: {list(identities_data.keys()) if isinstance(identities_data, dict) else 'not a dict'}")
        identities_list = identities_data.get("identities", []) if isinstance(identities_data, dict) else []
        LOGGER.info(f"identities_list type: {type(identities_list)}, length: {len(identities_list) if isinstance(identities_list, list) else 'not a list'}")
        identity_index = {}
        for ident in identities_list:
            if isinstance(ident, dict) and "identity_id" in ident:
                identity_index[ident["identity_id"]] = ident
            else:
                LOGGER.warning(f"Skipping malformed identity: {type(ident)}")

        clusters_output = []
        total_tracks = 0

        for cluster_id in episode_clusters:
            LOGGER.info(f"Processing cluster {cluster_id}")
            cluster_data = build_cluster_track_reps(ep_id, cluster_id, track_reps, cluster_centroids)
            LOGGER.info(f"cluster_data type: {type(cluster_data)}, keys: {list(cluster_data.keys()) if isinstance(cluster_data, dict) else 'not a dict'}")

            # Resolve crop URLs
            tracks_list = cluster_data.get("tracks", []) if isinstance(cluster_data, dict) else []
            for track in tracks_list:
                if isinstance(track, dict):
                    crop_key = track.get("crop_key")
                    if crop_key:
                        url = _resolve_crop_url(ep_id, crop_key, None)
                        track["crop_url"] = url

            # Get face count from identities
            identity = identity_index.get(cluster_id, {})
            LOGGER.info(f"identity for {cluster_id}: type={type(identity)}")
            if isinstance(identity, dict):
                faces_count = identity.get("size") or 0
            else:
                LOGGER.error(f"identity is not a dict for cluster {cluster_id}: {type(identity)}")
                faces_count = 0

            clusters_output.append({
                "cluster_id": cluster_id,
                "tracks": len(cluster_data.get("tracks", [])),
                "faces": faces_count,
                "cohesion": cluster_data.get("cohesion"),
                "centroid": cluster_data.get("cluster_centroid"),  # Include centroid vector
                "track_reps": cluster_data.get("tracks", []),
            })
            total_tracks += len(cluster_data.get("tracks", []))

        return {
            "person_id": person_id,
            "clusters": clusters_output,
            "total_clusters": len(clusters_output),
            "total_tracks": total_tracks,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.error(f"Failed to load clusters summary for person {person_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to load clusters summary: {exc}") from exc


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
        media_url = _resolve_face_media_url(ep_id, row)
        items.append(
            {
                "face_id": row.get("face_id"),
                "track_id": tid,
                "frame_idx": row.get("frame_idx"),
                "ts": row.get("ts"),
                "thumbnail_url": media_url,
                "media_url": media_url,
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
    first_faces = _first_face_lookup(ep_id)
    tracks_payload: List[Dict[str, Any]] = []
    track_ids: List[int] = []
    for raw_tid in identity.get("track_ids", []) or []:
        try:
            tid = int(raw_tid)
        except (TypeError, ValueError):
            continue
        track_ids.append(tid)
        track_row = track_lookup.get(tid, {})
        face_row = first_faces.get(tid)
        media_url = _resolve_face_media_url(ep_id, face_row)
        if not media_url:
            media_url = _resolve_thumb_url(ep_id, track_row.get("thumb_rel_path"), track_row.get("thumb_s3_key"))
        tracks_payload.append(
            {
                "track_id": tid,
                "faces_count": track_row.get("faces_count", 0),
                "thumbnail_url": media_url,
                "media_url": media_url,
            }
        )
    return {
        "identity": {
            "identity_id": identity_id,
            "label": identity.get("label"),
            "name": identity.get("name"),
            "track_ids": track_ids,
            "rep_thumbnail_url": _resolve_face_media_url(
                ep_id,
                first_faces.get(track_ids[0]) if track_ids else None,
            )
            or _resolve_thumb_url(ep_id, identity.get("rep_thumb_rel_path"), identity.get("rep_thumb_s3_key")),
            "rep_media_url": _resolve_face_media_url(
                ep_id,
                first_faces.get(track_ids[0]) if track_ids else None,
            )
            or _resolve_thumb_url(ep_id, identity.get("rep_thumb_rel_path"), identity.get("rep_thumb_s3_key")),
        },
        "tracks": tracks_payload,
    }


@router.get("/episodes/{ep_id}/tracks/{track_id}")
def track_detail(ep_id: str, track_id: int) -> dict:
    faces = [row for row in _load_faces(ep_id, include_skipped=False) if int(row.get("track_id", -1)) == track_id]
    frames = []
    for row in faces:
        media_url = _resolve_face_media_url(ep_id, row)
        frames.append(
            {
                "face_id": row.get("face_id"),
                "frame_idx": row.get("frame_idx"),
                "ts": row.get("ts"),
                "thumbnail_url": media_url,
                "media_url": media_url,
                "skip": row.get("skip"),
                "crop_rel_path": row.get("crop_rel_path"),
                "crop_s3_key": row.get("crop_s3_key"),
            }
        )
    track_row = next((row for row in _load_tracks(ep_id) if int(row.get("track_id", -1)) == track_id), None)
    preview_url = _resolve_face_media_url(ep_id, faces[0] if faces else None)
    if not preview_url:
        preview_url = _resolve_thumb_url(
            ep_id,
            (track_row or {}).get("thumb_rel_path"),
            (track_row or {}).get("thumb_s3_key"),
        )
    return {
        "track_id": track_id,
        "faces_count": len(frames),
        "thumbnail_url": preview_url,
        "media_url": preview_url,
        "frames": frames,
    }


@router.get("/episodes/{ep_id}/tracks/{track_id}/crops")
def list_track_crops(
    ep_id: str,
    track_id: int,
    sample: int = Query(1, ge=1, le=100, description="Return every Nth crop"),
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
            # Include quality metrics for sorting/filtering in UI
            item.setdefault("quality", meta.get("quality"))
            item.setdefault("conf", meta.get("conf"))

    if items:
        keys = [item.get("key") for item in items if item.get("key")]
        first_three = keys[:3]
        last_three = keys[-3:] if len(keys) > 3 else []
        _diag(
            "TILE_LIST",
            ep_id=ep_id,
            track_id=track_id,
            total_items=len(items),
            first_keys=first_three,
            last_keys=last_three,
            sample=sample,
            limit=limit,
        )

    return payload


@router.get("/episodes/{ep_id}/tracks/{track_id}/frames")
def list_track_frames(
    ep_id: str,
    track_id: int,
    sample: int = Query(1, ge=1, le=100, description="Return every Nth frame"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=TRACK_LIST_MAX_LIMIT),
) -> Dict[str, Any]:
    return _list_track_frame_media(ep_id, track_id, sample, page, page_size)


@router.get("/episodes/{ep_id}/tracks/{track_id}/integrity")
def track_integrity(ep_id: str, track_id: int) -> Dict[str, Any]:
    face_rows = _track_face_rows(ep_id, track_id)
    ctx, _ = _require_episode_context(ep_id)
    crops = _count_track_crops(ctx, track_id)
    faces_count = len(face_rows)
    return {
        "track_id": track_id,
        "faces_manifest": faces_count,
        "crops_files": crops,
        "ok": crops >= faces_count > 0,
    }


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


@router.post("/episodes/{ep_id}/tracks/{track_id}/name")
def assign_track_name(ep_id: str, track_id: int, body: IdentityNameRequest) -> dict:
    try:
        return identity_service.assign_track_name(ep_id, track_id, body.name, body.show)
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
                    # It's fine if the thumbnail file does not exist.
                    pass
            crop_rel = row.get("crop_rel_path")
            if isinstance(crop_rel, str):
                crop_file = frames_root / crop_rel
                try:
                    crop_file.unlink()
                except FileNotFoundError:
                    # It's fine if the crop file does not exist.
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
