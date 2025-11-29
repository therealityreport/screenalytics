from __future__ import annotations

import csv
import fcntl
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Sequence

from py_screenalytics.artifacts import get_path

from apps.api.services import roster as roster_service
from apps.api.services.storage import (
    StorageService,
    artifact_prefixes,
    episode_context_from_id,
)
from apps.shared.storage import s3_write_json, use_s3

STORAGE = StorageService()
LOGGER = logging.getLogger(__name__)


@contextmanager
def _episode_lock(ep_id: str) -> Generator[None, None, None]:
    """Acquire an exclusive lock for episode-level file operations.

    Uses fcntl advisory locking to prevent concurrent modifications
    to faces, tracks, and identities files for the same episode.
    """
    lock_path = _manifests_dir(ep_id) / ".episode.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = None
    try:
        lock_file = open(lock_path, "w")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()


def _manifests_dir(ep_id: str) -> Path:
    return get_path(ep_id, "detections").parent


def _faces_path(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "faces.jsonl"


def _identities_path(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "identities.json"


def _tracks_path(ep_id: str) -> Path:
    return get_path(ep_id, "tracks")


def _thumbs_root(ep_id: str) -> Path:
    return get_path(ep_id, "frames_root") / "thumbs"


def _crops_root(ep_id: str) -> Path:
    return get_path(ep_id, "frames_root") / "crops"


def _crop_url(ep_id: str, rel_path: str | None, s3_key: str | None) -> str | None:
    """Resolve a crop URL, preferring S3 with local fallback.

    Priority:
    1. Use provided s3_key if available
    2. Construct S3 key from rel_path if s3_key not provided
    3. Fall back to local file if S3 fails
    """
    constructed_s3_key = s3_key

    # If no S3 key provided, try to construct one from rel_path
    if not constructed_s3_key and rel_path:
        try:
            ep_ctx = episode_context_from_id(ep_id)
            prefixes = artifact_prefixes(ep_ctx)
            crops_prefix = prefixes.get("crops")
            if crops_prefix:
                crop_rel = rel_path
                if crop_rel.startswith("crops/"):
                    crop_rel = crop_rel[6:]
                constructed_s3_key = f"{crops_prefix}{crop_rel}"
        except (ValueError, KeyError):
            pass

    # Try S3 first
    if constructed_s3_key:
        url = STORAGE.presign_get(str(constructed_s3_key))
        if url:
            return url

    # Local fallback
    if rel_path:
        frames_root = get_path(ep_id, "frames_root")
        if rel_path.startswith("crops/"):
            local = frames_root / rel_path
        else:
            local = _crops_root(ep_id) / rel_path
        if local.exists():
            return str(local)

    return None


def _thumbnail_url(ep_id: str, rel_path: str | None, s3_key: str | None) -> str | None:
    """Resolve a thumbnail URL, preferring S3 with local fallback.

    Priority:
    1. Use provided s3_key if available
    2. Construct S3 key from rel_path if s3_key not provided
    3. Fall back to local file if S3 fails
    """
    constructed_s3_key = s3_key

    # If no S3 key provided, try to construct one from rel_path
    if not constructed_s3_key and rel_path:
        try:
            ep_ctx = episode_context_from_id(ep_id)
            prefixes = artifact_prefixes(ep_ctx)
            thumbs_prefix = prefixes.get("thumbs")
            if thumbs_prefix:
                constructed_s3_key = f"{thumbs_prefix}{rel_path}"
        except (ValueError, KeyError):
            pass

    # Try S3 first
    if constructed_s3_key:
        url = STORAGE.presign_get(str(constructed_s3_key))
        if url:
            return url

    # Local fallback - try crop path first if it looks like a crop
    if rel_path and (rel_path.startswith("crops/") or "crops/" in rel_path):
        frames_root = get_path(ep_id, "frames_root")
        crop_path = frames_root / rel_path
        if crop_path.exists():
            return str(crop_path)

    if rel_path:
        local = _thumbs_root(ep_id) / rel_path
        if local.exists():
            return str(local)

    return None


def _track_media_url(ep_id: str, track_row: Dict[str, Any] | None) -> str | None:
    """Resolve the best media URL for a track, prioritizing crops over thumbs.

    IMPORTANT: Crops are authoritative. best_crop_rel_path should be used first.
    """
    if not track_row:
        return None
    # Prioritize best crop - authoritative source for track preview
    crop = _crop_url(ep_id, track_row.get("best_crop_rel_path"), track_row.get("best_crop_s3_key"))
    if crop:
        return crop
    # Fall back to thumb only if crop is unavailable
    return _thumbnail_url(ep_id, track_row.get("thumb_rel_path"), track_row.get("thumb_s3_key"))


def _analytics_root(ep_id: str) -> Path:
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
    return data_root / "analytics" / ep_id


def _screentime_csv_path(ep_id: str) -> Path:
    return _analytics_root(ep_id) / "screentime.csv"


def _read_json_lines(path: Path) -> List[Dict[str, Any]]:
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


def _write_json_lines(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def load_faces(ep_id: str) -> List[Dict[str, Any]]:
    return _read_json_lines(_faces_path(ep_id))


def write_faces(ep_id: str, rows: List[Dict[str, Any]]) -> Path:
    return _write_json_lines(_faces_path(ep_id), rows)


def load_tracks(ep_id: str) -> List[Dict[str, Any]]:
    return _read_json_lines(_tracks_path(ep_id))


def write_tracks(ep_id: str, rows: List[Dict[str, Any]]) -> Path:
    return _write_json_lines(_tracks_path(ep_id), rows)


def load_identities(ep_id: str) -> Dict[str, Any]:
    path = _identities_path(ep_id)
    if not path.exists():
        return {"ep_id": ep_id, "identities": [], "stats": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ep_id": ep_id, "identities": [], "stats": {}}


def write_identities(ep_id: str, payload: Dict[str, Any]) -> Path:
    path = _identities_path(ep_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def update_identity_stats(ep_id: str, payload: Dict[str, Any]) -> None:
    faces_count = len(load_faces(ep_id))
    payload.setdefault("stats", {})
    payload["stats"]["faces"] = faces_count
    payload["stats"]["clusters"] = len(payload.get("identities", []))


def sync_manifests(ep_id: str, *paths: Path) -> None:
    try:
        ctx = episode_context_from_id(ep_id)
    except ValueError:
        return
    for path in paths:
        if path and path.exists():
            try:
                STORAGE.put_artifact(ctx, "manifests", path, path.name)
            except Exception:
                continue


def _identity_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    identities = payload.get("identities")
    if isinstance(identities, list):
        return identities
    if isinstance(payload, list):  # legacy format
        return payload  # type: ignore[return-value]
    return []


def _identity_name_lookup(payload: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for entry in _identity_rows(payload):
        key = entry.get("identity_id") or entry.get("id")
        name = entry.get("name")
        if not key or not name:
            continue
        mapping[str(key)] = str(name)
    return mapping


def _annotate_screentime_csv(ep_id: str, payload: Dict[str, Any]) -> None:
    csv_path = _screentime_csv_path(ep_id)
    if not csv_path.exists():
        return
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    except (OSError, csv.Error):
        return
    if not rows or not fieldnames:
        return
    key_field = "identity_id" if "identity_id" in fieldnames else None
    if key_field is None and "person_id" in fieldnames:
        key_field = "person_id"
    if key_field is None:
        return
    lookup = _identity_name_lookup(payload)
    if not lookup:
        return
    updated = False
    for row in rows:
        identifier = str(row.get(key_field, "") or "").strip()
        if not identifier:
            continue
        name = lookup.get(identifier)
        if not name or row.get("name") == name:
            continue
        row["name"] = name
        updated = True
    if not updated:
        return
    out_fields = list(fieldnames)
    if "name" not in out_fields:
        out_fields.append("name")
    try:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=out_fields)
            writer.writeheader()
            for row in rows:
                if "name" not in row:
                    row["name"] = ""
                writer.writerow(row)
    except (OSError, csv.Error) as exc:
        LOGGER.warning("Failed to update screentime.csv for %s: %s", ep_id, exc)


def _next_identity_id(entries: Sequence[Dict[str, Any]]) -> str:
    max_idx = 0
    for entry in entries:
        raw = entry.get("identity_id") or entry.get("id")
        if not raw:
            continue
        digits = "".join(ch for ch in str(raw) if ch.isdigit())
        try:
            idx = int(digits)
        except ValueError:
            continue
        max_idx = max(max_idx, idx)
    return f"id_{max_idx + 1:04d}"


def _track_dir_name(track_id: int) -> str:
    return f"track_{int(track_id):04d}"


def _crop_rel(track_id: int, frame_idx: int) -> str:
    return f"crops/{_track_dir_name(track_id)}/frame_{int(frame_idx):06d}.jpg"


def _thumb_rel(track_id: int, frame_idx: int) -> str:
    return f"{_track_dir_name(track_id)}/thumb_{int(frame_idx):06d}.jpg"


def _faces_by_track(rows: Sequence[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        try:
            track_id = int(row.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(track_id, []).append(row)
    for items in grouped.values():
        items.sort(key=lambda r: int(r.get("frame_idx", 0)))
    return grouped


def _next_track_id(tracks: Sequence[Dict[str, Any]]) -> int:
    max_id = 0
    for entry in tracks:
        try:
            max_id = max(max_id, int(entry.get("track_id", 0)))
        except (TypeError, ValueError):
            continue
    return max_id + 1 if max_id >= 0 else 1


def _track_lookup(tracks: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    for entry in tracks:
        try:
            track_id = int(entry.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        if track_id >= 0:
            lookup[track_id] = entry
    return lookup


def cluster_track_summary(
    ep_id: str,
    *,
    include: Iterable[str] | None = None,
    limit_per_cluster: int | None = None,
) -> Dict[str, Any]:
    include_set = {identity.lower() for identity in include} if include else None
    identities_payload = load_identities(ep_id)
    tracks = load_tracks(ep_id)
    faces = load_faces(ep_id)
    track_lookup = _track_lookup(tracks)

    # Load cluster centroids for cohesion data and track similarity computation
    cluster_cohesion: Dict[str, float] = {}
    cluster_centroids_data: Dict[str, Dict[str, Any]] = {}
    track_reps_data: Dict[str, Dict[str, Any]] = {}
    try:
        from apps.api.services.track_reps import load_cluster_centroids, load_track_reps
        import numpy as np

        centroids = load_cluster_centroids(ep_id)
        for cluster_id, cluster_data in centroids.items():
            if isinstance(cluster_data, dict):
                if cluster_data.get("cohesion") is not None:
                    cluster_cohesion[cluster_id] = cluster_data["cohesion"]
                if cluster_data.get("centroid") is not None:
                    cluster_centroids_data[cluster_id] = cluster_data

        # Load track reps for similarity computation
        track_reps_data = load_track_reps(ep_id)
    except Exception:
        pass  # Cohesion/similarity data not available
    face_counts: Dict[int, int] = {}
    for row in faces:
        try:
            tid = int(row.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        face_counts[tid] = face_counts.get(tid, 0) + 1

    # Helper to compute cosine similarity
    def _cosine_sim(a: Any, b: Any) -> float | None:
        try:
            import numpy as np
            a_vec = np.array(a, dtype=np.float32)
            b_vec = np.array(b, dtype=np.float32)
            norm_a = np.linalg.norm(a_vec) + 1e-12
            norm_b = np.linalg.norm(b_vec) + 1e-12
            return float(np.dot(a_vec / norm_a, b_vec / norm_b))
        except Exception:
            return None

    clusters: List[Dict[str, Any]] = []
    for identity in identities_payload.get("identities", []):
        identity_id = identity.get("identity_id")
        if not identity_id:
            continue
        if include_set and identity_id.lower() not in include_set:
            continue
        track_ids = identity.get("track_ids", []) or []

        # Get cluster centroid for similarity computation
        cluster_centroid = None
        cluster_data = cluster_centroids_data.get(identity_id)
        if cluster_data:
            cluster_centroid = cluster_data.get("centroid")

        tracks_payload: List[Dict[str, Any]] = []
        for raw_tid in track_ids:
            try:
                tid = int(raw_tid)
            except (TypeError, ValueError):
                continue
            track_row = track_lookup.get(tid)
            if not track_row:
                continue
            # Use _track_media_url to prioritize crops (authoritative) over thumbs
            thumb_url = _track_media_url(ep_id, track_row)

            # Get track rep data for similarity computation
            track_key = f"track_{tid:04d}"
            track_rep = track_reps_data.get(track_key)

            # Compute track similarity to cluster centroid
            track_similarity = None
            if cluster_centroid is not None and track_rep and track_rep.get("embed"):
                track_similarity = _cosine_sim(track_rep["embed"], cluster_centroid)
                if track_similarity is not None:
                    track_similarity = round(track_similarity, 3)

            # Get track internal consistency (how similar rep frame is to track centroid)
            internal_similarity = None
            if track_rep and track_rep.get("sim_to_centroid") is not None:
                internal_similarity = round(track_rep["sim_to_centroid"], 3)

            tracks_payload.append(
                {
                    "track_id": tid,
                    "faces": track_row.get("faces_count") or face_counts.get(tid) or 0,
                    "frames": track_row.get("frame_count"),
                    "rep_thumb_url": thumb_url,
                    "similarity": track_similarity,
                    "internal_similarity": internal_similarity,  # Track internal consistency
                }
            )
        if limit_per_cluster:
            limit = max(1, int(limit_per_cluster))
            tracks_payload = tracks_payload[:limit]
        clusters.append(
            {
                "identity_id": identity_id,
                "name": identity.get("name"),
                "label": identity.get("label"),
                "person_id": identity.get("person_id"),  # For filtering assigned vs unassigned
                "cohesion": cluster_cohesion.get(identity_id),  # Add cohesion from centroids
                "counts": {
                    "tracks": len(track_ids),
                    "faces": identity.get("faces") or sum(face_counts.get(int(tid), 0) for tid in track_ids),
                },
                "tracks": tracks_payload,
            }
        )
    return {"ep_id": ep_id, "clusters": clusters}


def _rebuild_track_entry(track_entry: Dict[str, Any], face_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not face_rows:
        return track_entry
    sorted_rows = sorted(face_rows, key=lambda row: int(row.get("frame_idx", 0)))
    track_entry["frame_count"] = len(sorted_rows)
    track_entry["faces_count"] = len(sorted_rows)
    ts_values = [row.get("ts") for row in sorted_rows if isinstance(row.get("ts"), (int, float))]
    if ts_values:
        track_entry["first_ts"] = float(min(ts_values))
        track_entry["last_ts"] = float(max(ts_values))
    sample_limit = min(len(sorted_rows), 10)
    bboxes = []
    for row in sorted_rows[:sample_limit]:
        entry = {
            "frame_idx": row.get("frame_idx"),
            "ts": row.get("ts"),
            "bbox_xyxy": row.get("bbox_xyxy"),
            "landmarks": row.get("landmarks"),
        }
        bboxes.append(entry)
    if bboxes:
        track_entry["bboxes_sampled"] = bboxes
    thumb_rel = sorted_rows[0].get("thumb_rel_path")
    thumb_s3 = sorted_rows[0].get("thumb_s3_key")
    if thumb_rel:
        track_entry["thumb_rel_path"] = thumb_rel
    if thumb_s3:
        track_entry["thumb_s3_key"] = thumb_s3
    crop_rel = sorted_rows[0].get("crop_rel_path")
    crop_s3 = sorted_rows[0].get("crop_s3_key")
    if crop_rel:
        track_entry.setdefault("best_crop_rel_path", crop_rel)
    if crop_s3:
        track_entry.setdefault("best_crop_s3_key", crop_s3)
    return track_entry


def _write_track_index(ep_id: str, track_id: int, face_rows: Sequence[Dict[str, Any]], ctx) -> None:
    frames_root = get_path(ep_id, "frames_root")
    crops_dir = frames_root / "crops" / _track_dir_name(track_id)
    index_path = crops_dir / "index.json"
    if not face_rows:
        try:
            index_path.unlink()
        except FileNotFoundError:
            # Track index may already be removed by another cleanup worker.
            pass
        return
    crops_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "key": f"{_track_dir_name(track_id)}/frame_{int(row.get('frame_idx', 0)):06d}.jpg",
            "frame_idx": int(row.get("frame_idx", 0)),
            "ts": row.get("ts"),
        }
        for row in sorted(face_rows, key=lambda r: int(r.get("frame_idx", 0)))
    ]
    index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    try:
        STORAGE.put_artifact(ctx, "crops", index_path, f"{_track_dir_name(track_id)}/index.json")
    except Exception:
        # Remote sync failures are non-fatal; artifacts stay locally.
        pass


def rename_identity(ep_id: str, identity_id: str, label: str | None) -> Dict[str, Any]:
    payload = load_identities(ep_id)
    identity = next(
        (item for item in payload.get("identities", []) if item.get("identity_id") == identity_id),
        None,
    )
    if not identity:
        raise ValueError("identity_not_found")
    normalized = (label or "").strip()
    identity["label"] = normalized or None
    update_identity_stats(ep_id, payload)
    identities_path = write_identities(ep_id, payload)
    sync_manifests(ep_id, identities_path)
    return identity


def merge_identities(ep_id: str, source_id: str, target_id: str) -> Dict[str, Any]:
    payload = load_identities(ep_id)
    identities = payload.get("identities", [])
    source = next((item for item in identities if item.get("identity_id") == source_id), None)
    target = next((item for item in identities if item.get("identity_id") == target_id), None)
    if not source or not target:
        raise ValueError("identity_not_found")
    # Consistently coerce all track_ids to int to avoid type confusion
    target_tids = {int(tid) for tid in (target.get("track_ids", []) or [])}
    source_tids = {int(tid) for tid in (source.get("track_ids", []) or [])}
    merged = target_tids | source_tids
    target["track_ids"] = sorted(merged)
    payload["identities"] = [item for item in identities if item.get("identity_id") != source_id]
    update_identity_stats(ep_id, payload)
    identities_path = write_identities(ep_id, payload)
    sync_manifests(ep_id, identities_path)
    return target


def move_track(ep_id: str, track_id: int, target_identity_id: str | None) -> Dict[str, Any]:
    payload = load_identities(ep_id)
    identities = payload.get("identities", [])
    source_identity = None
    target_identity = None
    for identity in identities:
        # Coerce track_ids to int for consistent comparison (handles string/int mismatch)
        track_ids = {int(tid) for tid in (identity.get("track_ids", []) or [])}
        if track_id in track_ids:
            source_identity = identity
        if target_identity_id and identity.get("identity_id") == target_identity_id:
            target_identity = identity
    if target_identity_id and target_identity is None:
        raise ValueError("target_not_found")
    # Remove from source identity (coerce to int for comparison)
    if source_identity:
        source_track_ids = {int(tid) for tid in source_identity.get("track_ids", [])}
        source_identity["track_ids"] = sorted(tid for tid in source_track_ids if tid != track_id)
    if target_identity is not None:
        target_identity.setdefault("track_ids", [])
        # Coerce existing track_ids to int and add new track
        target_track_ids = {int(tid) for tid in target_identity["track_ids"]}
        target_track_ids.add(track_id)
        target_identity["track_ids"] = sorted(target_track_ids)
    update_identity_stats(ep_id, payload)
    identities_path = write_identities(ep_id, payload)
    sync_manifests(ep_id, identities_path)

    # Automatically clean up any empty clusters (source may now be empty)
    cleanup_result = cleanup_empty_clusters(ep_id)

    return {
        "identity_id": target_identity_id,
        "track_ids": target_identity["track_ids"] if target_identity else [],
        "empty_clusters_removed": len(cleanup_result.get("removed_clusters", [])),
    }


def drop_track(ep_id: str, track_id: int) -> Dict[str, Any]:
    tracks = load_tracks(ep_id)
    kept_tracks = [row for row in tracks if int(row.get("track_id", -1)) != track_id]
    if len(kept_tracks) == len(tracks):
        raise ValueError("track_not_found")
    tracks_path = write_tracks(ep_id, kept_tracks)
    identities = load_identities(ep_id)
    for identity in identities.get("identities", []):
        # Coerce to int for consistent comparison
        identity["track_ids"] = sorted(
            int(tid) for tid in identity.get("track_ids", []) if int(tid) != track_id
        )
    update_identity_stats(ep_id, identities)
    identities_path = write_identities(ep_id, identities)
    sync_manifests(ep_id, tracks_path, identities_path)

    # Automatically clean up any empty clusters
    cleanup_result = cleanup_empty_clusters(ep_id)
    return {
        "track_id": track_id,
        "remaining_tracks": len(kept_tracks),
        "empty_clusters_removed": len(cleanup_result.get("removed_clusters", [])),
    }


def drop_frame(ep_id: str, track_id: int, frame_idx: int, delete_assets: bool = False) -> Dict[str, Any]:
    faces = load_faces(ep_id)
    removed = [
        row for row in faces if int(row.get("track_id", -1)) == track_id and int(row.get("frame_idx", -1)) == frame_idx
    ]
    if not removed:
        raise ValueError("frame_not_found")
    faces = [row for row in faces if row not in removed]
    faces_path = write_faces(ep_id, faces)
    thumbs_root = get_path(ep_id, "frames_root") / "thumbs"
    crops_root = get_path(ep_id, "frames_root")
    if delete_assets:
        for row in removed:
            thumb_rel = row.get("thumb_rel_path")
            if isinstance(thumb_rel, str):
                thumb_abs = thumbs_root / thumb_rel
                try:
                    thumb_abs.unlink()
                except FileNotFoundError:
                    # Thumbs are optional outputs; ignore if already deleted.
                    pass
            crop_rel = row.get("crop_rel_path")
            if isinstance(crop_rel, str):
                crop_abs = crops_root / crop_rel
                try:
                    crop_abs.unlink()
                except FileNotFoundError:
                    # Crops may already be removed by previous delete requests.
                    pass
    identities = load_identities(ep_id)
    update_identity_stats(ep_id, identities)
    identities_path = write_identities(ep_id, identities)
    sync_manifests(ep_id, faces_path, identities_path)
    return {"track_id": track_id, "frame_idx": frame_idx, "removed": len(removed)}


def cleanup_empty_clusters(ep_id: str, show_id: str | None = None) -> Dict[str, Any]:
    """Remove clusters with no tracks and update people.json accordingly.

    This should be called after any operation that might leave clusters empty:
    - drop_track()
    - move_track()
    - Merge operations

    Args:
        ep_id: Episode identifier
        show_id: Show identifier (extracted from ep_id if not provided)

    Returns:
        {
            "removed_clusters": List of identity_ids that were removed,
            "people_updated": List of person_ids that had clusters removed,
            "identities_before": Count before cleanup,
            "identities_after": Count after cleanup,
        }
    """
    from apps.api.services.people import PeopleService

    identities = load_identities(ep_id)
    all_identities = identities.get("identities", [])
    identities_before = len(all_identities)

    # Find empty clusters (no track_ids)
    empty_cluster_ids = []
    kept_identities = []
    for identity in all_identities:
        track_ids = identity.get("track_ids", []) or []
        if not track_ids:
            empty_cluster_ids.append(identity.get("identity_id"))
            LOGGER.info(
                "Removing empty cluster %s from episode %s",
                identity.get("identity_id"),
                ep_id,
            )
        else:
            kept_identities.append(identity)

    if not empty_cluster_ids:
        return {
            "removed_clusters": [],
            "people_updated": [],
            "identities_before": identities_before,
            "identities_after": identities_before,
        }

    # Update identities.json
    identities["identities"] = kept_identities
    update_identity_stats(ep_id, identities)
    identities_path = write_identities(ep_id, identities)
    sync_manifests(ep_id, identities_path)

    # Remove from people.json
    people_updated = []
    if show_id is None:
        try:
            ctx = episode_context_from_id(ep_id)
            show_id = ctx.show_slug
        except Exception:
            LOGGER.warning("Could not extract show_id from ep_id %s", ep_id)

    if show_id:
        people_service = PeopleService()
        for cluster_id in empty_cluster_ids:
            # Full cluster ID format: ep_id:identity_id
            full_cluster_id = f"{ep_id}:{cluster_id}"
            modified = people_service.remove_cluster_from_all_people(show_id, full_cluster_id)
            people_updated.extend(modified)

    LOGGER.info(
        "Cleaned up %d empty clusters from episode %s, updated %d people",
        len(empty_cluster_ids),
        ep_id,
        len(set(people_updated)),
    )

    return {
        "removed_clusters": empty_cluster_ids,
        "people_updated": list(set(people_updated)),
        "identities_before": identities_before,
        "identities_after": len(kept_identities),
    }


def _persist_identity_name(
    ep_id: str,
    payload: Dict[str, Any],
    identity_id: str,
    trimmed_name: str,
    show: str | None,
    cast_id: str | None = None,
) -> Dict[str, Any]:
    update_identity_stats(ep_id, payload)
    identities_path = write_identities(ep_id, payload)
    sync_manifests(ep_id, identities_path)
    if use_s3():
        try:
            s3_write_json(f"artifacts/manifests/{ep_id}/identities.json", payload)
        except Exception as exc:  # pragma: no cover - best effort sync
            LOGGER.warning("Failed to mirror identities for %s: %s", ep_id, exc)
    _annotate_screentime_csv(ep_id, payload)
    try:
        ctx = episode_context_from_id(ep_id)
        show_slug = show or ctx.show_slug
    except ValueError:
        show_slug = show or ""
    if show_slug:
        try:
            roster_service.add_if_missing(show_slug, trimmed_name)
        except ValueError:
            # Duplicate roster seeds are normal when multiple queues run.
            pass

        # Create or update People record for immediate visibility in People view
        try:
            from apps.api.services.people import PeopleService

            people_service = PeopleService()

            # Build cluster_id with episode prefix
            cluster_id_with_prefix = f"{ep_id}:{identity_id}"

            existing_person = None

            # PRIORITY 1: If cast_id provided, look for person with that cast_id first
            if cast_id:
                existing_person = next(
                    (p for p in people_service.list_people(show_slug) if p.get("cast_id") == cast_id),
                    None,
                )
                if existing_person:
                    LOGGER.info(f"Found existing person by cast_id {cast_id}: {existing_person['person_id']}")

            # PRIORITY 2: If no cast_id match, use alias-aware lookup by name
            if not existing_person:
                existing_person = people_service.find_person_by_name_or_alias(show_slug, trimmed_name)

            # PRIORITY 3: Fallback to case-insensitive primary-name match
            if not existing_person:
                existing_person = next(
                    (
                        person
                        for person in people_service.list_people(show_slug)
                        if people_service.normalize_name(person.get("name"))
                        == people_service.normalize_name(trimmed_name)
                    ),
                    None,
                )

            if existing_person:
                # Add cluster to existing person (if not already present)
                person_id = existing_person["person_id"]
                cluster_ids = existing_person.get("cluster_ids", [])
                if cluster_id_with_prefix not in cluster_ids:
                    people_service.add_cluster_to_person(
                        show_slug,
                        person_id,
                        cluster_id_with_prefix,
                        update_prototype=False,  # Don't update prototype from manual naming
                    )
                    LOGGER.info(
                        f"Added cluster {identity_id} to existing person {person_id} ({existing_person.get('name')})"
                    )

                # Add the input name as an alias if it's different from the primary name
                primary_name = existing_person.get("name") or ""
                if people_service.normalize_name(trimmed_name) != people_service.normalize_name(primary_name):
                    people_service.add_alias_to_person(show_slug, person_id, trimmed_name)
                    LOGGER.info(f"Added alias '{trimmed_name}' to person {person_id}")

                # Ensure cast_id is set if provided
                if cast_id and existing_person.get("cast_id") != cast_id:
                    people_service.update_person(show_slug, person_id, cast_id=cast_id)
                    LOGGER.info(f"Updated person {person_id} with cast_id {cast_id}")
            else:
                # Create new person with cast_id
                person = people_service.create_person(
                    show_slug,
                    name=trimmed_name,
                    cluster_ids=[cluster_id_with_prefix],
                    aliases=[],
                    cast_id=cast_id,  # Include cast_id when creating person
                )
                LOGGER.info(f"Created new person {person['person_id']} for {trimmed_name} with cluster {identity_id} and cast_id {cast_id}")

        except Exception as exc:
            # Don't fail the naming operation if People service fails
            LOGGER.warning(f"Failed to create/update People record for {trimmed_name}: {exc}")

        # Mark assignment as manual so it's protected during cleanup operations
        try:
            from apps.api.services.grouping import GroupingService

            grouping_service = GroupingService()
            grouping_service.mark_assignment_manual(ep_id, identity_id, cast_id=cast_id)
            LOGGER.info(f"Marked cluster {identity_id} as manually assigned (cast_id={cast_id})")
        except Exception as exc:
            # Don't fail the naming operation if marking manual fails
            LOGGER.warning(f"Failed to mark cluster {identity_id} as manual: {exc}")

    return {"ep_id": ep_id, "identity_id": identity_id, "name": trimmed_name}


def assign_identity_name(ep_id: str, identity_id: str, name: str, show: str | None = None, cast_id: str | None = None) -> Dict[str, Any]:
    payload = load_identities(ep_id)
    entries = _identity_rows(payload)
    target = next(
        (entry for entry in entries if (entry.get("identity_id") or entry.get("id")) == identity_id),
        None,
    )
    if target is None:
        raise ValueError("identity_not_found")
    trimmed = (name or "").strip()
    if not trimmed:
        raise ValueError("name_required")
    target["name"] = trimmed
    return _persist_identity_name(ep_id, payload, identity_id, trimmed, show, cast_id)


def assign_track_name(ep_id: str, track_id: int, name: str, show: str | None = None, cast_id: str | None = None) -> Dict[str, Any]:
    trimmed = (name or "").strip()
    if not trimmed:
        raise ValueError("name_required")
    try:
        track_id_int = int(track_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_track_id") from exc

    payload = load_identities(ep_id)
    entries = _identity_rows(payload)
    target_identity = None
    track_ids_for_identity: List[int] = []
    for entry in entries:
        raw_ids = entry.get("track_ids", []) or []
        normalized: List[int] = []
        for raw in raw_ids:
            try:
                normalized.append(int(raw))
            except (TypeError, ValueError):
                continue
        if track_id_int in normalized:
            target_identity = entry
            track_ids_for_identity = normalized
            break
    if target_identity is None:
        raise ValueError("track_not_found")

    identity_id = target_identity.get("identity_id") or target_identity.get("id")
    if not identity_id:
        identity_id = _next_identity_id(entries)
        target_identity["identity_id"] = identity_id

    if len(track_ids_for_identity) <= 1:
        target_identity["name"] = trimmed
        result = _persist_identity_name(ep_id, payload, identity_id, trimmed, show, cast_id)
        result["track_id"] = track_id_int
        result["split"] = False
        return result

    remaining_ids = [tid for tid in track_ids_for_identity if tid != track_id_int]
    target_identity["track_ids"] = remaining_ids

    new_identity_id = _next_identity_id(entries)
    new_identity = {
        "identity_id": new_identity_id,
        "label": None,
        "track_ids": [track_id_int],
        "name": trimmed,
    }
    payload.setdefault("identities", []).append(new_identity)

    result = _persist_identity_name(ep_id, payload, new_identity_id, trimmed, show, cast_id)
    result["track_id"] = track_id_int
    result["split"] = True
    result["source_identity_id"] = identity_id
    return result


def move_frames(
    ep_id: str,
    from_track_id: int,
    face_ids: Sequence[str],
    *,
    target_identity_id: str | None = None,
    new_identity_name: str | None = None,
    show_id: str | None = None,
) -> Dict[str, Any]:
    if not face_ids:
        raise ValueError("face_ids_required")
    try:
        ctx = episode_context_from_id(ep_id)
    except ValueError as exc:
        raise ValueError("invalid_ep_id") from exc

    # Acquire episode lock to prevent concurrent modifications
    with _episode_lock(ep_id):
        return _move_frames_locked(ep_id, from_track_id, face_ids, target_identity_id, new_identity_name, show_id, ctx)


def _move_frames_locked(
    ep_id: str,
    from_track_id: int,
    face_ids: Sequence[str],
    target_identity_id: str | None,
    new_identity_name: str | None,
    show_id: str | None,
    ctx: Any,
) -> Dict[str, Any]:
    """Internal implementation of move_frames, called while holding episode lock."""
    faces = load_faces(ep_id)
    face_map = {str(row.get("face_id")): row for row in faces}
    selected: List[Dict[str, Any]] = []
    for face_id in set(face_ids):
        row = face_map.get(face_id)
        if not row:
            raise ValueError(f"face_not_found:{face_id}")
        try:
            track_id = int(row.get("track_id", -1))
        except (TypeError, ValueError):
            raise ValueError(f"face_track_invalid:{face_id}")
        if track_id != from_track_id:
            raise ValueError(f"face_not_in_track:{face_id}")
        selected.append(row)

    if not selected:
        raise ValueError("selected_faces_missing")

    identities_payload = load_identities(ep_id)
    identities = _identity_rows(identities_payload)
    # Coerce track_ids to int for consistent comparison (handles string/int mismatch)
    source_identity = next(
        (entry for entry in identities if from_track_id in {int(tid) for tid in (entry.get("track_ids") or [])}),
        None,
    )

    target_identity = None
    if target_identity_id:
        target_identity = next(
            (entry for entry in identities if (entry.get("identity_id") or entry.get("id")) == target_identity_id),
            None,
        )
    trimmed_name = (new_identity_name or "").strip()
    if target_identity is None and trimmed_name:
        target_identity = next(
            (
                entry
                for entry in identities
                if isinstance(entry.get("name"), str) and entry["name"].lower() == trimmed_name.lower()
            ),
            None,
        )
    if target_identity is None and trimmed_name:
        new_identity_id = _next_identity_id(identities)
        target_identity = {
            "identity_id": new_identity_id,
            "label": None,
            "track_ids": [],
        }
        identities.append(target_identity)
    if target_identity is None:
        raise ValueError("target_identity_not_found")
    target_identity.setdefault("track_ids", [])
    if trimmed_name:
        target_identity["name"] = trimmed_name
        try:
            roster_service.add_if_missing(show_id or ctx.show_slug, trimmed_name)
        except ValueError:
            # Ignore duplicates when the name already exists on the roster.
            pass

    tracks = load_tracks(ep_id)
    source_track = next((row for row in tracks if int(row.get("track_id", -1)) == from_track_id), None)
    if source_track is None:
        raise ValueError("track_not_found")
    new_track_id = _next_track_id(tracks)
    prefixes = artifact_prefixes(ctx)
    crops_prefix = prefixes.get("crops", "")
    thumbs_prefix = prefixes.get("thumbs_tracks", "")
    frames_root = get_path(ep_id, "frames_root")
    thumbs_root = frames_root / "thumbs"

    for row in selected:
        frame_idx = int(row.get("frame_idx", 0))
        old_crop_path = frames_root / (row.get("crop_rel_path") or "")
        new_crop_rel = _crop_rel(new_track_id, frame_idx)
        new_crop_path = frames_root / new_crop_rel
        new_crop_path.parent.mkdir(parents=True, exist_ok=True)
        if old_crop_path.exists():
            try:
                old_crop_path.rename(new_crop_path)
            except OSError:
                # Parallel transfers may hold locks; skip and keep old path.
                pass
        old_thumb_rel = row.get("thumb_rel_path")
        old_thumb_path = thumbs_root / old_thumb_rel if old_thumb_rel else None
        new_thumb_rel = _thumb_rel(new_track_id, frame_idx)
        new_thumb_path = thumbs_root / new_thumb_rel
        new_thumb_path.parent.mkdir(parents=True, exist_ok=True)
        if old_thumb_path and old_thumb_path.exists():
            try:
                old_thumb_path.rename(new_thumb_path)
            except OSError:
                # Skip if filesystem renames collide; we'll refresh later.
                pass
        row["track_id"] = new_track_id
        row["crop_rel_path"] = new_crop_rel
        if crops_prefix:
            row["crop_s3_key"] = f"{crops_prefix}track_{new_track_id:04d}/frame_{frame_idx:06d}.jpg"
        row["thumb_rel_path"] = new_thumb_rel
        if thumbs_prefix:
            row["thumb_s3_key"] = f"{thumbs_prefix}track_{new_track_id:04d}/thumb_{frame_idx:06d}.jpg"

    faces_path = write_faces(ep_id, faces)
    faces_by_track = _faces_by_track(faces)

    new_track_entry = dict(source_track)
    new_track_entry["track_id"] = new_track_id
    new_track_entry = _rebuild_track_entry(new_track_entry, faces_by_track.get(new_track_id, []))
    tracks.append(new_track_entry)

    remaining_faces = faces_by_track.get(from_track_id, [])
    if remaining_faces:
        _rebuild_track_entry(source_track, remaining_faces)
    else:
        tracks = [row for row in tracks if int(row.get("track_id", -1)) != from_track_id]
        if source_identity:
            source_identity["track_ids"] = [tid for tid in source_identity.get("track_ids", []) if tid != from_track_id]

    _write_track_index(ep_id, from_track_id, remaining_faces, ctx)
    _write_track_index(ep_id, new_track_id, faces_by_track.get(new_track_id, []), ctx)

    target_identity["track_ids"] = sorted({int(tid) for tid in target_identity.get("track_ids", [])} | {new_track_id})
    identities_payload["identities"] = identities
    update_identity_stats(ep_id, identities_payload)

    identities_path = write_identities(ep_id, identities_payload)
    tracks_path = write_tracks(ep_id, tracks)
    _annotate_screentime_csv(ep_id, identities_payload)
    sync_manifests(ep_id, faces_path, tracks_path, identities_path)

    affected_ids = {target_identity.get("identity_id")}
    if source_identity and source_identity.get("identity_id"):
        affected_ids.add(source_identity["identity_id"])
    summary = cluster_track_summary(ep_id, include=affected_ids)

    return {
        "ep_id": ep_id,
        "moved_faces": len(selected),
        "new_track_id": new_track_id,
        "target_identity_id": target_identity.get("identity_id"),
        "target_name": target_identity.get("name"),
        "clusters": summary["clusters"],
    }
