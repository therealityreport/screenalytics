from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Set

import cv2  # type: ignore
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from py_screenalytics.artifacts import ensure_dirs, get_path
from py_screenalytics import run_layout
from py_screenalytics.facebank_seed import select_facebank_seeds, write_facebank_seeds
from tools import episode_run

from apps.api.services import roster as roster_service
from apps.api.services import identities as identity_service
from apps.api.services import metrics as metrics_service
from apps.api.services.archive import archive_service
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


def normalize_ep_id(ep_id: str) -> str:
    """Normalize ep_id to lowercase for case-insensitive handling.

    This ensures that 'RHOSLC-s06e02' and 'rhoslc-s06e02' are treated as the same episode
    throughout the API and file system operations.
    """
    return ep_id.strip().lower()


def _diag(tag: str, **kw) -> None:
    """Diagnostic logger enabled via DIAG_LOG=1."""
    if DIAG:
        LOGGER.info("[DIAG:%s] %s", tag, json.dumps(kw, ensure_ascii=False))


def _manifests_dir(ep_id: str) -> Path:
    return get_path(ep_id, "detections").parent


def _normalize_run_id(run_id: str | None) -> str | None:
    """Normalize optional run_id for run-scoped artifacts.

    Empty strings are treated as absent for backwards compatibility.
    """
    if run_id is None:
        return None
    candidate = str(run_id).strip()
    if not candidate:
        return None
    return run_layout.normalize_run_id(candidate)


def _manifests_dir_for_run(ep_id: str, run_id: str | None) -> Path:
    """Return run-scoped manifests directory when run_id is provided."""
    if not run_id:
        return _manifests_dir(ep_id)
    return run_layout.run_root(ep_id, run_id)


def _detections_path_for_run(ep_id: str, run_id: str | None) -> Path:
    if not run_id:
        return get_path(ep_id, "detections")
    return _manifests_dir_for_run(ep_id, run_id) / "detections.jsonl"


def _tracks_path_for_run(ep_id: str, run_id: str | None) -> Path:
    if not run_id:
        return get_path(ep_id, "tracks")
    return _manifests_dir_for_run(ep_id, run_id) / "tracks.jsonl"


def _faces_path_for_run(ep_id: str, run_id: str | None) -> Path:
    return _manifests_dir_for_run(ep_id, run_id) / "faces.jsonl"


def _identities_path_for_run(ep_id: str, run_id: str | None) -> Path:
    return _manifests_dir_for_run(ep_id, run_id) / "identities.json"


def _track_metrics_path_for_run(ep_id: str, run_id: str | None) -> Path:
    return _manifests_dir_for_run(ep_id, run_id) / "track_metrics.json"


def _faces_path(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "faces.jsonl"


def _faces_ops_path(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "faces_ops.jsonl"


@lru_cache(maxsize=8)
def _load_detections_by_frame(ep_id: str, mtime: float) -> Dict[int, List[Dict]]:
    """Load detections and index by frame_idx. Cached with mtime for invalidation."""
    detections_path = get_path(ep_id, "detections")
    result: Dict[int, List[Dict]] = {}
    if detections_path.exists():
        with open(detections_path, "r") as f:
            for line in f:
                try:
                    det = json.loads(line)
                    fidx = det.get("frame_idx")
                    if fidx is not None:
                        if fidx not in result:
                            result[fidx] = []
                        result[fidx].append(det)
                except json.JSONDecodeError:
                    continue
    return result


def _get_detections_by_frame(ep_id: str) -> Dict[int, List[Dict]]:
    """Get cached detections indexed by frame. Cache invalidates on file change."""
    detections_path = get_path(ep_id, "detections")
    mtime = detections_path.stat().st_mtime if detections_path.exists() else 0.0
    return _load_detections_by_frame(ep_id, mtime)


def _append_face_ops(ep_id: str, entries: Iterable[Dict[str, Any]]) -> None:
    entries = list(entries)
    if not entries:
        return
    path = _faces_ops_path(ep_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            payload = dict(entry)
            payload.setdefault("ts", timestamp)
            handle.write(json.dumps(payload) + "\n")


def _identities_path(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "identities.json"


def _runs_dir(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "runs"


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


_RUN_ACTIVITY_FILES: tuple[str, ...] = (
    "progress.json",
    "detect_track.json",
    "faces_embed.json",
    "cluster.json",
    "detections.jsonl",
    "tracks.jsonl",
    "faces.jsonl",
    "identities.json",
    "track_metrics.json",
)


def _latest_run_id_on_disk(ep_id: str) -> str | None:
    """Return the most recently updated run_id directory for this episode.

    Uses a small set of known phase markers/manifests/progress files to
    approximate "latest activity" without trusting legacy marker payloads.
    """
    run_ids = run_layout.list_run_ids(ep_id)
    if not run_ids:
        return None
    best_run_id: str | None = None
    best_mtime = 0.0
    for candidate in run_ids:
        try:
            run_root = run_layout.run_root(ep_id, candidate)
        except ValueError:
            continue
        mtime = 0.0
        for filename in _RUN_ACTIVITY_FILES:
            mtime = max(mtime, _safe_mtime(run_root / filename))
        if mtime > best_mtime:
            best_mtime = mtime
            best_run_id = candidate
    return best_run_id if best_mtime > 0 and best_run_id else None


def _resolve_active_run_id(ep_id: str) -> str | None:
    """Best-effort active run_id for UI defaults.

    Prefer active_run.json when it points to an existing run directory; otherwise
    fall back to the most recent run directory on disk.
    """
    try:
        candidate = run_layout.read_active_run_id(ep_id)
    except Exception:
        candidate = None
    if candidate:
        try:
            if run_layout.run_root(ep_id, candidate).exists():
                return candidate
        except ValueError:
            pass
    return _latest_run_id_on_disk(ep_id)


def _tracks_path(ep_id: str) -> Path:
    return get_path(ep_id, "tracks")


def _thumbs_root(ep_id: str) -> Path:
    return get_path(ep_id, "frames_root") / "thumbs"


def _crops_root(ep_id: str) -> Path:
    return get_path(ep_id, "frames_root") / "crops"


def _resolve_crop_url(ep_id: str, rel_path: str | None, s3_key: str | None) -> str | None:
    """Resolve crop URL via S3 presigning only. Artifacts must be in S3."""
    # Try provided S3 key first
    if s3_key:
        url = STORAGE.presign_get(s3_key)
        if url:
            return url

    # Construct S3 key from rel_path if not provided
    if rel_path:
        normalized = rel_path.strip()
        if normalized:
            try:
                ep_ctx = episode_context_from_id(ep_id)
                prefixes = artifact_prefixes(ep_ctx)
                crops_prefix = prefixes.get("crops")
                if crops_prefix:
                    crop_rel = normalized
                    if crop_rel.startswith("crops/"):
                        crop_rel = crop_rel[6:]
                    constructed_key = f"{crops_prefix}{crop_rel}"
                    url = STORAGE.presign_get(constructed_key)
                    if url:
                        return url
            except (ValueError, KeyError):
                pass

    return None


def _resolve_face_media_url(ep_id: str, row: Dict[str, Any] | None) -> str | None:
    if not row:
        return None
    thumb = _resolve_thumb_url(ep_id, row.get("thumb_rel_path"), row.get("thumb_s3_key"))
    if thumb:
        return thumb
    return _resolve_crop_url(ep_id, row.get("crop_rel_path"), row.get("crop_s3_key"))


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


def _embeds_root(ep_id: str) -> Path:
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
    return data_root / "embeds" / ep_id


def _episode_local_dirs(ep_id: str) -> List[Path]:
    dirs = [
        get_path(ep_id, "video").parent,
        get_path(ep_id, "frames_root"),
        _manifests_dir(ep_id),
        _analytics_root(ep_id),
        _embeds_root(ep_id),
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


def _safe_float(value) -> float | None:
    try:
        return float(value)
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


def _manifest_has_rows(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return True
    except OSError:
        return False
    return False


def _first_manifest_row(path: Path) -> Dict[str, Any] | None:
    if not path.exists() or not path.is_file():
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
                if isinstance(payload, dict):
                    return payload
    except OSError:
        return None
    return None


def _load_run_marker(ep_id: str, phase: str, *, run_id: str | None = None) -> Dict[str, Any] | None:
    marker_path = (
        _manifests_dir_for_run(ep_id, run_id) / f"{phase}.json" if run_id else _runs_dir(ep_id) / f"{phase}.json"
    )
    if not marker_path.exists():
        return None
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _compute_runtime_sec(started_at: str | None, finished_at: str | None) -> float | None:
    """Compute runtime in seconds from ISO timestamps."""
    if not started_at or not finished_at:
        return None
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        delta = (end - start).total_seconds()
        return delta if delta >= 0 else None
    except (ValueError, TypeError):
        return None


def _phase_status_from_marker(phase: str, marker: Dict[str, Any]) -> Dict[str, Any]:
    status_value = str(marker.get("status") or "unknown").lower()
    started_at = marker.get("started_at")
    finished_at = marker.get("finished_at")
    runtime_sec = _compute_runtime_sec(started_at, finished_at)
    return {
        "phase": phase,
        "status": status_value,
        "run_id": marker.get("run_id"),
        "faces": _safe_int(marker.get("faces")),
        "identities": _safe_int(marker.get("identities")),
        "detections": _safe_int(marker.get("detections")),
        "tracks": _safe_int(marker.get("tracks")),
        "detector": marker.get("detector"),
        "tracker": marker.get("tracker"),
        "device": marker.get("device"),
        "requested_device": marker.get("requested_device"),
        "resolved_device": marker.get("resolved_device"),
        "stride": _safe_int(marker.get("stride")),
        "det_thresh": _safe_float(marker.get("det_thresh")),
        "max_gap": _safe_int(marker.get("max_gap")),
        "scene_threshold": _safe_float(marker.get("scene_threshold")),
        "scene_min_len": _safe_int(marker.get("scene_min_len")),
        "scene_warmup_dets": _safe_int(marker.get("scene_warmup_dets")),
        "track_high_thresh": _safe_float(marker.get("track_high_thresh")),
        "new_track_thresh": _safe_float(marker.get("new_track_thresh")),
        "save_frames": marker.get("save_frames"),
        "save_crops": marker.get("save_crops"),
        "jpeg_quality": _safe_int(marker.get("jpeg_quality")),
        "thumb_size": _safe_int(marker.get("thumb_size")),
        "cluster_thresh": _safe_float(marker.get("cluster_thresh")),
        "min_cluster_size": _safe_int(marker.get("min_cluster_size")),
        "min_identity_sim": _safe_float(marker.get("min_identity_sim")),
        "started_at": started_at,
        "finished_at": finished_at,
        "version": marker.get("version"),
        "source": "marker",
        "runtime_sec": runtime_sec,
        "frames_total": _safe_int(marker.get("frames_total")),
        "video_duration_sec": _safe_float(marker.get("video_duration_sec")),
        "fps": _safe_float(marker.get("fps")),
    }


def _get_file_mtime_iso(path: Path) -> str | None:
    """Get the modification time of a file as an ISO timestamp."""
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except OSError:
        return None


def _faces_phase_status(ep_id: str, *, run_id: str | None = None) -> Dict[str, Any]:
    marker = _load_run_marker(ep_id, "faces_embed", run_id=run_id)
    if marker:
        result = _phase_status_from_marker("faces_embed", marker)
        # Add manifest existence info even when marker exists
        faces_path = _faces_path_for_run(ep_id, run_id)
        result["manifest_exists"] = faces_path.exists()
        # Prefer marker timestamp; fall back to manifest mtime if available
        result["last_run_at"] = marker.get("finished_at") or _get_file_mtime_iso(faces_path)
        faces_count = result.get("faces") or 0
        result["zero_rows"] = result["manifest_exists"] and faces_count == 0
        return result
    faces_path = _faces_path_for_run(ep_id, run_id)
    manifest_exists = faces_path.exists()
    faces_count = _count_nonempty_lines(faces_path)
    # SUCCESS if manifest exists (even with 0 rows), MISSING only if no manifest
    status_value = "success" if manifest_exists else "missing"
    source = "output" if manifest_exists else "absent"
    last_run_at = _get_file_mtime_iso(faces_path)
    return {
        "phase": "faces_embed",
        "status": status_value,
        "faces": faces_count,
        "identities": None,
        "device": None,
        "requested_device": None,
        "resolved_device": None,
        "save_frames": None,
        "save_crops": None,
        "jpeg_quality": None,
        "thumb_size": None,
        "started_at": None,
        "finished_at": None,
        "version": None,
        "source": source,
        "manifest_exists": manifest_exists,
        "zero_rows": manifest_exists and faces_count == 0,
        "last_run_at": last_run_at,
    }


def _extract_singleton_merge_stats(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract singleton merge stats from identities.json stats block."""
    stats_block = payload.get("stats") if isinstance(payload, dict) else None
    if not isinstance(stats_block, dict):
        return {}

    result: Dict[str, Any] = {}

    # Include the full singleton_stats block for backwards compatibility
    singleton_stats = stats_block.get("singleton_stats")
    if isinstance(singleton_stats, dict):
        result["singleton_stats"] = singleton_stats
        result["singleton_merge_enabled"] = singleton_stats.get("enabled", True)
        before = singleton_stats.get("before", {})
        after = singleton_stats.get("after", {})
        if isinstance(before, dict):
            result["singleton_fraction_before"] = _safe_float(before.get("singleton_fraction"))
        if isinstance(after, dict):
            result["singleton_fraction_after"] = _safe_float(after.get("singleton_fraction"))
            result["singleton_merge_merge_count"] = _safe_int(after.get("merge_count"))

    # Check singleton_merge block (newer format) - may have additional fields
    singleton_merge = stats_block.get("singleton_merge")
    if isinstance(singleton_merge, dict):
        result["singleton_merge_enabled"] = singleton_merge.get("enabled", True)
        if "singleton_fraction_before" not in result:
            result["singleton_fraction_before"] = _safe_float(singleton_merge.get("singleton_fraction_before"))
        if "singleton_fraction_after" not in result:
            result["singleton_fraction_after"] = _safe_float(singleton_merge.get("singleton_fraction_after"))
        result["singleton_merge_neighbor_top_k"] = _safe_int(singleton_merge.get("neighbor_top_k"))
        result["singleton_merge_similarity_thresh"] = _safe_float(singleton_merge.get("similarity_thresh"))
        if "singleton_merge_merge_count" not in result:
            result["singleton_merge_merge_count"] = _safe_int(singleton_merge.get("num_singleton_merges"))

    return result


def _cluster_phase_status(ep_id: str, *, run_id: str | None = None) -> Dict[str, Any]:
    """Get cluster phase status, checking both identities.json and track_metrics.json.

    For staleness detection, we need to check BOTH files because:
    - identities.json: Main cluster output
    - track_metrics.json: Written during clustering, may exist even when identities.json doesn't

    The cluster is considered to have run if EITHER file exists, and last_run_at
    is the max mtime of both files.
    """
    identities_path = _identities_path_for_run(ep_id, run_id)
    track_metrics_path = _track_metrics_path_for_run(ep_id, run_id)

    # Helper to get max mtime of multiple paths
    def _max_mtime_iso(*paths) -> str | None:
        mtimes = []
        for path in paths:
            if path.exists():
                mtime_iso = _get_file_mtime_iso(path)
                if mtime_iso:
                    mtimes.append(mtime_iso)
        if not mtimes:
            return None
        # ISO timestamps sort lexicographically, so max() works
        return max(mtimes)

    marker = _load_run_marker(ep_id, "cluster", run_id=run_id)
    if marker:
        result = _phase_status_from_marker("cluster", marker)
        # Add manifest existence info even when marker exists
        result["manifest_exists"] = identities_path.exists()
        # Use marker finished_at first; fall back to max of identities/track_metrics mtimes
        result["last_run_at"] = marker.get("finished_at") or _max_mtime_iso(identities_path, track_metrics_path)
        result["track_metrics_exists"] = track_metrics_path.exists()
        identities_count = result.get("identities") or 0
        result["zero_rows"] = result["manifest_exists"] and identities_count == 0
        # Try to add singleton merge stats from identities.json
        if identities_path.exists():
            try:
                payload = json.loads(identities_path.read_text(encoding="utf-8"))
                singleton_stats = _extract_singleton_merge_stats(payload)
                result.update(singleton_stats)
            except (OSError, json.JSONDecodeError):
                pass
        return result

    manifest_exists = identities_path.exists()
    track_metrics_exists = track_metrics_path.exists()
    # Cluster has run if either identities.json or track_metrics.json exists
    has_cluster_output = manifest_exists or track_metrics_exists

    faces_total = 0
    identities_count = 0
    singleton_merge_stats: Dict[str, Any] = {}

    if manifest_exists:
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
        singleton_merge_stats = _extract_singleton_merge_stats(payload)

    # Try to get cluster metrics from track_metrics.json (fallback source)
    if track_metrics_exists and not manifest_exists:
        try:
            metrics_data = json.loads(track_metrics_path.read_text(encoding="utf-8"))
            if isinstance(metrics_data, dict):
                cluster_block = metrics_data.get("cluster_metrics") or {}
                if isinstance(cluster_block, dict):
                    identities_count = cluster_block.get("identities_count", 0)
                    faces_total = cluster_block.get("faces_count", 0)
        except (OSError, json.JSONDecodeError):
            pass

    # SUCCESS if any cluster output exists (even with 0 identities), MISSING only if no output
    status_value = "success" if has_cluster_output else "missing"
    source = "output" if manifest_exists else ("metrics_fallback" if track_metrics_exists else "absent")
    # Use max of both files for staleness detection
    last_run_at = _max_mtime_iso(identities_path, track_metrics_path)

    return {
        "phase": "cluster",
        "status": status_value,
        "faces": faces_total,
        "identities": identities_count,
        "device": None,
        "requested_device": None,
        "resolved_device": None,
        "cluster_thresh": None,
        "min_cluster_size": None,
        "min_identity_sim": None,
        "started_at": None,
        "finished_at": None,
        "version": None,
        "source": source,
        "manifest_exists": manifest_exists,
        "track_metrics_exists": track_metrics_exists,
        "zero_rows": has_cluster_output and identities_count == 0,
        "last_run_at": last_run_at,
        **singleton_merge_stats,
    }


def _detect_track_phase_status(ep_id: str, *, run_id: str | None = None) -> Dict[str, Any]:
    """Get detect/track phase status including detector/tracker info.

    IMPORTANT: Returns the FACE detector (e.g., retinaface), NOT the scene detector.
    Scene detection (pyscenedetect) is a preliminary step, not the main detect/track operation.
    """
    marker = _load_run_marker(ep_id, "detect_track", run_id=run_id)
    if marker:
        result = _phase_status_from_marker("detect_track", marker)
        # Add manifest existence info even when marker exists
        tracks_path = _tracks_path_for_run(ep_id, run_id)
        result["manifest_exists"] = tracks_path.exists()
        # Prefer marker timestamp for stale detection; fall back to manifest mtime if unavailable
        # This ensures consistent timestamp comparison with faces_embed and cluster phases
        result["last_run_at"] = marker.get("finished_at") or _get_file_mtime_iso(tracks_path)
        tracks_count = result.get("tracks") or 0
        result["zero_rows"] = result["manifest_exists"] and tracks_count == 0
        return result
    tracks_path = _tracks_path_for_run(ep_id, run_id)
    detections_path = _detections_path_for_run(ep_id, run_id)

    detections_count = _count_nonempty_lines(detections_path)
    tracks_count = _count_nonempty_lines(tracks_path)
    manifest_exists = tracks_path.exists()
    last_run_at = _get_file_mtime_iso(tracks_path)

    # Infer detector/tracker from tracks.jsonl if available
    # Only accept FACE detectors (retinaface, yolov8face, etc.), NOT scene detectors
    detector = None
    tracker = None
    tracks_manifest_valid = False
    track_row = _first_manifest_row(tracks_path)
    if isinstance(track_row, dict):
        det_value = track_row.get("detector")
        if isinstance(det_value, str):
            det_value = det_value.strip()
            if det_value and det_value.lower() not in (
                "pyscenedetect",
                "internal",
                "off",
            ):
                detector = det_value
        tracker_value = track_row.get("tracker")
        if isinstance(tracker_value, str):
            tracker = tracker_value.strip() or None
        track_id_value = track_row.get("track_id")
        tracks_manifest_valid = detector is not None and tracker is not None and track_id_value is not None

    # Determine status
    if tracks_count > 0 and not tracks_manifest_valid:
        status_value = "invalid"
    elif tracks_count > 0:
        status_value = "success"
    elif detections_count > 0:
        status_value = "partial"  # Has detections but no tracks
    else:
        status_value = "missing"

    source = "output" if tracks_path.exists() or detections_path.exists() else "absent"

    return {
        "phase": "detect_track",
        "status": status_value,
        "detections": detections_count,
        "tracks": tracks_count,
        "detector": detector,
        "tracker": tracker,
        "device": None,
        "requested_device": None,
        "resolved_device": None,
        "stride": None,
        "det_thresh": None,
        "max_gap": None,
        "scene_threshold": None,
        "scene_min_len": None,
        "scene_warmup_dets": None,
        "track_high_thresh": None,
        "new_track_thresh": None,
        "faces": None,
        "identities": None,
        "started_at": None,
        "finished_at": None,
        "version": None,
        "source": source,
        "manifest_exists": manifest_exists,
        "zero_rows": manifest_exists and tracks_count == 0,
        "last_run_at": last_run_at,
    }


def _delete_episode_assets(ep_id: str, options) -> Dict[str, Any]:
    # Normalize ep_id to lowercase for case-insensitive handling
    ep_id = normalize_ep_id(ep_id)
    record = EPISODE_STORE.get(ep_id)
    if not record:
        raise HTTPException(status_code=404, detail="Episode not found")
    try:
        ep_ctx = episode_context_from_id(ep_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid episode id") from exc

    # Clean up people data BEFORE deleting local/S3 artifacts
    # This removes orphaned cluster_ids from people.json for this show
    people_cleanup = {
        "people_modified": 0,
        "clusters_removed": 0,
        "empty_people_removed": 0,
    }
    try:
        from apps.api.services.people import PeopleService

        people_service = PeopleService()
        show_id = record.show_ref  # Use show_ref from episode record
        people_cleanup = people_service.remove_episode_clusters(show_id, ep_id)
        LOGGER.info(
            "Cleaned up people data for %s: %d people modified, %d clusters removed, %d empty people removed",
            ep_id,
            people_cleanup["people_modified"],
            people_cleanup["clusters_removed"],
            people_cleanup["empty_people_removed"],
        )
    except Exception as exc:  # pragma: no cover - best effort cleanup
        LOGGER.warning("Failed to clean up people data for %s: %s", ep_id, exc)

    # Also clear person_id assignments from identities.json
    identities_cleared = 0
    try:
        from pathlib import Path

        identities_path = Path(f"data/manifests/{ep_id}/identities.json")
        if identities_path.exists():
            import json

            data = json.loads(identities_path.read_text(encoding="utf-8"))
            identities = data.get("identities", [])
            for identity in identities:
                if "person_id" in identity:
                    del identity["person_id"]
                    identities_cleared += 1
            if identities_cleared > 0:
                data["identities"] = identities
                identities_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                LOGGER.info(
                    "Cleared %d person_id assignment(s) from identities.json for %s",
                    identities_cleared,
                    ep_id,
                )
    except Exception as exc:  # pragma: no cover - best effort cleanup
        LOGGER.warning("Failed to clear identities.json for %s: %s", ep_id, exc)

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
        for key in (
            "frames",
            "crops",
            "manifests",
            "analytics",
            "thumbs_tracks",
            "thumbs_identities",
        ):
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
        "deleted": {
            "local_dirs": local_deleted,
            "s3_objects": s3_deleted,
            "people_cleanup": people_cleanup,
        },
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


def _refresh_similarity_indexes(
    ep_id: str,
    *,
    track_ids: Iterable[int] | None = None,
    identity_ids: Iterable[str] | None = None,
) -> None:
    """Regenerate track reps/centroids and refresh people prototypes if impacted."""
    track_set: Set[int] = set()
    for raw in track_ids or []:
        try:
            track_set.add(int(raw))
        except (TypeError, ValueError):
            continue

    identity_set: Set[str] = set()
    for raw in identity_ids or []:
        if not raw:
            continue
        identity_set.add(str(raw))

    identities_payload = _load_identities(ep_id)
    if track_set:
        track_lookup = _identity_lookup(identities_payload)
        for tid in track_set:
            identity = track_lookup.get(tid)
            if identity:
                identity_set.add(identity)

    try:
        from apps.api.services.track_reps import (
            generate_track_reps_and_centroids,
            load_cluster_centroids,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning("Cannot refresh track reps for %s: %s", ep_id, exc)
        return

    try:
        result = generate_track_reps_and_centroids(ep_id)
    except Exception as exc:  # pragma: no cover - surface but don't fail request
        LOGGER.error("Track rep regeneration failed for %s: %s", ep_id, exc)
        return

    sync_paths = []
    for path_key in ("track_reps_path", "cluster_centroids_path"):
        raw_path = result.get(path_key)
        if not raw_path:
            continue
        path_obj = Path(raw_path)
        if path_obj.exists():
            sync_paths.append(path_obj)
    if sync_paths:
        _sync_manifests(ep_id, *sync_paths)

    if not identity_set:
        return

    try:
        from apps.api.services.people import PeopleService, l2_normalize
        import numpy as np
    except Exception as exc:  # pragma: no cover - best-effort prototype refresh
        LOGGER.warning("PeopleService unavailable for %s similarity refresh: %s", ep_id, exc)
        return

    try:
        ep_ctx = episode_context_from_id(ep_id)
    except ValueError as exc:  # pragma: no cover - invalid IDs already filtered upstream
        LOGGER.warning("Unable to parse episode id %s for similarity refresh: %s", ep_id, exc)
        return

    show_id = ep_ctx.show_slug.upper()
    people_service = PeopleService()
    people = people_service.list_people(show_id)
    if not people:
        return

    cluster_refs = {f"{ep_id}:{identity_id}" for identity_id in identity_set}

    touched_person_ids: Set[str] = set()
    for identity in identities_payload.get("identities", []):
        if identity.get("identity_id") in identity_set and identity.get("person_id"):
            touched_person_ids.add(identity["person_id"])

    people_by_id = {person.get("person_id"): person for person in people if person.get("person_id")}

    if cluster_refs:
        for person in people:
            person_id = person.get("person_id")
            if not person_id:
                continue
            for cluster_id in person.get("cluster_ids") or []:
                if cluster_id in cluster_refs:
                    touched_person_ids.add(person_id)
                    break

    if not touched_person_ids:
        return

    centroid_cache: Dict[str, Dict[str, Any]] = {}

    def _centroid_vec(ep_slug: str, cluster_id: str):
        if ep_slug not in centroid_cache:
            try:
                centroids = load_cluster_centroids(ep_slug)
            except Exception as exc:  # pragma: no cover - best effort
                LOGGER.warning("Failed to load centroids for %s: %s", ep_slug, exc)
                centroid_cache[ep_slug] = {}
            else:
                centroid_cache[ep_slug] = centroids if isinstance(centroids, dict) else {}
        centroids = centroid_cache.get(ep_slug, {})
        record = centroids.get(cluster_id) if isinstance(centroids, dict) else None
        if not isinstance(record, dict):
            return None
        vector = record.get("centroid")
        if not isinstance(vector, list):
            return None
        return np.array(vector, dtype=np.float32)

    for person_id in touched_person_ids:
        person = people_by_id.get(person_id)
        if not person:
            continue
        original_clusters = list(person.get("cluster_ids") or [])
        updated_clusters: List[str] = []
        vectors = []
        for cluster_ref in original_clusters:
            if not isinstance(cluster_ref, str) or ":" not in cluster_ref:
                updated_clusters.append(cluster_ref)
                continue
            ep_slug, cluster_id = cluster_ref.split(":", 1)
            vec = _centroid_vec(ep_slug, cluster_id)
            if vec is None:
                if cluster_ref not in cluster_refs:
                    updated_clusters.append(cluster_ref)
                continue
            vectors.append(vec)
            updated_clusters.append(cluster_ref)

        updates: Dict[str, Any] = {}
        if vectors:
            stacked = np.stack(vectors, axis=0)
            proto = l2_normalize(np.mean(stacked, axis=0)).tolist()
            updates["prototype"] = proto
        elif not updated_clusters:
            updates["prototype"] = []

        if updated_clusters != original_clusters:
            updates["cluster_ids"] = updated_clusters

        if updates:
            people_service.update_person(show_id, person_id, **updates)


def _resolve_thumb_url(ep_id: str, rel_path: str | None, s3_key: str | None) -> str | None:
    """Resolve thumbnail URL via S3 presigning only. Artifacts must be in S3."""
    # Try provided S3 key first
    if s3_key:
        url = STORAGE.presign_get(s3_key)
        if url:
            return url

    # Construct S3 key from rel_path if not provided
    if rel_path:
        try:
            ep_ctx = episode_context_from_id(ep_id)
            prefixes = artifact_prefixes(ep_ctx)
            thumbs_prefix = prefixes.get("thumbs")
            if thumbs_prefix:
                constructed_key = f"{thumbs_prefix}{rel_path}"
                url = STORAGE.presign_get(constructed_key)
                if url:
                    return url
        except (ValueError, KeyError):
            pass

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
    scores: Dict[int, tuple[int, int, float, int]] = {}
    for row in faces:
        try:
            tid = int(row.get("track_id", -1))
            frame_idx = int(row.get("frame_idx", 10**9))
        except (TypeError, ValueError):
            continue
        skip_flag = 1 if row.get("skip") else 0
        has_thumb = 0 if (row.get("thumb_rel_path") or row.get("thumb_s3_key")) else 1
        quality_value = row.get("quality")
        if quality_value is None:
            quality_value = row.get("conf") or row.get("confidence")
        try:
            quality_score = float(quality_value) if quality_value is not None else 0.0
        except (TypeError, ValueError):
            quality_score = 0.0
        candidate_score = (skip_flag, has_thumb, -quality_score, frame_idx)
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
                entries.append(
                    {
                        "key": f"{track_component}/{jpeg.name}",
                        "frame_idx": idx,
                        "ts": None,
                    }
                )
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


def _list_track_frame_media(
    ep_id: str,
    track_id: int,
    sample: int,
    page: int,
    page_size: int,
    include_skipped: bool = False,
) -> Dict[str, Any]:
    sample = max(1, sample)
    page = max(1, page)
    page_size = max(1, min(page_size, TRACK_LIST_MAX_LIMIT))

    # Load faces with or without skipped entries based on parameter
    all_faces = _load_faces(ep_id, include_skipped=include_skipped)
    face_rows: Dict[int, Dict[str, Any]] = {}
    for row in all_faces:
        try:
            tid = int(row.get("track_id", -1))
            frame_idx = int(row.get("frame_idx", -1))
        except (TypeError, ValueError):
            continue
        if tid != track_id:
            continue
        face_rows.setdefault(frame_idx, row)
    crops = _discover_crop_entries(ep_id, track_id)
    ctx, prefixes = _require_episode_context(ep_id)
    crops_prefix = (prefixes or {}).get("crops") if prefixes else None

    # Load track centroid for similarity computation and quality scoring functions
    try:
        from apps.api.services.track_reps import (
            load_track_reps,
            _extract_quality_metrics,
            _compute_quality_score,
        )
        import numpy as np

        track_reps = load_track_reps(ep_id)
        track_rep = track_reps.get(f"track_{track_id:04d}", {})
        track_centroid = np.array(track_rep.get("embed", []), dtype=np.float32) if track_rep else None
    except Exception:
        track_centroid = None
        _extract_quality_metrics = None
        _compute_quality_score = None

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

        # First pass: compute quality scores for ALL frames to identify best frame
        best_frame_idx = None
        best_quality_score = -1.0
        if _extract_quality_metrics and _compute_quality_score:
            for idx in frame_indices:
                meta = face_rows.get(idx, {})
                if meta.get("skip"):
                    continue
                det_score, crop_std, box_area = _extract_quality_metrics(meta)
                quality_score = _compute_quality_score(det_score, crop_std, box_area)
                if quality_score > best_quality_score:
                    best_quality_score = quality_score
                    best_frame_idx = idx

        # Second pass: build paginated items with quality metadata
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

            # Extract quality metrics
            quality = None
            if _extract_quality_metrics and _compute_quality_score:
                det_score, crop_std, box_area = _extract_quality_metrics(meta)
                quality_score = _compute_quality_score(det_score, crop_std, box_area)
                quality = {
                    "det_score": round(float(det_score), 3),
                    "crop_std": round(float(crop_std), 1),
                    "box_area": round(float(box_area), 1),
                    "score": round(float(quality_score), 4),
                }

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
                    "quality": quality,
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
            "best_frame_idx": best_frame_idx,
        }

    # First pass: compute quality scores for ALL frames to identify best frame
    best_frame_idx = None
    best_quality_score = -1.0
    if _extract_quality_metrics and _compute_quality_score:
        for entry in crops:
            frame_idx = entry["frame_idx"]
            meta = face_rows.get(frame_idx, {})
            if meta.get("skip"):
                continue
            det_score, crop_std, box_area = _extract_quality_metrics(meta)
            quality_score = _compute_quality_score(det_score, crop_std, box_area)
            if quality_score > best_quality_score:
                best_quality_score = quality_score
                best_frame_idx = frame_idx

    # Second pass: build paginated items with quality metadata
    sampled_entries = crops[::sample]
    total_items = len(sampled_entries)
    start = (page - 1) * page_size
    end = start + page_size
    page_entries = sampled_entries[start:end] if start < total_items else []
    items: List[Dict[str, Any]] = []
    for entry in page_entries:
        frame_idx = entry["frame_idx"]
        meta = face_rows.get(frame_idx, {})
        # Skip frames that don't have a corresponding entry in faces.json
        # Without this, the UI would show frames that can't be moved/deleted
        if not meta:
            continue
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

        # Extract quality metrics
        quality = None
        if _extract_quality_metrics and _compute_quality_score and meta:
            det_score, crop_std, box_area = _extract_quality_metrics(meta)
            quality_score = _compute_quality_score(det_score, crop_std, box_area)
            quality = {
                "det_score": round(float(det_score), 3),
                "crop_std": round(float(crop_std), 1),
                "box_area": round(float(box_area), 1),
                "score": round(float(quality_score), 4),
            }

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
                "quality": quality,
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
        "best_frame_idx": best_frame_idx,
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
    created_at: str | None = None
    updated_at: str | None = None


class EpisodeListResponse(BaseModel):
    episodes: List[EpisodeSummary]


class EpisodeUpsert(BaseModel):
    ep_id: str = Field(..., min_length=3, description="Deterministic ep_id (slug-sXXeYY)")
    show_slug: str = Field(..., min_length=1)
    season: int = Field(..., ge=0, le=999)
    episode: int = Field(..., ge=0, le=999)
    title: str | None = Field(None, max_length=200)
    air_date: date | None = None


class EpisodeUpdateRequest(BaseModel):
    """Request body for updating episode metadata (non-identity fields only).

    Note: show/season/episode cannot be changed as they determine the ep_id
    which is used as a filesystem key throughout the system.
    """

    title: str | None = Field(None, max_length=200, description="Episode title")
    air_date: date | None = Field(None, description="Air date (YYYY-MM-DD)")


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


class BulkTrackAssignRequest(BaseModel):
    track_ids: List[int] = Field(..., min_length=1, description="List of track IDs to assign")
    name: str = Field(..., min_length=1, max_length=200, description="Name to assign")
    show: str | None = Field(None, description="Optional show slug override")
    cast_id: str | None = Field(None, description="Optional cast_id to link assignment")


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
    run_id: str | None = None
    faces: int | None = None
    identities: int | None = None
    detections: int | None = None
    tracks: int | None = None
    detector: str | None = None
    tracker: str | None = None
    device: str | None = None
    requested_device: str | None = None
    resolved_device: str | None = None
    stride: int | None = None
    det_thresh: float | None = None
    max_gap: int | None = None
    scene_threshold: float | None = None
    scene_min_len: int | None = None
    scene_warmup_dets: int | None = None
    track_high_thresh: float | None = None
    new_track_thresh: float | None = None
    save_frames: bool | None = None
    save_crops: bool | None = None
    jpeg_quality: int | None = None
    thumb_size: int | None = None
    cluster_thresh: float | None = None
    min_cluster_size: int | None = None
    min_identity_sim: float | None = None
    started_at: str | None = None
    finished_at: str | None = None
    version: str | None = None
    source: str | None = None
    runtime_sec: float | None = None  # Computed from started_at/finished_at
    # Singleton merge stats for cluster phase
    singleton_merge_enabled: bool | None = None
    singleton_fraction_before: float | None = None
    singleton_fraction_after: float | None = None
    singleton_merge_neighbor_top_k: int | None = None
    singleton_merge_merge_count: int | None = None
    singleton_merge_similarity_thresh: float | None = None
    singleton_stats: Dict[str, Any] | None = None  # Full singleton stats block
    # New fields for manifest existence and zero-result detection
    manifest_exists: bool | None = None
    zero_rows: bool | None = None
    last_run_at: str | None = None  # ISO timestamp of manifest mtime
    # Cluster-specific: indicates track_metrics.json exists (even if identities.json missing)
    track_metrics_exists: bool | None = None


class EpisodeStatusResponse(BaseModel):
    ep_id: str
    active_run_id: str | None = None
    detect_track: PhaseStatus
    faces_embed: PhaseStatus
    cluster: PhaseStatus
    # Pipeline state indicators
    scenes_ready: bool
    tracks_ready: bool
    faces_harvested: bool
    coreml_available: bool | None = None
    # Stale detection flags
    faces_stale: bool = False
    cluster_stale: bool = False
    # Fallback indicators for backwards compatibility
    faces_manifest_fallback: bool = False  # True if faces are stale relative to tracks
    tracks_only_fallback: bool = False  # True if cluster is stale


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
    include_s3: bool = True
    delete_raw: bool = False
    delete_local: bool = True


class PurgeAllIn(BaseModel):
    confirm: str
    delete_artifacts: bool = True
    delete_raw: bool = False
    delete_local: bool = True


@router.get("/episodes", response_model=EpisodeListResponse, tags=["episodes"])
def list_episodes() -> EpisodeListResponse:
    """List all episodes with timestamps for sorting by recent activity.

    Show slugs are normalized to UPPERCASE for consistent display.
    """
    records = EPISODE_STORE.list()
    episodes = [
        EpisodeSummary(
            ep_id=record.ep_id,
            # Normalize show_slug to uppercase for consistent display
            show_slug=record.show_ref.upper() if record.show_ref else record.show_ref,
            season_number=record.season_number,
            episode_number=record.episode_number,
            title=record.title,
            air_date=record.air_date,
            created_at=record.created_at,
            updated_at=record.updated_at,
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
                last_modified=(str(obj.get("last_modified")) if obj.get("last_modified") else None),
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
    """List all shows available in S3 with episode counts.

    Show codes are normalized to UPPERCASE and deduplicated case-insensitively.
    """
    raw_items = STORAGE.list_episode_videos_s3(limit=10000)

    # Group by show (normalized to uppercase for deduplication)
    show_episodes: Dict[str, int] = {}
    for obj in raw_items:
        show = obj.get("show")
        if show and isinstance(show, str):
            # Normalize to uppercase for case-insensitive grouping
            show_upper = show.upper()
            show_episodes[show_upper] = show_episodes.get(show_upper, 0) + 1

    # Convert to sorted list (already uppercase)
    shows = [S3Show(show=show, episode_count=count) for show, count in sorted(show_episodes.items())]
    return S3ShowsResponse(shows=shows, count=len(shows))


@router.get(
    "/episodes/s3_shows/{show}/episodes",
    response_model=S3EpisodesForShowResponse,
    tags=["episodes"],
)
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
    # Check both lowercase and uppercase show slugs for S3 v2 path (case-insensitive)
    v2_key = STORAGE.video_object_key_v2(record.show_ref.lower(), record.season_number, record.episode_number)
    v2_key_upper = STORAGE.video_object_key_v2(record.show_ref.upper(), record.season_number, record.episode_number)
    v1_key = STORAGE.video_object_key_v1(ep_id)
    v2_exists = STORAGE.object_exists(v2_key) or STORAGE.object_exists(v2_key_upper)
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


def _parse_iso_timestamp(ts: str | None) -> datetime | None:
    """Parse an ISO timestamp string to datetime."""
    if not ts:
        return None
    try:
        cleaned = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except (ValueError, TypeError):
        return None


def _phase_progress_value(status: str | None) -> float | None:
    """Map phase status to a coarse progress value for SSE clients."""
    if not status:
        return None
    status_lower = status.lower()
    mapping = {
        "success": 1.0,
        "stale": 0.9,
        "running": 0.6,
        "partial": 0.5,
        "invalid": 0.2,
        "missing": 0.0,
    }
    return mapping.get(status_lower, None)


def _status_to_events(ep_id: str, status: EpisodeStatusResponse) -> List[Dict[str, Any]]:
    flags = {
        "tracks_only_fallback": status.tracks_only_fallback,
        "faces_manifest_fallback": status.faces_manifest_fallback,
    }
    events: List[Dict[str, Any]] = []
    for phase_name, phase_status in (
        ("detect_track", status.detect_track),
        ("faces", status.faces_embed),
        ("cluster", status.cluster),
    ):
        if not phase_status:
            continue
        payload: Dict[str, Any] = {
            "episode_id": ep_id,
            "phase": phase_name,
            "event": "progress",
            "message": phase_status.status,
            "progress": _phase_progress_value(getattr(phase_status, "status", None)),
            "flags": flags,
            "manifest_mtime": getattr(phase_status, "last_run_at", None),
        }
        if phase_name == "cluster":
            payload["metrics"] = {
                "singleton_fraction_before": phase_status.singleton_fraction_before,
                "singleton_fraction_after": phase_status.singleton_fraction_after,
            }
        events.append(payload)
    return events


async def _status_snapshot(ep_id: str) -> EpisodeStatusResponse:
    return await asyncio.to_thread(_episode_run_status, ep_id, None)


def _episode_run_status(ep_id: str, run_id: str | None) -> EpisodeStatusResponse:
    run_id_norm = _normalize_run_id(run_id)

    tracks_path = _tracks_path_for_run(ep_id, run_id_norm)
    detections_path = _detections_path_for_run(ep_id, run_id_norm)
    detect_track_payload = _detect_track_phase_status(ep_id, run_id=run_id_norm)

    detections_manifest_ready = _manifest_has_rows(detections_path)
    tracks_manifest_ready = _manifest_has_rows(tracks_path)
    manifest_ready = detections_manifest_ready and tracks_manifest_ready
    if detect_track_payload.get("status") == "success" and not manifest_ready:
        detect_track_payload["status"] = "stale"
        detect_track_payload["source"] = "missing_artifact"
    detect_track_status = PhaseStatus(**detect_track_payload)
    faces_payload = _faces_phase_status(ep_id, run_id=run_id_norm)
    cluster_payload = _cluster_phase_status(ep_id, run_id=run_id_norm)

    # Stale detection: compare timestamps to detect outdated downstream artifacts
    detect_track_mtime = _parse_iso_timestamp(detect_track_payload.get("last_run_at"))
    faces_mtime = _parse_iso_timestamp(faces_payload.get("last_run_at"))
    cluster_mtime = _parse_iso_timestamp(cluster_payload.get("last_run_at"))

    faces_stale = False
    cluster_stale = False
    faces_manifest_fallback = False
    tracks_only_fallback = False

    # If detect/track is newer than faces, faces are stale
    if detect_track_mtime and faces_mtime and detect_track_mtime > faces_mtime:
        if faces_payload.get("manifest_exists"):
            faces_stale = True
            faces_manifest_fallback = True
            # Mark faces status as stale
            if faces_payload.get("status") == "success":
                faces_payload["status"] = "stale"

    # If detect/track or faces is newer than cluster, cluster is stale
    # Check both manifest_exists and track_metrics_exists for stale detection
    has_cluster_output = cluster_payload.get("manifest_exists") or cluster_payload.get("track_metrics_exists")
    if has_cluster_output:
        if detect_track_mtime and cluster_mtime and detect_track_mtime > cluster_mtime:
            cluster_stale = True
            # tracks_only_fallback indicates we're using metrics-only fallback
            tracks_only_fallback = not cluster_payload.get("manifest_exists") and cluster_payload.get("track_metrics_exists")
            if cluster_payload.get("status") == "success":
                cluster_payload["status"] = "stale"
        elif faces_mtime and cluster_mtime and faces_mtime > cluster_mtime:
            cluster_stale = True
            tracks_only_fallback = not cluster_payload.get("manifest_exists") and cluster_payload.get("track_metrics_exists")
            if cluster_payload.get("status") == "success":
                cluster_payload["status"] = "stale"
        # Even if not stale, mark tracks_only_fallback if using metrics fallback
        if not cluster_stale and not cluster_payload.get("manifest_exists") and cluster_payload.get("track_metrics_exists"):
            tracks_only_fallback = True

    faces_status = PhaseStatus(**faces_payload)
    cluster_status = PhaseStatus(**cluster_payload)

    # Compute pipeline state indicators
    # scenes_ready: True if scene detection has run (consider ready if tracks manifest has rows)
    scenes_ready = tracks_manifest_ready or detections_manifest_ready

    # tracks_ready: True only when both detections+tracks exist and API reports success
    tracks_ready = manifest_ready and detect_track_status.status == "success"

    # faces_harvested: True if faces.jsonl EXISTS (even with 0 rows)
    # This distinguishes "never ran" from "ran with zero results"
    faces_harvested = bool(faces_status.manifest_exists)

    coreml_available = getattr(episode_run, "COREML_PROVIDER_AVAILABLE", None)

    active_run_id: str | None = run_id_norm or _resolve_active_run_id(ep_id)

    return EpisodeStatusResponse(
        ep_id=ep_id,
        active_run_id=active_run_id,
        detect_track=detect_track_status,
        faces_embed=faces_status,
        cluster=cluster_status,
        scenes_ready=scenes_ready,
        tracks_ready=tracks_ready,
        faces_harvested=faces_harvested,
        coreml_available=coreml_available if isinstance(coreml_available, bool) else None,
        faces_stale=faces_stale,
        cluster_stale=cluster_stale,
        faces_manifest_fallback=faces_manifest_fallback,
        tracks_only_fallback=tracks_only_fallback,
    )


@router.get("/episodes/{ep_id}/status", response_model=EpisodeStatusResponse, tags=["episodes"])
def episode_run_status(ep_id: str, run_id: str | None = Query(None)) -> EpisodeStatusResponse:
    try:
        return _episode_run_status(ep_id, run_id)
    except ValueError as exc:
        # FastAPI will coerce query params to str; validate run_id explicitly.
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/episodes/{ep_id}/events",
    tags=["episodes"],
    response_class=StreamingResponse,
)
async def episode_events(
    ep_id: str,
    poll_ms: int = Query(1500, ge=200, le=10000),
    max_events: int = Query(250, ge=1, le=2000),
):
    ep_id_normalized = normalize_ep_id(ep_id)

    async def event_stream():
        # SSE stream keeps the web UI in sync without aggressive polling.
        sent = 0
        while sent < max_events:
            try:
                status = await _status_snapshot(ep_id_normalized)
            except HTTPException as exc:
                payload = {
                    "episode_id": ep_id_normalized,
                    "phase": "detect_track",
                    "event": "error",
                    "message": str(exc.detail),
                }
                yield f"data: {json.dumps(payload)}\n\n"
                break

            for payload in _status_to_events(ep_id_normalized, status):
                yield f"data: {json.dumps(payload)}\n\n"
                sent += 1
                if sent >= max_events:
                    break

            await asyncio.sleep(poll_ms / 1000)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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


@router.patch("/episodes/{ep_id}", tags=["episodes"])
def update_episode(ep_id: str, payload: EpisodeUpdateRequest) -> dict:
    """Update episode metadata (title, air_date only).

    Note: show/season/episode cannot be changed as they determine the ep_id
    which is used as a filesystem key.
    """
    ep_id = normalize_ep_id(ep_id)
    record = EPISODE_STORE.get(ep_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Episode {ep_id} not found")

    try:
        updated = EPISODE_STORE.update_metadata(
            ep_id=ep_id,
            title=payload.title,
            air_date=payload.air_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "ep_id": updated.ep_id,
        "show_slug": updated.show_ref,
        "season_number": updated.season_number,
        "episode_number": updated.episode_number,
        "title": updated.title,
        "air_date": updated.air_date,
        "updated_at": updated.updated_at,
    }


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


class MirrorArtifactsRequest(BaseModel):
    """Request to mirror specific artifacts from S3 to local storage."""

    artifacts: List[str] = ["faces", "identities"]  # faces.jsonl, identities.json


class MirrorArtifactsResponse(BaseModel):
    """Response from mirroring artifacts."""

    ep_id: str
    mirrored: Dict[str, bool]  # artifact -> success
    errors: Dict[str, str]  # artifact -> error message
    faces_manifest_exists: bool
    identities_manifest_exists: bool


@router.post("/episodes/{ep_id}/mirror_artifacts", tags=["episodes"])
def mirror_episode_artifacts(ep_id: str, payload: MirrorArtifactsRequest | None = None) -> MirrorArtifactsResponse:
    """Mirror faces/identities artifacts from S3 to local storage.

    This endpoint downloads manifest files (faces.jsonl, identities.json) from S3
    to the local file system, enabling clustering operations on machines that
    don't have direct S3 access or need local copies.

    Unlike /mirror which only mirrors the video, this mirrors the pipeline artifacts.
    """
    record = EPISODE_STORE.get(ep_id)
    if not record:
        raise HTTPException(status_code=404, detail="Episode not found")

    if payload is None:
        payload = MirrorArtifactsRequest()

    ensure_dirs(ep_id)
    mirrored: Dict[str, bool] = {}
    errors: Dict[str, str] = {}

    try:
        ep_ctx = episode_context_from_id(ep_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid episode id: {exc}") from exc

    # Get S3 prefixes for manifests
    from apps.api.services.storage import artifact_prefixes

    prefixes = artifact_prefixes(ep_ctx)
    manifests_prefix = prefixes.get("manifests", "")

    # Define artifact mappings: artifact_name -> (s3_key_suffix, local_path)
    manifests_dir = _manifests_dir(ep_id)
    artifact_map = {
        "faces": ("faces.jsonl", manifests_dir / "faces.jsonl"),
        "identities": ("identities.json", manifests_dir / "identities.json"),
        "tracks": ("tracks.jsonl", get_path(ep_id, "tracks")),
        "detections": ("detections.jsonl", get_path(ep_id, "detections")),
    }

    # Mirror requested artifacts
    for artifact_name in payload.artifacts:
        if artifact_name not in artifact_map:
            errors[artifact_name] = f"Unknown artifact: {artifact_name}"
            mirrored[artifact_name] = False
            continue

        s3_suffix, local_path = artifact_map[artifact_name]
        s3_key = f"{manifests_prefix}{s3_suffix}"

        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Try to download from S3
        if STORAGE.backend not in {"s3", "minio"} or STORAGE._client is None:
            # Local mode - check if file exists locally
            if local_path.exists():
                mirrored[artifact_name] = True
            else:
                errors[artifact_name] = "Storage backend is local and file doesn't exist"
                mirrored[artifact_name] = False
            continue

        try:
            # Check if object exists in S3
            STORAGE._client.head_object(Bucket=STORAGE.bucket, Key=s3_key)
            # Download the file
            STORAGE._client.download_file(STORAGE.bucket, s3_key, str(local_path))
            mirrored[artifact_name] = True
            LOGGER.info("Mirrored %s from s3://%s/%s to %s", artifact_name, STORAGE.bucket, s3_key, local_path)
        except Exception as exc:
            error_code = None
            if hasattr(exc, "response"):
                error_code = exc.response.get("Error", {}).get("Code")  # type: ignore[union-attr]
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                errors[artifact_name] = f"Not found in S3: {s3_key}"
            else:
                errors[artifact_name] = str(exc)
            mirrored[artifact_name] = False

    # Check final state of manifests
    faces_path = _faces_path(ep_id)
    identities_path = _identities_path(ep_id)

    return MirrorArtifactsResponse(
        ep_id=ep_id,
        mirrored=mirrored,
        errors=errors,
        faces_manifest_exists=faces_path.exists(),
        identities_manifest_exists=identities_path.exists(),
    )


@router.post("/episodes/{ep_id}/refresh_similarity", tags=["episodes"])
def refresh_similarity_values(ep_id: str) -> dict:
    """Recompute all similarity scores for the episode.

    This regenerates track representatives, cluster centroids, updates
    all similarity scores, and refreshes suggestions for unassigned clusters.

    Returns detailed step-by-step progress log with stats.
    """
    import time

    log_steps = []
    start_time = time.time()

    try:
        # Step 1: Load identities to count tracks and clusters (0-10%)
        step_start = time.time()
        identities_payload = _load_identities(ep_id)
        identities = identities_payload.get("identities", [])
        track_count = sum(len(i.get("track_ids", [])) for i in identities)
        cluster_count = len(identities)

        # Count assigned vs unassigned clusters
        assigned_count = sum(1 for i in identities if i.get("person_id"))
        unassigned_count = cluster_count - assigned_count

        log_steps.append({
            "step": "load_identities",
            "status": "success",
            "progress_pct": 10,
            "duration_ms": int((time.time() - step_start) * 1000),
            "cluster_count": cluster_count,
            "track_count": track_count,
            "assigned_count": assigned_count,
            "unassigned_count": unassigned_count,
            "details": [
                f"Loaded {cluster_count} clusters ({track_count} tracks)",
                f"Assigned: {assigned_count}, Unassigned: {unassigned_count}",
            ],
        })

        # Step 2: Generate track reps and centroids (10-50%)
        step_start = time.time()
        track_reps_count = 0
        centroids_count = 0
        try:
            from apps.api.services.track_reps import (
                generate_track_reps_and_centroids,
            )
            result = generate_track_reps_and_centroids(ep_id)
            track_reps_count = result.get("tracks_processed", 0)
            centroids_count = result.get("centroids_computed", 0)
            tracks_with_reps = result.get("tracks_with_reps", 0)
            tracks_skipped = result.get("tracks_skipped", 0)

            details = [
                f"Processed {track_reps_count} tracks",
                f"Computed {centroids_count} cluster centroids",
            ]
            if tracks_with_reps:
                details.append(f"Tracks with valid reps: {tracks_with_reps}")
            if tracks_skipped:
                details.append(f"Tracks skipped (no embeddings): {tracks_skipped}")

            log_steps.append({
                "step": "generate_track_reps",
                "status": "success",
                "progress_pct": 50,
                "duration_ms": int((time.time() - step_start) * 1000),
                "track_reps_count": track_reps_count,
                "centroids_count": centroids_count,
                "details": details,
            })
        except Exception as exc:
            log_steps.append({
                "step": "generate_track_reps",
                "status": "error",
                "progress_pct": 50,
                "duration_ms": int((time.time() - step_start) * 1000),
                "error": str(exc),
                "details": [f"Error: {exc}"],
            })
            LOGGER.error("Track rep regeneration failed for %s: %s", ep_id, exc)

        # Step 3: Update people prototypes (50-80%)
        step_start = time.time()
        people_updated = 0
        people_details = []
        try:
            from apps.api.services.people import PeopleService, l2_normalize
            import numpy as np
            from apps.api.services.track_reps import load_cluster_centroids

            ep_ctx = episode_context_from_id(ep_id)
            show_id = ep_ctx.show_slug.upper()
            people_service = PeopleService()
            people = people_service.list_people(show_id)

            if people:
                # Find all person_ids that have clusters in this episode
                touched_person_ids = set()
                for identity in identities:
                    if identity.get("person_id"):
                        touched_person_ids.add(identity["person_id"])

                # Update prototypes
                people_by_id = {p.get("person_id"): p for p in people if p.get("person_id")}
                try:
                    centroids_data = load_cluster_centroids(ep_id)
                except Exception:
                    centroids_data = {}

                for person_id in touched_person_ids:
                    person = people_by_id.get(person_id)
                    if not person:
                        continue
                    person_name = person.get("name") or person.get("display_name") or person_id
                    cluster_refs = person.get("cluster_ids") or []
                    vectors = []
                    ep_clusters = 0
                    for cluster_ref in cluster_refs:
                        if not isinstance(cluster_ref, str) or ":" not in cluster_ref:
                            continue
                        ep_slug, cluster_id = cluster_ref.split(":", 1)
                        if ep_slug == ep_id:
                            ep_clusters += 1
                            centroid_data = centroids_data.get(cluster_id, {})
                            if centroid_data and centroid_data.get("centroid"):
                                vectors.append(np.array(centroid_data["centroid"], dtype=np.float32))
                    if vectors:
                        stacked = np.stack(vectors, axis=0)
                        proto = l2_normalize(np.mean(stacked, axis=0)).tolist()
                        people_service.update_person(show_id, person_id, prototype=proto)
                        people_updated += 1
                        people_details.append(f"  • {person_name}: updated from {len(vectors)} centroid(s)")

            details = [f"Updated {people_updated} people prototypes"]
            if people_details:
                details.extend(people_details[:10])  # Limit to first 10 for brevity
                if len(people_details) > 10:
                    details.append(f"  ... and {len(people_details) - 10} more")

            log_steps.append({
                "step": "update_prototypes",
                "status": "success",
                "progress_pct": 80,
                "duration_ms": int((time.time() - step_start) * 1000),
                "people_updated": people_updated,
                "people_total": len(touched_person_ids) if people else 0,
                "details": details,
            })
        except Exception as exc:
            log_steps.append({
                "step": "update_prototypes",
                "status": "error",
                "progress_pct": 80,
                "duration_ms": int((time.time() - step_start) * 1000),
                "error": str(exc),
                "details": [f"Error: {exc}"],
            })
            LOGGER.warning("People prototype update failed for %s: %s", ep_id, exc)

        # Step 4: Refresh suggestions for unassigned clusters (80-100%)
        step_start = time.time()
        suggestions_count = 0
        suggestions_details = []
        suggestions_list = []
        try:
            from apps.api.services.grouping import GroupingService
            grouping_service = GroupingService()
            suggestions_result = grouping_service.suggest_from_assigned_clusters(ep_id)
            suggestions_list = suggestions_result.get("suggestions", [])
            suggestions_count = len(suggestions_list)

            # Build lookup for person names
            person_names = {}
            if people:
                for p in people:
                    pid = p.get("person_id")
                    if pid:
                        person_names[pid] = p.get("name") or p.get("display_name") or pid

            details = [f"Generated {suggestions_count} suggestions for unassigned clusters"]
            for suggestion in suggestions_list[:10]:  # Show first 10 suggestions
                cluster_id = suggestion.get("cluster_id", "?")
                suggested_person = suggestion.get("suggested_person_id", "?")
                distance = suggestion.get("distance", 0)
                person_name = person_names.get(suggested_person, suggested_person)
                confidence = max(0, min(100, int((1 - distance) * 100)))
                details.append(f"  • Cluster {cluster_id} → {person_name} ({confidence}% confidence)")
                suggestions_details.append({
                    "cluster_id": cluster_id,
                    "suggested_person_id": suggested_person,
                    "suggested_person_name": person_name,
                    "distance": round(distance, 4),
                    "confidence_pct": confidence,
                })

            if len(suggestions_list) > 10:
                details.append(f"  ... and {len(suggestions_list) - 10} more suggestions")

            log_steps.append({
                "step": "refresh_suggestions",
                "status": "success",
                "progress_pct": 100,
                "duration_ms": int((time.time() - step_start) * 1000),
                "suggestions_count": suggestions_count,
                "unassigned_clusters": unassigned_count,
                "details": details,
                "suggestions": suggestions_details[:20],  # Include first 20 in response
            })
        except FileNotFoundError:
            log_steps.append({
                "step": "refresh_suggestions",
                "status": "skipped",
                "progress_pct": 100,
                "duration_ms": int((time.time() - step_start) * 1000),
                "message": "No identities file found",
                "details": ["Skipped: No identities file found"],
            })
        except Exception as exc:
            log_steps.append({
                "step": "refresh_suggestions",
                "status": "error",
                "progress_pct": 100,
                "duration_ms": int((time.time() - step_start) * 1000),
                "error": str(exc),
                "details": [f"Error: {exc}"],
            })
            LOGGER.warning("Suggestions refresh failed for %s: %s", ep_id, exc)

        total_duration = int((time.time() - start_time) * 1000)

        return {
            "status": "success",
            "ep_id": ep_id,
            "message": "Similarity values refreshed successfully",
            "log": {
                "steps": log_steps,
                "total_duration_ms": total_duration,
            },
            "summary": {
                "clusters": cluster_count,
                "tracks": track_count,
                "assigned_clusters": assigned_count,
                "unassigned_clusters": unassigned_count,
                "centroids_computed": centroids_count,
                "people_updated": people_updated,
                "suggestions_generated": suggestions_count,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh similarity values: {str(e)}")


# Lazy Celery availability check
_celery_available_cache = None


def _check_celery_available() -> bool:
    """Check if Celery/Redis are available for async jobs."""
    global _celery_available_cache
    if _celery_available_cache is None:
        try:
            import redis
            from apps.api.config import REDIS_URL
            r = redis.from_url(REDIS_URL, socket_timeout=1)
            r.ping()
            _celery_available_cache = True
        except Exception as e:
            LOGGER.warning(f"Celery/Redis not available: {e}")
            _celery_available_cache = False
    return _celery_available_cache


class RefreshSimilarityRequest(BaseModel):
    """Request model for similarity refresh with optional execution mode."""
    execution_mode: Optional[Literal["redis", "local"]] = Field(
        "redis",
        description="Execution mode: 'redis' enqueues job via Celery, 'local' runs synchronously in-process"
    )


@router.post("/episodes/{ep_id}/refresh_similarity_async", status_code=202, tags=["episodes"])
def refresh_similarity_async(ep_id: str, body: RefreshSimilarityRequest = Body(default=RefreshSimilarityRequest())) -> dict:
    """Enqueue similarity refresh as background job (non-blocking).

    Execution Mode:
        - execution_mode="redis" (default): Enqueues job via Celery, returns 202 with job_id
        - execution_mode="local": Runs job synchronously in-process, returns result when done

    If Celery/Redis are unavailable in redis mode, returns an error.
    """
    execution_mode = body.execution_mode or "redis"

    # Handle local execution mode
    if execution_mode == "local":
        LOGGER.info(f"[{ep_id}] Running local refresh_similarity")
        try:
            # Import and run the similarity refresh directly
            from apps.api.services.similarity_refresh import refresh_similarity_indexes
            result = refresh_similarity_indexes(ep_id)
            return {
                "status": "completed",
                "ep_id": ep_id,
                "async": False,
                "execution_mode": "local",
                "message": "Executed synchronously (local mode)",
                "result": result,
            }
        except Exception as e:
            LOGGER.exception(f"[{ep_id}] Local refresh_similarity failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Redis/Celery mode - check if Celery is available
    if not _check_celery_available():
        LOGGER.info(f"[{ep_id}] Celery unavailable for refresh_similarity")
        return {
            "status": "error",
            "ep_id": ep_id,
            "async": False,
            "execution_mode": "redis",
            "message": "Background jobs unavailable (Celery/Redis not running). Please start background services or use execution_mode='local'.",
        }

    try:
        from apps.api.tasks import run_refresh_similarity_task, check_active_job

        # Check for active job
        active = check_active_job(ep_id, "refresh_similarity")
        if active:
            raise HTTPException(
                status_code=409,
                detail=f"Refresh job already in progress: {active}",
            )

        # Enqueue the task
        task = run_refresh_similarity_task.delay(episode_id=ep_id)

        return {
            "job_id": task.id,
            "status": "queued",
            "ep_id": ep_id,
            "async": True,
            "execution_mode": "redis",
            "message": "Refresh similarity job queued. Poll /celery_jobs/{job_id} for status.",
        }
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception(f"[{ep_id}] Failed to enqueue refresh_similarity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enqueue job: {str(e)}")


@router.get("/episodes/{ep_id}/recover_noise_tracks/preview", tags=["episodes"])
def recover_noise_tracks_preview(
    ep_id: str,
    frame_window: int = Query(8, ge=1, le=30, description="Number of frames to search before/after"),
    min_similarity: float = Query(0.70, ge=0.5, le=1.0, description="Minimum cosine similarity to merge"),
) -> dict:
    """Preview what would change if recovery were run (without making changes).

    Returns counts of single-frame tracks and estimated recoverable tracks.
    """
    from apps.api.services.identities import load_faces, _episode_lock

    try:
        with _episode_lock(ep_id):
            faces = load_faces(ep_id)

        if not faces:
            return {
                "status": "success",
                "single_frame_tracks": 0,
                "estimated_recoverable": 0,
                "frame_window": frame_window,
                "min_similarity": min_similarity,
            }

        # Count faces per track
        from collections import defaultdict
        faces_by_track: dict = defaultdict(int)
        for face in faces:
            track_id = face.get("track_id")
            if track_id is not None:
                faces_by_track[track_id] += 1

        single_frame_tracks = sum(1 for count in faces_by_track.values() if count == 1)

        # Estimate recoverable (rough estimate based on typical recovery rates)
        estimated_recoverable = int(single_frame_tracks * 0.15)  # ~15% typically recoverable

        return {
            "status": "success",
            "single_frame_tracks": single_frame_tracks,
            "multi_frame_tracks": sum(1 for count in faces_by_track.values() if count > 1),
            "estimated_recoverable": estimated_recoverable,
            "frame_window": frame_window,
            "min_similarity": min_similarity,
        }
    except Exception as e:
        LOGGER.exception(f"[{ep_id}] Failed to preview recovery: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/episodes/{ep_id}/recover_noise_tracks", tags=["episodes"])
def recover_noise_tracks(
    ep_id: str,
    frame_window: int = Query(8, ge=1, le=30, description="Number of frames to search before/after"),
    min_similarity: float = Query(0.70, ge=0.5, le=1.0, description="Minimum cosine similarity to merge"),
) -> dict:
    """Recover single-frame tracks by finding similar faces in adjacent frames.

    For each track that has only 1 face (single-frame track):
    1. Searches ±frame_window frames for similar faces (cosine similarity >= min_similarity)
    2. Merges matching faces into the single-frame track
    3. Updates faces.jsonl and tracks.jsonl with new assignments

    This helps convert "noise" clusters (single-frame-only) into reviewable clusters.

    Args:
        frame_window: Number of frames to search before/after (default: 8)
        min_similarity: Minimum cosine similarity to merge faces (default: 0.70)

    Returns:
        tracks_analyzed: Number of single-frame tracks examined
        tracks_expanded: Number of tracks that were expanded
        faces_merged: Total faces added to tracks
        details: List of {track_id, original_frame, added_frames}
    """
    from apps.api.services.track_recovery import recover_single_frame_tracks

    try:
        result = recover_single_frame_tracks(ep_id, frame_window=frame_window, min_similarity=min_similarity)

        # After recovery, refresh similarity indexes and recompute centroids for affected clusters
        if result.get("tracks_expanded", 0) > 0:
            affected_track_ids = [d.get("track_id") for d in result.get("details", []) if d.get("track_id") is not None]
            _refresh_similarity_indexes(ep_id, track_ids=affected_track_ids)

            # Recompute cluster centroids since track assignments have changed
            try:
                from apps.api.services.grouping import GroupingService
                grouping_service = GroupingService()
                grouping_service.compute_cluster_centroids(ep_id)
                LOGGER.info(f"[{ep_id}] Recomputed cluster centroids after recovery")
            except Exception as centroid_err:
                LOGGER.warning(f"[{ep_id}] Failed to recompute centroids after recovery: {centroid_err}")

        return result
    except Exception as e:
        LOGGER.exception(f"[{ep_id}] Failed to recover noise tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/episodes/{ep_id}/cleanup_empty_clusters", tags=["episodes"])
def cleanup_empty_clusters_endpoint(ep_id: str) -> dict:
    """Remove clusters that have no tracks/frames.

    Clusters can become empty when:
    - All tracks are dropped from a cluster
    - All tracks are moved to another cluster
    - Merge operations leave source clusters empty

    This endpoint removes empty clusters from identities.json and updates
    people.json to remove references to the deleted clusters.

    Returns:
        removed_clusters: List of identity_ids that were removed
        people_updated: List of person_ids that had clusters removed
        identities_before: Count before cleanup
        identities_after: Count after cleanup
    """
    from apps.api.services.identities import cleanup_empty_clusters

    try:
        result = cleanup_empty_clusters(ep_id)
        return {
            "status": "success",
            "ep_id": ep_id,
            **result,
        }
    except Exception as e:
        LOGGER.exception(f"[{ep_id}] Failed to cleanup empty clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
                "person_id": identity.get("person_id"),
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
def get_cluster_track_reps(
    ep_id: str,
    cluster_id: str,
    frames_per_track: int = Query(0, ge=0, le=20, description="Number of sample frames to include per track (0=none)"),
) -> dict:
    """Get representative frames with similarity scores for all tracks in a cluster.

    If frames_per_track > 0, includes sample frames for each track for inline display.
    """
    try:
        from apps.api.services.track_reps import (
            build_cluster_track_reps,
            load_track_reps,
            load_cluster_centroids,
        )

        track_reps = load_track_reps(ep_id)
        cluster_centroids = load_cluster_centroids(ep_id)

        result = build_cluster_track_reps(ep_id, cluster_id, track_reps, cluster_centroids)

        # Load faces for sample frames if requested
        all_faces = None
        if frames_per_track > 0:
            all_faces = _load_faces(ep_id, include_skipped=False)

        # Resolve crop URLs and add sample frames
        for track in result.get("tracks", []):
            crop_key = track.get("crop_key")
            if crop_key:
                # Use existing _resolve_crop_url helper
                url = _resolve_crop_url(ep_id, crop_key, None)
                track["crop_url"] = url

            # Add sample frames if requested
            if frames_per_track > 0 and all_faces:
                track_id_str = track.get("track_id", "")
                # Parse track_id to int
                if isinstance(track_id_str, str) and track_id_str.startswith("track_"):
                    try:
                        track_id_int = int(track_id_str.replace("track_", ""))
                    except ValueError:
                        track_id_int = None
                else:
                    try:
                        track_id_int = int(track_id_str)
                    except (ValueError, TypeError):
                        track_id_int = None

                if track_id_int is not None:
                    # Get all faces for this track, sorted by frame_idx
                    track_faces = sorted(
                        [f for f in all_faces if f.get("track_id") == track_id_int],
                        key=lambda f: f.get("frame_idx", 0)
                    )
                    total_frames = len(track_faces)
                    track["frame_count"] = total_frames

                    # Sample evenly across the track
                    if total_frames <= frames_per_track:
                        sampled = track_faces
                    else:
                        step = total_frames / frames_per_track
                        indices = [int(i * step) for i in range(frames_per_track)]
                        sampled = [track_faces[i] for i in indices if i < total_frames]

                    # Build frame entries with URLs
                    sample_frames = []
                    for face in sampled:
                        frame_url = _resolve_face_media_url(ep_id, face)
                        sample_frames.append({
                            "frame_idx": face.get("frame_idx"),
                            "crop_url": frame_url,
                            "quality": face.get("quality"),
                        })
                    track["sample_frames"] = sample_frames

        return result
    except Exception as exc:
        LOGGER.error(f"Failed to load track reps for cluster {cluster_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to load track representatives: {exc}") from exc


@router.get("/episodes/{ep_id}/people/{person_id}/clusters_summary")
def get_person_clusters_summary(ep_id: str, person_id: str) -> dict:
    """Get clusters summary with track representatives for a person in an episode."""
    try:
        from apps.api.services.track_reps import (
            build_cluster_track_reps,
            load_track_reps,
            load_cluster_centroids,
        )

        # Parse episode to get show
        ep_ctx = episode_context_from_id(ep_id)
        show_slug = ep_ctx.show_slug.upper()

        # Load identities first - we need this regardless of person lookup
        identities_data = _load_identities(ep_id)
        identities_list = identities_data.get("identities", []) if isinstance(identities_data, dict) else []

        # Try to find clusters via person registry first
        episode_clusters = []

        from apps.api.services.people import PeopleService
        people_service = PeopleService()
        person = people_service.get_person(show_slug, person_id)

        if person:
            # Person exists in registry - use their cluster_ids
            cluster_ids = person.get("cluster_ids", []) if isinstance(person, dict) else []
            if not isinstance(cluster_ids, list):
                LOGGER.warning(f"cluster_ids is not a list: {type(cluster_ids)}, value: {cluster_ids}")
                cluster_ids = []

            episode_clusters = [
                cid.split(":", 1)[1] if ":" in cid else cid
                for cid in cluster_ids
                if isinstance(cid, str) and cid.startswith(f"{ep_id}:")
            ]
        else:
            # Person not in registry - find identities assigned to this person_id
            # This handles auto-clustered people that don't have person records yet
            LOGGER.info(f"Person {person_id} not in registry, checking identities for this episode")
            for ident in identities_list:
                if isinstance(ident, dict) and ident.get("person_id") == person_id:
                    identity_id = ident.get("identity_id")
                    if identity_id:
                        episode_clusters.append(identity_id)

            if not episode_clusters:
                raise HTTPException(status_code=404, detail="Person not found")

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

        # Build identity index for face counts (identities_list already loaded above)
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
            LOGGER.info(
                f"cluster_data type: {type(cluster_data)}, keys: {list(cluster_data.keys()) if isinstance(cluster_data, dict) else 'not a dict'}"
            )

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

            clusters_output.append(
                {
                    "cluster_id": cluster_id,
                    "tracks": len(cluster_data.get("tracks", [])),
                    "faces": faces_count,
                    "cohesion": cluster_data.get("cohesion"),
                    "centroid": cluster_data.get("cluster_centroid"),  # Include centroid vector
                    "track_reps": cluster_data.get("tracks", []),
                }
            )
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
    identity = next(
        (item for item in payload.get("identities", []) if item.get("identity_id") == identity_id),
        None,
    )
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
            or _resolve_thumb_url(
                ep_id,
                identity.get("rep_thumb_rel_path"),
                identity.get("rep_thumb_s3_key"),
            ),
            "rep_media_url": _resolve_face_media_url(
                ep_id,
                first_faces.get(track_ids[0]) if track_ids else None,
            )
            or _resolve_thumb_url(
                ep_id,
                identity.get("rep_thumb_rel_path"),
                identity.get("rep_thumb_s3_key"),
            ),
        },
        "tracks": tracks_payload,
    }


# ============================================================================
# METRICS ENDPOINTS
# ============================================================================


@router.get("/episodes/{ep_id}/clusters/{cluster_id}/metrics")
def get_cluster_metrics(ep_id: str, cluster_id: str) -> dict:
    """Get computed metrics for a cluster (cohesion, isolation, ambiguity, temporal, quality)."""
    try:
        return metrics_service.compute_all_cluster_metrics(ep_id, cluster_id)
    except Exception as exc:
        LOGGER.error(f"Failed to compute cluster metrics for {cluster_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to compute metrics: {exc}") from exc


@router.get("/episodes/{ep_id}/tracks/{track_id}/metrics")
def get_track_metrics(ep_id: str, track_id: int) -> dict:
    """Get computed metrics for a track (consistency, person cohesion, quality)."""
    try:
        return metrics_service.compute_all_track_metrics(ep_id, track_id)
    except Exception as exc:
        LOGGER.error(f"Failed to compute track metrics for track {track_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to compute metrics: {exc}") from exc


@router.get("/episodes/{ep_id}/identities_with_metrics")
def list_identities_with_metrics(
    ep_id: str,
    include_metrics: bool = Query(True, description="Include computed metrics"),
) -> dict:
    """List identities with optional metrics for each cluster.

    Returns standard identity listing plus metrics for each cluster:
    - cohesion: Average similarity of tracks to centroid (with min/max range)
    - isolation: Distance to nearest cluster (higher = more isolated)
    - ambiguity: Gap between 1st and 2nd best match (higher = clearer)
    - temporal_consistency: Appearance consistency over time
    - avg_quality: Average quality score
    """
    payload = _load_identities(ep_id)
    track_lookup = {int(row.get("track_id", -1)): row for row in _load_tracks(ep_id)}
    first_faces = _first_face_lookup(ep_id)

    # Pre-load metrics data if needed
    if include_metrics:
        from apps.api.services.track_reps import load_track_reps, load_cluster_centroids
        track_reps = load_track_reps(ep_id)
        cluster_centroids = load_cluster_centroids(ep_id)
    else:
        track_reps = None
        cluster_centroids = None

    identities = []
    for identity in payload.get("identities", []):
        identity_id = identity.get("identity_id")
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

        entry = {
            "identity_id": identity_id,
            "label": identity.get("label"),
            "name": identity.get("name"),
            "person_id": identity.get("person_id"),
            "track_ids": track_ids,
            "faces": faces_total,
            "rep_thumbnail_url": preview_url,
            "rep_media_url": preview_url,
        }

        # Add metrics if requested
        if include_metrics and identity_id:
            try:
                cohesion = metrics_service.compute_cluster_cohesion(
                    ep_id, identity_id, track_reps, cluster_centroids
                )
                isolation = metrics_service.compute_cluster_isolation(
                    ep_id, identity_id, cluster_centroids
                )
                ambiguity = metrics_service.compute_cluster_ambiguity(
                    ep_id, identity_id, cluster_centroids
                )
                temporal = metrics_service.compute_temporal_consistency(ep_id, cluster_id=identity_id)
                quality = metrics_service.compute_aggregate_quality(ep_id, cluster_id=identity_id)

                entry["metrics"] = {
                    "cohesion": cohesion.get("cohesion"),
                    "min_similarity": cohesion.get("min_similarity"),
                    "max_similarity": cohesion.get("max_similarity"),
                    "isolation": isolation.get("isolation"),
                    "nearest_cluster": isolation.get("nearest_cluster"),
                    "ambiguity": ambiguity.get("ambiguity"),
                    "temporal_consistency": temporal.get("temporal_consistency"),
                    "avg_quality": quality.get("avg_quality"),
                    "quality_breakdown": quality.get("quality_breakdown"),
                }
            except Exception as exc:
                LOGGER.warning(f"Failed to compute metrics for {identity_id}: {exc}")
                entry["metrics"] = None

        identities.append(entry)

    return {"identities": identities, "stats": payload.get("stats", {})}


@router.get("/episodes/{ep_id}/unclustered_tracks")
def list_unclustered_tracks(ep_id: str) -> dict:
    """List tracks that are not in any identity cluster.

    Returns tracks from tracks.jsonl that have no corresponding identity_id
    in identities.json. These are tracks that were detected but not yet
    assigned to any cluster.
    """
    # Load all tracks
    all_tracks = _load_tracks(ep_id)

    # Build lookup of track_ids that are in identities
    identities_payload = _load_identities(ep_id)
    clustered_track_ids: set[int] = set()
    for identity in identities_payload.get("identities", []):
        for raw_tid in identity.get("track_ids", []) or []:
            try:
                clustered_track_ids.add(int(raw_tid))
            except (TypeError, ValueError):
                continue

    # Find tracks not in any identity
    first_faces = _first_face_lookup(ep_id)

    # Build face count lookups from faces.jsonl for tracks with missing faces_count
    # This handles cases where tracks.jsonl wasn't properly populated during pipeline
    # Count ALL faces (including skipped) for display - users need to see total faces
    faces_by_track: Dict[int, int] = {}  # Total faces (including skipped)
    skipped_by_track: Dict[int, int] = {}  # Skipped faces count
    emb_by_track: Dict[int, int] = {}  # Faces with embeddings count
    blur_by_track: Dict[int, List[float]] = {}  # Blur scores for avg calculation
    skip_reasons_by_track: Dict[int, List[str]] = {}  # Skip reasons for diagnostics
    for face in _load_faces(ep_id, include_skipped=True):  # Include ALL faces
        tid = face.get("track_id")
        if tid is not None:
            try:
                tid_int = int(tid)
                faces_by_track[tid_int] = faces_by_track.get(tid_int, 0) + 1
                # Count skipped separately
                skip_val = face.get("skip")
                if skip_val:
                    skipped_by_track[tid_int] = skipped_by_track.get(tid_int, 0) + 1
                    # Collect skip reasons for diagnostics
                    if tid_int not in skip_reasons_by_track:
                        skip_reasons_by_track[tid_int] = []
                    skip_reasons_by_track[tid_int].append(str(skip_val))
                # Count faces with embeddings
                if face.get("embedding") or face.get("has_embedding"):
                    emb_by_track[tid_int] = emb_by_track.get(tid_int, 0) + 1
                # Collect blur scores
                blur = face.get("blur")
                if blur is not None:
                    if tid_int not in blur_by_track:
                        blur_by_track[tid_int] = []
                    blur_by_track[tid_int].append(float(blur))
            except (TypeError, ValueError):
                continue

    unclustered = []
    for track in all_tracks:
        try:
            track_id = int(track.get("track_id", -1))
        except (TypeError, ValueError):
            continue

        if track_id < 0 or track_id in clustered_track_ids:
            continue

        # Build unclustered track entry
        # Fall back to counting faces from faces.jsonl if tracks.jsonl has no count
        faces_count = track.get("faces_count")
        if faces_count is None or faces_count == 0:
            faces_count = faces_by_track.get(track_id, 0)
        else:
            faces_count = int(faces_count)

        # Also populate skipped/embeddings from face data if tracks.jsonl doesn't have it
        faces_skipped = track.get("faces_skipped")
        if faces_skipped is None or faces_skipped == 0:
            faces_skipped = skipped_by_track.get(track_id, 0)
        else:
            faces_skipped = int(faces_skipped)

        faces_with_embeddings = track.get("faces_with_embeddings")
        if faces_with_embeddings is None or faces_with_embeddings == 0:
            faces_with_embeddings = emb_by_track.get(track_id, 0)
        else:
            faces_with_embeddings = int(faces_with_embeddings)

        # Determine singleton risk based on track properties
        if faces_count == 1:
            risk = "HIGH"
        elif faces_count <= 3:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        # Get thumbnail URL
        preview_url = _resolve_face_media_url(ep_id, first_faces.get(track_id))
        if not preview_url:
            preview_url = _resolve_thumb_url(
                ep_id,
                track.get("rep_thumb_rel_path"),
                track.get("rep_thumb_s3_key"),
            )

        # Calculate blur stats for this track
        blur_scores = blur_by_track.get(track_id, [])
        avg_blur = sum(blur_scores) / len(blur_scores) if blur_scores else None
        max_blur = max(blur_scores) if blur_scores else None

        # Determine skip reason with diagnostic details
        skip_reason = None
        skip_detail = None
        if faces_with_embeddings == 0:
            # No embeddings - check why faces were skipped
            track_skip_reasons = skip_reasons_by_track.get(track_id, [])
            if track_skip_reasons:
                # Parse skip reasons to find most common
                reason_counts: Dict[str, int] = {}
                for sr in track_skip_reasons:
                    # Extract base reason (e.g., "blurry" from "blurry:10.8")
                    base_reason = sr.split(":")[0] if ":" in sr else sr
                    reason_counts[base_reason] = reason_counts.get(base_reason, 0) + 1
                primary_reason = max(reason_counts.items(), key=lambda x: x[1])[0]
                skip_reason = f"no_embedding:{primary_reason}"
                skip_detail = f"All {faces_count} face(s) skipped. Primary reason: {primary_reason}. " \
                              f"Max blur: {max_blur}, threshold: 18.0"
            else:
                skip_reason = "no_embedding:unknown"
                skip_detail = f"Track has {faces_count} face(s) but no embeddings generated"
        else:
            # Has embeddings but not clustered - likely outlier or clustering not run
            skip_reason = "not_clustered"
            skip_detail = f"Track has {faces_with_embeddings} embedding(s) but wasn't assigned to a cluster. " \
                          "May be outlier (below min_identity_sim threshold) or clustering not run."

        unclustered.append({
            "track_id": track_id,
            "faces": faces_count,
            "faces_skipped": faces_skipped,
            "faces_with_embeddings": faces_with_embeddings,
            "singleton_risk": risk,
            "rep_thumbnail_url": preview_url,
            "avg_blur": round(avg_blur, 1) if avg_blur is not None else None,
            "max_blur": round(max_blur, 1) if max_blur is not None else None,
            "skip_reason": skip_reason,
            "skip_detail": skip_detail,
        })

    # Include config thresholds for reference (from config/pipeline/)
    config_thresholds = {
        "embedding": {
            "min_blur_score": 18.0,  # faces_embed_sampling.yaml
            "min_confidence": 0.45,
        },
        "clustering": {
            "cluster_thresh": 0.52,  # clustering.yaml
            "min_identity_sim": 0.45,
            "min_cluster_size": 1,
        },
    }

    return {
        "unclustered_tracks": unclustered,
        "count": len(unclustered),
        "total_tracks": len(all_tracks),
        "config_thresholds": config_thresholds,
    }


# --- Singleton Analysis (Batch) ---


class SingletonAnalysisRequest(BaseModel):
    """Request for batch singleton analysis."""

    include_archive: bool = Field(
        default=False,
        description="Compare singletons against archived item centroids",
    )
    min_similarity: float = Field(
        default=0.30,
        description="Minimum similarity threshold for suggestions",
        ge=0.0,
        le=1.0,
    )
    top_k: int = Field(
        default=3,
        description="Maximum suggestions per singleton",
        ge=1,
        le=10,
    )


@router.post("/episodes/{ep_id}/singleton_analysis")
def analyze_singletons_batch(
    ep_id: str,
    request: SingletonAnalysisRequest = Body(default=SingletonAnalysisRequest()),
) -> dict:
    """Batch analyze all singleton clusters against assigned track embeddings.

    This performs efficient batch comparison of ALL singleton clusters (clusters with
    exactly 1 track) against ALL assigned track embeddings from this episode.
    Results are grouped by suggested person for efficient UI review.

    Returns:
        - person_groups: List of person groups, each containing singletons suggested for that person
        - archive_matches: Singletons similar to archived items (if include_archive=True)
        - unmatched: Singletons with no good suggestions
        - stats: Summary statistics
    """
    from apps.api.services.grouping import GroupingService

    ep_id = normalize_ep_id(ep_id)
    grouping_service = GroupingService()

    try:
        result = grouping_service.analyze_singletons_batch(
            ep_id=ep_id,
            include_archive=request.include_archive,
            min_similarity=request.min_similarity,
            top_k=request.top_k,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/episodes/{ep_id}/singletons/{track_id}/nearby_suggestions")
def get_singleton_nearby_suggestions(
    ep_id: str,
    track_id: int,
    time_window: float = Query(10.0, description="Time window in seconds to search for nearby tracks"),
    top_k: int = Query(3, description="Number of top suggestions to return"),
) -> dict:
    """Suggest cast members based on nearby assigned tracks.

    This finds tracks within the specified time window that are already assigned
    to cast members, compares their embeddings to the singleton, and suggests
    the most similar cast member.

    Returns:
        - suggestions: List of cast member suggestions with similarity scores
        - nearby_tracks: List of assigned tracks found nearby
        - singleton_info: Info about the singleton track being analyzed
    """
    import numpy as np

    # Load all faces with embeddings
    all_faces = list(_load_faces(ep_id, include_skipped=True))

    # Get faces for the target singleton track
    singleton_faces = [f for f in all_faces if int(f.get("track_id", -1)) == track_id]
    if not singleton_faces:
        return {"error": "Track not found", "suggestions": [], "nearby_tracks": []}

    # Get singleton's timestamp range and embeddings
    singleton_timestamps = [f.get("ts", 0) for f in singleton_faces if f.get("ts") is not None]
    singleton_embeddings = [f.get("embedding") for f in singleton_faces if f.get("embedding")]

    if not singleton_timestamps:
        return {"error": "Track has no timestamps", "suggestions": [], "nearby_tracks": []}

    min_ts = min(singleton_timestamps)
    max_ts = max(singleton_timestamps)
    search_start = max(0, min_ts - time_window)
    search_end = max_ts + time_window

    # Compute singleton's mean embedding
    if not singleton_embeddings:
        return {"error": "Track has no embeddings", "suggestions": [], "nearby_tracks": []}

    singleton_mean = np.mean(singleton_embeddings, axis=0)
    singleton_mean = singleton_mean / (np.linalg.norm(singleton_mean) + 1e-8)

    # Load identities to find which tracks are assigned to cast members
    identities_path = _identities_path(ep_id)
    if not identities_path.exists():
        return {"error": "No identities found", "suggestions": [], "nearby_tracks": []}

    identities_data = json.loads(identities_path.read_text(encoding="utf-8"))
    identities = identities_data.get("identities", [])

    # Build track_id -> (identity_id, person_id, name) lookup
    track_to_assignment: Dict[int, Dict[str, Any]] = {}
    for identity in identities:
        person_id = identity.get("person_id")
        name = identity.get("name")
        if not person_id and not name:
            continue  # Not assigned
        for tid in identity.get("track_ids", []):
            try:
                track_to_assignment[int(tid)] = {
                    "identity_id": identity.get("identity_id"),
                    "person_id": person_id,
                    "name": name,
                }
            except (TypeError, ValueError):
                continue

    # Find nearby faces from assigned tracks
    nearby_faces_by_track: Dict[int, List[Dict]] = {}
    for face in all_faces:
        face_tid = face.get("track_id")
        face_ts = face.get("ts")
        face_emb = face.get("embedding")

        if face_tid is None or face_ts is None or face_emb is None:
            continue

        try:
            face_tid_int = int(face_tid)
        except (TypeError, ValueError):
            continue

        # Skip the singleton itself
        if face_tid_int == track_id:
            continue

        # Check if within time window
        if not (search_start <= face_ts <= search_end):
            continue

        # Check if track is assigned
        if face_tid_int not in track_to_assignment:
            continue

        # Collect this face
        if face_tid_int not in nearby_faces_by_track:
            nearby_faces_by_track[face_tid_int] = []
        nearby_faces_by_track[face_tid_int].append({
            "ts": face_ts,
            "embedding": face_emb,
        })

    if not nearby_faces_by_track:
        return {
            "suggestions": [],
            "nearby_tracks": [],
            "singleton_info": {
                "track_id": track_id,
                "timestamp_range": [min_ts, max_ts],
                "faces_count": len(singleton_faces),
            },
            "message": "No assigned tracks found within time window",
        }

    # Compute similarity to each nearby assigned track
    track_similarities: List[Dict[str, Any]] = []
    for nearby_tid, faces in nearby_faces_by_track.items():
        # Compute mean embedding for nearby track
        embeddings = [f["embedding"] for f in faces]
        nearby_mean = np.mean(embeddings, axis=0)
        nearby_mean = nearby_mean / (np.linalg.norm(nearby_mean) + 1e-8)

        # Cosine similarity
        similarity = float(np.dot(singleton_mean, nearby_mean))

        assignment = track_to_assignment[nearby_tid]
        track_similarities.append({
            "track_id": nearby_tid,
            "similarity": similarity,
            "faces_count": len(faces),
            "min_ts": min(f["ts"] for f in faces),
            "max_ts": max(f["ts"] for f in faces),
            "identity_id": assignment.get("identity_id"),
            "person_id": assignment.get("person_id"),
            "name": assignment.get("name"),
        })

    # Sort by similarity (highest first)
    track_similarities.sort(key=lambda x: x["similarity"], reverse=True)

    # Aggregate by cast member (person_id or name)
    cast_scores: Dict[str, Dict[str, Any]] = {}
    for ts in track_similarities:
        key = ts.get("person_id") or ts.get("name") or ts.get("identity_id")
        if not key:
            continue
        if key not in cast_scores:
            cast_scores[key] = {
                "person_id": ts.get("person_id"),
                "name": ts.get("name"),
                "best_similarity": ts["similarity"],
                "track_count": 0,
                "total_faces": 0,
            }
        cast_scores[key]["track_count"] += 1
        cast_scores[key]["total_faces"] += ts["faces_count"]
        # Keep best similarity
        if ts["similarity"] > cast_scores[key]["best_similarity"]:
            cast_scores[key]["best_similarity"] = ts["similarity"]

    # Convert to sorted list
    suggestions = sorted(
        cast_scores.values(),
        key=lambda x: x["best_similarity"],
        reverse=True
    )[:top_k]

    # Add confidence level
    for s in suggestions:
        sim = s["best_similarity"]
        if sim >= 0.70:
            s["confidence"] = "high"
        elif sim >= 0.55:
            s["confidence"] = "medium"
        else:
            s["confidence"] = "low"

    return {
        "suggestions": suggestions,
        "nearby_tracks": track_similarities[:10],  # Top 10 nearby tracks
        "singleton_info": {
            "track_id": track_id,
            "timestamp_range": [min_ts, max_ts],
            "faces_count": len(singleton_faces),
            "embeddings_count": len(singleton_embeddings),
        },
        "search_window": {
            "start": search_start,
            "end": search_end,
            "window_seconds": time_window,
        },
    }


@router.post("/episodes/{ep_id}/generate_singleton_suggestions")
def generate_singleton_suggestions(
    ep_id: str,
    top_k: int = Query(3, description="Number of top suggestions per singleton"),
) -> dict:
    """Generate cast suggestions for all singletons by comparing against assigned tracks.

    This searches ALL assigned tracks/clusters globally (not time-limited) and finds
    the most similar cast members for each singleton based on embedding similarity.

    Returns:
        - suggestions: Dict mapping track_id -> list of cast suggestions
        - stats: Summary statistics
        - errors: List of tracks that couldn't be processed
    """
    import numpy as np

    # Load all faces with embeddings
    all_faces = list(_load_faces(ep_id, include_skipped=True))

    # Count faces with embeddings
    faces_with_emb = [f for f in all_faces if f.get("embedding")]
    if not faces_with_emb:
        return {
            "error": "No faces have embeddings. Run 'faces_embed' job first.",
            "suggestions": {},
            "stats": {"total_faces": len(all_faces), "faces_with_embeddings": 0},
            "action_required": "faces_embed",
        }

    # Load identities to find assigned tracks
    identities_path = _identities_path(ep_id)
    if not identities_path.exists():
        return {"error": "No identities found", "suggestions": {}, "stats": {}}

    identities_data = json.loads(identities_path.read_text(encoding="utf-8"))
    identities = identities_data.get("identities", [])

    # Build track_id -> assignment lookup for assigned tracks
    assigned_track_ids: set = set()
    track_to_cast: Dict[int, Dict[str, Any]] = {}
    singleton_track_ids: set = set()  # Tracks in singleton clusters (1 track)
    unassigned_track_ids: set = set()  # Tracks in unassigned clusters

    for identity in identities:
        person_id = identity.get("person_id")
        name = identity.get("name")
        track_ids = identity.get("track_ids", [])

        for tid in track_ids:
            try:
                tid_int = int(tid)
                if person_id or name:
                    # This track is assigned
                    assigned_track_ids.add(tid_int)
                    track_to_cast[tid_int] = {
                        "identity_id": identity.get("identity_id"),
                        "person_id": person_id,
                        "name": name,
                        "cast_id": person_id,  # For quick assign
                    }
                else:
                    # Unassigned track
                    unassigned_track_ids.add(tid_int)
                    if len(track_ids) == 1:
                        singleton_track_ids.add(tid_int)
            except (TypeError, ValueError):
                continue

    # Load unclustered tracks
    all_tracks = list(_load_tracks(ep_id))
    clustered_track_ids = assigned_track_ids | unassigned_track_ids
    unclustered_track_ids = set()
    for t in all_tracks:
        try:
            tid = int(t.get("track_id", -1))
            if tid not in clustered_track_ids:
                unclustered_track_ids.add(tid)
        except (TypeError, ValueError):
            continue

    # All singletons = singleton clusters + unclustered tracks
    all_singleton_ids = singleton_track_ids | unclustered_track_ids

    if not assigned_track_ids:
        return {
            "error": "No assigned tracks found. Assign some tracks to cast members first.",
            "suggestions": {},
            "stats": {"singletons": len(all_singleton_ids), "assigned_tracks": 0},
        }

    # Build embeddings by track
    embeddings_by_track: Dict[int, List] = {}
    for face in faces_with_emb:
        tid = face.get("track_id")
        emb = face.get("embedding")
        if tid is not None and emb is not None:
            try:
                tid_int = int(tid)
                if tid_int not in embeddings_by_track:
                    embeddings_by_track[tid_int] = []
                embeddings_by_track[tid_int].append(emb)
            except (TypeError, ValueError):
                continue

    # Compute mean embedding for each assigned track
    assigned_embeddings: Dict[int, np.ndarray] = {}
    for tid in assigned_track_ids:
        if tid in embeddings_by_track:
            embs = embeddings_by_track[tid]
            mean_emb = np.mean(embs, axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
            assigned_embeddings[tid] = mean_emb

    if not assigned_embeddings:
        return {
            "error": "Assigned tracks have no embeddings. Run 'faces_embed' job first.",
            "suggestions": {},
            "stats": {"assigned_tracks": len(assigned_track_ids), "with_embeddings": 0},
            "action_required": "faces_embed",
        }

    # Generate suggestions for each singleton
    suggestions: Dict[str, List[Dict]] = {}
    errors: List[Dict] = []
    processed = 0

    for singleton_tid in all_singleton_ids:
        # Get singleton embeddings
        if singleton_tid not in embeddings_by_track:
            errors.append({"track_id": singleton_tid, "error": "no_embeddings"})
            continue

        singleton_embs = embeddings_by_track[singleton_tid]
        singleton_mean = np.mean(singleton_embs, axis=0)
        singleton_mean = singleton_mean / (np.linalg.norm(singleton_mean) + 1e-8)

        # Compare against all assigned tracks
        similarities: List[Dict] = []
        for assigned_tid, assigned_emb in assigned_embeddings.items():
            sim = float(np.dot(singleton_mean, assigned_emb))
            cast_info = track_to_cast[assigned_tid]
            similarities.append({
                "track_id": assigned_tid,
                "similarity": sim,
                "person_id": cast_info.get("person_id"),
                "name": cast_info.get("name"),
                "cast_id": cast_info.get("cast_id"),
                "identity_id": cast_info.get("identity_id"),
            })

        # Sort by similarity and group by cast member
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # Aggregate by cast (person_id or name)
        cast_scores: Dict[str, Dict] = {}
        for s in similarities:
            key = s.get("person_id") or s.get("name")
            if not key:
                continue
            if key not in cast_scores:
                cast_scores[key] = {
                    "person_id": s.get("person_id"),
                    "name": s.get("name"),
                    "cast_id": s.get("cast_id"),
                    "best_similarity": s["similarity"],
                    "track_count": 0,
                    "best_track_id": s["track_id"],
                }
            cast_scores[key]["track_count"] += 1
            if s["similarity"] > cast_scores[key]["best_similarity"]:
                cast_scores[key]["best_similarity"] = s["similarity"]
                cast_scores[key]["best_track_id"] = s["track_id"]

        # Sort by best similarity and take top_k
        top_suggestions = sorted(
            cast_scores.values(),
            key=lambda x: x["best_similarity"],
            reverse=True
        )[:top_k]

        # Add confidence levels
        for sug in top_suggestions:
            sim = sug["best_similarity"]
            if sim >= 0.70:
                sug["confidence"] = "high"
            elif sim >= 0.55:
                sug["confidence"] = "medium"
            else:
                sug["confidence"] = "low"

        suggestions[str(singleton_tid)] = top_suggestions
        processed += 1

    return {
        "suggestions": suggestions,
        "stats": {
            "total_singletons": len(all_singleton_ids),
            "processed": processed,
            "with_suggestions": len([s for s in suggestions.values() if s]),
            "errors": len(errors),
            "assigned_tracks_with_embeddings": len(assigned_embeddings),
        },
        "errors": errors[:20],  # Limit error list
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
                "similarity": row.get("similarity"),  # Include frame similarity score
            }
        )
    track_row = next(
        (row for row in _load_tracks(ep_id) if int(row.get("track_id", -1)) == track_id),
        None,
    )
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
    include_skipped: bool = Query(False, description="Include faces marked as skipped"),
) -> Dict[str, Any]:
    return _list_track_frame_media(ep_id, track_id, sample, page, page_size, include_skipped)


@router.get("/episodes/{ep_id}/tracks/{track_id}/integrity")
def track_integrity(ep_id: str, track_id: int) -> Dict[str, Any]:
    """Check integrity of a track's faces manifest vs crops on disk.

    Returns counts for all faces (including skipped) to properly detect
    cases where all faces were auto-skipped due to quality filters.
    """
    # Load ALL faces including skipped to get accurate counts
    all_faces = _load_faces(ep_id, include_skipped=True)
    track_faces = [f for f in all_faces if f.get("track_id") == track_id]

    total_faces = len(track_faces)
    skipped_faces = sum(1 for f in track_faces if f.get("skip"))
    active_faces = total_faces - skipped_faces

    ctx, _ = _require_episode_context(ep_id)
    crops = _count_track_crops(ctx, track_id)

    # Track is OK if it has ANY faces (even if all skipped) and crops exist
    ok = crops >= total_faces > 0

    return {
        "track_id": track_id,
        "faces_manifest": total_faces,
        "faces_active": active_faces,
        "faces_skipped": skipped_faces,
        "crops_files": crops,
        "ok": ok,
        "all_skipped": skipped_faces == total_faces and total_faces > 0,
    }


@router.post("/episodes/{ep_id}/faces/{face_id}/unskip")
def unskip_face(ep_id: str, face_id: str) -> Dict[str, Any]:
    """Remove skip flag from a face, making it active again.

    This allows manual override of auto-skip quality filters for faces
    that were incorrectly marked as low quality.
    """
    faces = _load_faces(ep_id, include_skipped=True)
    updated = False
    for face in faces:
        if face.get("face_id") == face_id:
            if "skip" in face:
                del face["skip"]
                updated = True
                break
            else:
                # Face exists but wasn't skipped
                return {"status": "already_active", "face_id": face_id}

    if not updated:
        raise HTTPException(status_code=404, detail="face_not_found")

    _write_faces(ep_id, faces)
    return {"status": "unskipped", "face_id": face_id}


@router.post("/episodes/{ep_id}/tracks/{track_id}/unskip_all")
def unskip_all_track_faces(ep_id: str, track_id: int) -> Dict[str, Any]:
    """Remove skip flag from all faces in a track.

    Convenience endpoint to unskip all faces in a track at once,
    useful when the entire track was incorrectly marked as low quality.
    """
    faces = _load_faces(ep_id, include_skipped=True)
    unskipped_count = 0
    for face in faces:
        if face.get("track_id") == track_id and "skip" in face:
            del face["skip"]
            unskipped_count += 1

    if unskipped_count == 0:
        return {"status": "no_skipped_faces", "track_id": track_id, "unskipped": 0}

    _write_faces(ep_id, faces)
    return {"status": "unskipped", "track_id": track_id, "unskipped": unskipped_count}


class ForceEmbedRequest(BaseModel):
    """Request body for force embed track endpoint."""

    quality_profile: str = "inclusive"  # inclusive, bypass, balanced, strict
    recompute_centroid: bool = True


@router.post("/episodes/{ep_id}/tracks/{track_id}/force_embed")
def force_embed_track(ep_id: str, track_id: int, body: ForceEmbedRequest = None) -> Dict[str, Any]:
    """Force embed low-quality faces in a track (Quality Rescue).

    This endpoint enables "quality rescue" for tracks where all faces were skipped
    by the quality gate (blurry faces). It:

    1. Removes skip flags from all faces in the track
    2. Marks faces as "rescued" with the quality profile used
    3. Optionally triggers centroid recomputation for the cluster

    After calling this endpoint, run the faces_embed job to generate embeddings
    for the newly unskipped faces.

    Args:
        ep_id: Episode identifier
        track_id: Track ID to force embed
        body: Optional request body with quality_profile and recompute_centroid flags

    Returns:
        Status with count of rescued faces and next steps
    """
    from apps.api.config import get_quality_profile, QUALITY_PROFILES

    body = body or ForceEmbedRequest()

    # Validate quality profile
    if body.quality_profile not in QUALITY_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid quality_profile. Must be one of: {list(QUALITY_PROFILES.keys())}",
        )

    profile = get_quality_profile(body.quality_profile)

    # Load all faces including skipped
    faces = _load_faces(ep_id, include_skipped=True)
    track_faces = [f for f in faces if f.get("track_id") == track_id]

    if not track_faces:
        raise HTTPException(status_code=404, detail="track_not_found")

    # Collect stats before modification
    total_faces = len(track_faces)
    skipped_faces = [f for f in track_faces if f.get("skip")]
    faces_with_embeddings = [f for f in track_faces if f.get("embedding")]

    if not skipped_faces:
        return {
            "status": "no_skipped_faces",
            "track_id": track_id,
            "total_faces": total_faces,
            "faces_with_embeddings": len(faces_with_embeddings),
            "message": "Track has no skipped faces to rescue",
        }

    # Collect skip reasons for reporting
    skip_reasons = [f.get("skip") for f in skipped_faces if f.get("skip")]

    # Remove skip flags and mark as rescued
    rescued_count = 0
    for face in faces:
        if face.get("track_id") == track_id and "skip" in face:
            original_skip = face.pop("skip")
            face["rescued"] = {
                "original_skip_reason": original_skip,
                "quality_profile": body.quality_profile,
                "rescued_at": datetime.utcnow().isoformat() + "Z",
            }
            rescued_count += 1

    _write_faces(ep_id, faces)

    # Determine if cluster centroid should be recomputed
    centroid_status = None
    if body.recompute_centroid and faces_with_embeddings:
        # Try to update the cluster centroid if there are any embeddings
        try:
            from apps.api.services.grouping import GroupingService

            grouping = GroupingService()
            centroid_result = grouping.compute_cluster_centroids(ep_id)
            centroid_status = f"Recomputed {len(centroid_result.get('centroids', {}))} centroids"
        except Exception as e:
            centroid_status = f"Centroid recomputation skipped: {str(e)}"

    return {
        "status": "rescued",
        "track_id": track_id,
        "rescued_faces": rescued_count,
        "total_faces": total_faces,
        "faces_with_embeddings": len(faces_with_embeddings),
        "quality_profile": body.quality_profile,
        "quality_warning": profile.get("warning"),
        "skip_reasons": list(set(skip_reasons))[:5],  # Sample of original reasons
        "centroid_status": centroid_status,
        "next_steps": (
            "Faces have been unskipped. Run 'faces_embed' job to generate embeddings, "
            "then 'Refresh Suggestions' to get cast matches."
            if not faces_with_embeddings
            else "Faces rescued. Centroid updated. Run 'Refresh Suggestions' to get cast matches."
        ),
    }


class BatchRescueRequest(BaseModel):
    """Request body for batch rescue of quality-only clusters."""
    cluster_ids: List[str] = []  # If empty, rescue ALL quality-only clusters
    quality_profile: str = "inclusive"  # inclusive, bypass, balanced, strict
    recompute_centroids: bool = True


@router.post("/episodes/{ep_id}/rescue_quality_clusters")
def rescue_quality_clusters(ep_id: str, body: BatchRescueRequest = None) -> Dict[str, Any]:
    """Batch rescue all quality-only clusters (no embeddings due to quality gate).

    This endpoint finds all clusters where ALL faces were skipped due to quality
    (blur, contrast, confidence) and rescues them by removing skip flags.

    Args:
        ep_id: Episode identifier
        body: Optional request body with cluster_ids filter and quality_profile

    Returns:
        Summary of rescued clusters and next steps
    """
    from apps.api.config import get_quality_profile, QUALITY_PROFILES
    from apps.api.services.grouping import GroupingService

    body = body or BatchRescueRequest()

    # Validate quality profile
    if body.quality_profile not in QUALITY_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid quality_profile. Must be one of: {list(QUALITY_PROFILES.keys())}",
        )

    profile = get_quality_profile(body.quality_profile)

    # Load all faces and identities
    faces = _load_faces(ep_id, include_skipped=True)
    identities_data = _load_identities(ep_id)
    identities = identities_data.get("identities", [])

    # Build cluster -> track_ids map
    cluster_to_tracks: Dict[str, List[int]] = {}
    for identity in identities:
        cluster_id = identity.get("identity_id")
        track_ids = identity.get("track_ids", [])
        if cluster_id:
            cluster_to_tracks[cluster_id] = [int(t) for t in track_ids]

    # Build track -> faces map and identify quality-only clusters
    track_faces: Dict[int, List[Dict]] = {}
    for face in faces:
        tid = face.get("track_id")
        if tid is not None:
            track_faces.setdefault(int(tid), []).append(face)

    # Find clusters where ALL faces are skipped (quality-only)
    quality_only_clusters: List[str] = []
    for cluster_id, track_ids in cluster_to_tracks.items():
        # If specific cluster_ids provided, filter
        if body.cluster_ids and cluster_id not in body.cluster_ids:
            continue

        all_faces_for_cluster = []
        for tid in track_ids:
            all_faces_for_cluster.extend(track_faces.get(tid, []))

        if not all_faces_for_cluster:
            continue

        # Check if ALL faces are skipped
        skipped_count = sum(1 for f in all_faces_for_cluster if f.get("skip"))
        embedded_count = sum(1 for f in all_faces_for_cluster if f.get("embedding"))

        if skipped_count == len(all_faces_for_cluster) and embedded_count == 0:
            quality_only_clusters.append(cluster_id)

    if not quality_only_clusters:
        return {
            "status": "no_clusters_to_rescue",
            "message": "No quality-only clusters found (all clusters have at least one embedded face)",
            "clusters_checked": len(cluster_to_tracks),
        }

    # Rescue all faces in quality-only clusters
    rescued_clusters: List[Dict[str, Any]] = []
    total_rescued_faces = 0

    for cluster_id in quality_only_clusters:
        track_ids = cluster_to_tracks.get(cluster_id, [])
        cluster_rescued = 0
        skip_reasons: List[str] = []
        best_face_score = 0.0

        for face in faces:
            tid = face.get("track_id")
            if tid is None or int(tid) not in track_ids:
                continue
            if "skip" not in face:
                continue

            # Track skip reasons and quality scores for reporting
            skip_reason = face.get("skip", "")
            skip_reasons.append(skip_reason)

            # Get quality score from skip_data if available
            skip_data = face.get("skip_data", {})
            blur = skip_data.get("blur_score", 0)
            conf = skip_data.get("confidence", 0)
            contrast = skip_data.get("contrast", 0)
            # Composite score: higher is better
            face_score = blur * 0.5 + conf * 30 + contrast * 0.5
            best_face_score = max(best_face_score, face_score)

            # Remove skip flag and mark as rescued
            original_skip = face.pop("skip")
            face["rescued"] = {
                "original_skip_reason": original_skip,
                "quality_profile": body.quality_profile,
                "rescued_at": datetime.utcnow().isoformat() + "Z",
                "batch_rescue": True,
            }
            cluster_rescued += 1

        if cluster_rescued > 0:
            rescued_clusters.append({
                "cluster_id": cluster_id,
                "track_ids": track_ids,
                "rescued_faces": cluster_rescued,
                "skip_reasons": list(set(skip_reasons))[:3],
                "best_quality_score": round(best_face_score, 2),
            })
            total_rescued_faces += cluster_rescued

    # Write updated faces back
    _write_faces(ep_id, faces)

    # Recompute centroids if requested
    centroid_status = None
    if body.recompute_centroids and rescued_clusters:
        try:
            grouping = GroupingService()
            centroid_result = grouping.compute_cluster_centroids(ep_id)
            centroid_status = f"Recomputed {len(centroid_result.get('centroids', {}))} centroids"
        except Exception as e:
            centroid_status = f"Centroid recomputation failed: {str(e)}"

    return {
        "status": "rescued",
        "rescued_clusters": len(rescued_clusters),
        "total_rescued_faces": total_rescued_faces,
        "quality_profile": body.quality_profile,
        "quality_warning": profile.get("warning"),
        "clusters": rescued_clusters,
        "centroid_status": centroid_status,
        "next_steps": (
            "Faces have been unskipped. Run 'faces_embed' job to generate embeddings, "
            "then 'Refresh Suggestions' to get cast matches."
        ),
    }


@router.post("/episodes/{ep_id}/tracks/{track_id}/frames/move")
def move_track_frames(ep_id: str, track_id: int, body: TrackFrameMoveRequest) -> dict:
    frame_ids = sorted({int(idx) for idx in body.frame_ids or []})
    if not frame_ids:
        raise HTTPException(status_code=400, detail="frame_ids_required")
    identities_payload = _load_identities(ep_id)
    track_identity_map = _identity_lookup(identities_payload)
    source_identity_id = track_identity_map.get(track_id)
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
    touched_tracks = [track_id]
    new_track_id = result.get("new_track_id")
    if isinstance(new_track_id, int):
        touched_tracks.append(new_track_id)
    touched_identity_ids = [
        source_identity_id,
        result.get("target_identity_id"),
        body.target_identity_id,
    ]
    _refresh_similarity_indexes(ep_id, track_ids=touched_tracks, identity_ids=touched_identity_ids)
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
    identities_payload = _load_identities(ep_id)
    track_identity_map = _identity_lookup(identities_payload)
    source_identity_id = track_identity_map.get(track_id)
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
    _update_identity_stats(ep_id, identities_payload)
    identities_path = _write_identities(ep_id, identities_payload)
    _sync_manifests(ep_id, faces_path, identities_path)
    _refresh_similarity_indexes(ep_id, track_ids=[track_id], identity_ids=[source_identity_id])
    return {
        "track_id": track_id,
        "deleted": len(removed),
        "remaining": len(kept),
    }


@router.post("/episodes/{ep_id}/identities/{identity_id}/rename")
def rename_identity(ep_id: str, identity_id: str, body: IdentityRenameRequest) -> dict:
    payload = _load_identities(ep_id)
    identity = next(
        (item for item in payload.get("identities", []) if item.get("identity_id") == identity_id),
        None,
    )
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


@router.post("/episodes/{ep_id}/tracks/bulk_assign")
def bulk_assign_tracks(ep_id: str, body: BulkTrackAssignRequest) -> dict:
    """Bulk assign multiple tracks to a cast member by name.

    This creates or updates identity assignments for each track, similar to
    calling assign_track_name for each track individually but more efficient.
    """
    assigned = 0
    failed = 0
    errors: list[str] = []

    for track_id in body.track_ids:
        try:
            identity_service.assign_track_name(
                ep_id, track_id, body.name, body.show, body.cast_id
            )
            assigned += 1
        except ValueError as exc:
            failed += 1
            errors.append(f"track {track_id}: {exc}")
        except Exception as exc:
            failed += 1
            errors.append(f"track {track_id}: {exc}")

    return {
        "assigned": assigned,
        "failed": failed,
        "errors": errors if errors else None,
    }


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
    _refresh_similarity_indexes(ep_id, identity_ids=[body.source_id, body.target_id])
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
    touched_identity_ids = [
        (source_identity or {}).get("identity_id"),
        body.target_identity_id,
    ]
    _refresh_similarity_indexes(ep_id, track_ids=[track_id], identity_ids=touched_identity_ids)
    return {
        "identity_id": body.target_identity_id,
        "track_ids": target_identity["track_ids"] if target_identity else [],
    }


@router.post("/episodes/{ep_id}/faces/move_frames")
def move_faces(ep_id: str, body: FaceMoveRequest) -> dict:
    identities_payload = _load_identities(ep_id)
    track_identity_map = _identity_lookup(identities_payload)
    source_identity_id = track_identity_map.get(body.from_track_id)
    try:
        result = identity_service.move_frames(
            ep_id,
            body.from_track_id,
            body.face_ids,
            target_identity_id=body.target_identity_id,
            new_identity_name=body.new_identity_name,
            show_id=body.show_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    touched_tracks = [body.from_track_id]
    new_track_id = result.get("new_track_id")
    if isinstance(new_track_id, int):
        touched_tracks.append(new_track_id)
    touched_identity_ids = [
        source_identity_id,
        result.get("target_identity_id"),
        body.target_identity_id,
    ]
    _refresh_similarity_indexes(ep_id, track_ids=touched_tracks, identity_ids=touched_identity_ids)
    return result


@router.post("/episodes/{ep_id}/frames/{frame_idx}/overlay")
def generate_frame_overlay(ep_id: str, frame_idx: int) -> dict:
    """Generate a full-frame image with bounding boxes for all faces in that frame.

    This extracts the frame from the video, draws colored bounding boxes for each
    track present in that frame, and saves the result as an overlay image.

    Returns:
        {
            "url": "path/to/overlay.jpg",
            "frame_idx": 804,
            "tracks": [{"track_id": 1, "bbox": [x1,y1,x2,y2]}, ...]
        }
    """
    import hashlib
    import numpy as np

    # Load faces for this episode
    faces = identity_service.load_faces(ep_id)
    if not faces:
        raise HTTPException(status_code=404, detail="No faces found for episode")

    # Filter to faces in the requested frame
    frame_faces = [f for f in faces if f.get("frame_idx") == frame_idx]
    if not frame_faces:
        raise HTTPException(status_code=404, detail=f"No faces found in frame {frame_idx}")

    # Get video path
    video_path = get_path(ep_id, "video")
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    # Extract the frame from video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video file")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise HTTPException(status_code=404, detail=f"Could not read frame {frame_idx}")

        # Draw bounding boxes for each face/track
        # Use different colors for different tracks
        colors = [
            (66, 133, 244),   # Blue
            (52, 168, 83),    # Green
            (251, 188, 4),    # Yellow
            (234, 67, 53),    # Red
            (154, 0, 255),    # Purple
            (0, 188, 212),    # Cyan
            (255, 152, 0),    # Orange
            (156, 39, 176),   # Deep Purple
        ]

        track_info = []
        track_ids_seen = set()
        for face in frame_faces:
            track_id = face.get("track_id")
            bbox = face.get("bbox_xyxy")
            if track_id is None or not bbox:
                continue

            # Get color based on track_id
            color_idx = track_id % len(colors)
            color = colors[color_idx]

            # Draw bbox
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Add track ID label
                label = f"T{track_id}"
                font_scale = 0.6
                thickness = 2
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                # Draw label background
                cv2.rectangle(frame, (x1, y1 - label_h - 4), (x1 + label_w + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

                if track_id not in track_ids_seen:
                    track_info.append({"track_id": track_id, "bbox": bbox})
                    track_ids_seen.add(track_id)
            except (TypeError, ValueError):
                continue

        # Save overlay image locally first
        import tempfile
        overlay_filename = f"frame_{frame_idx:06d}_overlay.jpg"

        # Create temp file for the overlay
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        success = cv2.imwrite(str(tmp_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Failed to save overlay image")

        # Upload to S3 and get presigned URL
        try:
            # S3 key for overlays: artifacts/{ep_id}/overlays/frame_XXXXXX_overlay.jpg
            s3_key = f"artifacts/{ep_id}/overlays/{overlay_filename}"

            # Upload using STORAGE service
            if STORAGE.backend in {"s3", "minio"} and STORAGE._client is not None:
                extra_args = {"ContentType": "image/jpeg"}
                STORAGE._client.upload_file(
                    str(tmp_path),
                    STORAGE.bucket,
                    s3_key,
                    ExtraArgs=extra_args,
                )
                # Get presigned URL
                url = STORAGE.presign_get(s3_key, expires_in=3600)
                if not url:
                    raise HTTPException(status_code=500, detail="Failed to generate presigned URL")
            else:
                # Local mode - save to artifacts directory
                artifacts_dir = get_path(ep_id, "frames_root").parent / "overlays"
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                local_path = artifacts_dir / overlay_filename
                import shutil
                shutil.copy(tmp_path, local_path)
                url = str(local_path)
        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

        return {
            "url": url,
            "frame_idx": frame_idx,
            "tracks": track_info,
        }
    finally:
        cap.release()


@router.get("/episodes/{ep_id}/timestamp/{timestamp_s}/preview")
def generate_timestamp_preview(
    ep_id: str,
    timestamp_s: float,
    include_unidentified: bool = Query(True, description="Include faces without cast assignment"),
) -> dict:
    """Generate a frame preview at a specific timestamp with named bounding boxes.

    This extracts the video frame at the given timestamp, draws bounding boxes for
    all detected faces, and labels them with cast member names (if assigned).

    Args:
        ep_id: Episode identifier
        timestamp_s: Timestamp in seconds (e.g., 125.5 for 2:05.5)
        include_unidentified: If True, show boxes for faces without cast assignment

    Returns:
        {
            "url": "path/to/overlay.jpg",
            "frame_idx": 3765,
            "timestamp_s": 125.5,
            "fps": 30.0,
            "faces": [
                {"track_id": 1, "bbox": [x1,y1,x2,y2], "name": "Lisa Barlow", "cast_id": "..."},
                {"track_id": 5, "bbox": [...], "name": null, "identity_id": "cluster_42"},
                ...
            ]
        }
    """
    import hashlib
    import numpy as np
    import tempfile
    import shutil
    from apps.api.services.people import PeopleService

    # Get video path and metadata
    video_path = get_path(ep_id, "video")
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video file")

    try:
        # Get FPS to convert timestamp to frame index
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0  # Fallback to 30fps

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps if fps > 0 else 0

        # Calculate frame index from timestamp
        frame_idx = int(timestamp_s * fps)

        # Clamp to valid range
        if frame_idx < 0:
            frame_idx = 0
        if frame_idx >= total_frames:
            frame_idx = total_frames - 1

        # Load all pipeline artifacts to determine status of each face
        # 1. Detections - raw face detections from RetinaFace
        # 2. Tracks - faces grouped into tracks by ByteTrack
        # 3. Faces (harvested) - faces that passed quality gating and were embedded

        tracks_path = get_path(ep_id, "tracks")

        # Load detections using cached helper (avoids re-parsing JSONL on repeated requests)
        detections_by_frame = _get_detections_by_frame(ep_id)
        frame_detections = detections_by_frame.get(frame_idx, [])

        # Load tracks - build mapping of track_id -> frame_span
        # tracks.jsonl uses frame_span: [start_frame, end_frame] format
        track_frame_spans = {}  # track_id -> (start_frame, end_frame)
        if tracks_path.exists():
            import json
            with open(tracks_path, "r") as f:
                for line in f:
                    try:
                        track = json.loads(line)
                        tid = track.get("track_id")
                        if tid is not None:
                            frame_span = track.get("frame_span")
                            if frame_span and len(frame_span) >= 2:
                                track_frame_spans[tid] = (frame_span[0], frame_span[1])
                    except json.JSONDecodeError:
                        continue

        # Build set of track_ids that are tracked at our frame
        # A track is "tracked" if frame_idx falls within its frame_span
        tracked_at_frame = set()
        for tid, (start_f, end_f) in track_frame_spans.items():
            if start_f <= frame_idx <= end_f:
                tracked_at_frame.add(tid)

        # Load faces for this episode (harvested faces)
        faces = identity_service.load_faces(ep_id)

        # Build set of track_ids that were harvested
        harvested_track_ids = set()
        if faces:
            for face in faces:
                tid = face.get("track_id")
                if tid is not None:
                    harvested_track_ids.add(tid)

        if not faces and not frame_detections:
            raise HTTPException(status_code=404, detail="No faces or detections found for episode")

        # Build harvested faces lookup by (track_id, frame_idx) for quick access
        harvested_faces_lookup = {}
        for f in faces:
            key = (f.get("track_id"), f.get("frame_idx"))
            if key[0] is not None:
                harvested_faces_lookup[key] = f

        # Use detections as primary source - this shows ALL detected faces
        # Detections have: track_id, frame_idx, bbox, conf, etc.
        original_frame_idx = frame_idx
        gap_seconds = 0.0

        # If no detections at exact frame, find closest frame with detections
        # Uses already-loaded detections_by_frame dict (no redundant file read)
        if not frame_detections and detections_by_frame:
            all_det_frames = sorted(detections_by_frame.keys())
            if all_det_frames:
                closest_frame = min(all_det_frames, key=lambda f: abs(f - frame_idx))
                gap = abs(closest_frame - frame_idx)
                gap_seconds = gap / fps if fps > 0 else 0

                frame_detections = detections_by_frame[closest_frame]
                frame_idx = closest_frame

                LOGGER.info(
                    f"[TIMESTAMP_PREVIEW] Requested frame {original_frame_idx} (ts={timestamp_s:.2f}s), "
                    f"closest detected frame is {closest_frame} (gap={gap} frames, {gap_seconds:.2f}s)"
                )

        if not frame_detections:
            raise HTTPException(
                status_code=404,
                detail=f"No faces found near timestamp {timestamp_s:.2f}s (frame {original_frame_idx})"
            )

        # Build mapping chain: track_id -> identity_id -> person_id -> cast (name, cast_id)
        identities_payload = _load_identities(ep_id)
        identities_list = identities_payload.get("identities", [])

        # track_id -> identity
        track_to_identity = {}
        identity_to_person = {}
        for identity in identities_list:
            identity_id = identity.get("identity_id")
            person_id = identity.get("person_id")
            if identity_id and person_id:
                identity_to_person[identity_id] = person_id
            for track_id in identity.get("track_ids", []) or []:
                try:
                    track_to_identity[int(track_id)] = identity
                except (TypeError, ValueError):
                    continue

        # Load people and cast for name mapping
        ep_ctx = episode_context_from_id(ep_id)
        show_id = ep_ctx.show_slug.upper()
        people_service = PeopleService()
        people = people_service.list_people(show_id)

        # Load cast members for name lookup (people may have name=None but cast_id set)
        from apps.api.services.cast import CastService
        cast_service = CastService()
        cast_members = cast_service.list_cast(show_id)
        cast_name_lookup = {c.get("cast_id"): c.get("name") for c in cast_members if c.get("cast_id")}

        # person_id -> cast info (name, cast_id)
        person_to_cast = {}
        for person in people:
            person_id = person.get("person_id")
            if person_id:
                # Try person's name first, then fall back to cast lookup
                name = person.get("name") or person.get("display_name")
                cast_id = person.get("cast_id")
                if not name and cast_id:
                    name = cast_name_lookup.get(cast_id)
                person_to_cast[person_id] = {
                    "name": name,
                    "cast_id": cast_id,
                }

        # Extract the video frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise HTTPException(status_code=404, detail=f"Could not read frame {frame_idx}")

        # Draw bounding boxes for each face/track
        colors = [
            (66, 133, 244),   # Blue
            (52, 168, 83),    # Green
            (251, 188, 4),    # Yellow
            (234, 67, 53),    # Red
            (154, 0, 255),    # Purple
            (0, 188, 212),    # Cyan
            (255, 152, 0),    # Orange
            (156, 39, 176),   # Deep Purple
        ]
        gray_color = (128, 128, 128)  # Gray for unidentified faces

        face_info = []
        track_ids_seen = set()

        # Iterate over ALL detections (not just harvested faces)
        for detection in frame_detections:
            track_id = detection.get("track_id")
            # Detections use "bbox" as [x1, y1, x2, y2] format
            bbox = detection.get("bbox") or detection.get("bbox_xyxy")
            conf = detection.get("conf", 0.0)

            if track_id is None or not bbox:
                continue

            # Skip duplicates
            if track_id in track_ids_seen:
                continue
            track_ids_seen.add(track_id)

            # Resolve track -> identity -> person -> cast
            identity = track_to_identity.get(track_id)
            identity_id = identity.get("identity_id") if identity else None
            person_id = identity.get("person_id") if identity else None
            cast_info = person_to_cast.get(person_id) if person_id else None

            name = cast_info.get("name") if cast_info else None
            cast_id = cast_info.get("cast_id") if cast_info else None

            # Skip unidentified if not requested
            if not include_unidentified and not name:
                continue

            # Determine pipeline status for this face
            is_tracked = track_id in tracked_at_frame
            is_harvested = track_id in harvested_track_ids
            is_clustered = identity_id is not None

            # Choose color: named cast get colors, unidentified get gray
            if name:
                color_idx = track_id % len(colors)
                color = colors[color_idx]
            else:
                color = gray_color

            # Draw bbox
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Create label: name if available, otherwise identity_id or track_id
                if name:
                    label = name
                elif identity_id:
                    label = f"[{identity_id[:12]}]"
                else:
                    label = f"T{track_id}"

                font_scale = 0.6
                thickness = 2
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )

                # Draw label background
                cv2.rectangle(
                    frame, (x1, y1 - label_h - 6), (x1 + label_w + 6, y1), color, -1
                )
                cv2.putText(
                    frame, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
                )

                # Determine WHY this face is unidentified
                unidentified_reason = None
                if not name:
                    if not is_harvested:
                        unidentified_reason = "not_harvested"  # Face didn't pass quality gating
                    elif not is_clustered:
                        unidentified_reason = "not_clustered"  # Face not assigned to identity
                    elif not person_id:
                        unidentified_reason = "no_person"  # Identity not linked to person
                    elif not cast_id:
                        unidentified_reason = "no_cast"  # Person not linked to cast
                    else:
                        unidentified_reason = "no_name"  # Cast has no name

                # Get quality scores from harvested face if available
                quality_scores = {}
                harvested_face = harvested_faces_lookup.get((track_id, frame_idx))
                if harvested_face:
                    quality_scores = {
                        "quality": round(harvested_face.get("quality", 0), 3),
                        "blur": round(harvested_face.get("blur", 0), 3),
                        "pose_yaw": round(harvested_face.get("pose_yaw", 0), 1),
                        "pose_pitch": round(harvested_face.get("pose_pitch", 0), 1),
                        "pose_roll": round(harvested_face.get("pose_roll", 0), 1),
                        "det_score": round(harvested_face.get("det_score", conf), 3),
                        "embedding_norm": round(harvested_face.get("embedding_norm", 0), 3),
                    }
                elif detection:
                    # Use detection scores if no harvested face
                    quality_scores = {
                        "det_score": round(conf, 3),
                    }
                    # Check if there's a nearby harvested face for this track
                    for (tid, fidx), hf in harvested_faces_lookup.items():
                        if tid == track_id:
                            quality_scores["quality"] = round(hf.get("quality", 0), 3)
                            quality_scores["blur"] = round(hf.get("blur", 0), 3)
                            break

                face_entry = {
                    "track_id": track_id,
                    "bbox": bbox,
                    "conf": round(conf, 3),
                    "name": name,
                    "cast_id": cast_id,
                    "identity_id": identity_id,
                    "person_id": person_id,
                    # Pipeline status
                    "detected": True,  # If it's in detections.jsonl, it was detected
                    "tracked": is_tracked,
                    "harvested": is_harvested,
                    "clustered": is_clustered,
                }

                # Add diagnostic info for unidentified faces
                if not name:
                    face_entry["unidentified_reason"] = unidentified_reason

                # Add quality scores
                if quality_scores:
                    face_entry["scores"] = quality_scores

                face_info.append(face_entry)
            except (TypeError, ValueError):
                continue

        # Save overlay image
        overlay_filename = f"frame_{frame_idx:06d}_preview.jpg"

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        success = cv2.imwrite(str(tmp_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Failed to save preview image")

        # Upload to S3 or save locally
        try:
            s3_key = f"artifacts/{ep_id}/previews/{overlay_filename}"

            if STORAGE.backend in {"s3", "minio"} and STORAGE._client is not None:
                extra_args = {"ContentType": "image/jpeg"}
                STORAGE._client.upload_file(
                    str(tmp_path),
                    STORAGE.bucket,
                    s3_key,
                    ExtraArgs=extra_args,
                )
                url = STORAGE.presign_get(s3_key, expires_in=3600)
                if not url:
                    raise HTTPException(status_code=500, detail="Failed to generate presigned URL")
            else:
                # Local mode - save to artifacts directory
                artifacts_dir = get_path(ep_id, "frames_root").parent / "previews"
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                local_path = artifacts_dir / overlay_filename
                shutil.copy(tmp_path, local_path)
                url = str(local_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        # Calculate gap info if we had to find a nearby frame
        actual_timestamp_s = round(frame_idx / fps, 3)
        gap_frames = abs(frame_idx - original_frame_idx)
        gap_seconds = round(gap_frames / fps, 2) if gap_frames > 0 else 0

        # Calculate pipeline summary stats
        num_detected = len(frame_detections)
        num_tracked = len(tracked_at_frame)
        num_harvested = sum(1 for f in face_info if f.get("harvested"))
        num_clustered = sum(1 for f in face_info if f.get("clustered"))

        return {
            "url": url,
            "frame_idx": frame_idx,
            "timestamp_s": actual_timestamp_s,
            "requested_timestamp_s": timestamp_s,
            "gap_frames": gap_frames,
            "gap_seconds": gap_seconds,
            "fps": fps,
            "duration_s": round(duration_s, 2),
            "faces": face_info,
            # Pipeline summary for this frame
            "pipeline_summary": {
                "detected": num_detected,
                "tracked": num_tracked,
                "harvested": num_harvested,
                "clustered": num_clustered,
            },
        }
    finally:
        cap.release()


@router.get("/episodes/{ep_id}/frame/{frame_idx}/preview")
def generate_frame_preview(
    ep_id: str,
    frame_idx: int,
    include_unidentified: bool = Query(True, description="Include faces without cast assignment"),
) -> dict:
    """Generate a frame preview at a specific frame index with named bounding boxes.

    This is a convenience wrapper around the timestamp preview that takes a frame index
    directly instead of a timestamp.

    Args:
        ep_id: Episode identifier
        frame_idx: Frame index (0-based)
        include_unidentified: If True, show boxes for faces without cast assignment

    Returns:
        Same format as /timestamp/{timestamp_s}/preview endpoint
    """
    # Get video FPS to convert frame index to timestamp
    video_path = get_path(ep_id, "video")
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video file")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
    finally:
        cap.release()

    # Convert frame index to timestamp
    timestamp_s = frame_idx / fps

    # Reuse the timestamp preview endpoint
    return generate_timestamp_preview(ep_id, timestamp_s, include_unidentified)


@router.get("/episodes/{ep_id}/frames_with_faces")
def get_frames_with_faces(
    ep_id: str,
    min_frame: int = Query(0, ge=0, description="Minimum frame index"),
    max_frame: Optional[int] = Query(None, ge=0, description="Maximum frame index (None for all)"),
) -> dict:
    """Get a sorted list of frame indices that have face detections.

    This is useful for navigation - jumping to the next/previous frame with faces.

    Args:
        ep_id: Episode identifier
        min_frame: Minimum frame index to include
        max_frame: Maximum frame index to include (None = no limit)

    Returns:
        {
            "frames": [0, 24, 48, 72, ...],  # Sorted list of frame indices with faces
            "count": 150,  # Total count
            "min_frame": 0,
            "max_frame": 3600
        }
    """
    detections_path = get_path(ep_id, "detections")
    if not detections_path.exists():
        return {"frames": [], "count": 0, "min_frame": min_frame, "max_frame": max_frame}

    frame_set = set()
    with open(detections_path, "r") as f:
        for line in f:
            try:
                det = json.loads(line)
                frame_idx = det.get("frame_idx")
                if frame_idx is not None:
                    if frame_idx >= min_frame:
                        if max_frame is None or frame_idx <= max_frame:
                            frame_set.add(frame_idx)
            except json.JSONDecodeError:
                continue

    frames = sorted(frame_set)
    return {
        "frames": frames,
        "count": len(frames),
        "min_frame": min_frame,
        "max_frame": max_frame,
    }


class VideoClipRequest(BaseModel):
    """Request model for video clip generation."""

    start_s: float = Field(..., ge=0, description="Start timestamp in seconds")
    end_s: float = Field(..., gt=0, description="End timestamp in seconds")
    include_unidentified: bool = Field(True, description="Include faces without cast assignment")
    output_fps: Optional[float] = Field(None, ge=1, le=60, description="Output FPS (default: source FPS)")


@router.post("/episodes/{ep_id}/video_clip")
def generate_video_clip(ep_id: str, body: VideoClipRequest) -> dict:
    """Generate a video clip with named face overlays.

    Creates a short video clip from the source video with bounding boxes and
    cast member names drawn on each frame.

    Args:
        ep_id: Episode identifier
        body: VideoClipRequest with start_s, end_s, include_unidentified, output_fps

    Returns:
        {
            "url": "path/to/clip.mp4",
            "start_s": 30.0,
            "end_s": 35.0,
            "duration_s": 5.0,
            "fps": 30.0,
            "frame_count": 150,
            "faces_detected": 42
        }
    """
    import tempfile
    import shutil
    from apps.api.services.people import PeopleService

    if body.end_s <= body.start_s:
        raise HTTPException(status_code=400, detail="end_s must be greater than start_s")

    max_duration = 30.0  # Max 30 second clips
    if body.end_s - body.start_s > max_duration:
        raise HTTPException(
            status_code=400,
            detail=f"Clip duration exceeds maximum of {max_duration} seconds"
        )

    # Get video path
    video_path = get_path(ep_id, "video")
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video file")

    try:
        # Get video metadata
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0:
            source_fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / source_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Clamp timestamps to video duration
        start_s = max(0, min(body.start_s, video_duration))
        end_s = max(start_s + 0.1, min(body.end_s, video_duration))

        # Calculate frame range
        start_frame = int(start_s * source_fps)
        end_frame = int(end_s * source_fps)

        # Use output FPS or source FPS
        output_fps = body.output_fps or source_fps

        # Load faces for this episode
        faces = identity_service.load_faces(ep_id)
        if not faces:
            raise HTTPException(status_code=404, detail="No faces found for episode")

        # Index faces by frame for fast lookup
        faces_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for face in faces:
            frame_idx = face.get("frame_idx")
            if frame_idx is not None:
                faces_by_frame.setdefault(frame_idx, []).append(face)

        # Build mapping chain: track_id -> identity -> person -> cast
        identities_payload = _load_identities(ep_id)
        identities_list = identities_payload.get("identities", [])

        track_to_identity = {}
        for identity in identities_list:
            identity_id = identity.get("identity_id")
            for track_id in identity.get("track_ids", []) or []:
                try:
                    track_to_identity[int(track_id)] = identity
                except (TypeError, ValueError):
                    continue

        # Load people and cast for name mapping
        ep_ctx = episode_context_from_id(ep_id)
        show_id = ep_ctx.show_slug.upper()
        people_service = PeopleService()
        people = people_service.list_people(show_id)

        # Load cast members for name lookup (people may have name=None but cast_id set)
        from apps.api.services.cast import CastService
        cast_service = CastService()
        cast_members = cast_service.list_cast(show_id)
        cast_name_lookup = {c.get("cast_id"): c.get("name") for c in cast_members if c.get("cast_id")}

        person_to_cast = {}
        for person in people:
            person_id = person.get("person_id")
            if person_id:
                # Try person's name first, then fall back to cast lookup
                name = person.get("name") or person.get("display_name")
                cast_id = person.get("cast_id")
                if not name and cast_id:
                    name = cast_name_lookup.get(cast_id)
                person_to_cast[person_id] = {
                    "name": name,
                    "cast_id": cast_id,
                }

        # Colors for different tracks
        colors = [
            (66, 133, 244),   # Blue
            (52, 168, 83),    # Green
            (251, 188, 4),    # Yellow
            (234, 67, 53),    # Red
            (154, 0, 255),    # Purple
            (0, 188, 212),    # Cyan
            (255, 152, 0),    # Orange
            (156, 39, 176),   # Deep Purple
        ]
        gray_color = (128, 128, 128)

        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_path), fourcc, output_fps, (width, height))

        if not writer.isOpened():
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Could not create video writer")

        try:
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames_written = 0
            total_faces_detected = 0

            # Calculate frame step for output FPS (if downsampling)
            frame_step = max(1, int(source_fps / output_fps)) if output_fps < source_fps else 1

            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Skip frames if downsampling
                if (frame_idx - start_frame) % frame_step != 0:
                    continue

                # Find faces near this frame (accounting for stride)
                # Check current frame and nearby frames within stride window
                frame_faces = []
                stride = 8  # Default stride from tracker config
                for check_frame in range(frame_idx - stride, frame_idx + stride + 1):
                    if check_frame in faces_by_frame:
                        frame_faces.extend(faces_by_frame[check_frame])
                        break  # Use first match

                track_ids_seen = set()
                for face in frame_faces:
                    track_id = face.get("track_id")
                    bbox = face.get("bbox_xyxy")
                    if track_id is None or not bbox:
                        continue
                    if track_id in track_ids_seen:
                        continue
                    track_ids_seen.add(track_id)

                    # Resolve identity chain
                    identity = track_to_identity.get(track_id)
                    identity_id = identity.get("identity_id") if identity else None
                    person_id = identity.get("person_id") if identity else None
                    cast_info = person_to_cast.get(person_id) if person_id else None
                    name = cast_info.get("name") if cast_info else None

                    # Skip unidentified if not requested
                    if not body.include_unidentified and not name:
                        continue

                    total_faces_detected += 1

                    # Choose color
                    if name:
                        color = colors[track_id % len(colors)]
                    else:
                        color = gray_color

                    # Draw bbox
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Create label
                        if name:
                            label = name
                        elif identity_id:
                            label = f"[{identity_id[:8]}]"
                        else:
                            label = f"T{track_id}"

                        font_scale = 0.6
                        thickness = 2
                        (label_w, label_h), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                        )

                        # Draw label background
                        cv2.rectangle(
                            frame, (x1, y1 - label_h - 6), (x1 + label_w + 6, y1), color, -1
                        )
                        cv2.putText(
                            frame, label, (x1 + 3, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
                        )
                    except (TypeError, ValueError):
                        continue

                # Add timestamp overlay
                current_time = frame_idx / source_fps
                mins = int(current_time // 60)
                secs = current_time % 60
                timestamp_text = f"{mins}:{secs:05.2f}"
                cv2.putText(
                    frame, timestamp_text, (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

                writer.write(frame)
                frames_written += 1

        finally:
            writer.release()

        if frames_written == 0:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="No frames written to clip")

        # Re-encode to H.264 for browser compatibility using ffmpeg
        import subprocess

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as h264_tmp:
            h264_path = Path(h264_tmp.name)

        try:
            # Use ffmpeg to convert to H.264 with AAC audio (browser compatible)
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(tmp_path),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",  # Required for browser compatibility
                "-movflags", "+faststart",  # Enable streaming
                "-an",  # No audio (source has no audio)
                str(h264_path)
            ]
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                timeout=120,
            )
            if result.returncode != 0:
                LOGGER.warning(f"ffmpeg conversion failed: {result.stderr.decode()}")
                # Fall back to original file if ffmpeg fails
                h264_path = tmp_path
            else:
                # Remove original temp file, use h264 version
                tmp_path.unlink(missing_ok=True)
                tmp_path = h264_path
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            LOGGER.warning(f"ffmpeg not available or timed out: {e}, using mp4v codec")
            h264_path.unlink(missing_ok=True)
            # Continue with original tmp_path

        # Generate output filename
        clip_filename = f"clip_{int(start_s)}s_{int(end_s)}s.mp4"

        # Upload to S3 or save locally
        try:
            s3_key = f"artifacts/{ep_id}/clips/{clip_filename}"

            if STORAGE.backend in {"s3", "minio"} and STORAGE._client is not None:
                extra_args = {"ContentType": "video/mp4"}
                STORAGE._client.upload_file(
                    str(tmp_path),
                    STORAGE.bucket,
                    s3_key,
                    ExtraArgs=extra_args,
                )
                url = STORAGE.presign_get(s3_key, expires_in=3600)
                if not url:
                    raise HTTPException(status_code=500, detail="Failed to generate presigned URL")
            else:
                # Local mode - save to artifacts directory
                artifacts_dir = get_path(ep_id, "frames_root").parent / "clips"
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                local_path = artifacts_dir / clip_filename
                shutil.copy(tmp_path, local_path)
                url = str(local_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        return {
            "url": url,
            "start_s": round(start_s, 2),
            "end_s": round(end_s, 2),
            "duration_s": round(end_s - start_s, 2),
            "fps": output_fps,
            "frame_count": frames_written,
            "faces_detected": total_faces_detected,
        }

    finally:
        cap.release()


@router.get("/episodes/{ep_id}/timeline_export")
def export_timeline_data(
    ep_id: str,
    interval_s: float = Query(1.0, ge=0.1, le=10.0, description="Time interval for binning (seconds)"),
    include_unassigned: bool = Query(True, description="Include unassigned tracks/clusters"),
    format: str = Query("json", description="Output format: json or csv"),
) -> dict:
    """Export timeline data showing who appears at each second of the video.

    Returns a second-by-second breakdown of detected faces, their identities,
    and cast member names. Useful for external analysis or verification.

    Args:
        ep_id: Episode identifier
        interval_s: Time interval for binning (default 1 second)
        include_unassigned: Include tracks without cast assignment
        format: Output format (json or csv)

    Returns:
        {
            "episode_id": "...",
            "duration_s": 2700.0,
            "interval_s": 1.0,
            "timeline": [
                {
                    "time_s": 0.0,
                    "frame_start": 0,
                    "frame_end": 30,
                    "faces": [
                        {"track_id": 1, "identity_id": "...", "name": "Lisa", "cast_id": "..."},
                        {"track_id": 5, "identity_id": "...", "name": null, "reason": "no_person"},
                    ]
                },
                ...
            ],
            "summary": {
                "total_intervals": 2700,
                "cast_appearances": {"Lisa": 450, "Mary": 380, ...},
                "unassigned_intervals": 120
            }
        }
    """
    import csv
    import io
    from apps.api.services.people import PeopleService

    # Get video metadata
    video_path = get_path(ep_id, "video")
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video file")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps
    finally:
        cap.release()

    # Load faces
    faces = identity_service.load_faces(ep_id)
    if not faces:
        raise HTTPException(status_code=404, detail="No faces found for episode")

    # Build mapping chains
    identities_payload = _load_identities(ep_id)
    identities_list = identities_payload.get("identities", [])

    track_to_identity = {}
    for identity in identities_list:
        identity_id = identity.get("identity_id")
        for track_id in identity.get("track_ids", []) or []:
            try:
                track_to_identity[int(track_id)] = identity
            except (TypeError, ValueError):
                continue

    # Load people
    ep_ctx = episode_context_from_id(ep_id)
    show_id = ep_ctx.show_slug.upper()
    people_service = PeopleService()
    people = people_service.list_people(show_id)

    person_to_cast = {}
    for person in people:
        person_id = person.get("person_id")
        if person_id:
            person_to_cast[person_id] = {
                "name": person.get("name") or person.get("display_name"),
                "cast_id": person.get("cast_id"),
            }

    # Index faces by timestamp interval
    faces_by_interval: Dict[int, List[Dict[str, Any]]] = {}
    for face in faces:
        ts = face.get("ts")
        if ts is None:
            frame_idx = face.get("frame_idx", 0)
            ts = frame_idx / fps

        interval_idx = int(ts / interval_s)
        if interval_idx not in faces_by_interval:
            faces_by_interval[interval_idx] = []

        track_id = face.get("track_id")
        identity = track_to_identity.get(track_id)
        identity_id = identity.get("identity_id") if identity else None
        person_id = identity.get("person_id") if identity else None
        cast_info = person_to_cast.get(person_id) if person_id else None
        name = cast_info.get("name") if cast_info else None
        cast_id = cast_info.get("cast_id") if cast_info else None

        # Determine reason for unassigned
        reason = None
        if not name:
            if not identity_id:
                reason = "not_clustered"
            elif not person_id:
                reason = "no_person"
            elif not cast_id:
                reason = "no_cast"
            else:
                reason = "no_name"

        # Skip unassigned if not requested
        if not include_unassigned and not name:
            continue

        face_entry = {
            "track_id": track_id,
            "identity_id": identity_id,
            "name": name,
            "cast_id": cast_id,
            "quality": round(face.get("quality", 0), 3),
        }
        if reason:
            face_entry["reason"] = reason

        faces_by_interval[interval_idx].append(face_entry)

    # Build timeline
    total_intervals = int(duration_s / interval_s) + 1
    timeline = []
    cast_appearances: Dict[str, int] = {}
    unassigned_intervals = 0

    for interval_idx in range(total_intervals):
        time_s = interval_idx * interval_s
        frame_start = int(time_s * fps)
        frame_end = int((time_s + interval_s) * fps)

        interval_faces = faces_by_interval.get(interval_idx, [])

        # Deduplicate by track_id
        seen_tracks = set()
        unique_faces = []
        for f in interval_faces:
            tid = f.get("track_id")
            if tid not in seen_tracks:
                seen_tracks.add(tid)
                unique_faces.append(f)

        # Count appearances
        has_unassigned = False
        for f in unique_faces:
            name = f.get("name")
            if name:
                cast_appearances[name] = cast_appearances.get(name, 0) + 1
            else:
                has_unassigned = True

        if has_unassigned:
            unassigned_intervals += 1

        timeline.append({
            "time_s": round(time_s, 2),
            "frame_start": frame_start,
            "frame_end": frame_end,
            "faces": unique_faces,
        })

    result = {
        "episode_id": ep_id,
        "duration_s": round(duration_s, 2),
        "interval_s": interval_s,
        "fps": fps,
        "total_frames": total_frames,
        "timeline": timeline,
        "summary": {
            "total_intervals": total_intervals,
            "cast_appearances": cast_appearances,
            "unassigned_intervals": unassigned_intervals,
        },
    }

    # Return CSV format if requested
    if format.lower() == "csv":
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "time_s", "frame_start", "frame_end",
            "track_id", "identity_id", "name", "cast_id", "quality", "reason"
        ])

        # Rows
        for interval in timeline:
            for face in interval.get("faces", []):
                writer.writerow([
                    interval["time_s"],
                    interval["frame_start"],
                    interval["frame_end"],
                    face.get("track_id"),
                    face.get("identity_id"),
                    face.get("name"),
                    face.get("cast_id"),
                    face.get("quality"),
                    face.get("reason", ""),
                ])

        csv_content = output.getvalue()

        # Save to S3 or return inline
        csv_filename = f"{ep_id}_timeline.csv"
        s3_key = f"artifacts/{ep_id}/exports/{csv_filename}"

        if STORAGE.backend in {"s3", "minio"} and STORAGE._client is not None:
            STORAGE._client.put_object(
                Bucket=STORAGE.bucket,
                Key=s3_key,
                Body=csv_content.encode("utf-8"),
                ContentType="text/csv",
            )
            url = STORAGE.presign_get(s3_key, expires_in=3600)
            return {
                "format": "csv",
                "url": url,
                "s3_key": s3_key,
                "rows": len(timeline),
            }
        else:
            # Return inline for local mode
            return {
                "format": "csv",
                "content": csv_content,
                "rows": len(timeline),
            }

    return result


@router.delete("/episodes/{ep_id}/identities/{identity_id}")
def delete_identity(ep_id: str, identity_id: str) -> dict:
    """Delete (archive) an identity/cluster.

    The cluster is moved to the archive where its centroid is stored.
    This allows matching faces in future episodes to be auto-archived.
    """
    payload = _load_identities(ep_id)
    identities = payload.get("identities", [])

    # Find the identity to archive before deleting
    identity_to_delete = None
    for item in identities:
        if item.get("identity_id") == identity_id:
            identity_to_delete = item
            break

    if not identity_to_delete:
        raise HTTPException(status_code=404, detail="Identity not found")

    # Extract show_id from episode_id (e.g., "rhoslc-s06e02" -> "RHOSLC")
    show_id = ep_id.split("-")[0].upper() if "-" in ep_id else ep_id.upper()

    # Archive the cluster with its centroid for future matching
    try:
        centroid = identity_to_delete.get("centroid")
        rep_crop_url = identity_to_delete.get("rep_crop_url")
        track_ids = identity_to_delete.get("track_ids", [])
        face_count = identity_to_delete.get("face_count", 0)

        archive_service.archive_cluster(
            show_id=show_id,
            episode_id=ep_id,
            cluster_id=identity_id,
            reason="user_deleted",
            centroid=centroid,
            rep_crop_url=rep_crop_url,
            track_ids=track_ids,
            face_count=face_count,
        )
        LOGGER.info(f"Archived cluster {identity_id} before deletion")
    except Exception as e:
        LOGGER.warning(f"Failed to archive cluster {identity_id}: {e}")
        # Continue with deletion even if archive fails

    # Remove the identity
    payload["identities"] = [item for item in identities if item.get("identity_id") != identity_id]
    _update_identity_stats(ep_id, payload)
    path = _write_identities(ep_id, payload)
    _sync_manifests(ep_id, path)

    return {"deleted": identity_id, "archived": True, "remaining": len(payload["identities"])}


@router.delete("/episodes/{ep_id}/tracks/{track_id}")
def delete_track(
    ep_id: str,
    track_id: int,
    payload: TrackDeleteRequest = Body(default=TrackDeleteRequest()),
) -> dict:
    identities_payload = _load_identities(ep_id)
    track_identity_map = _identity_lookup(identities_payload)
    source_identity_id = track_identity_map.get(track_id)
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
    for identity in identities_payload.get("identities", []):
        identity["track_ids"] = [tid for tid in identity.get("track_ids", []) if tid != track_id]
    _update_identity_stats(ep_id, identities_payload)
    identities_path = _write_identities(ep_id, identities_payload)
    _recount_track_faces(ep_id)
    _sync_manifests(ep_id, faces_path, tracks_path, identities_path)
    _refresh_similarity_indexes(ep_id, track_ids=[track_id], identity_ids=[source_identity_id])
    return {"track_id": track_id, "faces_deleted": payload.delete_faces}


@router.delete("/episodes/{ep_id}/frames")
def delete_frame(ep_id: str, payload: FrameDeleteRequest) -> dict:
    identities_payload = _load_identities(ep_id)
    track_identity_map = _identity_lookup(identities_payload)
    source_identity_id = track_identity_map.get(payload.track_id)
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
    _update_identity_stats(ep_id, identities_payload)
    identities_path = _write_identities(ep_id, identities_payload)
    _sync_manifests(ep_id, faces_path, identities_path)
    _refresh_similarity_indexes(ep_id, track_ids=[payload.track_id], identity_ids=[source_identity_id])
    return {
        "track_id": payload.track_id,
        "frame_idx": payload.frame_idx,
        "removed": len(removed_rows),
        "remaining": len(faces),
    }


@router.post("/episodes/{ep_id}/identities/{identity_id}/export_seeds", tags=["episodes"])
def export_facebank_seeds(ep_id: str, identity_id: str) -> Dict[str, Any]:
    """
    Select and export high-quality seed frames to permanent facebank.
    Only exports user-confirmed identities with person_id mappings.
    """
    # Validate identity_id format (prevent path traversal)
    if not identity_id or not re.match(r"^[a-zA-Z0-9_-]+$", identity_id):
        raise HTTPException(status_code=400, detail="Invalid identity_id format. Must match [a-zA-Z0-9_-]+")

    # Verify episode exists
    _require_episode_context(ep_id)

    # Load all faces for this identity
    all_faces = _load_faces(ep_id, include_skipped=False)
    identity_faces = [f for f in all_faces if f.get("identity_id") == identity_id]

    if not identity_faces:
        raise HTTPException(status_code=404, detail=f"No faces found for identity {identity_id} in episode {ep_id}")

    # Check if identity has a person_id mapping (cast member)
    identities = _load_identities(ep_id)
    if not isinstance(identities, dict):
        raise HTTPException(status_code=500, detail="Invalid identities data structure")

    identity_record = next((i for i in identities.get("identities", []) if i.get("identity_id") == identity_id), None)

    if not identity_record:
        raise HTTPException(status_code=404, detail=f"Identity {identity_id} not found in manifest for episode {ep_id}")

    person_id = identity_record.get("person_id")
    if not person_id:
        raise HTTPException(
            status_code=400,
            detail=f"Identity {identity_id} has no person_id mapping. Assign to a cast member first in the Faces Review UI.",
        )

    # Select seeds using quality criteria
    try:
        seeds = select_facebank_seeds(ep_id, identity_id, identity_faces)
    except Exception as exc:
        LOGGER.error(f"Failed to select facebank seeds for {ep_id}/{identity_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Seed selection failed: {exc}") from exc

    if not seeds:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No quality seeds available for {identity_id}. "
                "Check that faces have detection scores ≥0.75, sharpness ≥15.0, and similarity ≥0.70. "
                "Review frames in the Faces Review UI to confirm quality."
            ),
        )

    # Write to facebank
    facebank_root = Path(os.environ.get("SCREENALYTICS_FACEBANK_ROOT", "data/facebank")).expanduser()

    try:
        seeds_path = write_facebank_seeds(person_id, seeds, facebank_root)
    except (OSError, ValueError) as exc:
        LOGGER.error(f"Failed to write facebank seeds for {person_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to write seeds to facebank: {exc}") from exc

    # Log successful export
    LOGGER.info(
        "Exported %d facebank seeds for ep_id=%s identity=%s person=%s to %s",
        len(seeds),
        ep_id,
        identity_id,
        person_id,
        seeds_path,
    )

    # TODO: Emit refresh job for similarity recomputation across episodes
    # This would trigger re-indexing of embeddings in pgvector/FAISS
    # For now, similarity refresh must be triggered manually

    return {
        "status": "success",
        "person_id": person_id,
        "identity_id": identity_id,
        "ep_id": ep_id,
        "seeds_exported": len(seeds),
        "seeds_path": str(seeds_path),
        "refresh_required": True,
        "message": f"Exported {len(seeds)} high-quality seeds to facebank. Similarity refresh recommended.",
    }


# =============================================================================
# Enhancement #7: Real-time Collaboration Indicators (Presence Tracking)
# =============================================================================

# In-memory presence store (for single-instance deployments)
# For production, use Redis or similar for cross-instance presence
_PRESENCE_STORE: Dict[str, Dict[str, Any]] = {}
_PRESENCE_TTL_SECONDS = 60  # Presence expires after 60 seconds of no heartbeat


class PresenceHeartbeat(BaseModel):
    """Request body for presence heartbeat."""
    user_id: Optional[str] = Field(None, description="User identifier")
    user_name: Optional[str] = Field("Anonymous", description="Display name")


@router.get("/episodes/{ep_id}/presence", tags=["episodes"])
def get_presence(ep_id: str) -> Dict[str, Any]:
    """Get current viewers for an episode.

    Returns list of users currently viewing this episode,
    excluding viewers whose heartbeat has expired.
    """
    ep_id = normalize_ep_id(ep_id)
    now = datetime.now(timezone.utc).timestamp()

    # Clean up expired entries and collect active viewers
    viewers = []
    if ep_id in _PRESENCE_STORE:
        active = {}
        for user_key, presence in _PRESENCE_STORE[ep_id].items():
            if now - presence.get("last_seen", 0) < _PRESENCE_TTL_SECONDS:
                active[user_key] = presence
                viewers.append({
                    "user_id": presence.get("user_id"),
                    "name": presence.get("user_name", "Anonymous"),
                    "last_seen": presence.get("last_seen"),
                })
        _PRESENCE_STORE[ep_id] = active

    return {
        "ep_id": ep_id,
        "viewers": viewers,
        "count": len(viewers),
    }


@router.post("/episodes/{ep_id}/presence", tags=["episodes"])
def update_presence(ep_id: str, heartbeat: PresenceHeartbeat = Body(default=None)) -> Dict[str, Any]:
    """Update presence heartbeat for a user viewing an episode.

    Call this endpoint periodically (every 30s) to maintain presence.
    """
    ep_id = normalize_ep_id(ep_id)
    heartbeat = heartbeat or PresenceHeartbeat()

    user_id = heartbeat.user_id or "anonymous"
    user_name = heartbeat.user_name or "Anonymous"
    user_key = f"{user_id}_{hash(user_name) % 10000}"

    if ep_id not in _PRESENCE_STORE:
        _PRESENCE_STORE[ep_id] = {}

    _PRESENCE_STORE[ep_id][user_key] = {
        "user_id": user_id,
        "user_name": user_name,
        "last_seen": datetime.now(timezone.utc).timestamp(),
    }

    return {
        "status": "ok",
        "ep_id": ep_id,
        "user_id": user_id,
    }


@router.delete("/episodes/{ep_id}/presence", tags=["episodes"])
def leave_presence(ep_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Remove presence for a user (when leaving the page)."""
    ep_id = normalize_ep_id(ep_id)

    if ep_id in _PRESENCE_STORE:
        if user_id:
            # Remove specific user
            keys_to_remove = [k for k in _PRESENCE_STORE[ep_id] if k.startswith(user_id)]
            for k in keys_to_remove:
                _PRESENCE_STORE[ep_id].pop(k, None)
        else:
            # Clean up entire episode presence
            _PRESENCE_STORE.pop(ep_id, None)

    return {"status": "ok", "ep_id": ep_id}


@router.post("/episodes/{ep_id}/sync_thumbnails_to_s3", tags=["episodes"])
def sync_thumbnails_to_s3(ep_id: str) -> Dict[str, Any]:
    """Sync local thumbnail and crop files to S3.

    Scans the local thumbs/ and crops/ directories and uploads any files
    that exist locally but are missing from S3. Use this when thumbnails
    aren't loading due to missing S3 artifacts.

    Returns:
        uploaded_thumbs: Number of thumbnails uploaded
        uploaded_crops: Number of crops uploaded
        errors: List of any upload errors
    """
    ep_id = normalize_ep_id(ep_id)

    try:
        ep_ctx = episode_context_from_id(ep_id)
        prefixes = artifact_prefixes(ep_ctx)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    frames_root = get_path(ep_id, "frames_root")
    thumbs_dir = frames_root / "thumbs"
    crops_dir = frames_root / "crops"

    # Separate prefixes for track thumbs and identity thumbs
    thumbs_tracks_prefix = prefixes.get("thumbs_tracks", "")
    thumbs_identities_prefix = prefixes.get("thumbs_identities", "")
    crops_prefix = prefixes.get("crops", "")

    uploaded_thumbs = 0
    uploaded_crops = 0
    errors: List[str] = []

    # Sync track thumbnails (thumbs/track_xxxx/*.jpg)
    if thumbs_dir.exists() and thumbs_tracks_prefix:
        tracks_subdir = thumbs_dir
        for thumb_file in tracks_subdir.rglob("track_*/*.jpg"):
            # rel_path is like "track_0001/thumb_000120.jpg"
            rel_path = thumb_file.relative_to(thumbs_dir)
            s3_key = f"{thumbs_tracks_prefix}{rel_path}"

            # Check if already exists in S3 using proper existence check
            if STORAGE.object_exists(s3_key):
                continue  # Already in S3

            # Upload to S3
            success = STORAGE.put_artifact(ep_ctx, "thumbs_tracks", thumb_file, str(rel_path))
            if success:
                uploaded_thumbs += 1
            else:
                errors.append(f"Failed to upload thumb: {rel_path}")

    # Sync identity thumbnails (thumbs/identities/*.jpg)
    if thumbs_dir.exists() and thumbs_identities_prefix:
        identities_subdir = thumbs_dir / "identities"
        if identities_subdir.exists():
            for thumb_file in identities_subdir.rglob("*.jpg"):
                # rel_path is like "id_0001.jpg"
                rel_path = thumb_file.relative_to(identities_subdir)
                s3_key = f"{thumbs_identities_prefix}{rel_path}"

                # Check if already exists in S3 using proper existence check
                if STORAGE.object_exists(s3_key):
                    continue  # Already in S3

                # Upload to S3
                success = STORAGE.put_artifact(ep_ctx, "thumbs_identities", thumb_file, str(rel_path))
                if success:
                    uploaded_thumbs += 1
                else:
                    errors.append(f"Failed to upload identity thumb: {rel_path}")

    # Sync crops
    if crops_dir.exists() and crops_prefix:
        for crop_file in crops_dir.rglob("*.jpg"):
            rel_path = crop_file.relative_to(crops_dir)
            s3_key = f"{crops_prefix}{rel_path}"

            # Check if already exists in S3 using proper existence check
            if STORAGE.object_exists(s3_key):
                continue  # Already in S3

            # Upload to S3
            success = STORAGE.put_artifact(ep_ctx, "crops", crop_file, str(rel_path))
            if success:
                uploaded_crops += 1
            else:
                errors.append(f"Failed to upload crop: {rel_path}")

    return {
        "status": "success",
        "ep_id": ep_id,
        "uploaded_thumbs": uploaded_thumbs,
        "uploaded_crops": uploaded_crops,
        "errors": errors[:20] if errors else [],  # Limit error list
        "total_errors": len(errors),
    }


@router.get("/episodes/{ep_id}/artifact_status", tags=["episodes"])
def get_artifact_status(ep_id: str) -> Dict[str, Any]:
    """Get detailed artifact status for an episode (local and S3 counts).

    Returns counts of frames, crops, thumbnails, and manifests both locally and in S3.
    Useful for displaying sync status in the UI.
    """
    from apps.api.services.storage import artifact_prefixes, episode_context_from_id

    result: Dict[str, Any] = {
        "ep_id": ep_id,
        "local": {
            "frames": 0,
            "crops": 0,
            "thumbs_tracks": 0,
            "thumbs_identities": 0,
            "manifests": 0,
        },
        "s3": {
            "frames": 0,
            "crops": 0,
            "thumbs_tracks": 0,
            "thumbs_identities": 0,
            "manifests": 0,
        },
        "sync_status": "unknown",
        "s3_enabled": STORAGE.s3_enabled(),
    }

    # Get local artifact counts
    try:
        frames_root = get_path(ep_id, "frames_root")
        frames_dir = frames_root / "frames"
        crops_dir = frames_root / "crops"
        thumbs_tracks_dir = frames_root / "thumbs" / "tracks"
        thumbs_identities_dir = frames_root / "thumbs" / "identities"
        manifests_dir = get_path(ep_id, "detections").parent

        if frames_dir.exists():
            result["local"]["frames"] = len(list(frames_dir.glob("*.jpg")))
        if crops_dir.exists():
            result["local"]["crops"] = len(list(crops_dir.rglob("*.jpg")))
        if thumbs_tracks_dir.exists():
            result["local"]["thumbs_tracks"] = len(list(thumbs_tracks_dir.rglob("*.jpg")))
        if thumbs_identities_dir.exists():
            result["local"]["thumbs_identities"] = len(list(thumbs_identities_dir.rglob("*.jpg")))
        if manifests_dir.exists():
            manifest_extensions = {".json", ".jsonl", ".ndjson"}
            result["local"]["manifests"] = len([
                f for f in manifests_dir.iterdir()
                if f.is_file() and f.suffix.lower() in manifest_extensions
            ])
    except Exception as exc:
        LOGGER.warning("Failed to count local artifacts for %s: %s", ep_id, exc)

    # Get S3 artifact counts if S3 is enabled
    if STORAGE.s3_enabled():
        try:
            ep_ctx = episode_context_from_id(ep_id)
            prefixes = artifact_prefixes(ep_ctx)

            # Count frames in S3 (sample to avoid listing all)
            frames_keys = STORAGE.list_objects(prefixes["frames"], suffix=".jpg", max_items=1000)
            result["s3"]["frames"] = len(frames_keys)

            # Count crops in S3
            crops_keys = STORAGE.list_objects(prefixes["crops"], suffix=".jpg", max_items=1000)
            result["s3"]["crops"] = len(crops_keys)

            # Count track thumbnails in S3
            thumbs_tracks_keys = STORAGE.list_objects(prefixes["thumbs_tracks"], suffix=".jpg", max_items=1000)
            result["s3"]["thumbs_tracks"] = len(thumbs_tracks_keys)

            # Count identity thumbnails in S3
            thumbs_identities_keys = STORAGE.list_objects(prefixes["thumbs_identities"], suffix=".jpg", max_items=1000)
            result["s3"]["thumbs_identities"] = len(thumbs_identities_keys)

            # Count manifests in S3
            manifests_keys = STORAGE.list_objects(prefixes["manifests"], max_items=100)
            result["s3"]["manifests"] = len(manifests_keys)

        except Exception as exc:
            LOGGER.warning("Failed to count S3 artifacts for %s: %s", ep_id, exc)

    # Determine sync status
    local_total = sum(result["local"].values())
    s3_total = sum(result["s3"].values())

    if not STORAGE.s3_enabled():
        result["sync_status"] = "s3_disabled"
    elif local_total == 0 and s3_total == 0:
        result["sync_status"] = "empty"
    elif s3_total >= local_total and local_total > 0:
        result["sync_status"] = "synced"
    elif s3_total > 0 and s3_total < local_total:
        result["sync_status"] = "partial"
    elif local_total > 0 and s3_total == 0:
        result["sync_status"] = "pending"
    else:
        result["sync_status"] = "unknown"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Episode Pipeline Report Export
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/episodes/{ep_id}/report.pdf")
def get_episode_report_pdf(
    ep_id: str,
    include_appendix: bool = Query(False, description="Include raw data samples in appendix"),
    format: Literal["pdf", "snapshot"] = Query("pdf", description="Output format: pdf or snapshot (JSON)"),
    experiment_tag: Optional[str] = Query(None, description="Experiment tag/label for this run"),
    save_snapshot: bool = Query(True, description="Save snapshot for run comparison"),
) -> Response:
    """Generate a comprehensive PDF report of the episode pipeline run.

    This endpoint generates a detailed report containing:
    - Pipeline metadata and execution info
    - Effective configuration snapshot
    - Audio/diarization summary (if available)
    - Transcript summary (if available)
    - Speaker linking and cast assignments
    - Video/face detection analysis
    - Timing and frame details
    - Screen time and speaking time analytics
    - Errors, warnings, and anomalies
    - Run comparison (if previous snapshot exists)

    The report reads from existing manifest files and does NOT re-run any ML models.

    Args:
        ep_id: Episode identifier (e.g., 'rhoslc-s06e99')
        include_appendix: Include raw data samples in the appendix section
        format: Output format - 'pdf' for PDF document, 'snapshot' for JSON snapshot
        experiment_tag: Optional label for this run (for comparison tracking)
        save_snapshot: Whether to save snapshot for future run comparison

    Returns:
        PDF file or JSON snapshot depending on format parameter
    """
    from py_screenalytics.reports.snapshot_builder import (
        build_episode_snapshot,
        save_snapshot as persist_snapshot,
    )
    from py_screenalytics.reports.episode_report_pdf import EpisodeReportPDF
    from py_screenalytics.reports.metrics_export import export_metrics_deltas

    ep_id = normalize_ep_id(ep_id)

    # Resolve experiment_tag with precedence: query_param > env > config default
    resolved_tag = experiment_tag
    if resolved_tag is None:
        resolved_tag = os.environ.get("EXPERIMENT_TAG")
    if resolved_tag is None:
        # Load default from config
        import yaml
        config_path = PROJECT_ROOT / "config" / "pipeline" / "reports.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
                resolved_tag = config.get("default_experiment_tag", "") or None

    LOGGER.info(f"[report] Generating {format} report for {ep_id} (tag: {resolved_tag})")

    # Check that episode exists
    manifests_dir = _manifests_dir(ep_id)
    if not manifests_dir.exists():
        raise HTTPException(status_code=404, detail=f"Episode not found: {ep_id}")

    try:
        # Build the snapshot
        snapshot = build_episode_snapshot(
            ep_id,
            include_appendix=include_appendix,
            experiment_tag=resolved_tag,
        )

        # Save snapshot for future comparison
        if save_snapshot:
            try:
                persist_snapshot(snapshot)
                LOGGER.info(f"[report] Saved snapshot {snapshot.snapshot_id} for {ep_id}")
            except Exception as save_err:
                LOGGER.warning(f"[report] Failed to save snapshot: {save_err}")

        # Export metrics deltas if we have a run comparison
        if snapshot.run_comparison:
            try:
                export_metrics_deltas(snapshot)
                LOGGER.info(f"[report] Exported metrics_deltas.json for {ep_id}")
            except Exception as export_err:
                LOGGER.warning(f"[report] Failed to export metrics deltas: {export_err}")

        if format == "snapshot":
            # Return JSON snapshot
            return Response(
                content=snapshot.model_dump_json(indent=2),
                media_type="application/json",
                headers={
                    "Content-Disposition": f'attachment; filename="{ep_id}_snapshot.json"'
                },
            )

        # Generate PDF
        renderer = EpisodeReportPDF(snapshot)
        pdf_bytes = renderer.generate()

        LOGGER.info(f"[report] Generated PDF report for {ep_id}: {len(pdf_bytes)} bytes")

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{ep_id}_report.pdf"'
            },
        )

    except ImportError as e:
        LOGGER.error(f"[report] Missing dependency: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Missing dependency for PDF generation: {e}. Install reportlab with: pip install reportlab"
        )
    except FileNotFoundError as e:
        LOGGER.error(f"[report] Missing data for {ep_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        LOGGER.error(f"[report] Failed to generate report for {ep_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")


class SnapshotListItem(BaseModel):
    """Summary info for a saved snapshot."""
    snapshot_id: str
    generated_at: str
    experiment_tag: Optional[str] = None
    pinned: bool = False


@router.get("/episodes/{ep_id}/snapshots")
def list_episode_snapshots(ep_id: str) -> List[SnapshotListItem]:
    """List all saved snapshots for an episode.

    Returns:
        List of snapshot summaries, newest first
    """
    ep_id = normalize_ep_id(ep_id)
    snapshots_dir = PROJECT_ROOT / "data" / "snapshots" / ep_id

    if not snapshots_dir.exists():
        return []

    result = []
    for snapshot_file in sorted(snapshots_dir.glob("*.json"), reverse=True):
        if snapshot_file.name == "latest.json":
            continue  # Skip the symlink/copy
        try:
            with open(snapshot_file, "r") as f:
                data = json.load(f)
            result.append(SnapshotListItem(
                snapshot_id=data.get("snapshot_id", snapshot_file.stem),
                generated_at=data.get("generated_at", "unknown"),
                experiment_tag=data.get("pipeline_metadata", {}).get("experiment_tag"),
                pinned=data.get("pinned", False),
            ))
        except Exception as e:
            LOGGER.warning(f"[snapshots] Failed to read {snapshot_file}: {e}")
            continue

    return result


@router.delete("/episodes/{ep_id}/snapshots/{snapshot_id}")
def delete_episode_snapshot(ep_id: str, snapshot_id: str) -> Dict[str, str]:
    """Delete a specific snapshot.

    Args:
        ep_id: Episode identifier
        snapshot_id: Snapshot ID to delete

    Returns:
        Confirmation message

    Raises:
        HTTPException 404: Snapshot not found
        HTTPException 400: Cannot delete pinned snapshot
    """
    ep_id = normalize_ep_id(ep_id)
    snapshots_dir = PROJECT_ROOT / "data" / "snapshots" / ep_id
    snapshot_file = snapshots_dir / f"{snapshot_id}.json"

    if not snapshot_file.exists():
        raise HTTPException(status_code=404, detail=f"Snapshot not found: {snapshot_id}")

    # Check if pinned
    try:
        with open(snapshot_file, "r") as f:
            data = json.load(f)
        if data.get("pinned", False):
            raise HTTPException(
                status_code=400,
                detail="Cannot delete pinned snapshot. Unpin it first."
            )
    except HTTPException:
        raise
    except Exception:
        pass  # If we can't read it, allow deletion

    snapshot_file.unlink()
    LOGGER.info(f"[snapshots] Deleted snapshot {snapshot_id} for {ep_id}")

    # Update latest.json if needed
    latest_file = snapshots_dir / "latest.json"
    if latest_file.exists():
        try:
            with open(latest_file, "r") as f:
                latest_data = json.load(f)
            if latest_data.get("snapshot_id") == snapshot_id:
                # Find next most recent
                remaining = sorted(snapshots_dir.glob("*.json"), reverse=True)
                remaining = [f for f in remaining if f.name != "latest.json"]
                if remaining:
                    import shutil
                    shutil.copy2(remaining[0], latest_file)
                else:
                    latest_file.unlink()
        except Exception as e:
            LOGGER.warning(f"[snapshots] Failed to update latest.json: {e}")

    return {"status": "deleted", "snapshot_id": snapshot_id}
