from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Use the robust coerce_int from ui_helpers to handle edge cases (NaN, Inf, bool)
from ui_helpers import coerce_int  # noqa: F401 - re-exported for backwards compatibility


def quality_score(face_meta: Dict[str, Any]) -> float | None:
    quality = face_meta.get("quality") if isinstance(face_meta, dict) else None
    if isinstance(quality, dict):
        try:
            score = quality.get("score")
            return float(score) if score is not None else None
        except (TypeError, ValueError):
            return None
    return None


def scope_track_frames(frames: List[Dict[str, Any]], track_id: int) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Filter frames down to faces that belong to the active track."""
    scoped: List[Dict[str, Any]] = []
    missing: List[int] = []
    for frame in frames:
        frame_idx = frame.get("frame_idx")
        faces = frame.get("faces") if isinstance(frame.get("faces"), list) else []
        candidates = [face for face in faces if isinstance(face, dict)]

        # First, check if we have explicit face data for this track in the faces list
        # CRITICAL: Only use face-level data when available - it has the correct track-specific URLs
        faces_from_list = [
            face for face in candidates
            if coerce_int(face.get("track_id")) == track_id
        ]

        # Only fall back to frame-level data as a candidate if no explicit face data exists
        # This prevents frame-level URLs (which may point to wrong track) from overriding
        # face-level URLs when both are present
        if not faces_from_list:
            top_level_tid = coerce_int(frame.get("track_id"))
            if top_level_tid in (None, track_id):
                # Extract face fields from frame-level, excluding nested structures
                frame_as_face = {
                    k: v
                    for k, v in frame.items()
                    if k not in ("faces", "other_tracks")  # Exclude nested lists
                }
                if frame_as_face.get("face_id"):  # Only include if it has face metadata
                    candidates.append(frame_as_face)

        faces_for_track = []
        for face in candidates:
            tid = coerce_int(face.get("track_id"))
            if tid == track_id:
                faces_for_track.append(face)

        if not faces_for_track:
            try:
                missing.append(int(frame_idx))
            except (TypeError, ValueError):
                missing.append(frame_idx if frame_idx is not None else -1)
            continue

        # Prefer higher quality within the track for rendering
        faces_for_track.sort(key=lambda f: quality_score(f) or 0.0, reverse=True)
        primary = {**frame, **faces_for_track[0]}
        # CRITICAL: Only include faces for THIS track, never other tracks
        primary["faces"] = faces_for_track
        primary["track_id"] = coerce_int(primary.get("track_id")) or track_id
        scoped.append(primary)
    return scoped, missing


def best_track_frame_idx(frames: List[Dict[str, Any]], track_id: int, fallback: int | None) -> int | None:
    best_idx = None
    best_score = -1.0
    for frame in frames:
        frame_idx = coerce_int(frame.get("frame_idx"))
        if frame_idx is None:
            continue
        faces = frame.get("faces") if isinstance(frame.get("faces"), list) else []
        for face in faces:
            tid = coerce_int(face.get("track_id"))
            if tid != track_id:
                continue
            score = quality_score(face)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_idx = frame_idx
    return best_idx if best_idx is not None else fallback


def track_faces_debug(frames: List[Dict[str, Any]], track_id: int) -> List[str]:
    """Return lightweight debug lines summarizing track IDs present per frame."""
    lines: List[str] = []
    for frame in frames:
        frame_idx = coerce_int(frame.get("frame_idx"))
        faces = frame.get("faces") if isinstance(frame.get("faces"), list) else []
        track_ids = []
        for face in faces:
            tid = coerce_int(face.get("track_id"))
            if tid is not None:
                track_ids.append(tid)
        lines.append(f"frame {frame_idx}: tracks={sorted(set(track_ids))}")
    return lines
