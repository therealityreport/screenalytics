"""Track recovery service for expanding single-frame tracks.

Finds similar faces in adjacent frames and merges them into single-frame tracks.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from py_screenalytics.artifacts import get_path

from apps.api.services.identities import (
    load_faces,
    write_faces,
    load_tracks,
    write_tracks,
    sync_manifests,
    _episode_lock,
    _faces_path,
    _tracks_path,
)

LOGGER = logging.getLogger(__name__)

# Configuration
RECOVERY_FRAME_WINDOW = int(os.getenv("RECOVERY_FRAME_WINDOW", "8"))
RECOVERY_MIN_SIMILARITY = float(os.getenv("RECOVERY_MIN_SIMILARITY", "0.70"))


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = np.linalg.norm(vector) + 1e-12
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors (0-1)."""
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def recover_single_frame_tracks(
    ep_id: str,
    frame_window: int = RECOVERY_FRAME_WINDOW,
    min_similarity: float = RECOVERY_MIN_SIMILARITY,
) -> Dict[str, Any]:
    """
    Recover single-frame tracks by finding similar faces in adjacent frames.

    For each track with only 1 face:
    1. Load the face and its embedding
    2. Find other faces within ±frame_window frames
    3. Compute cosine similarity between embeddings
    4. If similarity >= min_similarity, reassign the adjacent face to this track

    Args:
        ep_id: Episode ID
        frame_window: Number of frames to search before/after (default: 8)
        min_similarity: Minimum cosine similarity to merge faces (default: 0.70)

    Returns:
        Dict with recovery results:
        - tracks_analyzed: Number of single-frame tracks processed
        - tracks_expanded: Number of tracks that were expanded
        - faces_merged: Total faces added to tracks
        - details: List of {track_id, original_frames, added_frames}
    """
    LOGGER.info(f"[{ep_id}] Starting single-frame track recovery (window={frame_window}, min_sim={min_similarity})")

    with _episode_lock(ep_id):
        faces = load_faces(ep_id)

        if not faces:
            LOGGER.warning(f"[{ep_id}] No faces found")
            return {
                "tracks_analyzed": 0,
                "tracks_expanded": 0,
                "faces_merged": 0,
                "details": [],
            }

        # Group faces by track_id
        faces_by_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for face in faces:
            track_id = face.get("track_id")
            if track_id is not None:
                faces_by_track[track_id].append(face)

        # Identify single-frame tracks
        single_frame_tracks = {
            tid: track_faces[0]
            for tid, track_faces in faces_by_track.items()
            if len(track_faces) == 1
        }

        # Identify multi-frame tracks (exclude faces from these tracks)
        multi_frame_track_ids = {
            tid for tid, track_faces in faces_by_track.items()
            if len(track_faces) > 1
        }

        LOGGER.info(f"[{ep_id}] Found {len(single_frame_tracks)} single-frame tracks, {len(multi_frame_track_ids)} multi-frame tracks")

        # Index all faces by frame_idx for fast lookup
        faces_by_frame: Dict[int, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
        for idx, face in enumerate(faces):
            frame_idx = face.get("frame_idx")
            if frame_idx is not None:
                faces_by_frame[frame_idx].append((idx, face))

        # Track modifications
        tracks_expanded = 0
        faces_merged = 0
        details = []
        modified_face_indices = set()

        for track_id, single_face in single_frame_tracks.items():
            embedding = single_face.get("embedding")
            frame_idx = single_face.get("frame_idx")

            if embedding is None or frame_idx is None:
                continue

            single_embed = np.array(embedding, dtype=np.float32)

            # Search adjacent frames for similar faces
            added_frames = []

            for delta in range(-frame_window, frame_window + 1):
                if delta == 0:
                    continue  # Skip the original frame

                check_frame = frame_idx + delta

                for face_idx, candidate in faces_by_frame.get(check_frame, []):
                    # Skip if already modified or in a multi-frame track
                    if face_idx in modified_face_indices:
                        continue

                    candidate_track_id = candidate.get("track_id")
                    if candidate_track_id in multi_frame_track_ids:
                        continue

                    # Skip if it's part of the same track already
                    if candidate_track_id == track_id:
                        continue

                    candidate_embedding = candidate.get("embedding")
                    if candidate_embedding is None:
                        continue

                    candidate_embed = np.array(candidate_embedding, dtype=np.float32)
                    similarity = cosine_similarity(single_embed, candidate_embed)

                    if similarity >= min_similarity:
                        # Merge this face into the single-frame track
                        old_track_id = candidate.get("track_id")
                        faces[face_idx]["track_id"] = track_id
                        modified_face_indices.add(face_idx)
                        added_frames.append({
                            "frame_idx": check_frame,
                            "similarity": round(similarity, 3),
                            "from_track": old_track_id,
                        })
                        LOGGER.debug(f"[{ep_id}] Merged face at frame {check_frame} (sim={similarity:.3f}) from track {old_track_id} -> {track_id}")

            if added_frames:
                tracks_expanded += 1
                faces_merged += len(added_frames)
                details.append({
                    "track_id": track_id,
                    "original_frame": frame_idx,
                    "added_frames": added_frames,
                })

        # Write updated faces if any changes were made
        if faces_merged > 0:
            write_faces(ep_id, faces)

            # Update tracks.jsonl statistics
            _update_tracks_statistics(ep_id, faces)

            # Sync manifests to S3
            sync_manifests(ep_id, _faces_path(ep_id), _tracks_path(ep_id))

            LOGGER.info(f"[{ep_id}] Track recovery complete: expanded {tracks_expanded} tracks with {faces_merged} faces")
        else:
            LOGGER.info(f"[{ep_id}] No tracks recovered")

        return {
            "tracks_analyzed": len(single_frame_tracks),
            "tracks_expanded": tracks_expanded,
            "faces_merged": faces_merged,
            "details": details,
        }


def _update_tracks_statistics(ep_id: str, faces: List[Dict[str, Any]]) -> None:
    """Update tracks.jsonl with updated face counts after recovery."""
    tracks = load_tracks(ep_id)

    if not tracks:
        return

    # Recount faces per track
    faces_by_track: Dict[int, int] = defaultdict(int)
    for face in faces:
        track_id = face.get("track_id")
        if track_id is not None:
            faces_by_track[track_id] += 1

    # Update track records
    for track in tracks:
        track_id = track.get("track_id")
        if track_id is not None and track_id in faces_by_track:
            track["faces_count"] = faces_by_track[track_id]

    write_tracks(ep_id, tracks)
