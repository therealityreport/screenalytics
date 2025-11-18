#!/usr/bin/env python
"""Split high face_embedding_spread tracks into multiple sub-tracks.

This module provides functionality to automatically split tracks that contain
multiple people (detected via high face_embedding_spread) into separate tracks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

try:
    from sklearn.cluster import AgglomerativeClustering

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from py_screenalytics.artifacts import get_path

LOGGER = logging.getLogger(__name__)


def split_high_spread_tracks(
    ep_id: str,
    spread_threshold: float = 0.35,
    *,
    dry_run: bool = False,
    manifests_dir: Path | None = None,
) -> Dict[str, Any]:
    """Automatically split tracks with high face_embedding_spread.

    Args:
        ep_id: Episode identifier
        spread_threshold: Maximum allowed spread (default: 0.35)
        dry_run: If True, only report what would be split without modifying files
        manifests_dir: Optional manifests directory (for testing)

    Returns:
        Dict with split statistics and track mapping
    """
    if not HAS_SKLEARN:
        raise RuntimeError("sklearn required for track splitting")

    if manifests_dir is None:
        manifests_dir = get_path(ep_id, "detections").parent
    tracks_path = manifests_dir / "tracks.jsonl"
    faces_path = manifests_dir / "faces.jsonl"

    if not tracks_path.exists() or not faces_path.exists():
        raise FileNotFoundError(f"Missing tracks.jsonl or faces.jsonl for {ep_id}")

    # Load all tracks
    tracks = []
    with tracks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tracks.append(json.loads(line))

    # Identify tracks needing split
    flagged_tracks: Set[int] = set()
    for track in tracks:
        track_id = int(track["track_id"])
        spread = track.get("face_embedding_spread")
        if spread is not None and float(spread) >= spread_threshold:
            flagged_tracks.add(track_id)

    if not flagged_tracks:
        LOGGER.info("[split_tracks] No tracks flagged for splitting")
        return {"split_count": 0, "flagged_tracks": []}

    LOGGER.info(
        "[split_tracks] Found %d tracks with spread >= %.3f",
        len(flagged_tracks),
        spread_threshold,
    )

    # Load all faces
    faces_by_track: Dict[int, List[Dict[str, Any]]] = {}
    with faces_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            face = json.loads(line)
            track_id = int(face["track_id"])
            if track_id in flagged_tracks:
                faces_by_track.setdefault(track_id, []).append(face)

    # Split each flagged track
    track_mapping: Dict[int, List[int]] = {}  # old_track_id -> [new_track_ids]
    next_track_id = max(t["track_id"] for t in tracks) + 1
    new_tracks: List[Dict[str, Any]] = []
    new_faces: List[Dict[str, Any]] = []

    for old_track_id in sorted(flagged_tracks):
        track_faces = faces_by_track.get(old_track_id, [])
        if len(track_faces) < 2:
            LOGGER.warning(
                f"[split_tracks] Track {old_track_id} has <2 faces, skipping"
            )
            continue

        # Extract embeddings
        embeddings = []
        for face in track_faces:
            emb = face.get("embedding")
            if emb:
                embeddings.append(np.array(emb, dtype=np.float32))

        if len(embeddings) < 2:
            LOGGER.warning(
                f"[split_tracks] Track {old_track_id} has <2 embeddings, skipping"
            )
            continue

        # Cluster faces within this track
        X = np.vstack(embeddings)
        n_clusters = min(len(embeddings), 5)  # Max 5 sub-tracks
        model = AgglomerativeClustering(
            n_clusters=n_clusters, metric="cosine", linkage="average"
        )
        labels = model.fit_predict(X)

        # Group faces by sub-cluster
        sub_groups: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            sub_groups.setdefault(int(label), []).append(idx)

        # Only split if we found multiple clusters
        unique_labels = set(labels)
        if len(unique_labels) == 1:
            LOGGER.info(
                f"[split_tracks] Track {old_track_id} clustered into 1 group, not splitting"
            )
            continue

        LOGGER.info(
            f"[split_tracks] Splitting track {old_track_id} into {len(unique_labels)} sub-tracks"
        )

        # Create new tracks and reassign faces
        new_track_ids = []
        for label, face_indices in sub_groups.items():
            new_track_id = next_track_id
            next_track_id += 1
            new_track_ids.append(new_track_id)

            # Create new track record (simplified - reuses original track metadata)
            original_track = next(t for t in tracks if t["track_id"] == old_track_id)
            new_track = {
                **original_track,
                "track_id": new_track_id,
                "stats": {
                    **original_track.get("stats", {}),
                    "split_from": old_track_id,
                    "sub_cluster": int(label),
                },
            }
            # Remove spread since sub-track should have lower spread
            new_track.pop("face_embedding_spread", None)
            new_tracks.append(new_track)

            # Reassign faces to new track
            for face_idx in face_indices:
                face = track_faces[face_idx].copy()
                face["track_id"] = new_track_id
                new_faces.append(face)

        track_mapping[old_track_id] = new_track_ids

    if dry_run:
        LOGGER.info(
            "[split_tracks] DRY RUN - would split %d tracks", len(track_mapping)
        )
        return {
            "dry_run": True,
            "split_count": len(track_mapping),
            "track_mapping": track_mapping,
        }

    # Write updated files
    # 1. Write new tracks (excluding old flagged ones)
    with tracks_path.open("w", encoding="utf-8") as f:
        for track in tracks:
            if track["track_id"] not in flagged_tracks:
                f.write(json.dumps(track) + "\n")
        for track in new_tracks:
            f.write(json.dumps(track) + "\n")

    # 2. Write new faces (excluding old flagged track faces)
    with faces_path.open("w", encoding="utf-8") as f_out:
        # Copy faces from non-flagged tracks
        with Path(str(faces_path) + ".bak").open("w", encoding="utf-8") as f_bak:
            with faces_path.open("r", encoding="utf-8") as f_in:
                for line in f_in:
                    f_bak.write(line)  # Backup
                    if not line.strip():
                        continue
                    face = json.loads(line)
                    if int(face["track_id"]) not in flagged_tracks:
                        f_out.write(line)

        # Write new faces from split tracks
        for face in new_faces:
            f_out.write(json.dumps(face) + "\n")

    LOGGER.info(
        "[split_tracks] Split %d tracks into %d new tracks",
        len(track_mapping),
        sum(len(v) for v in track_mapping.values()),
    )

    return {
        "split_count": len(track_mapping),
        "track_mapping": track_mapping,
        "new_tracks_count": len(new_tracks),
        "new_faces_count": len(new_faces),
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Split high-spread tracks")
    parser.add_argument("--ep-id", required=True, help="Episode ID")
    parser.add_argument(
        "--threshold", type=float, default=0.35, help="Spread threshold"
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files")
    args = parser.parse_args()

    result = split_high_spread_tracks(args.ep_id, args.threshold, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
