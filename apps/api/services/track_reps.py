"""Track representatives and cluster centroids computation service."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from py_screenalytics.artifacts import get_path

LOGGER = logging.getLogger(__name__)

# Quality gates for representative frames
REP_DET_MIN = float(os.getenv("REP_DET_MIN", "0.60"))
REP_STD_MIN = float(os.getenv("REP_STD_MIN", "1.0"))
REP_MAX_FRAMES_PER_TRACK = int(os.getenv("REP_MAX_FRAMES_PER_TRACK", "50"))


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = np.linalg.norm(vector) + 1e-12
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors (0-1)."""
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def _manifests_dir(ep_id: str) -> Path:
    return get_path(ep_id, "detections").parent


def _faces_path(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "faces.jsonl"


def _tracks_path(ep_id: str) -> Path:
    return get_path(ep_id, "tracks")


def _track_reps_path(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "track_reps.jsonl"


def _cluster_centroids_path(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "cluster_centroids.json"


def _identities_path(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "identities.json"


def _crops_root(ep_id: str) -> Path:
    return get_path(ep_id, "frames_root") / "crops"


def _discover_crop_path(ep_id: str, track_id: int, frame_idx: int) -> Optional[str]:
    """Find the crop file for a specific track and frame.

    Primary: data/frames/<ep_id>/crops/track_<ID>/frame_<frame_idx>.jpg
    Fallback 1: data/crops/<ep_id>/tracks/track_<ID>/frame_<frame_idx>.jpg
    Fallback 2: artifacts/crops/<show>/<season>/<episode>/tracks/track_<ID>/frame_<frame_idx>.jpg
    """
    track_component = f"track_{track_id:04d}"
    frame_filename = f"frame_{frame_idx:06d}.jpg"

    # Primary location
    frames_root = get_path(ep_id, "frames_root")
    primary = frames_root / "crops" / track_component / frame_filename
    if primary.exists():
        # Return relative path from crops root
        return f"crops/{track_component}/{frame_filename}"

    # Fallback locations
    fallback_root = Path(os.environ.get("SCREENALYTICS_CROPS_FALLBACK_ROOT", "data/crops")).expanduser()
    legacy1 = fallback_root / ep_id / "tracks" / track_component / frame_filename
    if legacy1.exists():
        return f"crops/{track_component}/{frame_filename}"

    # Try .png extension
    frame_filename_png = f"frame_{frame_idx:06d}.png"
    primary_png = frames_root / "crops" / track_component / frame_filename_png
    if primary_png.exists():
        return f"crops/{track_component}/{frame_filename_png}"

    legacy1_png = fallback_root / ep_id / "tracks" / track_component / frame_filename_png
    if legacy1_png.exists():
        return f"crops/{track_component}/{frame_filename_png}"

    return None


def _load_faces_for_track(ep_id: str, track_id: int) -> List[Dict[str, Any]]:
    """Load all faces for a specific track."""
    faces_path = _faces_path(ep_id)
    if not faces_path.exists():
        return []

    track_faces: List[Dict[str, Any]] = []
    with faces_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                face = json.loads(line)
            except json.JSONDecodeError:
                continue

            if face.get("track_id") == track_id:
                track_faces.append(face)

    return track_faces


def _passes_quality_gates(face: Dict[str, Any], crop_path: Optional[str]) -> bool:
    """Check if a face passes quality gates for representative selection."""
    # Detection confidence
    det_score = face.get("det_score") or face.get("conf") or face.get("quality", {}).get("det") or 0.0
    if det_score < REP_DET_MIN:
        return False

    # Crop standard deviation (sharpness)
    std = face.get("crop_std") or face.get("quality", {}).get("std") or 0.0
    if std < REP_STD_MIN:
        return False

    # Crop file must exist
    if not crop_path:
        return False

    # Skip flag
    if face.get("skip"):
        return False

    return True


def compute_track_representative(ep_id: str, track_id: int) -> Optional[Dict[str, Any]]:
    """Compute representative frame and centroid for a track.

    Returns:
        {
            "track_id": "track_0001",
            "rep_frame": 123,
            "crop_key": "crops/track_0001/frame_000123.jpg",
            "embed": [...512-d L2-norm...],
            "quality": {"det": 0.82, "std": 15.3}
        }
    """
    faces = _load_faces_for_track(ep_id, track_id)
    if not faces:
        return None

    # Sort by frame index
    faces.sort(key=lambda f: f.get("frame_idx", float('inf')))

    # Find first accepted frame
    rep_face = None
    rep_crop_key = None

    for face in faces:
        frame_idx = face.get("frame_idx")
        if frame_idx is None:
            continue

        crop_path = _discover_crop_path(ep_id, track_id, frame_idx)
        if _passes_quality_gates(face, crop_path):
            rep_face = face
            rep_crop_key = crop_path
            break

    # Fallback to first available crop if no frame passes gates
    if not rep_face:
        for face in faces:
            frame_idx = face.get("frame_idx")
            if frame_idx is None:
                continue
            crop_path = _discover_crop_path(ep_id, track_id, frame_idx)
            if crop_path:
                rep_face = face
                rep_crop_key = crop_path
                break

    if not rep_face:
        LOGGER.warning(f"No representative found for track {track_id} in {ep_id}")
        return None

    # Collect embeddings for track centroid (limited to first N accepted frames)
    embeddings: List[np.ndarray] = []
    for face in faces[:REP_MAX_FRAMES_PER_TRACK]:
        embedding = face.get("embedding")
        if not embedding:
            continue

        frame_idx = face.get("frame_idx")
        if frame_idx is None:
            continue

        crop_path = _discover_crop_path(ep_id, track_id, frame_idx)
        if _passes_quality_gates(face, crop_path):
            embeddings.append(np.array(embedding, dtype=np.float32))

    # Fallback: use all available embeddings if no accepted frames
    if not embeddings:
        for face in faces[:REP_MAX_FRAMES_PER_TRACK]:
            embedding = face.get("embedding")
            if embedding:
                embeddings.append(np.array(embedding, dtype=np.float32))

    if not embeddings:
        LOGGER.warning(f"No embeddings found for track {track_id} in {ep_id}")
        return None

    # Compute track centroid as mean of embeddings
    mean_embed = np.mean(embeddings, axis=0)
    track_centroid = l2_normalize(mean_embed)

    # Extract quality metrics
    det_score = rep_face.get("det_score") or rep_face.get("conf") or rep_face.get("quality", {}).get("det") or 0.0
    std = rep_face.get("crop_std") or rep_face.get("quality", {}).get("std") or 0.0

    return {
        "track_id": f"track_{track_id:04d}",
        "rep_frame": rep_face.get("frame_idx"),
        "crop_key": rep_crop_key,
        "embed": track_centroid.tolist(),
        "quality": {
            "det": round(float(det_score), 3),
            "std": round(float(std), 1),
        }
    }


def compute_all_track_reps(ep_id: str) -> List[Dict[str, Any]]:
    """Compute track representatives for all tracks in an episode."""
    tracks_path = _tracks_path(ep_id)
    if not tracks_path.exists():
        LOGGER.warning(f"No tracks file found for {ep_id}")
        return []

    track_reps: List[Dict[str, Any]] = []
    with tracks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                track = json.loads(line)
            except json.JSONDecodeError:
                continue

            track_id = track.get("track_id")
            if track_id is None:
                continue

            rep = compute_track_representative(ep_id, track_id)
            if rep:
                track_reps.append(rep)

    return track_reps


def write_track_reps(ep_id: str, track_reps: List[Dict[str, Any]]) -> Path:
    """Write track representatives to track_reps.jsonl."""
    path = _track_reps_path(ep_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for rep in track_reps:
            f.write(json.dumps(rep) + "\n")

    LOGGER.info(f"Wrote {len(track_reps)} track representatives to {path}")
    return path


def load_track_reps(ep_id: str) -> Dict[str, Dict[str, Any]]:
    """Load track representatives from track_reps.jsonl.

    Returns: Dict mapping track_id -> rep data
    """
    path = _track_reps_path(ep_id)
    if not path.exists():
        return {}

    reps: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rep = json.loads(line)
            except json.JSONDecodeError:
                continue

            track_id = rep.get("track_id")
            if track_id:
                reps[track_id] = rep

    return reps


def compute_cluster_centroids(ep_id: str, track_reps: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Compute cluster centroids from track representatives.

    Args:
        ep_id: Episode ID
        track_reps: Optional pre-loaded track representatives. If None, will load from disk.

    Returns:
        {
            "id_0001": {
                "centroid": [...512-d...],
                "tracks": ["track_0001", "track_0022", ...],
                "cohesion": 0.74
            },
            ...
        }
    """
    if track_reps is None:
        track_reps = load_track_reps(ep_id)

    if not track_reps:
        LOGGER.warning(f"No track representatives found for {ep_id}")
        return {}

    # Load identities to get cluster assignments
    identities_path = _identities_path(ep_id)
    if not identities_path.exists():
        LOGGER.warning(f"No identities file found for {ep_id}")
        return {}

    identities_data = json.loads(identities_path.read_text(encoding="utf-8"))
    identities = identities_data.get("identities", [])

    # Build cluster -> track mapping
    cluster_to_tracks: Dict[str, List[str]] = {}
    for identity in identities:
        cluster_id = identity.get("identity_id")
        if not cluster_id:
            continue

        track_ids = identity.get("track_ids", [])
        # Normalize track IDs to track_XXXX format
        normalized_tracks = []
        for tid in track_ids:
            if isinstance(tid, int):
                normalized_tracks.append(f"track_{tid:04d}")
            elif isinstance(tid, str):
                if tid.startswith("track_"):
                    normalized_tracks.append(tid)
                else:
                    try:
                        normalized_tracks.append(f"track_{int(tid):04d}")
                    except (TypeError, ValueError):
                        continue

        cluster_to_tracks[cluster_id] = normalized_tracks

    # Compute centroids
    centroids: Dict[str, Any] = {}
    for cluster_id, track_ids in cluster_to_tracks.items():
        # Collect track centroids for this cluster
        track_centroids: List[np.ndarray] = []
        valid_tracks: List[str] = []

        for track_id in track_ids:
            rep = track_reps.get(track_id)
            if not rep:
                continue

            embed = rep.get("embed")
            if not embed:
                continue

            track_centroids.append(np.array(embed, dtype=np.float32))
            valid_tracks.append(track_id)

        if not track_centroids:
            continue

        # Compute cluster centroid as mean of track centroids
        mean_centroid = np.mean(track_centroids, axis=0)
        cluster_centroid = l2_normalize(mean_centroid)

        # Compute cohesion as mean similarity of tracks to centroid
        similarities = [cosine_similarity(tc, cluster_centroid) for tc in track_centroids]
        cohesion = float(np.mean(similarities))

        centroids[cluster_id] = {
            "centroid": cluster_centroid.tolist(),
            "tracks": valid_tracks,
            "cohesion": round(cohesion, 3),
        }

    return centroids


def write_cluster_centroids(ep_id: str, centroids: Dict[str, Any]) -> Path:
    """Write cluster centroids to cluster_centroids.json."""
    path = _cluster_centroids_path(ep_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "ep_id": ep_id,
        "centroids": centroids,
        "computed_at": _now_iso(),
    }

    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    LOGGER.info(f"Wrote {len(centroids)} cluster centroids to {path}")
    return path


def load_cluster_centroids(ep_id: str) -> Dict[str, Any]:
    """Load cluster centroids from cluster_centroids.json."""
    path = _cluster_centroids_path(ep_id)
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("centroids", {})
    except json.JSONDecodeError:
        LOGGER.warning(f"Failed to parse cluster centroids for {ep_id}")
        return {}


def build_cluster_track_reps(
    ep_id: str,
    cluster_id: str,
    track_reps: Optional[Dict[str, Dict[str, Any]]] = None,
    cluster_centroids: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build track representatives with similarity scores for a cluster.

    Returns:
        {
            "cluster_id": "id_0001",
            "cluster_centroid": [...],
            "cohesion": 0.74,
            "tracks": [
                {
                    "track_id": "track_0001",
                    "rep_frame": 123,
                    "crop_key": "crops/track_0001/frame_000123.jpg",
                    "similarity": 0.82,
                    "quality": {"det": 0.82, "std": 15.3}
                },
                ...
            ],
            "total_tracks": 13
        }
    """
    if track_reps is None:
        track_reps = load_track_reps(ep_id)

    if cluster_centroids is None:
        cluster_centroids = load_cluster_centroids(ep_id)

    cluster_data = cluster_centroids.get(cluster_id)
    if not cluster_data:
        return {
            "cluster_id": cluster_id,
            "cluster_centroid": None,
            "cohesion": None,
            "tracks": [],
            "total_tracks": 0,
        }

    cluster_centroid_vec = np.array(cluster_data["centroid"], dtype=np.float32)
    track_ids = cluster_data.get("tracks", [])
    cohesion = cluster_data.get("cohesion")

    # Build track reps with similarity scores
    tracks_output: List[Dict[str, Any]] = []
    for track_id in track_ids:
        rep = track_reps.get(track_id)
        if not rep:
            # Include missing tracks with placeholder data
            tracks_output.append({
                "track_id": track_id,
                "rep_frame": None,
                "crop_key": None,
                "similarity": None,
                "quality": None,
            })
            continue

        # Compute similarity
        track_centroid_vec = np.array(rep["embed"], dtype=np.float32)
        similarity = cosine_similarity(track_centroid_vec, cluster_centroid_vec)

        tracks_output.append({
            "track_id": track_id,
            "rep_frame": rep.get("rep_frame"),
            "crop_key": rep.get("crop_key"),
            "similarity": round(float(similarity), 3),
            "quality": rep.get("quality"),
        })

    return {
        "cluster_id": cluster_id,
        "cluster_centroid": cluster_data["centroid"],
        "cohesion": cohesion,
        "tracks": tracks_output,
        "total_tracks": len(tracks_output),
    }


def generate_track_reps_and_centroids(ep_id: str) -> Dict[str, Any]:
    """Complete pipeline: compute track reps and cluster centroids.

    Returns: Summary with paths and counts
    """
    LOGGER.info(f"Generating track representatives for {ep_id}")
    track_reps_list = compute_all_track_reps(ep_id)
    track_reps_map = {rep["track_id"]: rep for rep in track_reps_list}

    track_reps_path = write_track_reps(ep_id, track_reps_list)

    LOGGER.info(f"Generating cluster centroids for {ep_id}")
    centroids = compute_cluster_centroids(ep_id, track_reps_map)

    centroids_path = write_cluster_centroids(ep_id, centroids)

    return {
        "track_reps_count": len(track_reps_list),
        "track_reps_path": str(track_reps_path),
        "cluster_centroids_count": len(centroids),
        "cluster_centroids_path": str(centroids_path),
    }
