"""Metrics computation service for SCREENALYTICS.

Computes similarity, cohesion, quality, and new metrics (Nov 2024):
- Temporal Consistency
- Ambiguity Score
- Cluster Isolation
- Confidence Trend

All metrics are normalized to 0-1 scale.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from py_screenalytics.artifacts import get_path
from apps.api.services.track_reps import (
    load_track_reps,
    load_cluster_centroids,
    cosine_similarity,
    l2_normalize,
)

LOGGER = logging.getLogger(__name__)


def _manifests_dir(ep_id: str) -> Path:
    return get_path(ep_id, "detections").parent


def _load_faces(ep_id: str) -> List[Dict[str, Any]]:
    """Load all faces from faces.jsonl."""
    faces_path = _manifests_dir(ep_id) / "faces.jsonl"
    if not faces_path.exists():
        return []

    faces = []
    with faces_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                faces.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return faces


def _load_identities(ep_id: str) -> Dict[str, Any]:
    """Load identities.json."""
    path = _manifests_dir(ep_id) / "identities.json"
    if not path.exists():
        return {"identities": []}
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================================
# CLUSTER METRICS
# ============================================================================


def compute_cluster_cohesion(
    ep_id: str,
    cluster_id: str,
    track_reps: Optional[Dict[str, Dict[str, Any]]] = None,
    cluster_centroids: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute cohesion metrics for a cluster.

    Returns:
        {
            "cohesion": float,      # Average similarity of tracks to centroid
            "min_similarity": float,
            "max_similarity": float,
            "track_count": int,
        }
    """
    if track_reps is None:
        track_reps = load_track_reps(ep_id)
    if cluster_centroids is None:
        cluster_centroids = load_cluster_centroids(ep_id)

    cluster_data = cluster_centroids.get(cluster_id)
    if not cluster_data:
        return {"cohesion": None, "min_similarity": None, "max_similarity": None, "track_count": 0}

    centroid = cluster_data.get("centroid")
    if not centroid:
        return {"cohesion": None, "min_similarity": None, "max_similarity": None, "track_count": 0}

    centroid_vec = np.array(centroid, dtype=np.float32)
    track_ids = cluster_data.get("tracks", [])

    similarities = []
    for track_id in track_ids:
        rep = track_reps.get(track_id)
        if not rep or not rep.get("embed"):
            continue
        track_embed = np.array(rep["embed"], dtype=np.float32)
        sim = cosine_similarity(track_embed, centroid_vec)
        similarities.append(sim)

    if not similarities:
        return {"cohesion": None, "min_similarity": None, "max_similarity": None, "track_count": len(track_ids)}

    return {
        "cohesion": float(np.mean(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "track_count": len(track_ids),
    }


def compute_cluster_isolation(
    ep_id: str,
    cluster_id: str,
    cluster_centroids: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute isolation (distance to nearest cluster) for a cluster.

    Returns:
        {
            "isolation": float,           # 1 - similarity to nearest cluster (higher = more isolated)
            "nearest_cluster": str,       # ID of nearest cluster
            "nearest_similarity": float,  # Similarity to nearest cluster
        }
    """
    if cluster_centroids is None:
        cluster_centroids = load_cluster_centroids(ep_id)

    cluster_data = cluster_centroids.get(cluster_id)
    if not cluster_data or not cluster_data.get("centroid"):
        return {"isolation": None, "nearest_cluster": None, "nearest_similarity": None}

    target_centroid = np.array(cluster_data["centroid"], dtype=np.float32)

    nearest_cluster = None
    nearest_similarity = -1.0

    for other_id, other_data in cluster_centroids.items():
        if other_id == cluster_id:
            continue
        other_centroid = other_data.get("centroid")
        if not other_centroid:
            continue

        other_vec = np.array(other_centroid, dtype=np.float32)
        sim = cosine_similarity(target_centroid, other_vec)

        if sim > nearest_similarity:
            nearest_similarity = sim
            nearest_cluster = other_id

    if nearest_cluster is None:
        return {"isolation": 1.0, "nearest_cluster": None, "nearest_similarity": None}

    # Isolation = 1 - similarity (higher similarity = lower isolation)
    isolation = 1.0 - nearest_similarity

    return {
        "isolation": float(isolation),
        "nearest_cluster": nearest_cluster,
        "nearest_similarity": float(nearest_similarity),
    }


def compute_cluster_ambiguity(
    ep_id: str,
    cluster_id: str,
    cluster_centroids: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute ambiguity score (gap between 1st and 2nd best match).

    For clusters assigned to a person, this measures how clearly they belong
    to that person vs the next best match.

    Returns:
        {
            "ambiguity": float,           # Gap between 1st and 2nd best (higher = clearer)
            "first_match": str,           # Best match cluster ID
            "first_similarity": float,
            "second_match": str,          # Second best cluster ID
            "second_similarity": float,
        }
    """
    if cluster_centroids is None:
        cluster_centroids = load_cluster_centroids(ep_id)

    cluster_data = cluster_centroids.get(cluster_id)
    if not cluster_data or not cluster_data.get("centroid"):
        return {"ambiguity": None, "first_match": None, "second_match": None}

    target_centroid = np.array(cluster_data["centroid"], dtype=np.float32)

    # Compute similarities to all other clusters
    similarities = []
    for other_id, other_data in cluster_centroids.items():
        if other_id == cluster_id:
            continue
        other_centroid = other_data.get("centroid")
        if not other_centroid:
            continue

        other_vec = np.array(other_centroid, dtype=np.float32)
        sim = cosine_similarity(target_centroid, other_vec)
        similarities.append((other_id, sim))

    if len(similarities) < 2:
        # Not enough clusters to compute ambiguity
        return {"ambiguity": 1.0, "first_match": None, "second_match": None}

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    first_id, first_sim = similarities[0]
    second_id, second_sim = similarities[1]

    # Ambiguity = gap between 1st and 2nd
    ambiguity = first_sim - second_sim

    return {
        "ambiguity": float(ambiguity),
        "first_match": first_id,
        "first_similarity": float(first_sim),
        "second_match": second_id,
        "second_similarity": float(second_sim),
    }


# ============================================================================
# TRACK METRICS
# ============================================================================


def compute_track_consistency(
    ep_id: str,
    track_id: int,
) -> Dict[str, Any]:
    """Compute frame-to-frame consistency within a track.

    Returns:
        {
            "track_similarity": float,    # Average similarity of frames to track centroid
            "min_similarity": float,
            "max_similarity": float,
            "excluded_frames": int,       # Frames below threshold
            "total_frames": int,
        }
    """
    faces = _load_faces(ep_id)
    track_faces = [f for f in faces if f.get("track_id") == track_id and f.get("embedding")]

    if not track_faces:
        return {
            "track_similarity": None,
            "min_similarity": None,
            "max_similarity": None,
            "excluded_frames": 0,
            "total_frames": 0,
        }

    # Compute track centroid
    embeddings = [np.array(f["embedding"], dtype=np.float32) for f in track_faces]
    centroid = l2_normalize(np.mean(embeddings, axis=0))

    # Compute similarities
    similarities = []
    for embed in embeddings:
        sim = cosine_similarity(embed, centroid)
        similarities.append(sim)

    # Count excluded frames (below threshold)
    threshold = 0.50
    excluded = sum(1 for s in similarities if s < threshold)

    return {
        "track_similarity": float(np.mean(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "excluded_frames": excluded,
        "total_frames": len(similarities),
    }


def compute_person_cohesion(
    ep_id: str,
    track_id: int,
    person_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute how well a track fits with other tracks of the same person.

    Returns:
        {
            "person_cohesion": float,     # Similarity to person's centroid
            "person_id": str,
        }
    """
    # Load identities to find the person for this track
    identities = _load_identities(ep_id)
    track_reps = load_track_reps(ep_id)

    # Find the identity containing this track
    target_identity = None
    for identity in identities.get("identities", []):
        track_ids = identity.get("track_ids", [])
        if track_id in track_ids or str(track_id) in [str(t) for t in track_ids]:
            target_identity = identity
            break

    if not target_identity:
        return {"person_cohesion": None, "person_id": None}

    person_id = person_id or target_identity.get("person_id")
    track_ids = target_identity.get("track_ids", [])

    if len(track_ids) <= 1:
        # Only one track, can't compute cohesion
        return {"person_cohesion": 1.0, "person_id": person_id}

    # Compute centroid from all tracks except this one
    track_id_str = f"track_{track_id:04d}"
    other_embeddings = []
    for tid in track_ids:
        tid_str = f"track_{tid:04d}" if isinstance(tid, int) else tid
        if tid_str == track_id_str:
            continue
        rep = track_reps.get(tid_str)
        if rep and rep.get("embed"):
            other_embeddings.append(np.array(rep["embed"], dtype=np.float32))

    if not other_embeddings:
        return {"person_cohesion": None, "person_id": person_id}

    # Compute centroid of other tracks
    other_centroid = l2_normalize(np.mean(other_embeddings, axis=0))

    # Get this track's embedding
    this_rep = track_reps.get(track_id_str)
    if not this_rep or not this_rep.get("embed"):
        return {"person_cohesion": None, "person_id": person_id}

    this_embed = np.array(this_rep["embed"], dtype=np.float32)
    cohesion = cosine_similarity(this_embed, other_centroid)

    return {
        "person_cohesion": float(cohesion),
        "person_id": person_id,
    }


# ============================================================================
# TEMPORAL METRICS
# ============================================================================


def compute_temporal_consistency(
    ep_id: str,
    cluster_id: Optional[str] = None,
    person_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute temporal consistency of appearance across time.

    Measures how consistent a person/cluster looks across different times
    in the episode. Lower scores may indicate costume changes, lighting
    changes, or potential misassignments.

    Returns:
        {
            "temporal_consistency": float,  # 0-1, higher = more consistent
            "time_span_seconds": float,     # Duration covered
            "sample_count": int,
        }
    """
    faces = _load_faces(ep_id)
    identities = _load_identities(ep_id)

    # Find tracks for this cluster/person
    track_ids = set()
    if cluster_id:
        for identity in identities.get("identities", []):
            if identity.get("identity_id") == cluster_id:
                track_ids.update(identity.get("track_ids", []))
                break
    elif person_id:
        for identity in identities.get("identities", []):
            if identity.get("person_id") == person_id:
                track_ids.update(identity.get("track_ids", []))

    if not track_ids:
        return {"temporal_consistency": None, "time_span_seconds": 0, "sample_count": 0}

    # Get faces for these tracks with timestamps
    track_faces = []
    for face in faces:
        if face.get("track_id") in track_ids and face.get("embedding"):
            ts = face.get("ts") or face.get("timestamp")
            if ts is not None:
                track_faces.append({
                    "ts": ts,
                    "embedding": np.array(face["embedding"], dtype=np.float32),
                })

    if len(track_faces) < 2:
        return {"temporal_consistency": 1.0, "time_span_seconds": 0, "sample_count": len(track_faces)}

    # Sort by timestamp
    track_faces.sort(key=lambda f: f["ts"])

    # Compute time span
    time_span = track_faces[-1]["ts"] - track_faces[0]["ts"]

    # Sample faces at different time points (to avoid over-representing dense sections)
    # Divide into 10 time bins and pick one from each
    n_bins = min(10, len(track_faces))
    bin_size = len(track_faces) // n_bins
    sampled = []
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size if i < n_bins - 1 else len(track_faces)
        # Pick the middle face from each bin
        mid_idx = (start_idx + end_idx) // 2
        sampled.append(track_faces[mid_idx])

    if len(sampled) < 2:
        return {"temporal_consistency": 1.0, "time_span_seconds": time_span, "sample_count": len(track_faces)}

    # Compute all pairwise similarities between time samples
    embeddings = [f["embedding"] for f in sampled]
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)

    consistency = float(np.mean(similarities))

    return {
        "temporal_consistency": consistency,
        "time_span_seconds": time_span,
        "sample_count": len(track_faces),
    }


# ============================================================================
# QUALITY METRICS
# ============================================================================


def compute_aggregate_quality(
    ep_id: str,
    cluster_id: Optional[str] = None,
    track_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute aggregate quality score for a cluster or track.

    Returns:
        {
            "avg_quality": float,
            "min_quality": float,
            "median_quality": float,
            "quality_breakdown": {
                "det": float,      # Average detection confidence
                "sharp": float,    # Average sharpness
                "area": float,     # Average face area
            }
        }
    """
    track_reps = load_track_reps(ep_id)
    identities = _load_identities(ep_id)

    # Determine which tracks to aggregate
    track_ids = []
    if track_id is not None:
        track_ids = [f"track_{track_id:04d}"]
    elif cluster_id:
        for identity in identities.get("identities", []):
            if identity.get("identity_id") == cluster_id:
                for tid in identity.get("track_ids", []):
                    track_ids.append(f"track_{tid:04d}" if isinstance(tid, int) else tid)
                break

    if not track_ids:
        return {"avg_quality": None, "min_quality": None, "median_quality": None, "quality_breakdown": None}

    # Collect quality scores
    qualities = []
    det_scores = []
    sharp_scores = []
    area_scores = []

    for tid in track_ids:
        rep = track_reps.get(tid)
        if not rep:
            continue
        quality = rep.get("quality", {})
        if isinstance(quality, dict):
            score = quality.get("score")
            if score is not None:
                qualities.append(score)
            if quality.get("det") is not None:
                det_scores.append(quality["det"])
            if quality.get("std") is not None:
                # Normalize std to 0-1 (assuming max ~100)
                sharp_scores.append(min(quality["std"] / 100.0, 1.0))
            if quality.get("box_area") is not None:
                # Normalize area to 0-1 (assuming max ~100000)
                area_scores.append(min(quality["box_area"] / 100000.0, 1.0))

    if not qualities:
        return {"avg_quality": None, "min_quality": None, "median_quality": None, "quality_breakdown": None}

    breakdown = {}
    if det_scores:
        breakdown["det"] = float(np.mean(det_scores))
    if sharp_scores:
        breakdown["sharp"] = float(np.mean(sharp_scores))
    if area_scores:
        breakdown["area"] = float(np.mean(area_scores))

    return {
        "avg_quality": float(np.mean(qualities)),
        "min_quality": float(np.min(qualities)),
        "median_quality": float(np.median(qualities)),
        "quality_breakdown": breakdown if breakdown else None,
    }


# ============================================================================
# COMBINED METRICS
# ============================================================================


def compute_all_cluster_metrics(
    ep_id: str,
    cluster_id: str,
) -> Dict[str, Any]:
    """Compute all metrics for a cluster.

    Returns combined dict with all cluster-level metrics.
    """
    # Load shared data once
    track_reps = load_track_reps(ep_id)
    cluster_centroids = load_cluster_centroids(ep_id)

    # Compute individual metrics
    cohesion = compute_cluster_cohesion(ep_id, cluster_id, track_reps, cluster_centroids)
    isolation = compute_cluster_isolation(ep_id, cluster_id, cluster_centroids)
    ambiguity = compute_cluster_ambiguity(ep_id, cluster_id, cluster_centroids)
    temporal = compute_temporal_consistency(ep_id, cluster_id=cluster_id)
    quality = compute_aggregate_quality(ep_id, cluster_id=cluster_id)

    return {
        "cluster_id": cluster_id,
        # Cohesion metrics
        "cohesion": cohesion.get("cohesion"),
        "min_similarity": cohesion.get("min_similarity"),
        "max_similarity": cohesion.get("max_similarity"),
        "track_count": cohesion.get("track_count"),
        # Isolation
        "isolation": isolation.get("isolation"),
        "nearest_cluster": isolation.get("nearest_cluster"),
        "nearest_similarity": isolation.get("nearest_similarity"),
        # Ambiguity
        "ambiguity": ambiguity.get("ambiguity"),
        "first_match": ambiguity.get("first_match"),
        "second_match": ambiguity.get("second_match"),
        # Temporal
        "temporal_consistency": temporal.get("temporal_consistency"),
        "time_span_seconds": temporal.get("time_span_seconds"),
        # Quality
        "avg_quality": quality.get("avg_quality"),
        "min_quality": quality.get("min_quality"),
        "quality_breakdown": quality.get("quality_breakdown"),
    }


def compute_all_track_metrics(
    ep_id: str,
    track_id: int,
) -> Dict[str, Any]:
    """Compute all metrics for a track.

    Returns combined dict with all track-level metrics.
    """
    consistency = compute_track_consistency(ep_id, track_id)
    person_cohesion = compute_person_cohesion(ep_id, track_id)
    quality = compute_aggregate_quality(ep_id, track_id=track_id)

    return {
        "track_id": track_id,
        # Track consistency
        "track_similarity": consistency.get("track_similarity"),
        "min_frame_similarity": consistency.get("min_similarity"),
        "max_frame_similarity": consistency.get("max_similarity"),
        "excluded_frames": consistency.get("excluded_frames"),
        "total_frames": consistency.get("total_frames"),
        # Person cohesion
        "person_cohesion": person_cohesion.get("person_cohesion"),
        "person_id": person_cohesion.get("person_id"),
        # Quality
        "avg_quality": quality.get("avg_quality"),
        "quality_breakdown": quality.get("quality_breakdown"),
    }
