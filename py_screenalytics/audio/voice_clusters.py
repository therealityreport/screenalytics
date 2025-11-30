"""Voice clustering for speaker identification.

Handles:
- Clustering diarization segments into unique voices
- Computing voice cluster centroids
- Managing episode-local voice cluster IDs
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import (
    DiarizationSegment,
    VoiceCluster,
    VoiceClusterSegment,
    VoiceClusteringConfig,
)

LOGGER = logging.getLogger(__name__)


def cluster_episode_voices(
    audio_path: Path,
    diarization_segments: List[DiarizationSegment],
    output_path: Path,
    config: Optional[VoiceClusteringConfig] = None,
    overwrite: bool = False,
) -> List[VoiceCluster]:
    """Cluster diarization segments into unique voices.

    Args:
        audio_path: Path to audio file
        diarization_segments: List of diarization segments
        output_path: Path for voice clusters JSON
        config: Voice clustering configuration
        overwrite: Whether to overwrite existing results

    Returns:
        List of VoiceCluster objects
    """
    if output_path.exists() and not overwrite:
        LOGGER.info(f"Voice clusters already exist: {output_path}")
        return _load_voice_clusters(output_path)

    config = config or VoiceClusteringConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Clustering {len(diarization_segments)} segments into voice clusters")

    # Extract embeddings for each segment
    from .diarization_pyannote import extract_speaker_embeddings

    segment_embeddings = extract_speaker_embeddings(
        audio_path,
        diarization_segments,
        config.embedding_model,
    )

    if not segment_embeddings:
        LOGGER.warning("No embeddings extracted, creating clusters from diarization labels")
        clusters = _clusters_from_diarization_labels(diarization_segments)
        _save_voice_clusters(clusters, output_path)
        return clusters

    # Check how many unique speakers diarization found
    unique_diar_speakers = set(seg.speaker for seg in diarization_segments)
    LOGGER.info(f"Diarization found {len(unique_diar_speakers)} unique speakers")

    # If diarization found very few speakers (1-2), do embedding-based clustering
    # to properly separate voices that diarization missed
    if len(unique_diar_speakers) <= 2 and len(segment_embeddings) >= 3:
        LOGGER.info("Diarization found few speakers - using embedding-based clustering")
        clusters = _cluster_by_embeddings(
            segment_embeddings,
            config.similarity_threshold,
            config.min_segments_per_cluster,
            config.centroid_method,
        )
    else:
        # Group embeddings by diarization speaker label
        speaker_embeddings: Dict[str, List[Tuple[DiarizationSegment, np.ndarray]]] = defaultdict(list)
        for segment, embedding in segment_embeddings:
            speaker_embeddings[segment.speaker].append((segment, embedding))

        # Refine clusters using embeddings
        clusters = _refine_clusters_with_embeddings(
            speaker_embeddings,
            config.similarity_threshold,
            config.min_segments_per_cluster,
            config.centroid_method,
        )

    # Assign stable IDs
    clusters = _assign_cluster_ids(clusters)

    # Save clusters
    _save_voice_clusters(clusters, output_path)

    LOGGER.info(f"Created {len(clusters)} voice clusters")

    return clusters


def _cluster_by_embeddings(
    segment_embeddings: List[Tuple[DiarizationSegment, np.ndarray]],
    similarity_threshold: float,
    min_segments: int,
    centroid_method: str,
) -> List[VoiceCluster]:
    """Cluster segments purely by embedding similarity, ignoring diarization labels.

    Uses agglomerative clustering with cosine similarity.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    if len(segment_embeddings) < 2:
        # Not enough for clustering
        return _single_cluster_from_embeddings(segment_embeddings, centroid_method)

    # Extract embeddings matrix
    embeddings = np.array([emb for _, emb in segment_embeddings])

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    # Compute pairwise cosine distances (1 - similarity)
    # Using correlation distance as proxy for cosine on normalized vectors
    distances = pdist(embeddings, metric='cosine')

    # Hierarchical clustering
    Z = linkage(distances, method='average')

    # Cut at threshold (convert similarity to distance)
    distance_threshold = 1 - similarity_threshold
    cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')

    # Group segments by cluster
    clusters_dict: Dict[int, List[Tuple[DiarizationSegment, np.ndarray]]] = defaultdict(list)
    for i, (segment, embedding) in enumerate(segment_embeddings):
        cluster_id = cluster_labels[i]
        clusters_dict[cluster_id].append((segment, embedding))

    LOGGER.info(f"Embedding clustering found {len(clusters_dict)} clusters")

    # Create VoiceCluster objects
    clusters = []
    for cluster_id, items in clusters_dict.items():
        if len(items) < min_segments:
            LOGGER.debug(f"Skipping cluster {cluster_id}: only {len(items)} segments")
            continue

        segments = [
            VoiceClusterSegment(
                start=seg.start,
                end=seg.end,
                diar_speaker=seg.speaker,
            )
            for seg, _ in items
        ]

        total_duration = sum(seg.end - seg.start for seg, _ in items)

        # Compute centroid
        cluster_embeddings = np.array([emb for _, emb in items])
        if centroid_method == "median":
            centroid = np.median(cluster_embeddings, axis=0)
        else:
            centroid = np.mean(cluster_embeddings, axis=0)

        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        cluster = VoiceCluster(
            voice_cluster_id="",  # Will be assigned later
            segments=segments,
            total_duration=total_duration,
            segment_count=len(items),
            centroid=centroid.tolist(),
        )
        clusters.append(cluster)

    return clusters


def _single_cluster_from_embeddings(
    segment_embeddings: List[Tuple[DiarizationSegment, np.ndarray]],
    centroid_method: str,
) -> List[VoiceCluster]:
    """Create a single cluster from all segments."""
    if not segment_embeddings:
        return []

    segments = [
        VoiceClusterSegment(
            start=seg.start,
            end=seg.end,
            diar_speaker=seg.speaker,
        )
        for seg, _ in segment_embeddings
    ]

    total_duration = sum(seg.end - seg.start for seg, _ in segment_embeddings)

    embeddings = np.array([emb for _, emb in segment_embeddings])
    if centroid_method == "median":
        centroid = np.median(embeddings, axis=0)
    else:
        centroid = np.mean(embeddings, axis=0)

    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    return [VoiceCluster(
        voice_cluster_id="VC_01",
        segments=segments,
        total_duration=total_duration,
        segment_count=len(segments),
        centroid=centroid.tolist(),
    )]


def _clusters_from_diarization_labels(
    segments: List[DiarizationSegment],
) -> List[VoiceCluster]:
    """Create voice clusters directly from diarization speaker labels."""
    speaker_segments: Dict[str, List[DiarizationSegment]] = defaultdict(list)

    for segment in segments:
        speaker_segments[segment.speaker].append(segment)

    clusters = []
    for i, (speaker, segs) in enumerate(sorted(speaker_segments.items())):
        cluster_segments = [
            VoiceClusterSegment(
                start=s.start,
                end=s.end,
                diar_speaker=s.speaker,
            )
            for s in segs
        ]

        total_duration = sum(s.end - s.start for s in segs)

        cluster = VoiceCluster(
            voice_cluster_id=f"VC_{i+1:02d}",
            segments=cluster_segments,
            total_duration=total_duration,
            segment_count=len(segs),
        )
        clusters.append(cluster)

    return clusters


def _refine_clusters_with_embeddings(
    speaker_embeddings: Dict[str, List[Tuple[DiarizationSegment, np.ndarray]]],
    similarity_threshold: float,
    min_segments: int,
    centroid_method: str,
) -> List[VoiceCluster]:
    """Refine speaker clusters using embedding similarity."""
    # First, compute centroid for each diarization speaker
    speaker_centroids: Dict[str, np.ndarray] = {}

    for speaker, items in speaker_embeddings.items():
        embeddings = np.array([emb for _, emb in items])

        if centroid_method == "median":
            centroid = np.median(embeddings, axis=0)
        else:  # mean
            centroid = np.mean(embeddings, axis=0)

        # Normalize centroid
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        speaker_centroids[speaker] = centroid

    # Check for speakers that should be merged
    speakers = list(speaker_centroids.keys())
    merge_groups: Dict[str, str] = {}  # speaker -> merged speaker

    for i, s1 in enumerate(speakers):
        if s1 in merge_groups:
            continue

        for s2 in speakers[i+1:]:
            if s2 in merge_groups:
                continue

            # Compute similarity
            sim = np.dot(speaker_centroids[s1], speaker_centroids[s2])

            if sim >= similarity_threshold:
                # Merge s2 into s1
                LOGGER.debug(f"Merging speaker {s2} into {s1} (similarity: {sim:.3f})")
                merge_groups[s2] = s1

    # Build final clusters
    final_speakers: Dict[str, List[Tuple[DiarizationSegment, np.ndarray]]] = defaultdict(list)

    for speaker, items in speaker_embeddings.items():
        target_speaker = merge_groups.get(speaker, speaker)
        final_speakers[target_speaker].extend(items)

    # Create VoiceCluster objects
    clusters = []

    for speaker, items in final_speakers.items():
        # Filter by minimum segments
        if len(items) < min_segments:
            LOGGER.debug(f"Skipping speaker {speaker}: only {len(items)} segments")
            continue

        segments = [
            VoiceClusterSegment(
                start=seg.start,
                end=seg.end,
                diar_speaker=seg.speaker,
            )
            for seg, _ in items
        ]

        total_duration = sum(seg.end - seg.start for seg, _ in items)

        # Compute centroid
        embeddings = np.array([emb for _, emb in items])
        if centroid_method == "median":
            centroid = np.median(embeddings, axis=0)
        else:
            centroid = np.mean(embeddings, axis=0)

        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        cluster = VoiceCluster(
            voice_cluster_id="",  # Will be assigned later
            segments=segments,
            total_duration=total_duration,
            segment_count=len(items),
            centroid=centroid.tolist(),
        )
        clusters.append(cluster)

    return clusters


def _assign_cluster_ids(clusters: List[VoiceCluster]) -> List[VoiceCluster]:
    """Assign stable IDs to clusters based on total duration (descending)."""
    # Sort by total duration descending
    clusters = sorted(clusters, key=lambda c: c.total_duration, reverse=True)

    for i, cluster in enumerate(clusters):
        cluster.voice_cluster_id = f"VC_{i+1:02d}"

    return clusters


def _save_voice_clusters(clusters: List[VoiceCluster], output_path: Path):
    """Save voice clusters to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [cluster.model_dump() for cluster in clusters]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_voice_clusters(clusters_path: Path) -> List[VoiceCluster]:
    """Load voice clusters from JSON file."""
    with clusters_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return [VoiceCluster(**item) for item in data]


def get_cluster_for_timestamp(
    clusters: List[VoiceCluster],
    timestamp: float,
) -> Optional[VoiceCluster]:
    """Find which voice cluster contains a given timestamp.

    Args:
        clusters: List of voice clusters
        timestamp: Timestamp in seconds

    Returns:
        VoiceCluster containing the timestamp, or None
    """
    for cluster in clusters:
        for segment in cluster.segments:
            if segment.start <= timestamp < segment.end:
                return cluster

    return None


def compute_cluster_centroid(
    embeddings: List[np.ndarray],
    method: str = "mean",
) -> np.ndarray:
    """Compute centroid from a list of embeddings.

    Args:
        embeddings: List of embedding vectors
        method: "mean" or "median"

    Returns:
        Normalized centroid vector
    """
    if not embeddings:
        raise ValueError("Cannot compute centroid from empty list")

    arr = np.array(embeddings)

    if method == "median":
        centroid = np.median(arr, axis=0)
    else:
        centroid = np.mean(arr, axis=0)

    # Normalize
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    return centroid
