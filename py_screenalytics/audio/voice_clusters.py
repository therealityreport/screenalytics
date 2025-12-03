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
    AudioSpeakerGroupsManifest,
    DiarizationSegment,
    SpeakerGroup,
    VoiceCluster,
    VoiceClusterSegment,
    VoiceClusterSourceGroup,
    VoiceClusteringConfig,
)
from .speaker_groups import compute_group_centroids, speaker_group_lookup

LOGGER = logging.getLogger(__name__)


def cluster_episode_voices(
    audio_path: Path,
    diarization_segments: List[DiarizationSegment] | None,
    output_path: Path,
    config: Optional[VoiceClusteringConfig] = None,
    overwrite: bool = False,
    speaker_groups_manifest: Optional[AudioSpeakerGroupsManifest] = None,
) -> List[VoiceCluster]:
    """Cluster speaker groups into unique voices.

    Args:
        audio_path: Path to audio file
        diarization_segments: List of diarization segments (deprecated; use speaker_groups_manifest)
        output_path: Path for voice clusters JSON
        config: Voice clustering configuration
        overwrite: Whether to overwrite existing results
        speaker_groups_manifest: Optional precomputed speaker groups

    Returns:
        List of VoiceCluster objects
    """
    if output_path.exists() and not overwrite:
        LOGGER.info(f"Voice clusters already exist: {output_path}")
        return _load_voice_clusters(output_path)

    config = config or VoiceClusteringConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if speaker_groups_manifest:
        clusters = _cluster_from_speaker_groups(
            audio_path,
            speaker_groups_manifest,
            config,
        )
    elif diarization_segments:
        LOGGER.warning(
            "Deprecated diarization_segments clustering path used. "
            "Provide speaker_groups_manifest for group-first clustering."
        )
        clusters = _cluster_from_segments_legacy(
            audio_path,
            diarization_segments,
            config,
        )
    else:
        LOGGER.warning("No diarization data provided for voice clustering")
        clusters = []

    clusters = _assign_cluster_ids(clusters)
    _save_voice_clusters(clusters, output_path)
    LOGGER.info("Created %d voice clusters", len(clusters))

    return clusters


def _cluster_from_speaker_groups(
    audio_path: Path,
    manifest: AudioSpeakerGroupsManifest,
    config: VoiceClusteringConfig,
) -> List[VoiceCluster]:
    """Cluster voices from speaker groups across sources."""
    group_lookup = speaker_group_lookup(manifest)
    if not group_lookup:
        return []

    try:
        centroids = compute_group_centroids(audio_path, manifest, config.embedding_model)
    except Exception as exc:  # pragma: no cover - depends on heavy deps
        LOGGER.warning("Failed to compute speaker group centroids: %s", exc)
        centroids = {}

    group_data = []
    for source in manifest.sources:
        for group in source.speakers:
            centroid = centroids.get(group.speaker_group_id)
            group_data.append(
                {
                    "source": source.source,
                    "group": group,
                    "centroid": centroid,
                }
            )

    # Determine cross-source merges using conservative mutual best match
    matches = _mutual_best_matches(group_data, config.similarity_threshold)

    assigned: set[str] = set()
    clusters: List[VoiceCluster] = []

    for g1_id, g2_id in matches:
        if g1_id in assigned or g2_id in assigned:
            continue
        groups = [group_lookup[g1_id], group_lookup[g2_id]]
        clusters.append(_cluster_from_groups(groups, centroids))
        assigned.add(g1_id)
        assigned.add(g2_id)

    # Remaining groups become single-group clusters
    for item in group_data:
        gid = item["group"].speaker_group_id
        if gid in assigned:
            continue
        clusters.append(_cluster_from_groups([item["group"]], centroids))
        assigned.add(gid)

    # Filter out empty clusters
    return [c for c in clusters if c.segment_count >= config.min_segments_per_cluster]


def _cluster_from_groups(
    groups: List[SpeakerGroup],
    centroids: Dict[str, np.ndarray],
) -> VoiceCluster:
    """Build a VoiceCluster from one or more speaker groups."""
    total_duration = sum(g.total_duration for g in groups)
    segment_count = sum(g.segment_count for g in groups)
    segments: List[VoiceClusterSegment] = []
    sources: List[VoiceClusterSourceGroup] = []
    centroid_vectors: List[np.ndarray] = []

    for group in groups:
        for seg in group.segments:
            segments.append(
                VoiceClusterSegment(
                    start=seg.start,
                    end=seg.end,
                    diar_speaker=group.speaker_label,
                    speaker_group_id=group.speaker_group_id,
                )
            )
        centroid_vec = centroids.get(group.speaker_group_id)
        if centroid_vec is not None:
            centroid_vectors.append(centroid_vec)
        sources.append(
            VoiceClusterSourceGroup(
                source=group.speaker_group_id.split(":", 1)[0],
                speaker_group_id=group.speaker_group_id,
                speaker_label=group.speaker_label,
                centroid=centroid_vec.tolist() if centroid_vec is not None else None,
            )
        )

    centroid = None
    if centroid_vectors:
        centroid = compute_cluster_centroid(centroid_vectors, method="mean").tolist()

    return VoiceCluster(
        voice_cluster_id="",
        segments=segments,
        sources=sources,
        speaker_group_ids=[g.speaker_group_id for g in groups],
        total_duration=total_duration,
        segment_count=segment_count,
        centroid=centroid,
    )


def _mutual_best_matches(
    group_data: List[Dict],
    similarity_threshold: float,
) -> List[Tuple[str, str]]:
    """Find mutual best centroid matches across different sources."""
    if len(group_data) < 2:
        return []

    # Build map of centroid by group_id
    centroid_map: Dict[str, np.ndarray] = {}
    for item in group_data:
        centroid = item.get("centroid")
        if centroid is not None:
            centroid_map[item["group"].speaker_group_id] = centroid

    best: Dict[str, Tuple[str, float]] = {}
    for i, item in enumerate(group_data):
        gid = item["group"].speaker_group_id
        centroid = centroid_map.get(gid)
        if centroid is None:
            continue
        for other in group_data[i + 1:]:
            other_gid = other["group"].speaker_group_id
            if other["source"] == item["source"]:
                continue
            other_centroid = centroid_map.get(other_gid)
            if other_centroid is None:
                continue
            sim = float(np.dot(centroid, other_centroid))
            if sim < similarity_threshold:
                continue
            if sim > best.get(gid, ("", -1))[1]:
                best[gid] = (other_gid, sim)
            if sim > best.get(other_gid, ("", -1))[1]:
                best[other_gid] = (gid, sim)

    matches: List[Tuple[str, str]] = []
    for gid, (candidate, _) in best.items():
        reciprocal = best.get(candidate, (None, -1))[0]
        if reciprocal == gid:
            pair = tuple(sorted([gid, candidate]))
            if pair not in matches:
                matches.append(pair)
    return matches


def _cluster_from_segments_legacy(
    audio_path: Path,
    diarization_segments: List[DiarizationSegment],
    config: VoiceClusteringConfig,
) -> List[VoiceCluster]:
    """Legacy segment-level clustering retained for backwards compatibility."""
    LOGGER.info("Clustering %d segments into voice clusters (legacy path)", len(diarization_segments))

    # If use_diarization_labels is set, skip embedding clustering entirely
    if config.use_diarization_labels:
        LOGGER.info("Using pyannote speaker labels directly (use_diarization_labels=True)")
        return _clusters_from_diarization_labels(diarization_segments)

    from .diarization_pyannote import extract_speaker_embeddings

    segment_embeddings = extract_speaker_embeddings(
        audio_path,
        diarization_segments,
        config.embedding_model,
    )

    if not segment_embeddings:
        LOGGER.warning("No embeddings extracted, creating clusters from diarization labels")
        return _clusters_from_diarization_labels(diarization_segments)

    unique_diar_speakers = set(seg.speaker for seg in diarization_segments)
    if len(unique_diar_speakers) <= 2 and len(segment_embeddings) >= 3:
        LOGGER.info("Diarization found few speakers - using embedding-based clustering")
        return _cluster_by_embeddings(
            segment_embeddings,
            config.similarity_threshold,
            config.min_segments_per_cluster,
            config.centroid_method,
        )

    speaker_embeddings: Dict[str, List[Tuple[DiarizationSegment, np.ndarray]]] = defaultdict(list)
    for segment, embedding in segment_embeddings:
        speaker_embeddings[segment.speaker].append((segment, embedding))

    return _refine_clusters_with_embeddings(
        speaker_embeddings,
        config.similarity_threshold,
        config.min_segments_per_cluster,
        config.centroid_method,
    )


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
            speaker_group_ids=[speaker],
            sources=[
                VoiceClusterSourceGroup(
                    source=speaker.split(":", 1)[0] if ":" in speaker else "diarization",
                    speaker_group_id=speaker,
                    speaker_label=speaker,
                )
            ],
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
