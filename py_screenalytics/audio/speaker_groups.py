"""Utilities for constructing and maintaining speaker group manifests.

Speaker groups are the primary abstraction for diarization results. They
aggregate diarization segments by source (pyannote, gpt4o, etc.) and provide
stable identifiers that downstream clustering, UI, and editing features can
reference.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .diarization_pyannote import extract_speaker_embeddings
from .models import (
    AudioSpeakerGroupsManifest,
    DiarizationSegment,
    SpeakerGroup,
    SpeakerGroupSource,
    SpeakerSegment,
    SpeakerSourceSummary,
)

LOGGER = logging.getLogger(__name__)


def _source_prefix(source: str) -> str:
    """Return a deterministic prefix for segment IDs for a diarization source."""
    if source.lower().startswith("py"):
        return "py"
    if source.lower().startswith("gpt4"):
        return "gpt4o"
    return source.lower().replace("-", "_")


def save_speaker_groups_manifest(manifest: AudioSpeakerGroupsManifest, output_path: Path) -> None:
    """Persist a speaker group manifest to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")


def load_speaker_groups_manifest(manifest_path: Path) -> AudioSpeakerGroupsManifest:
    """Load a speaker group manifest from disk."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return AudioSpeakerGroupsManifest(**data)


def build_speaker_groups_manifest(
    ep_id: str,
    source_segments: Dict[str, List[DiarizationSegment]],
    output_path: Path,
    overwrite: bool = False,
) -> AudioSpeakerGroupsManifest:
    """Build per-source speaker groups from diarization results.

    Args:
        ep_id: Episode identifier
        source_segments: Mapping of source name -> diarization segments
        output_path: Path to write manifest
        overwrite: Whether to overwrite if the manifest already exists
    """
    if output_path.exists() and not overwrite:
        LOGGER.info("Speaker groups already exist: %s", output_path)
        return load_speaker_groups_manifest(output_path)

    sources: List[SpeakerGroupSource] = []

    for source, segments in source_segments.items():
        if not segments:
            continue
        grouped = _build_groups_for_source(source, segments)
        sources.append(grouped)

    manifest = AudioSpeakerGroupsManifest(
        ep_id=ep_id,
        sources=sources,
    )
    save_speaker_groups_manifest(manifest, output_path)
    LOGGER.info(
        "Saved speaker groups for %s: %d sources -> %s",
        ep_id,
        len(sources),
        [s.source for s in sources],
    )
    return manifest


def _build_groups_for_source(
    source: str,
    segments: List[DiarizationSegment],
) -> SpeakerGroupSource:
    """Create groups for a diarization source."""
    prefix = _source_prefix(source)
    # Stable ordering by start time for deterministic IDs
    sorted_segments = sorted(segments, key=lambda s: (s.start, s.end))

    speaker_groups: Dict[str, List[SpeakerSegment]] = defaultdict(list)
    for idx, seg in enumerate(sorted_segments, start=1):
        segment_id = f"{prefix}_{idx:04d}"
        speaker_groups[seg.speaker].append(
            SpeakerSegment(
                segment_id=segment_id,
                start=seg.start,
                end=seg.end,
                diar_confidence=seg.confidence,
            )
        )

    speaker_group_objs: List[SpeakerGroup] = []
    total_segments = 0
    total_speech = 0.0

    for speaker_label, segs in speaker_groups.items():
        segs_sorted = sorted(segs, key=lambda s: (s.start, s.end))
        duration = sum(s.end - s.start for s in segs_sorted)
        total_segments += len(segs_sorted)
        total_speech += duration
        speaker_group_objs.append(
            SpeakerGroup(
                speaker_label=speaker_label,
                speaker_group_id=f"{source}:{speaker_label}",
                total_duration=duration,
                segment_count=len(segs_sorted),
                segments=segs_sorted,
            )
        )

    speaker_group_objs.sort(key=lambda g: g.speaker_group_id)
    summary = SpeakerSourceSummary(
        speakers=len(speaker_group_objs),
        segments=total_segments,
        speech_seconds=round(total_speech, 3),
    )

    return SpeakerGroupSource(
        source=source,
        summary=summary,
        speakers=speaker_group_objs,
    )


def speaker_group_lookup(manifest: AudioSpeakerGroupsManifest) -> Dict[str, SpeakerGroup]:
    """Convenience lookup for speaker_group_id -> SpeakerGroup."""
    lookup: Dict[str, SpeakerGroup] = {}
    for source in manifest.sources:
        for group in source.speakers:
            lookup[group.speaker_group_id] = group
    return lookup


def flatten_groups_to_segments(
    manifest: AudioSpeakerGroupsManifest,
    source_filter: Optional[str] = None,
) -> List[DiarizationSegment]:
    """Flatten groups back into DiarizationSegment entries.

    Speaker is set to the speaker_group_id to keep identifiers unique per source.
    """
    segments: List[DiarizationSegment] = []
    for source in manifest.sources:
        if source_filter and source.source != source_filter:
            continue
        for group in source.speakers:
            for seg in group.segments:
                segments.append(
                    DiarizationSegment(
                        start=seg.start,
                        end=seg.end,
                        speaker=group.speaker_group_id,
                        confidence=None,
                    )
                )
    return segments


def compute_group_centroids(
    audio_path: Path,
    manifest: AudioSpeakerGroupsManifest,
    embedding_model: str,
) -> Dict[str, np.ndarray]:
    """Compute centroid embeddings for each speaker group."""
    diar_segments = flatten_groups_to_segments(manifest)
    if not diar_segments:
        return {}

    embeddings = extract_speaker_embeddings(audio_path, diar_segments, embedding_model)
    grouped: Dict[str, List[np.ndarray]] = defaultdict(list)
    for seg, emb in embeddings:
        grouped[seg.speaker].append(emb)

    centroids: Dict[str, np.ndarray] = {}
    for group_id, group_embeddings in grouped.items():
        arr = np.array(group_embeddings)
        centroid = np.mean(arr, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[group_id] = centroid

    # Update manifest in-memory for convenience
    for source in manifest.sources:
        for group in source.speakers:
            group.centroid = centroids.get(group.speaker_group_id).tolist() if group.speaker_group_id in centroids else None

    return centroids


def update_summaries(manifest: AudioSpeakerGroupsManifest) -> AudioSpeakerGroupsManifest:
    """Recompute per-source summaries after edits."""
    for source in manifest.sources:
        total_segments = 0
        total_speech = 0.0
        for group in source.speakers:
            duration = sum(seg.end - seg.start for seg in group.segments)
            group.total_duration = duration
            group.segment_count = len(group.segments)
            total_segments += group.segment_count
            total_speech += duration
        source.summary = SpeakerSourceSummary(
            speakers=len(source.speakers),
            segments=total_segments,
            speech_seconds=round(total_speech, 3),
        )
    return manifest
