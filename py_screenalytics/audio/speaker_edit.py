"""Speaker editing utilities (Smart Split).

Provides deterministic heuristics to split a diarization segment that contains
multiple voices into subsegments and reassign them to the most likely speaker
groups.

Includes:
- smart_split_segment: Auto-detect split points via embeddings
- smart_split_segment_by_words: User-driven split at specific word boundaries
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import shutil
import numpy as np

from .diarization_pyannote import _save_diarization_manifest, extract_speaker_embeddings
from .episode_audio_pipeline import _get_audio_paths, _get_show_id, _load_config
from .models import (
    ASRSegment,
    AudioSpeakerGroupsManifest,
    DiarizationSegment,
    SmartSplitResult,
    SmartSplitSegment,
    SpeakerGroup,
    SpeakerSegment,
    WordTiming,
)
from .speaker_groups import (
    compute_group_centroids,
    load_speaker_groups_manifest,
    save_speaker_groups_manifest,
    speaker_group_lookup,
    update_summaries,
)
from .voice_bank import match_voice_clusters_to_bank
from .voice_clusters import cluster_episode_voices
from .fuse_diarization_asr import fuse_transcript
from .asr_openai import _load_asr_manifest
from .diarization_pyannote import _load_diarization_manifest

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Word-level Smart Split Types
# =============================================================================


@dataclass
class WordSplitBoundary:
    """A split boundary between words."""
    word_index: int  # Index of word AFTER which to split (0-based)
    split_time: Optional[float] = None  # Computed split time (end of word at index)


@dataclass
class SegmentWordInfo:
    """Word-level info for a segment."""
    segment_id: str
    start: float
    end: float
    words: List[Dict[str, Any]] = field(default_factory=list)  # [{w, t0, t1}, ...]
    text: str = ""


def _load_asr_words_for_segment(
    asr_path: Path,
    seg_start: float,
    seg_end: float,
    tolerance: float = 0.3,
) -> List[Dict[str, Any]]:
    """Load word-level timings from ASR manifest for a time range.

    Args:
        asr_path: Path to episode_asr_raw.jsonl
        seg_start: Segment start time
        seg_end: Segment end time
        tolerance: Time tolerance for matching

    Returns:
        List of word dicts with {w, t0, t1} sorted by t0
    """
    if not asr_path.exists():
        return []

    words = []
    try:
        with open(asr_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                seg_data = json.loads(line.strip())
                seg_words = seg_data.get("words") or []
                for w in seg_words:
                    t0 = w.get("t0", 0.0)
                    t1 = w.get("t1", 0.0)
                    # Check if word overlaps with segment
                    if t0 < seg_end + tolerance and t1 > seg_start - tolerance:
                        words.append({
                            "w": w.get("w", ""),
                            "t0": t0,
                            "t1": t1,
                        })
    except Exception as e:
        LOGGER.warning(f"Failed to load ASR words: {e}")

    return sorted(words, key=lambda x: x.get("t0", 0.0))


def get_segment_words(
    ep_id: str,
    source: str,
    speaker_group_id: str,
    segment_id: str,
) -> SegmentWordInfo:
    """Get word-level information for a specific segment.

    Args:
        ep_id: Episode identifier
        source: Diarization source (pyannote, gpt4o)
        speaker_group_id: Speaker group ID
        segment_id: Segment ID within the group

    Returns:
        SegmentWordInfo with words and timing
    """
    paths = _get_audio_paths(ep_id)

    # Load speaker groups to find segment
    manifest_path = paths.get("speaker_groups")
    if not manifest_path or not manifest_path.exists():
        raise FileNotFoundError(f"Speaker groups manifest not found for {ep_id}")

    manifest = load_speaker_groups_manifest(manifest_path)
    groups = speaker_group_lookup(manifest)

    if speaker_group_id not in groups:
        raise ValueError(f"Speaker group {speaker_group_id} not found")

    group = groups[speaker_group_id]
    target_seg = None
    for seg in group.segments:
        if seg.segment_id == segment_id:
            target_seg = seg
            break

    if target_seg is None:
        raise ValueError(f"Segment {segment_id} not found in group {speaker_group_id}")

    # Load words from ASR
    asr_path = paths.get("asr_raw")
    if not asr_path:
        raise FileNotFoundError(f"ASR path not configured for {ep_id}")

    words = _load_asr_words_for_segment(asr_path, target_seg.start, target_seg.end)
    text = " ".join(w.get("w", "") for w in words)

    return SegmentWordInfo(
        segment_id=segment_id,
        start=target_seg.start,
        end=target_seg.end,
        words=words,
        text=text,
    )


def smart_split_segment_by_words(
    ep_id: str,
    source: str,
    speaker_group_id: str,
    segment_id: str,
    split_word_indices: List[int],
    rebuild_downstream: bool = True,
) -> SmartSplitResult:
    """Split a diarization segment at specific word boundaries.

    This is a user-driven split that creates subsegments at precise word
    boundaries, allowing correct assignment of mixed-speaker segments.

    Args:
        ep_id: Episode identifier
        source: Diarization source (pyannote, gpt4o)
        speaker_group_id: Speaker group ID containing the segment
        segment_id: Segment ID to split
        split_word_indices: List of word indices AFTER which to split (0-based)
            Example: [6] splits after word 6, creating segments [0-6] and [7-end]
        rebuild_downstream: Whether to rebuild clusters/transcript after split

    Returns:
        SmartSplitResult with new segment IDs and time ranges

    Example:
        For segment with words: ["I", "didn't", "make", "out", "with", "someone", "and", "neither", "f***ing", "did", "Todd"]
        split_word_indices=[5] creates:
        - Segment A: "I didn't make out with someone" (words 0-5)
        - Segment B: "and neither f***ing did Todd" (words 6-10)
    """
    paths = _get_audio_paths(ep_id)
    manifest_path = paths.get("speaker_groups")
    if not manifest_path or not manifest_path.exists():
        raise FileNotFoundError(f"Speaker groups manifest not found for {ep_id}")

    manifest = load_speaker_groups_manifest(manifest_path)
    groups = speaker_group_lookup(manifest)

    if speaker_group_id not in groups:
        raise ValueError(f"Speaker group {speaker_group_id} not found")

    target_group = groups[speaker_group_id]
    target_source_entry = next((s for s in manifest.sources if s.source == source), None)
    if target_source_entry is None:
        raise ValueError(f"Source {source} not present in speaker groups")

    # Find target segment
    target_segment: Optional[SpeakerSegment] = None
    for seg in target_group.segments:
        if seg.segment_id == segment_id:
            target_segment = seg
            break

    if target_segment is None:
        raise ValueError(f"Segment {segment_id} not found in group {speaker_group_id}")

    seg_start = target_segment.start
    seg_end = target_segment.end

    # Load words for this segment
    asr_path = paths.get("asr_raw")
    if not asr_path:
        raise FileNotFoundError(f"ASR path not configured for {ep_id}")

    words = _load_asr_words_for_segment(asr_path, seg_start, seg_end)

    if not words:
        raise ValueError(f"No words found for segment {segment_id} ({seg_start:.2f}-{seg_end:.2f}s)")

    # Validate split indices
    valid_indices = sorted(set(i for i in split_word_indices if 0 <= i < len(words) - 1))
    if not valid_indices:
        raise ValueError(
            f"Invalid split indices {split_word_indices}. "
            f"Must be in range [0, {len(words) - 2}] for {len(words)} words."
        )

    # Compute split times from word boundaries
    # Split AFTER word at index i means split time = words[i].t1
    split_times = []
    for idx in valid_indices:
        word = words[idx]
        split_time = word.get("t1", 0.0)
        # Ensure split is within segment bounds
        if seg_start < split_time < seg_end:
            split_times.append(split_time)

    if not split_times:
        raise ValueError("No valid split times computed from word indices")

    # Build boundary list: [seg_start, split1, split2, ..., seg_end]
    boundaries = [seg_start] + sorted(split_times) + [seg_end]

    LOGGER.info(
        f"[{ep_id}] Word-level smart split: {segment_id} -> {len(boundaries)-1} subsegments "
        f"at boundaries {[f'{b:.2f}s' for b in boundaries]}"
    )

    # Generate new segment IDs
    prefix = source[:2]  # "py" for pyannote, "gp" for gpt4o
    existing_ids = [
        s.segment_id
        for src in manifest.sources
        for g in src.speakers
        for s in g.segments
    ]
    next_idx = _next_segment_suffix(existing_ids, f"{prefix}_")

    # Create subsegments
    subsegments: List[SpeakerSegment] = []
    for i in range(len(boundaries) - 1):
        sub_start = boundaries[i]
        sub_end = boundaries[i + 1]
        if sub_end - sub_start < 0.1:  # Skip tiny segments
            continue

        subsegments.append(SpeakerSegment(
            segment_id=f"{prefix}_{next_idx:04d}",
            start=sub_start,
            end=sub_end,
            diar_confidence=target_segment.diar_confidence,  # Inherit from parent
        ))
        next_idx += 1

    if not subsegments:
        raise ValueError("No valid subsegments created (all too short)")

    # Update manifest: remove original, add subsegments
    target_group.segments = [
        seg for seg in target_group.segments
        if seg.segment_id != segment_id
    ]
    target_group.segments.extend(subsegments)
    target_group.segments.sort(key=lambda s: s.start)

    # Update summaries
    update_summaries(manifest)
    save_speaker_groups_manifest(manifest, manifest_path)

    LOGGER.info(
        f"[{ep_id}] Created {len(subsegments)} subsegments from {segment_id}: "
        f"{[s.segment_id for s in subsegments]}"
    )

    # Update diarization manifest
    diar_path = paths.get("diarization_pyannote") if source == "pyannote" else paths.get("diarization_gpt4o")
    if diar_path and diar_path.exists():
        diar_segments = _load_diarization_manifest(diar_path)
        # Remove original segment
        filtered = [
            d for d in diar_segments
            if not (abs(d.start - seg_start) < 1e-3 and abs(d.end - seg_end) < 1e-3)
        ]
        # Add new subsegments
        for subseg in subsegments:
            filtered.append(DiarizationSegment(
                start=subseg.start,
                end=subseg.end,
                speaker=target_group.speaker_label,
            ))
        filtered.sort(key=lambda d: d.start)
        _save_diarization_manifest(filtered, diar_path)

        # Copy to primary diarization if pyannote
        if source == "pyannote" and paths.get("diarization"):
            try:
                shutil.copy(diar_path, paths["diarization"])
            except Exception as exc:
                LOGGER.warning(f"Failed to refresh primary diarization: {exc}")

    # Optionally rebuild downstream artifacts
    if rebuild_downstream:
        _rebuild_downstream_artifacts(ep_id, paths, source)

    # Build result
    return SmartSplitResult(
        ep_id=ep_id,
        source=source,
        original_segment_id=segment_id,
        new_segments=[
            SmartSplitSegment(
                segment_id=s.segment_id,
                start=s.start,
                end=s.end,
                speaker_group_id=speaker_group_id,
            )
            for s in subsegments
        ],
    )


def _rebuild_downstream_artifacts(ep_id: str, paths: dict, source: str) -> None:
    """Rebuild voice clusters, mapping, and transcript after a split."""
    config = _load_config()
    manifest_path = paths.get("speaker_groups")
    audio_path = _choose_audio_path(paths)

    if not audio_path or not manifest_path:
        LOGGER.warning(f"[{ep_id}] Cannot rebuild downstream: missing audio or manifest")
        return

    try:
        speaker_groups_manifest = load_speaker_groups_manifest(manifest_path)
        voice_clusters = cluster_episode_voices(
            audio_path,
            diarization_segments=None,
            output_path=paths["voice_clusters"],
            config=config.voice_clustering,
            overwrite=True,
            speaker_groups_manifest=speaker_groups_manifest,
        )
        voice_mapping = match_voice_clusters_to_bank(
            show_id=_get_show_id(ep_id),
            clusters=voice_clusters,
            output_path=paths["voice_mapping"],
            config=config.voice_bank,
            similarity_threshold=config.voice_clustering.similarity_threshold,
            overwrite=True,
        )

        diar_path = paths.get("diarization")
        if diar_path and diar_path.exists():
            diarization_segments = _load_diarization_manifest(diar_path)
            asr_segments = _load_asr_manifest(paths["asr_raw"])
            fuse_transcript(
                diarization_segments,
                asr_segments,
                voice_clusters,
                voice_mapping,
                speaker_groups_manifest,
                paths["transcript_jsonl"],
                paths["transcript_vtt"],
                include_speaker_notes=config.export.vtt_include_speaker_notes,
                overwrite=True,
                diarization_source=source,
            )
        LOGGER.info(f"[{ep_id}] Rebuilt downstream artifacts after word split")
    except Exception as e:
        LOGGER.warning(f"[{ep_id}] Failed to rebuild downstream artifacts: {e}")


def _choose_audio_path(paths: dict) -> Optional[Path]:
    """Pick the best available audio path for embeddings."""
    for key in ["vocals_enhanced", "vocals", "final_voice_only", "original"]:
        candidate = paths.get(key)
        if candidate and candidate.exists():
            return candidate
    return None


def _next_segment_suffix(existing_ids: List[str], prefix: str) -> int:
    max_idx = 0
    for seg_id in existing_ids:
        if not seg_id.startswith(prefix):
            continue
        try:
            suffix = seg_id.replace(prefix, "").strip("_")
            max_idx = max(max_idx, int(suffix))
        except ValueError:
            continue
    return max_idx + 1


def _dedupe_boundaries(boundaries: List[float], start: float, end: float) -> List[float]:
    uniq = []
    for b in sorted(boundaries):
        if b <= start or b >= end:
            continue
        if not uniq or abs(uniq[-1] - b) > 0.1:
            uniq.append(b)
    return [start] + uniq + [end]


def _boundary_candidates_from_other_source(
    manifest: AudioSpeakerGroupsManifest,
    current_source: str,
    start: float,
    end: float,
) -> List[float]:
    """Use the other diarization source to propose split points."""
    boundaries: List[float] = []
    for source in manifest.sources:
        if source.source == current_source:
            continue
        for group in source.speakers:
            for seg in group.segments:
                if seg.start >= end or seg.end <= start:
                    continue
                if start < seg.start < end:
                    boundaries.append(seg.start)
                if start < seg.end < end:
                    boundaries.append(seg.end)
    return boundaries


def _boundary_candidates_from_embeddings(
    audio_path: Path,
    start: float,
    end: float,
    embedding_model: str,
    expected_voices: int,
) -> List[float]:
    """Detect potential change points via sliding-window embedding similarity."""
    duration = end - start
    if duration <= 0.6:
        return []

    window = min(1.0, duration / max(2 * expected_voices, 2))
    step = window * 0.6

    windows: List[DiarizationSegment] = []
    t = start
    while t < end:
        w_end = min(end, t + window)
        windows.append(DiarizationSegment(start=t, end=w_end, speaker="probe"))
        t += step

    try:
        embeddings = extract_speaker_embeddings(audio_path, windows, embedding_model)
    except Exception as exc:  # pragma: no cover - depends on heavy deps
        LOGGER.warning("Sliding embedding extraction failed: %s", exc)
        return []

    if len(embeddings) < 2:
        return []

    # Normalize embeddings
    normed = []
    for _, emb in embeddings:
        norm = np.linalg.norm(emb)
        normed.append(emb / norm if norm > 0 else emb)

    sims: List[float] = []
    for i in range(len(normed) - 1):
        sims.append(float(np.dot(normed[i], normed[i + 1])))

    # Pick the largest drops (lowest sims)
    drops = sorted(
        [(sim, idx) for idx, sim in enumerate(sims)],
        key=lambda x: x[0],
    )
    split_candidates = []
    for _, idx in drops[: max(1, expected_voices - 1)]:
        # Boundary between window idx and idx+1
        split_time = windows[idx].end
        if start < split_time < end:
            split_candidates.append(split_time)

    return split_candidates


def _assign_subsegments(
    embeddings: Dict[str, np.ndarray],
    centroids: Dict[str, np.ndarray],
    original_group_id: str,
    similarity_threshold: float,
) -> Dict[str, Optional[str]]:
    """Assign embeddings to existing groups or mark as None when uncertain."""
    assignments: Dict[str, Optional[str]] = {}
    for seg_id, emb in embeddings.items():
        best_group = original_group_id
        best_sim = -1.0
        for group_id, centroid in centroids.items():
            sim = float(np.dot(emb, centroid))
            if sim > best_sim:
                best_sim = sim
                best_group = group_id
        if best_sim < similarity_threshold:
            assignments[seg_id] = original_group_id if best_group == original_group_id else None
        else:
            assignments[seg_id] = best_group
    return assignments


def _generate_subsegment_embeddings(
    audio_path: Path,
    subsegments: List[SpeakerSegment],
    embedding_model: str,
) -> Dict[str, np.ndarray]:
    diar_segments = [
        DiarizationSegment(start=s.start, end=s.end, speaker=s.segment_id) for s in subsegments
    ]
    try:
        embeddings = extract_speaker_embeddings(audio_path, diar_segments, embedding_model)
    except Exception as exc:  # pragma: no cover - heavy deps
        LOGGER.warning("Failed to extract embeddings for smart split: %s", exc)
        return {}
    results: Dict[str, np.ndarray] = {}
    for seg, emb in embeddings:
        norm = np.linalg.norm(emb)
        results[seg.speaker] = emb / norm if norm > 0 else emb
    return results


def _ensure_original_keeps_majority(
    assignments: Dict[str, Optional[str]],
    segments: List[SpeakerSegment],
    original_group_id: str,
) -> Dict[str, Optional[str]]:
    """If nothing assigned to original, force the longest chunk to stay."""
    if any(gid == original_group_id for gid in assignments.values() if gid):
        return assignments
    longest = max(segments, key=lambda s: s.end - s.start)
    assignments[longest.segment_id] = original_group_id
    return assignments


def smart_split_segment(
    ep_id: str,
    source: str,
    speaker_group_id: str,
    segment_id: Optional[str] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    expected_voices: int = 2,
) -> SmartSplitResult:
    """Smart-split a diarization segment into multiple speaker groups."""
    paths = _get_audio_paths(ep_id)
    manifest_path = paths.get("speaker_groups")
    if not manifest_path or not manifest_path.exists():
        raise FileNotFoundError(f"Speaker groups manifest not found for {ep_id}")

    manifest = load_speaker_groups_manifest(manifest_path)
    groups = speaker_group_lookup(manifest)
    if speaker_group_id not in groups:
        raise ValueError(f"Speaker group {speaker_group_id} not found")
    target_group = groups[speaker_group_id]
    target_source_entry = next((s for s in manifest.sources if s.source == source), None)
    if target_source_entry is None:
        raise ValueError(f"Source {source} not present in speaker groups")

    # Identify segment to split
    target_segment: Optional[SpeakerSegment] = None
    for seg in target_group.segments:
        if segment_id and seg.segment_id == segment_id:
            target_segment = seg
            break
        if start is not None and end is not None and abs(seg.start - start) < 1e-3 and abs(seg.end - end) < 1e-3:
            target_segment = seg
            break
    if target_segment is None:
        raise ValueError("Target segment not found in speaker group")

    seg_start = start if start is not None else target_segment.start
    seg_end = end if end is not None else target_segment.end

    audio_path = _choose_audio_path(paths)
    if not audio_path:
        raise FileNotFoundError("No audio available for smart split")

    config = _load_config()
    try:
        centroids = compute_group_centroids(audio_path, manifest, config.voice_clustering.embedding_model)
    except Exception as exc:  # pragma: no cover - heavy deps
        LOGGER.warning("Falling back to diarization labels for split (centroids failed): %s", exc)
        centroids = {}
    original_centroid = centroids.get(speaker_group_id)

    # Build boundaries
    boundaries: List[float] = []
    boundaries.extend(_boundary_candidates_from_other_source(manifest, source, seg_start, seg_end))
    boundaries.extend(
        _boundary_candidates_from_embeddings(
            audio_path,
            seg_start,
            seg_end,
            config.voice_clustering.embedding_model,
            expected_voices,
        )
    )
    boundary_list = _dedupe_boundaries(boundaries, seg_start, seg_end)

    # Build subsegments
    subsegments: List[SpeakerSegment] = []
    prefix = speaker_group_id.split(":", 1)[0]
    existing_ids = [s.segment_id for src in manifest.sources for g in src.speakers for s in g.segments]
    next_idx = _next_segment_suffix(existing_ids, f"{prefix}_")
    for i in range(len(boundary_list) - 1):
        s = boundary_list[i]
        e = boundary_list[i + 1]
        if e - s < 0.15:
            continue
        subsegments.append(
            SpeakerSegment(
                segment_id=f"{prefix}_{next_idx:04d}",
                start=s,
                end=e,
            )
        )
        next_idx += 1

    # Compute embeddings for subsegments
    sub_embeddings = _generate_subsegment_embeddings(
        audio_path,
        subsegments,
        config.voice_clustering.embedding_model,
    )

    # Build candidate centroids (same source preferred)
    candidate_centroids: Dict[str, np.ndarray] = {}
    for gid, centroid in centroids.items():
        if gid.split(":", 1)[0] != source:
            continue
        candidate_centroids[gid] = centroid
    if original_centroid is not None:
        candidate_centroids[speaker_group_id] = original_centroid

    assignments = _assign_subsegments(
        sub_embeddings,
        candidate_centroids,
        speaker_group_id,
        config.voice_clustering.similarity_threshold,
    )
    assignments = _ensure_original_keeps_majority(assignments, subsegments, speaker_group_id)

    # Create new groups if needed
    updated_groups: Dict[str, SpeakerGroup] = {}
    for src in manifest.sources:
        for g in src.speakers:
            updated_groups[g.speaker_group_id] = g

    new_group_counter = 1
    for subseg in subsegments:
        assigned_gid = assignments.get(subseg.segment_id, speaker_group_id)
        if assigned_gid not in updated_groups:
            # Create new group under the same source
            suffix = new_group_counter
            new_gid = f"{source}:{target_group.speaker_label}_SPLIT_{suffix:02d}"
            new_group_counter += 1
            new_group = SpeakerGroup(
                speaker_label=f"{target_group.speaker_label}_SPLIT_{suffix:02d}",
                speaker_group_id=new_gid,
                segments=[],
                total_duration=0.0,
                segment_count=0,
            )
            updated_groups[new_gid] = new_group
            target_source_entry.speakers.append(new_group)
            assignments[subseg.segment_id] = new_gid
        else:
            assignments[subseg.segment_id] = assigned_gid

    # Apply changes to manifest
    for seg in list(target_group.segments):
        if seg.segment_id == target_segment.segment_id:
            target_group.segments.remove(seg)

    for subseg in subsegments:
        gid = assignments.get(subseg.segment_id, speaker_group_id) or speaker_group_id
        updated_groups[gid].segments.append(subseg)

    update_summaries(manifest)
    save_speaker_groups_manifest(manifest, manifest_path)

    # Update diarization manifest for the source to reflect new splits
    diar_path = paths.get("diarization_pyannote") if source == "pyannote" else paths.get("diarization_gpt4o")
    if diar_path and diar_path.exists():
        diar_segments = _load_diarization_manifest(diar_path)
        filtered = [d for d in diar_segments if not (abs(d.start - seg_start) < 1e-3 and abs(d.end - seg_end) < 1e-3 and d.speaker == target_group.speaker_label)]
        for subseg in subsegments:
            filtered.append(
                DiarizationSegment(
                    start=subseg.start,
                    end=subseg.end,
                    speaker=assignments.get(subseg.segment_id, speaker_group_id).split(":", 1)[-1],
                )
            )
        filtered.sort(key=lambda d: d.start)
        _save_diarization_manifest(filtered, diar_path)
        if source == "pyannote" and paths.get("diarization"):
            try:
                shutil.copy(diar_path, paths["diarization"])
            except Exception as exc:
                LOGGER.warning("Failed to refresh primary diarization manifest: %s", exc)

    # Rebuild clusters, mapping, and transcript to keep artifacts consistent
    speaker_groups_manifest = load_speaker_groups_manifest(manifest_path)
    voice_clusters = cluster_episode_voices(
        audio_path,
        diarization_segments=None,
        output_path=paths["voice_clusters"],
        config=config.voice_clustering,
        overwrite=True,
        speaker_groups_manifest=speaker_groups_manifest,
    )
    voice_mapping = match_voice_clusters_to_bank(
        show_id=_get_show_id(ep_id),
        clusters=voice_clusters,
        output_path=paths["voice_mapping"],
        config=config.voice_bank,
        similarity_threshold=config.voice_clustering.similarity_threshold,
        overwrite=True,
    )

    diarization_segments = _load_diarization_manifest(paths["diarization"])
    asr_segments = _load_asr_manifest(paths["asr_raw"])
    fuse_transcript(
        diarization_segments,
        asr_segments,
        voice_clusters,
        voice_mapping,
        speaker_groups_manifest,
        paths["transcript_jsonl"],
        paths["transcript_vtt"],
        include_speaker_notes=config.export.vtt_include_speaker_notes,
        overwrite=True,
        diarization_source=source,
    )

    new_segments_payload = [
        SmartSplitSegment(
            segment_id=subseg.segment_id,
            start=subseg.start,
            end=subseg.end,
            speaker_group_id=assignments.get(subseg.segment_id, speaker_group_id) or speaker_group_id,
        )
        for subseg in subsegments
    ]

    return SmartSplitResult(
        ep_id=ep_id,
        source=source,
        original_segment_id=target_segment.segment_id,
        new_segments=new_segments_payload,
    )
