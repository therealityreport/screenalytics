"""Fuse diarization, ASR, and voice mapping into final transcript.

Handles:
- Aligning ASR segments with diarization
- Assigning speakers to transcript rows
- Generating JSONL and VTT output formats
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .models import (
    ASRSegment,
    AudioSpeakerGroupsManifest,
    DiarizationSegment,
    TranscriptRow,
    VoiceBankMatchResult,
    VoiceCluster,
    WordTiming,
)

LOGGER = logging.getLogger(__name__)


def fuse_transcript(
    diarization_segments: List[DiarizationSegment],
    asr_segments: List[ASRSegment],
    voice_clusters: List[VoiceCluster],
    voice_mapping: List[VoiceBankMatchResult],
    speaker_groups_manifest: Optional[AudioSpeakerGroupsManifest],
    output_jsonl: Path,
    output_vtt: Path,
    include_speaker_notes: bool = True,
    overwrite: bool = False,
    diarization_source: str = "nemo",
) -> List[TranscriptRow]:
    """Fuse diarization, ASR, and voice mapping into final transcript.

    Args:
        diarization_segments: Diarization results
        asr_segments: ASR results
        voice_clusters: Voice cluster data
        voice_mapping: Voice bank mapping results
        speaker_groups_manifest: Speaker group manifest (preferred mapping surface)
        output_jsonl: Path for JSONL transcript
        output_vtt: Path for VTT transcript
        include_speaker_notes: Whether to include speaker notes in VTT
        overwrite: Whether to overwrite existing files
        diarization_source: Source name to prefix diarization speakers when building group IDs

    Returns:
        List of TranscriptRow objects
    """
    if output_jsonl.exists() and output_vtt.exists() and not overwrite:
        LOGGER.info(f"Transcript already exists: {output_jsonl}")
        return _load_transcript_jsonl(output_jsonl)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_vtt.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        f"Fusing transcript: {len(diarization_segments)} diarization segments, "
        f"{len(asr_segments)} ASR segments, {len(voice_clusters)} voice clusters"
    )

    # Build lookup structures
    cluster_mapping = _build_cluster_mapping(voice_mapping)
    group_to_cluster = _build_group_cluster_lookup(voice_clusters)

    # Build word-level assignments to prevent cross-speaker rows
    words: List[dict] = []
    for asr_segment in asr_segments:
        if asr_segment.words:
            for w in asr_segment.words:
                diar_match = _find_best_diarization_match_word(w, diarization_segments)
                speaker_group_id = _speaker_group_from_diar(diar_match, diarization_source) if diar_match else None
                cluster_id = group_to_cluster.get(speaker_group_id or "", None)
                mapping = cluster_mapping.get(cluster_id or "", None)
                speaker_label = diar_match.speaker if diar_match else "UNKNOWN"
                if speaker_groups_manifest:
                    group_lookup = {
                        g.speaker_group_id: g
                        for src in speaker_groups_manifest.sources
                        for g in src.speakers
                    }
                    group = group_lookup.get(speaker_group_id or "")
                    if group:
                        speaker_label = group.speaker_label
                speaker_id, speaker_display_name, voice_bank_id, cluster_id = _resolve_speaker_fields(
                    speaker_group_id,
                    speaker_label,
                    cluster_id,
                    mapping,
                )
                # Extract NeMo overlap info if available
                is_overlap, secondary_speakers = _extract_overlap_info(diar_match)
                words.append({
                    "w": w.w,
                    "t0": w.t0,
                    "t1": w.t1,
                    "speaker_id": speaker_id,
                    "speaker_display_name": speaker_display_name,
                    "voice_cluster_id": cluster_id,
                    "voice_bank_id": voice_bank_id,
                    "conf": w.t1 - w.t0,  # placeholder; whisper does not include per-word conf
                    "overlap": is_overlap,
                    "secondary_speakers": secondary_speakers,
                })
        else:
            diar_match = _find_best_diarization_match(asr_segment, diarization_segments)
            speaker_group_id = _speaker_group_from_diar(diar_match, diarization_source) if diar_match else None
            cluster_id = group_to_cluster.get(speaker_group_id or "", None)
            mapping = cluster_mapping.get(cluster_id or "", None)
            speaker_label = diar_match.speaker if diar_match else "UNKNOWN"
            if speaker_groups_manifest:
                group_lookup = {
                    g.speaker_group_id: g
                    for src in speaker_groups_manifest.sources
                    for g in src.speakers
                }
                group = group_lookup.get(speaker_group_id or "")
                if group:
                    speaker_label = group.speaker_label
            speaker_id, speaker_display_name, voice_bank_id, cluster_id = _resolve_speaker_fields(
                speaker_group_id,
                speaker_label,
                cluster_id,
                mapping,
            )
            # Extract NeMo overlap info if available
            is_overlap, secondary_speakers = _extract_overlap_info(diar_match)
            words.append({
                "w": asr_segment.text,
                "t0": asr_segment.start,
                "t1": asr_segment.end,
                "speaker_id": speaker_id,
                "speaker_display_name": speaker_display_name,
                "voice_cluster_id": cluster_id,
                "voice_bank_id": voice_bank_id,
                "conf": asr_segment.confidence,
                "overlap": is_overlap,
                "secondary_speakers": secondary_speakers,
            })

    words = sorted(words, key=lambda w: w.get("t0", 0.0))

    transcript_rows = _rows_from_words(words)

    # Save outputs
    _save_transcript_jsonl(transcript_rows, output_jsonl)
    _save_transcript_vtt(transcript_rows, output_vtt, include_speaker_notes)

    LOGGER.info(f"Generated transcript: {len(transcript_rows)} rows")

    return transcript_rows


def _build_cluster_mapping(
    voice_mapping: List[VoiceBankMatchResult],
) -> Dict[str, VoiceBankMatchResult]:
    """Build lookup from voice_cluster_id to mapping result."""
    return {m.voice_cluster_id: m for m in voice_mapping}


def _build_group_cluster_lookup(
    voice_clusters: List[VoiceCluster],
) -> Dict[str, str]:
    """Map speaker_group_id -> voice_cluster_id.

    Handles multiple speaker ID formats for cross-source lookups:
    - Raw speaker label: "SPEAKER_00", "spk_00"
    - NeMo prefixed: "nemo:spk_00"
    - Legacy prefixes for backward compatibility: "pyannote:SPEAKER_00", "gpt4o:Speaker 1"
    """
    mapping: Dict[str, str] = {}
    for cluster in voice_clusters:
        for seg in cluster.segments:
            if seg.diar_speaker:
                # Map raw speaker label
                mapping[seg.diar_speaker] = cluster.voice_cluster_id
                # Map with nemo prefix (current format)
                mapping[f"nemo:{seg.diar_speaker}"] = cluster.voice_cluster_id
                # Legacy prefixes for backward compatibility with old manifests
                mapping[f"pyannote:{seg.diar_speaker}"] = cluster.voice_cluster_id
                mapping[f"gpt4o:{seg.diar_speaker}"] = cluster.voice_cluster_id
        for gid in cluster.speaker_group_ids:
            mapping[gid] = cluster.voice_cluster_id
        for source_group in cluster.sources:
            mapping[source_group.speaker_group_id] = cluster.voice_cluster_id
    return mapping


def _resolve_speaker_fields(
    speaker_group_id: Optional[str],
    diar_label: str,
    cluster_id: Optional[str],
    mapping: Optional[VoiceBankMatchResult],
) -> Tuple[str, str, str, str]:
    """Resolve speaker_id/display/voice ids with fallbacks."""
    if mapping:
        return (
            mapping.speaker_id,
            mapping.speaker_display_name,
            mapping.voice_bank_id,
            mapping.voice_cluster_id,
        )

    label = diar_label or "UNKNOWN"
    speaker_id = f"SPK_{label}"
    speaker_display_name = f"Speaker {label}"
    voice_bank_id = "voice_unknown"
    resolved_cluster = cluster_id or "VC_00"
    return speaker_id, speaker_display_name, voice_bank_id, resolved_cluster


def _find_best_diarization_match(
    asr_segment: ASRSegment,
    diarization_segments: List[DiarizationSegment],
) -> Optional[DiarizationSegment]:
    """Find the diarization segment with most overlap with ASR segment."""
    best_match = None
    best_overlap = 0.0

    for diar_seg in diarization_segments:
        # Calculate overlap
        overlap_start = max(asr_segment.start, diar_seg.start)
        overlap_end = min(asr_segment.end, diar_seg.end)
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_match = diar_seg

    return best_match


def _extract_overlap_info(diar_segment: Optional[Any]) -> Tuple[bool, List[str]]:
    """Extract overlap information from a diarization segment.

    Works with both NeMoDiarizationSegment (has overlap fields) and
    standard DiarizationSegment (no overlap fields).

    Args:
        diar_segment: A diarization segment (NeMo or standard)

    Returns:
        Tuple of (is_overlap, secondary_speakers)
    """
    if diar_segment is None:
        return False, []

    # Check for NeMo overlap fields
    is_overlap = getattr(diar_segment, "overlap", False)
    active_speakers = getattr(diar_segment, "active_speakers", [])
    primary_speaker = diar_segment.speaker

    # Secondary speakers are active speakers minus the primary
    secondary_speakers = [s for s in active_speakers if s != primary_speaker]

    return is_overlap, secondary_speakers


def _find_nearest_speaker(
    segment_start: float,
    segment_end: float,
    diarization_segments: List[DiarizationSegment],
) -> Optional[str]:
    """Find the nearest speaker when there's no direct overlap.

    Used as fallback in WhisperX-style merge when no diarization segment
    overlaps with the ASR segment.

    Args:
        segment_start: Start time of the ASR segment
        segment_end: End time of the ASR segment
        diarization_segments: List of diarization segments

    Returns:
        Speaker label from nearest segment, or None if no segments
    """
    if not diarization_segments:
        return None

    segment_mid = (segment_start + segment_end) / 2
    nearest_speaker = None
    min_distance = float("inf")

    for diar_seg in diarization_segments:
        # Distance to segment midpoint
        diar_mid = (diar_seg.start + diar_seg.end) / 2
        distance = abs(segment_mid - diar_mid)

        if distance < min_distance:
            min_distance = distance
            nearest_speaker = diar_seg.speaker

    return nearest_speaker


def assign_speakers_whisperx_style(
    asr_segments: List[ASRSegment],
    diarization_segments: List[DiarizationSegment],
    fill_nearest: bool = True,
) -> List[ASRSegment]:
    """Assign speakers using WhisperX timestamp intersection algorithm.

    This implements the official WhisperX-based algorithm from PyannoteAI docs:
    1. For each ASR segment/word, find all overlapping diarization segments
    2. Compute intersection duration with each
    3. Assign speaker with maximum overlap
    4. If no overlap and fill_nearest=True, use nearest segment by time

    This is more accurate than simple best-overlap matching because it
    considers total intersection duration across potentially multiple
    overlapping diarization segments.

    Args:
        asr_segments: List of ASR segments with text and timestamps
        diarization_segments: List of diarization segments with speaker labels
        fill_nearest: If True, use nearest speaker when no overlap (default True)

    Returns:
        List of ASRSegment objects with speaker field populated
    """
    LOGGER.info("Merging diarization with ASR transcripts using WhisperX-style intersection...")

    if not diarization_segments:
        LOGGER.warning("No diarization segments provided, cannot assign speakers")
        return asr_segments

    # Sort diarization segments by start time for efficient lookup
    sorted_diar = sorted(diarization_segments, key=lambda s: s.start)

    assigned_segments = []
    unknown_count = 0

    for asr_seg in asr_segments:
        # Find all overlapping diarization segments
        overlaps = []
        for diar_seg in sorted_diar:
            # Check for overlap: segments overlap if start < other_end and end > other_start
            if diar_seg.start < asr_seg.end and diar_seg.end > asr_seg.start:
                # Calculate intersection duration
                intersection_start = max(asr_seg.start, diar_seg.start)
                intersection_end = min(asr_seg.end, diar_seg.end)
                duration = intersection_end - intersection_start

                if duration > 0:
                    overlaps.append({
                        "speaker": diar_seg.speaker,
                        "duration": duration,
                    })

        # Assign speaker based on overlaps
        if overlaps:
            # Sum durations per speaker (handles multiple segments from same speaker)
            speaker_durations: Dict[str, float] = {}
            for overlap in overlaps:
                speaker = overlap["speaker"]
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + overlap["duration"]

            # Assign speaker with maximum total overlap
            assigned_speaker = max(speaker_durations.items(), key=lambda x: x[1])[0]
        elif fill_nearest:
            # No overlap - use nearest speaker
            assigned_speaker = _find_nearest_speaker(
                asr_seg.start, asr_seg.end, sorted_diar
            )
            if not assigned_speaker:
                assigned_speaker = "UNKNOWN"
                unknown_count += 1
        else:
            assigned_speaker = "UNKNOWN"
            unknown_count += 1

        # Create new segment with speaker assigned
        assigned_segments.append(ASRSegment(
            segment_id=asr_seg.segment_id,
            start=asr_seg.start,
            end=asr_seg.end,
            text=asr_seg.text,
            confidence=asr_seg.confidence,
            words=asr_seg.words,
            language=asr_seg.language,
            speaker=assigned_speaker,
        ))

    if unknown_count > 0:
        LOGGER.warning(f"Could not assign speaker to {unknown_count} segments")

    LOGGER.info(f"WhisperX-style speaker assignment complete: {len(assigned_segments)} segments")
    return assigned_segments


def _find_best_diarization_match_word(
    word: WordTiming,
    diarization_segments: List[DiarizationSegment],
) -> Optional[DiarizationSegment]:
    """Find the diarization segment with most overlap with a word."""
    asr_like = ASRSegment(start=word.t0, end=word.t1, text=word.w)
    return _find_best_diarization_match(asr_like, diarization_segments)


def _speaker_group_from_diar(
    diar_segment: DiarizationSegment,
    diarization_source: str,
) -> str:
    """Create a speaker_group_id for a diarization segment."""
    if ":" in diar_segment.speaker:
        return diar_segment.speaker
    return f"{diarization_source}:{diar_segment.speaker}"


def _find_cluster_for_segment(
    diar_segment: DiarizationSegment,
    voice_clusters: List[VoiceCluster],
) -> Optional[str]:
    """Find which voice cluster contains a diarization segment."""
    for cluster in voice_clusters:
        for seg in cluster.segments:
            # Check if diarization segment overlaps with cluster segment
            if seg.diar_speaker == diar_segment.speaker:
                # Check time overlap using proper interval overlap logic:
                # Two intervals [a, b) and [c, d) overlap iff a < d and c < b
                if diar_segment.start < seg.end and seg.start < diar_segment.end:
                    return cluster.voice_cluster_id

    return None


def _merge_consecutive_speaker_rows(
    rows: List[TranscriptRow],
    max_gap: float = 1.0,
) -> List[TranscriptRow]:
    """Merge consecutive rows from the same speaker.

    Args:
        rows: List of transcript rows
        max_gap: Maximum gap in seconds to merge

    Returns:
        Merged list of rows
    """
    if not rows:
        return rows

    merged = []
    current = rows[0]

    for row in rows[1:]:
        # Check if same speaker and small gap
        gap = row.start - current.end
        if row.speaker_id == current.speaker_id and gap <= max_gap:
            # Merge rows
            merged_words = None
            if current.words or row.words:
                merged_words = (current.words or []) + (row.words or [])

            # Average confidence
            if current.conf and row.conf:
                merged_conf = (current.conf + row.conf) / 2
            else:
                merged_conf = current.conf or row.conf

            # Merge overlap info: row has overlap if either row does
            merged_overlap = current.overlap or row.overlap
            # Combine secondary speakers, preserving uniqueness
            merged_secondary = list(current.secondary_speakers)
            for sp in row.secondary_speakers:
                if sp not in merged_secondary:
                    merged_secondary.append(sp)

            current = TranscriptRow(
                start=current.start,
                end=row.end,
                speaker_id=current.speaker_id,
                speaker_display_name=current.speaker_display_name,
                voice_cluster_id=current.voice_cluster_id,
                voice_bank_id=current.voice_bank_id,
                text=f"{current.text} {row.text}".strip(),
                conf=merged_conf,
                words=merged_words,
                overlap=merged_overlap,
                secondary_speakers=merged_secondary,
            )
        else:
            merged.append(current)
            current = row

    merged.append(current)
    return merged


def _rows_from_words(words: List[dict], max_gap: float = 0.8) -> List[TranscriptRow]:
    """Build transcript rows from word-level speaker assignments."""
    if not words:
        return []

    rows: List[TranscriptRow] = []
    current_words: List[dict] = []

    def _flush():
        if not current_words:
            return
        speaker_id = current_words[0]["speaker_id"]
        speaker_display_name = current_words[0]["speaker_display_name"]
        cluster_id = current_words[0]["voice_cluster_id"]
        voice_bank_id = current_words[0]["voice_bank_id"]
        text = " ".join(w["w"] for w in current_words if w.get("w"))
        start = current_words[0]["t0"]
        end = current_words[-1]["t1"]
        confs = [w.get("conf") for w in current_words if w.get("conf") is not None]
        conf = sum(confs) / len(confs) if confs else None

        # Aggregate overlap info: row has overlap if any word does
        has_overlap = any(w.get("overlap", False) for w in current_words)
        # Collect all unique secondary speakers from all words
        all_secondary: List[str] = []
        for w in current_words:
            for sp in w.get("secondary_speakers", []):
                if sp not in all_secondary:
                    all_secondary.append(sp)

        rows.append(
            TranscriptRow(
                start=start,
                end=end,
                speaker_id=speaker_id,
                speaker_display_name=speaker_display_name,
                voice_cluster_id=cluster_id,
                voice_bank_id=voice_bank_id,
                text=text.strip(),
                conf=conf,
                words=[
                    WordTiming(w=w["w"], t0=w["t0"], t1=w["t1"]) for w in current_words
                ],
                overlap=has_overlap,
                secondary_speakers=all_secondary,
            )
        )

    prev_word = None
    for w in words:
        if prev_word:
            gap = w.get("t0", 0.0) - prev_word.get("t1", 0.0)
            if (
                w.get("speaker_id") != prev_word.get("speaker_id")
                or gap > max_gap
            ):
                _flush()
                current_words = []
        current_words.append(w)
        prev_word = w

    _flush()
    return _merge_consecutive_speaker_rows(rows, max_gap=max_gap)


def _save_transcript_jsonl(rows: List[TranscriptRow], output_path: Path):
    """Save transcript to JSONL format."""
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")


def _load_transcript_jsonl(path: Path) -> List[TranscriptRow]:
    """Load transcript from JSONL format."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                rows.append(TranscriptRow(**data))
    return rows


def _save_transcript_vtt(
    rows: List[TranscriptRow],
    output_path: Path,
    include_speaker_notes: bool = True,
):
    """Save transcript to WebVTT format.

    Args:
        rows: Transcript rows
        output_path: Output file path
        include_speaker_notes: Whether to include NOTE lines with speaker info
    """
    with output_path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")

        for i, row in enumerate(rows, 1):
            # Format timestamps
            start_vtt = _seconds_to_vtt_time(row.start)
            end_vtt = _seconds_to_vtt_time(row.end)

            # Write speaker note with full voice metadata
            if include_speaker_notes:
                f.write(
                    f"NOTE speaker_id={row.speaker_id} "
                    f"speaker_display_name=\"{row.speaker_display_name}\" "
                    f"voice_cluster_id={row.voice_cluster_id} "
                    f"voice_bank_id={row.voice_bank_id}\n"
                )

            # Write cue
            f.write(f"{i}\n")
            f.write(f"{start_vtt} --> {end_vtt}\n")
            f.write(f"<v {row.speaker_display_name}>{row.text}\n\n")


def _seconds_to_vtt_time(seconds: float) -> str:
    """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def generate_vtt_from_jsonl(
    jsonl_path: Path,
    vtt_path: Path,
    include_speaker_notes: bool = True,
    overwrite: bool = False,
) -> Path:
    """Generate VTT file from existing JSONL transcript.

    Args:
        jsonl_path: Path to JSONL transcript
        vtt_path: Path for VTT output
        include_speaker_notes: Whether to include speaker notes
        overwrite: Whether to overwrite existing file

    Returns:
        Path to generated VTT file
    """
    if vtt_path.exists() and not overwrite:
        LOGGER.info(f"VTT already exists: {vtt_path}")
        return vtt_path

    rows = _load_transcript_jsonl(jsonl_path)
    _save_transcript_vtt(rows, vtt_path, include_speaker_notes)

    LOGGER.info(f"Generated VTT: {vtt_path}")
    return vtt_path


def generate_srt_from_jsonl(
    jsonl_path: Path,
    srt_path: Path,
    include_speaker_prefix: bool = True,
    overwrite: bool = False,
) -> Path:
    """Generate SRT file from existing JSONL transcript.

    Args:
        jsonl_path: Path to JSONL transcript
        srt_path: Path for SRT output
        include_speaker_prefix: Whether to prefix text with speaker name
        overwrite: Whether to overwrite existing file

    Returns:
        Path to generated SRT file
    """
    if srt_path.exists() and not overwrite:
        LOGGER.info(f"SRT already exists: {srt_path}")
        return srt_path

    rows = _load_transcript_jsonl(jsonl_path)

    with srt_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            start_srt = _seconds_to_srt_time(row.start)
            end_srt = _seconds_to_srt_time(row.end)

            text = row.text
            if include_speaker_prefix:
                text = f"{row.speaker_display_name}: {text}"

            f.write(f"{i}\n")
            f.write(f"{start_srt} --> {end_srt}\n")
            f.write(f"{text}\n\n")

    LOGGER.info(f"Generated SRT: {srt_path}")
    return srt_path


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
