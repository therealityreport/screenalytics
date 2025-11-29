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
from typing import Dict, List, Optional, Tuple

from .models import (
    ASRSegment,
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
    output_jsonl: Path,
    output_vtt: Path,
    include_speaker_notes: bool = True,
    overwrite: bool = False,
) -> List[TranscriptRow]:
    """Fuse diarization, ASR, and voice mapping into final transcript.

    Args:
        diarization_segments: Diarization results
        asr_segments: ASR results
        voice_clusters: Voice cluster data
        voice_mapping: Voice bank mapping results
        output_jsonl: Path for JSONL transcript
        output_vtt: Path for VTT transcript
        include_speaker_notes: Whether to include speaker notes in VTT
        overwrite: Whether to overwrite existing files

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
    cluster_lookup = {c.voice_cluster_id: c for c in voice_clusters}

    # Fuse segments
    transcript_rows = []

    for asr_segment in asr_segments:
        # Find overlapping diarization segment
        best_diar = _find_best_diarization_match(asr_segment, diarization_segments)

        if best_diar:
            # Find which voice cluster this belongs to
            cluster_id = _find_cluster_for_segment(best_diar, voice_clusters)
            mapping = cluster_mapping.get(cluster_id)

            if mapping:
                speaker_id = mapping.speaker_id
                speaker_display_name = mapping.speaker_display_name
                voice_bank_id = mapping.voice_bank_id
            else:
                # Fallback to diarization label
                speaker_id = f"SPK_{best_diar.speaker}"
                speaker_display_name = f"Speaker {best_diar.speaker}"
                voice_bank_id = "voice_unknown"
                cluster_id = cluster_id or "VC_00"
        else:
            # No diarization match - use unknown speaker
            speaker_id = "SPK_UNKNOWN"
            speaker_display_name = "Unknown Speaker"
            voice_bank_id = "voice_unknown"
            cluster_id = "VC_00"

        row = TranscriptRow(
            start=asr_segment.start,
            end=asr_segment.end,
            speaker_id=speaker_id,
            speaker_display_name=speaker_display_name,
            voice_cluster_id=cluster_id,
            voice_bank_id=voice_bank_id,
            text=asr_segment.text,
            conf=asr_segment.confidence,
            words=asr_segment.words,
        )
        transcript_rows.append(row)

    # Merge consecutive rows with same speaker
    transcript_rows = _merge_consecutive_speaker_rows(transcript_rows)

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


def _find_cluster_for_segment(
    diar_segment: DiarizationSegment,
    voice_clusters: List[VoiceCluster],
) -> Optional[str]:
    """Find which voice cluster contains a diarization segment."""
    for cluster in voice_clusters:
        for seg in cluster.segments:
            # Check if diarization segment is within cluster segment
            if seg.diar_speaker == diar_segment.speaker:
                # Check time overlap
                if (seg.start <= diar_segment.start < seg.end or
                    seg.start < diar_segment.end <= seg.end):
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
            )
        else:
            merged.append(current)
            current = row

    merged.append(current)
    return merged


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
