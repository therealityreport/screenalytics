"""Helpers for diarization comparison enrichment and canonical text overlays."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

LOGGER = logging.getLogger(__name__)


def _load_transcript_rows(transcript_path: Path) -> List[dict]:
    if not transcript_path.exists():
        return []
    rows: List[dict] = []
    try:
        with transcript_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to load transcript rows for comparison: %s", exc)
    return rows


def _words_in_range(rows: List[dict], start: float, end: float) -> List[dict]:
    """Collect word dicts overlapping a time range."""
    words: List[dict] = []
    for row in rows:
        for w in row.get("words") or []:
            t0 = w.get("t0", 0.0)
            t1 = w.get("t1", 0.0)
            if t0 < end and t1 > start:
                words.append(w)
    return sorted(words, key=lambda w: w.get("t0", 0.0))


def get_canonical_text_for_segment(
    transcript_path: Path,
    start: float,
    end: float,
    min_words: int = 2,
) -> Optional[str]:
    """Prefer Whisper/ASR words to build canonical text for a diarization segment."""
    rows = _load_transcript_rows(transcript_path)
    words = _words_in_range(rows, start, end)
    if len(words) < min_words:
        return None

    tokens: List[str] = []
    last_end = None
    for w in words:
        token = w.get("w", "").strip()
        if not token:
            continue
        # Basic spacing: add a space if there is a small gap
        if last_end is not None and w.get("t0", 0.0) - last_end > 0.02:
            tokens.append(" ")
        tokens.append(token)
        last_end = w.get("t1", last_end)

    text = "".join(tokens).strip()
    return text or None


def _speaker_ids_in_range(rows: List[dict], start: float, end: float) -> Set[str]:
    speakers: Set[str] = set()
    for row in rows:
        row_speaker = row.get("speaker_id")
        row_start = row.get("start", 0.0)
        row_end = row.get("end", 0.0)
        for w in row.get("words") or []:
            t0 = w.get("t0", 0.0)
            t1 = w.get("t1", 0.0)
            if t0 < end and t1 > start:
                spk = w.get("speaker_id") or w.get("speaker") or row_speaker
                if spk:
                    speakers.add(str(spk))
        if speakers and row_start < end and row_end > start and row_speaker:
            speakers.add(str(row_speaker))
    return speakers


def augment_diarization_comparison(
    comparison_path: Path,
    transcript_path: Path,
) -> Optional[dict]:
    """Augment diarization comparison with canonical text and mixed-speaker flags."""
    if not comparison_path.exists():
        return None
    try:
        data = json.loads(comparison_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to read comparison file: %s", exc)
        return None

    if not transcript_path.exists():
        return data

    rows = _load_transcript_rows(transcript_path)
    segments = data.get("segments", {})
    gpt4o_segments = segments.get("gpt4o", []) if isinstance(segments, dict) else []

    updated = False
    for seg in gpt4o_segments:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        canonical = seg.get("canonical_text")
        if not canonical:
            canonical = get_canonical_text_for_segment(transcript_path, start, end)
            if canonical:
                seg["canonical_text"] = canonical
                updated = True
        speakers = seg.get("speakers")
        if speakers is None:
            speaker_ids = list(_speaker_ids_in_range(rows, start, end))
            seg["speakers"] = speaker_ids
            seg["mixed_speaker"] = len(speaker_ids) > 1
            updated = True
        elif "mixed_speaker" not in seg:
            seg["mixed_speaker"] = len(speakers) > 1
            updated = True

    if updated:
        data.setdefault("segments", {})["gpt4o"] = gpt4o_segments
        comparison_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return data
