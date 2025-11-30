"""SCREENALYTICS Audio Pipeline Module.

This module provides audio processing capabilities for episode transcription:
- Audio extraction and normalization
- MDX-Extra stem separation (vocals vs accompaniment)
- Resemble Enhance audio enhancement
- Pyannote speaker diarization
- OpenAI Whisper / Gemini ASR
- Voice clustering and voice bank integration
- Transcript generation (JSONL + VTT)
"""

from __future__ import annotations

from .models import (
    AudioPipelineConfig,
    AudioPipelineResult,
    DiarizationSegment,
    ASRSegment,
    VoiceCluster,
    VoiceBankEntry,
    VoiceBankMatchResult,
    TranscriptRow,
    QCReport,
    QCStatus,
)

__all__ = [
    # Config and results
    "AudioPipelineConfig",
    "AudioPipelineResult",
    # Diarization
    "DiarizationSegment",
    # ASR
    "ASRSegment",
    # Voice clustering
    "VoiceCluster",
    "VoiceBankEntry",
    "VoiceBankMatchResult",
    # Transcript
    "TranscriptRow",
    # QC
    "QCReport",
    "QCStatus",
]
