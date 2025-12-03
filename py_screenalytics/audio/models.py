"""Pydantic models for the audio pipeline.

These models define the data structures used throughout the audio pipeline:
- Configuration models
- Pipeline result models
- Voice clustering and bank models
- Transcript models
- QC models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QCStatus(str, Enum):
    """Quality control status values."""
    OK = "ok"
    WARN = "warn"
    NEEDS_REVIEW = "needs_review"
    UNKNOWN = "unknown"
    FAILED = "failed"


# ============================================================================
# Configuration Models
# ============================================================================


class SeparationConfig(BaseModel):
    """Configuration for audio stem separation."""
    model_config = {"protected_namespaces": ()}
    provider: str = "mdx_extra"
    model_name: str = "mdx_extra_q"
    chunk_seconds: int = 15
    overlap_seconds: int = 2
    device: str = "auto"


class EnhanceConfig(BaseModel):
    """Configuration for audio enhancement."""
    provider: str = "resemble"
    mode: str = "studio"
    batch_seconds: int = 20
    max_retries: int = 3
    retry_delay_seconds: int = 2


class DiarizationConfig(BaseModel):
    """Configuration for speaker diarization.

    Note: min_speakers/max_speakers are HINTS, not guarantees. Pyannote may still
    detect fewer speakers if it determines they are similar. Use num_speakers to
    force a specific count when you know it ahead of time.

    Backend options:
    - "precision-2": pyannoteAI cloud API (requires PYANNOTEAI_API_KEY)
    - "oss-3.1": Local open-source pyannote/speaker-diarization-3.1 model

    When backend="precision-2" and PYANNOTEAI_API_KEY is not set, falls back to oss-3.1.

    Official API workflow:
    - Speaker range is automatically calculated from cast count (cast ± 3)
    - exclusive=True for clean segment merging (non-overlapping speakers)
    - Poll interval is 5-8 seconds per official docs
    - Webhook support for async completion
    """
    model_config = {"protected_namespaces": ()}
    provider: str = "pyannote"
    backend: str = "precision-2"  # "precision-2" (API) | "oss-3.1" (local)
    model_name: str = "pyannote/speaker-diarization-precision-2"
    min_speech: float = 0.2
    max_overlap: float = 0.1
    merge_gap_ms: int = 300
    min_speakers: int = 1
    max_speakers: int = 10  # Safe upper bound for reality TV
    # Force exact speaker count (overrides min/max if set)
    num_speakers: Optional[int] = None
    # API timeout in seconds for cloud diarization (precision-2)
    api_timeout_seconds: int = 900  # 15 minutes - allows for long episodes and API load

    # Official API workflow fields
    use_exclusive_diarization: bool = True  # Always use exclusive mode for clean merging
    webhook_url: Optional[str] = None  # Optional webhook URL for async notification
    api_poll_interval_base: float = 6.0  # Base polling interval (5-8s per docs)
    api_poll_interval_jitter: float = 2.0  # Random jitter for polling


class ASRConfig(BaseModel):
    """Configuration for automatic speech recognition.

    OpenAI model options:
    - whisper-1: Legacy model, supports word timestamps
    - gpt-4o-transcribe: Higher quality, no word timestamps
    - gpt-4o-mini-transcribe: Faster, good quality, no word timestamps
    - gpt-4o-transcribe-diarize: Includes speaker diarization (can replace pyannote)
    """
    provider: str = "openai_whisper"
    model: str = "gpt-4o-transcribe"  # Default to higher quality model
    language: str = "en"
    enable_word_timestamps: bool = True  # Only works with whisper-1
    chunk_duration_seconds: int = 30
    temperature: float = 0.0
    gemini_model: str = "gemini-2.0-flash-exp"
    gemini_use_for_cleanup: bool = True
    # For gpt-4o-transcribe-diarize: use known speaker references
    use_diarization_model: bool = False  # Set True to use gpt-4o-transcribe-diarize
    known_speaker_names: List[str] = Field(default_factory=list)
    known_speaker_audio_paths: List[str] = Field(default_factory=list)


class VoiceClusteringConfig(BaseModel):
    """Configuration for voice clustering."""
    # Lower threshold = more clusters (different voices separated)
    # Higher threshold = fewer clusters (similar voices merged)
    # 0.65 works well for separating distinct speakers
    similarity_threshold: float = 0.65
    min_segments_per_cluster: int = 1  # Allow single-segment clusters for trailers
    embedding_model: str = "pyannote/embedding"
    centroid_method: str = "mean"
    # Skip embedding-based clustering and use pyannote speaker labels directly
    # Set True when pyannote diarization is accurate and clustering causes over-merging
    use_diarization_labels: bool = False


class VoiceBankConfig(BaseModel):
    """Configuration for voice bank."""
    data_dir: str = "data/voice_bank"
    auto_create_unlabeled: bool = True
    max_unlabeled_per_episode: int = 20


class VoiceprintIdentificationConfig(BaseModel):
    """Configuration for voiceprint identification pass.

    HARD RULES (Manual-First, ML-Second):
    1. Manual assignments are preserved unless allow_high_conf_override=True
       AND ident_confidence >= high_conf_override_threshold (80%)
    2. "if_better" policy requires 10%+ improvement in score
    3. Minimum 10s clean speech required to create voiceprint
    4. Identification never silently relabels - low confidence goes to review queue
    """

    # Segment selection for voiceprint creation
    min_segment_duration: float = 4.0  # Minimum segment duration in seconds
    max_segment_duration: float = 20.0  # Maximum segment duration (Pyannote limit is 30s)
    max_segments_per_cast: int = 3  # Max segments to use for voiceprint creation
    min_total_clean_speech_per_cast: float = 10.0  # Must have 10s+ to create voiceprint

    # Voiceprint creation policy
    voiceprint_overwrite_policy: str = "if_missing"  # "if_missing" | "always" | "if_better"
    # For "if_better": score = total_duration_s * mean_confidence
    # Only overwrite if new_score > old_score * improvement_threshold
    if_better_improvement_threshold: float = 1.1  # Require 10% improvement

    # Identification matching (Pyannote API params)
    ident_matching_threshold: int = 60  # Minimum confidence for matching (0-100)
    ident_matching_exclusive: bool = True  # Prevent multiple speakers matching same voiceprint

    # Transcript regeneration - MANUAL-FIRST, ML-SECOND
    ident_conf_threshold: float = 60.0  # Minimum confidence to assign cast from identification
    high_conf_override_threshold: float = 80.0  # Threshold to override MANUAL assignments
    allow_high_conf_override: bool = False  # Must explicitly enable overriding manual assignments
    preserve_manual_assignments: bool = True  # HARD RULE: manual corrections always win

    # Review queue generation
    diar_conf_threshold: float = 70.0  # Flag segments below this diarization confidence
    review_queue_enabled: bool = True  # Generate review queue artifact


class QCConfig(BaseModel):
    """Configuration for quality control."""
    max_duration_drift_pct: float = 1.0
    min_snr_db: float = 14.0
    warn_snr_db: float = 12.0
    min_diarization_conf: float = 0.65
    max_der_sample_pct: float = 12.0
    min_asr_conf: float = 0.60
    min_cluster_duration_s: float = 2.0
    require_all_speaker_fields: bool = True


class ExportConfig(BaseModel):
    """Configuration for audio export."""
    sample_rate: int = 48000
    bit_depth: int = 24
    peak_dbfs: float = -1.0
    audio_format: str = "wav"
    vtt_include_speaker_notes: bool = True


class AudioPipelineConfig(BaseModel):
    """Complete audio pipeline configuration."""
    separation: SeparationConfig = Field(default_factory=SeparationConfig)
    enhance: EnhanceConfig = Field(default_factory=EnhanceConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    voice_clustering: VoiceClusteringConfig = Field(default_factory=VoiceClusteringConfig)
    voice_bank: VoiceBankConfig = Field(default_factory=VoiceBankConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @classmethod
    def from_yaml(cls, yaml_dict: Dict[str, Any]) -> "AudioPipelineConfig":
        """Create config from YAML dict (handles nested audio_pipeline key)."""
        pipeline_cfg = yaml_dict.get("audio_pipeline", yaml_dict)
        return cls(**pipeline_cfg)


# ============================================================================
# Diarization Models
# ============================================================================


def generate_segment_id(start: float, end: float, prefix: str = "seg") -> str:
    """
    Generate a stable segment ID based on timestamps.

    Args:
        start: Start time in seconds
        end: End time in seconds
        prefix: ID prefix (e.g., "diar", "asr", "vc")

    Returns:
        Stable segment ID like "diar_12345_67890" (start/end in ms)
    """
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    return f"{prefix}_{start_ms}_{end_ms}"


class DiarizationSegment(BaseModel):
    """A single diarization segment."""
    segment_id: Optional[str] = Field(None, description="Stable segment identifier")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    speaker: str = Field(..., description="Raw speaker label from diarization")
    confidence: Optional[float] = Field(None, description="Diarization confidence")
    overlap_ratio: Optional[float] = Field(None, description="Overlap ratio with other speakers")

    def get_segment_id(self) -> str:
        """Get or generate stable segment ID."""
        if self.segment_id:
            return self.segment_id
        return generate_segment_id(self.start, self.end, "diar")


# ============================================================================
# Speaker Group Models
# ============================================================================


class SpeakerSegment(BaseModel):
    """A diarization segment belonging to a speaker group."""
    segment_id: str = Field(..., description="Stable segment identifier")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    diar_confidence: Optional[float] = Field(None, description="Diarization confidence from pyannote (0-1)")


class SpeakerGroup(BaseModel):
    """Collection of segments from a diarization speaker."""
    speaker_label: str = Field(..., description="Original diarization speaker label")
    speaker_group_id: str = Field(..., description="Unique group identifier (source-prefixed)")
    total_duration: float = Field(0.0, description="Total duration of all segments in seconds")
    segment_count: int = Field(0, description="Number of segments")
    segments: List[SpeakerSegment] = Field(default_factory=list)
    centroid: Optional[List[float]] = Field(None, description="Optional embedding centroid for the group")


class SpeakerSourceSummary(BaseModel):
    """Summary statistics for a diarization source."""
    speakers: int = 0
    segments: int = 0
    speech_seconds: float = 0.0


class SpeakerGroupSource(BaseModel):
    """Speaker groups for a diarization source (e.g., pyannote, gpt4o)."""
    source: str
    summary: SpeakerSourceSummary
    speakers: List[SpeakerGroup] = Field(default_factory=list)


class AudioSpeakerGroupsManifest(BaseModel):
    """Manifest of speaker groups per diarization source."""
    ep_id: str
    schema_version: str = "audio_sg_v1"
    sources: List[SpeakerGroupSource] = Field(default_factory=list)


class SmartSplitSegment(BaseModel):
    """Resulting segment from a smart split operation."""
    segment_id: str
    start: float
    end: float
    speaker_group_id: str


class SmartSplitResult(BaseModel):
    """Smart split response payload."""
    ep_id: str
    source: str
    original_segment_id: str
    new_segments: List[SmartSplitSegment]


# ============================================================================
# ASR Models
# ============================================================================


class WordTiming(BaseModel):
    """Word-level timing information."""
    w: str = Field(..., description="Word text")
    t0: float = Field(..., description="Word start time")
    t1: float = Field(..., description="Word end time")


class ASRSegment(BaseModel):
    """A single ASR segment."""
    segment_id: Optional[str] = Field(None, description="Stable segment identifier")
    start: float = Field(..., description="Segment start time")
    end: float = Field(..., description="Segment end time")
    text: str = Field(..., description="Transcribed text")
    confidence: Optional[float] = Field(None, description="ASR confidence score")
    words: Optional[List[WordTiming]] = Field(None, description="Word-level timings")
    language: Optional[str] = Field(None, description="Detected language")
    speaker: Optional[str] = Field(None, description="Speaker label (from gpt-4o-transcribe-diarize)")

    def get_segment_id(self) -> str:
        """Get or generate stable segment ID."""
        if self.segment_id:
            return self.segment_id
        return generate_segment_id(self.start, self.end, "asr")


# ============================================================================
# Voice Clustering Models
# ============================================================================


class VoiceClusterSegment(BaseModel):
    """A segment belonging to a voice cluster."""
    segment_id: Optional[str] = Field(None, description="Stable segment identifier")
    start: float
    end: float
    diar_speaker: str
    speaker_group_id: Optional[str] = Field(None, description="Speaker group identifier for the segment")

    def get_segment_id(self) -> str:
        """Get or generate stable segment ID."""
        if self.segment_id:
            return self.segment_id
        return generate_segment_id(self.start, self.end, "vc")


class VoiceClusterSourceGroup(BaseModel):
    """A speaker group contributing to a voice cluster."""
    source: str
    speaker_group_id: str
    speaker_label: Optional[str] = None
    centroid: Optional[List[float]] = None


class VoiceCluster(BaseModel):
    """A voice cluster representing a unique voice within an episode."""
    voice_cluster_id: str = Field(..., description="Cluster ID (e.g., VC_01)")
    segments: List[VoiceClusterSegment] = Field(default_factory=list)
    sources: List[VoiceClusterSourceGroup] = Field(default_factory=list)
    speaker_group_ids: List[str] = Field(default_factory=list, description="Speaker groups mapped to this cluster")
    total_duration: float = Field(0.0, description="Total speech duration")
    segment_count: int = Field(0, description="Number of segments")
    centroid: Optional[List[float]] = Field(None, description="Centroid embedding")


class VoiceBankEntry(BaseModel):
    """An entry in the voice bank."""
    voice_bank_id: str = Field(..., description="Voice bank ID")
    show_id: Optional[str] = Field(None, description="Associated show")
    cast_id: Optional[str] = Field(None, description="Associated cast member")
    display_name: str = Field(..., description="Display name for the voice")
    embeddings: List[List[float]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_labeled: bool = Field(True, description="Whether this is a labeled (known) voice")


class VoiceBankMatchResult(BaseModel):
    """Result of matching a voice cluster to the voice bank."""
    voice_cluster_id: str
    voice_bank_id: str
    speaker_id: str
    speaker_display_name: str
    similarity: Optional[float] = None
    is_new_entry: bool = False


# ============================================================================
# Transcript Models
# ============================================================================


class TranscriptRow(BaseModel):
    """A single row in the episode transcript."""
    start: float = Field(..., description="Segment start time")
    end: float = Field(..., description="Segment end time")
    speaker_id: str = Field(..., description="Speaker ID (e.g., SPK_LISA_BARLOW)")
    speaker_display_name: str = Field(..., description="Human-readable speaker name")
    voice_cluster_id: str = Field(..., description="Episode voice cluster ID")
    voice_bank_id: str = Field(..., description="Voice bank entry ID")
    text: str = Field(..., description="Transcribed text")
    conf: Optional[float] = Field(None, description="Confidence score")
    words: Optional[List[WordTiming]] = Field(None, description="Word-level timings")


# ============================================================================
# QC Models
# ============================================================================


class QCMetric(BaseModel):
    """A single QC metric."""
    name: str
    value: float
    threshold: float
    passed: bool
    severity: str = "info"  # info, warn, error


class QCReport(BaseModel):
    """Quality control report for the audio pipeline."""
    ep_id: str
    status: QCStatus = QCStatus.UNKNOWN
    metrics: List[QCMetric] = Field(default_factory=list)
    duration_original_s: Optional[float] = None
    duration_final_s: Optional[float] = None
    duration_drift_pct: Optional[float] = None
    snr_db: Optional[float] = None
    mean_diarization_conf: Optional[float] = None
    mean_asr_conf: Optional[float] = None
    der_sample_pct: Optional[float] = None
    voice_cluster_count: int = 0
    labeled_voices: int = 0
    unlabeled_voices: int = 0
    transcript_row_count: int = 0
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


# ============================================================================
# Pipeline Result Models
# ============================================================================


@dataclass
class AudioArtifacts:
    """Paths to audio artifacts."""
    original: Optional[Path] = None
    vocals: Optional[Path] = None
    vocals_enhanced: Optional[Path] = None
    final_voice_only: Optional[Path] = None


@dataclass
class ManifestArtifacts:
    """Paths to manifest artifacts."""
    diarization: Optional[Path] = None
    diarization_pyannote: Optional[Path] = None
    diarization_gpt4o: Optional[Path] = None
    speaker_groups: Optional[Path] = None
    asr_raw: Optional[Path] = None
    voice_clusters: Optional[Path] = None
    voice_mapping: Optional[Path] = None
    transcript_jsonl: Optional[Path] = None
    transcript_vtt: Optional[Path] = None
    qc: Optional[Path] = None


@dataclass
class AudioPipelineResult:
    """Result of running the audio pipeline."""
    ep_id: str
    status: str = "unknown"
    qc_status: QCStatus = QCStatus.UNKNOWN
    audio_artifacts: AudioArtifacts = field(default_factory=AudioArtifacts)
    manifest_artifacts: ManifestArtifacts = field(default_factory=ManifestArtifacts)
    duration_original_s: float = 0.0
    duration_final_s: float = 0.0
    voice_cluster_count: int = 0
    labeled_voices: int = 0
    unlabeled_voices: int = 0
    transcript_row_count: int = 0
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ep_id": self.ep_id,
            "status": self.status,
            "qc_status": self.qc_status.value if isinstance(self.qc_status, QCStatus) else self.qc_status,
            "audio_artifacts": {
                "original": str(self.audio_artifacts.original) if self.audio_artifacts.original else None,
                "vocals": str(self.audio_artifacts.vocals) if self.audio_artifacts.vocals else None,
                "vocals_enhanced": str(self.audio_artifacts.vocals_enhanced) if self.audio_artifacts.vocals_enhanced else None,
                "final_voice_only": str(self.audio_artifacts.final_voice_only) if self.audio_artifacts.final_voice_only else None,
            },
            "manifest_artifacts": {
                "diarization": str(self.manifest_artifacts.diarization) if self.manifest_artifacts.diarization else None,
                "diarization_pyannote": str(self.manifest_artifacts.diarization_pyannote) if self.manifest_artifacts.diarization_pyannote else None,
                "diarization_gpt4o": str(self.manifest_artifacts.diarization_gpt4o) if self.manifest_artifacts.diarization_gpt4o else None,
                "speaker_groups": str(self.manifest_artifacts.speaker_groups) if self.manifest_artifacts.speaker_groups else None,
                "asr_raw": str(self.manifest_artifacts.asr_raw) if self.manifest_artifacts.asr_raw else None,
                "voice_clusters": str(self.manifest_artifacts.voice_clusters) if self.manifest_artifacts.voice_clusters else None,
                "voice_mapping": str(self.manifest_artifacts.voice_mapping) if self.manifest_artifacts.voice_mapping else None,
                "transcript_jsonl": str(self.manifest_artifacts.transcript_jsonl) if self.manifest_artifacts.transcript_jsonl else None,
                "transcript_vtt": str(self.manifest_artifacts.transcript_vtt) if self.manifest_artifacts.transcript_vtt else None,
                "qc": str(self.manifest_artifacts.qc) if self.manifest_artifacts.qc else None,
            },
            "duration_original_s": self.duration_original_s,
            "duration_final_s": self.duration_final_s,
            "voice_cluster_count": self.voice_cluster_count,
            "labeled_voices": self.labeled_voices,
            "unlabeled_voices": self.unlabeled_voices,
            "transcript_row_count": self.transcript_row_count,
            "error": self.error,
            "metrics": self.metrics,
        }
