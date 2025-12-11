"""NeMo MSDD speaker diarization integration.

Handles:
- Overlap-aware speaker diarization using NVIDIA NeMo MSDD
- Speaker embedding extraction using TitaNet/ECAPA-TDNN
- Multi-speaker segment detection with probability outputs

Backend: NeMo Neural Diarizer with MSDD (Multi-Scale Diarization Decoder)
Requires: GPU with CUDA support for optimal performance

Reference:
- https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/models.html
- https://github.com/NVIDIA/NeMo/blob/main/examples/speaker_tasks/diarization/neural_diarizer/
"""

from __future__ import annotations

import gc
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from .models import DiarizationConfig, DiarizationSegment, generate_segment_id

LOGGER = logging.getLogger(__name__)

# Global model cache
_NEMO_DIARIZER = None
_NEMO_EMBEDDING_MODEL = None
_NEMO_BACKEND = None
_NEMO_DEVICE = None


# ============================================================================
# NeMo-specific Configuration and Models
# ============================================================================


class NeMoDiarizationConfig(BaseModel):
    """Configuration for NeMo MSDD speaker diarization.

    Key parameters:
    - vad_model: Voice activity detection model
    - embedding_model: Speaker embedding model (TitaNet or ECAPA-TDNN)
    - msdd_model: Multi-scale diarization decoder model
    - overlap_threshold: Probability threshold for multi-speaker detection
    """

    model_config = {"protected_namespaces": ()}

    # VAD settings
    vad_model: str = "vad_multilingual_marblenet"
    vad_threshold: float = 0.5
    vad_min_speech_duration: float = 0.25
    vad_min_silence_duration: float = 0.1

    # Speaker embedding model
    embedding_model: str = "titanet_large"  # "titanet_large" | "ecapa_tdnn"
    embedding_dim: int = 192  # 192 for TitaNet-Large

    # MSDD model settings
    msdd_model: str = "diar_msdd_telephonic"
    msdd_model_path: Optional[str] = None  # Override with local path

    # Multi-scale window settings
    window_lengths: List[float] = Field(default_factory=lambda: [1.5, 1.25, 1.0, 0.75, 0.5])
    shift_lengths: List[float] = Field(default_factory=lambda: [0.75, 0.625, 0.5, 0.375, 0.25])
    scale_weights: List[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0])

    # Clustering settings
    max_num_speakers: int = 8
    min_num_speakers: int = 1
    oracle_num_speakers: Optional[int] = None  # Force exact count if known

    # Post-processing
    overlap_threshold: float = 0.5  # Probability threshold for multi-speaker segments
    merge_gap_ms: int = 300
    min_speech_duration: float = 0.2

    # Device settings
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    batch_size: int = 32

    # Output control
    output_speaker_probabilities: bool = True
    save_raw_rttm: bool = True


class NeMoDiarizationSegment(BaseModel):
    """A diarization segment with NeMo MSDD overlap-aware fields.

    Extends standard segment with:
    - overlap: Whether multiple speakers are active
    - active_speakers: All speakers active in this segment
    - speaker_probs: Per-speaker probability scores
    """

    segment_id: Optional[str] = Field(None, description="Stable segment identifier")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    speaker: str = Field(..., description="Primary speaker label")
    confidence: Optional[float] = Field(None, description="Confidence score (max probability)")

    # NeMo MSDD specific fields
    overlap: bool = Field(False, description="True if multiple speakers active")
    active_speakers: List[str] = Field(default_factory=list, description="All active speakers")
    speaker_probs: Dict[str, float] = Field(default_factory=dict, description="Per-speaker probabilities")

    def get_segment_id(self) -> str:
        """Get or generate stable segment ID."""
        if self.segment_id:
            return self.segment_id
        return generate_segment_id(self.start, self.end, "nemo")

    def to_standard_segment(self) -> DiarizationSegment:
        """Convert to standard DiarizationSegment for backward compatibility."""
        return DiarizationSegment(
            segment_id=self.get_segment_id(),
            start=self.start,
            end=self.end,
            speaker=self.speaker,
            confidence=self.confidence,
            overlap_ratio=len(self.active_speakers) - 1 if self.overlap else 0.0,
        )


class NeMoDiarizationResult(BaseModel):
    """Complete NeMo diarization result with embeddings and metadata."""

    segments: List[NeMoDiarizationSegment] = Field(default_factory=list)
    embeddings: Dict[str, List[float]] = Field(default_factory=dict, description="speaker_id -> embedding vector")
    speaker_count: int = 0
    total_speech_duration: float = 0.0
    overlap_duration: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Device and Environment Utilities
# ============================================================================


def _get_device(config: NeMoDiarizationConfig) -> str:
    """Determine compute device."""
    if config.device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return config.device


def _preprocess_audio_for_nemo(audio_path: Path, output_path: Path) -> Path:
    """Convert audio to mono 16kHz WAV format required by NeMo.

    Args:
        audio_path: Input audio file (any format)
        output_path: Where to save the preprocessed audio

    Returns:
        Path to the preprocessed audio file
    """
    import subprocess

    # Check if preprocessing is needed
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channels,sample_rate",
             "-of", "csv=p=0", str(audio_path)],
            capture_output=True, text=True, check=True
        )
        parts = result.stdout.strip().split(",")
        if len(parts) >= 2:
            channels = int(parts[0])
            sample_rate = int(parts[1])

            # Skip if already in correct format
            if channels == 1 and sample_rate == 16000:
                LOGGER.info(f"Audio already in NeMo format (mono 16kHz): {audio_path}")
                return audio_path
    except (subprocess.CalledProcessError, ValueError) as e:
        LOGGER.warning(f"Could not probe audio format: {e}, will convert anyway")

    LOGGER.info(f"Converting audio to mono 16kHz for NeMo: {audio_path}")

    # Convert to mono 16kHz using ffmpeg
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path),
         "-ac", "1",  # mono
         "-ar", "16000",  # 16kHz
         "-acodec", "pcm_s16le",  # 16-bit PCM
         str(output_path)],
        capture_output=True, check=True
    )

    LOGGER.info(f"Audio converted to mono 16kHz: {output_path}")
    return output_path


def _check_nemo_available() -> bool:
    """Check if NeMo toolkit is available."""
    try:
        import nemo.collections.asr as nemo_asr

        return True
    except ImportError:
        return False


def _check_cuda_available() -> bool:
    """Check if CUDA is available for GPU inference."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# ============================================================================
# Configuration Loading
# ============================================================================


@lru_cache(maxsize=1)
def _load_nemo_config(config_path: Optional[Path] = None) -> dict:
    """Load NeMo MSDD configuration from YAML file."""
    if config_path is None:
        # Find config relative to project
        candidates = [
            Path("config/nemo/msdd_inference.yaml"),
            Path(__file__).parents[3] / "config" / "nemo" / "msdd_inference.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path and config_path.exists():
        try:
            import yaml

            with config_path.open() as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            LOGGER.warning(f"Failed to load NeMo config from {config_path}: {e}")

    return {}


# ============================================================================
# Model Building and Caching
# ============================================================================


def _build_nemo_diarizer(config: NeMoDiarizationConfig, temp_dir: Path):
    """Build NeMo Neural Diarizer pipeline.

    Loads:
    - VAD model (MarbleNet)
    - Speaker embedding model (TitaNet)
    - MSDD model for overlap-aware decoding
    """
    try:
        from nemo.collections.asr.models.msdd_models import NeuralDiarizer
        from omegaconf import OmegaConf
    except ImportError as e:
        raise ImportError(
            "NeMo toolkit is required for MSDD diarization. "
            "Install with: pip install nemo_toolkit[asr]"
        ) from e

    device = _get_device(config)

    LOGGER.info(f"Building NeMo Neural Diarizer on device: {device}")

    # Build Hydra-style config for NeuralDiarizer
    # Reference: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/configs.html
    # Updated for NeMo 2.6+ config schema
    diarizer_cfg = OmegaConf.create(
        {
            # Root-level inference settings (required in NeMo 2.6+)
            "device": device,
            "verbose": True,
            "batch_size": 64,
            "num_workers": 0,  # Must be 0 on macOS to avoid pickle errors with spawn
            "sample_rate": 16000,
            "diarizer": {
                "manifest_filepath": str(temp_dir / "input_manifest.json"),  # Placeholder, updated per-call
                "out_dir": str(temp_dir),
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": False,
                "vad": {
                    "model_path": config.vad_model,
                    "external_vad_manifest": None,
                    "parameters": {
                        # Window/shift for frame-level VAD
                        "window_length_in_sec": 0.15,
                        "shift_length_in_sec": 0.01,
                        "smoothing": "median",
                        "overlap": 0.875,
                        # Thresholds
                        "onset": config.vad_threshold,
                        "offset": max(0.1, config.vad_threshold - 0.1),
                        # Padding
                        "pad_onset": 0.1,
                        "pad_offset": 0.1,
                        # Duration constraints
                        "min_duration_on": config.vad_min_speech_duration,
                        "min_duration_off": config.vad_min_silence_duration,
                        "filter_speech_first": True,
                    },
                },
                "speaker_embeddings": {
                    "model_path": config.embedding_model,
                    "parameters": {
                        # Multi-scale window settings - use lists for MSDD compatibility
                        "window_length_in_sec": config.window_lengths if isinstance(config.window_lengths, list) else [config.window_lengths],
                        "shift_length_in_sec": config.shift_lengths if isinstance(config.shift_lengths, list) else [config.shift_lengths],
                        "multiscale_weights": config.scale_weights if isinstance(config.scale_weights, list) else [config.scale_weights],
                        "save_embeddings": True,  # Required for clustering step
                    },
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": config.oracle_num_speakers is not None,
                        "max_num_speakers": config.max_num_speakers,
                        "enhanced_count_thres": 80,
                        "max_rp_threshold": 0.25,
                        "sparse_search_volume": 30,
                        "maj_vote_spk_count": False,
                    }
                },
                "msdd_model": {
                    "model_path": config.msdd_model_path or config.msdd_model,
                    "parameters": {
                        "infer_batch_size": 25,
                        "sigmoid_threshold": [config.overlap_threshold],
                        "seq_eval_mode": False,  # False = inference only, no RTTM ground truth needed
                        "split_infer": True,
                        "diar_window_length": 50,
                        "overlap_infer_spk_limit": 5,
                        # Additional parameters for NeMo 2.6+
                        "use_speaker_model_from_ckpt": True,
                        "use_clus_as_main": False,
                        "max_overlap_spks": 2,
                        "num_spks_per_model": 2,
                        "use_adaptive_thres": True,
                        "max_pred_length": 0,
                        "diar_eval_settings": [[0.25, True], [0.25, False], [0.0, False]],
                    },
                },
            },
        }
    )

    # Set oracle speaker count if provided
    if config.oracle_num_speakers is not None:
        diarizer_cfg.diarizer.clustering.parameters.oracle_num_speakers = True
        diarizer_cfg.diarizer.oracle_num_speakers = config.oracle_num_speakers

    diarizer = NeuralDiarizer(cfg=diarizer_cfg)

    return diarizer


def _get_nemo_diarizer(config: NeMoDiarizationConfig, temp_dir: Path):
    """Load or retrieve cached NeMo diarizer."""
    global _NEMO_DIARIZER, _NEMO_BACKEND, _NEMO_DEVICE

    requested_device = _get_device(config)

    # Check if we need to rebuild
    if (
        _NEMO_DIARIZER is not None
        and _NEMO_BACKEND == "nemo-msdd"
        and _NEMO_DEVICE == requested_device
    ):
        # Update paths for this run
        _NEMO_DIARIZER._cfg.diarizer.out_dir = str(temp_dir)
        _NEMO_DIARIZER._cfg.diarizer.manifest_filepath = str(temp_dir / "input_manifest.json")
        return _NEMO_DIARIZER

    # Build new diarizer
    diarizer = _build_nemo_diarizer(config, temp_dir)
    _NEMO_DIARIZER = diarizer
    _NEMO_BACKEND = "nemo-msdd"
    _NEMO_DEVICE = requested_device

    return diarizer


# ============================================================================
# Speaker Embedding Extraction
# ============================================================================


def _build_embedding_model(config: NeMoDiarizationConfig):
    """Build TitaNet or ECAPA-TDNN embedding model."""
    try:
        import nemo.collections.asr as nemo_asr
        import torch
    except ImportError as e:
        raise ImportError(
            "NeMo toolkit is required for speaker embeddings. "
            "Install with: pip install nemo_toolkit[asr]"
        ) from e

    device = _get_device(config)

    LOGGER.info(f"Loading NeMo embedding model: {config.embedding_model}")

    # Load pretrained model
    if config.embedding_model == "titanet_large":
        model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        )
    elif config.embedding_model == "ecapa_tdnn":
        model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_ecapa_tdnn"
        )
    else:
        # Try loading by name/path
        model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(config.embedding_model)

    model = model.to(device)
    model.eval()

    return model


def _get_embedding_model(config: NeMoDiarizationConfig):
    """Load or retrieve cached embedding model."""
    global _NEMO_EMBEDDING_MODEL

    if _NEMO_EMBEDDING_MODEL is not None:
        return _NEMO_EMBEDDING_MODEL

    model = _build_embedding_model(config)
    _NEMO_EMBEDDING_MODEL = model

    return model


def extract_speaker_embeddings_nemo(
    audio_path: Path,
    segments: List[DiarizationSegment],
    config: Optional[NeMoDiarizationConfig] = None,
) -> Dict[str, np.ndarray]:
    """Extract speaker embeddings using TitaNet/ECAPA-TDNN.

    Args:
        audio_path: Path to audio file (16kHz mono WAV recommended)
        segments: Diarization segments with speaker labels
        config: NeMo configuration

    Returns:
        Dict mapping speaker_id to centroid embedding (192-dim for TitaNet)
    """
    config = config or NeMoDiarizationConfig()
    model = _get_embedding_model(config)

    try:
        import torch
        import torchaudio
    except ImportError as e:
        raise ImportError("torchaudio required for embedding extraction") from e

    # Load audio
    waveform, sample_rate = torchaudio.load(str(audio_path))

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Group segments by speaker
    speaker_segments: Dict[str, List[DiarizationSegment]] = {}
    for seg in segments:
        if seg.speaker not in speaker_segments:
            speaker_segments[seg.speaker] = []
        speaker_segments[seg.speaker].append(seg)

    # Extract embeddings per speaker
    embeddings: Dict[str, List[np.ndarray]] = {}
    device = _get_device(config)

    for speaker, segs in speaker_segments.items():
        speaker_embeddings = []

        for seg in segs:
            # Skip very short segments
            if (seg.end - seg.start) < 0.5:
                continue

            # Extract segment audio
            start_sample = int(seg.start * sample_rate)
            end_sample = int(seg.end * sample_rate)
            segment_audio = waveform[:, start_sample:end_sample]

            if segment_audio.shape[1] < 8000:  # Min 0.5s at 16kHz
                continue

            # Get embedding
            with torch.no_grad():
                segment_audio = segment_audio.to(device)
                audio_length = torch.tensor([segment_audio.shape[1]], device=device)
                _, emb = model.forward(input_signal=segment_audio, input_signal_length=audio_length)
                speaker_embeddings.append(emb.cpu().numpy().flatten())

        if speaker_embeddings:
            embeddings[speaker] = speaker_embeddings

    # Compute centroids
    centroids: Dict[str, np.ndarray] = {}
    for speaker, embs in embeddings.items():
        centroid = np.mean(embs, axis=0)
        # Normalize to unit length
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[speaker] = centroid

    LOGGER.info(f"Extracted {len(centroids)} speaker embeddings ({config.embedding_dim}-dim)")
    return centroids


def extract_speaker_embeddings(
    audio_path: Path,
    segments: List[DiarizationSegment],
    embedding_model: str = "titanet_large",
) -> List[Tuple[DiarizationSegment, np.ndarray]]:
    """Extract speaker embeddings for each diarization segment.

    Compatible interface with pyannote version.

    Args:
        audio_path: Path to audio file
        segments: List of diarization segments
        embedding_model: Name of embedding model to use

    Returns:
        List of (segment, embedding) tuples
    """
    config = NeMoDiarizationConfig(embedding_model=embedding_model)
    centroids = extract_speaker_embeddings_nemo(audio_path, segments, config)

    # Return in compatible format
    results = []
    for seg in segments:
        if seg.speaker in centroids:
            results.append((seg, centroids[seg.speaker]))

    return results


# ============================================================================
# RTTM Parsing
# ============================================================================


def _parse_rttm_to_segments(
    rttm_path: Path,
    config: NeMoDiarizationConfig,
) -> List[NeMoDiarizationSegment]:
    """Parse NeMo RTTM output to segment objects.

    RTTM format: SPEAKER file 1 start dur <NA> <NA> speaker <NA> <NA>
    """
    segments = []

    if not rttm_path.exists():
        LOGGER.warning(f"RTTM file not found: {rttm_path}")
        return segments

    with rttm_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue

            # RTTM format: SPEAKER file 1 start dur <NA> <NA> speaker <NA> <NA>
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]

            segment = NeMoDiarizationSegment(
                start=start,
                end=start + duration,
                speaker=speaker,
                overlap=False,  # Will be updated if probs available
                active_speakers=[speaker],
                speaker_probs={speaker: 1.0},
            )
            segment.segment_id = segment.get_segment_id()
            segments.append(segment)

    return segments


def _augment_segments_with_probs(
    segments: List[NeMoDiarizationSegment],
    probs_path: Path,
    config: NeMoDiarizationConfig,
) -> List[NeMoDiarizationSegment]:
    """Augment segments with speaker probability information.

    MSDD outputs per-frame speaker probabilities which we aggregate
    to identify overlapping speech regions.
    """
    if not probs_path.exists():
        return segments

    try:
        probs = np.load(probs_path)
        # probs shape: (num_frames, num_speakers)
    except Exception as e:
        LOGGER.warning(f"Failed to load speaker probabilities: {e}")
        return segments

    # Frame rate based on MSDD config (typically 100fps = 10ms frames)
    frame_rate = 100

    augmented = []
    for seg in segments:
        start_frame = int(seg.start * frame_rate)
        end_frame = int(seg.end * frame_rate)

        if end_frame <= start_frame or end_frame > len(probs):
            augmented.append(seg)
            continue

        segment_probs = probs[start_frame:end_frame]

        # Average probabilities over segment
        mean_probs = np.mean(segment_probs, axis=0)

        # Find active speakers (above threshold)
        threshold = config.overlap_threshold
        active_mask = mean_probs > threshold
        active_speakers = [f"speaker_{i}" for i, active in enumerate(active_mask) if active]

        # Check for overlap
        is_overlap = len(active_speakers) > 1

        # Build speaker_probs dict
        speaker_probs = {f"speaker_{i}": float(p) for i, p in enumerate(mean_probs) if p > 0.1}

        augmented.append(
            NeMoDiarizationSegment(
                segment_id=seg.segment_id,
                start=seg.start,
                end=seg.end,
                speaker=seg.speaker,
                confidence=float(mean_probs.max()) if len(mean_probs) > 0 else None,
                overlap=is_overlap,
                active_speakers=active_speakers if active_speakers else [seg.speaker],
                speaker_probs=speaker_probs,
            )
        )

    return augmented


# ============================================================================
# Post-processing
# ============================================================================


def _merge_segments(
    segments: List[NeMoDiarizationSegment],
    merge_gap: float,
) -> List[NeMoDiarizationSegment]:
    """Merge nearby same-speaker segments."""
    if not segments:
        return segments

    # Sort by start time
    sorted_segs = sorted(segments, key=lambda s: s.start)
    merged = [sorted_segs[0]]

    for seg in sorted_segs[1:]:
        last = merged[-1]

        # Check if same speaker and close enough to merge
        if seg.speaker == last.speaker and (seg.start - last.end) <= merge_gap:
            # Merge: extend last segment
            merged[-1] = NeMoDiarizationSegment(
                segment_id=None,  # Will be regenerated
                start=last.start,
                end=seg.end,
                speaker=last.speaker,
                confidence=max(last.confidence or 0, seg.confidence or 0),
                overlap=last.overlap or seg.overlap,
                active_speakers=list(set(last.active_speakers + seg.active_speakers)),
                speaker_probs={
                    **last.speaker_probs,
                    **{k: max(v, last.speaker_probs.get(k, 0)) for k, v in seg.speaker_probs.items()},
                },
            )
            merged[-1].segment_id = merged[-1].get_segment_id()
        else:
            merged.append(seg)

    return merged


# ============================================================================
# Manifest I/O
# ============================================================================


def _save_diarization_manifest(
    segments: List[NeMoDiarizationSegment],
    output_path: Path,
) -> None:
    """Save diarization results to JSONL manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for seg in segments:
            f.write(seg.model_dump_json() + "\n")

    LOGGER.info(f"Saved {len(segments)} diarization segments to {output_path}")


def _load_diarization_manifest(input_path: Path) -> List[NeMoDiarizationSegment]:
    """Load diarization results from JSONL manifest."""
    segments = []

    if not input_path.exists():
        return segments

    with input_path.open() as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                segments.append(NeMoDiarizationSegment(**data))

    return segments


def load_diarization_manifest(input_path: Path) -> List[NeMoDiarizationSegment]:
    """Load diarization results from JSONL manifest.

    Public API wrapper for _load_diarization_manifest.
    """
    return _load_diarization_manifest(input_path)


def save_diarization_manifest(
    segments: List[Union[NeMoDiarizationSegment, "DiarizationSegment"]],
    output_path: Path,
) -> None:
    """Save diarization segments to JSONL manifest.

    Public API for saving diarization results. Accepts both NeMoDiarizationSegment
    and standard DiarizationSegment objects.

    Args:
        segments: List of diarization segments (NeMo or standard)
        output_path: Path to write JSONL manifest
    """
    from .models import DiarizationSegment

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for seg in segments:
            # Handle both NeMoDiarizationSegment and DiarizationSegment
            if isinstance(seg, NeMoDiarizationSegment):
                f.write(seg.model_dump_json() + "\n")
            elif isinstance(seg, DiarizationSegment):
                f.write(seg.model_dump_json() + "\n")
            else:
                # Fallback for dict-like objects
                f.write(json.dumps(dict(seg)) + "\n")

    LOGGER.info(f"Saved {len(segments)} diarization segments to {output_path}")


def _save_nemo_result(result: NeMoDiarizationResult, output_path: Path) -> None:
    """Save complete NeMo result including embeddings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save segments to JSONL
    _save_diarization_manifest(result.segments, output_path)

    # Save embeddings separately
    embeddings_path = output_path.with_suffix(".embeddings.json")
    with embeddings_path.open("w") as f:
        json.dump(
            {
                "embeddings": result.embeddings,
                "speaker_count": result.speaker_count,
                "total_speech_duration": result.total_speech_duration,
                "overlap_duration": result.overlap_duration,
                "metadata": result.metadata,
            },
            f,
            indent=2,
        )


def _load_nemo_result(input_path: Path) -> NeMoDiarizationResult:
    """Load complete NeMo result including embeddings."""
    segments = _load_diarization_manifest(input_path)

    embeddings_path = input_path.with_suffix(".embeddings.json")
    if embeddings_path.exists():
        with embeddings_path.open() as f:
            data = json.load(f)
        return NeMoDiarizationResult(
            segments=segments,
            embeddings=data.get("embeddings", {}),
            speaker_count=data.get("speaker_count", len(set(s.speaker for s in segments))),
            total_speech_duration=data.get("total_speech_duration", sum(s.end - s.start for s in segments)),
            overlap_duration=data.get("overlap_duration", sum(s.end - s.start for s in segments if s.overlap)),
            metadata=data.get("metadata", {}),
        )

    # Fallback: just segments
    return NeMoDiarizationResult(
        segments=segments,
        speaker_count=len(set(s.speaker for s in segments)),
        total_speech_duration=sum(s.end - s.start for s in segments),
        overlap_duration=sum(s.end - s.start for s in segments if s.overlap),
    )


# ============================================================================
# Main Diarization Function
# ============================================================================


def run_diarization_nemo(
    audio_path: Path,
    output_path: Path,
    config: Optional[NeMoDiarizationConfig] = None,
    overwrite: bool = False,
) -> NeMoDiarizationResult:
    """Run NeMo MSDD speaker diarization.

    Args:
        audio_path: Path to input audio file (16kHz WAV recommended)
        output_path: Path for diarization manifest (JSONL)
        config: NeMo diarization configuration
        overwrite: Whether to overwrite existing results

    Returns:
        NemoDiarizationResult with segments, embeddings, and metadata
    """
    global _NEMO_BACKEND

    # Check for existing results
    if output_path.exists() and not overwrite:
        LOGGER.info(f"NeMo diarization results already exist: {output_path}")
        return _load_nemo_result(output_path)

    config = config or NeMoDiarizationConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate environment
    if not _check_nemo_available():
        raise ImportError(
            "NeMo toolkit is not available. Install with: pip install nemo_toolkit[asr]"
        )

    device = _get_device(config)
    if device == "cpu":
        LOGGER.warning("NeMo MSDD is GPU-optimized. CPU inference will be slow.")

    _NEMO_BACKEND = "nemo-msdd"

    # Create temporary directory for NeMo outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Preprocess audio to mono 16kHz (NeMo requirement)
        preprocessed_audio = temp_dir / "audio_mono_16k.wav"
        actual_audio_path = _preprocess_audio_for_nemo(audio_path, preprocessed_audio)

        # Write input manifest (NeMo format)
        # All fields required by NeMo 2.6+ NeuralDiarizer
        manifest_path = temp_dir / "input_manifest.json"
        with manifest_path.open("w") as f:
            entry = {
                "audio_filepath": str(actual_audio_path.absolute()),
                "offset": 0,
                "duration": None,  # Full file
                "label": "infer",
                "text": "-",
                "num_speakers": config.oracle_num_speakers,  # None = auto-detect
                "rttm_filepath": None,  # No ground truth
                "uem_filepath": None,  # No scoring regions
            }
            f.write(json.dumps(entry) + "\n")

        # Get diarizer
        diarizer = _get_nemo_diarizer(config, temp_dir)

        # Update manifest path for this run
        diarizer._cfg.diarizer.manifest_filepath = str(manifest_path)

        # Set oracle speaker count if provided
        if config.oracle_num_speakers is not None:
            diarizer._cfg.diarizer.oracle_num_speakers = config.oracle_num_speakers

        LOGGER.info(f"Running NeMo MSDD diarization: {audio_path}")

        # Run diarization
        diarizer.diarize()

        # Parse RTTM output
        rttm_dir = temp_dir / "pred_rttms"
        rttm_files = list(rttm_dir.glob("*.rttm")) if rttm_dir.exists() else []

        if not rttm_files:
            LOGGER.warning(f"No RTTM files found in {rttm_dir}")
            segments = []
        else:
            rttm_path = rttm_files[0]  # Should be only one for single-file input
            segments = _parse_rttm_to_segments(rttm_path, config)

            # Copy RTTM to output location if requested
            if config.save_raw_rttm:
                raw_rttm_path = output_path.with_suffix(".rttm")
                import shutil

                shutil.copy(rttm_path, raw_rttm_path)

        # Parse speaker probabilities if available
        probs_dir = temp_dir / "speaker_outputs"
        if probs_dir.exists():
            probs_files = list(probs_dir.glob("*_speaker_probs.npy"))
            if probs_files:
                segments = _augment_segments_with_probs(segments, probs_files[0], config)

    # Post-processing
    if config.merge_gap_ms > 0:
        segments = _merge_segments(segments, config.merge_gap_ms / 1000.0)

    if config.min_speech_duration > 0:
        segments = [s for s in segments if (s.end - s.start) >= config.min_speech_duration]

    # Extract speaker embeddings
    standard_segments = [s.to_standard_segment() for s in segments]
    embeddings = extract_speaker_embeddings_nemo(audio_path, standard_segments, config)

    # Build result
    speaker_ids = set(s.speaker for s in segments)
    total_speech = sum(s.end - s.start for s in segments)
    overlap_duration = sum(s.end - s.start for s in segments if s.overlap)

    result = NeMoDiarizationResult(
        segments=segments,
        embeddings={spk: emb.tolist() for spk, emb in embeddings.items()},
        speaker_count=len(speaker_ids),
        total_speech_duration=total_speech,
        overlap_duration=overlap_duration,
        metadata={
            "backend": "nemo-msdd",
            "embedding_model": config.embedding_model,
            "msdd_model": config.msdd_model,
            "device": device,
            "overlap_threshold": config.overlap_threshold,
        },
    )

    # Save result
    _save_nemo_result(result, output_path)

    LOGGER.info(
        f"NeMo diarization complete: {len(segments)} segments, "
        f"{len(speaker_ids)} speakers, {overlap_duration:.1f}s overlap"
    )

    return result


def run_diarization(
    audio_path: Path,
    output_path: Path,
    config: Optional[DiarizationConfig] = None,
    overwrite: bool = False,
) -> List[DiarizationSegment]:
    """Run speaker diarization on audio file.

    Compatible interface with pyannote version.

    Args:
        audio_path: Path to input audio file
        output_path: Path for diarization manifest (JSONL)
        config: Diarization configuration (uses NeMo defaults)
        overwrite: Whether to overwrite existing results

    Returns:
        List of DiarizationSegment objects
    """
    # Convert DiarizationConfig to NeMoDiarizationConfig
    nemo_config = NeMoDiarizationConfig()
    if config:
        nemo_config.max_num_speakers = config.max_speakers
        nemo_config.min_num_speakers = config.min_speakers
        nemo_config.oracle_num_speakers = config.num_speakers
        nemo_config.merge_gap_ms = config.merge_gap_ms
        nemo_config.min_speech_duration = config.min_speech

    result = run_diarization_nemo(audio_path, output_path, nemo_config, overwrite)

    # Convert to standard segments for backward compatibility
    return [seg.to_standard_segment() for seg in result.segments]


# ============================================================================
# Memory Management
# ============================================================================


def unload_models() -> None:
    """Unload NeMo models to free GPU memory."""
    global _NEMO_DIARIZER, _NEMO_EMBEDDING_MODEL, _NEMO_BACKEND, _NEMO_DEVICE

    _NEMO_DIARIZER = None
    _NEMO_EMBEDDING_MODEL = None
    _NEMO_BACKEND = None
    _NEMO_DEVICE = None

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    LOGGER.info("NeMo diarization models unloaded")


def get_current_backend() -> Optional[str]:
    """Return currently loaded backend name."""
    return _NEMO_BACKEND
