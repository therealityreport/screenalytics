"""Pyannote speaker diarization integration.

Handles:
- Speaker diarization using Pyannote Audio
- Segment merging and confidence scoring
- Embedding extraction for voice clustering

THERMAL SAFETY: CPU thread limits are set BEFORE importing torch to prevent
laptop overheating during diarization. Override with SCREENALYTICS_DIARIZATION_THREADS.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
from functools import lru_cache

# CRITICAL: Set CPU thread limits BEFORE importing torch/numpy to prevent overheating.
# Default to 2 threads for thermal safety on laptops.
_DIARIZATION_THREADS = os.environ.get("SCREENALYTICS_DIARIZATION_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", _DIARIZATION_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _DIARIZATION_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _DIARIZATION_THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _DIARIZATION_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _DIARIZATION_THREADS)

import numpy as np

from .models import DiarizationConfig, DiarizationSegment

LOGGER = logging.getLogger(__name__)

# Global model cache
_DIARIZATION_PIPELINE = None
_EMBEDDING_MODEL = None


@lru_cache(maxsize=1)
def _load_env_token() -> Optional[str]:
    """Load a Pyannote/HF token from .env/.env.local if not in the environment."""
    candidate_keys = {"PYANNOTE_AUTH_TOKEN", "HF_TOKEN"}

    def _parse_env_file(path: Path) -> Optional[str]:
        if not path.exists():
            return None
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                key = key.strip()
                if key in candidate_keys:
                    return value.strip().strip('"').strip("'")
        except Exception:
            return None
        return None

    # Search upward for .env/.env.local starting from this file's parents
    start = Path(__file__).resolve()
    for parent in [start] + list(start.parents):
        for env_name in (".env.local", ".env"):
            token = _parse_env_file(parent / env_name)
            if token:
                return token
    return None


def _get_auth_token() -> Optional[str]:
    """Get Pyannote auth token from environment."""
    env_token = os.environ.get("PYANNOTE_AUTH_TOKEN") or os.environ.get("HF_TOKEN")
    if env_token:
        return env_token

    # Fallback to .env/.env.local if set but not exported
    file_token = _load_env_token()
    if file_token:
        # Set in os.environ so downstream libraries pick it up too
        os.environ.setdefault("PYANNOTE_AUTH_TOKEN", file_token)
        os.environ.setdefault("HF_TOKEN", file_token)
        LOGGER.info("Loaded PYANNOTE_AUTH_TOKEN from .env/.env.local")
        return file_token

    return None


def _get_diarization_pipeline(config: DiarizationConfig):
    """Load or retrieve cached diarization pipeline."""
    global _DIARIZATION_PIPELINE

    if _DIARIZATION_PIPELINE is not None:
        return _DIARIZATION_PIPELINE

    try:
        from pyannote.audio import Pipeline
        import torch

        auth_token = _get_auth_token()
        if not auth_token:
            LOGGER.warning(
                "No PYANNOTE_AUTH_TOKEN found (also checked .env/.env.local). "
                "Some models require authentication. Get a token from https://huggingface.co/settings/tokens "
                "and set PYANNOTE_AUTH_TOKEN in your environment or .env file."
            )

        LOGGER.info(f"Loading diarization pipeline: {config.model_name}")
        pipeline = Pipeline.from_pretrained(
            config.model_name,
            use_auth_token=auth_token,
        )

        # Move to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)

        _DIARIZATION_PIPELINE = pipeline
        return pipeline

    except ImportError as e:
        raise ImportError(
            "pyannote.audio is required for diarization. "
            "Install with: pip install pyannote.audio"
        ) from e


def _get_embedding_model(model_name: str = "pyannote/embedding"):
    """Load or retrieve cached embedding model."""
    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    try:
        from pyannote.audio import Model
        import torch

        auth_token = _get_auth_token()

        LOGGER.info(f"Loading embedding model: {model_name}")
        model = Model.from_pretrained(model_name, use_auth_token=auth_token)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        _EMBEDDING_MODEL = model
        return model

    except ImportError as e:
        raise ImportError(
            "pyannote.audio is required for speaker embeddings. "
            "Install with: pip install pyannote.audio"
        ) from e


def run_diarization(
    audio_path: Path,
    output_path: Path,
    config: Optional[DiarizationConfig] = None,
    overwrite: bool = False,
) -> List[DiarizationSegment]:
    """Run speaker diarization on audio file.

    Args:
        audio_path: Path to input audio file
        output_path: Path for diarization manifest (JSONL)
        config: Diarization configuration
        overwrite: Whether to overwrite existing results

    Returns:
        List of DiarizationSegment objects
    """
    if output_path.exists() and not overwrite:
        LOGGER.info(f"Diarization results already exist: {output_path}")
        return _load_diarization_manifest(output_path)

    config = config or DiarizationConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = _get_diarization_pipeline(config)

    LOGGER.info(f"Running diarization: {audio_path}")

    # Build diarization kwargs
    # num_speakers forces exact count and overrides min/max
    diarization_kwargs = {}
    if config.num_speakers is not None:
        LOGGER.info(f"Forcing {config.num_speakers} speakers (num_speakers override)")
        diarization_kwargs["num_speakers"] = config.num_speakers
    else:
        diarization_kwargs["min_speakers"] = config.min_speakers
        diarization_kwargs["max_speakers"] = config.max_speakers
        LOGGER.info(f"Speaker range: {config.min_speakers}-{config.max_speakers}")

    # Run diarization
    diarization = pipeline(audio_path, **diarization_kwargs)

    # Convert to segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment = DiarizationSegment(
            start=turn.start,
            end=turn.end,
            speaker=speaker,
            confidence=None,  # Pyannote doesn't provide per-segment confidence
        )
        segments.append(segment)

    # Merge short gaps between same-speaker segments
    if config.merge_gap_ms > 0:
        segments = _merge_segments(segments, config.merge_gap_ms / 1000.0)

    # Filter out very short segments
    if config.min_speech > 0:
        segments = [s for s in segments if (s.end - s.start) >= config.min_speech]

    # Calculate overlap ratios
    segments = _calculate_overlap_ratios(segments)

    # Save manifest
    _save_diarization_manifest(segments, output_path)

    LOGGER.info(f"Diarization complete: {len(segments)} segments from {len(set(s.speaker for s in segments))} speakers")

    return segments


def _merge_segments(
    segments: List[DiarizationSegment],
    max_gap: float,
) -> List[DiarizationSegment]:
    """Merge segments from the same speaker that are close together."""
    if not segments:
        return segments

    # Sort by start time
    segments = sorted(segments, key=lambda s: s.start)

    merged = [segments[0]]
    for segment in segments[1:]:
        last = merged[-1]

        # Check if same speaker and gap is small enough
        if segment.speaker == last.speaker and (segment.start - last.end) <= max_gap:
            # Merge by extending the last segment
            merged[-1] = DiarizationSegment(
                start=last.start,
                end=segment.end,
                speaker=last.speaker,
                confidence=last.confidence,
            )
        else:
            merged.append(segment)

    return merged


def _calculate_overlap_ratios(segments: List[DiarizationSegment]) -> List[DiarizationSegment]:
    """Calculate overlap ratio for each segment."""
    if not segments:
        return segments

    # Sort by start time
    segments = sorted(segments, key=lambda s: s.start)

    # Optimization: pre-compute overlap using interval sweep (O(n log n) vs O(n²))
    # Only check segments that could potentially overlap (within max segment duration)
    max_duration = max((s.end - s.start for s in segments), default=0)

    result = []
    for i, segment in enumerate(segments):
        overlap_duration = 0.0
        segment_duration = segment.end - segment.start

        # Only check nearby segments that could overlap
        # (segments are sorted by start time)
        for j in range(max(0, i - 50), min(len(segments), i + 50)):
            if i == j:
                continue
            other = segments[j]

            # Early exit: if other segment starts after current ends, no more overlaps possible
            if other.start >= segment.end:
                break
            # Skip if other segment ends before current starts
            if other.end <= segment.start:
                continue

            # Calculate overlap
            overlap_start = max(segment.start, other.start)
            overlap_end = min(segment.end, other.end)
            if overlap_end > overlap_start:
                overlap_duration += overlap_end - overlap_start

        overlap_ratio = overlap_duration / segment_duration if segment_duration > 0 else 0.0

        result.append(DiarizationSegment(
            start=segment.start,
            end=segment.end,
            speaker=segment.speaker,
            confidence=segment.confidence,
            overlap_ratio=overlap_ratio,
        ))

    return result


def _save_diarization_manifest(segments: List[DiarizationSegment], output_path: Path):
    """Save diarization segments to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for segment in segments:
            f.write(segment.model_dump_json() + "\n")


def _load_diarization_manifest(manifest_path: Path) -> List[DiarizationSegment]:
    """Load diarization segments from JSONL file."""
    segments = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                segments.append(DiarizationSegment(**data))
    return segments


def extract_speaker_embeddings(
    audio_path: Path,
    segments: List[DiarizationSegment],
    embedding_model: str = "pyannote/embedding",
) -> List[Tuple[DiarizationSegment, np.ndarray]]:
    """Extract speaker embeddings for each diarization segment.

    Args:
        audio_path: Path to audio file
        segments: List of diarization segments
        embedding_model: Name of embedding model to use

    Returns:
        List of (segment, embedding) tuples
    """
    try:
        from pyannote.audio import Inference
        import torch
        import torchaudio
    except ImportError as e:
        raise ImportError(
            "pyannote.audio and torchaudio are required for speaker embeddings."
        ) from e

    model = _get_embedding_model(embedding_model)

    # Create inference object
    inference = Inference(model, window="whole")

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    duration = waveform.shape[1] / sample_rate

    results = []
    for segment in segments:
        # Skip very short segments
        if (segment.end - segment.start) < 0.5:
            continue

        # Ensure bounds are valid
        start = max(0, segment.start)
        end = min(duration, segment.end)

        if end <= start:
            continue

        try:
            # Extract segment audio
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment_audio = waveform[:, start_sample:end_sample]

            # Get embedding
            with torch.no_grad():
                embedding = inference({"waveform": segment_audio, "sample_rate": sample_rate})

            results.append((segment, np.array(embedding)))
        except Exception as e:
            LOGGER.warning(f"Failed to extract embedding for segment {start:.2f}-{end:.2f}: {e}")

    LOGGER.info(f"Extracted {len(results)} embeddings from {len(segments)} segments")
    return results


def unload_models():
    """Unload diarization and embedding models to free memory."""
    global _DIARIZATION_PIPELINE, _EMBEDDING_MODEL

    _DIARIZATION_PIPELINE = None
    _EMBEDDING_MODEL = None

    import gc
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    LOGGER.info("Diarization models unloaded")
