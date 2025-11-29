"""Pyannote speaker diarization integration.

Handles:
- Speaker diarization using Pyannote Audio
- Segment merging and confidence scoring
- Embedding extraction for voice clustering
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .models import DiarizationConfig, DiarizationSegment

LOGGER = logging.getLogger(__name__)

# Global model cache
_DIARIZATION_PIPELINE = None
_EMBEDDING_MODEL = None


def _get_auth_token() -> Optional[str]:
    """Get Pyannote auth token from environment."""
    return os.environ.get("PYANNOTE_AUTH_TOKEN") or os.environ.get("HF_TOKEN")


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
                "No PYANNOTE_AUTH_TOKEN found. Some models require authentication. "
                "Get token from https://huggingface.co/settings/tokens"
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

    # Run diarization
    diarization = pipeline(
        audio_path,
        min_speakers=config.min_speakers,
        max_speakers=config.max_speakers,
    )

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

    result = []
    for i, segment in enumerate(segments):
        overlap_duration = 0.0
        segment_duration = segment.end - segment.start

        for j, other in enumerate(segments):
            if i == j:
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
