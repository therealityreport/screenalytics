"""Pyannote speaker diarization integration.

Handles:
- Speaker diarization using Pyannote Audio
- Segment merging and confidence scoring
- Embedding extraction for voice clustering

Backend: pyannoteAI Precision-2 cloud API (requires PYANNOTEAI_API_KEY)

THERMAL SAFETY: CPU thread limits are set BEFORE importing torch/numpy to prevent
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
from .circuit_breaker import get_pyannote_breaker, CircuitBreakerError

LOGGER = logging.getLogger(__name__)

# Global model cache
_DIARIZATION_PIPELINE = None
_DIARIZATION_BACKEND = None  # Track which backend is loaded
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
    """Get Pyannote auth token from environment (for OSS model / HuggingFace)."""
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


@lru_cache(maxsize=1)
def _load_pyannoteai_api_key() -> Optional[str]:
    """Load pyannoteAI API key from environment or .env file."""
    # Check environment first
    api_key = os.environ.get("PYANNOTEAI_API_KEY")
    if api_key:
        return api_key

    # Search for .env/.env.local files
    start = Path(__file__).resolve()
    for parent in [start] + list(start.parents):
        for env_name in (".env.local", ".env"):
            env_path = parent / env_name
            if not env_path.exists():
                continue
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if "=" not in stripped:
                        continue
                    key, value = stripped.split("=", 1)
                    if key.strip() == "PYANNOTEAI_API_KEY":
                        return value.strip().strip('"').strip("'") or None
            except Exception:
                pass
    return None


def _get_pyannoteai_api_key() -> Optional[str]:
    """Get pyannoteAI API key for Precision-2 backend."""
    api_key = _load_pyannoteai_api_key()
    if api_key:
        # Cache in environment for downstream use
        os.environ.setdefault("PYANNOTEAI_API_KEY", api_key)
    return api_key


def _run_precision2_diarization(
    audio_path: Path,
    config: DiarizationConfig,
) -> List[DiarizationSegment]:
    """Run diarization using pyannoteAI Precision-2 cloud API.

    Implements the official PyannoteAI workflow:
    - POST /v1/diarize with S3 signed URL
    - exclusive=true for clean segment merging
    - Poll GET /v1/jobs/{jobId} every 5-8 seconds
    - Parse exclusiveDiarization output (preferred over regular)

    Args:
        audio_path: Path to audio file
        config: Diarization configuration

    Returns:
        List of DiarizationSegment objects

    Raises:
        RuntimeError: If API key is missing or API call fails
    """
    from .pyannote_api import PyannoteAPIClient, PyannoteAPIError

    api_key = _get_pyannoteai_api_key()
    if not api_key:
        raise RuntimeError("PYANNOTEAI_API_KEY not set for Precision-2 backend")

    LOGGER.info("Using pyannote Precision-2 diarization backend (official HTTP API)")

    try:
        client = PyannoteAPIClient(
            api_key=api_key,
            poll_interval_base=config.api_poll_interval_base,
            poll_interval_jitter=config.api_poll_interval_jitter,
        )
    except PyannoteAPIError as e:
        raise RuntimeError(str(e)) from e

    # Upload audio to S3 and get presigned URL
    LOGGER.info(f"Uploading audio to S3 for PyannoteAI: {audio_path}")
    try:
        media_url = client.upload_and_get_url(audio_path)
        LOGGER.info("Audio uploaded, presigned URL generated")
    except PyannoteAPIError as e:
        raise RuntimeError(f"Failed to upload audio: {e}") from e

    # Determine speaker range
    if config.num_speakers is not None:
        # Force exact speaker count by setting min=max
        min_speakers = config.num_speakers
        max_speakers = config.num_speakers
        LOGGER.info(f"Forcing {config.num_speakers} speakers")
    else:
        min_speakers = config.min_speakers if config.min_speakers >= 1 else None
        max_speakers = config.max_speakers if config.max_speakers >= 1 else None
        LOGGER.info(f"Speaker range: {min_speakers}-{max_speakers}")

    # Submit diarization job with official API parameters
    # Always use exclusive=True per requirements
    try:
        job_id = client.submit_diarization(
            media_url=media_url,
            model="precision-2",
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            exclusive=config.use_exclusive_diarization,
            webhook_url=config.webhook_url,
        )
    except PyannoteAPIError as e:
        raise RuntimeError(f"Failed to submit diarization job: {e}") from e

    # If webhook is configured, we don't poll - the webhook will handle completion
    if config.webhook_url:
        LOGGER.info(f"Webhook configured, not polling. Job will complete via webhook: {job_id}")
        # Return empty list - the webhook will handle saving the result
        return []

    # Poll for completion with 5-8 second intervals per official docs
    LOGGER.info(f"Waiting for diarization (timeout: {config.api_timeout_seconds}s)...")
    try:
        result = client.poll_job(job_id, max_wait=config.api_timeout_seconds)
    except PyannoteAPIError as e:
        raise RuntimeError(f"Diarization job failed: {e}") from e

    if result.error:
        raise RuntimeError(f"Precision-2 diarization failed: {result.error}")

    # Parse result into segments
    # Prefer exclusiveDiarization for clean segment merging (non-overlapping)
    segments = []
    diarization_data = result.exclusive_diarization or result.diarization

    if diarization_data:
        for entry in diarization_data:
            if isinstance(entry, dict):
                segments.append(DiarizationSegment(
                    start=entry.get("start", 0),
                    end=entry.get("end", 0),
                    speaker=entry.get("speaker", "SPEAKER_00"),
                    confidence=entry.get("confidence"),
                ))

    # Log which output type was used
    if result.exclusive_diarization:
        LOGGER.info(f"PyannoteAI diarization succeeded; using exclusive output ({len(segments)} segments)")
    else:
        LOGGER.info(f"PyannoteAI diarization succeeded; using regular output ({len(segments)} segments)")

    # Save raw response for debugging/backup (Pyannote deletes after 24h)
    if result.raw_response:
        _save_raw_diarization_response(audio_path, result.raw_response)

    return segments


def _save_raw_diarization_response(audio_path: Path, raw_response: dict) -> None:
    """Save raw API response for debugging and backup.

    Pyannote deletes job outputs after 24 hours, so we save immediately.
    """
    try:
        # Derive output path from audio path
        manifests_dir = audio_path.parent.parent / "manifests" / audio_path.parent.name
        if not manifests_dir.exists():
            # Try alternative location
            manifests_dir = audio_path.parent
        manifests_dir.mkdir(parents=True, exist_ok=True)

        raw_path = manifests_dir / "audio_diarization_pyannote_raw.json"
        with raw_path.open("w", encoding="utf-8") as f:
            json.dump(raw_response, f, indent=2)
        LOGGER.debug(f"Saved raw diarization response to: {raw_path}")
    except Exception as e:
        LOGGER.warning(f"Failed to save raw diarization response: {e}")


def _build_oss31_pipeline(config: DiarizationConfig):
    """Build local OSS pyannote/speaker-diarization-3.1 pipeline.

    Args:
        config: Diarization configuration

    Returns:
        Pipeline instance
    """
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

        model_name = "pyannote/speaker-diarization-3.1"
        LOGGER.info(f"Using pyannote OSS 3.1 diarization backend (local): {model_name}")
        pipeline = Pipeline.from_pretrained(
            model_name,
            use_auth_token=auth_token,
        )

        # Move to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)

        return pipeline

    except ImportError as e:
        raise ImportError(
            "pyannote.audio is required for diarization. "
            "Install with: pip install pyannote.audio"
        ) from e


def _check_precision2_available() -> bool:
    """Check if Precision-2 backend is available (API key set)."""
    return _get_pyannoteai_api_key() is not None


def _build_pyannote_pipeline(config: DiarizationConfig):
    """Build the appropriate pyannote pipeline based on backend config.

    Note: For precision-2, this returns None since we use the SDK directly.
    The caller should check the backend and use _run_precision2_diarization.

    Args:
        config: Diarization configuration with backend setting

    Returns:
        Tuple of (Pipeline instance or None, backend name)

    Raises:
        RuntimeError: If backend is not precision-2 or API key is missing
    """
    backend = getattr(config, "backend", None) or "precision-2"

    if backend != "precision-2":
        raise RuntimeError(f"Unsupported diarization backend: {backend}. Only 'precision-2' is supported.")

    # For precision-2, we use the SDK directly in run_diarization
    if _check_precision2_available():
        return None, "precision-2"

    raise RuntimeError(
        "PYANNOTEAI_API_KEY not set. Precision-2 diarization requires a valid API key. "
        "Get one from https://www.pyannote.ai/ and set PYANNOTEAI_API_KEY in your .env file."
    )


def _get_diarization_pipeline(config: DiarizationConfig):
    """Load or retrieve cached diarization pipeline.

    Uses backend setting from config. Only precision-2 is supported.
    Caches the pipeline for reuse within the same process.
    """
    global _DIARIZATION_PIPELINE, _DIARIZATION_BACKEND

    # Check if we need to rebuild (different backend requested)
    requested_backend = getattr(config, "backend", None) or "precision-2"
    if _DIARIZATION_PIPELINE is not None and _DIARIZATION_BACKEND == requested_backend:
        return _DIARIZATION_PIPELINE

    # Build new pipeline
    pipeline, actual_backend = _build_pyannote_pipeline(config)
    _DIARIZATION_PIPELINE = pipeline
    _DIARIZATION_BACKEND = actual_backend

    return pipeline


def get_current_backend() -> Optional[str]:
    """Return the currently loaded diarization backend name."""
    return _DIARIZATION_BACKEND


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


def _run_oss_diarization(
    audio_path: Path,
    config: DiarizationConfig,
) -> List[DiarizationSegment]:
    """Run diarization using local OSS pyannote 3.1 pipeline.

    Args:
        audio_path: Path to audio file
        config: Diarization configuration

    Returns:
        List of DiarizationSegment objects
    """
    pipeline = _get_diarization_pipeline(config)

    # Build diarization kwargs
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

    return segments


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
    global _DIARIZATION_BACKEND

    if output_path.exists() and not overwrite:
        LOGGER.info(f"Diarization results already exist: {output_path}")
        return _load_diarization_manifest(output_path)

    config = config or DiarizationConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine backend - only precision-2 is supported
    backend = getattr(config, "backend", None) or "precision-2"

    if backend != "precision-2":
        raise RuntimeError(f"Unsupported diarization backend: {backend}. Only 'precision-2' is supported.")

    if not _check_precision2_available():
        raise RuntimeError(
            "PYANNOTEAI_API_KEY not set. Precision-2 diarization requires a valid API key. "
            "Get one from https://www.pyannote.ai/ and set PYANNOTEAI_API_KEY in your .env file."
        )

    # Use pyannoteAI SDK for Precision-2
    _DIARIZATION_BACKEND = "precision-2"
    LOGGER.info(f"Running diarization (precision-2): {audio_path}")
    segments = _run_precision2_diarization(audio_path, config)

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
    global _DIARIZATION_PIPELINE, _DIARIZATION_BACKEND, _EMBEDDING_MODEL

    _DIARIZATION_PIPELINE = None
    _DIARIZATION_BACKEND = None
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
