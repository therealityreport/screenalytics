"""Resemble Enhance audio enhancement API integration.

Handles:
- API communication with Resemble Enhance
- Audio chunking for long files
- Retry logic and rate limiting
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .models import EnhanceConfig

LOGGER = logging.getLogger(__name__)


def _get_api_key() -> str:
    """Get Resemble API key from environment."""
    key = os.environ.get("RESEMBLE_API_KEY")
    if not key:
        raise ValueError(
            "RESEMBLE_API_KEY environment variable is required for audio enhancement. "
            "Get your API key from https://resemble.ai"
        )
    return key


def enhance_audio_resemble(
    input_path: Path,
    output_path: Path,
    config: Optional[EnhanceConfig] = None,
    overwrite: bool = False,
) -> Path:
    """Enhance audio using Resemble Enhance API.

    Args:
        input_path: Path to input audio file
        output_path: Path for enhanced output
        config: Enhancement configuration
        overwrite: Whether to overwrite existing file

    Returns:
        Path to enhanced audio file
    """
    if output_path.exists() and not overwrite:
        LOGGER.info(f"Enhanced audio already exists: {output_path}")
        return output_path

    config = config or EnhanceConfig()
    api_key = _get_api_key()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import requests
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "requests and soundfile are required for Resemble enhancement. "
            "Install with: pip install requests soundfile"
        ) from e

    # Load audio
    LOGGER.info(f"Loading audio for enhancement: {input_path}")
    data, sample_rate = sf.read(input_path)

    # Ensure mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Calculate chunk size in samples
    chunk_samples = int(config.batch_seconds * sample_rate)
    total_samples = len(data)

    if total_samples <= chunk_samples:
        # Process whole file at once
        enhanced = _enhance_chunk(data, sample_rate, api_key, config)
    else:
        # Process in chunks
        LOGGER.info(f"Processing {total_samples / sample_rate:.1f}s audio in {chunk_samples / sample_rate}s chunks")

        enhanced_chunks = []
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = data[start:end]

            LOGGER.debug(f"Enhancing chunk {start / sample_rate:.1f}s - {end / sample_rate:.1f}s")
            enhanced_chunk = _enhance_chunk(chunk, sample_rate, api_key, config)
            enhanced_chunks.append(enhanced_chunk)

            progress = end / total_samples * 100
            LOGGER.info(f"Enhancement progress: {progress:.1f}%")

        # Concatenate chunks
        enhanced = np.concatenate(enhanced_chunks)

    # Save enhanced audio
    LOGGER.info(f"Saving enhanced audio: {output_path}")
    sf.write(output_path, enhanced, sample_rate)

    return output_path


def _enhance_chunk(
    audio_data: np.ndarray,
    sample_rate: int,
    api_key: str,
    config: EnhanceConfig,
) -> np.ndarray:
    """Enhance a single audio chunk via API.

    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate in Hz
        api_key: Resemble API key
        config: Enhancement configuration

    Returns:
        Enhanced audio data
    """
    import requests
    import soundfile as sf

    # Convert to WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    buffer.seek(0)
    audio_bytes = buffer.read()

    # API endpoint
    api_url = "https://api.resemble.ai/v2/enhance"

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    files = {
        "audio": ("audio.wav", audio_bytes, "audio/wav"),
    }

    data = {
        "mode": config.mode,
    }

    # Retry logic
    last_error = None
    for attempt in range(config.max_retries):
        try:
            response = requests.post(
                api_url,
                headers=headers,
                files=files,
                data=data,
                timeout=120,
            )

            if response.status_code == 429:
                # Rate limited
                retry_after = int(response.headers.get("Retry-After", config.retry_delay_seconds))
                LOGGER.warning(f"Rate limited, waiting {retry_after}s before retry")
                time.sleep(retry_after)
                continue

            response.raise_for_status()

            # Parse response
            enhanced_bytes = response.content

            # Load enhanced audio
            enhanced_buffer = io.BytesIO(enhanced_bytes)
            enhanced_data, enhanced_sr = sf.read(enhanced_buffer)

            # Resample if sample rate changed
            if enhanced_sr != sample_rate:
                try:
                    import librosa
                    enhanced_data = librosa.resample(
                        enhanced_data, orig_sr=enhanced_sr, target_sr=sample_rate
                    )
                except ImportError:
                    LOGGER.warning(
                        f"Sample rate changed from {sample_rate} to {enhanced_sr}, "
                        "but librosa not available for resampling"
                    )

            return enhanced_data

        except requests.RequestException as e:
            last_error = e
            LOGGER.warning(f"Enhancement attempt {attempt + 1} failed: {e}")
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay_seconds)

    raise RuntimeError(f"Enhancement failed after {config.max_retries} attempts: {last_error}")


def enhance_audio_local(
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    """Local audio enhancement fallback using simple processing.

    This is a fallback when the Resemble API is not available.
    Uses basic noise reduction and normalization.

    Args:
        input_path: Path to input audio
        output_path: Path for enhanced output
        overwrite: Whether to overwrite existing

    Returns:
        Path to enhanced audio
    """
    if output_path.exists() and not overwrite:
        LOGGER.info(f"Enhanced audio already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
        import noisereduce as nr
    except ImportError:
        LOGGER.warning("noisereduce not available, copying input to output")
        import shutil
        shutil.copy(input_path, output_path)
        return output_path

    LOGGER.info(f"Enhancing audio locally: {input_path}")
    data, sample_rate = sf.read(input_path)

    # Ensure mono for processing
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Apply noise reduction
    reduced = nr.reduce_noise(y=data, sr=sample_rate)

    # Normalize
    peak = np.max(np.abs(reduced))
    if peak > 0:
        reduced = reduced / peak * 0.9  # Leave 10% headroom

    # Save
    sf.write(output_path, reduced, sample_rate)
    LOGGER.info(f"Local enhancement complete: {output_path}")

    return output_path


def check_api_available() -> bool:
    """Check if Resemble API is available."""
    try:
        api_key = _get_api_key()
        return bool(api_key)
    except ValueError:
        return False
