"""Audio export utilities.

Handles:
- Final audio normalization and export
- Sample rate and bit depth conversion
- Level matching
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .models import ExportConfig

LOGGER = logging.getLogger(__name__)


def export_final_audio(
    input_path: Path,
    output_path: Path,
    config: Optional[ExportConfig] = None,
    overwrite: bool = False,
) -> Path:
    """Export final audio with normalization and level matching.

    Args:
        input_path: Path to input audio file
        output_path: Path for final output
        config: Export configuration
        overwrite: Whether to overwrite existing file

    Returns:
        Path to exported audio file
    """
    if output_path.exists() and not overwrite:
        LOGGER.info(f"Final audio already exists: {output_path}")
        return output_path

    config = config or ExportConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required for audio export")

    LOGGER.info(f"Exporting final audio: {input_path} -> {output_path}")

    # Load audio
    data, sr = sf.read(input_path)

    # Convert to mono if needed
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample if needed
    if sr != config.sample_rate:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=config.sample_rate)
            sr = config.sample_rate
        except ImportError:
            LOGGER.warning(f"librosa not available, keeping original sample rate {sr}")

    # Normalize to target peak level
    current_peak = np.max(np.abs(data))
    if current_peak > 0:
        target_linear = 10 ** (config.peak_dbfs / 20)
        data = data / current_peak * target_linear

    # Map bit depth to subtype
    subtype_map = {
        16: "PCM_16",
        24: "PCM_24",
        32: "PCM_32",
    }
    subtype = subtype_map.get(config.bit_depth, "PCM_24")

    # Save
    sf.write(output_path, data, sr, subtype=subtype)

    LOGGER.info(f"Exported {len(data) / sr:.2f}s audio at {sr}Hz, {config.bit_depth}bit")

    return output_path


def match_levels(
    reference_path: Path,
    target_path: Path,
    output_path: Path,
    config: Optional[ExportConfig] = None,
    overwrite: bool = False,
) -> Path:
    """Match audio levels between reference and target.

    Args:
        reference_path: Path to reference audio
        target_path: Path to target audio to adjust
        output_path: Path for level-matched output
        config: Export configuration
        overwrite: Whether to overwrite existing file

    Returns:
        Path to level-matched audio
    """
    if output_path.exists() and not overwrite:
        LOGGER.info(f"Level-matched audio already exists: {output_path}")
        return output_path

    config = config or ExportConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required for level matching")

    # Load both files
    ref_data, ref_sr = sf.read(reference_path)
    target_data, target_sr = sf.read(target_path)

    # Convert to mono
    if len(ref_data.shape) > 1:
        ref_data = np.mean(ref_data, axis=1)
    if len(target_data.shape) > 1:
        target_data = np.mean(target_data, axis=1)

    # Calculate RMS levels
    ref_rms = np.sqrt(np.mean(ref_data**2))
    target_rms = np.sqrt(np.mean(target_data**2))

    # Apply gain to match levels
    if target_rms > 0:
        gain = ref_rms / target_rms
        target_data = target_data * gain

        # Limit to prevent clipping
        peak = np.max(np.abs(target_data))
        target_linear = 10 ** (config.peak_dbfs / 20)
        if peak > target_linear:
            target_data = target_data / peak * target_linear

    # Map bit depth to subtype
    subtype_map = {
        16: "PCM_16",
        24: "PCM_24",
        32: "PCM_32",
    }
    subtype = subtype_map.get(config.bit_depth, "PCM_24")

    # Save
    sf.write(output_path, target_data, target_sr, subtype=subtype)

    LOGGER.info(f"Level-matched audio saved: {output_path}")

    return output_path


def convert_format(
    input_path: Path,
    output_path: Path,
    sample_rate: Optional[int] = None,
    bit_depth: Optional[int] = None,
    channels: Optional[int] = None,
    overwrite: bool = False,
) -> Path:
    """Convert audio to different format/parameters.

    Args:
        input_path: Path to input audio
        output_path: Path for converted output
        sample_rate: Target sample rate
        bit_depth: Target bit depth
        channels: Target channel count (1=mono, 2=stereo)
        overwrite: Whether to overwrite existing file

    Returns:
        Path to converted audio
    """
    if output_path.exists() and not overwrite:
        LOGGER.info(f"Converted audio already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required for format conversion")

    # Load audio
    data, sr = sf.read(input_path)

    # Handle channels
    if channels is not None:
        if channels == 1 and len(data.shape) > 1:
            data = np.mean(data, axis=1)
        elif channels == 2 and len(data.shape) == 1:
            data = np.stack([data, data], axis=1)

    # Resample if needed
    if sample_rate is not None and sr != sample_rate:
        try:
            import librosa
            if len(data.shape) > 1:
                # Process each channel
                channels_data = []
                for ch in range(data.shape[1]):
                    resampled = librosa.resample(data[:, ch], orig_sr=sr, target_sr=sample_rate)
                    channels_data.append(resampled)
                data = np.stack(channels_data, axis=1)
            else:
                data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate
        except ImportError:
            LOGGER.warning("librosa not available, keeping original sample rate")

    # Map bit depth to subtype
    if bit_depth:
        subtype_map = {
            16: "PCM_16",
            24: "PCM_24",
            32: "PCM_32",
        }
        subtype = subtype_map.get(bit_depth, "PCM_24")
    else:
        subtype = None

    # Save
    sf.write(output_path, data, sr, subtype=subtype)

    LOGGER.info(f"Converted audio saved: {output_path}")

    return output_path
