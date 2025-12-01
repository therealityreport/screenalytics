"""Audio I/O utilities for extraction and normalization.

Handles:
- Extracting audio from video files
- Sample rate and bit depth normalization
- Basic audio statistics (RMS, peak, LUFS)
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class AudioStats:
    """Statistics for an audio file."""
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: int
    rms_db: float
    peak_db: float
    lufs: Optional[float] = None


def _get_ffprobe_info(audio_path: Path) -> dict:
    """Get audio file info using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    import json
    return json.loads(result.stdout)


def _compute_audio_stats(audio_path: Path) -> AudioStats:
    """Compute audio statistics for a file."""
    try:
        import soundfile as sf
        data, sample_rate = sf.read(audio_path)

        if len(data.shape) == 1:
            channels = 1
        else:
            channels = data.shape[1]
            # Convert to mono for stats
            data = np.mean(data, axis=1)

        # Get bit depth from file info
        info = sf.info(audio_path)
        subtype = info.subtype
        bit_depth = 16  # default
        if "24" in subtype:
            bit_depth = 24
        elif "32" in subtype:
            bit_depth = 32
        elif "FLOAT" in subtype:
            bit_depth = 32

        duration = len(data) / sample_rate

        # Compute RMS
        rms = np.sqrt(np.mean(data**2))
        rms_db = 20 * np.log10(rms + 1e-10)

        # Compute peak
        peak = np.max(np.abs(data))
        peak_db = 20 * np.log10(peak + 1e-10)

        # LUFS requires pyloudnorm (optional)
        lufs = None
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(sample_rate)
            lufs = meter.integrated_loudness(data)
        except ImportError:
            LOGGER.debug("pyloudnorm not available, skipping LUFS calculation")
        except Exception as e:
            LOGGER.debug(f"LUFS calculation failed: {e}")

        return AudioStats(
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            rms_db=rms_db,
            peak_db=peak_db,
            lufs=lufs,
        )
    except ImportError:
        # Fallback using ffprobe
        LOGGER.debug("soundfile not available, using ffprobe for stats")
        info = _get_ffprobe_info(audio_path)
        stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})
        format_info = info.get("format", {})

        return AudioStats(
            duration_seconds=float(format_info.get("duration", 0)),
            sample_rate=int(stream.get("sample_rate", 48000)),
            channels=int(stream.get("channels", 2)),
            bit_depth=int(stream.get("bits_per_sample", 16)),
            rms_db=-20.0,  # placeholder
            peak_db=-1.0,  # placeholder
            lufs=None,
        )


def extract_audio_from_video(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 48000,
    bit_depth: int = 24,
    channels: int = 1,
    overwrite: bool = False,
) -> Tuple[Path, AudioStats]:
    """Extract audio from video file and normalize.

    Args:
        video_path: Path to source video file
        output_path: Path for output audio file
        sample_rate: Target sample rate in Hz
        bit_depth: Target bit depth (16, 24, or 32)
        channels: Number of output channels (1=mono, 2=stereo)
        overwrite: Whether to overwrite existing file

    Returns:
        Tuple of (output_path, AudioStats)
    """
    if output_path.exists() and not overwrite:
        LOGGER.info(f"Audio file already exists: {output_path}")
        stats = _compute_audio_stats(output_path)
        return output_path, stats

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Map bit depth to ffmpeg format
    if bit_depth == 16:
        acodec = "pcm_s16le"
    elif bit_depth == 24:
        acodec = "pcm_s24le"
    else:
        acodec = "pcm_s32le"

    # Build ffmpeg command - use -y (overwrite) always since we check existence above
    # -n can cause hangs if file exists but we already return early for that case
    cmd = [
        "ffmpeg",
        "-y",  # Always overwrite - we handle skip logic above
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", acodec,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(output_path),
    ]

    LOGGER.info(f"Extracting audio: {video_path} -> {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        # Truncate long error messages
        stderr = result.stderr.strip()
        if len(stderr) > 500:
            stderr = stderr[:500] + "... (truncated)"
        raise RuntimeError(f"ffmpeg audio extraction failed: {stderr}")

    stats = _compute_audio_stats(output_path)
    LOGGER.info(f"Audio extracted: {stats.duration_seconds:.2f}s, {sample_rate}Hz, {bit_depth}bit")

    return output_path, stats


def normalize_audio(
    input_path: Path,
    output_path: Path,
    target_peak_dbfs: float = -1.0,
    target_sample_rate: Optional[int] = None,
    overwrite: bool = False,
) -> Tuple[Path, AudioStats]:
    """Normalize audio file to target peak level.

    Args:
        input_path: Path to input audio file
        output_path: Path for output audio file
        target_peak_dbfs: Target peak level in dBFS
        target_sample_rate: Optional target sample rate
        overwrite: Whether to overwrite existing file

    Returns:
        Tuple of (output_path, AudioStats)
    """
    if output_path.exists() and not overwrite:
        LOGGER.info(f"Normalized audio already exists: {output_path}")
        stats = _compute_audio_stats(output_path)
        return output_path, stats

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build loudnorm filter
    filter_parts = [f"loudnorm=I=-16:TP={target_peak_dbfs}:LRA=11"]

    # Build command - use -y since we check existence above
    cmd = [
        "ffmpeg",
        "-y",  # Always overwrite - we handle skip logic above
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(input_path),
        "-af", ",".join(filter_parts),
    ]

    if target_sample_rate:
        cmd.extend(["-ar", str(target_sample_rate)])

    cmd.append(str(output_path))

    LOGGER.info(f"Normalizing audio: {input_path} -> {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if len(stderr) > 500:
            stderr = stderr[:500] + "... (truncated)"
        raise RuntimeError(f"ffmpeg normalization failed: {stderr}")

    stats = _compute_audio_stats(output_path)
    return output_path, stats


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of audio file in seconds."""
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration
    except ImportError:
        info = _get_ffprobe_info(audio_path)
        return float(info.get("format", {}).get("duration", 0))


def load_audio(
    audio_path: Path,
    target_sr: Optional[int] = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load audio file into numpy array.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (resample if different)
        mono: Whether to convert to mono

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        import soundfile as sf
        data, sr = sf.read(audio_path)

        # Convert to mono if needed
        if mono and len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Resample if needed
        if target_sr and sr != target_sr:
            try:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except ImportError:
                LOGGER.warning("librosa not available, skipping resample")

        return data, sr
    except ImportError:
        raise ImportError("soundfile is required for audio loading. Install with: pip install soundfile")


def save_audio(
    audio_data: np.ndarray,
    output_path: Path,
    sample_rate: int,
    bit_depth: int = 24,
) -> Path:
    """Save numpy array as audio file.

    Args:
        audio_data: Audio data as numpy array
        output_path: Output file path
        sample_rate: Sample rate in Hz
        bit_depth: Bit depth (16, 24, or 32)

    Returns:
        Path to saved file
    """
    try:
        import soundfile as sf

        # Map bit depth to subtype
        subtype_map = {
            16: "PCM_16",
            24: "PCM_24",
            32: "PCM_32",
        }
        subtype = subtype_map.get(bit_depth, "PCM_24")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio_data, sample_rate, subtype=subtype)

        return output_path
    except ImportError:
        raise ImportError("soundfile is required for audio saving. Install with: pip install soundfile")


def compute_snr(
    signal_path: Path,
    noise_estimate_seconds: float = 0.5,
) -> float:
    """Estimate signal-to-noise ratio from audio file.

    Uses the quietest portion of the audio as a noise estimate.

    Args:
        signal_path: Path to audio file
        noise_estimate_seconds: Duration to use for noise estimation

    Returns:
        Estimated SNR in dB
    """
    data, sr = load_audio(signal_path, mono=True)

    # Window size for noise estimation
    window_size = int(noise_estimate_seconds * sr)
    if window_size >= len(data):
        window_size = len(data) // 10

    # Find quietest window for noise estimate
    min_rms = float("inf")
    for i in range(0, len(data) - window_size, window_size):
        window = data[i:i + window_size]
        rms = np.sqrt(np.mean(window**2))
        if rms < min_rms and rms > 1e-10:  # Avoid silence
            min_rms = rms

    # Signal RMS
    signal_rms = np.sqrt(np.mean(data**2))

    # SNR
    if min_rms > 1e-10:
        snr = 20 * np.log10(signal_rms / min_rms)
    else:
        snr = 60.0  # Very clean signal

    return float(snr)
