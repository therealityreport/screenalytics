"""MDX-Extra stem separation for vocals/accompaniment.

Uses the Demucs/MDX-Extra model family for high-quality vocal separation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .models import SeparationConfig

LOGGER = logging.getLogger(__name__)

# Global model cache
_MDX_MODEL = None
_MDX_MODEL_NAME = None


def _get_mdx_model(model_name: str = "mdx_extra_q", device: str = "auto"):
    """Load or retrieve cached MDX model."""
    global _MDX_MODEL, _MDX_MODEL_NAME

    if _MDX_MODEL is not None and _MDX_MODEL_NAME == model_name:
        return _MDX_MODEL

    try:
        import demucs.separate
        from demucs.pretrained import get_model
        import torch

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        LOGGER.info(f"Loading MDX model '{model_name}' on device '{device}'")
        model = get_model(model_name)
        model.to(device)
        model.eval()

        _MDX_MODEL = model
        _MDX_MODEL_NAME = model_name

        return model
    except ImportError as e:
        raise ImportError(
            f"demucs is required for MDX separation. Install with: pip install demucs"
        ) from e


def separate_vocals(
    input_path: Path,
    output_dir: Path,
    config: Optional[SeparationConfig] = None,
    overwrite: bool = False,
) -> Tuple[Path, Path]:
    """Separate vocals from accompaniment using MDX-Extra.

    Args:
        input_path: Path to input audio file
        output_dir: Directory for output stems
        config: Separation configuration
        overwrite: Whether to overwrite existing files

    Returns:
        Tuple of (vocals_path, accompaniment_path)
    """
    config = config or SeparationConfig()

    vocals_path = output_dir / "episode_vocals.wav"
    accompaniment_path = output_dir / "episode_accompaniment.wav"

    if vocals_path.exists() and accompaniment_path.exists() and not overwrite:
        LOGGER.info(f"Separation outputs already exist: {vocals_path}")
        return vocals_path, accompaniment_path

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import demucs.separate
        import torch
        import torchaudio

        # Load the model
        model = _get_mdx_model(config.model_name, config.device)
        device = next(model.parameters()).device

        # Load audio
        LOGGER.info(f"Loading audio for separation: {input_path}")
        waveform, sample_rate = torchaudio.load(input_path)

        # Resample to model's expected sample rate if needed
        model_sr = model.samplerate
        if sample_rate != model_sr:
            LOGGER.info(f"Resampling from {sample_rate} to {model_sr}")
            resampler = torchaudio.transforms.Resample(sample_rate, model_sr)
            waveform = resampler(waveform)

        # Ensure stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        # Add batch dimension
        waveform = waveform.unsqueeze(0).to(device)

        # Process in chunks for memory efficiency
        chunk_samples = int(config.chunk_seconds * model_sr)
        overlap_samples = int(config.overlap_seconds * model_sr)

        total_samples = waveform.shape[-1]

        if total_samples <= chunk_samples:
            # Process whole file at once
            with torch.no_grad():
                sources = model(waveform)
        else:
            # Process in overlapping chunks
            LOGGER.info(f"Processing {total_samples / model_sr:.1f}s audio in chunks")

            # Get source names
            source_names = model.sources
            vocals_idx = source_names.index("vocals") if "vocals" in source_names else 0

            # Initialize output
            sources = torch.zeros(
                1, len(source_names), 2, total_samples,
                device=device, dtype=waveform.dtype
            )
            weights = torch.zeros(total_samples, device=device)

            # Process chunks
            hop = chunk_samples - overlap_samples
            for start in range(0, total_samples, hop):
                end = min(start + chunk_samples, total_samples)
                chunk = waveform[:, :, start:end]

                # Pad if needed
                if chunk.shape[-1] < chunk_samples:
                    pad_size = chunk_samples - chunk.shape[-1]
                    chunk = torch.nn.functional.pad(chunk, (0, pad_size))

                with torch.no_grad():
                    chunk_sources = model(chunk)

                # Trim padding
                actual_len = end - start
                chunk_sources = chunk_sources[:, :, :, :actual_len]

                # Blend with overlap
                blend_len = min(overlap_samples, actual_len, start)
                if blend_len > 0 and start > 0:
                    blend_weights = torch.linspace(0, 1, blend_len, device=device)
                    for i in range(len(source_names)):
                        sources[:, i, :, start:start+blend_len] = (
                            sources[:, i, :, start:start+blend_len] * (1 - blend_weights) +
                            chunk_sources[:, i, :, :blend_len] * blend_weights
                        )
                    sources[:, :, :, start+blend_len:end] = chunk_sources[:, :, :, blend_len:]
                else:
                    sources[:, :, :, start:end] = chunk_sources

                weights[start:end] = 1.0

                progress = (start + chunk_samples) / total_samples * 100
                LOGGER.debug(f"Separation progress: {min(progress, 100):.1f}%")

        # Extract vocals and accompaniment
        source_names = model.sources
        vocals_idx = source_names.index("vocals") if "vocals" in source_names else 0

        vocals = sources[0, vocals_idx].cpu()

        # Sum other sources for accompaniment
        accompaniment_indices = [i for i, name in enumerate(source_names) if name != "vocals"]
        if accompaniment_indices:
            accompaniment = sources[0, accompaniment_indices].sum(dim=0).cpu()
        else:
            # If no other sources, accompaniment is original minus vocals
            accompaniment = waveform[0].cpu() - vocals

        # Save stems
        LOGGER.info(f"Saving vocals: {vocals_path}")
        torchaudio.save(vocals_path, vocals, model_sr)

        LOGGER.info(f"Saving accompaniment: {accompaniment_path}")
        torchaudio.save(accompaniment_path, accompaniment, model_sr)

        return vocals_path, accompaniment_path

    except ImportError as e:
        LOGGER.error(f"Missing dependencies for MDX separation: {e}")
        raise ImportError(
            "demucs, torch, and torchaudio are required for MDX separation. "
            "Install with: pip install demucs torch torchaudio"
        ) from e


def separate_vocals_simple(
    input_path: Path,
    output_path: Path,
    model_name: str = "htdemucs",
    device: str = "auto",
    overwrite: bool = False,
) -> Path:
    """Simple vocal separation using demucs CLI-style interface.

    This is a simpler wrapper that just returns the vocals track.

    Args:
        input_path: Path to input audio
        output_path: Path for vocals output
        model_name: Demucs model name
        device: Device to use
        overwrite: Whether to overwrite existing

    Returns:
        Path to vocals audio file
    """
    if output_path.exists() and not overwrite:
        LOGGER.info(f"Vocals already exist: {output_path}")
        return output_path

    # Use the full separation function
    config = SeparationConfig(model_name=model_name, device=device)
    output_dir = output_path.parent

    vocals_path, _ = separate_vocals(input_path, output_dir, config, overwrite)

    # Rename if needed
    if vocals_path != output_path:
        import shutil
        shutil.move(vocals_path, output_path)

    return output_path


def unload_model():
    """Unload the MDX model to free memory."""
    global _MDX_MODEL, _MDX_MODEL_NAME
    if _MDX_MODEL is not None:
        del _MDX_MODEL
        _MDX_MODEL = None
        _MDX_MODEL_NAME = None

        # Force garbage collection
        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        LOGGER.info("MDX model unloaded")
