"""MDX-Extra stem separation for vocals/accompaniment.

Uses the Demucs/MDX-Extra model family for high-quality vocal separation.

THERMAL SAFETY: CPU thread limits are set BEFORE importing torch to prevent
laptop overheating during vocal separation. Override with SCREENALYTICS_SEPARATION_THREADS.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

# CRITICAL: Set CPU thread limits BEFORE importing torch to prevent overheating.
# Default to 2 threads for thermal safety on laptops.
_SEPARATION_THREADS = os.environ.get("SCREENALYTICS_SEPARATION_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", _SEPARATION_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _SEPARATION_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _SEPARATION_THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _SEPARATION_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _SEPARATION_THREADS)

from .models import SeparationConfig

LOGGER = logging.getLogger(__name__)

# Global model cache
_MDX_MODEL = None
_MDX_MODEL_NAME = None


def _get_mdx_model(model_name: str = "mdx_extra_q", device: str = "auto"):
    """Load or retrieve cached MDX model."""
    global _MDX_MODEL, _MDX_MODEL_NAME

    if _MDX_MODEL is not None and _MDX_MODEL_NAME == model_name:
        LOGGER.info(f"Using cached MDX model '{model_name}'")
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
        # Guardrail: demucs on mps/metal frequently stalls; prefer CPU
        if str(device).lower() in {"mps", "metal", "coreml", "apple"}:
            LOGGER.warning("Demucs on %s is unstable; falling back to CPU for separation", device)
            device = "cpu"

        LOGGER.info(f"Loading MDX model '{model_name}' on device '{device}'...")

        # Suppress demucs's confusing "Call apply_model on this" message
        # by temporarily redirecting stdout during model loading
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            model = get_model(model_name)

        model.to(device)
        model.eval()

        _MDX_MODEL = model
        _MDX_MODEL_NAME = model_name

        LOGGER.info(f"MDX model '{model_name}' loaded successfully on {device}")
        return model
    except ImportError as e:
        raise ImportError(
            f"demucs is required for MDX separation. Install with: pip install demucs"
        ) from e
    except Exception as e:
        LOGGER.error(f"Failed to load MDX model '{model_name}': {e}")
        raise RuntimeError(f"Failed to load MDX model: {e}") from e


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

    def _run_separation(device_choice: str, chunk_seconds: float, overlap_seconds: float) -> Tuple[Path, Path]:
        import demucs.separate
        from demucs.apply import apply_model
        import torch
        import torchaudio

        # Load the model
        model = _get_mdx_model(config.model_name, device_choice)
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

        # Add batch dimension: (channels, length) -> (1, channels, length)
        waveform = waveform.unsqueeze(0).to(device)

        # demucs BagOfModels require apply_model; use config chunk/overlap to set segment windows.
        segment_seconds = max(float(chunk_seconds), 1.0)
        overlap_ratio = 0.0
        if chunk_seconds > 0:
            overlap_ratio = min(max(overlap_seconds / chunk_seconds, 0.0), 0.95)

        # Calculate expected segments for progress estimation
        audio_duration_seconds = waveform.shape[-1] / model_sr
        estimated_segments = int(audio_duration_seconds / segment_seconds) + 1
        LOGGER.info(
            f"Running vocal separation: {audio_duration_seconds:.1f}s audio, "
            f"~{estimated_segments} segments (segment={segment_seconds}s, overlap={overlap_ratio:.2f}, device={device})..."
        )

        # Custom progress tracking class that logs outside the redirect context
        import sys
        import io
        import contextlib

        class ProgressTracker:
            def __init__(self, total_segments: int, logger):
                self.total = total_segments
                self.current = 0
                self.logger = logger
                self.last_log_pct = -10  # Log every 10%
                # Keep reference to real stderr for progress output
                self._real_stderr = sys.__stderr__

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def update(self, n=1):
                self.current += n
                pct = (self.current / max(self.total, 1)) * 100
                # Log every 10% or on completion
                if pct >= self.last_log_pct + 10 or self.current >= self.total:
                    msg = f"Separation progress: {pct:.0f}% ({self.current}/{self.total} segments)"
                    # Write to real stderr to bypass redirect, and also log properly
                    if self._real_stderr:
                        self._real_stderr.write(f"[INFO] {msg}\n")
                        self._real_stderr.flush()
                    self.last_log_pct = int(pct // 10) * 10

            def __iter__(self):
                return self

            def __len__(self):
                return self.total

        progress_tracker = ProgressTracker(estimated_segments, LOGGER)

        # Suppress demucs internal print statements (like "Call apply_model on this")
        # which are confusing and pollute the progress output
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sources = apply_model(
                model,
                waveform,
                overlap=overlap_ratio,
                segment=segment_seconds,
                device=device,
                split=True,
                shifts=1,
                progress=progress_tracker,  # Use our progress tracker
            )

        LOGGER.info("Vocal separation complete, extracting stems...")

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

    try:
        # Primary attempt
        return _run_separation(config.device, config.chunk_seconds, config.overlap_seconds)

    except ImportError as e:
        LOGGER.error(f"Missing dependencies for MDX separation: {e}")
        raise ImportError(
            "demucs, torch, and torchaudio are required for MDX separation. "
            "Install with: pip install demucs torch torchaudio"
        ) from e
    except Exception as e:
        LOGGER.error(f"Vocal separation failed: {e}")
        # Provide clearer error messages for common issues
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "cuda" in error_msg:
            raise RuntimeError(
                f"Vocal separation failed due to memory/GPU error: {e}. "
                "Try using device='cpu' or reducing chunk_seconds in config."
            ) from e
        if "call apply_model on this" in error_msg:
            raise RuntimeError(
                "Demucs failed to run the MDX model (internal apply_model error). "
                "Try reinstalling demucs/torch, clearing cached weights, or running with device='cpu' "
                "and low_power profile."
            ) from e
        elif "no such file" in error_msg or "not found" in error_msg:
            raise RuntimeError(
                f"Vocal separation failed - input file not found: {input_path}"
            ) from e
        # Fallback: retry once on CPU with safer chunk settings
        if config.device != "cpu":
            LOGGER.warning("Retrying vocal separation on CPU with safer chunk/overlap after failure...")
            try:
                # Use float division and ensure non-zero values
                fallback_chunk = max(8.0, config.chunk_seconds / 2) if config.chunk_seconds > 0 else 8.0
                fallback_overlap = max(1.0, config.overlap_seconds / 2) if config.overlap_seconds > 0 else 1.0
                return _run_separation("cpu", fallback_chunk, fallback_overlap)
            except Exception as inner_exc:
                raise RuntimeError(f"Vocal separation failed after CPU fallback: {inner_exc}") from inner_exc
        raise RuntimeError(f"Vocal separation failed: {e}") from e


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
        if output_path.exists():
            output_path.unlink()  # Remove existing file before move
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
