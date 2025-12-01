"""Celery tasks for the audio pipeline.

These tasks handle episode audio processing:
- Audio extraction
- Stem separation
- Enhancement
- Diarization
- Voice clustering
- Transcription
- QC and export

Usage:
    from apps.api.jobs_audio import episode_audio_pipeline_async
    result = episode_audio_pipeline_async.delay(ep_id="rhoslc-s06e02")
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Load environment variables from .env file for Celery workers
try:
    from dotenv import load_dotenv
    for env_path in [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",  # project root
    ]:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass  # dotenv not installed, rely on system environment

from celery import Task, chain, chord, group

from apps.api.celery_app import celery_app
from apps.api.tasks import _acquire_lock, _release_lock, _force_release_lock, check_active_job

LOGGER = logging.getLogger(__name__)

# Stage metadata for progress reporting
# Note: diarize and transcribe run in parallel, so they share a progress window
STAGE_INFO = {
    "ingest": {"order": 1, "name": "Extract Audio", "progress_start": 0.0},
    "separate": {"order": 2, "name": "Separate Vocals", "progress_start": 0.05},
    "enhance": {"order": 3, "name": "Enhance Audio", "progress_start": 0.15},
    "diarize": {"order": 4, "name": "Diarization", "progress_start": 0.25},
    "transcribe": {"order": 5, "name": "Transcription", "progress_start": 0.35},  # Fixed: was 0.25 (duplicate)
    "voices": {"order": 6, "name": "Voice Clustering", "progress_start": 0.50},
    "align": {"order": 7, "name": "Alignment", "progress_start": 0.65},
    "export": {"order": 8, "name": "Export", "progress_start": 0.75},
    "qc": {"order": 9, "name": "Quality Control", "progress_start": 0.85},
}


def _write_progress(ep_id: str, stage: str, message: str, stage_progress: float = 0.0, status: str = "running") -> None:
    """Write progress to JSON file for UI polling."""
    import json
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
    progress_file = data_root / "manifests" / ep_id / "audio_progress.json"
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    info = STAGE_INFO.get(stage, {"order": 0, "name": stage, "progress_start": 0.0})
    # Calculate overall progress based on stage position
    stage_weight = 1.0 / 9.0  # 9 stages
    overall = info["progress_start"] + (stage_progress * stage_weight)

    payload = {
        "progress": min(overall, 0.99),
        "step": stage,
        "step_name": info["name"],
        "step_order": info["order"],
        "total_steps": 9,
        "message": message,
        "step_progress": stage_progress,
        "timestamp": time.time(),
        "status": status,
    }
    progress_file.write_text(json.dumps(payload), encoding="utf-8")


def _write_progress_complete(ep_id: str, message: str = "Pipeline completed successfully") -> None:
    """Mark progress as complete."""
    import json
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
    progress_file = data_root / "manifests" / ep_id / "audio_progress.json"
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "progress": 1.0,
        "overall_progress": 1.0,
        "step": "complete",
        "step_name": "Complete",
        "step_order": 10,
        "total_steps": 9,
        "message": message,
        "step_progress": 1.0,
        "timestamp": time.time(),
        "status": "completed",
    }
    progress_file.write_text(json.dumps(payload), encoding="utf-8")
    LOGGER.info(f"[{ep_id}] Audio pipeline marked complete")


def _write_progress_error(ep_id: str, stage: str, error: str) -> None:
    """Mark progress as failed."""
    import json
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
    progress_file = data_root / "manifests" / ep_id / "audio_progress.json"
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    info = STAGE_INFO.get(stage, {"order": 0, "name": stage, "progress_start": 0.0})

    payload = {
        "progress": info["progress_start"],
        "step": stage,
        "step_name": info["name"],
        "step_order": info["order"],
        "total_steps": 9,
        "message": f"Error: {error}",
        "step_progress": 0.0,
        "timestamp": time.time(),
        "status": "error",
        "error": error,
    }
    progress_file.write_text(json.dumps(payload), encoding="utf-8")
    LOGGER.error(f"[{ep_id}] Audio pipeline failed at {stage}: {error}")


# Find project root
def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parents[2]

PROJECT_ROOT = _find_project_root()

# Audio queue names
AUDIO_QUEUES = {
    "ingest": "SCREENALYTICS_AUDIO_INGEST",
    "separate": "SCREENALYTICS_AUDIO_SEPARATE",
    "enhance": "SCREENALYTICS_AUDIO_ENHANCE",
    "diarize": "SCREENALYTICS_AUDIO_DIARIZE",
    "voices": "SCREENALYTICS_AUDIO_VOICES",
    "transcribe": "SCREENALYTICS_AUDIO_TRANSCRIBE",
    "align": "SCREENALYTICS_AUDIO_ALIGN",
    "qc": "SCREENALYTICS_AUDIO_QC",
    "export": "SCREENALYTICS_AUDIO_EXPORT",
}


class AudioPipelineTask(Task):
    """Base class for audio pipeline tasks."""

    abstract = True
    _pipeline_module = None

    @property
    def pipeline(self):
        """Lazy-load audio pipeline module."""
        if self._pipeline_module is None:
            sys.path.insert(0, str(PROJECT_ROOT))
            from py_screenalytics.audio import episode_audio_pipeline
            self._pipeline_module = episode_audio_pipeline
        return self._pipeline_module


# =============================================================================
# Individual Stage Tasks
# =============================================================================


@celery_app.task(bind=True, base=AudioPipelineTask, name="audio.ingest")
def episode_audio_ingest_task(
    self,
    ep_id: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Extract audio from video file."""
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting audio ingest for {ep_id}")
    _write_progress(ep_id, "ingest", "Extracting audio from video...", 0.0)

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from py_screenalytics.audio.io import extract_audio_from_video
        from py_screenalytics.audio.episode_audio_pipeline import _get_audio_paths, _load_config

        config = _load_config()
        paths = _get_audio_paths(ep_id)

        if not paths["video"].exists():
            _write_progress(ep_id, "ingest", f"Video not found: {paths['video']}", 1.0)
            return {
                "status": "error",
                "ep_id": ep_id,
                "stage": "ingest",
                "error": f"Video not found: {paths['video']}",
            }

        original_path, stats = extract_audio_from_video(
            paths["video"],
            paths["original"],
            sample_rate=config.export.sample_rate,
            bit_depth=config.export.bit_depth,
            overwrite=overwrite,
        )

        _write_progress(ep_id, "ingest", f"Audio extracted: {stats.duration_seconds:.1f}s", 1.0)
        return {
            "status": "success",
            "ep_id": ep_id,
            "stage": "ingest",
            "audio_path": str(original_path),
            "duration_s": stats.duration_seconds,
            "sample_rate": stats.sample_rate,
        }

    except Exception as e:
        LOGGER.exception(f"[{job_id}] Audio ingest failed: {e}")
        _write_progress_error(ep_id, "ingest", str(e))
        # Release pipeline lock so a new job can be started
        _force_release_lock(ep_id, "audio_pipeline")
        return {
            "status": "error",
            "ep_id": ep_id,
            "stage": "ingest",
            "error": str(e),
        }


def _check_required_file(path: Path, stage: str, file_desc: str) -> None:
    """Check if a required input file exists, raise if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"[{stage}] Required input '{file_desc}' not found at {path}. "
            f"Ensure previous pipeline stages completed successfully."
        )


@celery_app.task(bind=True, base=AudioPipelineTask, name="audio.separate")
def episode_audio_separate_task(
    self,
    ep_id: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Separate vocals from accompaniment using MDX-Extra."""
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting audio separation for {ep_id}")
    _write_progress(ep_id, "separate", "Separating vocals from accompaniment...", 0.0)

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from py_screenalytics.audio.separation_mdx import separate_vocals
        from py_screenalytics.audio.episode_audio_pipeline import _get_audio_paths, _load_config

        config = _load_config()
        paths = _get_audio_paths(ep_id)

        # Validate required input exists
        _check_required_file(paths["original"], "separate", "original audio")

        vocals_path, accompaniment_path = separate_vocals(
            paths["original"],
            paths["audio_dir"],
            config.separation,
            overwrite=overwrite,
        )

        _write_progress(ep_id, "separate", "Vocal separation complete", 1.0)
        return {
            "status": "success",
            "ep_id": ep_id,
            "stage": "separate",
            "vocals_path": str(vocals_path),
            "accompaniment_path": str(accompaniment_path),
        }

    except Exception as e:
        LOGGER.exception(f"[{job_id}] Audio separation failed: {e}")
        _write_progress_error(ep_id, "separate", str(e))
        _force_release_lock(ep_id, "audio_pipeline")
        return {
            "status": "error",
            "ep_id": ep_id,
            "stage": "separate",
            "error": str(e),
        }


@celery_app.task(bind=True, base=AudioPipelineTask, name="audio.enhance")
def episode_audio_enhance_task(
    self,
    ep_id: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Enhance vocals using Resemble API."""
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting audio enhancement for {ep_id}")
    _write_progress(ep_id, "enhance", "Enhancing audio quality...", 0.0)

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from py_screenalytics.audio.enhance_resemble import (
            enhance_audio_resemble,
            enhance_audio_local,
            check_api_available,
        )
        from py_screenalytics.audio.episode_audio_pipeline import _get_audio_paths, _load_config

        config = _load_config()
        paths = _get_audio_paths(ep_id)

        # Validate required input exists
        _check_required_file(paths["vocals"], "enhance", "separated vocals")

        if check_api_available():
            _write_progress(ep_id, "enhance", "Using Resemble API...", 0.3)
            enhanced_path = enhance_audio_resemble(
                paths["vocals"],
                paths["vocals_enhanced"],
                config.enhance,
                overwrite=overwrite,
            )
            method = "resemble_api"
        else:
            LOGGER.warning("Resemble API not available, using local enhancement")
            _write_progress(ep_id, "enhance", "Using local enhancement...", 0.3)
            enhanced_path = enhance_audio_local(
                paths["vocals"],
                paths["vocals_enhanced"],
                overwrite=overwrite,
            )
            method = "local"

        _write_progress(ep_id, "enhance", f"Enhancement complete ({method})", 1.0)
        return {
            "status": "success",
            "ep_id": ep_id,
            "stage": "enhance",
            "enhanced_path": str(enhanced_path),
            "method": method,
        }

    except Exception as e:
        LOGGER.exception(f"[{job_id}] Audio enhancement failed: {e}")
        _write_progress_error(ep_id, "enhance", str(e))
        _force_release_lock(ep_id, "audio_pipeline")
        return {
            "status": "error",
            "ep_id": ep_id,
            "stage": "enhance",
            "error": str(e),
        }


@celery_app.task(bind=True, base=AudioPipelineTask, name="audio.diarize")
def episode_audio_diarize_task(
    self,
    ep_id: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Run speaker diarization using Pyannote."""
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting diarization for {ep_id}")
    _write_progress(ep_id, "diarize", "Running speaker diarization...", 0.0)

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from py_screenalytics.audio.diarization_pyannote import run_diarization
        from py_screenalytics.audio.episode_audio_pipeline import _get_audio_paths, _load_config

        config = _load_config()
        paths = _get_audio_paths(ep_id)

        # Use enhanced vocals if available, fall back to vocals, then original
        audio_path = paths["vocals_enhanced"]
        if not audio_path.exists():
            audio_path = paths["vocals"]
        if not audio_path.exists():
            audio_path = paths["original"]

        # Validate at least one audio file exists
        if not audio_path.exists():
            raise FileNotFoundError(
                f"[diarize] No audio file found. Checked: vocals_enhanced, vocals, original. "
                f"Ensure previous pipeline stages (separate, enhance) completed successfully."
            )

        segments = run_diarization(
            audio_path,
            paths["diarization"],
            config.diarization,
            overwrite=overwrite,
        )

        speakers = set(s.speaker for s in segments)

        _write_progress(ep_id, "diarize", f"Found {len(speakers)} speakers, {len(segments)} segments", 1.0)
        return {
            "status": "success",
            "ep_id": ep_id,
            "stage": "diarize",
            "manifest_path": str(paths["diarization"]),
            "segment_count": len(segments),
            "speaker_count": len(speakers),
        }

    except Exception as e:
        LOGGER.exception(f"[{job_id}] Diarization failed: {e}")
        _write_progress_error(ep_id, "diarize", str(e))
        _force_release_lock(ep_id, "audio_pipeline")
        return {
            "status": "error",
            "ep_id": ep_id,
            "stage": "diarize",
            "error": str(e),
        }


@celery_app.task(bind=True, base=AudioPipelineTask, name="audio.voices")
def episode_audio_voices_task(
    self,
    previous_results: Any = None,  # Chord callback receives header results as first arg
    ep_id: str = "",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Cluster voices and map to voice bank.

    Note: When used as a chord callback, this function receives the results
    from the header tasks (diarize & transcribe) as the first argument.
    """
    # Handle both direct calls and chord callback calls
    # When called directly, previous_results might be the ep_id string
    if isinstance(previous_results, str) and not ep_id:
        ep_id = previous_results
        previous_results = None

    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting voice clustering for {ep_id}")
    _write_progress(ep_id, "voices", "Clustering voices...", 0.0)

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from py_screenalytics.audio.voice_clusters import cluster_episode_voices
        from py_screenalytics.audio.voice_bank import match_voice_clusters_to_bank
        from py_screenalytics.audio.diarization_pyannote import _load_diarization_manifest
        from py_screenalytics.audio.episode_audio_pipeline import (
            _get_audio_paths,
            _get_show_id,
            _load_config,
        )

        config = _load_config()
        paths = _get_audio_paths(ep_id)
        show_id = _get_show_id(ep_id)

        # Validate diarization completed
        _check_required_file(paths["diarization"], "voices", "diarization manifest")

        # Load diarization segments (prefer combined pyannote+GPT-4o if available)
        diar_path = paths.get("diarization_combined", paths["diarization"])
        if diar_path.exists():
            diarization_segments = _load_diarization_manifest(diar_path)
        else:
            diarization_segments = _load_diarization_manifest(paths["diarization"])

        # Use enhanced vocals if available
        audio_path = paths["vocals_enhanced"]
        if not audio_path.exists():
            audio_path = paths["vocals"]

        # Validate audio file exists
        if not audio_path.exists():
            raise FileNotFoundError(
                f"[voices] No audio file found. Checked: vocals_enhanced, vocals. "
                f"Ensure previous pipeline stages completed successfully."
            )

        # Cluster voices
        _write_progress(ep_id, "voices", "Extracting voice embeddings...", 0.3)
        voice_clusters = cluster_episode_voices(
            audio_path,
            diarization_segments,
            paths["voice_clusters"],
            config.voice_clustering,
            overwrite=overwrite,
        )

        # Map to voice bank
        _write_progress(ep_id, "voices", "Matching to voice bank...", 0.7)
        voice_mapping = match_voice_clusters_to_bank(
            show_id,
            voice_clusters,
            paths["voice_mapping"],
            config.voice_bank,
            config.voice_clustering.similarity_threshold,
            overwrite=overwrite,
        )

        labeled = sum(1 for m in voice_mapping if m.similarity is not None)
        unlabeled = len(voice_mapping) - labeled

        _write_progress(ep_id, "voices", f"{len(voice_clusters)} clusters, {labeled} labeled", 1.0)
        return {
            "status": "success",
            "ep_id": ep_id,
            "stage": "voices",
            "clusters_path": str(paths["voice_clusters"]),
            "mapping_path": str(paths["voice_mapping"]),
            "cluster_count": len(voice_clusters),
            "labeled_voices": labeled,
            "unlabeled_voices": unlabeled,
        }

    except Exception as e:
        LOGGER.exception(f"[{job_id}] Voice clustering failed: {e}")
        _write_progress_error(ep_id, "voices", str(e))
        _force_release_lock(ep_id, "audio_pipeline")
        return {
            "status": "error",
            "ep_id": ep_id,
            "stage": "voices",
            "error": str(e),
        }


@celery_app.task(bind=True, base=AudioPipelineTask, name="audio.transcribe")
def episode_audio_transcribe_task(
    self,
    ep_id: str,
    asr_provider: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Transcribe audio using ASR."""
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting transcription for {ep_id}")
    _write_progress(ep_id, "transcribe", "Starting transcription...", 0.0)

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from py_screenalytics.audio.episode_audio_pipeline import _get_audio_paths, _load_config

        config = _load_config()
        paths = _get_audio_paths(ep_id)

        # Use provider from args or config
        provider = asr_provider or config.asr.provider
        # Normalize legacy name
        if provider == "gemini":
            provider = "gemini_3"

        _write_progress(ep_id, "transcribe", f"Transcribing with {provider}...", 0.2)
        if provider == "gemini_3":
            from py_screenalytics.audio.asr_gemini import transcribe_audio
        else:
            from py_screenalytics.audio.asr_openai import transcribe_audio

        # Use enhanced vocals if available
        audio_path = paths["vocals_enhanced"]
        if not audio_path.exists():
            audio_path = paths["vocals"]

        # Validate audio file exists
        if not audio_path.exists():
            raise FileNotFoundError(
                f"[transcribe] No audio file found. Checked: vocals_enhanced, vocals. "
                f"Ensure previous pipeline stages (separate, enhance) completed successfully."
            )

        segments = transcribe_audio(
            audio_path,
            paths["asr_raw"],
            config.asr,
            overwrite=overwrite,
        )

        word_count = sum(len(s.words or []) for s in segments)

        _write_progress(ep_id, "transcribe", f"{len(segments)} segments, {word_count} words", 1.0)
        return {
            "status": "success",
            "ep_id": ep_id,
            "stage": "transcribe",
            "manifest_path": str(paths["asr_raw"]),
            "segment_count": len(segments),
            "word_count": word_count,
            "provider": provider,
        }

    except Exception as e:
        LOGGER.exception(f"[{job_id}] Transcription failed: {e}")
        _write_progress_error(ep_id, "transcribe", str(e))
        _force_release_lock(ep_id, "audio_pipeline")
        return {
            "status": "error",
            "ep_id": ep_id,
            "stage": "transcribe",
            "error": str(e),
        }


@celery_app.task(bind=True, base=AudioPipelineTask, name="audio.align")
def episode_audio_align_task(
    self,
    ep_id: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Fuse diarization, ASR, and voice mapping into final transcript."""
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting transcript alignment for {ep_id}")
    _write_progress(ep_id, "align", "Aligning transcript...", 0.0)

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from py_screenalytics.audio.fuse_diarization_asr import fuse_transcript
        from py_screenalytics.audio.diarization_pyannote import _load_diarization_manifest
        from py_screenalytics.audio.asr_openai import _load_asr_manifest
        from py_screenalytics.audio.voice_clusters import _load_voice_clusters
        from py_screenalytics.audio.voice_bank import _load_voice_mapping
        from py_screenalytics.audio.episode_audio_pipeline import _get_audio_paths, _load_config
        from py_screenalytics.audio.speaker_groups import load_speaker_groups_manifest

        config = _load_config()
        paths = _get_audio_paths(ep_id)

        # Load all inputs
        _write_progress(ep_id, "align", "Loading diarization and ASR data...", 0.2)
        diarization_segments = _load_diarization_manifest(paths["diarization"])
        asr_segments = _load_asr_manifest(paths["asr_raw"])
        voice_clusters = _load_voice_clusters(paths["voice_clusters"])
        voice_mapping = _load_voice_mapping(paths["voice_mapping"])
        speaker_groups_manifest = None
        if paths.get("speaker_groups") and paths["speaker_groups"].exists():
            speaker_groups_manifest = load_speaker_groups_manifest(paths["speaker_groups"])

        # Fuse into transcript
        _write_progress(ep_id, "align", "Fusing transcript...", 0.5)
        transcript_rows = fuse_transcript(
            diarization_segments,
            asr_segments,
            voice_clusters,
            voice_mapping,
            speaker_groups_manifest,
            paths["transcript_jsonl"],
            paths["transcript_vtt"],
            config.export.vtt_include_speaker_notes,
            overwrite=overwrite,
            diarization_source="pyannote",
        )

        _write_progress(ep_id, "align", f"Generated {len(transcript_rows)} transcript rows", 1.0)
        return {
            "status": "success",
            "ep_id": ep_id,
            "stage": "align",
            "transcript_jsonl": str(paths["transcript_jsonl"]),
            "transcript_vtt": str(paths["transcript_vtt"]),
            "row_count": len(transcript_rows),
        }

    except Exception as e:
        LOGGER.exception(f"[{job_id}] Transcript alignment failed: {e}")
        _write_progress_error(ep_id, "align", str(e))
        _force_release_lock(ep_id, "audio_pipeline")
        return {
            "status": "error",
            "ep_id": ep_id,
            "stage": "align",
            "error": str(e),
        }


@celery_app.task(bind=True, base=AudioPipelineTask, name="audio.qc")
def episode_audio_qc_task(
    self,
    ep_id: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Run quality control checks."""
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting QC for {ep_id}")
    _write_progress(ep_id, "qc", "Running quality checks...", 0.0)

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from py_screenalytics.audio.qc import run_qc_checks, get_qc_summary
        from py_screenalytics.audio.io import compute_snr, get_audio_duration
        from py_screenalytics.audio.diarization_pyannote import _load_diarization_manifest
        from py_screenalytics.audio.asr_openai import _load_asr_manifest
        from py_screenalytics.audio.voice_clusters import _load_voice_clusters
        from py_screenalytics.audio.voice_bank import _load_voice_mapping
        from py_screenalytics.audio.fuse_diarization_asr import _load_transcript_jsonl
        from py_screenalytics.audio.episode_audio_pipeline import _get_audio_paths, _load_config

        config = _load_config()
        paths = _get_audio_paths(ep_id)

        # Validate required pipeline artifacts exist
        required_artifacts = [
            (paths["diarization"], "diarization manifest"),
            (paths["asr_raw"], "ASR transcript"),
            (paths["voice_clusters"], "voice clusters"),
            (paths["voice_mapping"], "voice mapping"),
            (paths["transcript_jsonl"], "final transcript"),
        ]
        missing = [desc for path, desc in required_artifacts if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"[qc] Missing required artifacts: {', '.join(missing)}. "
                f"Ensure all previous pipeline stages completed successfully."
            )

        # Load data
        _write_progress(ep_id, "qc", "Loading pipeline artifacts...", 0.2)
        diarization_segments = _load_diarization_manifest(paths["diarization"])
        asr_segments = _load_asr_manifest(paths["asr_raw"])
        voice_clusters = _load_voice_clusters(paths["voice_clusters"])
        voice_mapping = _load_voice_mapping(paths["voice_mapping"])
        transcript_rows = _load_transcript_jsonl(paths["transcript_jsonl"])

        # Get durations and SNR
        _write_progress(ep_id, "qc", "Computing audio metrics...", 0.5)
        duration_original = get_audio_duration(paths["original"])
        duration_final = get_audio_duration(paths["final_voice_only"]) if paths["final_voice_only"].exists() else duration_original

        snr_db = None
        try:
            snr_db = compute_snr(paths["vocals_enhanced"])
        except Exception:
            pass

        # Run QC
        _write_progress(ep_id, "qc", "Validating results...", 0.7)
        qc_report = run_qc_checks(
            ep_id,
            config.qc,
            duration_original,
            duration_final,
            snr_db,
            diarization_segments,
            asr_segments,
            voice_clusters,
            voice_mapping,
            transcript_rows,
            paths["qc"],
            overwrite=overwrite,
        )

        summary = get_qc_summary(qc_report)

        # Mark pipeline as complete (QC is the final stage)
        _write_progress_complete(ep_id, f"QC {qc_report.status.value}: {summary}")

        # Release the pipeline lock (QC is the final stage in the chain)
        _force_release_lock(ep_id, "audio_pipeline")

        return {
            "status": "success",
            "ep_id": ep_id,
            "stage": "qc",
            "qc_path": str(paths["qc"]),
            "qc_status": qc_report.status.value,
            "summary": summary,
        }

    except Exception as e:
        LOGGER.exception(f"[{job_id}] QC failed: {e}")
        _write_progress_error(ep_id, "qc", str(e))
        _force_release_lock(ep_id, "audio_pipeline")
        return {
            "status": "error",
            "ep_id": ep_id,
            "stage": "qc",
            "error": str(e),
        }


@celery_app.task(bind=True, base=AudioPipelineTask, name="audio.export")
def episode_audio_export_task(
    self,
    ep_id: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Export final audio file."""
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting audio export for {ep_id}")
    _write_progress(ep_id, "export", "Exporting final audio...", 0.0)

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from py_screenalytics.audio.export import export_final_audio
        from py_screenalytics.audio.io import get_audio_duration
        from py_screenalytics.audio.episode_audio_pipeline import _get_audio_paths, _load_config

        config = _load_config()
        paths = _get_audio_paths(ep_id)

        # Use enhanced vocals as input
        input_path = paths["vocals_enhanced"]
        if not input_path.exists():
            input_path = paths["vocals"]

        _write_progress(ep_id, "export", "Writing final audio file...", 0.3)
        final_path = export_final_audio(
            input_path,
            paths["final_voice_only"],
            config.export,
            overwrite=overwrite,
        )

        duration = get_audio_duration(final_path)

        _write_progress(ep_id, "export", f"Exported {duration:.1f}s audio", 1.0)
        return {
            "status": "success",
            "ep_id": ep_id,
            "stage": "export",
            "final_path": str(final_path),
            "duration_s": duration,
        }

    except Exception as e:
        LOGGER.exception(f"[{job_id}] Audio export failed: {e}")
        _write_progress_error(ep_id, "export", str(e))
        _force_release_lock(ep_id, "audio_pipeline")
        return {
            "status": "error",
            "ep_id": ep_id,
            "stage": "export",
            "error": str(e),
        }


# =============================================================================
# Full Pipeline Task
# =============================================================================


@celery_app.task(bind=True, base=AudioPipelineTask, name="audio.pipeline")
def episode_audio_pipeline_task(
    self,
    ep_id: str,
    overwrite: bool = False,
    asr_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the complete audio pipeline for an episode.

    This task runs all stages sequentially in a single worker.
    For distributed execution, use episode_audio_pipeline_async.

    Args:
        ep_id: Episode identifier
        overwrite: Whether to overwrite existing artifacts
        asr_provider: Override ASR provider (openai_whisper or gemini_3)

    Returns:
        Result dict with status and all stage results
    """
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting full audio pipeline for {ep_id}")

    if not _acquire_lock(ep_id, "audio_pipeline", job_id):
        return {
            "status": "error",
            "error": "Another audio pipeline job is already running for this episode",
            "ep_id": ep_id,
        }

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from py_screenalytics.audio.episode_audio_pipeline import (
            run_episode_audio_pipeline,
            sync_audio_artifacts_to_s3,
        )

        def progress_callback(step: str, progress: float, message: str):
            self.update_state(
                state="PROGRESS",
                meta={
                    "step": step,
                    "progress": progress,
                    "message": message,
                    "ep_id": ep_id,
                },
            )

        result = run_episode_audio_pipeline(
            ep_id,
            overwrite=overwrite,
            asr_provider=asr_provider,
            progress_callback=progress_callback,
        )

        # Sync artifacts to S3 if pipeline succeeded
        if result.status == "succeeded":
            progress_callback("s3_sync", 0.5, "Syncing artifacts to S3...")
            s3_status = sync_audio_artifacts_to_s3(ep_id, result)
            result.metrics["s3_sync"] = s3_status
            progress_callback("s3_sync", 1.0, f"S3 sync: {s3_status.get('status', 'unknown')}")

        return result.to_dict()

    except Exception as e:
        LOGGER.exception(f"[{job_id}] Audio pipeline failed: {e}")
        return {
            "status": "error",
            "ep_id": ep_id,
            "error": str(e),
        }
    finally:
        _release_lock(ep_id, "audio_pipeline", job_id)


def episode_audio_pipeline_async(
    ep_id: str,
    overwrite: bool = False,
    asr_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Queue the audio pipeline as a Celery chain.

    The pipeline stages are:
    INGEST -> SEPARATE -> ENHANCE -> (DIARIZE & TRANSCRIBE in parallel) ->
    VOICES -> ALIGN -> QC -> EXPORT

    Args:
        ep_id: Episode identifier
        overwrite: Whether to overwrite existing artifacts
        asr_provider: Override ASR provider

    Returns:
        Dict with job_id and status
    """
    import uuid

    # Generate job_id first so we can acquire lock with it
    job_id = str(uuid.uuid4())

    # Check for existing job AND try to acquire lock atomically
    existing = check_active_job(ep_id, "audio_pipeline")
    if existing:
        return {
            "status": "error",
            "error": "Another audio pipeline job is already running",
            "existing_job_id": existing,
            "ep_id": ep_id,
        }

    # Try to acquire lock - this prevents race conditions where two requests
    # both pass check_active_job but then both try to start jobs
    if not _acquire_lock(ep_id, "audio_pipeline", job_id, ttl=7200):  # 2 hour TTL for long videos
        # Another job grabbed the lock between our check and now
        existing_again = check_active_job(ep_id, "audio_pipeline")
        return {
            "status": "error",
            "error": "Another audio pipeline job is already running (race condition)",
            "existing_job_id": existing_again,
            "ep_id": ep_id,
        }

    # Clear any stale progress from previous runs
    _write_progress(ep_id, "ingest", "Queuing pipeline...", 0.0, status="queued")

    # Build the chain
    # Sequential stages: ingest -> separate -> enhance
    # Parallel: diarize & transcribe
    # Sequential: voices -> align -> qc -> export

    pipeline_chain = chain(
        episode_audio_ingest_task.si(ep_id, overwrite),
        episode_audio_separate_task.si(ep_id, overwrite),
        episode_audio_enhance_task.si(ep_id, overwrite),
        chord(
            [
                episode_audio_diarize_task.si(ep_id, overwrite),
                episode_audio_transcribe_task.si(ep_id, asr_provider, overwrite),
            ],
            episode_audio_voices_task.s(ep_id, overwrite),  # callback already handles previous_results
        ),
        episode_audio_align_task.si(ep_id, overwrite),
        episode_audio_export_task.si(ep_id, overwrite),
        episode_audio_qc_task.si(ep_id, overwrite),
    )

    # Apply the chain with our pre-generated job_id
    result = pipeline_chain.apply_async(task_id=job_id)

    LOGGER.info(f"[{job_id}] Audio pipeline chain queued for {ep_id}")

    return {
        "job_id": result.id,
        "status": "queued",
        "ep_id": ep_id,
        "stages": [
            "ingest", "separate", "enhance",
            "diarize", "transcribe", "voices",
            "align", "export", "qc"
        ],
    }


# =============================================================================
# Phased Pipeline Functions (for UI buttons)
# =============================================================================


def episode_audio_files_async(
    ep_id: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Queue Phase 1: Create audio files (ingest -> separate -> enhance).

    Args:
        ep_id: Episode identifier
        overwrite: Whether to overwrite existing artifacts

    Returns:
        Dict with job_id and status
    """
    existing = check_active_job(ep_id, "audio_files")
    if existing:
        return {
            "status": "error",
            "error": "Another audio files job is already running",
            "existing_job_id": existing,
            "ep_id": ep_id,
        }

    # Chain: ingest -> separate -> enhance
    pipeline_chain = chain(
        episode_audio_ingest_task.si(ep_id, overwrite),
        episode_audio_separate_task.si(ep_id, overwrite),
        episode_audio_enhance_task.si(ep_id, overwrite),
    )

    result = pipeline_chain.apply_async()

    return {
        "job_id": result.id,
        "status": "queued",
        "ep_id": ep_id,
        "phase": "audio_files",
        "stages": ["ingest", "separate", "enhance"],
    }


def episode_audio_diarize_transcribe_async(
    ep_id: str,
    asr_provider: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Queue Phase 2: Diarization + Transcription + Initial Clustering.

    Args:
        ep_id: Episode identifier
        asr_provider: Override ASR provider
        overwrite: Whether to overwrite existing artifacts

    Returns:
        Dict with job_id and status
    """
    existing = check_active_job(ep_id, "audio_diarize")
    if existing:
        return {
            "status": "error",
            "error": "Another diarization job is already running",
            "existing_job_id": existing,
            "ep_id": ep_id,
        }

    # Run diarize and transcribe in parallel, then voice clustering
    pipeline_chain = chord(
        [
            episode_audio_diarize_task.si(ep_id, overwrite),
            episode_audio_transcribe_task.si(ep_id, asr_provider, overwrite),
        ],
        episode_audio_voices_task.s(ep_id, overwrite),
    )

    result = pipeline_chain.apply_async()

    return {
        "job_id": result.id,
        "status": "queued",
        "ep_id": ep_id,
        "phase": "diarize_transcribe",
        "stages": ["diarize", "transcribe", "voices"],
    }


def episode_audio_finalize_async(
    ep_id: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Queue Phase 4: Finalize transcript (align -> export -> qc).

    Args:
        ep_id: Episode identifier
        overwrite: Whether to overwrite existing artifacts

    Returns:
        Dict with job_id and status
    """
    existing = check_active_job(ep_id, "audio_finalize")
    if existing:
        return {
            "status": "error",
            "error": "Another finalize job is already running",
            "existing_job_id": existing,
            "ep_id": ep_id,
        }

    # Chain: align -> export -> qc
    pipeline_chain = chain(
        episode_audio_align_task.si(ep_id, overwrite),
        episode_audio_export_task.si(ep_id, overwrite),
        episode_audio_qc_task.si(ep_id, overwrite),
    )

    result = pipeline_chain.apply_async()

    return {
        "job_id": result.id,
        "status": "queued",
        "ep_id": ep_id,
        "phase": "finalize",
        "stages": ["align", "export", "qc"],
    }


# Export public API
__all__ = [
    "episode_audio_ingest_task",
    "episode_audio_separate_task",
    "episode_audio_enhance_task",
    "episode_audio_diarize_task",
    "episode_audio_voices_task",
    "episode_audio_transcribe_task",
    "episode_audio_align_task",
    "episode_audio_qc_task",
    "episode_audio_export_task",
    "episode_audio_pipeline_task",
    "episode_audio_pipeline_async",
    # Phased pipeline functions
    "episode_audio_files_async",
    "episode_audio_diarize_transcribe_async",
    "episode_audio_finalize_async",
    "AUDIO_QUEUES",
]
