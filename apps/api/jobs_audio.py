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

from celery import Task, chain, chord, group

from apps.api.celery_app import celery_app
from apps.api.tasks import _acquire_lock, _release_lock, check_active_job

LOGGER = logging.getLogger(__name__)

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

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from py_screenalytics.audio.io import extract_audio_from_video
        from py_screenalytics.audio.episode_audio_pipeline import _get_audio_paths, _load_config

        config = _load_config()
        paths = _get_audio_paths(ep_id)

        if not paths["video"].exists():
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
        return {
            "status": "error",
            "ep_id": ep_id,
            "stage": "ingest",
            "error": str(e),
        }


@celery_app.task(bind=True, base=AudioPipelineTask, name="audio.separate")
def episode_audio_separate_task(
    self,
    ep_id: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Separate vocals from accompaniment using MDX-Extra."""
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting audio separation for {ep_id}")

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from py_screenalytics.audio.separation_mdx import separate_vocals
        from py_screenalytics.audio.episode_audio_pipeline import _get_audio_paths, _load_config

        config = _load_config()
        paths = _get_audio_paths(ep_id)

        vocals_path, accompaniment_path = separate_vocals(
            paths["original"],
            paths["audio_dir"],
            config.separation,
            overwrite=overwrite,
        )

        return {
            "status": "success",
            "ep_id": ep_id,
            "stage": "separate",
            "vocals_path": str(vocals_path),
            "accompaniment_path": str(accompaniment_path),
        }

    except Exception as e:
        LOGGER.exception(f"[{job_id}] Audio separation failed: {e}")
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

        if check_api_available():
            enhanced_path = enhance_audio_resemble(
                paths["vocals"],
                paths["vocals_enhanced"],
                config.enhance,
                overwrite=overwrite,
            )
            method = "resemble_api"
        else:
            LOGGER.warning("Resemble API not available, using local enhancement")
            enhanced_path = enhance_audio_local(
                paths["vocals"],
                paths["vocals_enhanced"],
                overwrite=overwrite,
            )
            method = "local"

        return {
            "status": "success",
            "ep_id": ep_id,
            "stage": "enhance",
            "enhanced_path": str(enhanced_path),
            "method": method,
        }

    except Exception as e:
        LOGGER.exception(f"[{job_id}] Audio enhancement failed: {e}")
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

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from py_screenalytics.audio.diarization_pyannote import run_diarization
        from py_screenalytics.audio.episode_audio_pipeline import _get_audio_paths, _load_config

        config = _load_config()
        paths = _get_audio_paths(ep_id)

        # Use enhanced vocals if available, otherwise original
        audio_path = paths["vocals_enhanced"]
        if not audio_path.exists():
            audio_path = paths["vocals"]
        if not audio_path.exists():
            audio_path = paths["original"]

        segments = run_diarization(
            audio_path,
            paths["diarization"],
            config.diarization,
            overwrite=overwrite,
        )

        speakers = set(s.speaker for s in segments)

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

        # Load diarization segments
        diarization_segments = _load_diarization_manifest(paths["diarization"])

        # Use enhanced vocals if available
        audio_path = paths["vocals_enhanced"]
        if not audio_path.exists():
            audio_path = paths["vocals"]

        # Cluster voices
        voice_clusters = cluster_episode_voices(
            audio_path,
            diarization_segments,
            paths["voice_clusters"],
            config.voice_clustering,
            overwrite=overwrite,
        )

        # Map to voice bank
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

        if provider == "gemini_3":
            from py_screenalytics.audio.asr_gemini import transcribe_audio
        else:
            from py_screenalytics.audio.asr_openai import transcribe_audio

        # Use enhanced vocals if available
        audio_path = paths["vocals_enhanced"]
        if not audio_path.exists():
            audio_path = paths["vocals"]

        segments = transcribe_audio(
            audio_path,
            paths["asr_raw"],
            config.asr,
            overwrite=overwrite,
        )

        word_count = sum(len(s.words or []) for s in segments)

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

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from py_screenalytics.audio.fuse_diarization_asr import fuse_transcript
        from py_screenalytics.audio.diarization_pyannote import _load_diarization_manifest
        from py_screenalytics.audio.asr_openai import _load_asr_manifest
        from py_screenalytics.audio.voice_clusters import _load_voice_clusters
        from py_screenalytics.audio.voice_bank import _load_voice_mapping
        from py_screenalytics.audio.episode_audio_pipeline import _get_audio_paths, _load_config

        config = _load_config()
        paths = _get_audio_paths(ep_id)

        # Load all inputs
        diarization_segments = _load_diarization_manifest(paths["diarization"])
        asr_segments = _load_asr_manifest(paths["asr_raw"])
        voice_clusters = _load_voice_clusters(paths["voice_clusters"])
        voice_mapping = _load_voice_mapping(paths["voice_mapping"])

        # Fuse into transcript
        transcript_rows = fuse_transcript(
            diarization_segments,
            asr_segments,
            voice_clusters,
            voice_mapping,
            paths["transcript_jsonl"],
            paths["transcript_vtt"],
            config.export.vtt_include_speaker_notes,
            overwrite=overwrite,
        )

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

        # Load data
        diarization_segments = _load_diarization_manifest(paths["diarization"])
        asr_segments = _load_asr_manifest(paths["asr_raw"])
        voice_clusters = _load_voice_clusters(paths["voice_clusters"])
        voice_mapping = _load_voice_mapping(paths["voice_mapping"])
        transcript_rows = _load_transcript_jsonl(paths["transcript_jsonl"])

        # Get durations and SNR
        duration_original = get_audio_duration(paths["original"])
        duration_final = get_audio_duration(paths["final_voice_only"]) if paths["final_voice_only"].exists() else duration_original

        snr_db = None
        try:
            snr_db = compute_snr(paths["vocals_enhanced"])
        except Exception:
            pass

        # Run QC
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

        final_path = export_final_audio(
            input_path,
            paths["final_voice_only"],
            config.export,
            overwrite=overwrite,
        )

        duration = get_audio_duration(final_path)

        return {
            "status": "success",
            "ep_id": ep_id,
            "stage": "export",
            "final_path": str(final_path),
            "duration_s": duration,
        }

    except Exception as e:
        LOGGER.exception(f"[{job_id}] Audio export failed: {e}")
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
    # Check for existing job
    existing = check_active_job(ep_id, "audio_pipeline")
    if existing:
        return {
            "status": "error",
            "error": "Another audio pipeline job is already running",
            "existing_job_id": existing,
            "ep_id": ep_id,
        }

    # Build the chain
    # Sequential stages: ingest -> separate -> enhance
    # Parallel: diarize & transcribe
    # Sequential: voices -> align -> qc -> export

    pipeline_chain = chain(
        episode_audio_ingest_task.s(ep_id, overwrite),
        episode_audio_separate_task.s(ep_id, overwrite),
        episode_audio_enhance_task.s(ep_id, overwrite),
        # Parallel diarization and transcription
        chord(
            [
                episode_audio_diarize_task.s(ep_id, overwrite),
                episode_audio_transcribe_task.s(ep_id, asr_provider, overwrite),
            ],
            episode_audio_voices_task.s(ep_id, overwrite),
        ),
        episode_audio_align_task.s(ep_id, overwrite),
        episode_audio_export_task.s(ep_id, overwrite),
        episode_audio_qc_task.s(ep_id, overwrite),
    )

    # Apply the chain
    result = pipeline_chain.apply_async()

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
    "AUDIO_QUEUES",
]
