"""Audio pipeline API endpoints.

Provides HTTP endpoints for:
- Starting audio pipeline jobs
- Querying job status
- Downloading transcripts
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_screenalytics.artifacts import get_path

router = APIRouter(prefix="/jobs", tags=["audio"])
LOGGER = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================


class AudioPipelineRequest(BaseModel):
    """Request to start audio pipeline."""
    ep_id: str = Field(..., description="Episode identifier")
    run_mode: Literal["queue", "local"] = Field(
        "queue",
        description="Run mode: 'queue' for Celery, 'local' for synchronous",
    )
    overwrite: bool = Field(False, description="Overwrite existing artifacts")
    asr_provider: Optional[Literal["openai_whisper", "gemini_3"]] = Field(
        None,
        description="Override ASR provider",
    )


class AudioPipelineResponse(BaseModel):
    """Response from starting audio pipeline."""
    job_id: Optional[str] = None
    status: str
    ep_id: str
    run_mode: Optional[str] = None
    error: Optional[str] = None


class AudioStatusResponse(BaseModel):
    """Response from audio status query."""
    ep_id: str
    status: str
    qc_status: Optional[str] = None
    summary: Optional[dict] = None
    artifacts: Optional[dict] = None
    error: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================


def _get_audio_paths(ep_id: str) -> dict:
    """Get paths for audio artifacts."""
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))

    audio_dir = data_root / "audio" / ep_id
    manifests_dir = data_root / "manifests" / ep_id

    return {
        "audio_original": audio_dir / "episode_original.wav",
        "audio_vocals": audio_dir / "episode_vocals.wav",
        "audio_vocals_enhanced": audio_dir / "episode_vocals_enhanced.wav",
        "audio_final": audio_dir / "episode_final_voice_only.wav",
        "diarization": manifests_dir / "audio_diarization.jsonl",
        "asr_raw": manifests_dir / "audio_asr_raw.jsonl",
        "voice_clusters": manifests_dir / "audio_voice_clusters.json",
        "voice_mapping": manifests_dir / "audio_voice_mapping.json",
        "transcript_jsonl": manifests_dir / "episode_transcript.jsonl",
        "transcript_vtt": manifests_dir / "episode_transcript.vtt",
        "qc": manifests_dir / "audio_qc.json",
    }


def _check_artifacts_exist(ep_id: str) -> dict:
    """Check which audio artifacts exist for an episode."""
    paths = _get_audio_paths(ep_id)
    return {
        key: str(path) if path.exists() else None
        for key, path in paths.items()
    }


def _load_qc_status(ep_id: str) -> tuple[str, dict]:
    """Load QC status from file."""
    paths = _get_audio_paths(ep_id)
    qc_path = paths["qc"]

    if not qc_path.exists():
        return "unknown", {}

    try:
        with qc_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("status", "unknown"), {
            "duration_drift_pct": data.get("duration_drift_pct"),
            "snr_db": data.get("snr_db"),
            "mean_asr_conf": data.get("mean_asr_conf"),
            "mean_diarization_conf": data.get("mean_diarization_conf"),
            "voice_cluster_count": data.get("voice_cluster_count"),
            "labeled_voices": data.get("labeled_voices"),
            "unlabeled_voices": data.get("unlabeled_voices"),
        }
    except Exception as e:
        LOGGER.warning(f"Failed to load QC status for {ep_id}: {e}")
        return "error", {}


# =============================================================================
# API Endpoints
# =============================================================================


@router.post("/episode_audio_pipeline", response_model=AudioPipelineResponse)
async def start_audio_pipeline(req: AudioPipelineRequest) -> AudioPipelineResponse:
    """Start the audio pipeline for an episode.

    This endpoint kicks off the complete audio processing pipeline:
    1. Extract audio from video
    2. Separate vocals (MDX-Extra)
    3. Enhance audio (Resemble API)
    4. Speaker diarization (Pyannote)
    5. Transcription (OpenAI Whisper / Gemini)
    6. Voice clustering and bank mapping
    7. Transcript generation (JSONL + VTT)
    8. Quality control

    Args:
        req: Pipeline request with ep_id, run_mode, and options

    Returns:
        AudioPipelineResponse with job_id and status
    """
    # Validate episode exists
    video_path = get_path(req.ep_id, "video")
    if not video_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Video not found for episode {req.ep_id}. Mirror from S3 first.",
        )

    if req.run_mode == "queue":
        # Queue via Celery
        try:
            from apps.api.jobs_audio import episode_audio_pipeline_async

            result = episode_audio_pipeline_async(
                req.ep_id,
                overwrite=req.overwrite,
                asr_provider=req.asr_provider,
            )

            if result.get("status") == "error":
                return AudioPipelineResponse(
                    status="error",
                    ep_id=req.ep_id,
                    error=result.get("error"),
                )

            return AudioPipelineResponse(
                job_id=result.get("job_id"),
                status="queued",
                ep_id=req.ep_id,
                run_mode="queue",
            )

        except Exception as e:
            LOGGER.exception(f"Failed to queue audio pipeline: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to queue audio pipeline: {e}",
            )

    else:
        # Run locally (synchronous)
        try:
            from py_screenalytics.audio.episode_audio_pipeline import run_episode_audio_pipeline

            result = run_episode_audio_pipeline(
                req.ep_id,
                overwrite=req.overwrite,
                asr_provider=req.asr_provider,
            )

            return AudioPipelineResponse(
                status=result.status,
                ep_id=req.ep_id,
                run_mode="local",
                error=result.error,
            )

        except Exception as e:
            LOGGER.exception(f"Audio pipeline failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Audio pipeline failed: {e}",
            )


@router.get("/episode_audio_status", response_model=AudioStatusResponse)
async def get_audio_status(
    ep_id: str = Query(..., description="Episode identifier"),
) -> AudioStatusResponse:
    """Get audio pipeline status for an episode.

    Returns the current status, QC results, and available artifacts.

    Args:
        ep_id: Episode identifier

    Returns:
        AudioStatusResponse with status and summary
    """
    # Check which artifacts exist
    artifacts = _check_artifacts_exist(ep_id)

    # Determine overall status based on artifacts
    if artifacts.get("transcript_jsonl"):
        status = "succeeded"
    elif artifacts.get("audio_original"):
        status = "running"
    else:
        status = "not_started"

    # Load QC status if available
    qc_status, summary = _load_qc_status(ep_id)

    return AudioStatusResponse(
        ep_id=ep_id,
        status=status,
        qc_status=qc_status,
        summary=summary,
        artifacts=artifacts,
    )


@router.get("/episodes/{ep_id}/audio/transcript.vtt")
async def download_transcript_vtt(ep_id: str) -> Response:
    """Download WebVTT transcript for an episode.

    Args:
        ep_id: Episode identifier

    Returns:
        VTT file response
    """
    paths = _get_audio_paths(ep_id)
    vtt_path = paths["transcript_vtt"]

    if not vtt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"VTT transcript not found for episode {ep_id}",
        )

    content = vtt_path.read_text(encoding="utf-8")
    return Response(
        content=content,
        media_type="text/vtt",
        headers={
            "Content-Disposition": f'attachment; filename="{ep_id}_transcript.vtt"'
        },
    )


@router.get("/episodes/{ep_id}/audio/transcript.jsonl")
async def download_transcript_jsonl(ep_id: str) -> Response:
    """Download JSONL transcript for an episode.

    Args:
        ep_id: Episode identifier

    Returns:
        JSONL file response
    """
    paths = _get_audio_paths(ep_id)
    jsonl_path = paths["transcript_jsonl"]

    if not jsonl_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"JSONL transcript not found for episode {ep_id}",
        )

    content = jsonl_path.read_text(encoding="utf-8")
    return Response(
        content=content,
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="{ep_id}_transcript.jsonl"'
        },
    )


@router.get("/episodes/{ep_id}/audio/qc.json")
async def download_audio_qc(ep_id: str) -> dict:
    """Get audio QC report for an episode.

    Args:
        ep_id: Episode identifier

    Returns:
        QC report JSON
    """
    paths = _get_audio_paths(ep_id)
    qc_path = paths["qc"]

    if not qc_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio QC report not found for episode {ep_id}",
        )

    with qc_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/audio/prerequisites")
async def check_audio_prerequisites() -> dict:
    """Check if audio pipeline prerequisites are met.

    Returns status of required dependencies and API keys.

    Returns:
        Dict with status of each prerequisite
    """
    try:
        from py_screenalytics.audio.episode_audio_pipeline import check_pipeline_prerequisites
        return check_pipeline_prerequisites()
    except ImportError as e:
        return {
            "error": f"Audio module not available: {e}",
            "ffmpeg": False,
            "soundfile": False,
            "demucs": False,
            "pyannote": False,
            "openai": bool(os.environ.get("OPENAI_API_KEY")),
            "gemini": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
            "resemble": bool(os.environ.get("RESEMBLE_API_KEY")),
        }
