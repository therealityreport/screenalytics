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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Generator, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_screenalytics.artifacts import get_path

router = APIRouter(prefix="/jobs", tags=["audio"])
LOGGER = logging.getLogger(__name__)


def _stream_audio_pipeline(
    command: list[str],
    ep_id: str,
    timeout: int = 7200,
) -> Generator[str, None, None]:
    """Stream audio pipeline subprocess output as NDJSON.

    The audio pipeline CLI emits JSON progress lines directly.
    This function passes them through with minimal processing.

    Args:
        command: Command to execute
        ep_id: Episode identifier
        timeout: Max runtime in seconds (default 2 hours)

    Yields:
        NDJSON lines
    """
    start_time = time.time()

    # Set up environment
    env = os.environ.copy()
    # Clamp thread envs to keep laptop runs from pegging CPU
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    # Emit start message
    yield json.dumps({
        "phase": "init",
        "progress": 0,
        "message": f"Starting audio pipeline for {ep_id}...",
        "timestamp": time.time(),
    }) + "\n"

    try:
        LOGGER.info(f"Starting audio pipeline subprocess: {' '.join(command)}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(PROJECT_ROOT),
        )

        # Stream output lines as they arrive
        for line in iter(process.stdout.readline, ''):
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                process.kill()
                yield json.dumps({
                    "phase": "error",
                    "progress": 0,
                    "message": f"Audio pipeline timed out after {elapsed:.0f}s",
                    "timestamp": time.time(),
                }) + "\n"
                return

            # Try to parse as JSON (audio pipeline emits JSON progress)
            try:
                data = json.loads(line)
                # Pass through JSON lines from the pipeline
                yield line + "\n"
            except json.JSONDecodeError:
                # For plain log lines, wrap them
                if line.startswith('['):
                    # Looks like a log line [INFO] or [ERROR]
                    yield json.dumps({
                        "phase": "log",
                        "progress": 0,
                        "message": line,
                        "timestamp": time.time(),
                    }) + "\n"
                else:
                    # Other output
                    yield json.dumps({
                        "phase": "log",
                        "progress": 0,
                        "message": line,
                        "timestamp": time.time(),
                    }) + "\n"

        # Wait for process to complete
        process.wait()
        elapsed = time.time() - start_time

        if process.returncode == 0:
            yield json.dumps({
                "phase": "complete",
                "progress": 1.0,
                "message": f"Audio pipeline completed in {elapsed:.1f}s",
                "timestamp": time.time(),
            }) + "\n"
        else:
            yield json.dumps({
                "phase": "error",
                "progress": 0,
                "message": f"Audio pipeline failed with exit code {process.returncode}",
                "timestamp": time.time(),
            }) + "\n"

    except Exception as e:
        LOGGER.exception(f"Audio pipeline streaming error: {e}")
        yield json.dumps({
            "phase": "error",
            "progress": 0,
            "message": f"Audio pipeline error: {e}",
            "timestamp": time.time(),
        }) + "\n"


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
    asr_provider: Optional[Literal["openai_whisper", "gemini_3", "gemini"]] = Field(
        None,
        description="Override ASR provider (gemini_3 and gemini are equivalent)",
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


class VoiceAssignRequest(BaseModel):
    """Request to assign a voice cluster to a cast member or label."""
    voice_cluster_id: str = Field(..., description="Voice cluster ID to assign")
    cast_id: Optional[str] = Field(None, description="Cast member ID to assign to")
    custom_label: Optional[str] = Field(None, description="Custom label if not assigning to cast")


class VoiceAssignResponse(BaseModel):
    """Response from voice assignment."""
    voice_cluster_id: str
    speaker_id: str
    speaker_display_name: str
    voice_bank_id: str
    success: bool = True
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
        # Normalize provider before Celery chain (gemini → gemini_3)
        asr_provider = req.asr_provider
        if asr_provider == "gemini":
            asr_provider = "gemini_3"

        # Queue via Celery
        try:
            from apps.api.jobs_audio import episode_audio_pipeline_async

            result = episode_audio_pipeline_async(
                req.ep_id,
                overwrite=req.overwrite,
                asr_provider=asr_provider,
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
        # Run locally via streaming subprocess for real-time logs
        # Normalize ASR provider name (API uses gemini_3, CLI uses gemini)
        asr_provider = req.asr_provider or "openai_whisper"
        if asr_provider == "gemini_3":
            asr_provider = "gemini"

        # Build command for audio pipeline
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "audio_pipeline_run.py"),
            "--ep-id", req.ep_id,
            "--asr-provider", asr_provider,
        ]
        if req.overwrite:
            command.append("--overwrite")

        # Progress file for UI polling
        data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
        manifests_dir = data_root / "manifests" / req.ep_id
        manifests_dir.mkdir(parents=True, exist_ok=True)
        progress_file = manifests_dir / "audio_progress.json"
        command.extend(["--progress-file", str(progress_file)])

        # Return streaming response for live log updates
        return StreamingResponse(
            _stream_audio_pipeline(command, req.ep_id),
            media_type="application/x-ndjson",
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
            },
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


@router.post("/episodes/{ep_id}/audio/voices/assign", response_model=VoiceAssignResponse)
async def assign_voice_cluster(
    ep_id: str,
    req: VoiceAssignRequest,
) -> VoiceAssignResponse:
    """Assign a voice cluster to a cast member or custom label.

    Updates the voice mapping file with the new assignment.

    Args:
        ep_id: Episode identifier
        req: Voice assignment request

    Returns:
        VoiceAssignResponse with assignment details
    """
    paths = _get_audio_paths(ep_id)
    mapping_path = paths["voice_mapping"]

    if not mapping_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Voice mapping not found for episode {ep_id}",
        )

    # Validate request - must have either cast_id or custom_label
    if not req.cast_id and not req.custom_label:
        raise HTTPException(
            status_code=400,
            detail="Must provide either cast_id or custom_label",
        )

    try:
        # Load current mapping
        with mapping_path.open("r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        # Find the cluster in mapping
        cluster_entry = None
        cluster_idx = -1
        for i, entry in enumerate(mapping_data):
            if entry.get("voice_cluster_id") == req.voice_cluster_id:
                cluster_entry = entry
                cluster_idx = i
                break

        if cluster_entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"Voice cluster {req.voice_cluster_id} not found in mapping",
            )

        # Determine speaker info
        if req.cast_id:
            # Assign to cast member
            speaker_id = f"SPK_{req.cast_id.upper()}"
            speaker_display_name = req.cast_id.replace("_", " ").title()
            voice_bank_id = f"voice_{req.cast_id.lower()}"

            # Try to get cast member name from API
            try:
                from apps.api.routers.cast import get_cast_member
                show_id = ep_id.split("-")[0].upper()
                cast_info = await get_cast_member(show_id, req.cast_id)
                if cast_info and cast_info.get("name"):
                    speaker_display_name = cast_info["name"]
            except Exception:
                pass  # Use default name if cast lookup fails
        else:
            # Custom label
            speaker_id = f"SPK_CUSTOM_{req.custom_label.upper().replace(' ', '_')}"
            speaker_display_name = req.custom_label
            voice_bank_id = f"voice_custom_{req.custom_label.lower().replace(' ', '_')}"

        # Update the entry
        mapping_data[cluster_idx]["speaker_id"] = speaker_id
        mapping_data[cluster_idx]["speaker_display_name"] = speaker_display_name
        mapping_data[cluster_idx]["voice_bank_id"] = voice_bank_id
        mapping_data[cluster_idx]["similarity"] = 1.0  # Manual assignment = perfect match

        # Save updated mapping
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(mapping_data, f, indent=2)

        LOGGER.info(f"Assigned voice cluster {req.voice_cluster_id} to {speaker_display_name}")

        return VoiceAssignResponse(
            voice_cluster_id=req.voice_cluster_id,
            speaker_id=speaker_id,
            speaker_display_name=speaker_display_name,
            voice_bank_id=voice_bank_id,
            success=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception(f"Failed to assign voice cluster: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to assign voice cluster: {e}",
        )
