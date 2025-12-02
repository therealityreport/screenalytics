"""PyannoteAI HTTP client for official v1 API.

Implements the official PyannoteAI workflow:
- POST /v1/diarize with S3 signed URL
- Poll GET /v1/jobs/{jobId} with 5-8 second intervals
- Optional webhook support for async completion

Reference: https://docs.pyannote.ai/
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx

from .circuit_breaker import CircuitBreakerError, get_pyannote_breaker

LOGGER = logging.getLogger(__name__)

PYANNOTE_API_BASE = "https://api.pyannote.ai/v1"

# Job status values from official docs
STATUS_PENDING = "pending"
STATUS_CREATED = "created"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELED = "canceled"

# Terminal statuses (stop polling)
TERMINAL_STATUSES = {STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELED}


@dataclass
class DiarizationJobResult:
    """Result from a completed diarization job."""

    job_id: str
    status: str
    diarization: List[Dict[str, Any]] = field(default_factory=list)
    exclusive_diarization: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class VoiceprintJobResult:
    """Result from a completed voiceprint creation job."""

    job_id: str
    status: str
    voiceprint: Optional[str] = None  # Base64 encoded voiceprint blob
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class IdentificationJobResult:
    """Result from a completed identification job.

    Output format from /v1/identify:
    - diarization: [{speaker, start, end}, ...]
    - identification: [{speaker, start, end, diarizationSpeaker, match}, ...]
    - voiceprints: [{speaker, match, confidence: {label: score, ...}}, ...]
    """

    job_id: str
    status: str
    diarization: List[Dict[str, Any]] = field(default_factory=list)
    identification: List[Dict[str, Any]] = field(default_factory=list)
    voiceprints: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class PyannoteAPIError(Exception):
    """Exception for PyannoteAI API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, job_id: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.job_id = job_id


class PyannoteAPIClient:
    """HTTP client for PyannoteAI v1 API.

    Implements the official API workflow:
    1. Submit diarization job with S3 signed URL
    2. Poll for completion (or use webhook)
    3. Parse exclusive diarization output

    Example:
        client = PyannoteAPIClient(api_key="your-key")

        # Get S3 presigned URL for audio file
        audio_url = client.upload_and_get_url(audio_path)

        # Submit diarization job
        job_id = client.submit_diarization(
            media_url=audio_url,
            min_speakers=3,
            max_speakers=9,
            exclusive=True,
        )

        # Poll for result
        result = client.poll_job(job_id, max_wait=900.0)

        # Use exclusive diarization if available
        segments = result.exclusive_diarization or result.diarization
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        poll_interval_base: float = 6.0,
        poll_interval_jitter: float = 2.0,
    ):
        """Initialize PyannoteAI API client.

        Args:
            api_key: PyannoteAI API key (or from PYANNOTEAI_API_KEY env var)
            timeout: HTTP request timeout in seconds
            poll_interval_base: Base polling interval in seconds (5-8s per docs)
            poll_interval_jitter: Random jitter added to polling interval
        """
        self._api_key = api_key or os.environ.get("PYANNOTEAI_API_KEY")
        if not self._api_key:
            raise PyannoteAPIError("PYANNOTEAI_API_KEY not set")

        self._timeout = timeout
        self._poll_interval_base = poll_interval_base
        self._poll_interval_jitter = poll_interval_jitter
        self._breaker = get_pyannote_breaker()

        self._client = httpx.Client(
            base_url=PYANNOTE_API_BASE,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def upload_and_get_url(
        self,
        audio_path: Path,
        expiry_seconds: int = 3600,
    ) -> str:
        """Upload audio to S3 and return presigned GET URL.

        Args:
            audio_path: Path to local audio file
            expiry_seconds: URL expiry time (default 1 hour)

        Returns:
            Presigned S3 GET URL accessible to PyannoteAI

        Raises:
            PyannoteAPIError: If S3 is not configured or upload fails
        """
        try:
            from apps.api.services.storage import StorageService
        except ImportError as e:
            raise PyannoteAPIError(
                "StorageService not available. Ensure STORAGE_BACKEND=s3 is configured."
            ) from e

        storage = StorageService()

        if not storage.s3_enabled():
            raise PyannoteAPIError(
                "S3 storage not enabled. PyannoteAI API requires publicly accessible audio URL. "
                "Set STORAGE_BACKEND=s3 and configure S3 credentials."
            )

        # Upload to temp location
        job_uuid = str(uuid4())
        s3_key = f"temp/diarization/{job_uuid}/{audio_path.name}"

        LOGGER.info(f"Uploading audio to S3: {s3_key}")

        # Read file and upload
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        success = storage.upload_bytes(
            audio_data,
            s3_key,
            content_type="audio/wav" if audio_path.suffix.lower() == ".wav" else "audio/mpeg",
        )

        if not success:
            raise PyannoteAPIError(f"Failed to upload audio to S3: {s3_key}")

        # Generate presigned URL
        url = storage.presign_get(s3_key, expiry_seconds)
        if not url:
            raise PyannoteAPIError(f"Failed to generate presigned URL for: {s3_key}")

        LOGGER.info(f"Audio uploaded, presigned URL generated (expires in {expiry_seconds}s)")
        return url

    def submit_diarization(
        self,
        media_url: str,
        model: str = "precision-2",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        exclusive: bool = True,
        webhook_url: Optional[str] = None,
    ) -> str:
        """Submit diarization job to PyannoteAI.

        Request body matches official spec exactly:
        {
            "url": "<signed_s3_url>",
            "model": "precision-2",
            "exclusive": true,
            "minSpeakers": <int>,
            "maxSpeakers": <int>,
            "webhook": "<optional_url>"
        }

        Note: numSpeakers is NEVER used per requirements.

        Args:
            media_url: Publicly accessible audio URL
            model: Diarization model (default: precision-2)
            min_speakers: Minimum expected speakers (cast_count - 3)
            max_speakers: Maximum expected speakers (cast_count + 3)
            exclusive: Use exclusive diarization (required for clean merging)
            webhook_url: Optional webhook URL for async notification

        Returns:
            Job ID for polling or webhook correlation

        Raises:
            PyannoteAPIError: If submission fails
        """
        # Build request body matching official spec
        body: Dict[str, Any] = {
            "url": media_url,
            "model": model,
            "exclusive": exclusive,
        }

        # Add speaker constraints (NEVER use numSpeakers)
        if min_speakers is not None and min_speakers >= 1:
            body["minSpeakers"] = min_speakers
        if max_speakers is not None and max_speakers >= 1:
            body["maxSpeakers"] = max_speakers

        # Add optional webhook
        if webhook_url:
            body["webhook"] = webhook_url

        LOGGER.info(
            f"Submitting diarization job: model={model}, "
            f"speakers={min_speakers}-{max_speakers}, exclusive={exclusive}"
        )
        LOGGER.debug(f"Diarization request body: {json.dumps(body)}")

        try:
            with self._breaker:
                response = self._client.post("/diarize", json=body)
        except CircuitBreakerError as e:
            raise PyannoteAPIError(
                f"PyannoteAI API unavailable (circuit breaker open): {e}"
            ) from e

        if response.status_code != 200:
            error_text = response.text
            raise PyannoteAPIError(
                f"Diarization submission failed: {response.status_code} - {error_text}",
                status_code=response.status_code,
            )

        result = response.json()
        job_id = result.get("jobId")

        if not job_id:
            raise PyannoteAPIError(
                f"No jobId in response: {result}"
            )

        LOGGER.info(f"Diarization job submitted to PyannoteAI: jobId={job_id}")
        return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a diarization job.

        Args:
            job_id: Job ID from submit_diarization

        Returns:
            Job status response dict

        Raises:
            PyannoteAPIError: If status check fails
        """
        try:
            with self._breaker:
                response = self._client.get(f"/jobs/{job_id}")
        except CircuitBreakerError as e:
            raise PyannoteAPIError(
                f"PyannoteAI API unavailable (circuit breaker open): {e}",
                job_id=job_id,
            ) from e

        if response.status_code != 200:
            raise PyannoteAPIError(
                f"Job status check failed: {response.status_code} - {response.text}",
                status_code=response.status_code,
                job_id=job_id,
            )

        return response.json()

    def poll_job(
        self,
        job_id: str,
        max_wait: float = 900.0,
    ) -> DiarizationJobResult:
        """Poll for job completion with 5-8 second intervals.

        Per official docs:
        - Poll GET /v1/jobs/{jobId} every 5-8 seconds
        - Stop when status is: succeeded, failed, or canceled

        Args:
            job_id: Job ID to poll
            max_wait: Maximum wait time in seconds (default 15 min)

        Returns:
            DiarizationJobResult with segments

        Raises:
            PyannoteAPIError: If job fails or times out
        """
        elapsed = 0.0
        LOGGER.info(f"Polling PyannoteAI job: {job_id} (timeout: {max_wait}s)")

        while elapsed < max_wait:
            result = self.get_job_status(job_id)
            status = result.get("status", "unknown")

            LOGGER.debug(f"Job {job_id} status: {status}")

            if status in TERMINAL_STATUSES:
                return self._parse_job_result(job_id, result)

            # Calculate next poll interval (5-8 seconds per docs)
            jitter = random.uniform(-1.0, self._poll_interval_jitter)
            poll_interval = self._poll_interval_base + jitter

            LOGGER.info(f"Polling PyannoteAI job status... (interval: {poll_interval:.1f}s)")
            time.sleep(poll_interval)
            elapsed += poll_interval

        raise PyannoteAPIError(
            f"Diarization job timed out after {max_wait}s",
            job_id=job_id,
        )

    def _parse_job_result(
        self,
        job_id: str,
        result: Dict[str, Any],
    ) -> DiarizationJobResult:
        """Parse completed job result.

        Prefers exclusiveDiarization over regular diarization.

        Args:
            job_id: Job ID
            result: Raw API response

        Returns:
            Parsed DiarizationJobResult
        """
        status = result.get("status", "unknown")

        if status == STATUS_FAILED:
            error = result.get("error", "Unknown error")
            LOGGER.error(f"PyannoteAI diarization failed: {error}")
            return DiarizationJobResult(
                job_id=job_id,
                status=status,
                error=error,
                raw_response=result,
            )

        if status == STATUS_CANCELED:
            LOGGER.warning(f"PyannoteAI diarization was canceled: {job_id}")
            return DiarizationJobResult(
                job_id=job_id,
                status=status,
                error="Job was canceled",
                raw_response=result,
            )

        # Parse output
        output = result.get("output", {})

        # Prefer exclusive diarization for clean segment merging
        exclusive_diarization = output.get("exclusiveDiarization")
        diarization = output.get("diarization", [])

        if exclusive_diarization:
            LOGGER.info(
                f"Using exclusive diarization output ({len(exclusive_diarization)} segments)"
            )
        else:
            LOGGER.info(
                f"Using regular diarization output ({len(diarization)} segments)"
            )

        LOGGER.info(f"PyannoteAI diarization succeeded: jobId={job_id}")

        return DiarizationJobResult(
            job_id=job_id,
            status=status,
            diarization=diarization,
            exclusive_diarization=exclusive_diarization,
            raw_response=result,
        )

    # =========================================================================
    # Voiceprint Creation API
    # =========================================================================

    def submit_voiceprint(
        self,
        media_url: str,
        webhook_url: Optional[str] = None,
    ) -> str:
        """Submit voiceprint creation job to PyannoteAI.

        Per official docs:
        - Audio must be single-speaker only
        - Maximum duration: 30 seconds
        - POST /v1/voiceprint with {url: <audio_url>}

        Args:
            media_url: Publicly accessible audio URL (single speaker, <=30s)
            webhook_url: Optional webhook URL for async notification

        Returns:
            Job ID for polling

        Raises:
            PyannoteAPIError: If submission fails
        """
        body: Dict[str, Any] = {"url": media_url}

        if webhook_url:
            body["webhook"] = webhook_url

        LOGGER.info(f"Submitting voiceprint job to PyannoteAI")
        LOGGER.debug(f"Voiceprint request body: {json.dumps(body)}")

        try:
            with self._breaker:
                response = self._client.post("/voiceprint", json=body)
        except CircuitBreakerError as e:
            raise PyannoteAPIError(
                f"PyannoteAI API unavailable (circuit breaker open): {e}"
            ) from e

        if response.status_code != 200:
            error_text = response.text
            raise PyannoteAPIError(
                f"Voiceprint submission failed: {response.status_code} - {error_text}",
                status_code=response.status_code,
            )

        result = response.json()
        job_id = result.get("jobId")

        if not job_id:
            raise PyannoteAPIError(f"No jobId in response: {result}")

        LOGGER.info(f"Voiceprint job submitted to PyannoteAI: jobId={job_id}")
        return job_id

    def poll_voiceprint_job(
        self,
        job_id: str,
        max_wait: float = 300.0,
    ) -> VoiceprintJobResult:
        """Poll for voiceprint job completion.

        Args:
            job_id: Job ID to poll
            max_wait: Maximum wait time in seconds (default 5 min)

        Returns:
            VoiceprintJobResult with voiceprint blob

        Raises:
            PyannoteAPIError: If job fails or times out
        """
        elapsed = 0.0
        LOGGER.info(f"Polling PyannoteAI voiceprint job: {job_id} (timeout: {max_wait}s)")

        while elapsed < max_wait:
            result = self.get_job_status(job_id)
            status = result.get("status", "unknown")

            LOGGER.debug(f"Voiceprint job {job_id} status: {status}")

            if status in TERMINAL_STATUSES:
                return self._parse_voiceprint_result(job_id, result)

            jitter = random.uniform(-1.0, self._poll_interval_jitter)
            poll_interval = self._poll_interval_base + jitter

            LOGGER.info(f"Polling PyannoteAI voiceprint job status... (interval: {poll_interval:.1f}s)")
            time.sleep(poll_interval)
            elapsed += poll_interval

        raise PyannoteAPIError(
            f"Voiceprint job timed out after {max_wait}s",
            job_id=job_id,
        )

    def _parse_voiceprint_result(
        self,
        job_id: str,
        result: Dict[str, Any],
    ) -> VoiceprintJobResult:
        """Parse completed voiceprint job result."""
        status = result.get("status", "unknown")

        if status == STATUS_FAILED:
            error = result.get("error", "Unknown error")
            LOGGER.error(f"PyannoteAI voiceprint creation failed: {error}")
            return VoiceprintJobResult(
                job_id=job_id,
                status=status,
                error=error,
                raw_response=result,
            )

        if status == STATUS_CANCELED:
            LOGGER.warning(f"PyannoteAI voiceprint job was canceled: {job_id}")
            return VoiceprintJobResult(
                job_id=job_id,
                status=status,
                error="Job was canceled",
                raw_response=result,
            )

        # Parse output
        output = result.get("output", {})
        voiceprint = output.get("voiceprint")

        if not voiceprint:
            LOGGER.warning(f"No voiceprint in output for job {job_id}")

        LOGGER.info(f"PyannoteAI voiceprint succeeded: jobId={job_id}")

        return VoiceprintJobResult(
            job_id=job_id,
            status=status,
            voiceprint=voiceprint,
            raw_response=result,
        )

    # =========================================================================
    # Identification API
    # =========================================================================

    def submit_identification(
        self,
        media_url: str,
        voiceprints: List[Dict[str, str]],
        threshold: int = 60,
        exclusive: bool = True,
        webhook_url: Optional[str] = None,
    ) -> str:
        """Submit identification job to PyannoteAI.

        Per official docs:
        POST /v1/identify with:
        {
            "url": "<episode_audio_url>",
            "voiceprints": [
                {"label": "<cast_id>", "voiceprint": "<base64_blob>"},
                ...
            ],
            "matching": {
                "threshold": <0-100>,
                "exclusive": true
            }
        }

        Args:
            media_url: Publicly accessible audio URL for full episode
            voiceprints: List of {label: str, voiceprint: str} dicts
            threshold: Minimum confidence for matching (0-100, default 60)
            exclusive: Prevent multiple speakers matching same voiceprint
            webhook_url: Optional webhook URL for async notification

        Returns:
            Job ID for polling

        Raises:
            PyannoteAPIError: If submission fails
        """
        if not voiceprints:
            raise PyannoteAPIError("At least one voiceprint required for identification")

        body: Dict[str, Any] = {
            "url": media_url,
            "voiceprints": voiceprints,
            "matching": {
                "threshold": threshold,
                "exclusive": exclusive,
            },
        }

        if webhook_url:
            body["webhook"] = webhook_url

        LOGGER.info(
            f"Submitting identification job: {len(voiceprints)} voiceprints, "
            f"threshold={threshold}, exclusive={exclusive}"
        )
        LOGGER.debug(f"Identification request body (voiceprints truncated): "
                     f"url={media_url}, labels={[v['label'] for v in voiceprints]}")

        try:
            with self._breaker:
                response = self._client.post("/identify", json=body)
        except CircuitBreakerError as e:
            raise PyannoteAPIError(
                f"PyannoteAI API unavailable (circuit breaker open): {e}"
            ) from e

        if response.status_code != 200:
            error_text = response.text
            raise PyannoteAPIError(
                f"Identification submission failed: {response.status_code} - {error_text}",
                status_code=response.status_code,
            )

        result = response.json()
        job_id = result.get("jobId")

        if not job_id:
            raise PyannoteAPIError(f"No jobId in response: {result}")

        LOGGER.info(f"Identification job submitted to PyannoteAI: jobId={job_id}")
        return job_id

    def poll_identification_job(
        self,
        job_id: str,
        max_wait: float = 900.0,
    ) -> IdentificationJobResult:
        """Poll for identification job completion.

        Args:
            job_id: Job ID to poll
            max_wait: Maximum wait time in seconds (default 15 min)

        Returns:
            IdentificationJobResult with diarization, identification, and voiceprints

        Raises:
            PyannoteAPIError: If job fails or times out
        """
        elapsed = 0.0
        LOGGER.info(f"Polling PyannoteAI identification job: {job_id} (timeout: {max_wait}s)")

        while elapsed < max_wait:
            result = self.get_job_status(job_id)
            status = result.get("status", "unknown")

            LOGGER.debug(f"Identification job {job_id} status: {status}")

            if status in TERMINAL_STATUSES:
                return self._parse_identification_result(job_id, result)

            jitter = random.uniform(-1.0, self._poll_interval_jitter)
            poll_interval = self._poll_interval_base + jitter

            LOGGER.info(f"Polling PyannoteAI identification job status... (interval: {poll_interval:.1f}s)")
            time.sleep(poll_interval)
            elapsed += poll_interval

        raise PyannoteAPIError(
            f"Identification job timed out after {max_wait}s",
            job_id=job_id,
        )

    def _parse_identification_result(
        self,
        job_id: str,
        result: Dict[str, Any],
    ) -> IdentificationJobResult:
        """Parse completed identification job result.

        Output format:
        {
            "output": {
                "diarization": [{speaker, start, end}, ...],
                "identification": [{speaker, start, end, diarizationSpeaker, match}, ...],
                "voiceprints": [{speaker, match, confidence: {label: score}}, ...]
            }
        }
        """
        status = result.get("status", "unknown")

        if status == STATUS_FAILED:
            error = result.get("error", "Unknown error")
            LOGGER.error(f"PyannoteAI identification failed: {error}")
            return IdentificationJobResult(
                job_id=job_id,
                status=status,
                error=error,
                raw_response=result,
            )

        if status == STATUS_CANCELED:
            LOGGER.warning(f"PyannoteAI identification job was canceled: {job_id}")
            return IdentificationJobResult(
                job_id=job_id,
                status=status,
                error="Job was canceled",
                raw_response=result,
            )

        # Parse output
        output = result.get("output", {})
        diarization = output.get("diarization", [])
        identification = output.get("identification", [])
        voiceprints = output.get("voiceprints", [])

        LOGGER.info(
            f"PyannoteAI identification succeeded: jobId={job_id}, "
            f"{len(diarization)} diarization segments, "
            f"{len(identification)} identification segments, "
            f"{len(voiceprints)} voiceprint matches"
        )

        return IdentificationJobResult(
            job_id=job_id,
            status=status,
            diarization=diarization,
            identification=identification,
            voiceprints=voiceprints,
            raw_response=result,
        )


def calculate_speaker_range(cast_count: int) -> tuple[int, int]:
    """Calculate speaker range from cast count using ±3 rule.

    Rules per requirements:
    - minSpeakers = max(1, cast_count - 3)
    - maxSpeakers = cast_count + 3
    - If cast_count is 0, fallback to [1, 6]
    - Never use numSpeakers

    Args:
        cast_count: Number of cast members for episode

    Returns:
        Tuple of (min_speakers, max_speakers)
    """
    if cast_count <= 0:
        LOGGER.info("No cast data, using fallback speaker range: 1-6")
        return (1, 6)

    min_speakers = max(1, cast_count - 3)
    max_speakers = cast_count + 3

    LOGGER.info(
        f"Using cast-based speaker range: {min_speakers}-{max_speakers} (cast_count={cast_count})"
    )

    return (min_speakers, max_speakers)
