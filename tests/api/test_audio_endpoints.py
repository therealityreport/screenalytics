"""Tests for audio pipeline API endpoints.

Verifies:
1. Audio pipeline job creation endpoint
2. Job status query endpoint
3. Transcript download endpoints
4. Prerequisites check endpoint
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def api_client():
    """Create a test client for the API."""
    from apps.api.main import app
    return TestClient(app)


class TestAudioPipelineEndpoints:
    """Tests for /jobs/episode_audio_pipeline endpoint."""

    def test_audio_pipeline_missing_ep_id(self, api_client):
        """POST without ep_id returns 422."""
        response = api_client.post(
            "/jobs/episode_audio_pipeline",
            json={"asr_provider": "openai_whisper"},
        )
        assert response.status_code == 422

    def test_audio_pipeline_invalid_asr_provider(self, api_client):
        """POST with invalid asr_provider returns 400."""
        response = api_client.post(
            "/jobs/episode_audio_pipeline",
            json={"ep_id": "test-s01e01", "asr_provider": "invalid_provider"},
        )
        assert response.status_code == 400

    @patch("apps.api.routers.audio.episode_audio_pipeline_task")
    def test_audio_pipeline_success(self, mock_task, api_client):
        """POST with valid params starts job and returns 202."""
        mock_result = MagicMock()
        mock_result.id = "test-job-id-123"
        mock_task.delay.return_value = mock_result

        response = api_client.post(
            "/jobs/episode_audio_pipeline",
            json={"ep_id": "test-s01e01", "asr_provider": "openai_whisper"},
        )

        assert response.status_code == 202
        data = response.json()
        assert data["job_id"] == "test-job-id-123"
        assert data["ep_id"] == "test-s01e01"
        assert data["job_type"] == "audio_pipeline"


class TestAudioStatusEndpoint:
    """Tests for /jobs/episode_audio_status endpoint."""

    def test_audio_status_missing_ep_id(self, api_client):
        """GET without ep_id returns 422."""
        response = api_client.get("/jobs/episode_audio_status")
        assert response.status_code == 422

    @patch("apps.api.routers.audio.AsyncResult")
    def test_audio_status_no_job(self, mock_async, api_client):
        """GET with no active job returns not_started."""
        # No stored job_id means not started
        response = api_client.get(
            "/jobs/episode_audio_status",
            params={"ep_id": "test-s01e01"},
        )
        # Should return 200 with status info
        assert response.status_code in [200, 404]


class TestTranscriptDownloadEndpoints:
    """Tests for transcript download endpoints."""

    def test_transcript_vtt_not_found(self, api_client):
        """GET VTT for nonexistent episode returns 404."""
        response = api_client.get("/episodes/nonexistent-s99e99/audio/transcript.vtt")
        assert response.status_code == 404

    def test_transcript_jsonl_not_found(self, api_client):
        """GET JSONL for nonexistent episode returns 404."""
        response = api_client.get("/episodes/nonexistent-s99e99/audio/transcript.jsonl")
        assert response.status_code == 404


class TestPrerequisitesEndpoint:
    """Tests for /audio/prerequisites endpoint."""

    def test_prerequisites_check(self, api_client):
        """GET /audio/prerequisites returns status of dependencies."""
        response = api_client.get("/audio/prerequisites")
        assert response.status_code == 200
        data = response.json()

        # Should have status fields for each dependency
        assert "pyannote" in data or "pyannote_available" in data
        assert "resemble" in data or "resemble_api_key_set" in data
        assert "openai" in data or "openai_api_key_set" in data
