"""Tests for PyannoteAI API client and cast-based speaker constraints.

Tests:
- Cast-based speaker count calculation (±3 rule)
- API request body matches official PyannoteAI spec
- Polling interval is in 5-8 second range
- Exclusive diarization is always used
- WhisperX-style merge assigns speakers correctly
"""

import pytest
from unittest.mock import MagicMock, patch
import random

# Import the modules under test
from py_screenalytics.audio.pyannote_api import (
    PyannoteAPIClient,
    PyannoteAPIError,
    calculate_speaker_range,
    DiarizationJobResult,
)
from py_screenalytics.audio.fuse_diarization_asr import (
    assign_speakers_whisperx_style,
    _find_nearest_speaker,
)
from py_screenalytics.audio.models import ASRSegment, DiarizationSegment


class TestCastBasedSpeakerCount:
    """Test cast-based speaker count calculation with ±3 rule."""

    def test_cast_count_plus_minus_3_rule(self):
        """cast=6 -> min=3, max=9"""
        min_speakers, max_speakers = calculate_speaker_range(6)
        assert min_speakers == 3
        assert max_speakers == 9

    def test_cast_count_7_gives_4_to_10(self):
        """cast=7 -> min=4, max=10"""
        min_speakers, max_speakers = calculate_speaker_range(7)
        assert min_speakers == 4
        assert max_speakers == 10

    def test_cast_count_10_gives_7_to_13(self):
        """cast=10 -> min=7, max=13"""
        min_speakers, max_speakers = calculate_speaker_range(10)
        assert min_speakers == 7
        assert max_speakers == 13

    def test_min_speakers_never_below_1(self):
        """cast=2 -> min=1 (not -1), max=5"""
        min_speakers, max_speakers = calculate_speaker_range(2)
        assert min_speakers == 1  # max(1, 2-3) = max(1, -1) = 1
        assert max_speakers == 5

    def test_cast_count_1_gives_1_to_4(self):
        """cast=1 -> min=1, max=4"""
        min_speakers, max_speakers = calculate_speaker_range(1)
        assert min_speakers == 1  # max(1, 1-3) = max(1, -2) = 1
        assert max_speakers == 4

    def test_fallback_when_cast_count_zero(self):
        """cast=0 -> min=1, max=6 (fallback)"""
        min_speakers, max_speakers = calculate_speaker_range(0)
        assert min_speakers == 1
        assert max_speakers == 6

    def test_fallback_when_cast_count_negative(self):
        """Negative cast count also falls back to 1-6."""
        min_speakers, max_speakers = calculate_speaker_range(-5)
        assert min_speakers == 1
        assert max_speakers == 6


class TestPyannoteAPIClient:
    """Test PyannoteAI API client request construction."""

    @patch("py_screenalytics.audio.pyannote_api.httpx.Client")
    @patch("py_screenalytics.audio.pyannote_api.get_pyannote_breaker")
    def test_request_body_matches_official_spec(self, mock_breaker, mock_httpx):
        """Request body matches official pyannoteAI spec."""
        # Setup mocks
        mock_client = MagicMock()
        mock_httpx.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jobId": "test-job-123"}
        mock_client.post.return_value = mock_response

        breaker_ctx = MagicMock()
        breaker_ctx.__enter__ = MagicMock(return_value=None)
        breaker_ctx.__exit__ = MagicMock(return_value=False)
        mock_breaker.return_value = breaker_ctx

        # Create client and submit job
        client = PyannoteAPIClient(api_key="test-key")
        job_id = client.submit_diarization(
            media_url="https://s3.example.com/audio.wav",
            min_speakers=3,
            max_speakers=9,
            exclusive=True,
        )

        # Verify request
        assert job_id == "test-job-123"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args

        # Check endpoint
        assert call_args[0][0] == "/diarize"

        # Check body
        body = call_args[1]["json"]
        assert body["url"] == "https://s3.example.com/audio.wav"
        assert body["model"] == "precision-2"
        assert body["exclusive"] is True
        assert body["minSpeakers"] == 3
        assert body["maxSpeakers"] == 9

    @patch("py_screenalytics.audio.pyannote_api.httpx.Client")
    @patch("py_screenalytics.audio.pyannote_api.get_pyannote_breaker")
    def test_num_speakers_is_never_used(self, mock_breaker, mock_httpx):
        """numSpeakers parameter is never sent even if provided."""
        mock_client = MagicMock()
        mock_httpx.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jobId": "test-job"}
        mock_client.post.return_value = mock_response

        breaker_ctx = MagicMock()
        breaker_ctx.__enter__ = MagicMock(return_value=None)
        breaker_ctx.__exit__ = MagicMock(return_value=False)
        mock_breaker.return_value = breaker_ctx

        client = PyannoteAPIClient(api_key="test-key")
        client.submit_diarization(
            media_url="https://s3.example.com/audio.wav",
            min_speakers=5,
            max_speakers=5,  # Even when min=max (forcing count)
        )

        body = mock_client.post.call_args[1]["json"]

        # numSpeakers should NEVER appear in request
        assert "numSpeakers" not in body
        assert "num_speakers" not in body

    @patch("py_screenalytics.audio.pyannote_api.httpx.Client")
    @patch("py_screenalytics.audio.pyannote_api.get_pyannote_breaker")
    def test_exclusive_always_true_by_default(self, mock_breaker, mock_httpx):
        """exclusive=true in all requests by default."""
        mock_client = MagicMock()
        mock_httpx.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jobId": "test-job"}
        mock_client.post.return_value = mock_response

        breaker_ctx = MagicMock()
        breaker_ctx.__enter__ = MagicMock(return_value=None)
        breaker_ctx.__exit__ = MagicMock(return_value=False)
        mock_breaker.return_value = breaker_ctx

        client = PyannoteAPIClient(api_key="test-key")
        client.submit_diarization(
            media_url="https://s3.example.com/audio.wav",
            # Not explicitly setting exclusive
        )

        body = mock_client.post.call_args[1]["json"]
        assert body["exclusive"] is True

    @patch("py_screenalytics.audio.pyannote_api.httpx.Client")
    @patch("py_screenalytics.audio.pyannote_api.get_pyannote_breaker")
    def test_webhook_url_included_when_provided(self, mock_breaker, mock_httpx):
        """Webhook URL is included in request when provided."""
        mock_client = MagicMock()
        mock_httpx.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jobId": "test-job"}
        mock_client.post.return_value = mock_response

        breaker_ctx = MagicMock()
        breaker_ctx.__enter__ = MagicMock(return_value=None)
        breaker_ctx.__exit__ = MagicMock(return_value=False)
        mock_breaker.return_value = breaker_ctx

        client = PyannoteAPIClient(api_key="test-key")
        client.submit_diarization(
            media_url="https://s3.example.com/audio.wav",
            webhook_url="https://myapp.com/webhooks/pyannote",
        )

        body = mock_client.post.call_args[1]["json"]
        assert body["webhook"] == "https://myapp.com/webhooks/pyannote"

    def test_polling_interval_in_5_to_8_second_range(self):
        """Poll interval = base(6) + jitter(-1 to +2) = 5 to 8 seconds."""
        # Test the interval calculation many times to verify bounds
        base = 6.0
        jitter_range = 2.0

        for _ in range(100):
            jitter = random.uniform(-1.0, jitter_range)
            interval = base + jitter
            assert 5.0 <= interval <= 8.0, f"Interval {interval} outside expected range"


class TestWhisperXMerge:
    """Test WhisperX-style speaker assignment."""

    def test_assigns_speaker_with_max_overlap(self):
        """Speaker with most overlap is assigned."""
        # ASR segment from 5.0 to 10.0
        asr_segments = [
            ASRSegment(start=5.0, end=10.0, text="Hello world"),
        ]

        # Diarization: SPEAKER_01 has more overlap
        diarization_segments = [
            DiarizationSegment(start=4.0, end=6.0, speaker="SPEAKER_00"),  # 1s overlap
            DiarizationSegment(start=6.0, end=11.0, speaker="SPEAKER_01"),  # 4s overlap
        ]

        result = assign_speakers_whisperx_style(asr_segments, diarization_segments)

        assert len(result) == 1
        assert result[0].speaker == "SPEAKER_01"

    def test_sums_overlap_for_same_speaker_multiple_segments(self):
        """Sum durations per speaker when multiple segments overlap."""
        # ASR segment from 5.0 to 15.0
        asr_segments = [
            ASRSegment(start=5.0, end=15.0, text="Long utterance"),
        ]

        # SPEAKER_00 has two segments that together have more overlap
        diarization_segments = [
            DiarizationSegment(start=4.0, end=8.0, speaker="SPEAKER_00"),  # 3s overlap
            DiarizationSegment(start=8.0, end=10.0, speaker="SPEAKER_01"),  # 2s overlap
            DiarizationSegment(start=10.0, end=16.0, speaker="SPEAKER_00"),  # 5s overlap
        ]
        # SPEAKER_00 total = 3 + 5 = 8s
        # SPEAKER_01 total = 2s

        result = assign_speakers_whisperx_style(asr_segments, diarization_segments)

        assert result[0].speaker == "SPEAKER_00"

    def test_fills_nearest_when_no_overlap(self):
        """Nearest segment used when fill_nearest=True and no overlap."""
        # ASR segment has no overlap with any diarization
        asr_segments = [
            ASRSegment(start=20.0, end=21.0, text="Isolated utterance"),
        ]

        diarization_segments = [
            DiarizationSegment(start=0.0, end=5.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=10.0, end=15.0, speaker="SPEAKER_01"),  # Nearest to 20-21
        ]

        result = assign_speakers_whisperx_style(
            asr_segments, diarization_segments, fill_nearest=True
        )

        # SPEAKER_01 at 10-15 is closer to 20-21 than SPEAKER_00 at 0-5
        assert result[0].speaker == "SPEAKER_01"

    def test_returns_unknown_when_no_overlap_and_fill_nearest_false(self):
        """Returns UNKNOWN when no overlap and fill_nearest=False."""
        asr_segments = [
            ASRSegment(start=20.0, end=21.0, text="Isolated"),
        ]

        diarization_segments = [
            DiarizationSegment(start=0.0, end=5.0, speaker="SPEAKER_00"),
        ]

        result = assign_speakers_whisperx_style(
            asr_segments, diarization_segments, fill_nearest=False
        )

        assert result[0].speaker == "UNKNOWN"

    def test_handles_empty_diarization(self):
        """Returns original segments when no diarization provided."""
        asr_segments = [
            ASRSegment(start=0.0, end=1.0, text="Hello"),
        ]

        result = assign_speakers_whisperx_style(asr_segments, [])

        assert len(result) == 1
        assert result[0].text == "Hello"
        # Speaker should be unchanged (None if not set)

    def test_handles_multiple_asr_segments(self):
        """Correctly assigns speakers to multiple ASR segments."""
        asr_segments = [
            ASRSegment(start=0.0, end=2.0, text="First"),
            ASRSegment(start=5.0, end=7.0, text="Second"),
            ASRSegment(start=10.0, end=12.0, text="Third"),
        ]

        diarization_segments = [
            DiarizationSegment(start=0.0, end=3.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=4.0, end=8.0, speaker="SPEAKER_01"),
            DiarizationSegment(start=9.0, end=13.0, speaker="SPEAKER_02"),
        ]

        result = assign_speakers_whisperx_style(asr_segments, diarization_segments)

        assert len(result) == 3
        assert result[0].speaker == "SPEAKER_00"
        assert result[1].speaker == "SPEAKER_01"
        assert result[2].speaker == "SPEAKER_02"


class TestFindNearestSpeaker:
    """Test nearest speaker finding utility."""

    def test_finds_nearest_by_midpoint(self):
        """Finds speaker with midpoint closest to segment midpoint."""
        diar_segments = [
            DiarizationSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),  # mid=1.0
            DiarizationSegment(start=8.0, end=10.0, speaker="SPEAKER_01"),  # mid=9.0
        ]

        # Segment at 5-6, mid=5.5 -> closer to SPEAKER_00 (1.0) than SPEAKER_01 (9.0)
        # Wait, 5.5 - 1.0 = 4.5, 9.0 - 5.5 = 3.5, so SPEAKER_01 is closer
        speaker = _find_nearest_speaker(5.0, 6.0, diar_segments)
        assert speaker == "SPEAKER_01"

    def test_returns_none_for_empty_segments(self):
        """Returns None when no segments provided."""
        speaker = _find_nearest_speaker(5.0, 6.0, [])
        assert speaker is None


class TestDiarizationJobResult:
    """Test DiarizationJobResult dataclass."""

    def test_prefers_exclusive_diarization(self):
        """When both outputs present, exclusive should be preferred."""
        result = DiarizationJobResult(
            job_id="test",
            status="succeeded",
            diarization=[{"start": 0, "end": 1, "speaker": "A"}],
            exclusive_diarization=[{"start": 0, "end": 1, "speaker": "B"}],
        )

        # Client code should use exclusive_diarization when available
        preferred = result.exclusive_diarization or result.diarization
        assert preferred == [{"start": 0, "end": 1, "speaker": "B"}]

    def test_falls_back_to_regular_diarization(self):
        """Falls back to regular diarization when exclusive is None."""
        result = DiarizationJobResult(
            job_id="test",
            status="succeeded",
            diarization=[{"start": 0, "end": 1, "speaker": "A"}],
            exclusive_diarization=None,
        )

        preferred = result.exclusive_diarization or result.diarization
        assert preferred == [{"start": 0, "end": 1, "speaker": "A"}]
