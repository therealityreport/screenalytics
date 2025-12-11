"""End-to-end tests for the audio pipeline.

Verifies:
1. Full pipeline produces all required artifacts
2. Transcript rows have speaker/voice fields
3. QC status is at least warn (not needs_review)
4. Voice clusters and mappings are consistent
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestAudioPipelineE2E:
    """End-to-end tests for the audio pipeline."""

    @pytest.fixture
    def mock_audio_file(self):
        """Create a minimal mock audio file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create video directory structure
            video_dir = Path(tmpdir) / "videos" / "test-s01e01"
            video_dir.mkdir(parents=True)

            # Create a placeholder video file
            video_path = video_dir / "episode.mp4"
            video_path.write_bytes(b"MOCK_VIDEO")

            yield tmpdir, video_path

    @patch("py_screenalytics.audio.io.extract_audio_from_video")
    @patch("py_screenalytics.audio.separation_mdx.separate_vocals")
    @patch("py_screenalytics.audio.enhance_resemble.check_api_available")
    @patch("py_screenalytics.audio.enhance_resemble.enhance_audio_local")
    @patch("py_screenalytics.audio.diarization_nemo.run_diarization_nemo")
    @patch("py_screenalytics.audio.voice_clusters.cluster_episode_voices")
    @patch("py_screanalytics.audio.voice_bank.match_voice_clusters_to_bank")
    @patch("py_screanalytics.audio.asr_openai.transcribe_audio")
    @patch("py_screanalytics.audio.fuse_diarization_asr.fuse_transcript")
    @patch("py_screanalytics.audio.export.export_final_audio")
    @patch("py_screanalytics.audio.io.get_audio_duration")
    @patch("py_screanalytics.audio.io.compute_snr")
    def test_pipeline_produces_all_artifacts(
        self,
        mock_compute_snr,
        mock_get_duration,
        mock_export,
        mock_fuse,
        mock_transcribe,
        mock_match_bank,
        mock_cluster,
        mock_diarize,
        mock_enhance,
        mock_check_api,
        mock_separate,
        mock_extract,
        mock_audio_file,
    ):
        """Pipeline produces all required artifacts."""
        from py_screenalytics.audio.episode_audio_pipeline import run_episode_audio_pipeline
        from py_screenalytics.audio.models import (
            ASRSegment,
            DiarizationSegment,
            TranscriptRow,
            VoiceBankMatchResult,
            VoiceCluster,
            VoiceClusterSegment,
            QCStatus,
        )

        tmpdir, video_path = mock_audio_file

        # Setup mocks
        audio_dir = Path(tmpdir) / "audio" / "test-s01e01"
        manifests_dir = Path(tmpdir) / "manifests" / "test-s01e01"
        audio_dir.mkdir(parents=True)
        manifests_dir.mkdir(parents=True)

        # Mock audio extraction
        original_path = audio_dir / "episode_original.wav"
        original_path.write_bytes(b"MOCK_AUDIO")
        mock_extract.return_value = (
            original_path,
            MagicMock(duration_seconds=3600.0, sample_rate=48000),
        )

        # Mock separation
        vocals_path = audio_dir / "episode_vocals.wav"
        vocals_path.write_bytes(b"MOCK_VOCALS")
        mock_separate.return_value = (vocals_path, audio_dir / "episode_accompaniment.wav")

        # Mock enhancement
        mock_check_api.return_value = False
        enhanced_path = audio_dir / "episode_vocals_enhanced.wav"
        enhanced_path.write_bytes(b"MOCK_ENHANCED")
        mock_enhance.return_value = enhanced_path

        # Mock diarization
        diar_segments = [
            DiarizationSegment(start=0.0, end=30.0, speaker="SPEAKER_00", confidence=0.95),
            DiarizationSegment(start=30.0, end=60.0, speaker="SPEAKER_01", confidence=0.90),
        ]
        mock_diarize.return_value = diar_segments

        # Mock voice clustering
        voice_clusters = [
            VoiceCluster(
                voice_cluster_id="VC_01",
                segments=[VoiceClusterSegment(start=0.0, end=30.0, diar_speaker="SPEAKER_00")],
                total_duration=30.0,
                segment_count=1,
                centroid=[0.1] * 256,
            ),
            VoiceCluster(
                voice_cluster_id="VC_02",
                segments=[VoiceClusterSegment(start=30.0, end=60.0, diar_speaker="SPEAKER_01")],
                total_duration=30.0,
                segment_count=1,
                centroid=[0.2] * 256,
            ),
        ]
        mock_cluster.return_value = voice_clusters

        # Mock voice bank mapping - one labeled, one unlabeled
        voice_mapping = [
            VoiceBankMatchResult(
                voice_cluster_id="VC_01",
                voice_bank_id="voice_test_cast",
                speaker_id="SPK_TEST_CAST",
                speaker_display_name="Test Cast",
                similarity=0.85,
            ),
            VoiceBankMatchResult(
                voice_cluster_id="VC_02",
                voice_bank_id="voice_unlabeled_01",
                speaker_id="SPK_UNLABELED_01",
                speaker_display_name="Unlabeled Voice 1",
                similarity=None,
            ),
        ]
        mock_match_bank.return_value = voice_mapping

        # Mock ASR
        asr_segments = [
            ASRSegment(start=0.0, end=15.0, text="Hello from speaker one", confidence=0.95),
            ASRSegment(start=30.0, end=45.0, text="And hello from speaker two", confidence=0.92),
        ]
        mock_transcribe.return_value = asr_segments

        # Mock fusion
        transcript_rows = [
            TranscriptRow(
                start=0.0,
                end=15.0,
                text="Hello from speaker one",
                speaker_id="SPK_TEST_CAST",
                speaker_display_name="Test Cast",
                voice_cluster_id="VC_01",
                voice_bank_id="voice_test_cast",
                conf=0.95,
            ),
            TranscriptRow(
                start=30.0,
                end=45.0,
                text="And hello from speaker two",
                speaker_id="SPK_UNLABELED_01",
                speaker_display_name="Unlabeled Voice 1",
                voice_cluster_id="VC_02",
                voice_bank_id="voice_unlabeled_01",
                conf=0.92,
            ),
        ]
        mock_fuse.return_value = transcript_rows

        # Mock export
        final_path = audio_dir / "episode_final_voice_only.wav"
        final_path.write_bytes(b"MOCK_FINAL")
        mock_export.return_value = final_path

        # Mock duration and SNR
        mock_get_duration.return_value = 3600.0
        mock_compute_snr.return_value = 22.0

        # Run the pipeline
        result = run_episode_audio_pipeline(
            ep_id="test-s01e01",
            overwrite=True,
            data_root=Path(tmpdir),
        )

        # Verify result
        assert result.status == "succeeded"
        assert result.qc_status in [QCStatus.OK, QCStatus.WARN]

        # Check voice stats
        assert result.voice_cluster_count == 2
        assert result.labeled_voices == 1
        assert result.unlabeled_voices == 1
        assert result.transcript_row_count == 2

        # Verify artifacts
        assert result.audio_artifacts.original is not None
        assert result.audio_artifacts.vocals is not None
        assert result.audio_artifacts.vocals_enhanced is not None
        assert result.audio_artifacts.final_voice_only is not None

        assert result.manifest_artifacts.diarization is not None
        assert result.manifest_artifacts.voice_clusters is not None
        assert result.manifest_artifacts.voice_mapping is not None
        assert result.manifest_artifacts.transcript_jsonl is not None
        assert result.manifest_artifacts.transcript_vtt is not None
        assert result.manifest_artifacts.qc is not None

    def test_pipeline_prerequisites_check(self):
        """Prerequisites check returns expected structure."""
        from py_screenalytics.audio.episode_audio_pipeline import check_pipeline_prerequisites

        status = check_pipeline_prerequisites()

        assert "ffmpeg" in status
        assert "soundfile" in status
        assert "demucs" in status
        assert "pyannote" in status
        assert "openai" in status
        assert "gemini" in status
        assert "resemble" in status


class TestArtifactConsistency:
    """Tests for artifact consistency validation."""

    def test_transcript_cluster_ids_in_clusters(self):
        """All transcript voice_cluster_ids exist in voice_clusters.json."""
        from py_screanalytics.audio.models import TranscriptRow, VoiceCluster, VoiceClusterSegment

        voice_clusters = [
            VoiceCluster(
                voice_cluster_id="VC_01",
                segments=[VoiceClusterSegment(start=0.0, end=10.0, diar_speaker="SPEAKER_00")],
                total_duration=10.0,
                segment_count=1,
            ),
            VoiceCluster(
                voice_cluster_id="VC_02",
                segments=[VoiceClusterSegment(start=10.0, end=20.0, diar_speaker="SPEAKER_01")],
                total_duration=10.0,
                segment_count=1,
            ),
        ]

        transcript_rows = [
            TranscriptRow(
                start=0.0,
                end=5.0,
                text="Test",
                speaker_id="SPK_01",
                speaker_display_name="Speaker 1",
                voice_cluster_id="VC_01",
                voice_bank_id="voice_01",
            ),
            TranscriptRow(
                start=10.0,
                end=15.0,
                text="Test 2",
                speaker_id="SPK_02",
                speaker_display_name="Speaker 2",
                voice_cluster_id="VC_02",
                voice_bank_id="voice_02",
            ),
        ]

        cluster_ids = {c.voice_cluster_id for c in voice_clusters}
        transcript_cluster_ids = {r.voice_cluster_id for r in transcript_rows}

        # All transcript cluster IDs should be in voice_clusters
        assert transcript_cluster_ids.issubset(cluster_ids)

    def test_transcript_cluster_ids_in_mapping(self):
        """All transcript voice_cluster_ids exist in voice_mapping.json."""
        from py_screenalytics.audio.models import TranscriptRow, VoiceBankMatchResult

        voice_mapping = [
            VoiceBankMatchResult(
                voice_cluster_id="VC_01",
                voice_bank_id="voice_01",
                speaker_id="SPK_01",
                speaker_display_name="Speaker 1",
            ),
            VoiceBankMatchResult(
                voice_cluster_id="VC_02",
                voice_bank_id="voice_02",
                speaker_id="SPK_02",
                speaker_display_name="Speaker 2",
            ),
        ]

        transcript_rows = [
            TranscriptRow(
                start=0.0,
                end=5.0,
                text="Test",
                speaker_id="SPK_01",
                speaker_display_name="Speaker 1",
                voice_cluster_id="VC_01",
                voice_bank_id="voice_01",
            ),
        ]

        mapping_cluster_ids = {m.voice_cluster_id for m in voice_mapping}
        transcript_cluster_ids = {r.voice_cluster_id for r in transcript_rows}

        # All transcript cluster IDs should be in voice_mapping
        assert transcript_cluster_ids.issubset(mapping_cluster_ids)
