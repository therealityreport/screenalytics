"""Tests for diarization + ASR fusion.

Verifies:
1. Transcript rows include all speaker/voice fields
2. Mapping is consistent with voice_clusters and voice_mapping
3. VTT includes speaker metadata in NOTE lines
4. JSONL format is correct
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestFuseTranscript:
    """Tests for fuse_transcript function."""

    def test_transcript_rows_have_required_fields(self):
        """All transcript rows have speaker_id, speaker_display_name, voice_cluster_id, voice_bank_id."""
        from py_screenalytics.audio.fuse_diarization_asr import fuse_transcript
        from py_screenalytics.audio.models import (
            ASRSegment,
            DiarizationSegment,
            VoiceBankMatchResult,
            VoiceCluster,
            VoiceClusterSegment,
            WordTiming,
        )

        diarization_segments = [
            DiarizationSegment(start=0.0, end=5.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=5.0, end=10.0, speaker="SPEAKER_01"),
        ]

        asr_segments = [
            ASRSegment(
                start=0.0,
                end=4.5,
                text="Hello, how are you?",
                confidence=0.95,
                words=[
                    WordTiming(w="Hello", t0=0.0, t1=0.5),
                    WordTiming(w="how", t0=0.6, t1=0.8),
                    WordTiming(w="are", t0=0.9, t1=1.0),
                    WordTiming(w="you", t0=1.1, t1=1.3),
                ],
            ),
            ASRSegment(
                start=5.5,
                end=9.0,
                text="I'm doing great!",
                confidence=0.92,
            ),
        ]

        voice_clusters = [
            VoiceCluster(
                voice_cluster_id="VC_01",
                segments=[VoiceClusterSegment(start=0.0, end=5.0, diar_speaker="SPEAKER_00")],
                total_duration=5.0,
                segment_count=1,
            ),
            VoiceCluster(
                voice_cluster_id="VC_02",
                segments=[VoiceClusterSegment(start=5.0, end=10.0, diar_speaker="SPEAKER_01")],
                total_duration=5.0,
                segment_count=1,
            ),
        ]

        voice_mapping = [
            VoiceBankMatchResult(
                voice_cluster_id="VC_01",
                voice_bank_id="voice_lisa_barlow",
                speaker_id="SPK_LISA_BARLOW",
                speaker_display_name="Lisa Barlow",
                similarity=0.89,
            ),
            VoiceBankMatchResult(
                voice_cluster_id="VC_02",
                voice_bank_id="voice_unlabeled_01",
                speaker_id="SPK_UNLABELED_01",
                speaker_display_name="Unlabeled Voice 1",
                similarity=None,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "transcript.jsonl"
            vtt_path = Path(tmpdir) / "transcript.vtt"

            rows = fuse_transcript(
                diarization_segments,
                asr_segments,
                voice_clusters,
                voice_mapping,
                None,
                jsonl_path,
                vtt_path,
                include_speaker_notes=True,
                overwrite=True,
            )

            # All rows must have required fields
            for row in rows:
                assert row.speaker_id, f"Row missing speaker_id: {row}"
                assert row.speaker_display_name, f"Row missing speaker_display_name: {row}"
                assert row.voice_cluster_id, f"Row missing voice_cluster_id: {row}"
                assert row.voice_bank_id, f"Row missing voice_bank_id: {row}"

            # Check specific values
            assert rows[0].speaker_id == "SPK_LISA_BARLOW"
            assert rows[0].voice_bank_id == "voice_lisa_barlow"

    def test_jsonl_output_format(self):
        """JSONL file has correct format with all fields."""
        from py_screenalytics.audio.fuse_diarization_asr import fuse_transcript
        from py_screenalytics.audio.models import (
            ASRSegment,
            DiarizationSegment,
            VoiceBankMatchResult,
            VoiceCluster,
            VoiceClusterSegment,
        )

        diarization_segments = [
            DiarizationSegment(start=0.0, end=5.0, speaker="SPEAKER_00"),
        ]

        asr_segments = [
            ASRSegment(start=0.5, end=4.0, text="Test message", confidence=0.90),
        ]

        voice_clusters = [
            VoiceCluster(
                voice_cluster_id="VC_01",
                segments=[VoiceClusterSegment(start=0.0, end=5.0, diar_speaker="SPEAKER_00")],
                total_duration=5.0,
                segment_count=1,
            ),
        ]

        voice_mapping = [
            VoiceBankMatchResult(
                voice_cluster_id="VC_01",
                voice_bank_id="voice_test",
                speaker_id="SPK_TEST",
                speaker_display_name="Test Speaker",
                similarity=0.85,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "transcript.jsonl"
            vtt_path = Path(tmpdir) / "transcript.vtt"

            fuse_transcript(
                diarization_segments,
                asr_segments,
                voice_clusters,
                voice_mapping,
                None,
                jsonl_path,
                vtt_path,
                overwrite=True,
            )

            # Verify JSONL content
            with jsonl_path.open("r") as f:
                lines = f.readlines()

            assert len(lines) >= 1

            row = json.loads(lines[0])
            assert "start" in row
            assert "end" in row
            assert "text" in row
            assert row["speaker_id"] == "SPK_TEST"
            assert row["speaker_display_name"] == "Test Speaker"
            assert row["voice_cluster_id"] == "VC_01"
            assert row["voice_bank_id"] == "voice_test"

    def test_vtt_output_includes_speaker_notes(self):
        """VTT file includes NOTE lines with speaker metadata."""
        from py_screenalytics.audio.fuse_diarization_asr import fuse_transcript
        from py_screenalytics.audio.models import (
            ASRSegment,
            DiarizationSegment,
            VoiceBankMatchResult,
            VoiceCluster,
            VoiceClusterSegment,
        )

        diarization_segments = [
            DiarizationSegment(start=0.0, end=5.0, speaker="SPEAKER_00"),
        ]

        asr_segments = [
            ASRSegment(start=0.5, end=4.0, text="Hello world", confidence=0.95),
        ]

        voice_clusters = [
            VoiceCluster(
                voice_cluster_id="VC_01",
                segments=[VoiceClusterSegment(start=0.0, end=5.0, diar_speaker="SPEAKER_00")],
                total_duration=5.0,
                segment_count=1,
            ),
        ]

        voice_mapping = [
            VoiceBankMatchResult(
                voice_cluster_id="VC_01",
                voice_bank_id="voice_lisa",
                speaker_id="SPK_LISA",
                speaker_display_name="Lisa",
                similarity=0.90,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "transcript.jsonl"
            vtt_path = Path(tmpdir) / "transcript.vtt"

            fuse_transcript(
                diarization_segments,
                asr_segments,
                voice_clusters,
                voice_mapping,
                None,
                jsonl_path,
                vtt_path,
                include_speaker_notes=True,
                overwrite=True,
            )

            # Verify VTT content
            vtt_content = vtt_path.read_text()

            assert "WEBVTT" in vtt_content
            assert "NOTE speaker_id=SPK_LISA" in vtt_content
            assert 'speaker_display_name="Lisa"' in vtt_content
            assert "voice_cluster_id=VC_01" in vtt_content
            assert "voice_bank_id=voice_lisa" in vtt_content
            assert "<v Lisa>Hello world" in vtt_content

    def test_rows_split_when_speaker_changes(self):
        """Transcript rows split when words have different speaker_ids."""
        from py_screenalytics.audio.fuse_diarization_asr import fuse_transcript
        from py_screenalytics.audio.models import (
            ASRSegment,
            DiarizationSegment,
            VoiceBankMatchResult,
            VoiceCluster,
            VoiceClusterSegment,
            WordTiming,
        )

        diarization_segments = [
            DiarizationSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=2.0, end=4.0, speaker="SPEAKER_01"),
        ]

        asr_segments = [
            ASRSegment(
                start=0.0,
                end=4.0,
                text="hello there friend",
                confidence=0.9,
                words=[
                    WordTiming(w="hello", t0=0.0, t1=0.5),
                    WordTiming(w="there", t0=0.6, t1=1.0),
                    WordTiming(w="friend", t0=2.2, t1=2.8),
                ],
            )
        ]

        voice_clusters = [
            VoiceCluster(
                voice_cluster_id="VC_01",
                segments=[VoiceClusterSegment(start=0.0, end=2.0, diar_speaker="SPEAKER_00")],
                speaker_group_ids=["pyannote:SPEAKER_00"],
                total_duration=2.0,
                segment_count=1,
            ),
            VoiceCluster(
                voice_cluster_id="VC_02",
                segments=[VoiceClusterSegment(start=2.0, end=4.0, diar_speaker="SPEAKER_01")],
                speaker_group_ids=["pyannote:SPEAKER_01"],
                total_duration=2.0,
                segment_count=1,
            ),
        ]

        voice_mapping = [
            VoiceBankMatchResult(
                voice_cluster_id="VC_01",
                voice_bank_id="voice_a",
                speaker_id="SPK_A",
                speaker_display_name="Speaker A",
                similarity=0.8,
            ),
            VoiceBankMatchResult(
                voice_cluster_id="VC_02",
                voice_bank_id="voice_b",
                speaker_id="SPK_B",
                speaker_display_name="Speaker B",
                similarity=0.8,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "transcript.jsonl"
            vtt_path = Path(tmpdir) / "transcript.vtt"

            rows = fuse_transcript(
                diarization_segments,
                asr_segments,
                voice_clusters,
                voice_mapping,
                None,
                jsonl_path,
                vtt_path,
                include_speaker_notes=True,
                overwrite=True,
                diarization_source="pyannote",
            )

            assert len(rows) == 2, "Expected split rows when speakers change within ASR segment"
            assert rows[0].speaker_id != rows[1].speaker_id


class TestTimestampFormatting:
    """Tests for timestamp formatting utilities."""

    def test_seconds_to_vtt_time(self):
        """VTT time format is correct."""
        from py_screenalytics.audio.fuse_diarization_asr import _seconds_to_vtt_time

        assert _seconds_to_vtt_time(0.0) == "00:00:00.000"
        assert _seconds_to_vtt_time(1.5) == "00:00:01.500"
        assert _seconds_to_vtt_time(61.234) == "00:01:01.234"
        assert _seconds_to_vtt_time(3661.0) == "01:01:01.000"

    def test_seconds_to_srt_time(self):
        """SRT time format is correct."""
        from py_screenalytics.audio.fuse_diarization_asr import _seconds_to_srt_time

        assert _seconds_to_srt_time(0.0) == "00:00:00,000"
        assert _seconds_to_srt_time(1.5) == "00:00:01,500"
        assert _seconds_to_srt_time(61.234) == "00:01:01,234"
