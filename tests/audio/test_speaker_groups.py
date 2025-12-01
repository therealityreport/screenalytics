"""Tests for speaker group manifest creation and smart split editing."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest


def test_build_speaker_groups_manifest(tmp_path):
    """Speaker groups manifest captures per-source groups and summaries."""
    from py_screenalytics.audio.models import DiarizationSegment
    from py_screenalytics.audio.speaker_groups import build_speaker_groups_manifest

    manifest_path = tmp_path / "audio_speaker_groups.json"
    segments = [
        DiarizationSegment(start=0.0, end=5.0, speaker="PY_SPK_00"),
        DiarizationSegment(start=6.0, end=8.0, speaker="PY_SPK_01"),
        DiarizationSegment(start=9.0, end=12.0, speaker="PY_SPK_00"),
    ]

    manifest = build_speaker_groups_manifest(
        ep_id="test-s01e01",
        source_segments={"pyannote": segments},
        output_path=manifest_path,
        overwrite=True,
    )

    assert manifest_path.exists()
    assert manifest.sources[0].summary.speakers == 2
    assert manifest.sources[0].summary.segments == 3

    data = json.loads(manifest_path.read_text())
    first_group = data["sources"][0]["speakers"][0]
    assert first_group["speaker_group_id"].startswith("pyannote:")
    assert first_group["segments"][0]["segment_id"].startswith("py_")


def test_smart_split_segment_reassigns_voice(tmp_path, monkeypatch):
    """Smart split reassigns subsegments to a better-matching speaker group."""
    from py_screenalytics.audio.diarization_pyannote import _save_diarization_manifest
    from py_screenalytics.audio.models import DiarizationSegment, VoiceBankMatchResult, VoiceCluster, VoiceClusterSegment
    from py_screenalytics.audio.speaker_edit import smart_split_segment
    from py_screenalytics.audio.speaker_groups import build_speaker_groups_manifest

    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "test-s01e01"

    audio_dir = tmp_path / "audio" / ep_id
    manifests_dir = tmp_path / "manifests" / ep_id
    audio_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True)

    # Minimal audio placeholder
    (audio_dir / "episode_vocals.wav").write_bytes(b"AUDIO")

    # Build diarization + speaker groups (two speakers, one segment to split)
    diar_segments = [
        DiarizationSegment(start=0.0, end=4.0, speaker="SPEAKER_00"),
        DiarizationSegment(start=4.5, end=6.0, speaker="SPEAKER_01"),
    ]
    manifest_path = manifests_dir / "audio_speaker_groups.json"
    build_speaker_groups_manifest(
        ep_id,
        {"pyannote": diar_segments},
        manifest_path,
        overwrite=True,
    )
    _save_diarization_manifest(diar_segments, manifests_dir / "audio_diarization_pyannote.jsonl")
    shutil.copy(manifests_dir / "audio_diarization_pyannote.jsonl", manifests_dir / "audio_diarization.jsonl")
    (manifests_dir / "audio_asr_raw.jsonl").write_text("", encoding="utf-8")

    # Patch heavy operations
    def fake_centroids(audio_path, manifest, model):
        return {
            "pyannote:SPEAKER_00": np.array([1.0, 0.0]),
            "pyannote:SPEAKER_01": np.array([0.0, 1.0]),
        }

    monkeypatch.setattr("py_screenalytics.audio.speaker_edit.compute_group_centroids", fake_centroids)

    def fake_extract(audio_path, segments, model):
        results = []
        for seg in segments:
            vec = np.array([1.0, 0.0]) if seg.start < 2.0 else np.array([0.0, 1.0])
            results.append((seg, vec))
        return results

    monkeypatch.setattr("py_screenalytics.audio.speaker_edit.extract_speaker_embeddings", fake_extract)

    def fake_cluster(audio_path, diarization_segments, output_path, config, overwrite=False, speaker_groups_manifest=None):
        return [
            VoiceCluster(
                voice_cluster_id="VC_01",
                speaker_group_ids=["pyannote:SPEAKER_00"],
                sources=[],
                segments=[VoiceClusterSegment(start=0.0, end=4.0, diar_speaker="SPEAKER_00")],
                total_duration=4.0,
                segment_count=1,
            ),
            VoiceCluster(
                voice_cluster_id="VC_02",
                speaker_group_ids=["pyannote:SPEAKER_01"],
                sources=[],
                segments=[VoiceClusterSegment(start=4.5, end=6.0, diar_speaker="SPEAKER_01")],
                total_duration=1.5,
                segment_count=1,
            ),
        ]

    monkeypatch.setattr("py_screenalytics.audio.speaker_edit.cluster_episode_voices", fake_cluster)

    def fake_match(show_id, clusters, output_path, config=None, similarity_threshold=0.0, overwrite=False):
        return [
            VoiceBankMatchResult(
                voice_cluster_id=cl.voice_cluster_id,
                voice_bank_id=f"voice_{cl.voice_cluster_id.lower()}",
                speaker_id=f"SPK_{cl.voice_cluster_id}",
                speaker_display_name=cl.voice_cluster_id,
                similarity=0.9,
            )
            for cl in clusters
        ]

    monkeypatch.setattr("py_screenalytics.audio.speaker_edit.match_voice_clusters_to_bank", fake_match)
    monkeypatch.setattr("py_screenalytics.audio.speaker_edit.fuse_transcript", lambda *args, **kwargs: [])

    result = smart_split_segment(
        ep_id=ep_id,
        source="pyannote",
        speaker_group_id="pyannote:SPEAKER_00",
        segment_id="py_0001",
        expected_voices=2,
    )

    assert len(result.new_segments) >= 2
    # At least one segment should be reassigned away from the original group
    reassigned = [seg for seg in result.new_segments if seg.speaker_group_id != "pyannote:SPEAKER_00"]
    assert reassigned, "Expected a subsegment to be moved to a different speaker group"
