"""Test that cluster track representatives return all tracks."""

import json
from pathlib import Path
import pytest

def test_cluster_track_reps_returns_all_tracks(tmp_path):
    """Given a cluster with 13 tracks, track_reps returns 13 items with crop_key and similarity."""
    from apps.api.services.track_reps import (
        write_track_reps,
        write_cluster_centroids,
        build_cluster_track_reps,
    )

    # Setup mock data
    ep_id = "test-s01e01"

    # Create 13 track representatives
    track_reps_list = []
    for i in range(1, 14):
        track_reps_list.append({
            "track_id": f"track_{i:04d}",
            "rep_frame": 100 + i,
            "crop_key": f"crops/track_{i:04d}/frame_{100+i:06d}.jpg",
            "embed": [0.1] * 512,  # Mock embedding
            "quality": {"det": 0.8, "std": 10.0},
        })

    # Create cluster centroids with one cluster containing all 13 tracks
    cluster_centroids = {
        "id_0001": {
            "centroid": [0.1] * 512,
            "tracks": [f"track_{i:04d}" for i in range(1, 14)],
            "cohesion": 0.85,
        }
    }

    # Convert to dict for lookup
    track_reps_map = {rep["track_id"]: rep for rep in track_reps_list}

    # Build cluster track reps
    result = build_cluster_track_reps(ep_id, "id_0001", track_reps_map, cluster_centroids)

    # Assertions
    assert result["cluster_id"] == "id_0001"
    assert result["total_tracks"] == 13, f"Expected 13 tracks, got {result['total_tracks']}"
    assert len(result["tracks"]) == 13, f"Expected 13 track items, got {len(result['tracks'])}"
    assert result["cohesion"] == 0.85

    # Check that each track has required fields
    for track in result["tracks"]:
        assert "track_id" in track
        assert "crop_key" in track
        assert "similarity" in track
        assert track["crop_key"] is not None, f"Track {track['track_id']} missing crop_key"
        assert track["similarity"] is not None, f"Track {track['track_id']} missing similarity"
        assert 0.0 <= track["similarity"] <= 1.0, f"Similarity out of range: {track['similarity']}"


def test_cluster_track_reps_includes_missing_tracks():
    """Tracks without embeddings should still appear in results with None placeholders."""
    from apps.api.services.track_reps import build_cluster_track_reps

    ep_id = "test-s01e01"

    # Only 2 tracks have representatives
    track_reps_map = {
        "track_0001": {
            "track_id": "track_0001",
            "rep_frame": 100,
            "crop_key": "crops/track_0001/frame_000100.jpg",
            "embed": [0.1] * 512,
            "quality": {"det": 0.8, "std": 10.0},
        },
        "track_0002": {
            "track_id": "track_0002",
            "rep_frame": 200,
            "crop_key": "crops/track_0002/frame_000200.jpg",
            "embed": [0.1] * 512,
            "quality": {"det": 0.8, "std": 10.0},
        },
    }

    # Cluster has 4 tracks total
    cluster_centroids = {
        "id_0001": {
            "centroid": [0.1] * 512,
            "tracks": ["track_0001", "track_0002", "track_0003", "track_0004"],
            "cohesion": 0.75,
        }
    }

    result = build_cluster_track_reps(ep_id, "id_0001", track_reps_map, cluster_centroids)

    assert result["total_tracks"] == 4, "Should include all 4 tracks even if some lack embeddings"
    assert len(result["tracks"]) == 4

    # Check that missing tracks have None values
    track_ids = [t["track_id"] for t in result["tracks"]]
    assert "track_0003" in track_ids
    assert "track_0004" in track_ids

    # Find the missing tracks
    missing_tracks = [t for t in result["tracks"] if t["track_id"] in ["track_0003", "track_0004"]]
    for track in missing_tracks:
        assert track["crop_key"] is None
        assert track["similarity"] is None
        assert track["rep_frame"] is None
