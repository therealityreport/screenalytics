"""Unit tests for track_fusion.json diagnostics + attribution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_track_fusion_diagnostics_reconcile_final_pairs(tmp_path: Path) -> None:
    """If final_pairs > 0, diagnostics must show a non-zero gating or reid path."""
    sys.path.insert(0, str(PROJECT_ROOT))

    from FEATURES.body_tracking.src.track_fusion import TrackFusion, fuse_face_body_tracks

    faces_path = tmp_path / "faces.jsonl"
    body_tracks_path = tmp_path / "body_tracks.jsonl"
    output_path = tmp_path / "track_fusion.json"

    # Face track 1 overlaps body track 100001 in 3 frames → should create one fused pair.
    _write_jsonl(
        faces_path,
        [
            {"track_id": 1, "frame_idx": 0, "ts": 0.0, "bbox_xyxy": [0, 0, 5, 5]},
            {"track_id": 1, "frame_idx": 1, "ts": 0.1, "bbox_xyxy": [0, 0, 5, 5]},
            {"track_id": 1, "frame_idx": 2, "ts": 0.2, "bbox_xyxy": [0, 0, 5, 5]},
        ],
    )

    body_track_row = {
        "track_id": 100001,
        "start_frame": 0,
        "end_frame": 2,
        "detections": [
            {"frame_idx": 0, "bbox": [0, 0, 10, 10]},
            {"frame_idx": 1, "bbox": [0, 0, 10, 10]},
            {"frame_idx": 2, "bbox": [0, 0, 10, 10]},
        ],
    }
    _write_jsonl(body_tracks_path, [body_track_row])

    fusion = TrackFusion(iou_threshold=0.02, min_overlap_ratio=0.7, face_in_upper_body=True, upper_body_fraction=0.5)
    fuse_face_body_tracks(
        fusion=fusion,
        face_tracks_path=faces_path,
        body_tracks_path=body_tracks_path,
        output_path=output_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    diagnostics = payload.get("diagnostics")
    assert isinstance(diagnostics, dict)

    final_pairs = diagnostics.get("final_pairs")
    assert final_pairs == 1

    iou_pass = diagnostics.get("iou_pass")
    reid_pass = diagnostics.get("reid_pass")
    assert isinstance(iou_pass, int)
    assert isinstance(reid_pass, int)
    assert iou_pass > 0 or reid_pass > 0

    # Distribution stats must be recorded (not N/A) for non-empty overlap candidates.
    iou_dist = diagnostics.get("iou_distribution")
    overlap_dist = diagnostics.get("overlap_ratio_distribution")
    assert isinstance(iou_dist, dict)
    assert isinstance(overlap_dist, dict)
    assert iou_dist.get("max", 0) > 0
    assert overlap_dist.get("max", 0) > 0

    identities = payload.get("identities")
    assert isinstance(identities, dict)
    assert identities, "expected fused identities"
    any_attributed = False
    for identity in identities.values():
        if not isinstance(identity, dict):
            continue
        attribution = identity.get("attribution")
        if not isinstance(attribution, dict):
            continue
        if attribution.get("source") == "iou":
            any_attributed = True
            assert isinstance(attribution.get("best_iou"), (int, float))
    assert any_attributed, "expected at least one IoU-attributed fused identity"


def test_track_fusion_records_reid_comparisons_and_attribution() -> None:
    """Re-ID path must record comparisons and attribute fused identities."""
    sys.path.insert(0, str(PROJECT_ROOT))

    import numpy as np

    from FEATURES.body_tracking.src.track_fusion import TrackFusion

    face_tracks = {
        1: {
            "track_id": 1,
            "detections": [],
            "start_frame": 0,
            "end_frame": 10,
        }
    }
    body_tracks = {
        100001: {
            "track_id": 100001,
            "detections": [],
            "start_frame": 12,
            "end_frame": 20,
        }
    }

    # 1 face embedding for face track 1.
    face_embeddings = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    face_embeddings_meta = [{"track_id": 1}]

    # 1 body embedding row for body track 100001 (perfect match).
    body_embeddings = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    body_embeddings_meta = [{"track_id": 100001, "frame_idx": 12}]

    fusion = TrackFusion(reid_similarity_threshold=0.70, max_gap_seconds=30.0)
    identities = fusion.fuse_tracks(
        face_tracks=face_tracks,
        body_tracks=body_tracks,
        body_embeddings=body_embeddings,
        body_embeddings_meta=body_embeddings_meta,
        face_embeddings=face_embeddings,
        face_embeddings_meta=face_embeddings_meta,
    )

    diagnostics = getattr(fusion, "last_diagnostics", {})
    assert diagnostics.get("reid_comparisons") == 1
    assert diagnostics.get("reid_pass") == 1
    assert diagnostics.get("final_pairs") == 1
    assert diagnostics.get("reid_pairs") == 1
    assert diagnostics.get("iou_pairs") == 0

    assert identities, "expected fused identities via re-id"
    any_reid_attributed = False
    for identity in identities.values():
        payload = identity.to_dict()
        attribution = payload.get("attribution")
        if isinstance(attribution, dict) and attribution.get("source") == "reid":
            any_reid_attributed = True
            assert isinstance(attribution.get("best_similarity"), (int, float))
    assert any_reid_attributed, "expected at least one Re-ID attributed fused identity"
