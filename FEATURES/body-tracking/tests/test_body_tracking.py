"""
Tests for body tracking pipeline.
"""

import json
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

from FEATURES.body_tracking.src.detect_bodies import BodyDetection, BodyDetector
from FEATURES.body_tracking.src.track_bodies import (
    BodyTrack,
    BodyTracker,
    SimpleIoUTracker,
    track_bodies,
)
from FEATURES.body_tracking.src.track_fusion import (
    FaceBodyAssociation,
    FusedIdentity,
    TrackFusion,
)
from FEATURES.body_tracking.src.screentime_compare import (
    ScreenTimeBreakdown,
    ScreenTimeComparator,
    TimeSegment,
)


class TestBodyDetection:
    """Tests for body detection."""

    def test_body_detection_dataclass(self):
        """Test BodyDetection serialization."""
        det = BodyDetection(
            frame_idx=100,
            timestamp=4.16,
            bbox=[100.0, 200.0, 300.0, 500.0],
            score=0.95,
            class_id=0,
        )

        d = det.to_dict()
        assert d["frame_idx"] == 100
        assert d["timestamp"] == 4.16
        assert d["bbox"] == [100.0, 200.0, 300.0, 500.0]
        assert d["score"] == 0.95

        # Round-trip
        det2 = BodyDetection.from_dict(d)
        assert det2.frame_idx == det.frame_idx
        assert det2.bbox == det.bbox

    def test_body_detector_init(self):
        """Test BodyDetector initialization (no model load)."""
        detector = BodyDetector(
            model_name="yolov8n",
            confidence_threshold=0.5,
            device="cpu",
        )
        assert detector.model_name == "yolov8n"
        assert detector.confidence_threshold == 0.5
        assert detector._model is None  # Lazy load


class TestBodyTracking:
    """Tests for body tracking."""

    def test_body_track_properties(self):
        """Test BodyTrack computed properties."""
        dets = [
            BodyDetection(frame_idx=10, timestamp=0.4, bbox=[0, 0, 100, 200], score=0.9),
            BodyDetection(frame_idx=11, timestamp=0.45, bbox=[5, 0, 105, 200], score=0.85),
            BodyDetection(frame_idx=12, timestamp=0.5, bbox=[10, 0, 110, 200], score=0.88),
        ]
        track = BodyTrack(track_id=1, detections=dets)

        assert track.start_frame == 10
        assert track.end_frame == 12
        assert track.frame_count == 3
        assert abs(track.duration - 0.1) < 0.01

    def test_simple_iou_tracker(self):
        """Test SimpleIoUTracker."""
        tracker = SimpleIoUTracker(iou_threshold=0.3, max_age=5)

        # Frame 0: Two persons
        bboxes_0 = np.array([
            [100, 100, 200, 300],  # Person A
            [400, 100, 500, 300],  # Person B
        ])
        scores_0 = np.array([0.9, 0.85])
        ids_0 = tracker.update(0, bboxes_0, scores_0)
        assert len(ids_0) == 2
        assert ids_0[0] != ids_0[1]

        # Frame 1: Persons moved slightly
        bboxes_1 = np.array([
            [105, 100, 205, 300],  # Person A moved
            [405, 100, 505, 300],  # Person B moved
        ])
        scores_1 = np.array([0.88, 0.82])
        ids_1 = tracker.update(1, bboxes_1, scores_1)
        assert ids_1[0] == ids_0[0]  # Same person A
        assert ids_1[1] == ids_0[1]  # Same person B

        # Frame 2: Person A gone, new person C
        bboxes_2 = np.array([
            [405, 100, 505, 300],  # Person B
            [600, 100, 700, 300],  # Person C (new)
        ])
        scores_2 = np.array([0.9, 0.8])
        ids_2 = tracker.update(2, bboxes_2, scores_2)
        assert ids_2[0] == ids_0[1]  # Still person B
        assert ids_2[1] not in [ids_0[0], ids_0[1]]  # New person C

    def test_iou_computation(self):
        """Test IoU computation."""
        tracker = SimpleIoUTracker()

        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 150, 150])  # 50% overlap

        iou = tracker._compute_iou(box1, box2)
        # Intersection: 50x50=2500, Union: 2*10000 - 2500 = 17500
        expected_iou = 2500 / 17500
        assert abs(iou - expected_iou) < 0.01

    def test_body_tracker_init(self):
        """Test BodyTracker initialization."""
        tracker = BodyTracker(
            tracker_type="bytetrack",
            track_thresh=0.5,
            track_buffer=60,
            id_offset=100000,
        )
        assert tracker.tracker_type == "bytetrack"
        assert tracker.id_offset == 100000


class TestTrackFusion:
    """Tests for face-body track fusion."""

    def test_face_in_body_score(self):
        """Test face-in-body scoring."""
        fusion = TrackFusion(min_overlap_ratio=0.7)

        # Face fully inside body upper region
        face_box = [150, 50, 250, 150]
        body_box = [100, 0, 300, 400]
        score = fusion.compute_face_in_body_score(face_box, body_box)
        assert score >= 0.9  # High score

        # Face outside body
        face_box = [500, 50, 600, 150]
        body_box = [100, 0, 300, 400]
        score = fusion.compute_face_in_body_score(face_box, body_box)
        assert score == 0.0

        # Face in lower body (should be penalized)
        face_box = [150, 300, 250, 400]
        body_box = [100, 0, 300, 400]
        score = fusion.compute_face_in_body_score(face_box, body_box)
        assert score < 0.5  # Penalized for being in lower body

    def test_iou_computation(self):
        """Test IoU computation."""
        fusion = TrackFusion()

        box1 = [0, 0, 100, 100]
        box2 = [0, 0, 100, 100]  # Same box
        iou = fusion.compute_iou(box1, box2)
        assert iou == 1.0

        box3 = [200, 200, 300, 300]  # No overlap
        iou = fusion.compute_iou(box1, box3)
        assert iou == 0.0

    def test_fused_identity_dataclass(self):
        """Test FusedIdentity serialization."""
        identity = FusedIdentity(
            identity_id="fused_0001",
            face_track_ids=[1, 2],
            body_track_ids=[100001, 100002],
            face_visible_frames=100,
            body_only_frames=50,
            total_frames=150,
        )

        d = identity.to_dict()
        assert d["identity_id"] == "fused_0001"
        assert d["face_track_ids"] == [1, 2]
        assert d["body_track_ids"] == [100001, 100002]
        assert d["face_visible_frames"] == 100
        assert d["body_only_frames"] == 50
        assert d["total_frames"] == 150

    def test_association_dataclass(self):
        """Test FaceBodyAssociation serialization."""
        assoc = FaceBodyAssociation(
            face_track_id=1,
            body_track_id=100001,
            confidence=0.85,
            method="iou",
            frame_range=(100, 200),
        )

        d = assoc.to_dict()
        assert d["face_track_id"] == 1
        assert d["body_track_id"] == 100001
        assert d["confidence"] == 0.85
        assert d["method"] == "iou"
        assert d["frame_range"] == [100, 200]


class TestScreenTimeComparison:
    """Tests for screen time comparison."""

    def test_time_segment_properties(self):
        """Test TimeSegment computed properties."""
        seg = TimeSegment(
            start_frame=0,
            end_frame=24,
            start_time=0.0,
            end_time=1.0,
            segment_type="face",
        )
        assert seg.duration == 1.0
        assert seg.frame_count == 25

    def test_frames_to_segments(self):
        """Test conversion of frames to segments."""
        comparator = ScreenTimeComparator(fps=24.0, merge_short_gaps=False)

        # Continuous frames
        frames = [0, 1, 2, 3, 4]
        segments = comparator._frames_to_segments(frames, "face")
        assert len(segments) == 1
        assert segments[0].start_frame == 0
        assert segments[0].end_frame == 4

        # Gap in frames
        frames = [0, 1, 2, 10, 11, 12]
        segments = comparator._frames_to_segments(frames, "face")
        assert len(segments) == 2
        assert segments[0].frame_count == 3
        assert segments[1].frame_count == 3

    def test_merge_segments(self):
        """Test segment merging."""
        comparator = ScreenTimeComparator(fps=24.0, merge_short_gaps=True, max_merge_gap_seconds=0.5)

        # Two segments with small gap (should merge)
        seg1 = TimeSegment(0, 10, 0.0, 0.4, "face")
        seg2 = TimeSegment(15, 25, 0.6, 1.0, "face")  # Gap of 0.2s
        merged = comparator._merge_segments([seg1, seg2])
        assert len(merged) == 1
        assert merged[0].start_frame == 0
        assert merged[0].end_frame == 25

        # Two segments with large gap (should not merge)
        seg3 = TimeSegment(50, 60, 2.0, 2.4, "face")  # Gap of 1.0s
        merged = comparator._merge_segments([seg1, seg3])
        assert len(merged) == 2

        # Different types (should not merge)
        seg4 = TimeSegment(15, 25, 0.6, 1.0, "body")
        merged = comparator._merge_segments([seg1, seg4])
        assert len(merged) == 2

    def test_screen_time_breakdown(self):
        """Test screen time breakdown computation."""
        comparator = ScreenTimeComparator(fps=24.0, merge_short_gaps=False)

        # Face frames: 0-100
        # Body frames: 50-150
        face_frames = list(range(0, 100))
        body_frames = list(range(50, 150))

        breakdown = comparator.compute_breakdown("test_id", face_frames, body_frames)

        assert breakdown.identity_id == "test_id"
        assert breakdown.face_only_frames == 100
        assert breakdown.combined_frames == 150  # Union of 0-150
        assert breakdown.duration_gain > 0  # Should have gain from body-only frames
        assert breakdown.duration_gain_pct > 0

    def test_empty_frames(self):
        """Test with empty frame lists."""
        comparator = ScreenTimeComparator()

        # No face or body frames
        breakdown = comparator.compute_breakdown("empty", [], [])
        assert breakdown.face_only_duration == 0.0
        assert breakdown.combined_duration == 0.0
        assert breakdown.duration_gain == 0.0

        # Only body frames
        breakdown = comparator.compute_breakdown("body_only", [], list(range(100)))
        assert breakdown.face_only_duration == 0.0
        assert breakdown.combined_duration > 0
        assert breakdown.body_only_duration > 0


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_detection_to_tracking_flow(self):
        """Test detection → tracking data flow."""
        # Create sample detections
        detections = [
            BodyDetection(0, 0.0, [100, 100, 200, 300], 0.9, 0),
            BodyDetection(0, 0.0, [400, 100, 500, 300], 0.85, 0),
            BodyDetection(1, 0.04, [105, 100, 205, 300], 0.88, 0),
            BodyDetection(1, 0.04, [405, 100, 505, 300], 0.82, 0),
            BodyDetection(2, 0.08, [110, 100, 210, 300], 0.9, 0),
            BodyDetection(2, 0.08, [410, 100, 510, 300], 0.87, 0),
        ]

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for det in detections:
                f.write(json.dumps(det.to_dict()) + "\n")
            det_path = Path(f.name)

        try:
            # Create tracker and run
            tracker = BodyTracker(
                tracker_type="bytetrack",
                track_buffer=30,
                id_offset=100000,
            )

            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                tracks_path = Path(f.name)

            track_bodies(tracker, det_path, tracks_path)

            # Verify tracks were created
            tracks = []
            with open(tracks_path) as f:
                for line in f:
                    tracks.append(json.loads(line))

            assert len(tracks) >= 2  # At least 2 persons

            # Each track should have detections
            for track in tracks:
                assert "track_id" in track
                assert "detections" in track
                assert len(track["detections"]) > 0

        finally:
            det_path.unlink(missing_ok=True)
            tracks_path.unlink(missing_ok=True)

    def test_fusion_with_mock_data(self):
        """Test fusion with mock face/body tracks."""
        fusion = TrackFusion(
            iou_threshold=0.5,
            min_overlap_ratio=0.7,
        )

        # Mock face track
        face_tracks = {
            1: {
                "track_id": 1,
                "detections": [
                    {"frame_idx": 0, "bbox": [150, 50, 250, 150]},
                    {"frame_idx": 1, "bbox": [155, 50, 255, 150]},
                    {"frame_idx": 2, "bbox": [160, 50, 260, 150]},
                ],
            }
        }

        # Mock body track (overlapping with face)
        body_tracks = {
            100001: {
                "track_id": 100001,
                "detections": [
                    {"frame_idx": 0, "bbox": [100, 0, 300, 400]},
                    {"frame_idx": 1, "bbox": [105, 0, 305, 400]},
                    {"frame_idx": 2, "bbox": [110, 0, 310, 400]},
                    {"frame_idx": 3, "bbox": [115, 0, 315, 400]},  # Face gone
                    {"frame_idx": 4, "bbox": [120, 0, 320, 400]},
                ],
            }
        }

        identities = fusion.fuse_tracks(face_tracks, body_tracks)

        # Should create one fused identity
        assert len(identities) >= 1

        # Check identity structure
        for identity_id, identity in identities.items():
            assert identity.identity_id == identity_id
            # If face and body were associated
            if identity.face_track_ids and identity.body_track_ids:
                assert 1 in identity.face_track_ids
                assert 100001 in identity.body_track_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
