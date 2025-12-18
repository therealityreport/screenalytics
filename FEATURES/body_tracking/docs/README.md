# Body Tracking Feature

## Overview

This feature adds person body tracking to maintain identity when faces aren't visible:
- **YOLO Person Detection** - Detect all persons in frame
- **ByteTrack** - Temporal tracking of body boxes
- **OSNet Re-ID** - Person re-identification embeddings
- **Track Fusion** - Associate face tracks with body tracks
- **Screen Time Comparison** - Face-only vs face+body metrics

## Why This Feature?

Current face-only tracking loses identity when:
- Cast turns away from camera (back of head)
- Face is occluded by objects or other people
- Profile angle exceeds embedding threshold

This results in underreported screen time and fragmented identities.

## Quick Start

```bash
# Run full pipeline on episode
python -m FEATURES.body_tracking --episode-id rhoslc-s06e01

# With verbose output
python -m FEATURES.body_tracking --episode-id rhoslc-s06e01 --verbose

# Skip existing artifacts
python -m FEATURES.body_tracking --episode-id rhoslc-s06e01 --skip-existing

# Run specific stage only
python -m FEATURES.body_tracking --episode-id rhoslc-s06e01 --stage detect
```

## Pipeline Stages

### 1. Detection (`--stage detect`)
Runs YOLO person detection on every frame (or sampled frames).

**Output:** `body_detections.jsonl`
```json
{"frame_idx": 100, "timestamp": 4.16, "bbox": [100, 200, 300, 500], "score": 0.95, "class_id": 0}
```

### 2. Tracking (`--stage track`)
Runs ByteTrack to associate detections into tracks.

**Output:** `body_tracks.jsonl`
```json
{
  "track_id": 100001,
  "start_frame": 100,
  "end_frame": 500,
  "duration": 16.67,
  "frame_count": 400,
  "detections": [...]
}
```

### 3. Embedding (`--stage embed`)
Computes OSNet Re-ID embeddings for representative frames per track.

**Output:** `body_embeddings.npy` + `body_embeddings_meta.json`

### 4. Fusion (`--stage fuse`)
Associates face tracks with body tracks using:
- IoU-based matching when face is visible
- Re-ID handoff when face disappears

**Output:** `track_fusion.json`
```json
{
  "num_face_tracks": 25,
  "num_body_tracks": 30,
  "num_fused_identities": 8,
  "identities": {
    "fused_0001": {
      "face_track_ids": [1, 2],
      "body_track_ids": [100001, 100002],
      "face_visible_frames": 500,
      "body_only_frames": 200,
      "total_frames": 700
    }
  }
}
```

### 5. Comparison (`--stage compare`)
Compares face-only screen time with face+body screen time.

**Output:** `screentime_comparison.json`
```json
{
  "summary": {
    "total_identities": 8,
    "identities_with_gain": 6,
    "total_face_only_duration": 125.5,
    "total_body_duration": 160.2,
    "total_fused_duration": 115.0,
    "total_combined_duration": 170.7,
    "total_duration_gain": 45.2,
    "face_total_s": 125.5,
    "body_total_s": 160.2,
    "fused_total_s": 115.0,
    "combined_total_s": 170.7,
    "gain_total_s": 45.2,
    "avg_duration_gain_pct": 36.0
  },
  "breakdowns": [...]
}
```

## Components

### Body Detector (`src/detect_bodies.py`)
- Uses YOLOv8 (configurable model: n/s/m)
- Filters to COCO "person" class (id=0)
- Supports batch inference for performance

### Body Tracker (`src/track_bodies.py`)
- Uses `supervision.ByteTrack` when available
- Falls back to `SimpleIoUTracker` if supervision not installed
- Applies ID offset (100000+) to avoid collision with face tracks

### Body Embedder (`src/body_embeddings.py`)
- Uses torchreid's OSNet (512-d embeddings)
- Extracts body crops with configurable margin
- Samples representative frames per track (default: 5)

### Track Fusion (`src/track_fusion.py`)
- IoU-based association for overlapping face/body boxes
- Re-ID handoff for temporal gaps up to 30 seconds
- Union-find for building fused identities

### Screen Time Comparator (`src/screentime_compare.py`)
- Segment-based duration calculation
- Configurable gap merging (default: 1 second)
- Per-identity and aggregate metrics

## Configuration

### Body Detection (`config/pipeline/body_detection.yaml`)

```yaml
body_tracking:
  enabled: true

person_detection:
  model: yolov8n              # yolov8n | yolov8s | yolov8m
  confidence_threshold: 0.50
  nms_iou_threshold: 0.45
  min_height_px: 50
  min_width_px: 25
  device: auto                # auto | cuda | cpu | mps

person_tracking:
  tracker: bytetrack
  track_thresh: 0.50
  track_buffer: 120           # Frames to keep lost tracks
  id_offset: 100000           # Avoid collision with face IDs

person_reid:
  enabled: true
  model: osnet_x1_0
  embedding_dim: 256

performance:
  detection_batch_size: 4
  reid_batch_size: 32
```

### Track Fusion (`config/pipeline/track_fusion.yaml`)

```yaml
iou_association:
  iou_threshold: 0.02
  min_overlap_ratio: 0.7
  face_in_upper_body: true    # Face should be in upper 50% of body

reid_handoff:
  similarity_threshold: 0.70
  max_gap_seconds: 30

screen_time:
  merge_short_gaps: true
  max_merge_gap_seconds: 1.0
```

## Output Artifacts

All artifacts are stored in `data/manifests/{ep_id}/body_tracking/`:

| File | Description |
|------|-------------|
| `body_detections.jsonl` | Raw person detections per frame |
| `body_tracks.jsonl` | Tracked persons over time |
| `body_embeddings.npy` | Re-ID embedding vectors |
| `body_embeddings_meta.json` | Metadata for embeddings |
| `track_fusion.json` | Face↔body associations |
| `screentime_comparison.json` | Face-only vs combined metrics |
| `body_metrics.json` | Pipeline summary and stats |

## Dependencies

```
ultralytics>=8.2.0      # YOLO person detection
torchreid>=1.4.0        # OSNet Re-ID embeddings
supervision>=0.20.0     # ByteTrack (optional, has fallback)
numpy
opencv-python
pyyaml
```

## Running Tests

```bash
# Unit tests (no ML dependencies required)
pytest FEATURES/body_tracking/tests/test_body_tracking.py -v

# Skip tests requiring ultralytics/torchreid
pytest FEATURES/body_tracking/tests/ -v -k "not integration"
```

## Related Documentation

- [Full TODO](../../../docs/todo/feature_body_tracking_reid_fusion.md)
- [Feature Overview](../../../docs/features/vision_alignment_and_body_tracking.md)
- [ACCEPTANCE_MATRIX.md](../../../ACCEPTANCE_MATRIX.md) - Sections 3.10-3.12
- [Skills: body-tracking](../../../.claude/skills/body-tracking/SKILL.md)
