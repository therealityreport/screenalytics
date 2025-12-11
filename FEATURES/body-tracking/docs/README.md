# Body Tracking Feature

## Overview

This feature adds person body tracking to maintain identity when faces aren't visible:
- **YOLO Person Detection** - Detect all persons in frame
- **ByteTrack** - Temporal tracking of body boxes
- **OSNet Re-ID** - Person re-identification embeddings
- **Track Fusion** - Associate face tracks with body tracks

## Why This Feature?

Current face-only tracking loses identity when:
- Cast turns away from camera (back of head)
- Face is occluded by objects or other people
- Profile angle exceeds embedding threshold

This results in underreported screen time and fragmented identities.

## Components

### Person Detector (`src/person_detector.py`)
Detects persons using YOLOv8, filtering to COCO "person" class.

### Person Embedder (`src/person_embedder.py`)
Generates 256-d Re-ID embeddings using OSNet.

### Track Fusion (`src/track_fusion.py`)
Associates face tracks with body tracks using IoU and Re-ID handoff.

## Configuration

See `config/pipeline/body_detection.yaml`:

```yaml
body_tracking:
  enabled: true

person_detection:
  model: yolov8n
  confidence_threshold: 0.50

person_reid:
  model: osnet_x1_0
  embedding_dim: 256
```

## Screen Time Output

```json
{
  "identity_id": "ID_001",
  "face_visible_duration": 125.5,
  "body_only_duration": 45.2,
  "total_duration": 170.7
}
```

## Related Documentation

- [Full TODO](../../../docs/todo/feature_body_tracking_reid_fusion.md)
- [Feature Overview](../../../docs/features/vision_alignment_and_body_tracking.md)
- [ACCEPTANCE_MATRIX.md](../../../ACCEPTANCE_MATRIX.md) - Sections 3.10-3.12
