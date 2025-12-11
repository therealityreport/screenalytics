# Face Alignment Feature

FAN-based face alignment with 68-point landmarks for improved embedding quality.

## Overview

This feature adds robust face alignment to the Screenalytics pipeline using:
- **FAN (Face Alignment Network)** - 68-point 2D landmark detection
- **LUVLi** - Per-landmark uncertainty for quality gating (future)
- **3DDFA_V2** - Selective 3D head pose estimation (future)

## Why This Feature?

Current 5-point alignment (InsightFace) has limitations:
- Imprecise alignment for non-frontal faces
- No quality signal for filtering bad faces
- Cannot handle profiles or partial occlusions well

## Quick Start

```bash
# Run alignment on an episode
python -m FEATURES.face_alignment --episode-id rhobh-s05e02

# CPU mode with stride
python -m FEATURES.face_alignment --episode-id rhobh-s05e02 --device cpu --stride 5

# Export aligned crops
python -m FEATURES.face_alignment --episode-id rhobh-s05e02 --export-crops
```

## Components

### FAN 2D Aligner (`src/run_fan_alignment.py`)
- `FANAligner` class - Lazy-loaded FAN model wrapper
- 68-point landmark extraction
- 5-point extraction for ArcFace alignment
- Similarity transform for canonical pose

### Face Alignment Runner (`src/face_alignment_runner.py`)
- `FaceAlignmentRunner` - Pipeline orchestration
- `FaceAlignmentConfig` - Config loading from YAML

### Detection Loaders (`src/load_detections.py`)
- `load_face_detections()` - Load from detections.jsonl
- `load_face_tracks()` - Load from tracks.jsonl
- `get_representative_frames()` - Select high-quality frames

### Export Utilities (`src/export_aligned_faces.py`)
- `export_aligned_faces()` - Write to JSONL
- `compute_alignment_stats()` - Summary statistics

### Future: LUVLi Quality Gate (`src/luvli_quality.py`)
Estimates per-landmark uncertainty to filter low-quality faces.

### Future: 3DDFA_V2 (`src/ddfa_v2.py`)
Extracts 3D head pose (yaw, pitch, roll) for challenging faces.

## Output Artifacts

Written to `data/manifests/{ep_id}/face_alignment/`:

| File | Description |
|------|-------------|
| `aligned_faces.jsonl` | Per-face alignment results |
| `alignment_stats.json` | Summary statistics |
| `aligned_crops/` | Optional aligned face images |

### aligned_faces.jsonl Schema

```json
{
  "detection_id": 123,
  "frame_idx": 456,
  "track_id": 7,
  "bbox": [100.0, 100.0, 200.0, 250.0],
  "landmarks_68": [[x1, y1], [x2, y2], ...],
  "alignment_quality": 0.85,
  "pose_yaw": 15.0,
  "pose_pitch": -5.0,
  "pose_roll": 2.0
}
```

## Alignment Quality

The `alignment_quality` field in aligned_faces.jsonl provides a quality score (0.0-1.0, higher is better).

**Current Implementation:** Heuristic-based
- **size_score (30%):** Larger faces relative to minimum threshold score higher
- **aspect_ratio_score (30%):** Faces with typical frontal aspect ratio (~0.85) score higher
- **landmark_spread_score (40%):** Well-distributed landmarks with good symmetry score higher

**Confidence:** 0.6 (heuristic-based; will increase to ~0.9 with LUVLi model)

**Future:** The heuristic will be replaced with LUVLi model-based uncertainty estimation, which provides per-landmark confidence scores.

**Usage:**
```python
from FEATURES.face_alignment.src.alignment_quality import filter_by_quality

passed, rejected = filter_by_quality(aligned_faces, min_quality=0.6)
```

## Configuration

See `config/pipeline/face_alignment.yaml`:

```yaml
model:
  type: "2d"           # 2d | 3d
  device: "cuda"       # cuda | cpu

processing:
  stride: 1            # Process every Nth frame
  min_detection_score: 0.5
  min_face_size: 20

quality:
  threshold: 0.6       # Skip faces below this
  enabled: true

output:
  crop_size: 112
  save_crops: false
```

## Usage

```python
from FEATURES.face_alignment.src.run_fan_alignment import FANAligner, run_fan_alignment

# Low-level: direct aligner usage
aligner = FANAligner(device="cuda", model_type="2d")
landmarks = aligner.align_batch(images, bboxes)

# High-level: full pipeline
from FEATURES.face_alignment.src.face_alignment_runner import (
    FaceAlignmentRunner,
    FaceAlignmentConfig,
)

config = FaceAlignmentConfig.from_yaml("config/pipeline/face_alignment.yaml")
runner = FaceAlignmentRunner(episode_id, manifest_dir, config)
runner.run_full_pipeline(video_path)
```

## Testing

```bash
# Run unit tests
pytest FEATURES/face-alignment/tests/ -v

# Run with coverage
pytest FEATURES/face-alignment/tests/ --cov=FEATURES.face_alignment
```

## Rollback

To disable face alignment and revert to InsightFace 5-point:

```yaml
# config/pipeline/face_alignment.yaml
face_alignment:
  enabled: false
  aligner: insightface  # Use InsightFace 5-point
```

## Related Documentation

- [TODO with LUVLi/3DDFA_V2 Planning](../TODO.md)
- [Full Feature Doc](../../../docs/todo/feature_face_alignment_fan_luvli_3ddfa.md)
- [Feature Overview](../../../docs/features/vision_alignment_and_body_tracking.md)
- [ACCEPTANCE_MATRIX.md](../../../ACCEPTANCE_MATRIX.md) - Sections 3.7-3.9
