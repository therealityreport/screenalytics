# Face Alignment Feature

## Overview

This feature adds robust face alignment to the Screenalytics pipeline using:
- **FAN (Face Alignment Network)** - 68-point 2D landmark detection
- **LUVLi** - Per-landmark uncertainty for quality gating
- **3DDFA_V2** - Selective 3D head pose estimation

## Why This Feature?

Current 5-point alignment (InsightFace) has limitations:
- Imprecise alignment for non-frontal faces
- No quality signal for filtering bad faces
- Cannot handle profiles or partial occlusions well

## Components

### FAN 2D Aligner (`src/fan_aligner.py`)
Detects 68 facial landmarks for improved alignment accuracy.

### LUVLi Quality Gate (`src/luvli_quality.py`)
Estimates per-landmark uncertainty to filter low-quality faces.

### 3DDFA_V2 (`src/ddfa_v2.py`)
Extracts 3D head pose (yaw, pitch, roll) for challenging faces.

## Configuration

See `config/pipeline/alignment.yaml`:

```yaml
aligner: fan_2d
quality_gating:
  enabled: true
  min_alignment_quality: 0.60
```

## Usage

```python
from FEATURES.face_alignment.src.fan_aligner import FAN2DAligner
from FEATURES.face_alignment.src.luvli_quality import compute_alignment_quality

aligner = FAN2DAligner()
landmarks = aligner.detect(face_crop)
quality = compute_alignment_quality(landmarks)
```

## Related Documentation

- [Full TODO](../../../docs/todo/feature_face_alignment_fan_luvli_3ddfa.md)
- [Feature Overview](../../../docs/features/vision_alignment_and_body_tracking.md)
- [ACCEPTANCE_MATRIX.md](../../../ACCEPTANCE_MATRIX.md) - Sections 3.7-3.9
