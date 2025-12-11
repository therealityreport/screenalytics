# Vision Analytics Feature

## Overview

This feature adds advanced visibility analytics:
- **Face Mesh** - 468-point dense landmarks via MediaPipe
- **Visibility Fraction** - How much of face is visible (0-1)
- **Gaze Direction** - Coarse gaze estimation (left/center/right)
- **CenterFace** - [Future] CPU-friendly detector

## Why This Feature?

Current visibility metrics are binary (face visible or not). This misses:
- Partial visibility (half-face, profile)
- Gaze information (looking at camera vs away)
- Regional occlusion details

## Components

### Face Mesh (`src/face_mesh.py`)
Extracts 468-point mesh on close-up faces using MediaPipe.

### Visibility (`src/visibility.py`)
Computes visibility fraction and regional breakdown.

### Gaze (`src/gaze.py`)
Estimates coarse gaze direction from iris landmarks.

### CenterFace (`src/centerface.py`)
[STUB] CPU-friendly detector for future implementation.

## Configuration

See `config/pipeline/analytics.yaml`:

```yaml
face_mesh:
  enabled: true
  execution:
    closeup_threshold: 0.05
    sample_rate: 30

gaze:
  enabled: true
  use_iris: true
```

## Output

```json
{
  "visibility_fraction": 0.85,
  "visibility_breakdown": {
    "left_eye": 1.0,
    "right_eye": 0.7,
    "nose": 1.0,
    "mouth": 0.9
  },
  "gaze": {
    "horizontal": "center",
    "vertical": "center",
    "looking_at_camera": true
  }
}
```

## Related Documentation

- [Full TODO](../../../docs/todo/feature_mesh_and_advanced_visibility.md)
- [Feature Overview](../../../docs/features/vision_alignment_and_body_tracking.md)
- [ACCEPTANCE_MATRIX.md](../../../ACCEPTANCE_MATRIX.md) - Sections 3.14-3.15
