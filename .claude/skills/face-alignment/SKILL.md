---
name: face-alignment
description: Face alignment, landmark extraction, and quality gating. Use when debugging alignment issues, reviewing rejected faces, or tuning quality thresholds.
---

# Face Alignment Skill

Use this skill to debug face alignment issues and tune quality thresholds.

## When to Use

- Faces being rejected but you don't know why
- Identity fragmentation that might be alignment-related
- Need to adjust alignment quality thresholds
- Debugging head pose estimation
- Comparing FAN vs InsightFace alignment quality

## Sub-agents

| Sub-agent | Purpose |
|-----------|---------|
| **FaceAlign2DSubagent** | FAN 68-point 2D landmark extraction |
| **FaceAlign3DSubagent** | 3DDFA_V2 dense 3D alignment |
| **AlignmentQualitySubagent** | LUVLi uncertainty scoring |

## Key Skills

### `align_face_boxes()`
Run alignment on detected face boxes.

```python
from FEATURES.face_alignment.src.fan_aligner import FAN2DAligner

aligner = FAN2DAligner()
landmarks = aligner.detect(face_crop)
aligned_crop = aligner.align(face_crop, landmarks)
```

### `compute_alignment_quality()`
Get per-face quality scores using LUVLi uncertainty.

```python
from FEATURES.face_alignment.src.luvli_quality import compute_alignment_quality

quality = compute_alignment_quality(landmarks)
if quality < config.min_alignment_quality:
    # Skip embedding for this face
```

### `export_alignment_metrics()`
Generate alignment quality reports for an episode.

## Config Reference

**File:** `config/pipeline/alignment.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `aligner` | `fan_2d` | Alignment backend: `fan_2d`, `fan_3d`, `insightface` |
| `min_alignment_quality` | 0.60 | Quality gate threshold [0, 1] |
| `run_3d_every_n_frames` | 10 | 3DDFA sampling rate |
| `max_yaw_for_embedding` | 75 | Skip profiles beyond this angle |
| `fallback_on_failure` | true | Use InsightFace if FAN fails |

## Common Issues

### "alignment:quality_gate"

**Cause:** Face below quality threshold

**Check:** `alignment_quality` in diagnostic

**Fix:** Lower threshold in config:
```yaml
quality_gating:
  min_alignment_quality: 0.50  # default is 0.60
```

### High `landmark_jitter`

**Cause:** Unstable landmarks across frames

**Check:** Per-frame landmark variance in track

**Fix:**
- Ensure FAN model loaded correctly
- Check input face crop quality
- May need temporal smoothing

### "alignment:extreme_pose"

**Cause:** Head pose beyond embedding threshold

**Check:** `head_pose.yaw` in face metadata

**Fix:** Increase pose limit:
```yaml
ddfa_v2:
  pose_limits:
    max_yaw_for_embedding: 80  # default is 75
```

## Diagnostic Output

```json
{
  "face_id": "F_42_100",
  "alignment_quality": 0.72,
  "landmarks_detected": 68,
  "head_pose": {
    "yaw": -25.5,
    "pitch": 10.2,
    "roll": 3.1
  },
  "quality_breakdown": {
    "eyes": 0.85,
    "nose": 0.80,
    "mouth": 0.65,
    "chin": 0.55
  },
  "aligner_used": "fan_2d"
}
```

## Key Files

| File | Purpose |
|------|---------|
| `FEATURES/face-alignment/src/fan_aligner.py` | FAN 2D/3D landmarks |
| `FEATURES/face-alignment/src/luvli_quality.py` | Quality scoring |
| `FEATURES/face-alignment/src/ddfa_v2.py` | 3D head pose |
| `config/pipeline/alignment.yaml` | Configuration |
| `tests/ml/test_face_alignment.py` | Integration tests |

## Related Skills

- [pipeline-insights](../pipeline-insights/SKILL.md) - General pipeline debugging
- [cluster-quality](../cluster-quality/SKILL.md) - Clustering metrics
