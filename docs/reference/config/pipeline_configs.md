# Pipeline Configuration Reference

**Version:** 2.0
**Last Updated:** 2025-11-18

---

## 1. Overview

All pipeline behavior is **config-driven** (no hardcoded thresholds). This document provides a key-by-key reference for all pipeline YAML configs.

**Location:** `config/pipeline/*.yaml`

---

## 2. Detection Config (`detection.yaml`)

### 2.1 Model
```yaml
model_id: retinaface_r50
```
- **Type:** string
- **Default:** `retinaface_r50`
- **Valid values:** `retinaface_r50`, `retinaface_mnet` (mobile)
- **Effect:** Which RetinaFace ONNX weights to load

### 2.2 Thresholds
```yaml
min_size: 90
confidence_th: 0.8
iou_th: 0.5
```

| Key | Type | Default | Range | Effect |
|-----|------|---------|-------|--------|
| `min_size` | int | 90 | 32–256 | Minimum face size (pixels); lower = more small faces, slower |
| `confidence_th` | float | 0.8 | 0.0–1.0 | Detection confidence threshold; lower = more faces, more false positives |
| `iou_th` | float | 0.5 | 0.0–1.0 | NMS IoU threshold; lower = stricter NMS (fewer overlaps) |

### 2.3 Adaptive Confidence
```yaml
adaptive_confidence: false
min_confidence: 0.6
max_confidence: 0.9
```

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `adaptive_confidence` | bool | false | Enable adaptive thresholding for low-light scenes |
| `min_confidence` | float | 0.6 | Minimum threshold for dark scenes |
| `max_confidence` | float | 0.9 | Maximum threshold for well-lit scenes |

### 2.4 NMS Mode
```yaml
nms_mode: hard
soft_nms_sigma: 0.5
```

| Key | Type | Default | Valid Values | Effect |
|-----|------|---------|--------------|--------|
| `nms_mode` | string | hard | `hard`, `soft` | NMS algorithm; soft-NMS better for overlapping faces (hugs, crowds) |
| `soft_nms_sigma` | float | 0.5 | 0.1–1.0 | Gaussian decay parameter for soft-NMS (lower = more aggressive) |

### 2.5 Pose Quality
```yaml
check_pose_quality: true
max_yaw_angle: 45.0
```

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `check_pose_quality` | bool | true | Enable pose-based landmark filtering |
| `max_yaw_angle` | float | 45.0 | Max head rotation (degrees); faces beyond this angle discarded |

### 2.6 Letterbox Detection
```yaml
auto_crop_letterbox: false
letterbox_threshold: 20
```

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `auto_crop_letterbox` | bool | false | Auto-detect and crop black bars (letterbox/pillarbox) |
| `letterbox_threshold` | int | 20 | Pixel intensity threshold (0–255) for detecting black bars |

---

## 3. Tracking Config (`tracking.yaml`)

### 3.1 ByteTrack Spatial Matching
```yaml
track_thresh: 0.70
match_thresh: 0.85
track_buffer: 90
max_track_buffer: 300
```

| Key | Type | Default | Range | Effect |
|-----|------|---------|-------|--------|
| `track_thresh` | float | 0.70 | 0.0–1.0 | Min confidence to continue track; lower = more tracks, more noise |
| `match_thresh` | float | 0.85 | 0.0–1.0 | IoU threshold for bbox association; higher = stricter matching |
| `track_buffer` | int | 90 | 30–300 | Frames to keep track alive after last detection (~3–4s); higher = fewer ID switches, longer tracks |
| `max_track_buffer` | int | 300 | 90–600 | Cap on buffer to prevent runaway memory |

### 3.2 New Track Threshold
```yaml
new_track_thresh: 0.85
```

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `new_track_thresh` | float | 0.85 | Higher threshold for starting new tracks (reduces ghost tracks) |

### 3.3 AppearanceGate
```yaml
gate_enabled: true
```

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `gate_enabled` | bool | true | Enable appearance-based track splitting (via embeddings) |

**Appearance gate thresholds (via environment variables):**
- `TRACK_GATE_APPEAR_HARD`: default 0.65 (hard split if similarity < 65%)
- `TRACK_GATE_APPEAR_SOFT`: default 0.75 (soft split if similarity < 75%)
- `TRACK_GATE_APPEAR_STREAK`: default 3 (consecutive low-sim frames before split)
- `TRACK_GATE_IOU`: default 0.50 (split if spatial jump, IoU < 50%)
- `TRACK_GATE_EMB_EVERY`: default 24 (extract embeddings every N frames)

### 3.4 Global Motion Compensation (GMC)
Set via `SCREENALYTICS_GMC_METHOD` environment variable:
- `"sparseOptFlow"` (default): Sparse optical flow for camera motion
- `"orb"`: ORB feature matching
- `"off"`: Disable GMC

---

## 4. Performance Profiles (`performance_profiles.yaml`)

### 4.1 Profiles
```yaml
fast_cpu:
  description: "Optimized for fanless/low-power devices"
  coreml_input_size: 384
  detection_fps_limit: 15
  frame_stride: 10
  min_size: 120
  adaptive_confidence: true
  nms_mode: hard
  check_pose_quality: false

balanced:
  description: "Balanced performance and quality"
  coreml_input_size: 480
  detection_fps_limit: 24
  frame_stride: 5
  min_size: 90
  adaptive_confidence: false
  nms_mode: hard
  check_pose_quality: true

high_accuracy:
  description: "Maximum quality for powerful systems"
  coreml_input_size: 640
  detection_fps_limit: 30
  frame_stride: 1
  min_size: 64
  adaptive_confidence: true
  nms_mode: soft
  soft_nms_sigma: 0.5
  check_pose_quality: true
```

| Profile | Device | Stride | FPS | Use Case |
|---------|--------|--------|-----|----------|
| `fast_cpu` | CPU (fanless) | 10 | 15 | Exploratory, thermal-constrained |
| `balanced` | CPU/MPS | 5 | 24 | Standard local dev |
| `high_accuracy` | GPU (CUDA) | 1 | 30 | Production, max recall |

---

## 5. Faces Embed Sampling (`faces_embed_sampling.yaml`)

### 5.1 Quality Gating
```yaml
min_quality: 0.7
min_confidence: 0.8
min_face_size: 64
max_blur: 100
```

| Key | Type | Default | Range | Effect |
|-----|------|---------|-------|--------|
| `min_quality` | float | 0.7 | 0.0–1.0 | Combined quality score threshold; higher = fewer but better faces |
| `min_confidence` | float | 0.8 | 0.0–1.0 | RetinaFace confidence threshold |
| `min_face_size` | int | 64 | 32–256 | Minimum face size (pixels) |
| `max_blur` | int | 100 | 0–500 | Laplacian variance threshold (lower = more blur tolerated) |

### 5.2 Volume Control
```yaml
max_crops_per_track: 50
max_crops_per_episode: 5000
```

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `max_crops_per_track` | int | 50 | Limit crops per track to avoid embedding thousands of near-identical faces |
| `max_crops_per_episode` | int | 5000 | Global episode limit (optional) |

### 5.3 Sampling Strategy
```yaml
sampling_mode: uniform
sample_interval: 24
```

| Key | Type | Default | Valid Values | Effect |
|-----|------|---------|--------------|--------|
| `sampling_mode` | string | uniform | `uniform`, `quality-weighted`, `stratified` | How to sample frames along track |
| `sample_interval` | int | 24 | 1–60 | Sample every Nth frame (uniform mode only) |

### 5.4 Crop Generation
```yaml
thumb_size: 224
jpeg_quality: 90
check_pose_quality: true
max_yaw_angle: 45.0
```

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `thumb_size` | int | 224 | Output crop size (before resize to 112x112 for ArcFace) |
| `jpeg_quality` | int | 90 | JPEG quality for saved crops (0–100) |
| `check_pose_quality` | bool | true | Enable pose filtering |
| `max_yaw_angle` | float | 45.0 | Max head rotation (degrees) |

---

## 6. Clustering Config (TBD: `recognition.yaml`)

**Note:** Currently passed via CLI/API args; future config file:

```yaml
algorithm: agglomerative
linkage: average
cluster_thresh: 0.58
min_cluster_size: 2
outlier_mode: singleton
pooling_method: mean
```

| Key | Type | Default | Valid Values | Effect |
|-----|------|---------|--------------|--------|
| `algorithm` | string | agglomerative | `agglomerative`, `dbscan`, `hdbscan` | Clustering algorithm |
| `linkage` | string | average | `ward`, `average`, `complete` | Hierarchical linkage method |
| `cluster_thresh` | float | 0.58 | 0.0–1.0 | Cosine similarity threshold; lower = more clusters |
| `min_cluster_size` | int | 2 | 1–10 | Minimum tracks per cluster |
| `outlier_mode` | string | singleton | `singleton`, `noise` | How to handle outliers (cluster_id=null vs -1) |
| `pooling_method` | string | mean | `mean`, `median`, `max` | Track-level embedding pooling |

---

## 7. Audio Config (TBD: `audio.yaml`)

**Placeholder for future audio diarization/ASR config:**

```yaml
diarization:
  model: pyannote/speaker-diarization
  min_speakers: 1
  max_speakers: 10

asr:
  model: faster-whisper/large-v2
  language: en
  beam_size: 5
```

---

## 8. Screen Time Config (`screen_time_v2.yaml`)

```yaml
screen_time_presets:
  bravo_default:
    quality_min: 0.7
    gap_tolerance_s: 1.0
    screen_time_mode: track
    edge_padding_s: 0.5
    track_coverage_min: 0.5
    use_video_decode: false
```

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `quality_min` | float | 0.7 | Minimum quality for screentime inclusion |
| `gap_tolerance_s` | float | 1.0 | Max gap (seconds) to merge adjacent appearances |
| `screen_time_mode` | string | track | `track` or `frame` aggregation |
| `edge_padding_s` | float | 0.5 | Edge padding (seconds) |
| `track_coverage_min` | float | 0.5 | Minimum track coverage fraction |
| `use_video_decode` | bool | false | Decode video for frame-level validation |

---

## 9. Config Override Precedence

1. **CLI args** (highest priority): `--stride 3 --device cuda`
2. **Environment variables**: `TRACK_THRESH=0.75`
3. **Profile-specific config**: `performance_profiles.yaml:balanced`
4. **Stage-specific config**: `tracking.yaml`, `detection.yaml`
5. **Default values** (lowest priority): Hardcoded in code

**Example:**
```bash
export TRACK_THRESH=0.75
python tools/episode_run.py --profile balanced --stride 5
```
Effective config:
- `stride: 5` (CLI overrides profile)
- `track_thresh: 0.75` (env overrides profile + stage config)
- Other params from `balanced` profile

---

## 10. References

- [Pipeline Overview](../../pipeline/overview.md)
- [Detect & Track](../../pipeline/detect_track_faces.md)
- [Faces Harvest](../../pipeline/faces_harvest.md)
- [Cluster Identities](../../pipeline/cluster_identities.md)
- [Performance Tuning](../../ops/performance_tuning_faces_pipeline.md)

---

**Maintained by:** Screenalytics Engineering
