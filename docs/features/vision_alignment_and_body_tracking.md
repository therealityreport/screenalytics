# Vision Alignment & Body Tracking

Version: 1.0
Status: In Development
Last Updated: 2025-12-11

---

## Overview

This document describes the extension of Screenalytics from face-only tracking to a comprehensive person tracking system that includes:

1. **Robust Face Alignment** - Using FAN (Face Alignment Network) for 68-point 2D landmarks with quality gating via LUVLi uncertainty estimation
2. **Body Tracking + Person Re-ID** - Track cast members even when faces aren't visible using YOLO person detection and OSNet Re-ID embeddings
3. **Optimized Embedding Engines** - TensorRT and ONNXRuntime backends for 5-10x embedding throughput improvement
4. **Advanced Visibility Analytics** - Face mesh, gaze direction, and fine-grained visibility metrics

---

## Goals

| Goal | Metric | Target |
|------|--------|--------|
| Reduce identity fragmentation | Track splits per episode | ≥10% reduction |
| Capture hidden screen time | Body-only duration captured | ≥30% of lost time |
| Improve embedding throughput | Faces/sec @ batch=32 | ≥5x vs PyTorch |
| Better quality gating | False embedding rate | ≤5% of faces |

---

## Updated Pipeline Architecture

```
Video Frames
    │
    ├─────────────────────────────────────────────────────────────┐
    │                                                             │
    ▼                                                             ▼
┌─────────────────────────┐                         ┌─────────────────────────┐
│   Face Detection        │                         │   Person Detection      │
│   (RetinaFace)          │                         │   (YOLOv8)              │
└───────────┬─────────────┘                         └───────────┬─────────────┘
            │                                                   │
            ▼                                                   ▼
┌─────────────────────────┐                         ┌─────────────────────────┐
│   Face Alignment        │                         │   Person Tracking       │
│   (FAN 2D/3D)           │                         │   (ByteTrack)           │
│   68/98-point landmarks │                         └───────────┬─────────────┘
└───────────┬─────────────┘                                     │
            │                                                   ▼
            ▼                                       ┌─────────────────────────┐
┌─────────────────────────┐                         │   Person Re-ID          │
│   Alignment Quality     │                         │   (OSNet 256-d)         │
│   Gate (LUVLi)          │                         └───────────┬─────────────┘
│   min_quality: 0.60     │                                     │
└───────────┬─────────────┘                                     │
            │                                                   │
    ┌───────┴───────┐                                           │
    │               │                                           │
    ▼               ▼                                           │
 [PASS]          [SKIP]                                         │
    │                                                           │
    ▼                                                           │
┌─────────────────────────┐                                     │
│   Face Embedding        │                                     │
│   (ArcFace TensorRT)    │                                     │
│   512-d, FP16           │                                     │
└───────────┬─────────────┘                                     │
            │                                                   │
            ▼                                                   │
┌─────────────────────────┐                                     │
│   Face Tracking         │                                     │
│   (ByteTrack)           │                                     │
└───────────┬─────────────┘                                     │
            │                                                   │
            └───────────────────┬───────────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────┐
                  │   Face↔Body Association │
                  │   (IoU + Re-ID handoff) │
                  └───────────┬─────────────┘
                              │
                              ▼
                  ┌─────────────────────────┐
                  │   Identity Clustering   │
                  │   (HAC + Centroid)      │
                  └───────────┬─────────────┘
                              │
                              ▼
                  ┌─────────────────────────┐
                  │   Unified Timeline      │
                  │   ├─ face_visible_time  │
                  │   └─ body_only_time     │
                  └─────────────────────────┘
```

---

## Feature Modules

### 1. Face Alignment (Priority 1)

| Component | Library | Purpose |
|-----------|---------|---------|
| FAN 2D | face-alignment 1.3.5+ | 68-point 2D landmarks |
| LUVLi | custom integration | Per-landmark uncertainty |
| 3DDFA_V2 | cleardusk/3DDFA_V2 | 3D head pose (selective) |

**Integration Points:**
- `tools/episode_run.py:2033` - `_prepare_face_crop()` function
- `tools/episode_run.py:1710` - `ArcFaceEmbedder` class

**Config:** `config/pipeline/alignment.yaml`

**Feature Sandbox:** `FEATURES/face-alignment/`

**TODO Doc:** [docs/todo/feature_face_alignment_fan_luvli_3ddfa.md](../todo/feature_face_alignment_fan_luvli_3ddfa.md)

---

### 2. Body Tracking + Re-ID (Priority 2)

| Component | Library | Purpose |
|-----------|---------|---------|
| Person Detector | ultralytics 8.2+ | YOLOv8 COCO person class |
| Person Tracker | ByteTrack | Temporal consistency |
| Person Re-ID | torchreid 1.4+ | OSNet 256-d embeddings |
| Track Fusion | custom | Face↔body association |

**Integration Points:**
- `tools/episode_run.py:1309` - `ByteTrackAdapter` (reuse for body)
- `tools/episode_run.py:982` - `DetectionSample` dataclass

**Config:** `config/pipeline/body_detection.yaml`, `config/pipeline/track_fusion.yaml`

**Feature Sandbox:** `FEATURES/body-tracking/`

**TODO Doc:** [docs/todo/feature_body_tracking_reid_fusion.md](../todo/feature_body_tracking_reid_fusion.md)

---

### 3. Embedding Engines (Priority 1)

| Component | Backend | Purpose |
|-----------|---------|---------|
| TensorRT ArcFace | tensorrtx / ONNX→TRT | GPU-optimized FP16 inference |
| PyTorch Reference | InsightFace_Pytorch | Training/evaluation baseline |
| ONNXRuntime C++ | Future | High-load microservice |

**Integration Points:**
- `tools/episode_run.py:1710` - `ArcFaceEmbedder` class (replace/extend)

**Storage:** S3/MinIO with versioning: `{model}-{version}-sm{arch}.trt`

**Config:** `config/pipeline/embedding.yaml`

**Feature Sandbox:** `FEATURES/embedding-engines/`

**TODO Doc:** [docs/todo/feature_arcface_tensorrt_onnxruntime.md](../todo/feature_arcface_tensorrt_onnxruntime.md)

---

### 4. Vision Analytics (Priority 3)

| Component | Library | Purpose |
|-----------|---------|---------|
| Face Mesh | MediaPipe | 468-point dense mesh |
| Visibility | custom | Face visibility fraction |
| Gaze | custom | Coarse gaze direction |
| CenterFace | CenterFace | CPU-friendly detector (future) |

**Config:** `config/pipeline/analytics.yaml`

**Feature Sandbox:** `FEATURES/vision-analytics/`

**TODO Doc:** [docs/todo/feature_mesh_and_advanced_visibility.md](../todo/feature_mesh_and_advanced_visibility.md)

---

## Data Schema Extensions

### Extended Track Schema (`tracks.jsonl`)

```json
{
  "track_id": 42,
  "track_type": "face",           // NEW: "face" | "body" | "fused"
  "face_track_id": 42,            // Original face track
  "body_track_id": 127,           // NEW: Associated body track
  "first_ts": 10.5,
  "last_ts": 45.2,
  "frame_count": 1040,
  "face_embedding": [...],        // 512-d ArcFace
  "body_embedding": [...],        // NEW: 256-d OSNet
  "alignment_quality_mean": 0.78, // NEW: LUVLi quality
  "head_pose_mean": {             // NEW: 3DDFA stats
    "yaw": -12.5,
    "pitch": 5.2,
    "roll": 2.1
  }
}
```

### Extended Identity Schema (`identities.json`)

```json
{
  "identity_id": "ID_001",
  "track_ids": [42, 127, 256],
  "person_id": "cast_lisa_barlow",
  "face_visible_duration": 125.5,  // NEW: seconds with face
  "body_only_duration": 45.2,      // NEW: seconds body-only
  "total_duration": 170.7,         // face + body
  "visibility_breakdown": {        // NEW: detailed stats
    "frontal": 0.65,
    "profile": 0.25,
    "back_of_head": 0.10
  }
}
```

---

## Risk & Rollback

| Feature | Config Flag | Rollback Behavior |
|---------|-------------|-------------------|
| FAN Alignment | `aligner: insightface` | Use 5-point InsightFace |
| Quality Gate | `alignment_gate_enabled: false` | Skip quality check |
| Body Tracking | `body_tracking_enabled: false` | Face-only mode |
| TensorRT | `embedding_backend: pytorch` | PyTorch fallback |
| Face Mesh | `mesh_enabled: false` | Skip mesh inference |

---

## Related Documentation

- [ACCEPTANCE_MATRIX.md](../../ACCEPTANCE_MATRIX.md) - Sections 3.7-3.15
- [FEATURES_GUIDE.md](../../FEATURES_GUIDE.md) - Feature sandbox workflow
- [docs/pipeline/overview.md](../pipeline/overview.md) - Existing pipeline docs

---

## Skills & Agents

| Skill | Purpose | Key Functions |
|-------|---------|---------------|
| `face-alignment` | Alignment debugging | `align_face_boxes()`, `compute_alignment_quality()` |
| `body-tracking` | Body tracking debugging | `track_persons_over_time()`, `associate_body_with_face_tracks()` |
| `embedding-engine` | Embedding optimization | `embed_faces_batched()`, `compare_embedding_backends()` |
| `vision-analytics` | Visibility analysis | `compute_visibility_labels()`, `generate_screen_time_breakdown()` |
| `qa-acceptance` | Acceptance testing | `run_acceptance_checks()`, `summarize_regressions_and_passes()` |

---

**Maintainers:** Screenalytics Engineering
