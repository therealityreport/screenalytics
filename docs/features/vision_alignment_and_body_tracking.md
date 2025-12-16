# Vision Alignment & Body Tracking

Version: 1.1
Status: In Development
Last Updated: 2025-12-16
Tags: face_alignment, arcface_tensorrt, vision_analytics, body_tracking

---

## Overview

This document describes the extension of Screenalytics from face-only tracking to a comprehensive person tracking system that includes:

1. **Robust Face Alignment** - FAN 68-point 2D landmarks (implemented); quality gating is a heuristic stub; 3DDFA_V2 is planned-only.
2. **Body Tracking + Person Re-ID** - Optional FEATURE sandbox for tracking cast members when faces are not visible (YOLO + OSNet + fusion).
3. **Optimized Embedding Engines** - ArcFace TensorRT work is scaffold-only today; real GPU parity/testing and integration are future work.
4. **Advanced Visibility Analytics** - Vision analytics (mesh/gaze/visibility) is planned-only today (docs/config only).

---

## Implementation Status Snapshot (current)

This is the blunt “what exists right now” snapshot (scaffold ≠ production-ready):

- **Face Alignment (`face_alignment`)** — `partial`
  - FAN 68-point landmarks: implemented and wired in the embedding (faces_embed / “harvest”) stage.
  - “LUVLi quality”: heuristic gate only (not a real LUVLi model).
  - “3D alignment / 3DDFA”: not started; current “pose” is a solvePnP placeholder from FAN landmarks.
- **Body Tracking + Re-ID (`body_tracking`)** — `partial`
  - YOLO person detection + ByteTrack tracking: implemented behind an opt-in config flag.
  - OSNet Re-ID + face↔body fusion + screentime comparison: implemented, but optional/best-effort (heavy deps).
  - Default behavior remains face-only unless body artifacts exist.
- **ArcFace TensorRT (`arcface_tensorrt`)** — `scaffold_only`
  - Code is wired and can be selected as an embedding backend, but treat it as experimental until GPU parity/eval is proven.
  - When TensorRT/PyCUDA/engine is missing, the pipeline falls back to the PyTorch baseline.
- **Vision Analytics (`vision_analytics`)** — `not_started`
  - Docs/config only; no runnable implementation or artifacts yet.

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

Note: the diagram below shows a **target** architecture; several components are partial/scaffold/planned-only today (see status snapshot above).

```
episode.mp4 / decoded frames
    │
    ├─► [1] Face Detect (RetinaFace) ─► detections.jsonl
    │        │
    │        └─► [2] Face Track (ByteTrack) ─► tracks.jsonl
    │                 │
    │                 └─► [3] Face Embed (“harvest”)
    │                       ├─ (optional) Face Alignment (FAN 68) ─► face_alignment/aligned_faces.jsonl
    │                       ├─ (optional) Quality Gate (heuristic “LUVLi”) ─► drop/downweight
    │                       └─ Embedding Engine (PyTorch baseline / TensorRT experimental) ─► faces.jsonl + faces.npy
    │
    ├─► [4] Cluster Identities ─► identities.json
    │
    ├─► (optional) Body Tracking + Re-ID (FEATURE sandbox; heavy deps)
    │        ├─ Person Detect (YOLO) ─► body_tracking/body_detections.jsonl
    │        ├─ Person Track (ByteTrack) ─► body_tracking/body_tracks.jsonl
    │        ├─ Person Re-ID (OSNet) ─► body_tracking/body_embeddings.*
    │        └─ Face↔Body Fusion ─► body_tracking/track_fusion.json + screentime_comparison.json
    │
    ├─► (planned) Vision Analytics (mesh/gaze/visibility) ─► vision_analytics/* (planned)
    │
    └─► [5] Screentime Analyze
           ├─ identities + tracks + faces
           ├─ + body_tracking/screentime_comparison.json (if present)
           └─ + vision_analytics/* (planned)
             └─► screentime.json / screentime.csv
```

---

## Feature Modules

**Current reality (as of this doc update):**
- These modules are still treated as **FEATURE / best-effort** capabilities (optional deps, safe fallbacks).
- The main pipeline stays operational in face-only mode and **consumes extra outputs only when the artifacts exist**:
  - Embedding can use `face_alignment/aligned_faces.jsonl` when present (otherwise skips alignment gating).
  - Screentime can use `body_tracking/screentime_comparison.json` when present (otherwise face-only metrics).
- ArcFace TensorRT is wired as an embedding backend, but should be considered experimental until parity is demonstrated on target GPUs.
- Vision analytics (mesh/visibility/gaze) is planned-only today (docs/config placeholders; no runnable implementation).

See also:
- Machine-readable snapshot: [docs/_meta/feature_status.json](../_meta/feature_status.json)
- Historical audit (may be stale): [docs/plans/complete/audit/vision_alignment_body_tracking_status.md](../plans/complete/audit/vision_alignment_body_tracking_status.md)

### Truth Table (artifact-gated consumption)

The “umbrella” rule is simple: **if an artifact exists, downstream stages may consume it; if it doesn’t, the pipeline falls back**.

| Output artifact | Produced by | Consumed by | If missing |
|---|---|---|---|
| `face_alignment/aligned_faces.jsonl` | faces_embed (or legacy `python -m FEATURES.face_alignment`) | faces_embed gating + export/debug | Skip alignment gating; embed proceeds normally |
| `body_tracking/body_tracks.jsonl` | body_tracking (opt-in) | body_tracking_fusion | Skip fusion/comparison |
| `body_tracking/track_fusion.json` | body_tracking_fusion | export/debug + UI (optional) | No fused face↔body association |
| `body_tracking/screentime_comparison.json` | body_tracking_fusion | screentime analytics/UI | Face-only screentime metrics |
| `vision_analytics/*` | (planned-only) | (none yet) | No visibility analytics inputs |

### 1. Face Alignment (Priority 1)

| Component | Library | Purpose |
|-----------|---------|---------|
| FAN 2D | face-alignment 1.3.5+ | 68-point 2D landmarks |
| LUVLi (style) | heuristic + scaffolding | Uncertainty/visibility fields (not true LUVLi yet) |
| 3DDFA_V2 | planned | 3D head pose (selective) |

**Integration Points:**
- Default path: runs inside the embedding (faces_embed) stage when `config/pipeline/face_alignment.yaml` enables it.
- Requires optional dependency: `face-alignment` (`pip install face-alignment`)
- Legacy sandbox runner: `python -m FEATURES.face_alignment`
- Pipeline consumption: `_load_alignment_quality_index()` in `tools/episode_run.py`

**Config:** `config/pipeline/face_alignment.yaml` (canonical)

**Feature Sandbox:** `FEATURES/face_alignment/`

**TODO Doc:** [docs/plans/in_progress/feature_face_alignment_fan_luvli_3ddfa.md](../plans/in_progress/feature_face_alignment_fan_luvli_3ddfa.md)

---

### 2. Body Tracking + Re-ID (Priority 2)

| Component | Library | Purpose |
|-----------|---------|---------|
| Person Detector | ultralytics 8.2+ | YOLOv8 COCO person class |
| Person Tracker | ByteTrack | Temporal consistency |
| Person Re-ID | torchreid 1.4+ | OSNet 256-d embeddings |
| Track Fusion | custom | Face↔body association |

**Integration Points:**
- Pipeline (opt-in): `tools/episode_run.py` runs best-effort body tracking + fusion when `config/pipeline/body_detection.yaml` enables it.
- Standalone sandbox runner: `FEATURES/body_tracking/src/body_tracking_runner.py` (`python -m FEATURES.body_tracking`)
- Screentime reads comparison artifact (if present): `apps/api/services/screentime.py`

**Config:** `config/pipeline/body_detection.yaml`, `config/pipeline/track_fusion.yaml`

**Feature Sandbox:** `FEATURES/body_tracking/`

**TODO Doc:** [docs/plans/in_progress/feature_body_tracking_reid_fusion.md](../plans/in_progress/feature_body_tracking_reid_fusion.md)

---

### 3. Embedding Engines (Priority 1)

| Component | Backend | Purpose |
|-----------|---------|---------|
| TensorRT ArcFace | ONNX→TRT | Scaffold for GPU-optimized FP16 inference (pending GPU parity test/eval) |
| PyTorch/ONNX Runtime (in-proc) | `insightface` Python package | Reference runtime (models via `scripts/fetch_models.py`) |
| ONNXRuntime C++ | planned | High-load microservice (future architecture) |

**Integration Points:**
- Backend selection: `config/pipeline/embedding.yaml` + `tools/episode_run.py`
- TensorRT implementation: `FEATURES/arcface_tensorrt/src/`

**Storage:** S3/MinIO with versioning: `{model}-{version}-sm{arch}.trt`

**Config:** `config/pipeline/embedding.yaml`

**Feature Sandbox:** `FEATURES/arcface_tensorrt/` (implementation), `FEATURES/embedding_engines/` (docs-only planning)

**TODO Doc:** [docs/plans/in_progress/feature_arcface_tensorrt_onnxruntime.md](../plans/in_progress/feature_arcface_tensorrt_onnxruntime.md)

---

### 4. Vision Analytics (Priority 3)

| Component | Library | Purpose |
|-----------|---------|---------|
| Face Mesh | MediaPipe | 468-point dense mesh |
| Visibility | custom | Face visibility fraction |
| Gaze | custom | Coarse gaze direction |
| CenterFace | CenterFace | CPU-friendly detector (future) |

**Config:** `config/pipeline/analytics.yaml`

**Feature Sandbox:** `FEATURES/vision_analytics/` (docs-only today)

**TODO Doc:** [docs/plans/in_progress/feature_mesh_and_advanced_visibility.md](../plans/in_progress/feature_mesh_and_advanced_visibility.md)

---

## Data Schema Extensions (Proposed / Future)

The schemas below describe a **target** model, not a production guarantee today.
Current implementations write dedicated artifacts under `data/manifests/{ep_id}/face_alignment/` and
`data/manifests/{ep_id}/body_tracking/` (see the status snapshot / `docs/_meta/feature_status.json`).

### Proposed Track Schema (`tracks.jsonl`)

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

### Proposed Identity Schema (`identities.json`)

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
| FAN Alignment | `config/pipeline/face_alignment.yaml` → `face_alignment.enabled: false` | Do not generate alignment artifacts |
| Quality Gate | `config/pipeline/embedding.yaml` → `face_alignment.enabled: false` or `min_alignment_quality: 0.0` | Disable gating |
| Body Tracking | `config/pipeline/body_detection.yaml` → `body_tracking.enabled: false` | Face-only mode |
| TensorRT | `config/pipeline/embedding.yaml` → `embedding.backend: pytorch` | PyTorch fallback |
| Face Mesh | `config/pipeline/analytics.yaml` → `face_mesh.enabled: false` | Skip mesh inference (planned-only today) |

---

## Related Documentation

- [ACCEPTANCE_MATRIX.md](../../ACCEPTANCE_MATRIX.md) - Sections 3.7-3.15
- [Feature sandboxes](feature_sandboxes.md) - Feature sandbox workflow
- [docs/pipeline/overview.md](../pipeline/overview.md) - Existing pipeline docs
- [docs/plans/complete/audit/vision_alignment_body_tracking_status.md](../plans/complete/audit/vision_alignment_body_tracking_status.md) - Code-vs-docs audit

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
