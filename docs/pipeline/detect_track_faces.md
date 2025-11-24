# Detect & Track Faces — Screenalytics Pipeline

**Version:** 2.0
**Last Updated:** 2025-11-18

---

## 1. Overview

The **Detect & Track** stage is the foundation of the Screenalytics visual pipeline. It transforms raw video frames into temporal face tracks by:

1. **Detection (RetinaFace):** Finding face bounding boxes and landmarks in each frame
2. **Tracking (ByteTrack + AppearanceGate):** Associating detections across frames into consistent tracks

This stage produces `detections.jsonl`, `tracks.jsonl`, and `track_metrics.json` artifacts that feed downstream embedding and clustering stages.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT: episode.mp4                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ VIDEO DECODE                                                      │
│ - FFmpeg/PyAV frame extraction                                   │
│ - Stride: Process every Nth frame                                │
│ - FPS override: Resample to target FPS                           │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ SCENE DETECTION (optional)                                        │
│ - PySceneDetect or internal HSV histogram                        │
│ - Detect hard cuts (scene changes)                               │
│ - Trigger tracker resets at scene boundaries                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ FACE DETECTION (RetinaFace)                                       │
│ - InsightFace RetinaFace R50 (ONNX)                              │
│ - Detect face bboxes + 5-point landmarks                         │
│ - Filter by min_size, confidence_th, NMS                         │
│ - Output: detections.jsonl (det_v1 schema)                       │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ FACE TRACKING (ByteTrack + AppearanceGate)                       │
│ - ByteTrack: IoU-based bbox association                          │
│ - AppearanceGate: Embedding-based track splitting                │
│ - Track lifecycle: birth, continuation, termination              │
│ - Output: tracks.jsonl (track_v1 schema)                         │
│ - Metrics: track_metrics.json                                    │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ OPTIONAL EXPORTERS                                                │
│ - --save-frames: Export sampled frames as JPG                    │
│ - --save-crops: Export face crops per track                      │
│ - JPEG quality configurable                                      │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ OUTPUTS                                                           │
│ - data/manifests/{ep_id}/detections.jsonl                        │
│ - data/manifests/{ep_id}/tracks.jsonl                            │
│ - data/manifests/{ep_id}/track_metrics.json                      │
│ - data/manifests/{ep_id}/progress.json (live updates)            │
│ - data/frames/{ep_id}/frames/*.jpg (optional)                    │
│ - data/frames/{ep_id}/crops/{track_id}/*.jpg (optional)          │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. CLI Usage

### 3.1 Basic Detection + Tracking
```bash
python tools/episode_run.py \
  --ep-id rhobh-s05e02 \
  --video data/videos/rhobh-s05e02/episode.mp4 \
  --stride 3 \
  --device auto
```

### 3.2 With Performance Profile (Recommended)
```bash
# CPU-safe (fanless devices, exploratory)
python tools/episode_run.py \
  --ep-id <ep_id> --video <path> \
  --profile fast_cpu

# Balanced (standard local dev)
python tools/episode_run.py \
  --ep-id <ep_id> --video <path> \
  --profile balanced

# High accuracy (GPU, max recall)
python tools/episode_run.py \
  --ep-id <ep_id> --video <path> \
  --profile high_accuracy \
  --device cuda
```

### 3.3 With Exporters
```bash
python tools/episode_run.py \
  --ep-id <ep_id> --video <path> \
  --save-frames \
  --save-crops \
  --jpeg-quality 90
```

### 3.4 Advanced Options
```bash
python tools/episode_run.py \
  --ep-id <ep_id> --video <path> \
  --detector retinaface \
  --tracker bytetrack \
  --stride 3 \
  --fps 8 \
  --device auto \
  --max-gap 30 \
  --scene-detector pyscenedetect \
  --scene-threshold 27.0 \
  --save-frames --save-crops
```

---

## 4. API Usage

### 4.1 Synchronous (SSE Streaming)
```bash
POST /jobs/detect_track
Content-Type: application/json

{
  "ep_id": "rhobh-s05e02",
  "profile": "balanced",
  "save_frames": true,
  "save_crops": true
}

# Response: SSE stream with progress events
# => {"phase": "detect", "frames_done": 120, "frames_total": 2400, ...}
# => {"phase": "track", "frames_done": 2400, "frames_total": 2400, ...}
# => {"phase": "done", "tracks_born": 42, "tracks_lost": 40, ...}
```

### 4.2 Asynchronous (Polling)
```bash
POST /jobs/detect_track_async
Content-Type: application/json

{
  "ep_id": "rhobh-s05e02",
  "profile": "balanced"
}

# Response:
{
  "job_id": "job-abc123",
  "state": "running",
  "ep_id": "rhobh-s05e02"
}

# Poll progress:
GET /jobs/job-abc123/progress
{
  "phase": "track",
  "frames_done": 1200,
  "frames_total": 2400,
  "elapsed_sec": 45.2,
  "eta_sec": 47.8,
  "fps_detected": 24.0,
  "analyzed_fps": 8.0
}

# Cancel:
POST /jobs/job-abc123/cancel
```

---

## 5. Configuration

### 5.1 Detection Config (`config/pipeline/detection.yaml`)
```yaml
model_id: retinaface_r50
min_size: 90              # Minimum face size in pixels
confidence_th: 0.8        # Detection confidence threshold
iou_th: 0.5               # NMS IoU threshold

# Adaptive confidence (fixes low-light/high-contrast)
adaptive_confidence: false
min_confidence: 0.6       # Min threshold for dark scenes
max_confidence: 0.9       # Max threshold for well-lit scenes

# NMS mode
nms_mode: hard            # "hard" or "soft"
soft_nms_sigma: 0.5       # Gaussian decay for soft-NMS

# Pose quality check
check_pose_quality: true
max_yaw_angle: 45.0       # Max head rotation (degrees)

# Letterbox detection
auto_crop_letterbox: false
letterbox_threshold: 20   # Pixel intensity for black bars
```

### 5.2 Tracking Config (`config/pipeline/tracking.yaml`)
```yaml
# ByteTrack spatial matching
track_thresh: 0.70        # Min confidence to track
match_thresh: 0.85        # IoU threshold for bbox matching
track_buffer: 90          # Frames to keep track alive (≈3-4s)
max_track_buffer: 300     # Maximum buffer cap

# Separate thresholds for new vs continuing tracks
new_track_thresh: 0.85    # Higher threshold for new tracks

# AppearanceGate (track splitting based on embeddings)
gate_enabled: true

# Appearance gate thresholds (via environment variables)
# TRACK_GATE_APPEAR_HARD: 0.65 (hard split if similarity < 65%)
# TRACK_GATE_APPEAR_SOFT: 0.75 (soft split if similarity < 75%)
# TRACK_GATE_APPEAR_STREAK: 3 (consecutive low-sim frames before split)
# TRACK_GATE_IOU: 0.50 (split if spatial jump, IoU < 50%)
# TRACK_GATE_EMB_EVERY: 24 (extract embeddings every 24 frames)

# Global Motion Compensation (GMC)
# SCREENALYTICS_GMC_METHOD: "sparseOptFlow" | "orb" | "off"
```

### 5.3 Performance Profiles (`config/pipeline/performance_profiles.yaml`)
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

---

## 6. Artifacts

### 6.1 `detections.jsonl` (det_v1 schema)
One JSON object per line, one line per detection:

```jsonl
{"ep_id": "rhobh-s05e02", "frame_idx": 42, "ts_s": 1.75, "bbox": [0.1, 0.2, 0.3, 0.4], "landmarks": [0.15, 0.25, 0.25, 0.25, 0.2, 0.3, 0.15, 0.35, 0.25, 0.35], "conf": 0.95, "model_id": "retinaface_r50_v1", "schema_version": "det_v1"}
```

**Fields:**
- `ep_id`: Episode identifier
- `frame_idx`: Zero-based frame number
- `ts_s`: Timestamp in seconds
- `bbox`: `[x1, y1, x2, y2]` normalized (0–1) coordinates
- `landmarks`: Flattened `[x, y] * 5` facial landmarks (left eye, right eye, nose, left mouth, right mouth)
- `conf`: Detector confidence (0–1)
- `model_id`: Model identifier (e.g., `"retinaface_r50_v1"`)
- `schema_version`: `"det_v1"`

### 6.2 `tracks.jsonl` (track_v1 schema)
One JSON object per line, one line per track:

```jsonl
{"track_id": "track-00001", "ep_id": "rhobh-s05e02", "start_s": 1.5, "end_s": 12.3, "frame_span": [30, 246], "sample_thumbs": ["crops/track-00001/frame_0030.jpg", "crops/track-00001/frame_0138.jpg"], "stats": {"detections": 216, "avg_conf": 0.92}, "schema_version": "track_v1"}
```

**Fields:**
- `track_id`: Deterministic track identifier (e.g., `"track-00001"`)
- `ep_id`: Episode identifier
- `start_s`, `end_s`: Timestamps for track span (seconds)
- `frame_span`: `[start_frame, end_frame]` (inclusive)
- `sample_thumbs`: List of thumbnail paths (relative to artifacts root)
- `stats`:
  - `detections`: Number of detections in this track
  - `avg_conf`: Average detection confidence
- `schema_version`: `"track_v1"`

### 6.3 `track_metrics.json`
Summary metrics for the entire episode:

```json
{
  "ep_id": "rhobh-s05e02",
  "tracks_born": 42,
  "tracks_lost": 40,
  "id_switches": 2,
  "longest_tracks": [
    {"track_id": "track-00005", "length": 512},
    {"track_id": "track-00012", "length": 384},
    {"track_id": "track-00003", "length": 256}
  ],
  "avg_tracks_per_frame": 2.3,
  "tracks_per_minute": 18.5,
  "short_track_fraction": 0.12,
  "total_frames": 2400,
  "analyzed_fps": 8.0,
  "elapsed_sec": 120.5,
  "device": "cpu",
  "detector": "retinaface",
  "tracker": "bytetrack",
  "scene_cuts": 5
}
```

**Key Metrics:**
- `tracks_born`: Total tracks created
- `tracks_lost`: Tracks terminated
- `id_switches`: Track ID reassignments (lower is better; indicates tracker instability)
- `longest_tracks`: Top 5 longest tracks (by frame count)
- `avg_tracks_per_frame`: Average concurrent tracks per frame
- `tracks_per_minute`: Derived metric (tracks_born / episode_duration_min); high values may indicate ghost tracks
- `short_track_fraction`: Fraction of tracks shorter than threshold (e.g., < 30 frames or < 1 second)
- `scene_cuts`: Number of scene changes detected (if scene detection enabled)

---

## 7. Metrics & Quality Gates

### 7.1 Acceptable Ranges (CPU)
| Metric | Target | Warning Threshold | Indicates |
|--------|--------|-------------------|-----------|
| **tracks_per_minute** | 10–30 | > 50 | Too many ghost tracks; increase `track_thresh` or `min_box_area` |
| **short_track_fraction** | < 0.20 | > 0.30 | Too many fleeting tracks; increase `min_track_len` or `track_buffer` |
| **id_switch_rate** | < 0.05 | > 0.10 | Tracker instability; adjust `match_thresh` or `track_buffer` |
| **Runtime (1hr episode)** | ≤ 3× realtime | > 5× realtime | Too slow; use faster profile or increase `stride` |

### 7.2 Acceptable Ranges (GPU)
| Metric | Target | Warning Threshold | Indicates |
|--------|--------|-------------------|-----------|
| **tracks_per_minute** | 10–30 | > 50 | Same as CPU |
| **short_track_fraction** | < 0.15 | > 0.25 | Same as CPU |
| **id_switch_rate** | < 0.03 | > 0.08 | Same as CPU |
| **Runtime (1hr episode)** | ≤ 10 min | > 15 min | Too slow; check GPU utilization |

### 7.3 Guardrails
The pipeline emits **warnings** if:
- `tracks_per_minute > 50` → "Too many tracks detected; consider increasing track_thresh or adjusting scene detection"
- `short_track_fraction > 0.3` → "High fraction of short tracks; consider increasing min_track_len or track_buffer"
- `id_switch_rate > 0.1` → "High ID switch rate; tracker may be unstable; review match_thresh and appearance gate settings"

---

## 8. Good vs Bad Runs

### 8.1 Example: Good Run
```json
{
  "ep_id": "rhobh-s05e02",
  "tracks_born": 28,
  "tracks_lost": 26,
  "id_switches": 1,
  "longest_tracks": [
    {"track_id": "track-00005", "length": 512},
    {"track_id": "track-00012", "length": 384}
  ],
  "avg_tracks_per_frame": 1.8,
  "tracks_per_minute": 12.3,
  "short_track_fraction": 0.08,
  "scene_cuts": 3
}
```
**Why it's good:**
- `tracks_per_minute = 12.3` (reasonable for TV episode)
- `short_track_fraction = 0.08` (very low; most tracks are substantial)
- `id_switches = 1` (stable tracker)
- `scene_cuts = 3` (reasonable scene changes)

### 8.2 Example: Bad Run (Too Many Tracks)
```json
{
  "ep_id": "rhobh-s05e02",
  "tracks_born": 10842,
  "tracks_lost": 10840,
  "id_switches": 58,
  "avg_tracks_per_frame": 12.5,
  "tracks_per_minute": 482.1,
  "short_track_fraction": 0.87
}
```
**Why it's bad:**
- `tracks_per_minute = 482.1` ⚠️ **WAY too high** (likely ghost tracks from background motion or low `track_thresh`)
- `short_track_fraction = 0.87` ⚠️ **87% of tracks are fleeting** (< 1 second)
- `id_switches = 58` ⚠️ **Tracker very unstable**

**Fixes:**
1. Increase `track_thresh` from 0.70 to 0.80+
2. Increase `new_track_thresh` from 0.85 to 0.90+
3. Increase `min_box_area` to filter small background faces
4. Enable `check_pose_quality: true` to discard unreliable landmarks
5. Adjust scene detection to reduce false positives

---

## 9. Common Issues & Troubleshooting

### 9.1 Too Many Tracks (Exploding Track Count)
**Symptom:** `tracks_per_minute > 50`, `short_track_fraction > 0.5`

**Causes:**
- `track_thresh` too low (allowing low-confidence detections to create tracks)
- `min_box_area` too small (tracking tiny background faces)
- Scene detection off or too sensitive (creating tracker resets on every frame)

**Fixes:**
1. Increase `track_thresh` to 0.75–0.85
2. Increase `new_track_thresh` to 0.85–0.90
3. Set `min_box_area: 400` (px²) to filter small faces
4. Enable scene detection with appropriate threshold (`scene_threshold: 27.0` for PySceneDetect)
5. Increase `min_track_len` filter (drop tracks < N frames)

### 9.2 Missed Faces (Low Recall)
**Symptom:** Known faces not detected or tracked

**Causes:**
- `stride` too high (skipping frames)
- `min_size` too large (filtering small/distant faces)
- `confidence_th` too high (rejecting valid detections)

**Fixes:**
1. Decrease `stride` from 10 → 3 or 1
2. Decrease `min_size` from 90 → 64 or 48
3. Decrease `confidence_th` from 0.8 → 0.6 (but watch for false positives)
4. Enable `adaptive_confidence: true` for low-light scenes

### 9.3 ID Switches (Tracker Instability)
**Symptom:** `id_switch_rate > 0.1`, tracks frequently reassigned

**Causes:**
- `match_thresh` too low (allowing spurious associations)
- `track_buffer` too short (tracks dying and reviving with new IDs)
- Appearance gate too aggressive (splitting tracks on temporary pose changes)

**Fixes:**
1. Increase `match_thresh` to 0.85–0.90
2. Increase `track_buffer` to 120–180 frames
3. Adjust appearance gate thresholds:
   - Decrease `TRACK_GATE_APPEAR_HARD` to 0.60 (less aggressive splitting)
   - Increase `TRACK_GATE_APPEAR_STREAK` to 5 (require more consecutive low-sim frames)
4. Enable GMC (Global Motion Compensation) for camera pans

### 9.4 Blank/Gray Crops
**Symptom:** Face crops are blank or gray rectangles

**Causes:**
- Out-of-bounds bounding boxes
- Invalid landmarks
- Crop generation logic bug

**Fixes:**
1. Enable `check_pose_quality: true` to discard unreliable landmarks
2. Ensure bbox clamping in crop generation code
3. Review `crops_debug.jsonl` (if `DEBUG_THUMBS=1`) to identify failed crops
4. See [faces_harvest.md](faces_harvest.md) for crop debugging

### 9.5 Performance Issues (Overheating, Slow Processing)
**Symptom:** CPU thermal throttling, fans at max, processing < 1 FPS

**Causes:**
- Profile too aggressive for hardware (e.g., `high_accuracy` on fanless MacBook Air)
- Too many concurrent threads (PyTorch, ONNX, OpenCV all using default thread counts)
- Exporters enabled (writing thousands of frames/crops to disk)

**Fixes:**
1. Use `--profile fast_cpu` for thermally constrained devices
2. Limit threading:
   ```bash
   export OMP_NUM_THREADS=2
   export MKL_NUM_THREADS=2
   python tools/episode_run.py ...
   ```
3. Disable exporters (`--no-save-frames --no-save-crops`)
4. Increase `stride` to 5–10 for exploratory passes
5. See [docs/ops/performance_tuning_faces_pipeline.md](../ops/performance_tuning_faces_pipeline.md)

---

## 10. Device Selection

### 10.1 Auto-Detection
```bash
python tools/episode_run.py --ep-id <ep_id> --video <path> --device auto
```
**Order:** CUDA GPU → Apple CoreML (MPS) → CPU

### 10.2 Explicit Selection
```bash
# Force CPU
python tools/episode_run.py --device cpu

# Force CUDA GPU
python tools/episode_run.py --device cuda

# Force Apple Silicon (MPS/CoreML)
python tools/episode_run.py --device mps
```

### 10.3 CoreML-Specific Tuning (macOS)
```bash
# Reduce CoreML input size for thermal management
python tools/episode_run.py --device coreml --coreml-det-size 384
```

---

## 11. Scene Detection

### 11.1 Modes
```bash
# PySceneDetect (recommended, cleanest resets)
python tools/episode_run.py --scene-detector pyscenedetect --scene-threshold 27.0

# Internal HSV histogram (legacy fallback)
python tools/episode_run.py --scene-detector internal --scene-threshold 0.5

# Disable scene detection
python tools/episode_run.py --scene-detector off
```

### 11.2 Parameters
- `--scene-threshold`: Sensitivity (PySceneDetect: 20–30; internal: 0–2)
- `--scene-min-len`: Minimum scene length (frames)
- `--scene-warmup-dets`: Number of warmup detections after scene cut

---

## 12. Integration Tests

### 12.1 Run Real Detection Test
```bash
RUN_ML_TESTS=1 pytest tests/ml/test_detect_track_real.py -v
```

**What it tests:**
- RetinaFace detection on sample clip
- ByteTrack association
- Artifact schemas (`detections.jsonl`, `tracks.jsonl`)
- Quality metrics within acceptable ranges

### 12.2 Expected Assertions
```python
assert track_metrics["tracks_per_minute"] < 50
assert track_metrics["short_track_fraction"] < 0.3
assert track_metrics["id_switch_rate"] < 0.1
assert len(tracks) > 0
```

---

## 13. References

- [Pipeline Overview](overview.md) — Full pipeline stages
- [Faces Harvest](faces_harvest.md) — Embedding and crop generation
- [Cluster Identities](cluster_identities.md) — Identity grouping
- [Performance Tuning](../ops/performance_tuning_faces_pipeline.md) — Speed vs accuracy tuning
- [Troubleshooting](../ops/troubleshooting_faces_pipeline.md) — Common issues
- [Artifact Schemas](../reference/artifacts_faces_tracks_identities.md) — Complete schema reference
- [Config Reference](../reference/config/pipeline_configs.md) — Key-by-key config docs

---

**Maintained by:** Screenalytics Engineering
**Next Review:** After Phase 2 completion
