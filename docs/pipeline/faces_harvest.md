# Faces Harvest (Embedding) — Screenalytics Pipeline

**Version:** 2.0
**Last Updated:** 2025-11-18

---

## 1. Overview

The **Faces Harvest** (embedding) stage extracts face crops from video frames and generates **512-dimensional unit-norm embeddings** using ArcFace. This stage bridges detection/tracking and identity clustering by:

1. Generating face crops from RetinaFace bounding boxes and landmarks
2. Filtering crops by quality (confidence, size, blur, pose)
3. Extracting ArcFace embeddings (ONNX)
4. Limiting volume per track/episode to avoid embedding thousands of near-identical faces

**Outputs:** `faces.jsonl`, `faces.npy`, and optional face crops (`crops/{track_id}/*.jpg`).

---

## 2. Architecture

```
tracks.jsonl + episode.mp4
    ↓
[CROP GENERATION]
- Read tracks from tracks.jsonl
- For each track, sample frames per sampling strategy
- Extract face crops using RetinaFace bbox + landmarks
- Apply quality gating (confidence, size, blur, pose)
    ↓
[EMBEDDING EXTRACTION]
- Resize crops to 112x112 (ArcFace input size)
- Normalize pixels
- Run ArcFace ONNX model
- Produce 512-d unit-norm embeddings
    ↓
[ARTIFACTS]
- faces.jsonl (one entry per face, with track_id linkage)
- faces.npy (Nx512 float32 matrix)
- crops/{track_id}/*.jpg (optional, for UI/debugging)
```

---

## 3. CLI Usage

### 3.1 Basic Embedding (No Crops)
```bash
python tools/episode_run.py --ep-id <ep_id> --faces-embed
```

### 3.2 With Face Crops Export
```bash
python tools/episode_run.py --ep-id <ep_id> --faces-embed --save-crops
```

### 3.3 Quality Gating
```bash
python tools/episode_run.py --ep-id <ep_id> --faces-embed \
  --min-face-conf 0.8 \
  --min-face-size 64 \
  --face-validate
```

### 3.4 Volume Control
```bash
python tools/episode_run.py --ep-id <ep_id> --faces-embed \
  --max-crops-per-track 50 \
  --save-crops
```

---

## 4. API Usage

```bash
POST /jobs/faces_embed
Content-Type: application/json

{
  "ep_id": "rhobh-s05e02",
  "save_crops": true,
  "min_quality": 0.7,
  "max_crops_per_track": 50
}

# Response: SSE stream or async job_id
```

---

## 5. Configuration

### 5.1 Sampling Config (`config/pipeline/faces_embed_sampling.yaml`)
```yaml
# Quality gating
min_quality: 0.7          # Combined quality score threshold
min_confidence: 0.8       # RetinaFace detection confidence
min_face_size: 64         # Minimum face size (pixels)
max_blur: 100             # Laplacian variance threshold (lower = more blur)

# Volume control
max_crops_per_track: 50   # Limit crops per track
max_crops_per_episode: 5000  # Global episode limit (optional)

# Sampling strategy
sampling_mode: uniform    # "uniform" | "quality-weighted" | "stratified"
sample_interval: 24       # Sample every Nth frame (uniform mode)

# Crop generation
thumb_size: 224           # Output crop size (before resize to 112x112 for ArcFace)
jpeg_quality: 90          # JPEG quality for saved crops

# Pose filtering
check_pose_quality: true
max_yaw_angle: 45.0       # Max head rotation (degrees)
```

---

## 6. Artifacts

### 6.1 `faces.jsonl`
One JSON object per line, one line per face:

```jsonl
{"face_id": "face-00001", "track_id": "track-00001", "frame_idx": 42, "bbox": [0.1, 0.2, 0.3, 0.4], "landmarks": [0.15, 0.25, 0.25, 0.25, 0.2, 0.3, 0.15, 0.35, 0.25, 0.35], "embedding": [0.012, -0.045, ...], "quality_score": 0.87, "crop_path": "crops/track-00001/frame_0042.jpg"}
```

**Fields:**
- `face_id`: Unique face identifier (e.g., `"face-00001"`)
- `track_id`: Links back to `tracks.jsonl`
- `frame_idx`: Zero-based frame number
- `bbox`: `[x1, y1, x2, y2]` normalized (0–1) coordinates
- `landmarks`: Flattened `[x, y] * 5` facial landmarks
- `embedding`: 512-d float32 array (unit-norm)
- `quality_score`: Combined quality metric (0–1)
- `crop_path`: Relative path to face crop (optional, empty if `--save-crops` not used)
- `rejection_reason`: (Optional) If face was rejected, reason (e.g., `"low_quality"`, `"pose_too_extreme"`)

### 6.2 `faces.npy`
NumPy array of shape `(N, 512)`, dtype `float32`, where each row is a unit-norm embedding corresponding to the same row index in `faces.jsonl`.

```python
import numpy as np
embeddings = np.load("data/embeds/{ep_id}/faces.npy")
print(embeddings.shape)  # (N, 512)
print(np.linalg.norm(embeddings[0]))  # ≈ 1.0 (unit norm)
```

### 6.3 Face Crops (`crops/{track_id}/*.jpg`)
Face crops saved as JPEG images under `data/frames/{ep_id}/crops/{track_id}/`.

**Naming:** `frame_{frame_idx:06d}.jpg`

**Example:**
```
data/frames/rhobh-s05e02/crops/
  track-00001/
    frame_000042.jpg
    frame_000084.jpg
    ...
  track-00002/
    frame_000056.jpg
    ...
```

---

## 7. Quality Gating

### 7.1 Quality Score Computation
```python
quality_score = (
    0.4 * detection_confidence +
    0.3 * size_score +
    0.2 * blur_score +
    0.1 * pose_score
)
```

**Components:**
- **detection_confidence:** RetinaFace confidence (0–1)
- **size_score:** Normalized face size (larger = better; capped at 1.0)
- **blur_score:** Laplacian variance normalized (higher variance = sharper = better)
- **pose_score:** Based on yaw angle from landmarks (frontal = 1.0, extreme = 0.0)

> **⚠️ CURRENT LIMITATION: Pose/Expression Gating Disabled**
> 
> Pose and expression extraction is **not currently implemented** in `tools/episode_run.py`.
> The `_analyze_pose_expression()` function returns `(None, None, None)`, which means:
> - **pose_score** is effectively 0 in the quality score calculation
> - Yaw/pitch angle checks are **skipped** (no pose-based rejection)
> - Expression filtering is **disabled**
> 
> **Impact on Quality:**
> - Profile views, extreme head rotations, and unusual expressions are included
> - May reduce clustering accuracy by mixing frontal and profile embeddings
> - ArcFace embeddings from extreme poses can differ significantly from frontal views
> 
> **Configuration Exists But Not Enforced:**
> - `max_yaw_angle: 45.0` (degrees) - defined but not checked
> - `max_pitch_angle: 30.0` (degrees) - defined but not checked
> - `allowed_expressions: [neutral, smile, happy, unknown]` - defined but not checked
> 
> See [tools/episode_run.py:628-685](../tools/episode_run.py) for implementation details.

### 7.2 Rejection Criteria
Faces are **rejected** (not embedded) if:
- `quality_score < min_quality` (default: 0.7)
- `detection_confidence < min_confidence` (default: 0.8)
- Face size < `min_face_size` pixels (default: 64)
- Blur variance < `max_blur` threshold
- Yaw angle > `max_yaw_angle` degrees (default: 45°) when `check_pose_quality: true`

**Rejected faces:**
- Not written to `faces.jsonl`
- Not embedded
- Optionally logged to `crops_debug.jsonl` (if `DEBUG_THUMBS=1`)

---

## 8. Volume Control

### 8.1 Per-Track Limits
To avoid embedding thousands of near-identical crops from long tracks:

```yaml
max_crops_per_track: 50
```

**Sampling strategies:**
- **uniform:** Sample every Nth frame along track timeline
- **quality-weighted:** Prefer high-quality frames (high confidence, low blur, frontal pose)
- **stratified:** Divide track into time bins, sample uniformly within each bin

### 8.2 Per-Episode Limits (Optional)
```yaml
max_crops_per_episode: 5000
```

Limits total crops across all tracks. If limit exceeded, lowest-quality crops are dropped first.

---

## 9. Crop Generation

### 9.1 Alignment
Crops are generated using **RetinaFace landmarks** for consistent face alignment:

1. Detect 5-point landmarks (left eye, right eye, nose, left mouth, right mouth)
2. Compute affine transform to canonical pose
3. Apply transform and crop face region
4. Resize to `thumb_size` (default: 224x224) before saving
5. Resize to 112x112 for ArcFace input

### 9.2 Handling Out-of-Bounds Boxes
- **Clamp bbox:** If bbox extends beyond frame boundaries, clamp to `[0, 1]`
- **Reject if too small:** If clamped bbox results in face < `min_face_size` pixels, reject
- **No blank crops:** Never write gray/blank rectangles

### 9.3 Debugging Blank Crops
If crops are blank or gray:

1. Enable debug logging:
   ```bash
   DEBUG_THUMBS=1 python tools/episode_run.py --ep-id <ep_id> --faces-embed --save-crops
   ```
2. Review `data/frames/{ep_id}/crops_debug.jsonl`:
   ```jsonl
   {"frame_idx": 42, "track_id": "track-00001", "bbox": [0.1, 0.2, 0.3, 0.4], "crop_path": "crops/track-00001/frame_000042.jpg", "status": "success", "quality_score": 0.87}
   {"frame_idx": 56, "track_id": "track-00002", "bbox": [-0.1, 0.2, 0.05, 0.4], "crop_path": "", "status": "rejected", "reason": "bbox_out_of_bounds"}
   ```
3. Analyze failures:
   ```bash
   python tools/debug_thumbs.py data/frames/{ep_id}/crops_debug.jsonl
   ```

---

## 10. Embedding Extraction

### 10.1 Model
- **ArcFace ONNX:** `arcface_r100_v1` (InsightFace)
- **Input:** 112x112 RGB face crop, normalized
- **Output:** 512-d float32 vector (unit-norm)

### 10.2 Normalization
```python
# Pixel normalization
crop_rgb = crop_bgr[:, :, ::-1]  # BGR → RGB
crop_normalized = (crop_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5

# Embedding normalization
embedding_raw = arcface_model.run(crop_normalized)
embedding = embedding_raw / np.linalg.norm(embedding_raw)  # Unit-norm
```

### 10.3 Batch Processing (Optional)
For GPU efficiency, faces can be batched:

```yaml
embed_batch_size: 8  # Process 8 faces per batch
```

---

## 11. Performance

### 11.1 Timing Breakdown (Typical Episode)
| Stage | Time (CPU) | Time (GPU) |
|-------|------------|------------|
| Crop generation | 15s | 10s |
| ArcFace embedding | 120s | 25s |
| Write artifacts | 5s | 5s |
| **Total** | **140s** | **40s** |

### 11.2 Optimization Tips
1. **Reduce crops per track:** Lower `max_crops_per_track` from 50 → 20
2. **Increase sampling interval:** Sample every 24 frames instead of 12
3. **Disable crop export:** Skip `--save-crops` (only embed, no disk writes)
4. **Use GPU:** `--embed-device cuda` for 3–5× speedup
5. **Quality-weighted sampling:** Prefer high-quality frames, skip low-quality

---

## 12. Common Issues

### 12.1 Blank/Gray Crops
**Symptom:** Face crops are blank rectangles

**Causes:**
- Out-of-bounds bounding boxes
- Invalid landmarks
- Crop generation bug

**Fixes:**
1. Enable `check_pose_quality: true`
2. Review `crops_debug.jsonl` (with `DEBUG_THUMBS=1`)
3. Ensure bbox clamping in crop generation code
4. Check RetinaFace detection quality (may need re-detection with better settings)

### 12.2 Too Many Faces (Memory Exhaustion)
**Symptom:** OOM error during embedding

**Causes:**
- No volume control (embedding all crops from long episode)
- `max_crops_per_track` too high

**Fixes:**
1. Set `max_crops_per_track: 50` (or lower)
2. Set `max_crops_per_episode: 5000`
3. Use quality-weighted sampling (embed best frames only)

### 12.3 Embedding Dimension Mismatch
**Symptom:** `faces.npy` shape is not `(N, 512)`

**Causes:**
- Wrong ArcFace model loaded
- Model output not unit-normalized

**Fixes:**
1. Confirm model ID: `arcface_r100_v1` (512-d output)
2. Verify normalization: `np.linalg.norm(embeddings[i]) ≈ 1.0`

### 12.4 Poor Clustering Results
**Symptom:** Clustering produces too many singletons or over-merged clusters

**Causes:**
- Low-quality embeddings (blur, occlusion, extreme pose)
- Too few crops per track (under-sampling)

**Fixes:**
1. Increase `min_quality` threshold (0.7 → 0.8)
2. Enable `check_pose_quality: true`
3. Increase `max_crops_per_track` (20 → 50) for better track-level pooling
4. Review `quality_score` distribution in `faces.jsonl`

---

## 13. Integration Tests

### 13.1 Run Faces Embed Test
```bash
RUN_ML_TESTS=1 pytest tests/ml/test_faces_embed.py -v
```

**What it tests:**
- Crop generation from tracks
- ArcFace embedding extraction
- Artifact schemas (`faces.jsonl`, `faces.npy`)
- Quality gating behavior
- Volume control (max_crops_per_track)

### 13.2 Expected Assertions
```python
assert faces_jsonl_exists
assert faces_npy_shape == (N, 512)
assert all(np.isclose(np.linalg.norm(embeddings[i]), 1.0) for i in range(N))
assert all(face["quality_score"] >= min_quality for face in faces)
assert len(faces_per_track) <= max_crops_per_track
```

---

## 14. References

- [Pipeline Overview](overview.md) — Full pipeline stages
- [Detect & Track](detect_track_faces.md) — Detection and tracking
- [Cluster Identities](cluster_identities.md) — Identity grouping
- [Artifact Schemas](../reference/artifacts_faces_tracks_identities.md) — Complete schema reference
- [Config Reference](../reference/config/pipeline_configs.md) — Key-by-key config docs
- [Troubleshooting](../ops/troubleshooting_faces_pipeline.md) — Common issues

---

**Maintained by:** Screenalytics Engineering
**Next Review:** After Phase 2 completion
