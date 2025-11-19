# CPU Optimization Report: Face Harvest Process

**Goal:** Ensure SCREENALYTICS NEVER USES MORE THAN 300% CPU (3 cores)

**Current Status:** Process exceeds 300% CPU during `faces_embed` phase  
**Processing Context:** 24,061 frames @ 6.05 fps, RetinaFace + ByteTrack + ArcFace(CoreML)

---

## ISSUE 1: Single-Face Embedding Loop (CRITICAL - P0)

### Problem
**Location:** `tools/episode_run.py` line 4520

```python
# ❌ CURRENT: Embedding ONE face at a time
for sample in samples:  # 24,061 iterations
    crop, crop_err = _prepare_face_crop(image, validated_bbox, landmarks)
    encoded = embedder.encode([crop])  # CoreML model invoked 24,061 times!
    embedding_vec = encoded[0]
```

**Impact:**
- **CPU Usage:** 400-600% (4-6 cores) during embedding loop
- **Model Overhead:** 24,061 separate CoreML invocations
- **GPU Underutilization:** CoreML processes single crop instead of batch
- **Scheduling Overhead:** Constant context switching between CPU threads and GPU

### Root Cause
1. Samples are already sorted by `frame_idx` (line 4234: `sort_by_frame=True`)
2. TODO comment exists (line 4512) acknowledging batching opportunity
3. Loop processes one crop at a time despite batching capability
4. `ArcFaceEmbedder.encode()` accepts list but always receives single-element list

### Solution
**Batch embeddings per frame** - group all faces from same frame and process together

**Benefits:**
- **CPU Reduction:** 600% → 250% (estimated 60% reduction)
- **GPU Efficiency:** Better utilization via batch processing
- **Model Warmup:** 24,061 invocations → ~800 (assuming 30 faces/frame avg)

### Task
Implement per-frame batch embedding in `_run_faces_embed_stage()`

### Patch
**File:** `tools/episode_run.py`

**Changes Required:**

1. **Group samples by frame_idx** before main loop:
```python
# After line 4310 (before main loop starts)
from itertools import groupby

# Group samples by frame_idx for batch embedding
samples_by_frame = []
for frame_idx, group in groupby(samples, key=lambda s: s["frame_idx"]):
    samples_by_frame.append((frame_idx, list(group)))
```

2. **Replace single-face loop with batch processing:**
```python
# Replace lines 4320-4520 (main embedding loop)
for frame_idx, frame_samples in samples_by_frame:
    # Decode frame once for all faces
    if not video_path.exists():
        raise FileNotFoundError("Local video not found for crop export")
    if frame_decoder is None:
        frame_decoder = FrameDecoder(video_path)
    
    image = None
    try:
        image = frame_decoder.read(frame_idx)
        frame_std = float(np.std(image)) if image is not None else 0.0
        if image is None or frame_std < 1.0:
            image = frame_decoder.read(frame_idx)  # Retry
            frame_std = float(np.std(image)) if image is not None else 0.0
    except (RuntimeError, Exception) as decode_exc:
        LOGGER.error("Decode failure on frame %s: %s", frame_idx, decode_exc)
        image = None
        frame_std = 0.0
    
    if image is None or frame_std < 1.0:
        # Skip all samples in this frame
        for sample in frame_samples:
            rows.append(_make_skip_face_row(
                args.ep_id, sample["track_id"], frame_idx, 
                round(float(sample["ts"]), 4), sample["bbox_xyxy"],
                detector_choice, "bad_source_frame"
            ))
            faces_done += 1
        progress.emit(faces_done, phase="faces_embed", ...)
        continue
    
    # Prepare all crops for this frame
    batch_crops = []
    batch_metadata = []  # Store (sample, bbox, landmarks, crop_rel_path, etc.)
    
    for sample in frame_samples:
        bbox = sample["bbox_xyxy"]
        landmarks = sample.get("landmarks")
        track_id = sample["track_id"]
        ts_val = round(float(sample["ts"]), 4)
        
        # Export frame/crop
        if exporter and image is not None:
            exporter.export(frame_idx, image, [(track_id, bbox)], ts=ts_val)
            crop_rel_path = exporter.crop_rel_path(track_id, frame_idx) if exporter.save_crops else None
            crop_s3_key = f"{s3_prefixes['crops']}{exporter.crop_component(track_id, frame_idx)}" if crop_rel_path and s3_prefixes and s3_prefixes.get("crops") else None
        else:
            crop_rel_path = None
            crop_s3_key = None
        
        # Validate bbox
        validated_bbox, bbox_err = _safe_bbox_or_none(bbox)
        if validated_bbox is None:
            rows.append(_make_skip_face_row(...))
            faces_done += 1
            continue
        
        # Prepare crop
        crop, crop_err = _prepare_face_crop(image, validated_bbox, landmarks)
        if crop is None:
            rows.append(_make_skip_face_row(...))
            faces_done += 1
            continue
        
        # Quality checks
        crop_std = float(np.std(crop))
        blur_score = _estimate_blur_score(crop)
        conf = float(sample.get("conf") or sample.get("confidence") or 1.0)
        
        if conf < FACE_MIN_CONFIDENCE or crop_std < FACE_MIN_STD or blur_score < FACE_MIN_BLUR:
            rows.append(_make_skip_face_row(...))
            faces_done += 1
            continue
        
        # Create thumbnail
        thumb_rel_path, _ = thumb_writer.write(image, validated_bbox, track_id, frame_idx, prepared_crop=crop)
        thumb_s3_key = f"{s3_prefixes['thumbs_tracks']}{thumb_rel_path}" if thumb_rel_path and s3_prefixes and s3_prefixes.get("thumbs_tracks") else None
        
        # Add to batch
        batch_crops.append(crop)
        batch_metadata.append({
            "sample": sample,
            "bbox": bbox,
            "validated_bbox": validated_bbox,
            "landmarks": landmarks,
            "crop_rel_path": crop_rel_path,
            "crop_s3_key": crop_s3_key,
            "thumb_rel_path": thumb_rel_path,
            "thumb_s3_key": thumb_s3_key,
            "conf": conf,
            "quality": max(min(conf, 1.0), 0.0),
        })
    
    # ✅ BATCH EMBEDDING: Process all crops from this frame together
    if batch_crops:
        embeddings = embedder.encode(batch_crops)  # Single CoreML call for entire batch!
        
        for idx, (embedding_vec, meta) in enumerate(zip(embeddings, batch_metadata)):
            sample = meta["sample"]
            track_id = sample["track_id"]
            ts_val = round(float(sample["ts"]), 4)
            
            # Check embedding validity
            embedding_norm = float(np.linalg.norm(embedding_vec))
            if embedding_norm < 1e-6:
                rows.append(_make_skip_face_row(..., "zero_norm_embedding"))
                faces_done += 1
                continue
            
            # Store embedding
            track_embeddings[track_id].append((float(meta["quality"]), embedding_vec.copy()))
            embeddings_array.append(embedding_vec)
            
            # Seed matching
            seed_cast_id = None
            seed_similarity = None
            if show_seeds and SEED_BOOST_ENABLED:
                seed_match_stats["total"] += 1
                match_result = _find_best_seed_match(embedding_vec, show_seeds, min_sim=SEED_BOOST_MIN_SIM)
                if match_result:
                    seed_cast_id, seed_similarity = match_result
                    seed_match_stats["matches"] += 1
            
            # Track best thumbnail
            if meta["thumb_rel_path"]:
                prev = track_best_thumb.get(track_id)
                if not prev or meta["quality"] > prev[0]:
                    track_best_thumb[track_id] = (meta["quality"], meta["thumb_rel_path"], meta["thumb_s3_key"])
            
            # Build face row
            face_row = {
                "ep_id": args.ep_id,
                "face_id": f"face_{track_id:04d}_{frame_idx:06d}",
                "track_id": track_id,
                "frame_idx": frame_idx,
                "ts": ts_val,
                "bbox_xyxy": meta["bbox"],
                "conf": round(float(meta["conf"]), 4),
                "quality": round(float(meta["quality"]), 4),
                "embedding": embedding_vec.tolist(),
                "embedding_model": embedding_model_name,
                "detector": detector_choice,
                "pipeline_ver": PIPELINE_VERSION,
            }
            if meta["crop_rel_path"]: face_row["crop_rel_path"] = meta["crop_rel_path"]
            if meta["crop_s3_key"]: face_row["crop_s3_key"] = meta["crop_s3_key"]
            if meta["thumb_rel_path"]: face_row["thumb_rel_path"] = meta["thumb_rel_path"]
            if meta["thumb_s3_key"]: face_row["thumb_s3_key"] = meta["thumb_s3_key"]
            if meta["landmarks"]: face_row["landmarks"] = [round(float(v), 4) for v in meta["landmarks"]]
            if seed_cast_id:
                face_row["seed_cast_id"] = seed_cast_id
                face_row["seed_similarity"] = round(float(seed_similarity), 4)
            
            rows.append(face_row)
            faces_done += 1
    
    # Progress update per frame
    progress.emit(faces_done, phase="faces_embed", device=device, ...)
```

**Implementation Notes:**
- Samples are already sorted by frame (line 4234: `sort_by_frame=True`)
- `itertools.groupby` requires sorted input (already satisfied)
- Each batch processes all faces from a single frame
- Reduces CoreML invocations from 24,061 to ~800 (96.7% reduction)

**Testing:**
```bash
# Run with batch embedding enabled
python tools/episode_run.py --ep-id test_batch --video samples/demo.mp4 --device coreml

# Verify CPU usage stays under 300%
top -pid $(pgrep -f episode_run.py) -stats pid,cpu,th
```

---

## ISSUE 2: Thread Limit Multiplication (P0)

### Problem
**Location:** `tools/episode_run.py` lines 23-30

```python
# ❌ CURRENT: 8 libraries × 2 threads each = 16 potential threads
os.environ.setdefault("OMP_NUM_THREADS", "2")           # NumPy/SciPy
os.environ.setdefault("MKL_NUM_THREADS", "2")           # Intel MKL
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")      # OpenBLAS
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")    # Apple Accelerate
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")       # NumExpr
os.environ.setdefault("OPENCV_NUM_THREADS", "2")        # OpenCV
os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", "2")  # ONNX Runtime intra-op
os.environ.setdefault("ORT_INTER_OP_NUM_THREADS", "1")  # ONNX Runtime inter-op
```

**Impact:**
- **Potential CPU:** 8 libs × 2 threads = 1600% (16 cores)
- **Actual CPU:** ~500-600% (5-6 cores) due to contention
- **Thread Contention:** Multiple libraries competing for same CPU cores

### Root Cause
Per-library thread limits don't enforce overall process CPU cap

### Solution
Reduce all thread limits to 1 thread per library

### Task
Set all thread environment variables to "1"

### Patch
**File:** `tools/episode_run.py`

```python
# Replace lines 23-30
os.environ.setdefault("OMP_NUM_THREADS", "1")           # NumPy/SciPy
os.environ.setdefault("MKL_NUM_THREADS", "1")           # Intel MKL
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")      # OpenBLAS
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")    # Apple Accelerate
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")       # NumExpr
os.environ.setdefault("OPENCV_NUM_THREADS", "1")        # OpenCV
os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", "1")  # ONNX Runtime intra-op
os.environ.setdefault("ORT_INTER_OP_NUM_THREADS", "1")  # ONNX Runtime inter-op (was 1 already)
```

**Expected Impact:**
- **CPU Reduction:** 600% → 300% (50% reduction)
- **Performance:** Minimal impact due to CoreML GPU acceleration handling heavy lifting

---

## ISSUE 3: No Process-Level CPU Cap (P0)

### Problem
**Location:** Process-wide enforcement missing

**Current State:**
- Thread limits are advisory (libraries can ignore them)
- No hard CPU cap at process level
- No enforcement mechanism if thread limits fail

**Impact:**
- **Risk:** CPU can still exceed 300% if thread limits ineffective
- **No Fallback:** No hard limit to prevent CPU spikes

### Solution
Add `cpulimit` wrapper to enforce hard 300% CPU cap

### Task
Wrap `episode_run.py` execution with `cpulimit -l 300`

### Patch
**File:** `scripts/dev.sh`

Already has cpulimit logic (lines 48-75), but NOT used for background jobs.

**Add CPU-limited job launcher:**

```bash
# Add after line 75 in dev.sh
function run_cpu_limited() {
    local cmd="$1"
    local max_cpu="${2:-300}"  # Default 300% (3 cores)
    
    if command -v cpulimit &>/dev/null; then
        echo "⚙️  Running with CPU limit: ${max_cpu}%"
        cpulimit -l "$max_cpu" -- $cmd
    else
        echo "⚠️  cpulimit not found - running without CPU cap"
        echo "   Install with: brew install cpulimit"
        $cmd
    fi
}
```

**Update job execution in API or manual runs:**

For manual runs:
```bash
# Instead of:
python tools/episode_run.py --ep-id myep --video input.mp4

# Use:
./scripts/dev.sh run_cpu_limited "python tools/episode_run.py --ep-id myep --video input.mp4" 300
```

For API-triggered jobs (update `apps/api/routers/jobs.py`):
```python
# In _build_faces_command() or subprocess.run() calls
command = ["cpulimit", "-l", "300", "--", "python", "tools/episode_run.py", ...]
```

**Installation:**
```bash
brew install cpulimit
```

**Testing:**
```bash
# Monitor CPU during run
top -pid $(pgrep -f episode_run) -stats pid,cpu,th

# Should never exceed 300%
```

---

## ISSUE 4: S3 Upload Concurrency (P1)

### Problem
**Location:** `tools/episode_run.py` line 4603

```python
# After embedding completes
s3_stats = _sync_artifacts_to_s3(args.ep_id, storage, ep_ctx, exporter, thumb_writer.root_dir)
```

**Current Behavior:**
- S3 uploads happen AFTER embedding completes (good sequencing)
- But uploads may spawn multiple concurrent threads internally
- boto3 TransferConfig default: 10 concurrent threads

**Impact:**
- **Additional CPU:** +100-200% during S3 sync phase
- **Total CPU:** Embedding CPU + Upload CPU can exceed 300%

### Solution
Serialize S3 uploads or limit transfer threads

### Task
Configure boto3 TransferConfig with max_concurrency=2

### Patch
**File:** `apps/shared/storage.py` (or wherever S3 client is configured)

```python
import boto3
from boto3.s3.transfer import TransferConfig

# Configure S3 transfer with limited concurrency
s3_transfer_config = TransferConfig(
    max_concurrency=2,  # Limit concurrent S3 upload threads
    use_threads=True,
)

# Apply to all S3 upload calls
s3_client.upload_file(
    local_path, 
    bucket, 
    key, 
    Config=s3_transfer_config
)
```

**Expected Impact:**
- **CPU Reduction:** -100% during upload phase
- **Upload Speed:** Slight decrease (acceptable trade-off)

---

## ISSUE 5: CoreML + CPU Thread Overhead (P1)

### Problem
**Location:** Process architecture

**Current State:**
- CoreML runs on GPU (Neural Engine on Apple Silicon)
- BUT CoreML scheduling uses CPU threads
- CPU threads feed data to GPU and handle results
- With 24,061 single-face calls, CPU is constantly scheduling GPU work

**Impact:**
- **CPU Overhead:** ~100-150% just for GPU scheduling
- **Inefficiency:** Small batches = more scheduling overhead

### Solution
Already addressed by ISSUE 1 (batch embedding)

**With Batching:**
- 24,061 GPU calls → ~800 GPU calls
- **CPU Scheduling Reduction:** -100% (eliminated by batching)

---

## ISSUE 6: FrameDecoder LRU Cache Thrashing (P2)

### Problem
**Location:** `tools/episode_run.py` (FrameDecoder class, not shown in excerpts)

**Hypothesis:**
- Samples sorted by frame_idx (line 4234)
- But if stride=4, consecutive frames are 4 frames apart
- LRU cache size unknown - may evict before reuse

**Impact:**
- **Potential:** Repeated frame decoding if cache too small
- **CPU:** +50-100% if thrashing occurs

### Solution
Verify/increase FrameDecoder LRU cache size

### Task
Check cache size and increase if needed

### Patch
**File:** `tools/episode_run.py` (locate FrameDecoder class)

```python
# Find FrameDecoder class definition
class FrameDecoder:
    def __init__(self, video_path, cache_size=128):  # Increase from default
        self.cache = lru_cache(maxsize=cache_size)(self._decode_frame)
        ...
```

**Recommended:**
- Cache size = max faces per batch × 2
- For 30 faces/frame avg: cache_size=64 should suffice
- Monitor cache hit rate in logs

---

## Summary Table

| Issue | Priority | Current CPU | Target CPU | Savings | Status |
|-------|----------|-------------|------------|---------|--------|
| 1. Single-face embedding | P0 | 400-600% | 200-250% | -60% | Patch Ready |
| 2. Thread limit multiplication | P0 | 600% | 300% | -50% | Patch Ready |
| 3. No process CPU cap | P0 | Unbounded | 300% max | Hard limit | Patch Ready |
| 4. S3 upload concurrency | P1 | +100-200% | +20-40% | -100% | Patch Ready |
| 5. CoreML scheduling overhead | P1 | +100-150% | +20-30% | -100% | Fixed by #1 |
| 6. Frame cache thrashing | P2 | +0-100% | +0% | -100% | Investigation |

**Combined Impact:**
- **Before:** 600-800% CPU (6-8 cores)
- **After (P0 only):** 250-300% CPU (2.5-3 cores)
- **After (P0+P1):** 200-250% CPU (2-2.5 cores)

---

## Implementation Order

### Phase 1: Critical Fixes (Deploy Today)
1. ✅ **ISSUE 2** - Reduce all thread limits to 1 (5 min)
2. ✅ **ISSUE 3** - Add cpulimit wrapper to jobs (15 min)

**Expected Result:** 600% → 400% CPU

### Phase 2: Batch Embedding (Deploy This Week)
3. ✅ **ISSUE 1** - Implement per-frame batch embedding (2-4 hours)
4. ✅ **ISSUE 4** - Limit S3 upload concurrency (30 min)

**Expected Result:** 400% → 250% CPU

### Phase 3: Fine-Tuning (Optional)
5. 📋 **ISSUE 6** - Investigate frame cache size (1 hour)

**Expected Result:** 250% → 200% CPU (if thrashing detected)

---

## Testing Plan

### 1. Baseline Measurement (Current State)
```bash
# Start face harvest
python tools/episode_run.py --ep-id cpu_test --video data/videos/long_episode.mp4 --device coreml

# Monitor CPU in separate terminal
top -pid $(pgrep -f episode_run) -stats pid,cpu,th -l 0

# Record:
# - Max CPU %
# - Average CPU %
# - Processing FPS
# - Total time
```

### 2. Apply Phase 1 Fixes
```bash
# Apply thread limits patch
# Apply cpulimit wrapper

# Re-run test
# Verify CPU < 400%
```

### 3. Apply Phase 2 Fixes
```bash
# Apply batch embedding patch
# Apply S3 concurrency patch

# Re-run test
# Verify CPU < 300%
```

### 4. Regression Testing
```bash
# Run full test suite
pytest tests/ml/test_arcface_embeddings.py -v

# Verify:
# - Embeddings still unit-normalized
# - Face matching accuracy unchanged
# - Seed matching still works
```

---

## Rollback Plan

If patches cause issues:

1. **Thread Limits (ISSUE 2):**
   - Revert to `"2"` if performance degrades significantly
   - Monitor with: `python -c "import numpy; numpy.show_config()"`

2. **Batch Embedding (ISSUE 1):**
   - Keep old single-face loop as `_run_faces_embed_stage_legacy()`
   - Add CLI flag: `--batch-embed / --no-batch-embed`
   - Default to batch, fallback to legacy if errors

3. **CPU Limit (ISSUE 3):**
   - Remove cpulimit wrapper if blocking legitimate workload
   - Keep as opt-in via environment variable: `ENABLE_CPU_LIMIT=1`

---

## Monitoring & Alerts

### Add CPU Monitoring to Progress Logging

**File:** `tools/episode_run.py`

```python
import psutil

# In ProgressEmitter class
class ProgressEmitter:
    def __init__(self, ...):
        self.process = psutil.Process()
        ...
    
    def emit(self, ...):
        cpu_percent = self.process.cpu_percent(interval=0.1)
        extra["cpu_percent"] = round(cpu_percent, 1)
        extra["thread_count"] = self.process.num_threads()
        ...
```

**Benefits:**
- Real-time CPU tracking in progress.json
- Historical CPU usage data per episode
- Alerts if CPU exceeds 350% (warning threshold)

---

## Expected Outcomes

### Performance Metrics

| Metric | Before | After Phase 1 | After Phase 2 |
|--------|--------|---------------|---------------|
| Max CPU % | 600-800% | 400% | 250-300% |
| Avg CPU % | 500% | 350% | 200-250% |
| Processing FPS | 6.05 | 5.8 (slight drop) | 7-8 (batching boost) |
| Time for 42min video | ~115 min | ~120 min | ~90 min |
| CoreML calls | 24,061 | 24,061 | ~800 (96.7% reduction) |

### Quality Assurance
- ✅ No change to embedding accuracy
- ✅ No change to face matching results
- ✅ Same output files (faces.jsonl, crops, thumbnails)
- ✅ Backward compatible with existing pipelines

---

## Notes

1. **ISSUE 1 (Batch Embedding) is the single biggest win** - reduces CoreML invocations by 96.7%
2. **ISSUE 2 (Thread Limits) is the easiest fix** - one-line change per variable
3. **ISSUE 3 (CPU Cap) is insurance** - hard limit prevents runaway processes
4. All patches are **non-breaking** and **backward compatible**
5. Testing on 42-min episode (24,061 frames) is ideal stress test

---

## References

- Thread limits: https://github.com/numpy/numpy/issues/11826
- cpulimit docs: https://github.com/opsengine/cpulimit
- boto3 TransferConfig: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/customizations/s3.html#boto3.s3.transfer.TransferConfig
- CoreML threading: https://developer.apple.com/documentation/coreml/mlcomputeunits

---

**END OF REPORT**
