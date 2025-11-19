# Detect/Track Performance Improvements & Bug Fixes

## Current State Analysis
- **42-minute video**: 26,807 detections, 4,352 tracks
- **Current max gap**: 0.5 seconds (15 frames at 30fps) - tracks split too frequently
- **Track buffer**: 15-25 frames (configurable via YAML)
- **Embedding extraction**: Every 10 frames (GATE_EMB_EVERY_DEFAULT)
- **No track skipping**: All tracks processed every frame
- **No crop skipping**: All crops saved for all tracks
- **CPU limits**: Thread limits set but not enforced globally (no 250% cap)

---

## 8 BUGS/ISSUES

### Bug #1: Max Gap Too Short Causes Excessive Track Fragmentation
**Issue**: `TRACK_MAX_GAP_SEC = 0.5` (15 frames at 30fps) causes tracks to split prematurely when faces are temporarily occluded or detection confidence drops briefly.

**Impact**: Creates 4,352 tracks instead of fewer, longer tracks. Each track split requires new track ID assignment and increases processing overhead.

**Current Code Location**: `tools/episode_run.py:221`
```python
TRACK_MAX_GAP_SEC = float(os.environ.get("TRACK_MAX_GAP_SEC", "0.5"))
```

**Suggested Fix**: Increase default to 2.0 seconds (60 frames at 30fps) to allow tracks to bridge temporary gaps.

**Patch**:
```python
# Line 221 in tools/episode_run.py
TRACK_MAX_GAP_SEC = float(os.environ.get("TRACK_MAX_GAP_SEC", "2.0"))  # Increased from 0.5 to 2.0
```

---

### Bug #2: No Track Processing Skip - All Tracks Processed Every Frame
**Issue**: Every tracked object is processed for embedding extraction and crop saving on every frame, even when not needed.

**Impact**: Unnecessary CPU/GPU usage. For 4,352 tracks, this means processing thousands of embeddings and crops per frame.

**Current Code Location**: `tools/episode_run.py:3636-3707` - All tracked_objects processed without skipping.

**Suggested Fix**: Add track skipping logic to process every Nth track (e.g., every 6th track).

**Patch**:
```python
# After line 3636 in tools/episode_run.py, add track skip counter
# Add near top of file with other constants (around line 82):
TRACK_PROCESS_SKIP = max(int(os.environ.get("SCREENALYTICS_TRACK_PROCESS_SKIP", "6")), 1)

# In _run_full_pipeline function, add track counter (around line 3230):
track_process_counter = 0  # Add after line 3230

# Modify the tracked_objects loop (around line 3636):
active_ids: set[int] = set()
for obj_idx, obj in enumerate(tracked_objects):
    # Skip processing every Nth track
    if obj_idx % TRACK_PROCESS_SKIP != 0:
        # Still record track for continuity, but skip expensive operations
        active_ids.add(obj.track_id)
        export_id = recorder.record(
            tracker_track_id=obj.track_id,
            frame_idx=frame_idx,
            ts=ts,
            bbox=obj.bbox,
            class_label=FACE_CLASS_LABEL,
            landmarks=None,
            confidence=(float(obj.conf) if obj.conf is not None else None),
            force_new_track=False,
        )
        continue
    
    active_ids.add(obj.track_id)
    # ... rest of existing processing code ...
```

---

### Bug #3: No Crop Saving Skip - All Crops Saved for All Tracks
**Issue**: Crops are saved for every track on every frame where they appear, even when skipping track processing.

**Impact**: Excessive disk I/O and storage usage. For 4,352 tracks over 42 minutes, this creates millions of crop files.

**Current Code Location**: `tools/episode_run.py:3700-3701`

**Suggested Fix**: Add crop skip logic to save crops every Nth track (e.g., every 8th track).

**Patch**:
```python
# Add constant near top (around line 82):
TRACK_CROP_SKIP = max(int(os.environ.get("SCREENALYTICS_TRACK_CROP_SKIP", "8")), 1)

# Modify crop saving logic (around line 3700):
crop_records: list[tuple[int, list[float]]] = []
for obj_idx, obj in enumerate(tracked_objects):
    # ... existing processing ...
    export_id = recorder.record(...)
    
    # Only save crops for every Nth track
    if frame_exporter and frame_exporter.save_crops and (obj_idx % TRACK_CROP_SKIP == 0):
        crop_records.append((export_id, bbox_list))
```

---

### Bug #4: Thread Limits Not Enforced Globally - No 250% CPU Cap
**Issue**: Thread limits are set via environment variables but not enforced at runtime. Multiple processes (Streamlit + API + detect/track) can exceed 250% CPU total.

**Impact**: System overload, thermal throttling, degraded performance.

**Current Code Location**: `tools/episode_run.py:23-30` - Sets env vars but doesn't enforce limits.

**Suggested Fix**: Add CPU usage monitoring and throttling to cap total CPU usage at 250%.

**Patch**:
```python
# Add after imports (around line 33):
import psutil
import threading

# Add global CPU limiter class (after line 300):
class CPULimiter:
    """Enforce global CPU usage cap across all processes."""
    def __init__(self, max_cpu_percent: float = 250.0):
        self.max_cpu_percent = max_cpu_percent
        self._lock = threading.Lock()
        self._monitoring = False
    
    def check_and_throttle(self):
        """Check total CPU usage and throttle if needed."""
        with self._lock:
            total_cpu = psutil.cpu_percent(interval=0.1) * psutil.cpu_count()
            if total_cpu > self.max_cpu_percent:
                # Throttle by sleeping briefly
                time.sleep(0.01)
    
    def get_current_usage(self) -> float:
        """Get current total CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1) * psutil.cpu_count()

# Global instance
_CPU_LIMITER = CPULimiter(max_cpu_percent=250.0)

# In _run_full_pipeline main loop (around line 3277), add throttling:
while True:
    ok, frame = cap.read()
    if not ok:
        break
    
    # Throttle if CPU usage too high
    _CPU_LIMITER.check_and_throttle()
    
    # ... rest of frame processing ...
```

---

### Bug #5: Embedding Extraction Happens Even When Track Skipped
**Issue**: Gate embeddings are extracted for all tracks even when track processing is skipped, wasting GPU/CPU cycles.

**Impact**: Unnecessary embedding computations when tracks are being skipped anyway.

**Current Code Location**: `tools/episode_run.py:3558-3635` - Gate embedding extraction happens before track skipping.

**Suggested Fix**: Skip embedding extraction for skipped tracks.

**Patch**:
```python
# Modify gate embedding section (around line 3558):
if should_embed_gate and gate_embedder and tracked_objects:
    embed_inputs: list[np.ndarray] = []
    embed_track_ids: list[int] = []
    
    for obj_idx, obj in enumerate(tracked_objects):
        # Skip embedding for tracks that will be skipped in processing
        if obj_idx % TRACK_PROCESS_SKIP != 0:
            continue
        
        # ... existing embedding extraction code ...
```

---

### Bug #6: Track Buffer Too Small for Long Gaps
**Issue**: Default track buffer (15-25 frames) is smaller than the max gap, causing tracks to die before max gap is reached.

**Impact**: Tracks split prematurely even with increased max gap.

**Current Code Location**: `tools/episode_run.py:208-211`, `config/pipeline/tracking.yaml:7`

**Suggested Fix**: Ensure track buffer >= max gap frames to allow tracks to bridge gaps.

**Patch**:
```python
# Modify ByteTrackRuntimeConfig.scaled_buffer (around line 934):
def scaled_buffer(self, stride: int) -> int:
    """Scale buffer by stride, ensuring it's at least max_gap_frames."""
    effective = max(int(round(self.track_buffer_base * stride)), self.track_buffer_base)
    # Ensure buffer is at least as large as max gap to prevent premature track death
    max_gap_frames = int(round(30.0 * TRACK_MAX_GAP_SEC))  # Assume 30fps if unknown
    return max(effective, max_gap_frames)
```

---

### Bug #7: No Batching of Embedding Extractions
**Issue**: Embeddings are extracted one-by-one in a loop, missing opportunities for batch processing optimization.

**Impact**: Slower embedding extraction, especially on GPU where batching provides significant speedup.

**Current Code Location**: `tools/episode_run.py:3612-3635` - Embeddings extracted sequentially.

**Current Code**: Already batches! But could optimize batch size based on available memory.

**Suggested Fix**: Add configurable batch size and memory-aware batching.

**Patch**:
```python
# Add constant (around line 82):
EMBEDDING_BATCH_SIZE = max(int(os.environ.get("SCREENALYTICS_EMBEDDING_BATCH_SIZE", "32")), 1)

# Modify embedding extraction (around line 3612):
if embed_inputs:
    # Process in batches for better GPU utilization
    encoded_list = []
    for batch_start in range(0, len(embed_inputs), EMBEDDING_BATCH_SIZE):
        batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, len(embed_inputs))
        batch_inputs = embed_inputs[batch_start:batch_end]
        batch_encoded = gate_embedder.encode(batch_inputs)
        encoded_list.extend(batch_encoded)
    
    encoded = encoded_list
    # ... rest of existing code ...
```

---

### Bug #8: Excessive Track Recording on Every Frame
**Issue**: TrackRecorder.record() is called for every tracked object on every frame, even when track hasn't changed.

**Impact**: Unnecessary dictionary lookups and accumulator updates.

**Current Code Location**: `tools/episode_run.py:3669-3678`

**Suggested Fix**: Only record when track bbox/confidence changes significantly or on periodic intervals.

**Patch**:
```python
# Modify TrackRecorder to track last recorded state (around line 1481):
class TrackRecorder:
    def __init__(self, *, max_gap: int, remap_ids: bool) -> None:
        # ... existing init ...
        self._last_recorded: dict[int, dict] = {}  # track_id -> {frame_idx, bbox, conf}
    
    def record(
        self,
        *,
        tracker_track_id: int,
        frame_idx: int,
        ts: float,
        bbox: list[float] | np.ndarray,
        class_label: int | str,
        landmarks: list[float] | None = None,
        confidence: float | None = None,
        force_new_track: bool = False,
        skip_if_unchanged: bool = True,  # New parameter
    ) -> int:
        # ... existing mapping logic ...
        
        # Skip recording if track unchanged and skip_if_unchanged=True
        if skip_if_unchanged and export_id in self._last_recorded:
            last = self._last_recorded[export_id]
            if (frame_idx - last["frame_idx"] < 5 and  # Within 5 frames
                np.allclose(bbox, last["bbox"], rtol=0.05)):  # Bbox similar
                return export_id
        
        # Record track
        track = self._accumulators.get(export_id)
        # ... existing recording logic ...
        
        # Update last recorded state
        self._last_recorded[export_id] = {
            "frame_idx": frame_idx,
            "bbox": bbox_values,
            "conf": confidence,
        }
        return export_id
```

---

## 8 IMPROVEMENTS/FIXES

### Improvement #1: Increase Max Gap to 2.0 Seconds for Longer Tracks
**Current**: 0.5 seconds (15 frames at 30fps)
**Target**: 2.0 seconds (60 frames at 30fps)

**Rationale**: Allows tracks to bridge temporary occlusions and detection gaps, reducing track fragmentation.

**Implementation**: See Bug #1 patch above.

**Expected Impact**: Reduce track count from 4,352 to ~2,000-3,000 tracks (30-50% reduction).

---

### Improvement #2: Implement Track Processing Skip (Every 6th Track)
**Current**: All tracks processed every frame
**Target**: Process every 6th track, skip others

**Rationale**: Reduces CPU/GPU load while maintaining track continuity. Skipped tracks still recorded but don't get expensive embedding/crop processing.

**Implementation**: See Bug #2 patch above.

**Expected Impact**: 
- 83% reduction in embedding extractions per frame
- 83% reduction in crop operations per frame
- ~40-50% CPU/GPU usage reduction

---

### Improvement #3: Implement Crop Saving Skip (Every 8th Track)
**Current**: Crops saved for all tracks
**Target**: Save crops every 8th track

**Rationale**: Reduces disk I/O and storage usage while maintaining representative crop samples.

**Implementation**: See Bug #3 patch above.

**Expected Impact**:
- 87.5% reduction in crop file writes
- Significant disk I/O reduction
- Storage space savings

---

### Improvement #4: Add Global CPU Usage Cap (250% Total)
**Current**: No global CPU limit enforcement
**Target**: Cap total CPU usage at 250% across all processes

**Rationale**: Prevents system overload and thermal throttling, ensures responsive system.

**Implementation**: See Bug #4 patch above.

**Expected Impact**:
- Prevents system overload
- More consistent performance
- Better thermal management

---

### Improvement #5: Increase Track Buffer to Match Max Gap
**Current**: 15-25 frames
**Target**: At least 60 frames (2 seconds at 30fps) to match max gap

**Rationale**: Track buffer must be >= max gap to prevent premature track death.

**Implementation**: See Bug #6 patch above.

**Expected Impact**: Fewer premature track splits, longer tracks.

---

### Improvement #6: Optimize Embedding Batch Size
**Current**: Processes all embeddings in one batch (could be very large)
**Target**: Configurable batch size (default 32) for memory efficiency

**Rationale**: Better GPU memory utilization and faster processing with optimal batch sizes.

**Implementation**: See Bug #7 patch above.

**Expected Impact**:
- 20-30% faster embedding extraction on GPU
- Better memory utilization
- More stable performance

---

### Improvement #7: Skip Unchanged Track Recording
**Current**: Records every track on every frame
**Target**: Only record when track changes significantly or every N frames

**Rationale**: Reduces unnecessary accumulator updates and dictionary operations.

**Implementation**: See Bug #8 patch above.

**Expected Impact**:
- 10-15% reduction in track recording overhead
- Fewer accumulator updates
- Lower memory churn

---

### Improvement #8: Add Adaptive Frame Stride Based on Track Density
**Current**: Fixed frame stride (default 1)
**Target**: Adaptive stride that increases when track density is high

**Rationale**: When many tracks are active, reduce detection frequency to save CPU while maintaining tracking quality.

**Implementation**:
```python
# Add adaptive stride logic (around line 3110):
def _adaptive_stride(base_stride: int, active_tracks: int, max_tracks: int = 50) -> int:
    """Increase stride when track density is high."""
    if active_tracks > max_tracks:
        # Increase stride by 1 for every 25 tracks over max
        stride_multiplier = 1 + ((active_tracks - max_tracks) // 25)
        return base_stride * stride_multiplier
    return base_stride

# In main loop (around line 3317):
current_stride = _adaptive_stride(frame_stride, len(tracked_objects))
should_sample = frame_idx % current_stride == 0
```

**Expected Impact**:
- 20-30% reduction in detection calls when track density is high
- Better CPU utilization
- Maintains tracking quality

---

## Summary of Expected Improvements

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Track Count | 4,352 | ~2,000-3,000 | 30-50% reduction |
| CPU Usage (peak) | Unbounded | 250% cap | Controlled |
| Embedding Ops/frame | All tracks | Every 6th | 83% reduction |
| Crop Saves/frame | All tracks | Every 8th | 87.5% reduction |
| Track Buffer | 15-25 frames | 60+ frames | 2-4x increase |
| Max Gap | 0.5s (15 frames) | 2.0s (60 frames) | 4x increase |
| Processing Time | Baseline | -40-50% | Significant speedup |

---

## Implementation Priority

1. **P0 (Critical)**: Bug #1 (Max Gap), Bug #4 (CPU Cap), Bug #6 (Track Buffer)
2. **P1 (High)**: Bug #2 (Track Skip), Bug #3 (Crop Skip), Improvement #5 (Track Buffer)
3. **P2 (Medium)**: Bug #5 (Embedding Skip), Bug #7 (Batching), Bug #8 (Recording Skip)
4. **P3 (Low)**: Improvement #8 (Adaptive Stride)

---

## Testing Recommendations

1. Run detect/track on 42-minute test video
2. Measure: track count, CPU usage, processing time, crop count
3. Verify: tracks are longer, fewer total tracks, CPU stays under 250%
4. Validate: tracking accuracy maintained despite skipping
