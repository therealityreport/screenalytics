# Remaining High-Priority Optimizations - Implementation Guide

## D1: Efficient Frame Skipping with cap.grab() [HIGHEST IMPACT]

**Location:** `tools/episode_run.py` line ~3653

**Current Code:**
```python
while True:
    ok, frame = cap.read()
    if not ok:
        break

    # ... frame processing ...

    force_detect = frames_since_cut < scene_warmup
    should_sample = frame_idx % frame_stride == 0
    if not (should_sample or force_detect):
        frame_idx += 1
        frames_since_cut += 1
        continue
```

**Replace With:**
```python
while True:
    # D1: Use grab() to skip frame decode for frames we won't analyze
    # This avoids decoding ~83% of frames when stride=6
    ok = cap.grab()
    if not ok:
        break

    # Determine if we need to actually decode this frame
    force_detect = frames_since_cut < scene_warmup
    should_sample = frame_idx % frame_stride == 0
    at_scene_cut = next_cut is not None and frame_idx >= next_cut

    # Skip decode if we won't process this frame
    if not (should_sample or force_detect or at_scene_cut):
        frame_idx += 1
        frames_since_cut += 1
        continue

    # Retrieve (decode) only frames we'll actually process
    frame_ok, frame = cap.retrieve()
    if not frame_ok:
        LOGGER.warning(
            "Failed to retrieve frame %d for %s after successful grab",
            frame_idx,
            args.ep_id,
        )
        frame_idx += 1
        frames_since_cut += 1
        continue

    # Guard against empty/None frames before detection
    if frame is None or frame.size == 0:
        LOGGER.warning(
            "Skipping frame %d for %s: empty or None frame from video capture",
            frame_idx,
            args.ep_id,
        )
        frame_idx += 1
        frames_since_cut += 1
        continue

    # Continue with scene cut detection and processing...
```

**Impact:** Avoids decoding 83% of frames (48,000 frames saved on a 40min episode at stride=6).

---

## F1: Scene Cut Cooldown [QUICK WIN]

**Location:** `tools/episode_run.py`

**Step 1:** Add CLI argument (around line 2753)
```python
parser.add_argument(
    "--scene-cut-cooldown",
    type=int,
    default=24,
    help="Minimum frames between scene cut resets (default: 24)",
)
```

**Step 2:** Add tracking variable before main loop (around line 3547)
```python
cut_ix = 0
next_cut = scene_cuts[cut_ix] if scene_cuts else None
frames_since_cut = 10**9
last_cut_reset = -999  # F1: Track last reset to prevent thrashing
scene_cut_cooldown = getattr(args, "scene_cut_cooldown", 24)
```

**Step 3:** Update scene cut logic (around line 3669)
```python
if next_cut is not None and frame_idx >= next_cut:
    # F1: Only reset if we're past cooldown period
    if frame_idx - last_cut_reset >= scene_cut_cooldown:
        reset_tracker = getattr(tracker_adapter, "reset", None)
        if callable(reset_tracker):
            reset_tracker()
        if appearance_gate:
            appearance_gate.reset_all()
        recorder.on_cut(frame_idx)
        frames_since_cut = 0
        last_cut_reset = frame_idx  # F1: Record reset time
        if progress:
            # ... progress emit ...

    # Always advance to next cut
    cut_ix += 1
    next_cut = scene_cuts[cut_ix] if cut_ix < len(scene_cuts) else None
```

**Impact:** Prevents repeated resets when multiple cuts detected within 24 frames of each other.

---

## B4: TrackRecorder Skip Unchanged Frames [MODERATE WIN]

**Location:** `tools/episode_run.py` - `TrackRecorder` class (line ~1536)

**Step 1:** Add to `__init__` (line ~1539)
```python
def __init__(self, *, max_gap: int, remap_ids: bool) -> None:
    self.max_gap = max(1, int(max_gap))
    self.remap_ids = remap_ids
    self._next_export_id = 1
    self._mapping: dict[int, dict[str, int]] = {}
    self._active_exports: set[int] = set()
    self._accumulators: dict[int, TrackAccumulator] = {}
    self._last_recorded: dict[int, dict] = {}  # B4: Track last recorded state
    self.metrics = {
        "tracks_born": 0,
        "tracks_lost": 0,
        "id_switches": 0,
        "forced_splits": 0,
    }
```

**Step 2:** Update `record()` method (line ~1553)
```python
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
    skip_if_unchanged: bool = False,  # B4: New parameter
) -> int:
    if isinstance(bbox, np.ndarray):
        bbox_values = bbox.tolist()
    else:
        bbox_values = bbox

    # B4: Skip if unchanged
    if skip_if_unchanged and tracker_track_id in self._last_recorded:
        last = self._last_recorded[tracker_track_id]
        frame_gap = frame_idx - last["frame_idx"]
        if frame_gap < 5:  # Only check recent frames
            bbox_similar = np.allclose(bbox_values, last["bbox"], rtol=0.05)
            if bbox_similar:
                return last["export_id"]  # Return existing ID without update

    # ... existing record logic ...

    # B4: Update last recorded state
    self._last_recorded[export_id] = {
        "frame_idx": frame_idx,
        "bbox": bbox_values,
        "export_id": export_id,
    }

    return export_id
```

**Step 3:** Use in lightweight updates (line ~3786)
```python
if obj_idx % TRACK_PROCESS_SKIP != 0:
    # Lightweight continuity update only
    recorder.record(
        tracker_track_id=obj.track_id,
        frame_idx=frame_idx,
        ts=ts,
        bbox=obj.bbox,
        class_label=FACE_CLASS_LABEL,
        landmarks=None,
        confidence=float(obj.conf) if obj.conf is not None else None,
        force_new_track=False,
        skip_if_unchanged=True,  # B4: Skip if bbox hasn't changed
    )
    continue
```

**Impact:** Further reduces redundant track accumulator updates.

---

## C3: Async Frame/Crop Exporter [I/O OPTIMIZATION]

**Location:** `tools/episode_run.py` - `FrameExporter` class (line ~2171)

**Step 1:** Add queue and worker to `__init__`
```python
import queue
import threading

class FrameExporter:
    def __init__(self, ep_id: str, save_frames: bool, save_crops: bool, jpeg_quality: int, debug_logger):
        # ... existing init ...

        # C3: Async export queue
        self._export_queue: queue.Queue = queue.Queue(maxsize=64)
        self._worker_thread: threading.Thread | None = None
        self._shutdown = False

        if save_frames or save_crops:
            self._worker_thread = threading.Thread(
                target=self._export_worker,
                name="frame-export-worker",
                daemon=True,
            )
            self._worker_thread.start()
```

**Step 2:** Add worker method
```python
def _export_worker(self) -> None:
    """Background worker that performs actual JPEG encoding and I/O."""
    while True:
        try:
            item = self._export_queue.get(timeout=0.5)
        except queue.Empty:
            if self._shutdown:
                break
            continue

        if item is None:  # Shutdown sentinel
            break

        task_type, args = item
        try:
            if task_type == "frame":
                frame_idx, frame, ts = args
                # ... existing frame save logic ...
            elif task_type == "crop":
                export_id, frame, bbox, ts = args
                # ... existing crop save logic ...
        except Exception as exc:
            LOGGER.error("Frame export worker error: %s", exc)
        finally:
            self._export_queue.task_done()
```

**Step 3:** Update `export()` method
```python
def export(self, frame_idx: int, frame, crop_records: list, ts: float | None = None) -> None:
    """Enqueue frame/crop export instead of doing it synchronously."""
    if self.save_frames:
        self._export_queue.put(("frame", (frame_idx, frame.copy(), ts)))

    if self.save_crops and crop_records:
        for export_id, bbox in crop_records:
            self._export_queue.put(("crop", (export_id, frame.copy(), bbox, ts)))
```

**Step 4:** Add `close()` method
```python
def close(self) -> None:
    """Shutdown worker and wait for pending exports."""
    if self._worker_thread:
        self._shutdown = True
        self._export_queue.put(None)  # Sentinel
        self._worker_thread.join(timeout=30)

    # ... existing write_indexes logic ...
```

**Step 5:** Call `close()` in main pipeline (line ~4033)
```python
recorder.finalize()
if frame_exporter:
    frame_exporter.close()  # C3: Wait for async exports to complete
    # frame_exporter.write_indexes()  # Now called in close()
```

**Impact:** Moves JPEG encoding off hot path, reduces frame processing latency.

---

## G3: Persist Gate Embeddings to Tracks [AVOIDS RECOMPUTATION]

**Location:** `tools/episode_run.py`

**Step 1:** Add field to TrackAccumulator (line ~610)
```python
@dataclass
class TrackAccumulator:
    track_id: int
    class_id: int | str
    first_ts: float
    last_ts: float
    frame_count: int = 0
    bboxes_sampled: list[list[float]] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    landmarks_sampled: list[list[float]] = field(default_factory=list)
    gate_embedding: list[float] | None = None  # G3: Persist gate embedding
```

**Step 2:** Update `TrackRecorder.record()` signature (line ~1553)
```python
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
    skip_if_unchanged: bool = False,
    gate_embedding: np.ndarray | None = None,  # G3: Accept gate embedding
) -> int:
    # ... existing logic ...

    track.add(ts, frame_idx, bbox_values, confidence=confidence, landmarks=landmarks)

    # G3: Store gate embedding if provided
    if gate_embedding is not None:
        track.gate_embedding = gate_embedding.tolist()

    return export_id
```

**Step 3:** Pass embeddings when recording (line ~3797+)
```python
export_id = recorder.record(
    tracker_track_id=obj.track_id,
    frame_idx=frame_idx,
    ts=ts,
    bbox=obj.bbox,
    class_label=class_value,
    landmarks=landmarks,
    confidence=(float(obj.conf) if obj.conf is not None else None),
    force_new_track=force_split,
    gate_embedding=gate_embeddings.get(obj.track_id),  # G3: Pass gate embedding
)
```

**Step 4:** Include in track rows (line ~1628)
```python
def to_row(self) -> dict:
    row = {
        "track_id": self.track_id,
        "class_id": self.class_id,
        # ... existing fields ...
    }
    if self.gate_embedding:
        row["gate_embedding"] = self.gate_embedding  # G3: Include in output
    return row
```

**Step 5:** Use in faces_embed stage (line ~4320+)
Check if track already has gate_embedding and skip ArcFace recomputation.

**Impact:** Avoids recomputing ArcFace embeddings in faces_embed stage if gate embedding exists.

---

## Summary of Expected Combined Impact

After implementing all remaining tasks:

| Optimization | CPU Savings | I/O Savings | Memory Savings |
|--------------|-------------|-------------|----------------|
| D1: cap.grab() | ~15% (decode) | - | - |
| F1: Scene cooldown | ~5% | - | - |
| B4: Skip unchanged | ~10% | - | ~20% |
| C3: Async exporter | ~5% (latency) | Smoother | - |
| G3: Persist gate emb | ~2% | - | - |

**Total Additional Savings:** ~30-40% on top of existing 80-90% reduction.

**Final Expected Performance:**
- Wall-clock: 25 min → **6-8 min** (4x faster)
- CPU: 450% → **~200%** (55% reduction)
- Smoother I/O with no spikes

---

## Testing Commands

```bash
# Test with all optimizations
python tools/episode_run.py \
  --ep-id "TEST-S01E01" \
  --video path/to/episode.mp4 \
  --stride 6 \
  --max-gap 60 \
  --max-gap-sec 2.0 \
  --min-track-length 3 \
  --track-sample-limit 6 \
  --scene-cut-cooldown 24 \
  --save-crops \
  --device auto

# Monitor CPU
watch -n 1 'ps aux | grep episode_run | grep -v grep'

# Check results
wc -l data/episodes/TEST-S01E01/manifests/tracks.jsonl
wc -l data/episodes/TEST-S01E01/manifests/detections.jsonl
ls -lh data/episodes/TEST-S01E01/frames/crops/ | wc -l
```
