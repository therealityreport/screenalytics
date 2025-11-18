# Tracking Accuracy Tuning Guide

## Problem: Multi-Person Tracks and Low Similarity Scores

**Symptoms:**
- Tracks contain multiple different people
- Similarity scores drop dramatically mid-track (e.g., 85% → 47%)
- False detections (ears, partial faces) included in tracks
- Incorrect identity assignments during clustering

**Example:** Track 3 containing Kyle's ear + Lisa Rinna faces, but labeled as Brandi Glanville

---

## Root Causes

### 1. ByteTrack is Purely Spatial
- Uses IoU (bbox overlap) only, **not appearance/identity**
- If Kyle's ear bbox overlaps with Lisa's face bbox in next frame → same track!
- Cannot distinguish between people, only spatial continuity

### 2. Appearance Gate Can't Reject Without Embeddings
The appearance gate (identity verifier) can only work when ArcFace embeddings are extracted:

```python
# From episode_run.py:641
if similarity is not None and similarity < self.config.appear_t_hard:
    split = True  # Reject mismatch
```

**Problem:** If `emb_every` is set too high, embeddings are only extracted every N frames. When embedding is None, similarity is None, and the gate **cannot reject** the detection!

**Your case:** 47% similarity should trigger immediate hard split (< 60% threshold), but if no embedding was extracted for that frame, the gate is blind.

### 3. Detection Threshold Too Low
- `RETINAFACE_SCORE_THRESHOLD = 0.5` allows low-quality false positives
- Ears, partial faces, reflections can enter tracking pipeline

### 4. Track Buffer Too Long
- `track_buffer: 30` frames = 1 full second at 30fps
- Allows tracks to persist too long without detections
- Increases chance of incorrect spatial associations

---

## Solution: Strict Tracking Configuration

### Quick Start

Use the pre-configured strict mode:

```bash
./scripts/detect_track_strict.sh <ep_id>
```

### What Changed

| Parameter | Default | Strict | Impact |
|-----------|---------|--------|--------|
| **Detection threshold** | 0.5 | 0.65 | Filters low-quality detections (ears, partial faces) |
| **Appearance hard split** | 60% | 75% | More aggressive rejection of mismatches |
| **Appearance soft split** | 70% | 82% | Earlier streak counting for subtle drift |
| **Embedding extraction** | Every N frames | **Every frame** | Gate can always verify identity |
| **Track buffer** | 30 frames | 15 frames | Less persistence without detections |
| **ByteTrack IoU** | 0.8 | 0.85 | Tighter spatial matching required |

### Expected Results

**Before (default config):**
- Track 3: Frames 23-48, similarity 85% → 47%, contains Kyle + Lisa + labeled as Brandi
- Multiple people per track
- False positives included

**After (strict config):**
- Track 3a: Frames 23-30, similarity 85-84%, Kyle only
- Track 3b: Frames 39-48, similarity 85%+, Lisa only (split at frame 39 due to <75% similarity)
- Single person per track
- Fewer false positives

---

## Advanced Tuning

### Manual Configuration

Edit `config/pipeline/tracking-strict.yaml` or set environment variables:

```bash
# Appearance gate thresholds
export TRACK_GATE_APPEAR_HARD=0.75    # Hard split threshold
export TRACK_GATE_APPEAR_SOFT=0.82    # Soft split threshold
export TRACK_GATE_APPEAR_STREAK=2     # Consecutive low-sim frames
export TRACK_GATE_IOU=0.40            # Spatial continuity threshold
export TRACK_GATE_EMB_EVERY=1         # Extract embeddings every N frames (1=every frame)

# Detection threshold
python tools/episode_run.py --det-thresh 0.65 ...
```

### If Strict Mode Too Aggressive

**Symptoms:** Too many short tracks, valid continuous shots split unnecessarily

**Adjustments:**
1. Lower appearance thresholds slightly:
   ```bash
   export TRACK_GATE_APPEAR_HARD=0.70  # Was 0.75
   export TRACK_GATE_APPEAR_SOFT=0.78  # Was 0.82
   ```

2. Increase track buffer:
   ```yaml
   track_buffer: 20  # Was 15 in strict, 30 in default
   ```

3. Lower detection threshold if losing valid faces:
   ```bash
   --det-thresh 0.60  # Was 0.65
   ```

### If Still Getting Multi-Person Tracks

**Increase aggressiveness:**
1. Raise appearance hard threshold:
   ```bash
   export TRACK_GATE_APPEAR_HARD=0.80  # Was 0.75
   ```

2. Extract embeddings more frequently (if using stride > 1):
   ```bash
   export TRACK_GATE_EMB_EVERY=1  # Every frame
   ```

3. Increase ByteTrack match threshold:
   ```yaml
   match_thresh: 0.90  # Was 0.85
   ```

---

## Understanding the Logs

### Appearance Gate Split Logs

```
[gate] split track=42 f=156 sim=0.632 iou=0.892 reason=hard
```

- `track=42`: ByteTrack ID being split
- `f=156`: Frame number where split occurred
- `sim=0.632`: Cosine similarity (63.2%)
- `iou=0.892`: Spatial IoU (89.2% overlap)
- `reason=hard`: Split because similarity < `appear_t_hard` (75%)

**Reasons:**
- `hard`: Similarity < hard threshold (immediate split)
- `streak`: Similarity < soft threshold for N consecutive frames
- `iou`: Spatial jump (bbox moved too far)

### Gate Summary Stats

After detect/track completes:

```json
{
  "gate": {
    "splits": {
      "hard": 23,    // Immediate similarity rejections
      "streak": 12,  // Gradual drift rejections
      "iou": 5,      // Spatial jump rejections
      "total": 40
    },
    "avg_similarity": 0.87
  }
}
```

**Good indicators:**
- `avg_similarity > 0.80` → tracks are identity-coherent
- `hard splits > 0` → gate is actively rejecting mismatches
- Low `streak` splits → minimal gradual drift

---

## Technical Deep Dive

### Appearance Gate Logic

From `tools/episode_run.py:625-679`:

```python
def process(self, tracker_id, bbox, embedding, frame_idx):
    # Compute similarity to track prototype
    similarity = _cosine_similarity(embedding, state.proto)

    # Hard split: immediate rejection
    if similarity is not None and similarity < self.config.appear_t_hard:
        split = True
        reason = "hard"

    # Soft split: consecutive low-similarity frames
    elif similarity is not None and similarity < self.config.appear_t_soft:
        state.low_sim_streak += 1
        if state.low_sim_streak >= self.config.appear_streak:
            split = True
            reason = "streak"

    # Spatial split: bbox moved too far
    if iou < self.config.gate_iou:
        split = True
        reason = "iou"

    # Update prototype with exponential moving average
    if not split and embedding is not None:
        mixed = proto_momentum * state.proto + (1 - proto_momentum) * embedding
        state.proto = _l2_normalize(mixed)
```

**Key insight:** If `embedding is None`, all appearance checks fail and only IoU check remains. This is why `--gate-emb-every 1` is critical!

### ByteTrack IoU Matching

From `FEATURES/tracking/src/bytetrack_runner.py:55-70`:

```python
def iou(box_a, box_b):
    # Compute intersection area
    inter_area = inter_w * inter_h

    # Compute union area
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter_area

    return inter_area / denom if denom > 0 else 0.0
```

**IoU thresholds:**
- `match_thresh: 0.8` → Requires 80% overlap to associate detection with existing track
- `match_thresh: 0.85` (strict) → Requires 85% overlap (tighter spatial matching)

---

## Recommended Workflow

1. **Test with strict config first:**
   ```bash
   ./scripts/detect_track_strict.sh test-ep-id
   ```

2. **Review tracks in UI:**
   - Check similarity scores in track frames
   - Verify single-person constraint
   - Look for false positives (ears, partial faces)

3. **Tune if needed:**
   - Too aggressive → lower thresholds
   - Still multi-person → raise thresholds

4. **Set as default (optional):**
   ```bash
   # In .bashrc or .zshrc
   export TRACK_GATE_APPEAR_HARD=0.75
   export TRACK_GATE_APPEAR_SOFT=0.82
   export TRACK_GATE_EMB_EVERY=1
   ```

5. **Re-run detect/track for all episodes:**
   ```bash
   for ep in $(ls data/episodes/); do
       ./scripts/detect_track_strict.sh "$ep"
   done
   ```

---

## Performance Impact

### Strict Config Overhead

**Embedding extraction every frame:**
- Default: ~1.5-2.0x detect/track time (embeddings every 3-5 frames)
- Strict: ~2.5-3.0x detect/track time (embeddings every frame)
- **Worth it:** Ensures gate can always verify identity

**Mitigation:**
- CoreML acceleration helps (5-10+ fps on Apple Silicon)
- 480x480 detection size reduces thermal load
- Consider extracting embeddings every 2 frames if 1 is too slow:
  ```bash
  export TRACK_GATE_EMB_EVERY=2
  ```

### Thermal Management

If computer overheats with strict config:
1. Reduce detection size (already set to 480x480)
2. Reduce embedding frequency:
   ```bash
   export TRACK_GATE_EMB_EVERY=2  # Every other frame
   ```
3. Process fewer frames:
   ```bash
   --every 2  # Process every 2nd frame
   ```

---

## FAQ

**Q: Why 47% similarity didn't trigger a split?**
A: Likely no embedding was extracted for that frame (embedding=None), so the appearance gate couldn't verify identity.

**Q: Why are ears being detected as faces?**
A: RetinaFace threshold too low (0.5). Strict config raises to 0.65.

**Q: Can I tune thresholds per-episode?**
A: Yes, pass CLI args:
```bash
python tools/episode_run.py --ep-id EP123 \
  --det-thresh 0.70 \
  --gate-appear-hard 0.78 \
  detect track
```

**Q: What if I want even stricter (no multi-person tracks allowed)?**
A: Set very high appearance thresholds:
```bash
export TRACK_GATE_APPEAR_HARD=0.85
export TRACK_GATE_APPEAR_SOFT=0.90
```

**Q: Will this affect clustering/identities?**
A: Yes, positively! Cleaner single-person tracks → more accurate clusters → better identity assignments.

---

## Related Files

- **Strict config:** `config/pipeline/tracking-strict.yaml`
- **Strict script:** `scripts/detect_track_strict.sh`
- **ByteTrack implementation:** `FEATURES/tracking/src/bytetrack_runner.py`
- **Appearance gate:** `tools/episode_run.py:611-695`
- **Detection logic:** `tools/episode_run.py:2900-3600`

---

## Next Steps

After tuning tracking:
1. **Re-run clustering:** `python tools/episode_run.py --ep-id <ep_id> cluster`
2. **Review identities:** Check if Lisa/Kyle/Brandi are correctly separated
3. **Link to cast:** `python scripts/link_people_to_cast.py` to assign cluster IDs to cast names

