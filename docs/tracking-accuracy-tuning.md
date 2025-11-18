# Tracking Accuracy Tuning Guide

## Default Configuration (Strict Mode)

**As of Nov 18, 2025:** Strict tracking is now the **DEFAULT** configuration.

All `detect track` commands now use stricter thresholds for better single-person track accuracy.

### Current Defaults

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Detection threshold** | 0.65 | Filters low-quality detections (ears, partial faces) |
| **Appearance hard split** | 75% | Immediate rejection if similarity < 75% |
| **Appearance soft split** | 82% | Start streak counting if similarity < 82% |
| **Embedding extraction** | Every 5 frames | Balance between accuracy and performance |
| **Track buffer** | 15 frames | Less persistence without detections (0.5s at 30fps) |
| **ByteTrack IoU** | 0.85 | Tighter spatial matching required |

### Old Defaults (Pre-Nov 18)

For reference, the old (too permissive) defaults were:
- Detection threshold: 0.5 (too low, allowed ears/false positives)
- Appearance hard split: 60% (too permissive)
- Appearance soft split: 70% (too permissive)
- Embedding extraction: Variable (often too sparse)
- Track buffer: 30 frames (too long, allowed incorrect associations)
- ByteTrack IoU: 0.8 (too permissive)

---

## Problem: Multi-Person Tracks and Low Similarity Scores

**Symptoms:**
- Tracks contain multiple different people
- Similarity scores drop dramatically mid-track (e.g., 85% → 47%)
- False detections (ears, partial faces) included in tracks
- Incorrect identity assignments during clustering

**Example:** Track 3 containing Kyle's ear + Lisa Rinna faces, but labeled as Brandi Glanville

### Root Causes

#### 1. ByteTrack is Purely Spatial
- Uses IoU (bbox overlap) only, **not appearance/identity**
- If Kyle's ear bbox overlaps with Lisa's face bbox in next frame → same track!
- Cannot distinguish between people, only spatial continuity

#### 2. Appearance Gate Can't Reject Without Embeddings
The appearance gate (identity verifier) can only work when ArcFace embeddings are extracted:

```python
# From episode_run.py:641
if similarity is not None and similarity < self.config.appear_t_hard:
    split = True  # Reject mismatch
```

**Problem:** If `emb_every` is set too high, embeddings are only extracted every N frames. When embedding is None, similarity is None, and the gate **cannot reject** the detection!

**Example case:** 47% similarity should trigger immediate hard split (< 75% threshold), but if no embedding was extracted for that frame, the gate is blind.

#### 3. Detection Threshold Too Low
- Old default of 0.5 allowed low-quality false positives
- Ears, partial faces, reflections could enter tracking pipeline
- **Fixed:** Now 0.65 by default

#### 4. Track Buffer Too Long
- Old default of 30 frames = 1 full second at 30fps
- Allowed tracks to persist too long without detections
- Increased chance of incorrect spatial associations
- **Fixed:** Now 15 frames by default

---

## Ultra-Strict Mode (Every Frame Embeddings)

For cases where you need **absolute maximum accuracy** and can tolerate 2-3x slower processing:

```bash
./scripts/detect_track_strict.sh <ep_id>
```

**Difference from default:**
- Extracts embeddings **EVERY frame** instead of every 5 frames
- Ensures appearance gate can verify identity on every detection
- All other settings identical to default (0.65 det thresh, 0.75/0.82 gates, etc.)

**When to use:**
- Critical scenes with rapid person changes
- Dense crowds with people crossing paths
- High-value footage where accuracy matters more than speed

**Performance impact:**
- ~2-3x slower than default due to ArcFace inference every frame
- May cause thermal throttling on long episodes
- Consider using 480x480 detection size to mitigate heat

---

## Expected Results

### Before (Old Default Config)
```
Track 3: frames 23-48
├─ Frames 23-36: 85% similarity (Kyle's ear/face)
└─ Frames 39-48: 47% similarity (Lisa Rinna) ❌ Should have split!

Issues:
- Multiple people in single track
- Low similarity scores not triggering splits
- False positives (ears) included
- Incorrect cluster assignment (labeled as Brandi)
```

### After (New Default Config)
```
Track 3a: frames 23-30, 85% similarity (Kyle only) ✅
Track 3b: frames 39-48, 85%+ similarity (Lisa only) ✅

Improvements:
- Split at frame 39 due to < 75% similarity
- Single person per track
- Fewer false positives
- Correct identity assignments
```

---

## Advanced Tuning

### If Default Is Too Aggressive

**Symptoms:** Too many short tracks, valid continuous shots split unnecessarily

**Adjustments:**
1. Lower appearance thresholds slightly:
   ```bash
   export TRACK_GATE_APPEAR_HARD=0.70  # Default is 0.75
   export TRACK_GATE_APPEAR_SOFT=0.78  # Default is 0.82
   ```

2. Increase track buffer:
   ```bash
   # Edit config/pipeline/tracking.yaml
   track_buffer: 20  # Default is 15
   ```

3. Lower detection threshold if losing valid faces:
   ```bash
   python tools/episode_run.py --det-thresh 0.60 ...  # Default is 0.65
   ```

### If Still Getting Multi-Person Tracks

**Increase aggressiveness:**
1. Raise appearance hard threshold:
   ```bash
   export TRACK_GATE_APPEAR_HARD=0.80  # Default is 0.75
   ```

2. Extract embeddings more frequently:
   ```bash
   export TRACK_GATE_EMB_EVERY=3  # Default is 5 (or use ultra-strict script for 1)
   ```

3. Increase ByteTrack match threshold:
   ```bash
   # Edit config/pipeline/tracking.yaml
   match_thresh: 0.90  # Default is 0.85
   ```

---

## Manual Configuration

### Environment Variables

```bash
# Appearance gate thresholds
export TRACK_GATE_APPEAR_HARD=0.75    # Hard split threshold (default)
export TRACK_GATE_APPEAR_SOFT=0.82    # Soft split threshold (default)
export TRACK_GATE_APPEAR_STREAK=2     # Consecutive low-sim frames (default)
export TRACK_GATE_IOU=0.40            # Spatial continuity threshold (default)
export TRACK_GATE_EMB_EVERY=5         # Extract embeddings every N frames (default)
```

### Config File

Edit `config/pipeline/tracking.yaml`:

```yaml
track_thresh: 0.65      # Min confidence to track
match_thresh: 0.85      # IoU threshold for bbox matching
track_buffer: 15        # Frames to keep track alive
```

### CLI Arguments

```bash
python tools/episode_run.py \
  --ep-id EP123 \
  --det-thresh 0.70 \
  --gate-appear-hard 0.78 \
  --gate-emb-every 3 \
  detect track
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

## Performance Impact

### Default Config (Emb Every 5 Frames)

**Processing time:**
- Baseline (old defaults): 1.0x
- Current defaults: ~1.3-1.5x (embeddings every 5 frames)

**Worth it:** Significantly better track accuracy with minimal slowdown

### Ultra-Strict (Emb Every Frame)

**Processing time:**
- ~2.5-3.0x slower than old defaults
- ~2.0x slower than current defaults

**Mitigation:**
- CoreML acceleration helps (5-10+ fps on Apple Silicon)
- 480x480 detection size reduces thermal load (already default for CoreML)
- Consider every 2-3 frames instead of every frame:
  ```bash
  export TRACK_GATE_EMB_EVERY=2
  ```

### Thermal Management

If computer overheats:
1. Detection size already optimized (480x480 for CoreML)
2. Reduce embedding frequency:
   ```bash
   export TRACK_GATE_EMB_EVERY=7  # Default is 5
   ```
3. Process fewer frames:
   ```bash
   python tools/episode_run.py --every 2 ...  # Process every 2nd frame
   ```

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

**Key insight:** If `embedding is None`, all appearance checks fail and only IoU check remains. This is why `emb_every` is critical!

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
- Old default: `0.8` → Required 80% overlap
- New default: `0.85` → Requires 85% overlap (tighter spatial matching)

---

## FAQ

**Q: Why did you make strict mode the default?**
A: The old defaults allowed too many multi-person tracks and false positives. Better to have slower but accurate tracking by default.

**Q: Can I go back to the old permissive defaults?**
A: Yes, set environment variables:
```bash
export TRACK_GATE_APPEAR_HARD=0.60
export TRACK_GATE_APPEAR_SOFT=0.70
export TRACK_GATE_IOU=0.35
export TRACK_GATE_EMB_EVERY=0
```
And edit `config/pipeline/tracking.yaml` to restore old values. Not recommended.

**Q: When should I use ultra-strict mode (every frame)?**
A: Only for critical scenes or when default (every 5 frames) still shows multi-person tracks. Expect 2x slowdown.

**Q: Why 47% similarity didn't trigger a split?**
A: Likely no embedding was extracted for that frame (old defaults had sparse extraction). New defaults extract every 5 frames.

**Q: Will this affect clustering/identities?**
A: Yes, positively! Cleaner single-person tracks → more accurate clusters → better identity assignments.

**Q: Can I tune thresholds per-episode?**
A: Yes, use CLI args or environment variables before running detect/track.

---

## Recommended Workflow

1. **Default mode works for most cases:**
   ```bash
   python tools/episode_run.py --ep-id <ep_id> detect track
   ```

2. **Review tracks in UI:**
   - Check similarity scores in track frames
   - Verify single-person constraint
   - Look for false positives (ears, partial faces)

3. **If still seeing multi-person tracks, use ultra-strict:**
   ```bash
   ./scripts/detect_track_strict.sh <ep_id>
   ```

4. **Re-run clustering after fixing tracks:**
   ```bash
   python tools/episode_run.py --ep-id <ep_id> cluster
   ```

5. **Link to cast:**
   ```bash
   python scripts/link_people_to_cast.py
   ```

---

## Related Files

- **Default config:** `config/pipeline/tracking.yaml` (strict settings)
- **Ultra-strict script:** `scripts/detect_track_strict.sh` (embeddings every frame)
- **ByteTrack implementation:** `FEATURES/tracking/src/bytetrack_runner.py`
- **Appearance gate:** `tools/episode_run.py:611-695`
- **Detection logic:** `tools/episode_run.py:2900-3600`
- **Gate defaults:** `tools/episode_run.py:131-136`
