# ByteTrack Optimization Configuration Reference

## Quick Reference: Before vs After

### Track Continuity Settings

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `TRACK_MAX_GAP_SEC` | 0.5s (15 frames) | 2.5s (75 frames) | Prevent premature track splits |
| `TRACK_BUFFER_BASE_DEFAULT` | 15 | 90 | Allow longer gaps between detections |
| `Buffer scaling formula` | `stride / 3` | `stride * 1` | Proportional scaling for stride |
| `BYTE_TRACK_MATCH_THRESH` | 0.85 | 0.72 | Tolerate bbox jitter/occlusions |
| `TRACK_NEW_THRESH` | 0.70 | 0.55 | Lower barrier for new tracks |

### Data Sampling Settings

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `TRACK_SAMPLE_LIMIT` | None (unlimited) | 8 | Cap JSON size per track |
| Embedding throttle | Every frame | Every 30 frames/track | Reduce CPU overhead |
| Progress events | Every frame | Every 30 frames | Reduce I/O overhead |

### Face Harvesting Settings

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| Crop export interval | Every frame | Every 8 frames | 2 crops/sec @ 30fps |
| Min confidence | 0.60 (detection) | 0.75 (export) | High-quality faces only |
| Min face size | 20px (detection) | 50px (export) | Ignore tiny/distant faces |

## Environment Variables (Optional Overrides)

```bash
# Track continuity (CRITICAL)
export TRACK_MAX_GAP_SEC=2.5              # Allow 2.5-second gaps
export BYTE_TRACK_BUFFER=90               # Base buffer frames
export BYTE_TRACK_MATCH_THRESH=0.72       # IoU matching threshold

# Track thresholds
export BYTE_TRACK_NEW_TRACK_THRESH=0.55   # New track creation threshold
export SCREANALYTICS_TRACK_HIGH_THRESH=0.45  # High confidence threshold

# Sampling
export SCREANALYTICS_TRACK_SAMPLE_LIMIT=8 # Max samples per track in JSON

# Appearance gate (hardcoded in AppearanceGate class)
# - Embedding throttle: 30 frames per track
# - No environment variable needed
```

## Calculated Values for Different Strides

### Stride = 6 (Processing 5 fps @ 30fps video)

| Metric | Value | Calculation |
|--------|-------|-------------|
| Track buffer (frames) | 540 | 90 * 6 |
| Max gap (frames) | 75 | 2.5s * 30fps |
| Detection frames (42 min) | ~12,600 | (42 * 60 * 30) / 6 |
| Export frames (42 min) | ~9,450 | (42 * 60 * 30) / 8 |

### Stride = 1 (Processing every frame)

| Metric | Value | Calculation |
|--------|-------|-------------|
| Track buffer (frames) | 90 | 90 * 1 |
| Max gap (frames) | 75 | 2.5s * 30fps |
| Detection frames (42 min) | ~75,600 | 42 * 60 * 30 |
| Export frames (42 min) | ~9,450 | (42 * 60 * 30) / 8 |

## Expected Track Counts

### 42-Minute Video with 5-10 Cast Members

| Scenario | Expected Tracks | Notes |
|----------|----------------|-------|
| Ideal (no fragmentation) | 50-100 | ~5-10 tracks per cast member |
| Good (minimal fragmentation) | 100-200 | Some scene transitions cause splits |
| Acceptable | 200-500 | Moderate fragmentation from cuts/occlusions |
| **OLD BEHAVIOR** | **4,352** | **Massive over-fragmentation (FIXED)** |

## CPU Performance Targets

### 42-Minute Video @ Stride=6

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| Detection | ~12,600 frames | ~12,600 frames | 0% (unchanged) |
| Tracking | ~12,600 updates | ~12,600 updates | 0% (unchanged) |
| Embeddings (gate) | ~12,600 * N tracks | ~(12,600/30) * N tracks | ~97% |
| Progress events | ~12,600 | ~420 | ~97% |
| Crop exports | ~26,807 | ~1,500-2,500 | ~90% |

### Target CPU Utilization

- **Detection (RetinaFace CoreML)**: 100-150%
- **Tracking (ByteTrack)**: 20-40%
- **Embeddings (ArcFace, throttled)**: 10-20%
- **Export/I/O**: 10-20%
- **Total**: <300% sustained (down from >300%)

## Quality Filter Impact

### Face Crop Export Pipeline

```
All detections (26,807)
  ↓
Frame interval filter (every 8 frames) → ~9,450 candidates
  ↓
Confidence filter (>0.75) → ~5,000 candidates
  ↓
Size filter (>50x50px) → ~1,500-2,500 saved
```

### Filter Rejection Rates

| Filter | Typical Rejection Rate | Purpose |
|--------|----------------------|---------|
| Frame interval | ~65% | Reduce volume, maintain temporal coverage |
| Low confidence (<0.75) | ~50% of remaining | Eliminate blurry/occluded faces |
| Small size (<50px) | ~60% of remaining | Ignore distant/partial faces |
| **Combined** | **~90-92%** | **High-quality faces only** |

## Monitoring Commands

### During Run

```bash
# Watch CPU usage
top -pid $(pgrep -f episode_run.py)

# Monitor progress
watch -n 2 'cat data/jobs/test-progress.json | jq .'

# Track count (after completion)
grep -c "^{" data/manifests/rhobh-s05e01/tracks.jsonl

# Face crops saved
find data/frames/rhobh-s05e01/crops -name "*.jpg" | wc -l
```

### Post-Run Analysis

```bash
# Track statistics
python -c "
import json
from pathlib import Path
tracks = [json.loads(line) for line in Path('data/manifests/rhobh-s05e01/tracks.jsonl').read_text().splitlines()]
print(f'Total tracks: {len(tracks)}')
print(f'Avg frame_count: {sum(t[\"frame_count\"] for t in tracks) / len(tracks):.1f}')
print(f'Avg samples: {sum(len(t.get(\"bboxes_sampled\", [])) for t in tracks) / len(tracks):.1f}')
"

# Detection statistics  
wc -l data/manifests/rhobh-s05e01/detections.jsonl
```

## Troubleshooting

### Still seeing >200 tracks?

1. Check TRACK_MAX_GAP_SEC: `grep TRACK_MAX_GAP_SEC tools/episode_run.py`
2. Verify buffer scaling: Look for `scaled_buffer()` method
3. Scene cuts may legitimately create more tracks

### CPU still >300%?

1. Verify embedding throttle is active (check AppearanceGate.should_extract_embedding)
2. Check progress rate-limiting (should emit every 30 frames)
3. Confirm export throttle (check FrameExporter._quality_filtered_count)

### Too few face crops?

1. Reduce min_confidence from 0.75 to 0.70
2. Reduce min_face_size from 50 to 40
3. Check quality_filtered count in FrameExporter.summary()

### JSON files too large?

1. Verify TRACK_SAMPLE_LIMIT=8 is active
2. Check track.bboxes_sampled length in tracks.jsonl
3. Reduce limit to 6 or 4 if needed

## Validation Checklist

- [ ] Track count: 50-200 (not 4,352)
- [ ] Samples per track: ≤8 in JSON
- [ ] CPU usage: <300% sustained
- [ ] Face crops: 1,500-2,500 (not 26,807)
- [ ] Quality filtered: ~7,500-8,000
- [ ] Debug logs: Not at ERROR level
- [ ] Progress events: ~420 (not 12,600)
- [ ] Embedding throttle: Active (30-frame intervals)


