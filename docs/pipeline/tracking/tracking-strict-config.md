# Strict Tracking Configuration

## Overview

The strict tracking configuration reduces false associations and multi-person track contamination by using tighter thresholds for spatial matching, appearance similarity, and track management.

## Configuration Values

### ByteTrack Spatial Matching (Strict)
- **track_thresh**: `0.65` (was `0.5`) - Minimum confidence to track, filters low-quality detections
- **match_thresh**: `0.85` (was `0.75`) - IoU threshold for bbox matching, tighter spatial matching
- **track_buffer**: `15` (was `30`) - Frames to keep track alive, reduces incorrect associations

### Appearance Gate Thresholds (Strict)
- **gate_appear_hard**: `0.75` (was `0.60`) - Hard split if similarity < 75%
- **gate_appear_soft**: `0.82` (was `0.70`) - Soft split if similarity < 82%
- **gate_appear_streak**: `2` - Consecutive low-sim frames before split
- **gate_iou**: `0.40` (was `0.35`) - Split if spatial jump (IoU < 40%)
- **gate_emb_every**: `5` - Extract embeddings every 5 frames

## How It's Applied

The strict configuration is now the **default** and is applied through multiple layers:

### 1. Hardcoded Defaults (Primary)
File: `tools/episode_run.py` (lines 152-169)
```python
TRACK_BUFFER_BASE_DEFAULT = 15
BYTE_TRACK_MATCH_THRESH_DEFAULT = 0.85
TRACK_HIGH_THRESH_DEFAULT = 0.65
GATE_APPEAR_T_HARD_DEFAULT = 0.75
GATE_APPEAR_T_SOFT_DEFAULT = 0.82
```

### 2. YAML Configuration (Backup)
File: `config/pipeline/tracking.yaml`
- Loaded automatically at module import time
- Only applied if environment variables are not set
- Provides declarative config for future modifications

### 3. Environment Variables (Override)
File: `scripts/set_strict_tracking.sh`
```bash
export SCREANALYTICS_TRACK_BUFFER=15
export BYTE_TRACK_MATCH_THRESH=0.85
export SCREANALYTICS_TRACK_HIGH_THRESH=0.65
```

## Usage

### Default Mode (Strict)
Simply run `episode_run.py` normally - strict settings are now the default:
```bash
python tools/episode_run.py --ep-id rhobh-s05e01
```

### Ultra-Strict Mode (Embeddings Every Frame)
For maximum accuracy at 2-3x slower speed:
```bash
./scripts/detect_track_strict.sh rhobh-s05e01
```

### Custom Settings via Environment
To override specific parameters:
```bash
export BYTE_TRACK_MATCH_THRESH=0.90  # Even stricter
python tools/episode_run.py --ep-id rhobh-s05e01
```

### Load All Strict Settings Manually
```bash
source scripts/set_strict_tracking.sh
python tools/episode_run.py --ep-id rhobh-s05e01
```

## Impact

### Pros
- Fewer multi-person track contaminations
- Better single-person track purity
- More accurate appearance-based splits
- Shorter track lifetimes reduce incorrect associations

### Cons
- More total tracks (due to stricter splitting)
- May split valid tracks if person changes pose/lighting significantly
- Slightly more sensitive to occlusions

## Verification

To verify strict settings are active, check the logs at runtime:
```
[INFO] Applied track_buffer=15 from YAML config
[INFO] Applied match_thresh=0.85 from YAML config
[INFO] Applied track_thresh=0.65 from YAML config
```

Or check the tracker config summary in the final output JSON.

## Migration from Old Default

If you have existing episodes processed with the old loose settings (track_buffer=30, match_thresh=0.75, track_thresh=0.5), you should re-run detection/tracking:

```bash
# Re-process with strict defaults
python tools/episode_run.py --ep-id <episode_id>
```

The new strict defaults will produce cleaner, more accurate tracks.
