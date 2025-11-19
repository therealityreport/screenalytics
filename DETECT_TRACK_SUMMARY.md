# Detect/Track Performance Improvements - Executive Summary

## Current State
- **42-minute video**: 26,807 detections, 4,352 tracks
- **Max gap**: 0.5 seconds (15 frames at 30fps) - tracks split too frequently
- **No track skipping**: All 4,352 tracks processed every frame
- **No crop skipping**: All crops saved for all tracks
- **No CPU cap**: Unbounded CPU usage across Streamlit + API + detect/track

## Answer to Your Question

**Q: What is the current max time between face detections that a track will continue?**

**A**: Currently **0.5 seconds** (15 frames at 30fps). This is set by `TRACK_MAX_GAP_SEC = 0.5` in `tools/episode_run.py:221`. 

The `TrackRecorder.record()` method checks if the gap between frames exceeds `max_gap` (which is calculated from `TRACK_MAX_GAP_SEC * fps`). If the gap is larger, it creates a new track ID, causing track fragmentation.

## 8 Bugs/Issues Identified

1. **Max Gap Too Short** - 0.5s causes excessive track fragmentation
2. **No Track Processing Skip** - All tracks processed every frame
3. **No Crop Saving Skip** - All crops saved for all tracks
4. **No Global CPU Cap** - No enforcement of 250% CPU limit
5. **Embedding Extraction for Skipped Tracks** - Wastes GPU/CPU cycles
6. **Track Buffer Too Small** - Smaller than max gap, causes premature track death
7. **No Embedding Batching Optimization** - Missing batch size configuration
8. **Excessive Track Recording** - Records unchanged tracks every frame

## 8 Improvements Proposed

1. **Increase Max Gap to 2.0s** - Allow tracks to bridge temporary gaps (60 frames at 30fps)
2. **Skip Processing Every 6th Track** - Reduce CPU/GPU load by 83%
3. **Skip Crop Saving Every 8th Track** - Reduce disk I/O by 87.5%
4. **Add Global CPU Cap at 250%** - Prevent system overload
5. **Increase Track Buffer to Match Max Gap** - Prevent premature track death
6. **Optimize Embedding Batch Size** - Better GPU utilization (default 32)
7. **Skip Unchanged Track Recording** - Reduce accumulator overhead
8. **Skip Embeddings for Skipped Tracks** - Avoid unnecessary computations

## Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Track Count | 4,352 | ~2,000-3,000 | 30-50% reduction |
| Max Gap | 0.5s (15 frames) | 2.0s (60 frames) | 4x increase |
| CPU Usage | Unbounded | Capped at 250% | Controlled |
| Embedding Ops | All tracks | Every 6th | 83% reduction |
| Crop Saves | All tracks | Every 8th | 87.5% reduction |
| Processing Time | Baseline | -40-50% | Significant speedup |

## Files Created

1. **DETECT_TRACK_IMPROVEMENTS.md** - Detailed analysis with code locations and patches
2. **patches/** - 8 patch files ready to apply
3. **DETECT_TRACK_SUMMARY.md** - This summary document

## Quick Start

Apply all patches:
```bash
cd /workspace
for patch in patches/patch_*.patch; do
    patch -p1 < "$patch"
done
```

Install dependency (for CPU limiter):
```bash
pip install psutil
```

Configure via environment variables:
```bash
export TRACK_MAX_GAP_SEC=2.0
export SCREENALYTICS_TRACK_PROCESS_SKIP=6
export SCREENALYTICS_TRACK_CROP_SKIP=8
export SCREENALYTICS_EMBEDDING_BATCH_SIZE=32
```

## Priority Order

1. **P0 (Critical)**: Max gap increase, CPU cap, track buffer fix
2. **P1 (High)**: Track skip, crop skip
3. **P2 (Medium)**: Embedding optimizations, recording skip
4. **P3 (Low)**: Adaptive stride (future enhancement)
