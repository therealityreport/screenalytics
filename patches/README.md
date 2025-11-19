# Detect/Track Performance Improvement Patches

This directory contains patches to improve detect/track performance, reduce CPU usage, and create longer tracks.

## Patch Files

1. **patch_01_max_gap_increase.patch** - Increase max gap from 0.5s to 2.0s for longer tracks
2. **patch_02_track_processing_skip.patch** - Skip processing every 6th track
3. **patch_03_crop_saving_skip.patch** - Save crops every 8th track only
4. **patch_04_cpu_limiter.patch** - Add global CPU usage cap at 250%
5. **patch_05_track_buffer_match_max_gap.patch** - Ensure track buffer >= max gap
6. **patch_06_embedding_batch_optimization.patch** - Optimize embedding batch processing
7. **patch_07_skip_unchanged_track_recording.patch** - Skip recording unchanged tracks
8. **patch_08_skip_embedding_for_skipped_tracks.patch** - Skip embeddings for skipped tracks

## Applying Patches

To apply all patches:
```bash
cd /workspace
for patch in patches/patch_*.patch; do
    patch -p1 < "$patch"
done
```

To apply individual patches:
```bash
patch -p1 < patches/patch_01_max_gap_increase.patch
```

## Dependencies

Patch 04 (CPU limiter) requires `psutil`:
```bash
pip install psutil
```

## Configuration

All patches respect environment variables for configuration:

- `TRACK_MAX_GAP_SEC` - Max gap in seconds (default: 2.0)
- `SCREENALYTICS_TRACK_PROCESS_SKIP` - Process every Nth track (default: 6)
- `SCREENALYTICS_TRACK_CROP_SKIP` - Save crops every Nth track (default: 8)
- `SCREENALYTICS_EMBEDDING_BATCH_SIZE` - Embedding batch size (default: 32)

## Expected Results

- Track count: 4,352 → ~2,000-3,000 (30-50% reduction)
- CPU usage: Capped at 250% total
- Embedding ops: 83% reduction (every 6th track)
- Crop saves: 87.5% reduction (every 8th track)
- Processing time: 40-50% faster
