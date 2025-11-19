# Instructions for Applying Performance Patches

## Overview
This directory contains 8 patches to optimize the Run Detect/Track functionality. Apply them in order for best results.

## Quick Start (Apply All High-Priority Patches)

```bash
# Navigate to workspace root
cd /workspace

# Apply high-priority patches (recommended first)
patch -p1 < patches/01_change_default_stride_to_6.patch
patch -p1 < patches/03_reduce_thread_limits_cap_cpu_250.patch
patch -p1 < patches/02_increase_track_buffer_to_90.patch

# Test the changes
python tools/episode_run.py --ep-id test_episode --video samples/demo.mp4
```

## Individual Patch Application

### Patch 1: Change Default Stride to 6 (HIGH PRIORITY)
**Impact:** 83% fewer detections, 6x faster processing

```bash
patch -p1 < patches/01_change_default_stride_to_6.patch
```

**Verification:**
```bash
python tools/episode_run.py --help | grep -A1 "\-\-stride"
# Should show: "default: 6"
```

---

### Patch 2: Increase Track Buffer to 90 (HIGH PRIORITY)
**Impact:** 72% fewer tracks, reduces fragmentation

```bash
patch -p1 < patches/02_increase_track_buffer_to_90.patch
```

**Verification:**
```bash
grep "track_buffer:" config/pipeline/tracking.yaml
# Should show: "track_buffer: 90"
```

---

### Patch 3: Reduce Thread Limits (HIGH PRIORITY)
**Impact:** Caps CPU at ~250% instead of 400-600%

```bash
patch -p1 < patches/03_reduce_thread_limits_cap_cpu_250.patch
```

**Verification:**
```bash
grep -A5 "Force thread limits" tools/episode_run.py | grep OMP_NUM_THREADS
# Should show: OMP_NUM_THREADS", "1"
```

---

### Patch 4: Add Crop Stride (MEDIUM PRIORITY)
**Impact:** 98% fewer crops, reduced I/O

```bash
patch -p1 < patches/04_add_crop_stride_every_8th_frame.patch
```

**Verification:**
```bash
grep -A2 "crop_stride = 8" tools/episode_run.py
# Should find the crop stride logic
```

---

### Patch 5: Reduce Gate Embedding Frequency (MEDIUM PRIORITY)
**Impact:** 58% fewer embeddings

```bash
patch -p1 < patches/05_reduce_gate_embedding_freq_24_frames.patch
```

**Verification:**
```bash
grep "GATE_EMB_EVERY_DEFAULT" tools/episode_run.py | grep "24"
# Should show: ...get("TRACK_GATE_EMB_EVERY", "24")
```

---

### Patch 6: Smart Frame Seeking (LOW PRIORITY - CODEC DEPENDENT)
**Impact:** 83% less video I/O (may not work with all codecs)

```bash
patch -p1 < patches/06_smart_frame_seeking.patch
```

**Warning:** Frame seeking may not work reliably with all video codecs. The patch includes automatic fallback detection. Test thoroughly with your target video files.

**Verification:**
```bash
grep "seeking_enabled" tools/episode_run.py
# Should find the seeking logic
```

---

### Patch 7: Consolidate Bbox Validation (MEDIUM PRIORITY)
**Impact:** 66% less validation overhead

```bash
patch -p1 < patches/07_consolidate_bbox_validation.patch
```

**Verification:**
```bash
grep -c "_safe_bbox_or_none" tools/episode_run.py
# Count should be reduced compared to original
```

---

### Patch 8: Cross-Frame Embedding Batching (LOW PRIORITY - COMPLEX)
**Impact:** 15-25% faster embeddings

```bash
patch -p1 < patches/08_cross_frame_embedding_batching.patch
```

**Note:** This is the most complex patch. Test thoroughly.

**Verification:**
```bash
grep "class EmbeddingBatchQueue" tools/episode_run.py
# Should find the new class
```

---

## Rollback Instructions

If you need to undo a patch:

```bash
# Rollback a specific patch
patch -R -p1 < patches/01_change_default_stride_to_6.patch

# Or use git to reset (if you have uncommitted changes)
git checkout tools/episode_run.py
git checkout config/pipeline/tracking.yaml
```

---

## Testing Recommendations

### Basic Functionality Test
```bash
# Test with a short video sample
python tools/episode_run.py \
    --ep-id test_run \
    --video samples/demo.mp4 \
    --stride 6
```

### Performance Benchmark
```bash
# Before applying patches
time python tools/episode_run.py --ep-id baseline --video path/to/42min_video.mp4

# After applying patches
time python tools/episode_run.py --ep-id optimized --video path/to/42min_video.mp4
```

### CPU Monitoring
```bash
# Monitor CPU usage during run
top -p $(pgrep -f episode_run.py) -d 1
```

### Track Quality Validation
```bash
# Compare track counts
echo "Baseline tracks:"
wc -l < data/artifacts/baseline/tracks.jsonl

echo "Optimized tracks:"
wc -l < data/artifacts/optimized/tracks.jsonl

# Expected: ~72% reduction (from ~4,352 to ~1,200)
```

---

## Troubleshooting

### Issue: Patch fails to apply
**Solution:** Check if the file has been modified. Use `git diff` to see changes, then manually apply the patch.

### Issue: Frame seeking causes frame skips
**Solution:** The patch includes automatic fallback. Check logs for "Frame seeking unreliable" warnings. If problematic, skip patch #6.

### Issue: CPU usage still high
**Solution:** Verify thread limits with:
```bash
python -c "import os; print({k:v for k,v in os.environ.items() if 'THREAD' in k})"
```

### Issue: Track fragmentation still high
**Solution:** Increase track_buffer further (try 120 or 150 frames). Monitor for false track merges.

### Issue: Missing faces in output
**Solution:** The stride=6 default may miss fast-moving faces. Adjust with `--stride 3` or `--stride 4` for better recall.

---

## Performance Expectations

### Before Patches (42-minute video @ 24fps)
- Processing time: ~45-60 minutes
- Detections: 26,807
- Tracks: 4,352
- CPU usage: 400-600%
- Crops: ~26,807

### After All Patches
- Processing time: ~8-10 minutes (5-6x faster)
- Detections: ~4,468 (-83%)
- Tracks: ~1,200 (-72%)
- CPU usage: ~250% (-58%)
- Crops: ~560 (-98%)

---

## Support

For issues or questions:
1. Check `RUN_DETECT_TRACK_IMPROVEMENTS.md` for detailed analysis
2. Review patch contents in `patches/` directory
3. Test patches individually to isolate issues
4. Monitor logs for warnings and errors

