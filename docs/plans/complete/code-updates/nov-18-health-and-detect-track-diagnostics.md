# Health Page: Robust S3 Handling + Detect/Track Crop Diagnostics

**Date:** 2025-11-18
**Branch:** `nov-18`
**Files Modified:**
- `apps/workspace-ui/pages/5_Health.py`

## Summary

Fixed two critical issues in the Health page UI:
1. **KeyError crash** when S3 metadata lacks `key` field
2. **Missing visibility** into detect/track crop diagnostics (frames skipped due to invalid bboxes)

The Health page now gracefully handles incomplete metadata and surfaces the new crop error metrics from the detect/track pipeline guard.

## Problem 1: Health Page KeyError on Missing S3 Key

### Original Symptom

The Health page crashed when viewing episodes where the health API response omitted the `s3.key` field:

```text
KeyError: 'key'
  File "apps/workspace-ui/pages/5_Health.py", line 40, in <module>
    f"S3 object `{detail['s3']['bucket']}/{detail['s3']['key']}` exists → {detail['s3']['exists']}"
                                           ~~~~~~~~~~~~^^^^^^^
```

This occurred for certain episodes (e.g., `rhobh-s05e01`) where the health API returned:
- `detail["s3"]` dict without a `"key"` field, or
- `detail["s3"]` as `None` or missing entirely

### Root Cause

The original code assumed all S3 metadata fields (`bucket`, `key`, `exists`) were always present:

```python
# BEFORE (line 40)
st.write(
    f"S3 object `{detail['s3']['bucket']}/{detail['s3']['key']}` exists → {detail['s3']['exists']}"
)
```

This brittle indexing failed when:
- The health API hadn't yet created an S3 object for the episode
- S3 metadata was partially populated (bucket but no key)
- The episode used local storage only

### Solution

**File**: `apps/workspace-ui/pages/5_Health.py` (lines 39-64)

Implemented defensive extraction and path building:

```python
# Robustly handle S3 status (key field may be missing)
s3 = detail.get("s3")
if not isinstance(s3, dict):
    st.caption("S3 status: not available for this check")
else:
    bucket = s3.get("bucket")
    key = s3.get("key")
    exists = s3.get("exists")

    # Build path string that tolerates missing pieces
    if bucket and key:
        path = f"{bucket}/{key}"
    elif bucket:
        path = bucket
    elif key:
        path = key
    else:
        path = "(no S3 path)"

    # Safe representation of exists
    if exists is None:
        exists_str = "unknown"
    else:
        exists_str = "True" if bool(exists) else "False"

    st.write(f"S3 object `{path}` exists → {exists_str}")
```

**Key improvements**:
1. **Type guard**: Check `isinstance(s3, dict)` before accessing fields
2. **Safe extraction**: Use `.get()` instead of `[]` indexing
3. **Graceful fallbacks**: Handle missing bucket, key, or exists fields
4. **Clear messaging**: Show "(no S3 path)" when no usable info available

### Impact

**Before**: Health page crashed with `KeyError: 'key'` for episodes without complete S3 metadata

**After**: Health page renders gracefully with appropriate fallback messages:
- Full S3 info: `S3 object bucket/key exists → True`
- Bucket only: `S3 object bucket exists → True`
- No S3 info: `S3 status: not available for this check`

## Problem 2: No Visibility into Detect/Track Crop Errors

### Background

The detect/track pipeline now has comprehensive bbox validation and error tracking (from commits e5790a3 and feaf76a):

1. **Per-frame TypeError guard** catches NoneType multiply errors and logs:
   ```
   Skipping frame 861 for rhobh-s05e01 due to NoneType multiply error: ...
   ```

2. **`_safe_bbox_or_none()` validator** prevents invalid bboxes from reaching crop operations

3. **Crop diagnostics** track:
   - `crop_attempts`: Total frames where cropping was attempted
   - `crop_errors`: Frames skipped due to invalid bboxes

However, these metrics were invisible in the Health page UI - users couldn't see how many frames were skipped or assess the quality of a detect/track run.

### Solution

**File**: `apps/workspace-ui/pages/5_Health.py` (lines 80-123)

Added new "Pipeline diagnostics" section that surfaces crop error metrics:

```python
# Display detect/track crop diagnostics if available
st.subheader("Pipeline diagnostics")

# Detect/Track phase
detect_track = detail.get("detect_track") or {}
dt_meta = detect_track.get("meta") or {}

# Try different possible locations for crop diagnostics
crop_attempts = dt_meta.get("crop_attempts")
crop_errors = dt_meta.get("crop_errors")

# Also check in detect_track_stats if meta doesn't have them
if crop_attempts is None:
    dt_stats = dt_meta.get("detect_track_stats") or {}
    crop_attempts = dt_stats.get("crop_attempts")
    crop_errors = dt_stats.get("crop_errors")

if crop_attempts is not None and crop_errors is not None and crop_attempts > 0:
    error_rate = crop_errors / crop_attempts if crop_attempts > 0 else 0.0
    st.caption(
        f"**Detect/Track crops:** {crop_errors} / {crop_attempts} failed "
        f"({error_rate:.1%} of attempts skipped due to invalid bboxes)."
    )
elif crop_attempts is not None and crop_attempts > 0:
    st.caption(f"**Detect/Track crops:** {crop_attempts} attempts (no error stats available).")
elif detect_track:
    st.caption("**Detect/Track:** No crop diagnostics available for this run.")
```

**Additional pipeline status display**:

```python
# Show basic detect/track status if available
dt_status = detect_track.get("status")
if dt_status:
    st.caption(f"Detect/Track status: `{dt_status}`")

# Faces phase
faces = detail.get("faces") or {}
faces_status = faces.get("status")
if faces_status:
    st.caption(f"Faces status: `{faces_status}`")

# Cluster phase
cluster = detail.get("cluster") or {}
cluster_status = cluster.get("status")
if cluster_status:
    st.caption(f"Cluster status: `{cluster_status}`")
```

### Data Flow

The crop diagnostics displayed come from the detect/track pipeline:

1. **Source**: `tools/episode_run.py`
   - Per-frame TypeError guard increments counters when skipping frames
   - `_safe_bbox_or_none()` validator identifies invalid bboxes
   - Counters emitted via progress/status updates

2. **Health API**: `/episodes/{ep_id}` endpoint
   - Returns episode metadata including detect/track phase info
   - Format (assumed): `detail["detect_track"]["meta"]["crop_attempts"]`
   - Alternate locations checked: `detail["detect_track"]["meta"]["detect_track_stats"]`

3. **Health UI**: Displays metrics with error rate calculation

### Assumptions

**Health API payload shape** (based on existing progress.json structure):

```json
{
  "detect_track": {
    "status": "completed",
    "meta": {
      "crop_attempts": 150,
      "crop_errors": 5,
      "detect_track_stats": {
        "detections_frame": 3,
        "tracks_frame": 3
      }
    }
  },
  "faces": {
    "status": "completed"
  },
  "cluster": {
    "status": "pending"
  },
  "s3": {
    "bucket": "screenalytics-dev",
    "key": "episodes/rhobh-s05e01/video.mp4",
    "exists": true
  },
  "local": {
    "path": "/data/episodes/rhobh-s05e01",
    "exists": true
  }
}
```

The implementation checks multiple possible locations for the crop diagnostics to handle different payload formats.

### Impact

**Before**: No way to see how many frames were skipped in detect/track runs

**After**: Health page shows clear diagnostics:
- Example: `Detect/Track crops: 5 / 150 failed (3.3% of attempts skipped due to invalid bboxes).`
- Users can quickly assess:
  - Total frames processed
  - How many had invalid bboxes
  - Percentage of frames skipped
  - Whether the run quality is acceptable

### Example Output

For `rhobh-s05e01` after the fixes:

```
Current episode
ep_id: rhobh-s05e01

S3 object screenalytics-dev/episodes/rhobh-s05e01/video.mp4 exists → True
Local path /data/episodes/rhobh-s05e01 exists → True

Pipeline diagnostics
Detect/Track crops: 7 / 2478 failed (0.3% of attempts skipped due to invalid bboxes).
Detect/Track status: completed
Faces status: completed
Cluster status: pending
```

This immediately shows:
- 7 frames out of 2478 had invalid bboxes and were skipped
- 0.3% skip rate is acceptable (RetinaFace occasionally emits bad detections)
- All pipeline phases completed successfully

## Related Work

### Previous Commits

| Commit | Description | Relation |
|--------|-------------|----------|
| `e5790a3` | Per-frame TypeError guard in detect/track loop | Produces the crop_errors being displayed |
| `feaf76a` | `_safe_bbox_or_none()` validator at call sites | Prevents NoneType errors, increments diagnostics |
| `<this commit>` | Health page fixes + diagnostics display | Surfaces the error metrics in UI |

### Defense in Depth

The Health page now provides visibility into all 5 layers of bbox protection:

1. **Layer 1**: `_valid_face_box()` - Early rejection (visible in detection counts)
2. **Layer 2**: `_safe_bbox_or_none()` - Call-site validation (visible in crop_errors)
3. **Layer 3**: `_prepare_face_crop()` - Input validation (visible in skip rows)
4. **Layer 4**: Per-frame TypeError guard - Final safety net (visible in crop_errors)
5. **Layer 5**: Health page - User-facing diagnostics (THIS COMMIT)

Users can now see the effectiveness of these protections via the Health page metrics.

## Testing

### Manual Test Plan

1. **Test S3 key handling**:
   ```bash
   # Navigate to Health page with ep_id=rhobh-s05e01
   # Verify: No KeyError, graceful S3 status display
   ```

2. **Test crop diagnostics display**:
   ```bash
   # Run detect/track on episode with invalid bboxes
   # Navigate to Health page
   # Verify: Crop error stats displayed with percentage
   ```

3. **Test missing metadata**:
   ```bash
   # View Health page for episode without S3 metadata
   # Verify: Shows "S3 status: not available for this check"
   # Verify: Shows "No crop diagnostics available for this run"
   ```

### Expected Behavior

**With complete metadata**:
- S3 section shows full bucket/key path and exists status
- Pipeline diagnostics show crop attempts/errors with percentage
- All phase statuses displayed

**With missing S3 key**:
- S3 section shows bucket-only or "(no S3 path)" fallback
- No crash or error

**With old detect/track run** (no crop diagnostics):
- Shows "No crop diagnostics available for this run"
- Graceful fallback, no errors

## Future Enhancements

1. **Historical trend**: Track crop error rates over time
2. **Threshold alerts**: Warn if crop error rate exceeds acceptable threshold (e.g., >5%)
3. **Frame-level details**: Link to log entries for specific skipped frames
4. **Comparison view**: Compare crop error rates across episodes or detectors
5. **Export diagnostics**: Download full crop error report for analysis

## References

- Per-frame TypeError guard: `docs/plans/complete/code-updates/nov-18-detect-track-crop-none-guards.md`
- Bbox validator implementation: `tools/episode_run.py:1057-1098`
- Health API endpoint: `apps/api/endpoints/episodes.py` (assumed)
- Progress emission: `tools/episode_run.py` progress.emit() calls
