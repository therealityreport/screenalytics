# Episode Detail: Detect/Track Manifest Fallback

**Date:** 2025-11-17
**Branch:** `nov-17`
**Files Modified:** `apps/workspace-ui/pages/2_Episode_Detail.py`

## Summary

Updated the Episode Detail page to support manifest-based fallback for determining Detect/Track completion status. The page now derives `tracks_ready` from both:

1. The status payload from `/episodes/{ep_id}/status` (primary signal)
2. The presence of detect/track manifests on disk (fallback when API status is missing or stale)

This ensures that the Detect/Track step shows as complete and Faces Harvest controls are properly enabled whenever the underlying manifests exist, even if the status API is missing, stale, or lagging.

## Changes

### File: `apps/workspace-ui/pages/2_Episode_Detail.py`

#### 1. Added manifest checking helper function (lines ~267-275)

```python
def _check_detect_track_manifests() -> bool:
    """
    Check if detect/track manifests exist locally.
    Returns True if tracks.jsonl exists and is readable.
    """
    try:
        return tracks_path.exists() and tracks_path.is_file()
    except Exception:
        return False
```

**Purpose:** Safely checks if `tracks.jsonl` exists locally, with exception handling for missing/inaccessible files.

#### 2. Refactored status computation with manifest fallback (lines ~304-326)

**Before:**
```python
detect_status_value = str(detect_phase_status.get("status") or "missing").lower()
tracks_ready_status = bool((status_payload or {}).get("tracks_ready"))
# ... later ...
tracks_ready = tracks_ready_status
```

**After:**
```python
# Check if detect/track manifests exist locally for fallback
manifests_present = _check_detect_track_manifests()

# Get status values from API
detect_status_value = str(detect_phase_status.get("status") or "missing").lower()
tracks_ready_status = bool((status_payload or {}).get("tracks_ready"))

# Apply manifest-based fallback for detect/track status
using_manifest_fallback = False
if detect_status_value != "success" and manifests_present:
    # Promote status to success when manifests exist but status API is missing/stale
    detect_status_value = "success"
    using_manifest_fallback = True

# Derive tracks_ready from both status API and manifest fallback
tracks_ready = tracks_ready_status or manifests_present
```

**Key changes:**
- Introduced `manifests_present` boolean by calling `_check_detect_track_manifests()`
- Added `using_manifest_fallback` flag to track when status is promoted based on manifests
- Promoted `detect_status_value` to `"success"` when manifests exist but API status is not success
- Computed `tracks_ready` as the logical OR of `tracks_ready_status` (API signal) and `manifests_present` (manifest fallback)

#### 3. Added manifest-fallback diagnostic caption (lines ~350-352)

In the Pipeline Status card's Detect/Track column, added a diagnostic caption that appears only when using the manifest-based fallback:

```python
if detect_status_value == "success":
    st.success("✅ **Detect/Track**: Complete")
    # ... detector/tracker info ...
    # Show manifest-fallback caption when status was inferred from manifests
    if using_manifest_fallback:
        st.caption("ℹ️ _Detect/Track completion inferred from manifests (status API missing/stale)._")
```

**Purpose:** Provides clear debugging information when the UI is relying on manifests instead of the status API.

#### 4. Updated Faces Harvest "Ready to run" check (line ~376)

**Before:**
```python
elif tracks_ready_status:
    st.info("⏳ **Faces Harvest**: Ready to run")
```

**After:**
```python
elif tracks_ready:
    st.info("⏳ **Faces Harvest**: Ready to run")
```

**Purpose:** Uses the manifest-aware `tracks_ready` flag instead of the API-only `tracks_ready_status`.

#### 5. Removed duplicate `tracks_ready` assignment (line ~653)

**Before:**
```python
tracks_ready = tracks_ready_status
faces_ready = faces_ready_state
```

**After:**
```python
# tracks_ready is already computed above with manifest fallback (line ~321)
faces_ready = faces_ready_state
```

**Purpose:** Eliminated duplicate assignment since `tracks_ready` is now computed earlier with the manifest fallback logic.

## How the Manifest-Based Fallback Works

### Manifest Paths Checked

The fallback checks for the existence of:

- **`tracks.jsonl`** (primary indicator): Retrieved via `get_path(ep_id, "tracks")`

The helper function `_check_detect_track_manifests()` verifies:
1. The file exists (`tracks_path.exists()`)
2. The file is a regular file (`tracks_path.is_file()`)
3. Any exceptions during the check are caught and treated as "not present"

### Effective Status Determination

The effective detect/track status is determined using the following logic:

1. **Primary signal (API status):** If `/episodes/{ep_id}/status` returns a `detect_track` status of `"success"`, use that directly.
2. **Fallback signal (manifests):** If the API status is not `"success"` (e.g., `"missing"`, `"failed"`, or any other value) **AND** `tracks.jsonl` exists locally:
   - Promote `detect_status_value` to `"success"`
   - Set `using_manifest_fallback = True` to trigger the diagnostic caption
3. **`tracks_ready` flag:** Computed as `tracks_ready_status OR manifests_present`, meaning the UI considers tracks ready if either:
   - The status API says `tracks_ready=true`, **OR**
   - The manifest file exists locally

## UI Behavior

### When API status reports success

- **Detect/Track status card:** Shows "Complete" with detector/tracker info and detection/track counts
- **Faces Harvest controls:** Enabled (assuming other requirements are met)
- **Manifest-fallback caption:** Does **not** appear

### When API status is missing/stale but manifests exist

- **Detect/Track status card:** Shows "Complete" (status promoted to success)
- **Faces Harvest controls:** Enabled (assuming other requirements are met)
- **Manifest-fallback caption:** **Appears** with text:
  - `"ℹ️ Detect/Track completion inferred from manifests (status API missing/stale)."`
- **User action required:** None - simply refreshing the page is enough to recover the correct state

### When no Detect/Track run has occurred

- **Detect/Track status card:** Shows "Not started" or equivalent
- **Faces Harvest controls:** Disabled
- **Manifest-fallback caption:** Does **not** appear

## Centralized Logic

All Detect/Track completion logic is centralized in a single location (lines ~304-326) and reused throughout the page:

- **Pipeline Status card** (Detect/Track column): Uses `detect_status_value` and `using_manifest_fallback`
- **Pipeline Status card** (Faces Harvest column): Uses `tracks_ready` to determine if harvest is ready
- **Faces Harvest controls**: Uses `tracks_ready` to enable/disable the "Run Faces Harvest" button

This ensures consistent behavior across all UI elements and prevents logic duplication.

## Known Limitations and Edge Cases

### Partially-written or corrupted manifests

The current implementation only checks for file existence and does not validate the contents of `tracks.jsonl`. If the file exists but is:

- Empty
- Partially written (e.g., interrupted mid-write)
- Corrupted (invalid JSON lines)

The UI will still treat Detect/Track as complete. Future enhancements could add lightweight validation (e.g., checking file size > 0 bytes or attempting to read the first line).

### Stale manifests from previous runs

If a Detect/Track job fails or is interrupted, but `tracks.jsonl` exists from a previous successful run, the UI will show the step as complete based on the old manifest. This is generally acceptable since:

- The manifest represents a valid prior completion
- Users can rerun Detect/Track to generate fresh manifests
- The status API will eventually catch up once the new run completes

### Detection manifests not checked

The fallback currently only checks `tracks.jsonl` and does not verify the presence of:

- `detections.jsonl`
- Individual detection frame manifests

This is intentional, as `tracks.jsonl` is the primary indicator of a complete Detect/Track run. Future enhancements could add additional manifest checks if needed.

### Manifest presence without status API

In rare cases where the status API is permanently unavailable or never records completion (e.g., due to a bug or DB issue), the UI will correctly fall back to manifests. However, the diagnostic caption will always appear until the status API is updated.

## Testing

### Manual testing scenarios

1. **Successful Detect/Track with API status:**
   - Run Detect/Track to completion
   - Verify `/episodes/{ep_id}/status` shows `detect_track.status = "success"`
   - Load Episode Detail page
   - Expected: Detect/Track shows as complete, no fallback caption, Faces Harvest is enabled

2. **Successful Detect/Track with missing API status:**
   - Run Detect/Track to completion (manifests exist)
   - Manually clear or corrupt the status API entry (or test with an episode that has manifests but no status record)
   - Load Episode Detail page
   - Expected: Detect/Track shows as complete, fallback caption appears, Faces Harvest is enabled

3. **No Detect/Track run:**
   - Use an episode that has never run Detect/Track (no manifests, no status)
   - Load Episode Detail page
   - Expected: Detect/Track shows as "Not started", no fallback caption, Faces Harvest is disabled

4. **Failed Detect/Track with no manifests:**
   - Run Detect/Track but have it fail (e.g., missing weights, error during processing)
   - Verify no manifests exist
   - Load Episode Detail page
   - Expected: Detect/Track shows as failed/error, no fallback caption, Faces Harvest is disabled

## Related Files

- [apps/workspace-ui/pages/2_Episode_Detail.py](../../apps/workspace-ui/pages/2_Episode_Detail.py) - Episode Detail page (modified)
- [py_screenalytics/artifacts.py](../../py_screenalytics/artifacts.py) - Artifact path resolution (`get_path()`)
- [apps/api/routers/episodes.py](../../apps/api/routers/episodes.py) - Episode status API endpoint

## Future Enhancements

1. **Content validation:** Add lightweight checks to verify manifest files are not empty or corrupted
2. **Extended manifest checks:** Optionally check for `detections.jsonl` in addition to `tracks.jsonl`
3. **Auto-refresh status:** Add a background task to automatically refresh the status API when manifests are detected but status is missing
4. **Timestamp comparison:** Compare manifest modification time with status `finished_at` to detect stale status records
