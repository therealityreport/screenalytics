# Faces Manifest / Crops Mismatch Debugging

## Symptoms

1. Cluster preview shows a valid image in Faces Review
2. Clicking "View frames" shows "0 crops loaded"
3. Warning: "Crops on disk are missing for this track. Faces manifest=0 · crops=N"
4. Or info message: "All N faces in this track were auto-skipped (quality filters)"

## Root Causes

### 1. All Faces Skipped (Quality Filters) - Most Common

The pipeline may mark all faces in a track as "skip" due to quality filters:
- `skip: "blurry:13.9"` - Low crop standard deviation
- `skip: "low_det"` - Low detection confidence
- `skip: "small_box"` - Bounding box too small

The Frames View excludes skipped faces by default, showing 0 crops.
However, the cluster preview uses `include_skipped=True` so it finds a valid thumbnail.

**Detection:**
```bash
# Check if track faces have skip flags
grep '"track_id": 78' data/manifests/rhoslc-s06e88/faces.jsonl | grep skip

# Count total vs skipped faces for a track
grep -c '"track_id": 78' data/manifests/rhoslc-s06e88/faces.jsonl
```

**Fix:**
- Enable "Show skipped faces" toggle in Frames View to see the skipped faces
- Click "Unskip all" button to remove skip flags from all faces in the track
- Or use API: `POST /episodes/{ep_id}/tracks/{track_id}/unskip_all`

### 2. Manifest Entries Missing

Faces.jsonl entries were never created for the track, but crops exist on disk.

**Detection:**
```bash
# Check if track has ANY entries (including skipped)
grep -c '"track_id": 78' data/manifests/<ep_id>/faces.jsonl
# Returns 0 if missing

# Check if crops exist on disk
ls data/frames/<ep_id>/crops/track_0078/
```

**Fix:**
- Re-run face embedding stage for the episode
- Or manually create faces.jsonl entries from crop files

### 3. Faces Without Embeddings

Face entries exist but the embedding stage was never run. The face records have
bounding boxes and crop paths but no `embedding` field.

**Detection:**
```bash
# Check if any face entries have embeddings
grep '"track_id": 78' data/manifests/<ep_id>/faces.jsonl | grep -c embedding
# Returns 0 if no embeddings

# Sample a face entry to see its fields
grep '"track_id": 78' data/manifests/<ep_id>/faces.jsonl | head -1 | python3 -m json.tool
```

**Fix:**
- Re-run face embedding stage: `python tools/episode_run.py <ep_id> --stage embed`
- The track representative service will still select a crop based on quality score alone

**Note:** Tracks without embeddings will have `no_embeddings: true` flag in track_reps.jsonl.

### 4. Crops Deleted But Manifest Not Updated

Crop files were deleted but faces.jsonl entries still exist.

**Detection:**
```bash
# Check manifest count
grep -c '"track_id": 78' data/manifests/<ep_id>/faces.jsonl

# Check disk count
ls -1 data/frames/<ep_id>/crops/track_0078/*.jpg | wc -l
```

**Fix:**
- Re-run face embedding stage to regenerate crops
- Or remove orphaned manifest entries

## API Endpoints for Debugging

### Check Track Integrity
```bash
curl "http://localhost:8000/episodes/{ep_id}/tracks/{track_id}/integrity"
```

Response:
```json
{
  "track_id": 78,
  "faces_manifest": 6,      // Total face entries including skipped
  "faces_active": 0,        // Entries without skip flag
  "faces_skipped": 6,       // Entries with skip flag
  "crops_files": 6,         // Crop files on disk
  "ok": true,               // crops >= faces_manifest > 0
  "all_skipped": true       // All faces are skipped
}
```

### List Frames Including Skipped
```bash
curl "http://localhost:8000/episodes/{ep_id}/tracks/{track_id}/frames?include_skipped=true"
```

### Unskip Single Face
```bash
curl -X POST "http://localhost:8000/episodes/{ep_id}/faces/{face_id}/unskip"
```

### Unskip All Faces in Track
```bash
curl -X POST "http://localhost:8000/episodes/{ep_id}/tracks/{track_id}/unskip_all"
```

## Prevention

1. The integrity check now properly counts ALL faces (including skipped)
2. The UI shows an info message when all faces are skipped, not a false "missing" warning
3. Users can toggle "Show skipped faces" to review auto-skipped faces
4. "Unskip all" button allows quick override of quality filters

## File Locations

| File | Purpose |
|------|---------|
| `data/manifests/{ep_id}/faces.jsonl` | Face entries with embeddings, crop paths, skip flags |
| `data/frames/{ep_id}/crops/track_XXXX/` | Crop image files |
| `data/manifests/{ep_id}/track_reps.jsonl` | Track representatives for cluster previews |
| `data/manifests/{ep_id}/tracks.jsonl` | Track metadata including best_crop_rel_path |
