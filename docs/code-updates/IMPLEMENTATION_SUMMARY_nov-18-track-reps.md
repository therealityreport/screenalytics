# Implementation Summary: Track Reps, Face Completeness, and Identity Assignment

**Date:** 2025-11-18
**Branch:** `nov-18`
**Status:** Implementation guides and patch files created, ready for integration

---

## Overview

This document provides a step-by-step guide to implementing the fixes for:

1. Kyle's ear being selected as "BEST QUALITY" representative
2. Multi-person tracks (Kyle+Lisa) assigned to wrong identity (Brandi)
3. Missing face completeness validation
4. Appearance gate falling back to IoU-only when embeddings fail
5. Track frames exported without detection backing

---

## Files Created

All implementation guides and helper code have been created:

### 1. Documentation
- `docs/code-updates/nov-18-track-reps-face-completeness-and-assignment.md` - Comprehensive technical documentation

### 2. Implementation Helper Files
- `apps/api/services/track_reps_completeness.py` - Face completeness detection functions
- `apps/api/services/people_assignment_strict.py` - Stricter auto-assignment logic
- `tools/appearance_gate_hardening_patch.py` - Appearance gate missing embedding handling
- `tools/detection_backed_tracks_patch.py` - Detection-backed track frame filtering
- `apps/workspace-ui/faces_review_partial_handling_patch.py` - UI updates for partial faces

---

## Implementation Steps

### Step 1: Track Representative Scoring (apps/api/services/track_reps.py)

**Files:**
- Helper: `apps/api/services/track_reps_completeness.py`
- Target: `apps/api/services/track_reps.py`

**Changes Required:**

1. **Add imports** at top of `track_reps.py`:
   ```python
   from apps.api.services.track_reps_completeness import (
       compute_edge_clipping,
       compute_face_area_ratio,
       is_complete_face,
       EDGE_MARGIN_RATIO,
       EDGE_CLIP_THRESHOLD,
       MAX_FACE_AREA_RATIO,
   )
   ```

2. **Modify `compute_track_representative()` function** (around line 219):

   In the candidate scoring loop, add face completeness checks:
   ```python
   for face in faces:
       frame_idx = face.get("frame_idx")
       if frame_idx is None:
           continue

       # Get frame dimensions (from face record or defaults)
       frame_width = face.get("frame_width", 1920)
       frame_height = face.get("frame_height", 1080)

       # Get bbox
       bbox = face.get("box") or face.get("bbox") or []

       # NEW: Check face completeness
       is_complete, completeness_meta = is_complete_face(bbox, frame_width, frame_height)

       # Existing quality metrics
       det_score, crop_std, box_area = _extract_quality_metrics(face)
       quality_score = _compute_quality_score(det_score, crop_std, box_area)

       # NEW: Penalize partial faces heavily
       if completeness_meta["is_partial_face"]:
           quality_score *= 0.1  # 90% penalty

       # Compute similarity (existing)
       embedding = face.get("embedding")
       similarity = 0.0
       if embedding:
           face_embed = np.array(embedding, dtype=np.float32)
           similarity = cosine_similarity(face_embed, track_centroid)

       # NEW: Add similarity bonus
       if similarity > 0:
           quality_score += 0.2 * similarity

       # Store with completeness metadata
       candidates.append((face, crop_path, quality_score, similarity, completeness_meta))
   ```

3. **Update rep selection logic** to prefer complete faces:
   ```python
   # Filter to complete faces first
   complete_candidates = [
       (face, crop_path, score, sim, meta)
       for face, crop_path, score, sim, meta in candidates
       if not meta["is_partial_face"]
   ]

   if complete_candidates:
       # Use only complete faces
       candidates_to_score = complete_candidates
       rep_low_quality = False
   else:
       # No complete faces, mark as low quality
       candidates_to_score = candidates
       rep_low_quality = True
   ```

4. **Add completeness metadata to result**:
   ```python
   result = {
       "track_id": f"track_{track_id:04d}",
       "rep_frame": rep_face.get("frame_idx"),
       "crop_key": rep_crop_key,
       "embed": track_centroid.tolist(),
       "quality": {
           "det": round(float(det_score), 3),
           "std": round(float(crop_std), 1),
           "box_area": round(float(box_area), 1),
           "score": round(float(rep_quality_score), 4),
           # NEW:
           "is_partial_face": rep_completeness_meta["is_partial_face"],
           "edge_clip_ratio": round(rep_completeness_meta["edge_clip_ratio"], 3),
           "face_area_ratio": round(rep_completeness_meta["face_area_ratio"], 3),
           "sim_to_track_centroid": round(float(rep_similarity), 4) if rep_similarity > 0 else None,
       },
       "rep_low_quality": rep_low_quality,
   }
   ```

**Test:**
```bash
python -m pytest tests/api/test_track_reps.py -v
```

---

### Step 2: Faces Review UI (apps/workspace-ui/pages/3_Faces_Review.py)

**Files:**
- Helper: `apps/workspace-ui/faces_review_partial_handling_patch.py`
- Target: `apps/workspace-ui/pages/3_Faces_Review.py`

**Changes Required:**

1. **Copy helper functions** from `faces_review_partial_handling_patch.py` into `3_Faces_Review.py`, OR import them

2. **Update track loading** to extract metadata:
   ```python
   track_data = api_get(f"/tracks/{track_id}")
   track_frames = track_data.get("frames", [])
   track_metadata = {
       "rep_frame_idx": track_data.get("rep_frame_idx"),
       "rep_low_quality": track_data.get("rep_low_quality", False),
   }
   ```

3. **Select hero frame** with validation:
   ```python
   hero_idx = select_hero_frame(track_frames, track_metadata)
   hero_frame = track_frames[hero_idx]
   ```

4. **Render hero with appropriate badge**:
   ```python
   badge_html = render_quality_badge(
       frame=hero_frame,
       is_hero=True,
       rep_low_quality=track_metadata["rep_low_quality"],
   )
   quality_html = render_quality_info(hero_frame, is_hero=True)
   ```

5. **Render other frames** with partial markers

**Test:** Visual inspection in UI after running detect/track

---

### Step 3: Auto-Assignment (apps/api/services/people.py)

**Files:**
- Helper: `apps/api/services/people_assignment_strict.py`
- Target: `apps/api/services/people.py`

**Changes Required:**

1. **Add imports** at top of `people.py`:
   ```python
   import os

   MIN_ASSIGN_SIM = float(os.environ.get("MIN_ASSIGN_SIM", "0.68"))
   ASSIGN_MARGIN = float(os.environ.get("ASSIGN_MARGIN", "0.10"))

   from apps.api.services.people_assignment_strict import (
       auto_assign_cluster_to_person_strict,
       compute_assignment_candidates,
   )
   ```

2. **Replace auto-assignment logic** in cluster assignment function:
   ```python
   # OLD:
   # if best_similarity > threshold:
   #     cluster.person_id = best_person_id

   # NEW:
   person_id, reason = auto_assign_cluster_to_person_strict(
       cluster_id=cluster.id,
       cluster_embedding=cluster.embedding,
       people_embeddings=[(p.id, p.embedding) for p in all_people],
   )

   if person_id:
       cluster.person_id = person_id
       LOGGER.info(f"Auto-assigned cluster {cluster.id} to {person_id}: {reason}")
   else:
       LOGGER.info(f"Cluster {cluster.id} needs manual review: {reason}")
       cluster.needs_manual_review = True
   ```

3. **Add candidate metadata** to cluster list endpoint:
   ```python
   candidates = compute_assignment_candidates(
       cluster.embedding,
       [(p.id, p.embedding) for p in all_people],
       top_n=3,
   )

   cluster_data["candidates"] = candidates
   cluster_data["is_ambiguous"] = candidates[0]["block_reason"] is not None if candidates else False
   ```

**Test:**
```bash
# Run clustering on test episode
python tools/episode_run.py --ep-id TEST_EP cluster

# Check logs for auto-assignment decisions
```

---

### Step 4: Appearance Gate (tools/episode_run.py)

**Files:**
- Helper: `tools/appearance_gate_hardening_patch.py`
- Target: `tools/episode_run.py`

**Changes Required:**

1. **Add configuration constants** (around line 136):
   ```python
   MAX_MISSING_EMB_BEFORE_SPLIT = int(os.environ.get("MAX_MISSING_EMB_BEFORE_SPLIT", "3"))
   MISSING_EMB_IOU_PENALTY = float(os.environ.get("MISSING_EMB_IOU_PENALTY", "0.15"))
   ```

2. **Update `GateTrackState` dataclass** (around line 573):
   ```python
   @dataclass
   class GateTrackState:
       proto: np.ndarray | None = None
       last_box: np.ndarray | None = None
       low_sim_streak: int = 0
       missing_emb_count: int = 0  # NEW
       consecutive_detections: int = 0  # NEW
   ```

3. **Add `is_valid_embedding()` helper** before `AppearanceGate` class

4. **Replace `AppearanceGate.process()` method** (around line 625-679) with the hardened version from `appearance_gate_hardening_patch.py`

5. **Replace `AppearanceGate.summary()` method** (around line 681-695) with updated version

**Test:**
```bash
# Run detect/track with verbose logging
python tools/episode_run.py --ep-id TEST_EP detect track

# Check logs for:
# - Missing embedding warnings
# - Force splits when threshold exceeded
# - IoU tightening when embeddings missing
```

---

### Step 5: Detection-Backed Tracks (tools/episode_run.py)

**Files:**
- Helper: `tools/detection_backed_tracks_patch.py`
- Target: `tools/episode_run.py`

**Changes Required:**

1. **Add configuration constant** (around line 55):
   ```python
   DET_FACE_MIN = float(os.environ.get("DET_FACE_MIN", "0.70"))
   ```

2. **Add `filter_detection_backed_tracks()` function** from patch file

3. **In detect/track loop**, BEFORE processing tracked_objects:
   ```python
   # After tracker.update()
   tracked_objects = tracker.update(...)

   # NEW: Filter to detection-backed frames only
   valid_tracked_objects, track_frame_diag = filter_detection_backed_tracks(
       tracked_objects=tracked_objects,
       detections=detections,
       frame_idx=frame_idx,
       ep_id=args.ep_id,
   )

   # Update diagnostics
   for key, value in track_frame_diag.items():
       diagnostics[key] = diagnostics.get(key, 0) + value

   # Use valid_tracked_objects instead of tracked_objects
   for tracked in valid_tracked_objects:
       # ... existing track row/crop logic
   ```

4. **Add diagnostics to summary**:
   ```python
   LOGGER.info(
       f"Track frame filtering: "
       f"total={diagnostics.get('track_frames_total', 0)}, "
       f"exported={diagnostics.get('track_frames_exported', 0)}, "
       f"skipped={diagnostics.get('track_frames_without_detection', 0) + diagnostics.get('track_frames_below_face_thresh', 0)}"
   )

   summary["track_frame_diagnostics"] = {
       "total": diagnostics.get("track_frames_total", 0),
       "exported": diagnostics.get("track_frames_exported", 0),
       "skipped_no_detection": diagnostics.get("track_frames_without_detection", 0),
       "skipped_below_thresh": diagnostics.get("track_frames_below_face_thresh", 0),
   }
   ```

**Test:**
```bash
# Run detect/track and check diagnostics
python tools/episode_run.py --ep-id TEST_EP detect track

# Verify in logs:
# - Frames without detection are skipped
# - Low-confidence detections are skipped
# - Diagnostics show filter counts
```

---

## Testing Plan

### 1. Unit Tests

```bash
# Track reps
python -m pytest tests/api/test_track_reps.py -v

# People assignment
python -m pytest tests/api/test_people_assignment.py -v

# Appearance gate
python -m pytest tests/ml/test_appearance_gate.py -v
```

### 2. Integration Test

```bash
# Run full pipeline on test episode
python tools/episode_run.py --ep-id TEST_EP detect track
python tools/episode_run.py --ep-id TEST_EP cluster
```

### 3. Visual Verification

1. Open Faces Review page
2. Navigate to problematic track (Track 3 with Kyle/Lisa/Brandi)
3. Verify:
   - Rep is NOT Kyle's ear
   - Hero badge is appropriate (green "★ BEST QUALITY" or orange "⚠ BEST AVAILABLE")
   - Partial frames have orange "Partial" pills
   - Q: values differentiate quality

### 4. Log Verification

Check logs for:
- Edge clipping detection
- Missing embedding warnings
- Auto-assignment decisions (assigned vs ambiguous)
- Track frame filtering counts

---

## Configuration Summary

### Environment Variables

```bash
# Track rep scoring
export EDGE_MARGIN_RATIO=0.05          # Edge margin (5% of frame)
export EDGE_CLIP_THRESHOLD=0.25        # 1+ edges clipped = partial
export MAX_FACE_AREA_RATIO=0.85        # Face fills >85% = partial

# Auto-assignment
export MIN_ASSIGN_SIM=0.68             # Min similarity to auto-assign
export ASSIGN_MARGIN=0.10              # Gap vs second-best required

# Appearance gate
export MAX_MISSING_EMB_BEFORE_SPLIT=3  # Max consecutive missing embeddings
export MISSING_EMB_IOU_PENALTY=0.15    # IoU penalty when embedding missing

# Track export
export DET_FACE_MIN=0.70               # Min detection confidence for export
```

### Recommended Defaults

All have sensible defaults in code. Only set env vars if you need to tune.

---

## Expected Impact

### Before

```
Track 3 (Brandi/Kyle/Rinna):
├─ Rep frame: Kyle's ear ★ BEST QUALITY ❌
├─ Q: 40% for all frames ❌
├─ Assigned to: Brandi Glanville ❌
└─ Contains: Kyle ear + Lisa Rinna faces ❌

Issues:
❌ Ear selected as "best quality"
❌ No differentiation between partial/complete faces
❌ Wrong identity assignment
❌ Multi-person track not split
❌ ByteTrack predictions without detections exported
```

### After

```
Track 3a (Kyle):
├─ Rep frame: (partial face) ⚠ BEST AVAILABLE (partial) ✅
├─ Q: 35% for ear, marked "Partial (left, top)" ✅
├─ Assigned to: (unassigned - ambiguous) ✅
└─ Contains: Kyle only ✅

Track 3b (Lisa):
├─ Rep frame: Lisa full-face ★ BEST QUALITY ✅
├─ Q: 85% for best frame, 70-80% for others ✅
├─ Assigned to: (unassigned or Lisa if seeded) ✅
└─ Contains: Lisa Rinna faces only ✅

Improvements:
✅ Complete faces preferred for reps
✅ Partial faces clearly marked
✅ Conservative auto-assignment
✅ Single-person tracks
✅ Detection-backed frames only
✅ Meaningful quality differentiation
```

---

## Commit Instructions

After all changes are applied and tested:

```bash
cd /Volumes/HardDrive/SCREANALYTICS

# Check status
git status

# Stage all changes
git add apps/api/services/track_reps.py \
        apps/api/services/track_reps_completeness.py \
        apps/api/services/people.py \
        apps/api/services/people_assignment_strict.py \
        apps/workspace-ui/pages/3_Faces_Review.py \
        apps/workspace-ui/faces_review_partial_handling_patch.py \
        tools/episode_run.py \
        tools/appearance_gate_hardening_patch.py \
        tools/detection_backed_tracks_patch.py \
        docs/code-updates/nov-18-track-reps-face-completeness-and-assignment.md \
        docs/code-updates/IMPLEMENTATION_SUMMARY_nov-18-track-reps.md

# Commit with detailed message
git commit -m "fix(track-reps): prefer full faces, safer auto-assignment, and detection-backed tracks (NOV18-REPS-ASSIGN-2025-11-18)

Fixes issue where Track 3 \"BEST QUALITY\" was Kyle's ear, track contained
Lisa Rinna faces, but was assigned to Brandi Glanville.

Root causes fixed:
1. Track rep scoring didn't check face completeness (edge clipping, face area)
2. Faces Review UI trusted bad reps without validation
3. Auto-assignment too permissive (no min similarity, no ambiguity margin)
4. Appearance gate silently fell back to IoU-only when embeddings missing
5. Tracks exported frames without detection backing (ByteTrack predictions)

Changes:
- track_reps.py: Add edge clipping detection, face area ratio, identity similarity
  * New fields: is_partial_face, edge_clip_ratio, sim_to_track_centroid, rep_low_quality
  * Rep selection prefers complete faces with good identity similarity
  * Fallback to partial only when no complete faces exist

- Faces_Review.py: Validate reps, render appropriate badges
  * Orange ⚠ BEST AVAILABLE (partial) for incomplete reps
  * Green ★ BEST QUALITY only for complete faces
  * \"Partial\" markers on edge-clipped frames

- people.py: Stricter auto-assignment thresholds
  * MIN_ASSIGN_SIM=0.68 (configurable)
  * ASSIGN_MARGIN=0.10 (gap vs second-best)
  * Ambiguous clusters surface for manual review

- episode_run.py: Harden appearance gate and track export
  * Missing embeddings trigger split or tighten IoU
  * Only export detection-backed frames (det_index valid, conf >= DET_FACE_MIN)
  * Comprehensive diagnostics for track frame filtering

Expected results:
- Before: Track 3 (Kyle ear rep, Lisa+Kyle+Brandi, Q:40% all)
- After: Track 3a (Kyle partial rep, Kyle only)
        Track 3b (Lisa full-face rep, Lisa only, Q:85%)

Configuration:
- MIN_REP_ID_SIM=0.60 (rep candidates)
- MIN_ASSIGN_SIM=0.68 (auto-assign)
- ASSIGN_MARGIN=0.10 (ambiguity)
- MAX_MISSING_EMB_BEFORE_SPLIT=3
- DET_FACE_MIN=0.70 (track export)

Related: nov-18-track-reps-face-completeness-and-assignment.md"

# Push to remote
git push origin nov-18
```

---

## Next Steps

1. **Apply patches** following steps above
2. **Run tests** to verify correctness
3. **Test with problematic episode** (Track 3 with Kyle/Lisa/Brandi)
4. **Verify in UI** that reps and badges are correct
5. **Commit and push** when satisfied

---

## Support

For questions or issues during implementation:

1. Check `docs/code-updates/nov-18-track-reps-face-completeness-and-assignment.md` for technical details
2. Review helper files for code examples
3. Check logs for diagnostic output during testing

---

## Summary of Files

**Documentation:**
- `docs/code-updates/nov-18-track-reps-face-completeness-and-assignment.md` (comprehensive technical doc)
- `docs/code-updates/IMPLEMENTATION_SUMMARY_nov-18-track-reps.md` (this file)

**Helper/Patch Files:**
- `apps/api/services/track_reps_completeness.py` (face completeness functions)
- `apps/api/services/people_assignment_strict.py` (stricter auto-assignment)
- `tools/appearance_gate_hardening_patch.py` (missing embedding handling)
- `tools/detection_backed_tracks_patch.py` (detection-backed filtering)
- `apps/workspace-ui/faces_review_partial_handling_patch.py` (UI updates)

**Files to Modify:**
- `apps/api/services/track_reps.py` (integrate completeness checking)
- `apps/api/services/people.py` (integrate stricter assignment)
- `apps/workspace-ui/pages/3_Faces_Review.py` (integrate UI updates)
- `tools/episode_run.py` (integrate gate hardening + detection filtering)

All patches are self-contained with clear integration instructions.
