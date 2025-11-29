# Faces Review Guide

Complete walkthrough of the Faces & Tracks Review page (`apps/workspace-ui/pages/3_Faces_Review.py`). Use it after **Detect/Track → Faces Embed → Cluster** are complete. This page is for QA, cast assignment, and track/frame cleanup.

## Prerequisites
- Episode has detections, tracks, faces embeds, and clustering artifacts.
- Local video is mirrored (episode header shows ✅); otherwise use **Mirror from S3**.
- Use face-first detectors (RetinaFace or YOLOv8-face) for best similarity numbers.

## Page Map (top row)
- **Similarity Scores Guide** expander: in-page legend; full doc at `docs/similarity-scores-guide.md`.
- **Help** link: opens this guide.
- **Episode header**: detector/tracker, S3 keys, Mirror from S3, Episode Detail shortcut.
- **Cluster Cleanup** popover: pick actions, presets, preview/dry-run, backup/undo, protect recent edits.
- **Auto-assign toggle**: enables ≥85% similarity auto-link after refresh.
- **Refresh Values**: recomputes similarity, pulls facebank/assigned suggestions, optional auto-link.
- **Save Progress**: flushes assignments and refreshes Smart Suggestions.
- **Smart Suggestions**: recompute and jump to Smart Suggestions page.
- **Recover Noise Tracks** popover: expand single-frame tracks using adjacent-frame similarity; includes preview + undo.

## Recommended Flow
1) Confirm episode context in the header; mirror video if missing.  
2) Open **Cluster Cleanup** → choose preset (**Quick** = split only, **Standard** = split+reembed+group). Preview or dry-run if unsure. Keep **Protect recently-edited** on to avoid undoing manual fixes. Run cleanup (auto-backups and undo provided).  
3) Optional: **Recover Noise Tracks**. Start with defaults (±8 frames, ≥70% sim). Use **Preview** to estimate impact; the run creates a backup and records history.  
4) **Refresh Values** with auto-assign enabled to tighten similarities and link high-confidence clusters to cast.  
5) Work the **Needs Cast Assignment** queue: review auto-people + unassigned clusters, use cast/facebank suggestions, and save often.  
6) Use **Cast Lineup** cards to jump into a person, then:
   - **Person view**: check cluster counts, cohesion badge, “View all tracks” for outlier detection.
   - **Cluster view**: inspect track reps, add to comparison, assign/rename/merge/delete as needed.
   - **Track view**: sample slider, auto page sizing, Track Health badges, carousel, move/rename/archive track, generate overlays, move/delete selected frames.  
7) Use **Cluster Comparison** (sidebar) for up to 3 clusters; merge when same person or all unassigned.  
8) When finished, **Save Progress**; rerun **Smart Suggestions** to refresh downstream assists.

## Job/Action Reference
- **Cluster Cleanup** (unassigned-only)
  - Actions: `split_tracks` (fix mixed tracks), `reembed` (refresh embeddings), `group_clusters` (auto-group unassigned). Defaults come from the chosen preset.
  - Shows preview stats and warnings; dry-run describes planned changes.
  - Auto-backup before run; **Undo Last Cleanup** available when backups exist.
  - Honors “Protect recently-edited” identities to avoid overwriting fresh manual work.
- **Refresh Similarity & Auto-link**
  - Recomputes cluster/track/frame similarities, then (if toggle on) auto-assigns clusters to cast when facebank similarity ≥85%. Stores cast suggestions for the queue.
- **Recover Noise Tracks**
  - Targets single-frame tracks; expands using crops within ±`frame_window` (default 8) if similarity ≥ `min_similarity` (default 70%).
  - Preview summarizes counts; full run creates a backup and logs tracks expanded/faces merged with undo support.
- **Smart Suggestions**
  - Recomputes suggestions from current assignments and opens the Smart Suggestions page for bulk acceptance.

## Views and Controls
- **People (default)**
  - Cast Lineup grid (5 cols), cast filter, featured thumbnails from facebank/rep crops/best cluster crops.
  - Metrics per cast/person: clusters, tracks, frames, cohesion badge.
  - **Needs Cast Assignment** queue merges auto-people + unassigned clusters. Uses facebank, assigned-cluster, and cross-episode suggestions.
  - Each unassigned card shows filtered tracks (single-frame noise is suppressed on multi-track clusters), quality badges, and quick assign/delete controls.
- **Person view**
  - Aggregates all episode clusters for a person; shows totals and cohesion.
  - “View all tracks” opens Cast Tracks view sorted for outlier hunting.
- **Cluster view**
  - Track rep grid with similarity/quality badges, pagination/sampling, quick assignment to cast name, move/merge/delete cluster, add to comparison set.
  - Prefetches adjacent clusters for faster navigation.
- **Cast Tracks view**
  - One crop per track across the person, sortable by cast track score to catch mis-assignments.
- **Track view**
  - Sampling slider (1–20), adaptive page size, best-frame pinning, Track Health (Track→Cluster, Cast Track Score, frame similarity stats, quality, drift).
  - Frame carousel and grid show per-face thumbnails only for the active track; includes overlay generation.
  - Actions: move track to another identity, unassign, archive (with auto-archive payload), move selected frames to another/new identity or delete selected frames.
- **Cluster Comparison (sidebar)**
  - Up to 3 clusters; shows cast/person, cohesion, representative crops, and similarity matrix. Merge enabled when all unassigned or same person.

## Tips and Safety
- Local fallback banner means crops are served from disk instead of S3—expect possible staleness.
- Warnings appear when tracks were produced with legacy detectors; rerun detect/track for better embeddings.
- Assignments invalidate caches automatically; use **Save Progress** after bursts of edits.
- For storage-sensitive runs, watch sample sliders and page sizes in Track view; refresh crops if thumbnails look stale.
