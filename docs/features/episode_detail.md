# Episode Detail Page

The Episode Detail page (`apps/workspace-ui/pages/2_Episode_Detail.py`) is the central hub for managing a single episode's processing pipeline and reviewing its state.

## Run Selector

Episodes support multiple processing "runs" (attempts). Each run is isolated and maintains its own:
- Detection/tracking artifacts
- Face embeddings
- Identity assignments
- Clustering results

### UI Controls

- **Run selector dropdown**: Choose from existing runs or create a new attempt
- **Active run indicator**: Shows current run_id in use
- **New attempt button**: Creates a fresh run_id (UUID format)

### Run Scoping

When navigating to Faces Review or Smart Suggestions:
- The selected run_id is passed via URL query param (`?run_id=...`)
- Child pages operate within that run's scope
- All mutations affect only the selected run

## Export Run Debug Report

The Debug/Export section generates a comprehensive PDF report capturing the run's pipeline configuration, statistics, artifact status, and actionable tuning suggestions.

### Endpoint

```
GET /episodes/{ep_id}/runs/{run_id}/export
```

Returns: `application/pdf` - Screen Time Run Debug Report

### Report Sections

The PDF report includes 12 sections plus an appendix:

**0. Run Inputs & Lineage**
- Episode ID, Run ID, Git SHA, Generated timestamp
- Video metadata (duration, frame rate, resolution if available)
- Model versions (Face Detector, Tracker, Embedding, Body Detector, Re-ID)

**1. Face Detect**
- Detection counts
- Configuration from `detection.yaml` (model_id, confidence_th, min_size, wide_shot_mode)
- Artifacts with sizes and record counts

**2. Face Track**
- Track metrics (born, lost, ID switches, forced splits, scene cuts)
- **Diagnostic warnings** for alarming metrics (high forced splits, high ID switches)
- Configuration from `tracking.yaml` (track_thresh, match_thresh, track_buffer)
- Artifacts with sizes

**3. Face Harvest / Embed**
- Harvested and aligned face counts
- **Diagnostic warning** for low alignment rate
- Configuration from `embedding.yaml` (backend, min_alignment_quality)
- Artifacts with sizes and record counts

**4. Body Detect**
- Body detection counts
- Configuration from `body_detection.yaml` (model, confidence_threshold, detect_every_n_frames)
- Artifacts with sizes

**5. Body Track**
- Body track counts
- Configuration from `body_detection.yaml → person_tracking` (tracker, thresholds, id_offset)
- Artifacts with sizes

**6. Track Fusion**
- **Clarified metrics**: Face Tracks, Body Tracks, Total Tracked IDs (union), Actual Fused Pairs
- Configuration from `track_fusion.yaml` (iou_threshold, reid_handoff settings)
- Artifacts with sizes

**7. Cluster**
- Cluster statistics (count, singleton fraction, mixed tracks, outlier tracks, low cohesion)
- **Diagnostic warnings** for high singleton fraction or mixed tracks
- Configuration from `clustering.yaml` (algorithm, distance metric, cluster_thresh, singleton_merge)
- Artifacts with sizes and identity counts

**8. Faces Review (DB State)**
- Assigned identities, locked identities, unassigned counts
- **DB error context** with impact explanation if DB unavailable
- Data sources

**9. Smart Suggestions**
- Batch counts, suggestion counts, dismissed/applied/pending
- Data sources

**10. Screen Time Analyze**
- **Screen Time Breakdown Table**: Face-only, Body-only, Combined Total, Gain from Body Tracking
- Percentages and explanatory notes
- Configuration from `screen_time_v2.yaml` (active preset, quality_min, gap_tolerance_s)
- Artifacts with sizes

**11. What Likely Needs Tuning**
- Heuristic-based suggestions derived from the metrics
- Stage, Issue, Suggested Action format
- Covers: Detection, Tracking, Embedding, Clustering, Body Tracking, Track Fusion, Screen Time

**Appendix: Artifact Manifest**
- Complete listing with: Filename, Status, Size, Record Count, Pipeline Stage

## Pipeline Jobs

Each job creates execution records in the `job_runs` table:

| Field | Description |
|-------|-------------|
| job_name | Operation performed |
| status | queued/running/succeeded/failed |
| started_at / finished_at | Timestamps |
| artifact_index_json | Output files with sizes/mtimes |
| error_text | Error details if failed |

Jobs are tracked automatically when using the local execution mode.

## Related Files

- [apps/workspace-ui/pages/2_Episode_Detail.py](../../apps/workspace-ui/pages/2_Episode_Detail.py)
- [apps/api/routers/episodes.py](../../apps/api/routers/episodes.py) - Export endpoint
- [apps/api/services/run_export.py](../../apps/api/services/run_export.py) - Bundle builder
- [apps/api/services/run_persistence.py](../../apps/api/services/run_persistence.py) - DB operations
