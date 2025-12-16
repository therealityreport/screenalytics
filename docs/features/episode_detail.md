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

## Export Run Debug Bundle

The Debug/Export section allows exporting a complete snapshot of a run's state for debugging or archival.

### Endpoint

```
GET /episodes/{ep_id}/runs/{run_id}/export
```

### Export Options

| Toggle | Description |
|--------|-------------|
| Include Artifacts | Raw JSONL files (tracks, faces, identities) |
| Include Images | Thumbnails, crops, frames (large) |
| Include Logs | Pipeline logs and job markers |

### Bundle Contents

The exported ZIP contains:

**Always included:**
- `run_summary.json` - Metadata, schema version, artifact paths
- `jobs.json` - Job execution history from DB
- `identity_assignments.json` - Current identity/person mappings
- `identity_locks.json` - Locked identities for this run
- `smart_suggestion_batches.json` - Suggestion batch metadata
- `smart_suggestions.json` - Individual suggestions with evidence
- `smart_suggestions_applied.json` - Audit trail of applied suggestions

**With include_artifacts=True:**
- `detections.jsonl`
- `tracks.jsonl`
- `track_metrics.json`
- `faces.jsonl`
- `identities.json`
- `cluster_centroids.json`
- `body_tracking/` directory
- `analytics/` directory

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
