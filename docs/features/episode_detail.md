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

## Debug / Export (PDF report + ZIP bundle)

The Debug/Export section allows exporting either:
- a **PDF debug report** (default), or
- a **ZIP debug bundle** (raw artifacts + logs) for debugging/archival.

### Endpoint

```
GET /episodes/{ep_id}/runs/{run_id}/export
```

### Formats

| Query param | Value | Output |
|---|---|---|
| `format` | `pdf` (default) | `application/pdf` debug report |
| `format` | `zip` | `application/zip` raw debug bundle |

### Export Options

These toggles apply to **ZIP** exports only:

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
- `debug_report.pdf` - Same PDF debug report (best-effort; included when generation succeeds)

**With include_artifacts=True:**
- `detections.jsonl`
- `tracks.jsonl`
- `track_metrics.json`
- `faces.jsonl`
- `identities.json`
- `cluster_centroids.json`
- `body_tracking/` directory
- `analytics/` directory

### PDF Report Contents (high level)

The PDF report is a self-contained summary intended for “what happened and why”:
- Run lineage + config snapshots
- A quick **Run Health** section (DB connectivity, body-tracking artifact presence, fusion sanity checks)
- Phase-by-phase counts and diagnostics (detect/track/embed/cluster/screentime)

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
