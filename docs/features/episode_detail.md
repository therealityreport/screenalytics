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

The Debug/Export section generates a PDF report capturing the run's pipeline statistics, artifact status, and review state.

### Endpoint

```
GET /episodes/{ep_id}/runs/{run_id}/export
```

Returns: `application/pdf` - Screen Time Run Debug Report

### Report Sections

The PDF report includes:

1. **Cover / Executive Summary**
   - Episode ID, Run ID, Generated timestamp
   - Total face/body tracks, clusters, fused identities, screen time gain

2. **Face Detect** - Detection counts, artifact size

3. **Face Track** - Track metrics (born, lost, ID switches, scene cuts)

4. **Face Harvest / Embed** - Harvested faces, aligned faces, embedding file status

5. **Body Detect** - Body detection counts

6. **Body Track** - Body track counts

7. **Track Fusion** - Face/body track counts, fused identity count

8. **Cluster** - Cluster statistics (count, faces, mixed tracks, low cohesion)

9. **Faces Review** - Assigned/locked identity counts (from DB)

10. **Smart Suggestions** - Batch/suggestion counts, dismissed/applied counts (from DB)

11. **Screen Time Analyze** - Duration gains, face-only vs combined time

12. **Appendix: Artifact Manifest** - Complete listing of all artifacts with status and size

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
