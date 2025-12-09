# API Module - CLAUDE.md

FastAPI backend for Screenalytics.

## Architecture

```
routers/          → HTTP endpoints (thin layer)
    ↓
services/         → Business logic
    ↓
storage.py        → S3/local file abstraction
```

## Key Files

| File | Purpose |
|------|---------|
| `routers/episodes.py` | Episode CRUD, status, pipeline triggers |
| `services/grouping.py` | Clustering algorithms, merge logic |
| `services/identities.py` | Identity/cast linking, suggestions |
| `services/track_reps.py` | Representative frame selection |
| `services/storage.py` | S3/MinIO/local file operations |
| `models.py` | Pydantic request/response models |

## Services Overview

### grouping.py
- `cluster_faces()` - Main clustering entry point
- `compute_centroids()` - Calculate cluster centroid embeddings
- `merge_singleton_clusters()` - Quality rescue for singletons
- `_validate_embedding_dimensions()` - Filter mismatched embeddings

### identities.py
- `get_cast_suggestions()` - AI-powered cast matching
- `link_identity_to_cast()` - Commit cast assignment
- `get_temporal_suggestions()` - Time-based cluster hints
- `get_rescued_clusters()` - Quality rescue candidates

### track_reps.py
- `select_representative_frames()` - Best frames for thumbnails
- `compute_track_quality()` - Track similarity metrics

### storage.py
- `get_artifact_path()` - Resolve local/S3 paths
- `get_presigned_url()` - Generate S3 presigned URLs
- `sync_to_s3()` - Upload local artifacts to S3

## Manifest Files

Located in `data/manifests/{ep_id}/`:

| File | Format | Contents |
|------|--------|----------|
| `faces.jsonl` | JSONL | Face detections per frame |
| `tracks.jsonl` | JSONL | Track sequences |
| `identities.json` | JSON | Cluster assignments |
| `cast_links.json` | JSON | Identity → cast mappings |
| `track_metrics.json` | JSON | Quality metrics per track |
| `cluster_centroids.json` | JSON | Centroid embeddings |

## Error Handling Pattern

```python
from models import ApiResult

def some_endpoint():
    try:
        result = do_work()
        return ApiResult(success=True, data=result)
    except SpecificError as e:
        logger.error(f"[ENDPOINT] Failed: {e}")
        return ApiResult(success=False, error=str(e))
```

## Adding New Endpoints

1. Add Pydantic models to `models.py`
2. Add route in appropriate router file
3. Implement logic in services layer
4. New fields should be `Optional` for backwards compat

## Long-Running Jobs

Background jobs use Celery:

```python
from celery_app import celery

@celery.task
def run_pipeline_stage(ep_id: str, stage: str):
    ...
```

Progress tracking via SSE endpoints.

## Testing

```bash
# Run API tests
pytest tests/api/ -v

# Specific test
pytest tests/api/test_episode_status.py -v

# Syntax check
python -m py_compile apps/api/routers/episodes.py
```

## Common Issues

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| 500 on clustering | Embedding dimension mismatch | Check `_validate_embedding_dimensions()` |
| Stale status | Cache not invalidated | Call storage invalidation |
| Missing thumbnails | S3 sync failed | Check storage.py logs |
| Cast suggestions empty | Facebank not loaded | Verify facebank path |
