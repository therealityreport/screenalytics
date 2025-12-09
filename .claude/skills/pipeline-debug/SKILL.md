# Pipeline Debug Skill

Use this skill to diagnose failed or stuck episode pipeline jobs.

## When to Use

- Episode shows "failed" or "stuck" status
- Manifests missing or malformed
- Unexpected clustering results
- Pipeline stage errors in logs

## Diagnostic Workflow

### Step 1: Check Episode Status

```bash
# Via API
curl http://localhost:8000/episodes/{ep_id}

# Or check manifest directory
ls -la data/manifests/{ep_id}/
```

### Step 2: Review Manifests

Check for presence and validity:

| File | Required For |
|------|--------------|
| `faces.jsonl` | Faces ready |
| `tracks.jsonl` | Tracks ready |
| `identities.json` | Clustering complete |
| `cast_links.json` | Cast linking |
| `track_metrics.json` | Quality metrics |

### Step 3: Check Pipeline Logs

Look for these markers:

```
[PHASE] - Stage start/end
[JOB]   - Background job status
[GUARDRAIL] - Quality checks
```

Common error patterns:

| Pattern | Meaning |
|---------|---------|
| `[PHASE] detect FAILED` | Detection crashed |
| `[GUARDRAIL] low quality` | Quality below threshold |
| `[JOB] timeout` | Job exceeded time limit |

### Step 4: Verify S3 Sync (if applicable)

```python
# Check STORAGE_BACKEND
import os
print(os.environ.get("STORAGE_BACKEND", "local"))
```

If S3, check sync status in storage.py logs.

### Step 5: Check Thresholds

Compare `track_metrics.json` against:
- `config/pipeline/thresholds.yaml`
- `ACCEPTANCE_MATRIX.md` (if exists)

## Key Files

| File | Purpose |
|------|---------|
| `tools/episode_run.py` | CLI entry point |
| `apps/api/services/identities.py` | Clustering logic |
| `apps/api/services/grouping.py` | Merge/cluster algorithms |
| `config/pipeline/*.yaml` | Threshold configs |

## Common Fixes

### Missing faces.jsonl

```bash
# Re-run detection
python tools/episode_run.py {ep_id} --stage detect
```

### Stuck in clustering

1. Check embedding dimensions match (512-d for ArcFace)
2. Verify facebank is loaded
3. Check memory usage

### Quality too low

1. Review `track_metrics.json`
2. Adjust thresholds in config
3. Consider manual curation

## Checklist

- [ ] Episode status checked
- [ ] All manifests present
- [ ] Logs reviewed for errors
- [ ] S3 sync verified (if applicable)
- [ ] Thresholds within acceptable range
