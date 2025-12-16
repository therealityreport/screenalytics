# Smart Suggestions Page

The Smart Suggestions page (`apps/workspace-ui/pages/3_Smart_Suggestions.py`) uses AI to suggest cast assignments for unassigned clusters.

## Run Scoping

All operations are scoped to a specific run_id:

- **Resolution**: Same as Faces Review (query param → active → most recent)
- **API calls**: All include run_id in query params via `_merge_run_params()`

## Suggestion Batches

Suggestions are organized into batches for auditability.

### Batch Lifecycle

1. **Generate**: `POST /episodes/{ep_id}/smart_suggestions/generate`
   - Creates new `suggestion_batches` row
   - Inserts all suggestions into `suggestions` table
   - Returns batch_id

2. **Review**: UI displays suggestions from current batch
   - Batch selector dropdown to switch between batches
   - Each batch is immutable after creation

3. **Apply/Dismiss**: User acts on individual suggestions

### Batch Table Schema

| Column | Description |
|--------|-------------|
| batch_id | UUID primary key |
| ep_id, run_id | Scope identifiers |
| generator_version | Algorithm version used |
| generator_config_json | Parameters (min_similarity, top_k) |
| created_at | Timestamp |

## Suggestions

Individual cast match suggestions within a batch.

### Suggestion Table Schema

| Column | Description |
|--------|-------------|
| suggestion_id | UUID primary key |
| batch_id | FK to suggestion_batches |
| target_identity_id | Cluster to assign |
| suggested_person_id | Cast member to assign to |
| confidence | Match confidence (0-1) |
| evidence_json | Supporting data |
| dismissed | Boolean, default false |

## Suggestion Actions

### Dismiss

Mark suggestions as dismissed (hidden from default view):

```
POST /episodes/{ep_id}/smart_suggestions/dismiss?run_id=...
{
  "batch_id": "...",
  "suggestion_ids": ["...", "..."],
  "dismissed": true
}
```

Dismissed suggestions can be restored by setting `dismissed: false`.

### Apply Single

Apply one suggestion:

```
POST /episodes/{ep_id}/smart_suggestions/apply?run_id=...
{
  "batch_id": "...",
  "suggestion_id": "..."
}
```

**Lock enforcement**: Returns `{"status": "skipped", "reason": "locked"}` for locked identities.

### Apply All

Apply all suggestions in a batch:

```
POST /episodes/{ep_id}/smart_suggestions/apply_all?run_id=...
{
  "batch_id": "..."
}
```

**Requirements**:
- `batch_id` is required in request body
- `run_id` is required in query params

**Lock enforcement**: Skips locked identities and reports count.

**Response**:
```json
{
  "status": "success",
  "counts": {
    "applied": 15,
    "skipped_locked": 3,
    "skipped_dismissed": 2,
    "skipped_already_assigned": 5
  }
}
```

## Apply Tracking

All applies are recorded for audit trail.

### Suggestion Applies Table

| Column | Description |
|--------|-------------|
| apply_id | UUID primary key |
| batch_id | FK to suggestion_batches |
| suggestion_id | FK to suggestions |
| ep_id, run_id | Scope identifiers |
| applied_at | Timestamp |
| applied_by | Optional user identifier |
| changes_json | Before/after state |

## UI Components

### Batch Selector

Dropdown showing all batches for the current run:
- Format: `{batch_id} · {created_at} · {generator_version}`
- Switching batches reloads suggestions

### Apply All Button

Primary action button at top of bulk actions section:
- Requires confirmation for >3 suggestions
- Shows result counts after completion
- Respects identity locks

### Bulk Actions

Secondary actions in columns:
- **Accept High**: Apply high-confidence suggestions
- **Dismiss Low**: Hide low-confidence suggestions
- **Skip Temporal**: Hide temporal-only matches
- **Triage Singletons**: Filter to high-risk single-track clusters

## Run Isolation

Batches and suggestions are fully run-scoped:
- Dismissing in run A doesn't affect run B
- Each run has its own batch history
- Apply tracking is run-scoped

## Related Files

- [apps/workspace-ui/pages/3_Smart_Suggestions.py](../../apps/workspace-ui/pages/3_Smart_Suggestions.py)
- [apps/api/routers/grouping.py](../../apps/api/routers/grouping.py) - Suggestion endpoints
- [apps/api/services/run_persistence.py](../../apps/api/services/run_persistence.py) - DB operations
- [db/migrations/0004_run_debug_persistence.sql](../../db/migrations/0004_run_debug_persistence.sql) - Schema
