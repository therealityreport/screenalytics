# Faces Review Page

The Faces Review page (`apps/workspace-ui/pages/3_Faces_Review.py`) provides manual curation of identity assignments and face quality.

## Run Scoping

All operations on this page are scoped to a specific run_id:

- **Resolution order**: Query param → active_run_id file → most recent run
- **Enforcement**: Page blocks if no valid run_id found
- **API calls**: All mutations include run_id in query params

### Query Parameter

Navigate with `?ep_id={ep_id}&run_id={run_id}` to open a specific run.

## Identity Assignment

Clusters (grouped face tracks) can be assigned to cast members:

### Assignment Methods

| Method | Description |
|--------|-------------|
| Manual | User explicitly assigns via dropdown |
| Smart Suggestions | AI-suggested matches accepted by user |
| Auto | Pipeline auto-assignment based on facebank |

### Mutation Endpoints

All mutations require `run_id` query param:

- `POST /episodes/{ep_id}/clusters/group` - Assign cluster to cast
- `POST /identities/{ep_id}/rename` - Rename identity label
- `POST /identities/{ep_id}/merge` - Merge two identities
- `POST /identities/{ep_id}/move_track` - Move track to different identity

## Identity Locks

Prevent accidental modification of curated identities.

### Lock States

| State | Behavior |
|-------|----------|
| Locked | Cannot be modified by Smart Suggestions or auto-assign |
| Unlocked | Normal mutation allowed |

### API Endpoints

```
POST /episodes/{ep_id}/identities/{identity_id}/lock?run_id=...
POST /episodes/{ep_id}/identities/{identity_id}/unlock?run_id=...
GET /episodes/{ep_id}/identity_locks?run_id=...
```

### Lock Enforcement

- Smart Suggestions `apply` skips locked identities (returns `status: "skipped"`)
- Smart Suggestions `apply_all` counts skipped locks in response
- Locks are persisted in `identity_locks` table

### Lock Payload

```json
{
  "locked_by": "user@example.com",
  "reason": "Reviewed and confirmed correct"
}
```

## Run Isolation

Each run maintains independent state:

- Identity assignments don't leak between runs
- Locks are run-scoped
- Cluster data is stored in run-specific directories

### Data Paths

```
data/manifests/{ep_id}/runs/{run_id}/
├── tracks.jsonl
├── faces.jsonl
├── identities.json
└── cluster_centroids.json
```

## Related Files

- [apps/workspace-ui/pages/3_Faces_Review.py](../../apps/workspace-ui/pages/3_Faces_Review.py)
- [apps/api/routers/face_review.py](../../apps/api/routers/face_review.py)
- [apps/api/routers/identities.py](../../apps/api/routers/identities.py)
- [apps/api/services/face_review.py](../../apps/api/services/face_review.py)
- [apps/api/services/identities.py](../../apps/api/services/identities.py)
