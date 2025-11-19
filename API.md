# API Reference — Screenalytics

Version: 2.0
Last Updated: 2025-11-18

---

## Base URL

```
http://localhost:8000  # Local dev
https://api.screenalytics.example.com  # Production
```

---

## Authentication

**Current:** None (local dev)
**Future:** API keys or session tokens (TBD)

---

## Response Envelope

All responses follow this format:

```json
{
  "ok": true,
  "data": { ... },
  "meta": {
    "request_id": "req-abc123",
    "timestamp": "2025-11-18T12:34:56Z"
  }
}
```

**Error format:**
```json
{
  "ok": false,
  "error": {
    "code": "EPISODE_NOT_FOUND",
    "message": "Episode 'rhobh-s05e02' not found",
    "details": {}
  },
  "meta": {
    "request_id": "req-abc123"
  }
}
```

---

## Endpoints

### Episodes

#### `POST /episodes`
Create a new episode.

**Request:**
```json
{
  "ep_id": "rhobh-s05e02",
  "show_slug": "rhobh",
  "season": 5,
  "episode": 2,
  "title": "Reunion Part 2",
  "air_date": "2015-04-14"
}
```

**Response:**
```json
{
  "ok": true,
  "data": {
    "ep_id": "rhobh-s05e02",
    "show_slug": "rhobh",
    "season": 5,
    "episode": 2,
    "title": "Reunion Part 2",
    "air_date": "2015-04-14",
    "created_at": "2025-11-18T12:34:56Z"
  }
}
```

#### `GET /episodes/{ep_id}`
Fetch episode metadata.

**Response:**
```json
{
  "ok": true,
  "data": {
    "ep_id": "rhobh-s05e02",
    "show_slug": "rhobh",
    "season": 5,
    "episode": 2,
    "title": "Reunion Part 2",
    "has_video": true,
    "has_tracks": true,
    "has_identities": false
  }
}
```

#### `DELETE /episodes/{ep_id}`
Delete episode and optionally purge artifacts.

**Query params:**
- `delete_s3`: `true` | `false` (default: `false`)

**Response:**
```json
{
  "ok": true,
  "data": {
    "ep_id": "rhobh-s05e02",
    "local_files_deleted": 142,
    "s3_objects_deleted": 1245
  }
}
```

---

### Jobs

#### `POST /jobs/detect_track`
Run detect/track synchronously (SSE streaming).

**Request:**
```json
{
  "ep_id": "rhobh-s05e02",
  "profile": "balanced",
  "save_frames": true,
  "save_crops": false
}
```

**Response (SSE stream):**
```
data: {"phase": "detect", "frames_done": 120, "frames_total": 2400, "elapsed_sec": 15.2, "eta_sec": 288.8}

data: {"phase": "track", "frames_done": 2400, "frames_total": 2400, "elapsed_sec": 305.0}

data: {"phase": "done", "tracks_born": 42, "tracks_lost": 40, "id_switches": 2, "elapsed_sec": 320.5}
```

#### `POST /jobs/detect_track_async`
Run detect/track asynchronously (polling).

**Request:** Same as `/jobs/detect_track`

**Response:**
```json
{
  "ok": true,
  "data": {
    "job_id": "job-abc123",
    "state": "running",
    "ep_id": "rhobh-s05e02",
    "created_at": "2025-11-18T12:34:56Z"
  }
}
```

#### `GET /jobs/{job_id}/progress`
Poll job progress.

**Response:**
```json
{
  "ok": true,
  "data": {
    "job_id": "job-abc123",
    "state": "running",
    "phase": "track",
    "frames_done": 1200,
    "frames_total": 2400,
    "elapsed_sec": 150.0,
    "eta_sec": 150.0
  }
}
```

**States:** `pending` | `running` | `succeeded` | `failed` | `canceled`

#### `POST /jobs/{job_id}/cancel`
Cancel a running job.

**Response:**
```json
{
  "ok": true,
  "data": {
    "job_id": "job-abc123",
    "state": "canceled"
  }
}
```

#### `POST /jobs/faces_embed`
Run faces embedding (SSE streaming).

**Request:**
```json
{
  "ep_id": "rhobh-s05e02",
  "save_crops": true,
  "min_quality": 0.7,
  "max_crops_per_track": 50
}
```

**Response:** SSE stream with progress updates

#### `POST /jobs/faces_embed_async`
Run faces embedding asynchronously.

**Request:** Same as `/jobs/faces_embed`

**Response:** Job ID and state

#### `POST /jobs/cluster`
Run identity clustering (SSE streaming).

**Request:**
```json
{
  "ep_id": "rhobh-s05e02",
  "cluster_thresh": 0.58,
  "min_cluster_size": 2
}
```

**Response:** SSE stream with progress updates

#### `POST /jobs/cluster_async`
Run identity clustering asynchronously.

**Request:** Same as `/jobs/cluster`

**Response:** Job ID and state

#### `POST /jobs/episode_cleanup_async`
Run episode cleanup workflow.

**Request:**
```json
{
  "ep_id": "rhobh-s05e02",
  "actions": ["split_tracks", "reembed", "recluster", "group_clusters"],
  "write_back": true
}
```

**Response:**
```json
{
  "ok": true,
  "data": {
    "job_id": "job-xyz789",
    "state": "running",
    "actions": ["split_tracks", "reembed", "recluster", "group_clusters"]
  }
}
```

---

### Identities (Moderation)

#### `POST /episodes/{ep_id}/identities/merge`
Merge multiple identities into one.

**Request:**
```json
{
  "source_ids": ["identity-00002", "identity-00003"],
  "target_id": "identity-00001"
}
```

**Response:**
```json
{
  "ok": true,
  "data": {
    "target_id": "identity-00001",
    "track_ids": ["track-00001", "track-00005", "track-00012", "track-00042", "track-00056"],
    "sources_deleted": ["identity-00002", "identity-00003"]
  }
}
```

#### `POST /episodes/{ep_id}/identities/split`
Split tracks from an identity into a new identity.

**Request:**
```json
{
  "identity_id": "identity-00001",
  "track_ids_to_split": ["track-00005", "track-00012"]
}
```

**Response:**
```json
{
  "ok": true,
  "data": {
    "original_id": "identity-00001",
    "new_id": "identity-00008",
    "tracks_moved": ["track-00005", "track-00012"]
  }
}
```

#### `POST /episodes/{ep_id}/tracks/{track_id}/move`
Move a track between identities.

**Request:**
```json
{
  "from_identity_id": "identity-00001",
  "to_identity_id": "identity-00002"
}
```

**Response:**
```json
{
  "ok": true,
  "data": {
    "track_id": "track-00005",
    "from": "identity-00001",
    "to": "identity-00002"
  }
}
```

#### `DELETE /episodes/{ep_id}/tracks/{track_id}`
Soft-delete a track.

**Response:**
```json
{
  "ok": true,
  "data": {
    "track_id": "track-00042",
    "deleted": true
  }
}
```

#### `PATCH /episodes/{ep_id}/identities/{identity_id}`
Update identity metadata (lock, labels).

**Request:**
```json
{
  "locked": true,
  "labels": {
    "person_id": "lisa-vanderpump",
    "name": "Lisa Vanderpump"
  }
}
```

**Response:**
```json
{
  "ok": true,
  "data": {
    "identity_id": "identity-00001",
    "locked": true,
    "labels": {
      "person_id": "lisa-vanderpump",
      "name": "Lisa Vanderpump"
    }
  }
}
```

---

### Facebank

#### `POST /cast/{cast_id}/seeds/upload`
Upload a reference face image for a cast member.

**Request (multipart/form-data):**
```
file: <image.jpg>
person_id: "lisa-vanderpump"
```

**Response:**
```json
{
  "ok": true,
  "data": {
    "seed_id": "seed-abc123",
    "person_id": "lisa-vanderpump",
    "display_path": "facebank/lisa-vanderpump/seed-abc123_d.png",
    "embed_path": "facebank/lisa-vanderpump/seed-abc123_e.png",
    "embedding_extracted": true
  }
}
```

#### `GET /episodes/{ep_id}/tracks/{track_id}/crops`
Get presigned URLs for track crops.

**Query params:**
- `sample`: `1–100` (default: `3`, sample every Nth crop)
- `limit`: `1–200` (default: `40`)
- `start_after`: cursor (opaque, from `next_start_after`)

**Response:**
```json
{
  "ok": true,
  "data": {
    "track_id": "track-00001",
    "crops": [
      {
        "frame_idx": 42,
        "crop_path": "crops/track-00001/frame_000042.jpg",
        "presigned_url": "https://s3.amazonaws.com/screenalytics/...?X-Amz-Signature=...",
        "expires_at": "2025-11-18T13:34:56Z"
      }
    ],
    "next_start_after": "frame_000138",
    "has_more": true
  }
}
```

---

### Progress Tracking

#### `GET /episodes/{ep_id}/progress`
Get live progress for episode (any running job).

**Response:**
```json
{
  "ok": true,
  "data": {
    "ep_id": "rhobh-s05e02",
    "phase": "track",
    "frames_done": 1200,
    "frames_total": 2400,
    "elapsed_sec": 150.0,
    "fps_detected": 24.0,
    "analyzed_fps": 8.0,
    "eta_sec": 150.0
  }
}
```

---

### Health

#### `GET /healthz`
Health check.

**Response:**
```json
{
  "status": "ok",
  "version": "2.0.0",
  "uptime_sec": 12345
}
```

---

## Error Codes

| Code | HTTP Status | Meaning |
|------|-------------|---------|
| `EPISODE_NOT_FOUND` | 404 | Episode does not exist |
| `JOB_NOT_FOUND` | 404 | Job ID not found |
| `TRACK_NOT_FOUND` | 404 | Track ID not found |
| `IDENTITY_NOT_FOUND` | 404 | Identity ID not found |
| `INVALID_REQUEST` | 400 | Malformed request body |
| `VALIDATION_ERROR` | 422 | Request failed validation |
| `INTERNAL_ERROR` | 500 | Server error |
| `JOB_ALREADY_RUNNING` | 409 | Job already in progress for this episode |

---

## Rate Limiting

**Current:** None (local dev)
**Future:** 100 requests/minute per IP (TBD)

---

## Webhooks (Future)

Screenalytics can POST events to external URLs:

**Events:**
- `job.completed`
- `job.failed`
- `identity.merged`
- `identity.split`

**Payload:**
```json
{
  "event": "job.completed",
  "ep_id": "rhobh-s05e02",
  "job_id": "job-abc123",
  "timestamp": "2025-11-18T12:34:56Z",
  "data": { ... }
}
```

---

## Documentation

**For implementation details, see:**

- **[Pipeline Overview](docs/pipeline/overview.md)** — What each job does
- **[Artifact Schemas](docs/reference/artifacts_faces_tracks_identities.md)** — Response payload structures
- **[Troubleshooting](docs/ops/troubleshooting_faces_pipeline.md)** — Common API errors

---

**Maintained by:** Screenalytics Engineering
