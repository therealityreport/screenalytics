# API Reference — Screenalytics

Version: 2.1
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
  "stride": 6,
  "fps": null,
  "device": "auto",
  "save_frames": false,
  "save_crops": false,
  "jpeg_quality": 85,
  "detector": "retinaface",
  "tracker": "bytetrack",
  "max_gap": 30,
  "det_thresh": null,
  "scene_detector": "pyscenedetect",
  "scene_threshold": 27.0,
  "scene_min_len": 12,
  "scene_warmup_dets": 3,
  "track_high_thresh": 0.5,
  "new_track_thresh": 0.5,
  "track_buffer": 30,
  "min_box_area": 20.0
}
```

**Request Parameters:**
- `ep_id` (string, required): Episode identifier
- `profile` (enum, optional): Performance profile (`fast_cpu` alias for `low_power`, `balanced`, `high_accuracy`)
  - Overrides default values for `stride`, `fps`, and detection parameters
  - See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for profile details
- `stride` (int, default: 6): Frame sampling stride (every Nth frame)
- `fps` (float, optional): Target FPS for sampling (null = use source FPS)
- `device` (enum, default: "auto"): Execution device (`auto`, `cpu`, `mps`, `coreml`, `metal`, `apple`, `cuda`)
- `save_frames` (bool, default: false): Export sampled full frames
- `save_crops` (bool, default: false): Export per-track crops
- `jpeg_quality` (int, default: 85): JPEG quality for exports (1-100)
- `detector` (string, default: "retinaface"): Face detector backend
- `tracker` (string, default: "bytetrack"): Object tracker backend
- `max_gap` (int, default: 30): Maximum frames to interpolate across gaps
- `det_thresh` (float, optional): Detection confidence threshold (0.0-1.0)
- `scene_detector` (enum, default: "pyscenedetect"): Scene detector (`pyscenedetect`, `internal`, `off`)
- `scene_threshold` (float, default: 27.0): Scene cut sensitivity
- `scene_min_len` (int, default: 12): Minimum scene length in frames
- `scene_warmup_dets` (int, default: 3): Detections to accumulate before scene start
- `track_high_thresh` (float, default: 0.5): High confidence track matching threshold
- `new_track_thresh` (float, default: 0.5): New track creation threshold
- `track_buffer` (int, default: 30): Frames to buffer lost tracks before deletion
- `min_box_area` (float, default: 20.0): Minimum bounding box area

**Profile Resolution Order:**
1. Explicit request parameters (highest priority)
2. Environment variables
3. Profile preset values
4. Stage config defaults
5. Hardcoded fallbacks (lowest priority)

**Response (SSE stream):**
```
data: {"phase": "detect", "frames_done": 120, "frames_total": 2400, "elapsed_sec": 15.2, "eta_sec": 288.8}

data: {"phase": "track", "frames_done": 2400, "frames_total": 2400, "elapsed_sec": 305.0}

data: {"phase": "done", "tracks_born": 42, "tracks_lost": 40, "id_switches": 2, "elapsed_sec": 320.5}
```

See [docs/pipeline/detect_track_faces.md](docs/pipeline/detect_track_faces.md) for pipeline details.

---

#### `POST /jobs/detect_track_async`
Run detect/track asynchronously (polling).

**Request:** Same as `/jobs/detect_track`

**Response:**
```json
{
  "job_id": "job-abc123",
  "state": "running",
  "ep_id": "rhobh-s05e02",
  "started_at": "2025-11-18T12:34:56Z"
}
```

#### `GET /jobs/{job_id}`
Get job status and details.

**Response:**
```json
{
  "job_id": "job-abc123",
  "job_type": "detect_track",
  "ep_id": "rhobh-s05e02",
  "state": "succeeded",
  "started_at": "2025-11-18T12:34:56Z",
  "ended_at": "2025-11-18T12:40:12Z",
  "progress_file": "/path/to/progress.json",
  "command": ["python3", "tools/episode_run.py", "--ep-id", "rhobh-s05e02", ...],
  "requested": {
    "stride": 6,
    "device": "auto",
    "profile": "balanced"
  },
  "summary": {
    "stage": "detect_track",
    "tracks_total": 42,
    "detections_total": 1245
  },
  "track_metrics": {
    "ep_id": "rhobh-s05e02",
    "generated_at": "2025-11-18T12:40:12Z",
    "metrics": {
      "total_detections": 1245,
      "total_tracks": 42,
      "duration_minutes": 42.5,
      "tracks_per_minute": 0.99,
      "short_track_count": 8,
      "short_track_fraction": 0.190,
      "id_switch_rate": 0.024
    },
    "cluster_metrics": {
      "singleton_count": 5,
      "singleton_fraction": 0.238,
      "largest_cluster_size": 8,
      "largest_cluster_fraction": 0.381,
      "total_clusters": 21,
      "total_tracks": 42,
      "outlier_tracks": 3,
      "low_cohesion_identities": 2
    },
    "scene_cuts": {
      "count": 18,
      "detector": "pyscenedetect"
    },
    "crop_stats": {
      "crop_attempts": 2100,
      "crop_errors": {
        "near_uniform_gray": 12,
        "tiny_file": 3
      },
      "blank_crops": 15,
      "blank_fraction": 0.0071
    }
  },
  "error": null,
  "return_code": 0
}
```

**States:** `pending` | `running` | `succeeded` | `failed` | `canceled`

**Metrics Fields** (included when job completes):

**Detect/Track Metrics** (`metrics`):
- `total_detections` (int): Total face detections across all frames
- `total_tracks` (int): Number of unique tracks created
- `duration_minutes` (float): Episode duration in minutes
- `tracks_per_minute` (float): Track generation rate (lower is better, target: 10-30)
- `short_track_count` (int): Tracks below minimum length threshold
- `short_track_fraction` (float): Fraction of short tracks (target: < 0.20)
- `id_switch_rate` (float): Identity fragmentation rate (target: < 0.05)

**Cluster Metrics** (`cluster_metrics`, available after clustering):
- `singleton_count` (int): Number of single-track clusters
- `singleton_fraction` (float): Fraction of singleton clusters (target: < 0.30)
- `largest_cluster_size` (int): Size of largest identity cluster
- `largest_cluster_fraction` (float): Largest cluster as fraction of total (target: < 0.40)
- `total_clusters` (int): Number of identity clusters (typical: 5-15 for TV episode)
- `total_tracks` (int): Total tracks assigned to clusters
- `outlier_tracks` (int): Tracks flagged as outliers
- `low_cohesion_identities` (int): Clusters with low internal similarity

**Scene Cuts** (`scene_cuts`):
- `count` (int): Number of scene boundaries detected
- `detector` (string): Scene detector used

**Crop Stats** (`crop_stats`):
- `crop_attempts` (int): Total crop extraction attempts
- `crop_errors` (object): Error counts by type
- `blank_crops` (int): Near-uniform or corrupt crops
- `blank_fraction` (float): Fraction of blank crops (warning threshold: > 0.10)

See [ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md) for metric thresholds and quality gates.

---

#### `GET /jobs/{job_id}/progress`
Poll job progress (live updates).

**Response:**
```json
{
  "phase": "track",
  "frames_done": 1200,
  "frames_total": 2400,
  "elapsed_sec": 150.0,
  "eta_sec": 150.0,
  "fps_detected": 24.0,
  "analyzed_fps": 8.0,
  "track_metrics": {
    "metrics": {
      "total_detections": 650,
      "total_tracks": 22,
      "duration_minutes": 21.2,
      "tracks_per_minute": 1.04
    }
  }
}
```

**Note:** `track_metrics` is included once metrics are available (typically after job completion). During execution, partial metrics may be included.

**Cleanup Job Progress:**

For `episode_cleanup` jobs, progress includes phase tracking:

```json
{
  "state": "running",
  "phase": "reembed",
  "phase_index": 2,
  "phase_total": 4,
  "phase_progress": 0.5,
  "total_elapsed_seconds": 127.8,
  "track_metrics": {...}
}
```

**Cleanup Progress Fields:**
- `phase` (string): Current cleanup phase (`split_tracks`, `reembed`, `recluster`, `group_clusters`)
- `phase_index` (int): Current phase number (1-indexed)
- `phase_total` (int): Total number of phases to execute
- `phase_progress` (float): Phase completion (0.0 = starting, 1.0 = complete)
- `total_elapsed_seconds` (float): Elapsed time since cleanup started

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

#### `POST /jobs/faces_embed_async`
Run face embedding asynchronously (requires existing tracks).

**Request:**
```json
{
  "ep_id": "rhobh-s05e02",
  "profile": "balanced",
  "device": "auto",
  "save_frames": false,
  "save_crops": false,
  "jpeg_quality": 85,
  "thumb_size": 256
}
```

**Request Parameters:**
- `ep_id` (string, required): Episode identifier (must have tracks.jsonl)
- `profile` (enum, optional): Performance profile (`fast_cpu` alias for `low_power`, `balanced`, `high_accuracy`)
  - Controls quality gating thresholds and sampling strategy
- `device` (enum, optional): Execution device (default: "auto")
- `save_frames` (bool, default: false): Export sampled frames alongside crops
- `save_crops` (bool, default: false): Export crops to local storage + S3
- `jpeg_quality` (int, default: 85): JPEG quality for face crops (1-100)
- `thumb_size` (int, default: 256): Square thumbnail size in pixels (64-512)

**Response:**
```json
{
  "job_id": "job-def456",
  "state": "running",
  "ep_id": "rhobh-s05e02",
  "started_at": "2025-11-18T12:45:00Z"
}
```

See [docs/pipeline/faces_harvest.md](docs/pipeline/faces_harvest.md) for embedding pipeline details.

---

#### `POST /jobs/cluster_async`
Run identity clustering asynchronously (requires faces.jsonl and faces.npy).

**Request:**
```json
{
  "ep_id": "rhobh-s05e02",
  "profile": "balanced",
  "device": "auto",
  "cluster_thresh": 0.58,
  "min_cluster_size": 2,
  "min_identity_sim": 0.45
}
```

**Request Parameters:**
- `ep_id` (string, required): Episode identifier (must have faces.jsonl and faces.npy)
- `profile` (enum, optional): Performance profile (`fast_cpu` alias for `low_power`, `balanced`, `high_accuracy`)
  - Controls clustering threshold and similarity parameters
- `device` (enum, optional): Execution device (default: "auto")
- `cluster_thresh` (float, default: 0.58): Cosine similarity threshold for clustering (0.2-0.99)
- `min_cluster_size` (int, default: 2): Minimum tracks per identity cluster
- `min_identity_sim` (float, default: 0.45): Minimum similarity for track to remain in cluster (0.0-0.99)

**Response:**
```json
{
  "job_id": "job-ghi789",
  "state": "running",
  "ep_id": "rhobh-s05e02",
  "started_at": "2025-11-18T12:50:00Z"
}
```

See [docs/pipeline/cluster_identities.md](docs/pipeline/cluster_identities.md) for clustering algorithm details.

---

#### `POST /jobs/episode_cleanup_async`
Run episode cleanup workflow (outlier removal, re-embedding, re-clustering).

**Request:**
```json
{
  "ep_id": "rhobh-s05e02",
  "profile": "balanced",
  "stride": 4,
  "fps": null,
  "device": "auto",
  "embed_device": "auto",
  "detector": "retinaface",
  "tracker": "bytetrack",
  "max_gap": 30,
  "det_thresh": null,
  "scene_detector": "pyscenedetect",
  "scene_threshold": 27.0,
  "scene_min_len": 12,
  "scene_warmup_dets": 3,
  "cluster_thresh": 0.58,
  "min_cluster_size": 2,
  "min_identity_sim": 0.45,
  "thumb_size": 256,
  "save_frames": false,
  "save_crops": false,
  "jpeg_quality": 85,
  "actions": ["split_tracks", "reembed", "recluster", "group_clusters"],
  "write_back": true
}
```

**Request Parameters:**
- `ep_id` (string, required): Episode identifier
- `profile` (enum, optional): Performance profile (`fast_cpu` alias for `low_power`, `balanced`, `high_accuracy`; applies to all cleanup stages)
- `actions` (array, default: all): Cleanup phases to run
  - `split_tracks`: Split long tracks at scene boundaries
  - `reembed`: Re-extract face embeddings for outlier tracks
  - `recluster`: Re-cluster identities after cleanup
  - `group_clusters`: Merge similar clusters
- `write_back` (bool, default: true): Write cleaned artifacts back to episode directory
- *(Other parameters same as detect_track and cluster endpoints)*

**Response:**
```json
{
  "job_id": "job-xyz789",
  "state": "running",
  "ep_id": "rhobh-s05e02",
  "started_at": "2025-11-18T13:00:00Z"
}
```

See [docs/pipeline/episode_cleanup.md](docs/pipeline/episode_cleanup.md) for cleanup workflow details.

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
  "version": "0.1.0",
  "redis": "ok",
  "storage": "ok",
  "db": "ok",
  "details": {
    "storage": "optional detail on degraded/error state"
  }
}
```

Returns HTTP 200 when Redis, storage, and (if configured) Postgres respond quickly; returns HTTP 503 with a `details` map when any dependency is unavailable.

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

### Configuration & Profiles
- **[CONFIG_GUIDE.md](CONFIG_GUIDE.md)** — Quick reference for all config parameters and performance profiles
- **[docs/reference/config/pipeline_configs.md](docs/reference/config/pipeline_configs.md)** — Complete config parameter reference
- **[ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md)** — Metric thresholds and quality gates

### Pipeline Documentation
- **[docs/pipeline/overview.md](docs/pipeline/overview.md)** — End-to-end pipeline architecture
- **[docs/pipeline/detect_track_faces.md](docs/pipeline/detect_track_faces.md)** — Detection & tracking stage
- **[docs/pipeline/faces_harvest.md](docs/pipeline/faces_harvest.md)** — Face sampling & embedding stage
- **[docs/pipeline/cluster_identities.md](docs/pipeline/cluster_identities.md)** — Identity clustering stage
- **[docs/pipeline/episode_cleanup.md](docs/pipeline/episode_cleanup.md)** — Cleanup workflow

### Schemas & Artifacts
- **[docs/reference/artifacts_faces_tracks_identities.md](docs/reference/artifacts_faces_tracks_identities.md)** — Artifact schemas (detections.jsonl, tracks.jsonl, identities.json, etc.)
- **[docs/reference/schemas/identities_v1_spec.md](docs/reference/schemas/identities_v1_spec.md)** — Identity cluster schema specification
- **[docs/reference/metrics/derived_metrics.md](docs/reference/metrics/derived_metrics.md)** — Metrics calculation and guardrails

### Operations & Troubleshooting
- **[docs/ops/troubleshooting_faces_pipeline.md](docs/ops/troubleshooting_faces_pipeline.md)** — Common API errors and debugging
- **[docs/ops/performance_tuning_faces_pipeline.md](docs/ops/performance_tuning_faces_pipeline.md)** — Speed vs accuracy tuning
- **[docs/ops/monitoring_logging_faces_pipeline.md](docs/ops/monitoring_logging_faces_pipeline.md)** — Logging and monitoring

---

**Maintained by:** Screenalytics Engineering
**Last Updated:** 2025-11-18
**Version:** 2.1
