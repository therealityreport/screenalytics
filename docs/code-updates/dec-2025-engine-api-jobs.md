# Engine, API & Jobs Refactor

**Date:** December 2025
**Branch:** `feature/2025-12-engine-api-jobs`
**Author:** Claude (AI-assisted)

## Overview

This update makes the episode processing engine the single orchestration path and adds a job API layer for asynchronous episode runs.

## Key Components

### 1. Episode Engine (`py_screenalytics/pipeline/`)

The engine provides a UI-agnostic API for running the screen-time pipeline.

#### Module Structure

```
py_screenalytics/pipeline/
├── __init__.py           # Public API exports
├── constants.py          # Artifact paths, defaults, thresholds
├── episode_engine.py     # Main orchestration logic
└── stages.py             # Stage implementations wrapper
```

#### EpisodeRunConfig

Configuration dataclass with all pipeline parameters:

```python
from py_screenalytics.pipeline import EpisodeRunConfig

config = EpisodeRunConfig(
    # Device settings
    device="auto",              # auto, cpu, cuda, coreml
    embed_device=None,          # Optional separate device for embedding

    # Detection settings
    detector="retinaface",
    det_thresh=0.65,

    # Tracking settings
    tracker="bytetrack",
    stride=1,                   # Frame stride
    track_buffer=15,

    # Scene detection
    scene_detector="pyscenedetect",
    scene_threshold=27.0,

    # Embedding settings
    max_samples_per_track=16,

    # Clustering settings
    cluster_thresh=0.75,
    min_identity_sim=0.50,

    # Export settings
    save_crops=True,
    save_frames=False,
    thumb_size=256,
    jpeg_quality=85,

    # Dev-mode options
    reuse_detections=False,     # Skip detect_track if artifacts exist
    reuse_embeddings=False,     # Skip faces_embed if artifacts exist
    force_recluster=True,       # Always rerun clustering
)
```

#### EpisodeRunResult

Result dataclass with aggregate metrics:

```python
result = run_episode("rhobh-s05e14", "/path/to/video.mp4", config)

# Access results
print(f"Success: {result.success}")
print(f"Tracks: {result.tracks_count}")
print(f"Faces: {result.faces_count}")
print(f"Identities: {result.identities_count}")
print(f"Runtime: {result.runtime_sec:.1f}s")

# Per-stage details
for stage in result.stages:
    print(f"  {stage.stage}: {stage.runtime_sec:.1f}s, success={stage.success}")

# JSON export
json_data = result.to_dict()
json_str = result.to_json()
```

#### Main Functions

```python
from py_screenalytics.pipeline import (
    run_episode,        # Run full pipeline
    run_stage,          # Run single stage
    run_detect_track,   # Direct stage call
    run_faces_embed,    # Direct stage call
    run_cluster,        # Direct stage call
)

# Full pipeline
result = run_episode("rhobh-s05e14", "/path/to/video.mp4", config)

# Single stage
from py_screenalytics.pipeline import PipelineStage
result = run_stage("rhobh-s05e14", "/path/to/video.mp4", config, PipelineStage.DETECT_TRACK)
```

### 2. Canonical Artifact Paths (`constants.py`)

All artifact locations are defined in `py_screenalytics/pipeline/constants.py`:

```python
from py_screenalytics.pipeline import get_artifact_path, ArtifactKind

# Get path for any artifact
path = get_artifact_path("rhobh-s05e14", ArtifactKind.FACES)
# -> data/manifests/rhobh-s05e14/faces.jsonl

# Or use string keys
path = get_artifact_path("rhobh-s05e14", "tracks")
# -> data/manifests/rhobh-s05e14/tracks.jsonl
```

**Artifact Types:**

| Kind | Path Pattern |
|------|--------------|
| `video` | `videos/{ep_id}/episode.mp4` |
| `detections` | `manifests/{ep_id}/detections.jsonl` |
| `tracks` | `manifests/{ep_id}/tracks.jsonl` |
| `faces` | `manifests/{ep_id}/faces.jsonl` |
| `identities` | `manifests/{ep_id}/identities.json` |
| `track_reps` | `manifests/{ep_id}/track_reps.jsonl` |
| `frames_root` | `frames/{ep_id}` |
| `crops_dir` | `frames/{ep_id}/crops` |
| `thumbs_dir` | `frames/{ep_id}/thumbs` |
| `faces_embeddings` | `manifests/{ep_id}/faces_embeddings.npy` |

### 3. Job API Layer

#### POST /jobs/episode-run

Start a full episode processing pipeline job.

**Request:**
```json
{
  "ep_id": "rhobh-s05e14",
  "device": "auto",
  "stride": 1,
  "det_thresh": 0.65,
  "cluster_thresh": 0.75,
  "save_crops": true,
  "save_frames": false,
  "reuse_detections": false,
  "reuse_embeddings": false,
  "profile": "balanced"
}
```

**Response:**
```json
{
  "job_id": "abc123def456",
  "status": "running"
}
```

#### GET /jobs/{job_id}

Poll job status.

**Response (running):**
```json
{
  "job_id": "abc123def456",
  "job_type": "episode_run",
  "ep_id": "rhobh-s05e14",
  "state": "running",
  "started_at": "2025-12-03T15:00:00Z",
  "ended_at": null,
  "summary": null,
  "error": null
}
```

**Response (succeeded):**
```json
{
  "job_id": "abc123def456",
  "job_type": "episode_run",
  "ep_id": "rhobh-s05e14",
  "state": "succeeded",
  "started_at": "2025-12-03T15:00:00Z",
  "ended_at": "2025-12-03T15:45:00Z",
  "summary": {
    "episode_id": "rhobh-s05e14",
    "success": true,
    "tracks_count": 142,
    "faces_count": 1850,
    "identities_count": 12,
    "runtime_sec": 2700.5,
    "stages": [...]
  },
  "error": null
}
```

### 4. Dev-Mode Options

For faster iteration during development:

#### Artifact Reuse

Skip stages if their artifacts already exist:

```python
config = EpisodeRunConfig(
    reuse_detections=True,   # Skip detect_track if detections/tracks exist
    reuse_embeddings=True,   # Skip faces_embed if faces/embeddings exist
    force_recluster=True,    # Always run clustering (for threshold tuning)
)
```

#### Check Existing Artifacts

```python
from py_screenalytics.pipeline import check_artifacts_exist

# Check what artifacts exist for detect_track stage
artifacts = check_artifacts_exist("rhobh-s05e14", "detect_track")
# {"detections": True, "tracks": True}

# Check faces_embed artifacts
artifacts = check_artifacts_exist("rhobh-s05e14", "faces_embed")
# {"faces": True, "faces_embeddings": False}
```

### 5. Streamlit Integration

The Streamlit UI can use the job API:

```python
import requests

# Start job
resp = requests.post(f"{API_BASE}/jobs/episode-run", json={
    "ep_id": ep_id,
    "device": "coreml",
    "stride": 4,
    "save_crops": True,
})
job_id = resp.json()["job_id"]

# Store in session state
st.session_state["episode_run_job_id"] = job_id

# Poll status
while True:
    status = requests.get(f"{API_BASE}/jobs/{job_id}").json()
    if status["state"] in ("succeeded", "failed"):
        break
    time.sleep(5)

# On success, load artifacts
if status["state"] == "succeeded":
    summary = status["summary"]
    st.success(f"Found {summary['identities_count']} identities!")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Entry Points                             │
├─────────────────────────────────────────────────────────────┤
│  CLI: tools/episode_run.py (thin wrapper)                   │
│  API: POST /jobs/episode-run                                │
│  UI:  Streamlit → Job API polling                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              py_screenalytics.pipeline                       │
├─────────────────────────────────────────────────────────────┤
│  run_episode(ep_id, video_path, config) -> EpisodeRunResult │
│  run_stage(ep_id, video_path, config, stage) -> StageResult │
├─────────────────────────────────────────────────────────────┤
│  EpisodeRunConfig - All pipeline parameters                  │
│  EpisodeRunResult - Aggregate results + per-stage details   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage Implementations (stages.py)              │
├─────────────────────────────────────────────────────────────┤
│  run_detect_track() - RetinaFace + ByteTrack               │
│  run_faces_embed()  - ArcFace embeddings                   │
│  run_cluster()      - Agglomerative clustering             │
└─────────────────────────────────────────────────────────────┘
```

## Files Changed

| File | Change |
|------|--------|
| `py_screenalytics/pipeline/__init__.py` | Updated exports |
| `py_screenalytics/pipeline/episode_engine.py` | Refactored to use stages module |
| `py_screenalytics/pipeline/stages.py` | NEW - Stage wrapper functions |
| `apps/api/services/jobs.py` | Added `start_episode_run_job()` |
| `apps/api/routers/jobs.py` | Added `POST /jobs/episode-run` |

## Usage Examples

### Python (Direct Engine)

```python
from py_screenalytics.pipeline import run_episode, EpisodeRunConfig

config = EpisodeRunConfig(
    device="coreml",
    stride=4,
    save_crops=True,
    cluster_thresh=0.72,
)

result = run_episode("rhobh-s05e14", "/path/to/video.mp4", config)

if result.success:
    print(f"Found {result.identities_count} identities in {result.runtime_sec:.1f}s")
else:
    print(f"Failed: {result.error}")
```

### API (Job-based)

```bash
# Start job
curl -X POST http://localhost:8000/jobs/episode-run \
  -H "Content-Type: application/json" \
  -d '{"ep_id": "rhobh-s05e14", "device": "auto", "stride": 4}'

# Poll status
curl http://localhost:8000/jobs/{job_id}
```

### Dev Mode (Fast Iteration)

```python
# Skip detection if it's already done, just re-cluster with new threshold
config = EpisodeRunConfig(
    reuse_detections=True,
    reuse_embeddings=True,
    cluster_thresh=0.70,  # Try different threshold
)

result = run_episode("rhobh-s05e14", "/path/to/video.mp4", config)
# Only clustering runs, detection/embedding skipped
```

## Future Work

- [ ] CLI thin wrapper over engine
- [ ] Golden episode regression tests
- [ ] Model caching across calls
- [ ] Streamlit full integration with job polling
