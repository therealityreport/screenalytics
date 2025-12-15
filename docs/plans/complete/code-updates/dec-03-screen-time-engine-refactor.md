# Screen Time Engine Refactor

**Date:** December 3, 2025
**Branch:** `feature/screen-time-engine-refactor`
**Author:** Claude (AI-assisted)

## Summary

This update introduces a UI-agnostic episode processing engine that centralizes the screen-time pipeline logic. The engine provides a clean, single entry point (`run_episode()`) that can be invoked from CLI tools, API endpoints, or Streamlit UIs without any UI-specific dependencies.

## Key Changes

### New Module: `py_screenalytics/pipeline/`

A new pipeline module has been created with the following structure:

```
py_screenalytics/pipeline/
├── __init__.py           # Public API exports
├── constants.py          # Artifact paths and defaults
└── episode_engine.py     # Main engine logic
```

### New Data Structures

#### `EpisodeRunConfig`

A dataclass capturing all pipeline parameters with sensible defaults:

```python
from py_screenalytics.pipeline import EpisodeRunConfig

config = EpisodeRunConfig(
    device="auto",           # auto, cpu, cuda, coreml
    stride=1,                # Frame stride for detection
    det_thresh=0.65,         # Detection threshold
    cluster_thresh=0.75,     # Clustering similarity threshold
    save_crops=True,         # Export face crops
    # ... many more options with defaults
)
```

**Categories of configuration:**
- **Device/execution:** `device`, `embed_device`, `allow_cpu_fallback`
- **Detection:** `detector`, `det_thresh`, `coreml_det_size`
- **Tracking:** `tracker`, `stride`, `track_buffer`, `match_thresh`, etc.
- **Scene detection:** `scene_detector`, `scene_threshold`, `scene_min_len`
- **Embedding:** `max_samples_per_track`, `min_samples_per_track`
- **Clustering:** `cluster_thresh`, `min_identity_sim`, `min_cluster_size`
- **Export:** `save_frames`, `save_crops`, `thumb_size`, `jpeg_quality`
- **Output:** `data_root`, `progress_callback`, `progress_file`

#### `EpisodeRunResult`

A comprehensive result structure with per-stage details:

```python
result = run_episode("rhobh-s05e14", "/path/to/video.mp4", config)

# Aggregate metrics
print(f"Tracks: {result.tracks_count}")
print(f"Faces: {result.faces_count}")
print(f"Identities: {result.identities_count}")
print(f"Runtime: {result.runtime_sec:.1f}s")

# Per-stage results
for stage in result.stages:
    print(f"  {stage.stage}: {stage.runtime_sec:.1f}s, success={stage.success}")

# JSON export
print(result.to_json())
```

#### `StageResult`

Per-stage result structure:
- `stage`: Stage name (detect_track, faces_embed, cluster)
- `success`: Boolean success status
- `runtime_sec`: Stage runtime
- `frames_processed`, `detections_count`, `tracks_count`, etc.
- `artifacts`: Dictionary of produced artifact paths
- `error`: Error message if failed

### Artifact Contract

The `constants.py` module defines canonical artifact paths:

```python
from py_screenalytics.pipeline.constants import get_artifact_path, ArtifactKind

# Get path for any artifact
detections = get_artifact_path("rhobh-s05e14", ArtifactKind.DETECTIONS)
# -> data/manifests/rhobh-s05e14/detections.jsonl

faces = get_artifact_path("rhobh-s05e14", "faces")  # String also works
# -> data/manifests/rhobh-s05e14/faces.jsonl
```

**Artifact types:**
| Kind | Path Pattern |
|------|--------------|
| `video` | `videos/{ep_id}/episode.mp4` |
| `detections` | `manifests/{ep_id}/detections.jsonl` |
| `tracks` | `manifests/{ep_id}/tracks.jsonl` |
| `faces` | `manifests/{ep_id}/faces.jsonl` |
| `identities` | `manifests/{ep_id}/identities.json` |
| `track_reps` | `manifests/{ep_id}/track_reps.jsonl` |
| `frames_root` | `frames/{ep_id}` |
| `faces_embeddings` | `manifests/{ep_id}/faces_embeddings.npy` |

### Main Entry Point

```python
from py_screenalytics.pipeline import run_episode, EpisodeRunConfig

# Minimal usage - uses all defaults
result = run_episode("rhobh-s05e14", "/path/to/video.mp4")

# With custom config
config = EpisodeRunConfig(
    device="coreml",
    stride=2,
    save_crops=True,
    cluster_thresh=0.7,
)
result = run_episode("rhobh-s05e14", "/path/to/video.mp4", config)

if result.success:
    print(f"Found {result.identities_count} identities in {result.runtime_sec:.1f}s")
else:
    print(f"Pipeline failed: {result.error}")
```

### Running Individual Stages

```python
from py_screenalytics.pipeline import run_stage, PipelineStage

# Run just detection/tracking
result = run_stage("rhobh-s05e14", "/path/to/video.mp4", config, PipelineStage.DETECT_TRACK)

# Run just embedding
result = run_stage("rhobh-s05e14", "/path/to/video.mp4", config, PipelineStage.FACES_EMBED)

# Run just clustering
result = run_stage("rhobh-s05e14", "/path/to/video.mp4", config, PipelineStage.CLUSTER)
```

### Progress Callbacks

The engine supports progress callbacks for real-time updates:

```python
def on_progress(event):
    if event["type"] == "stage_start":
        print(f"Starting {event['stage']}...")
    elif event["type"] == "stage_complete":
        print(f"Completed {event['stage']} in {event['runtime_sec']:.1f}s")

config = EpisodeRunConfig(progress_callback=on_progress)
result = run_episode("rhobh-s05e14", "/path/to/video.mp4", config)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Entry Points                             │
├─────────────────────────────────────────────────────────────┤
│  CLI: tools/episode_run.py                                  │
│  API: apps/api/routers/jobs.py                              │
│  UI:  apps/workspace-ui/pages/2_Episode_Detail.py           │
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
│  StageResult - Single stage result                          │
├─────────────────────────────────────────────────────────────┤
│  constants.py - Artifact paths, thresholds, model names     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage Implementations                           │
│  (currently in tools/episode_run.py, to be refactored)      │
├─────────────────────────────────────────────────────────────┤
│  _run_detect_track_stage() - RetinaFace + ByteTrack         │
│  _run_faces_embed_stage()  - ArcFace embeddings             │
│  _run_cluster_stage()      - Agglomerative clustering       │
└─────────────────────────────────────────────────────────────┘
```

## Migration Path

### Phase 1 (This PR): Foundation ✅
- [x] Create `py_screenalytics/pipeline/` module
- [x] Define `EpisodeRunConfig` with all parameters
- [x] Define `EpisodeRunResult` and `StageResult`
- [x] Define canonical artifact paths in `constants.py`
- [x] Create `run_episode()` entry point (delegates to existing CLI)

### Phase 2: CLI Integration (Optional)
- [ ] Refactor `tools/episode_run.py` to use engine
- [ ] CLI becomes thin wrapper: parse args → build config → call engine

### Phase 3: API Integration (Optional)
- [ ] Update `apps/api/routers/jobs.py` to use engine
- [ ] API calls `run_stage()` directly instead of subprocess

### Phase 4: Performance Optimization (Future)
- [ ] Model caching across stages (load once, reuse)
- [ ] Artifact caching (skip stages if outputs exist)
- [ ] Parallel stage execution where possible

## Testing

Basic import and instantiation test:

```bash
python3 -c "
from py_screenalytics.pipeline import (
    run_episode, run_stage,
    EpisodeRunConfig, EpisodeRunResult,
    PipelineStage, get_artifact_path,
)
config = EpisodeRunConfig(device='cpu', stride=2)
print(f'Config: device={config.device}, stride={config.stride}')
path = get_artifact_path('test-s01e01', 'faces')
print(f'Faces path: {path}')
print('All imports passed!')
"
```

## Files Changed

| File | Change |
|------|--------|
| `py_screenalytics/pipeline/__init__.py` | NEW - Module exports |
| `py_screenalytics/pipeline/constants.py` | NEW - Artifact paths and defaults |
| `py_screenalytics/pipeline/episode_engine.py` | NEW - Engine logic |

## Backward Compatibility

- **100% backward compatible** - No changes to existing code
- Existing CLI (`tools/episode_run.py`) unchanged
- Existing API endpoints unchanged
- New engine is additive only

## Usage Examples

### From CLI (future integration)

```bash
# Current (unchanged)
python tools/episode_run.py --ep-id rhobh-s05e14 --video /path/to/video.mp4

# Future (after CLI integration)
python -m py_screenalytics.pipeline --ep-id rhobh-s05e14 --video /path/to/video.mp4
```

### From Python

```python
from py_screenalytics.pipeline import run_episode, EpisodeRunConfig

config = EpisodeRunConfig(
    device="coreml",
    save_crops=True,
    cluster_thresh=0.72,
)

result = run_episode("rhobh-s05e14", "/path/to/video.mp4", config)

# Access results
print(f"Success: {result.success}")
print(f"Identities: {result.identities_count}")
print(f"Runtime: {result.runtime_sec:.1f}s")

# Per-stage details
for stage in result.stages:
    print(f"  {stage.stage}: {stage.runtime_sec:.1f}s")
```

### From API (future integration)

```python
# In apps/api/routers/jobs.py
from py_screenalytics.pipeline import run_stage, EpisodeRunConfig, PipelineStage

@router.post("/jobs/detect_track")
def start_detect_track(request: DetectTrackRequest):
    config = EpisodeRunConfig(
        device=request.device or "auto",
        stride=request.stride or 1,
        save_crops=request.save_crops or False,
    )
    result = run_stage(request.ep_id, request.video_path, config, PipelineStage.DETECT_TRACK)
    return result.to_dict()
```
