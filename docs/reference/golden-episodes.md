# Golden Episodes

Golden episodes are small, hand-validated test cases used for regression detection in the screen-time pipeline.

## What Are Golden Episodes?

Golden episodes are short video clips that:

1. **Are small enough to run reasonably fast** - typically under 5 minutes
2. **Have visually verified "reasonable" Screen Time behavior** - the metrics produced match human expectations
3. **Serve as regression tests** - when pipeline changes cause metrics to drift outside expected ranges, it signals a problem

## Why Golden Episodes?

Changes to the screen-time pipeline can inadvertently affect:

- Face detection accuracy
- Track formation and maintenance
- Identity clustering quality
- Screen-time metric computation

Golden episodes catch these regressions by comparing actual metrics against known-good baselines.

## Current Golden Episodes

### Golden Episode #1: rhobh-s05e17

| Property | Value |
|----------|-------|
| Episode ID | `rhobh-s05e17` |
| Description | Short RHOBH clip (1.7 minutes) |
| Duration | ~103 seconds |
| Video Path | `data/videos/rhobh-s05e17/episode.mp4` |

**Expected Metrics:**

| Metric | Range | Baseline |
|--------|-------|----------|
| tracks_per_minute | 42.0 - 72.0 | 56.9 |
| short_track_fraction | 0.24 - 0.54 | 0.39 |
| id_switch_rate | 0.0 - 0.07 | 0.02 |
| identities_count | 37 - 57 | 47 |
| faces_count | 522 - 722 | 622 |

**Notes:**
- Initial baseline captured 2025-12-04 from existing artifacts
- Ranges are intentionally wide for initial setup
- Ranges should be tightened after manual validation

## Running Golden Episode Tests

### Using pytest

```bash
# Run all golden episode tests
pytest tests/test_golden_screen_time.py -v

# Run specific test
pytest tests/test_golden_screen_time.py::test_golden_episode[rhobh-s05e17] -v
```

### Using the standalone script

```bash
# Check all golden episodes
python -m tests.test_golden_screen_time

# Check specific episode
python -m tests.test_golden_screen_time rhobh-s05e17

# List available golden episodes
python -m tests.test_golden_screen_time --list

# Run pipeline before checking (slower but more thorough)
python -m tests.test_golden_screen_time --run-pipeline

# Force full pipeline run (no artifact reuse)
python -m tests.test_golden_screen_time --run-pipeline --no-reuse
```

## When to Run Golden Episode Tests

Run golden episode tests after:

1. **Modifying tracking logic** - ByteTrack parameters, appearance gating
2. **Changing detection code** - RetinaFace thresholds, NMS settings
3. **Updating clustering** - cluster thresholds, distance metrics
4. **Altering metric computation** - screen-time calculations, track metrics

## Adding a New Golden Episode

### 1. Select a suitable clip

Choose a clip that is:
- Short (under 5 minutes)
- Representative of typical content
- Has clear, identifiable faces
- Produces "reasonable" pipeline results

### 2. Process the clip

```python
from py_screenalytics.pipeline import run_episode, EpisodeRunConfig

config = EpisodeRunConfig(
    device="coreml",  # or "auto", "cuda"
    stride=6,
    det_thresh=0.65,
    cluster_thresh=0.70,
    save_crops=True,
)

result = run_episode("new-episode-id", "/path/to/video.mp4", config)
print(f"Success: {result.success}")
print(f"Tracks: {result.tracks_count}")
print(f"Identities: {result.identities_count}")
```

### 3. Compute baseline metrics

```bash
python3 << 'EOF'
import json
from pathlib import Path

ep_id = "new-episode-id"
data_root = Path("data")

# Load artifacts and compute metrics
# (see tests/test_golden_screen_time.py for metric computation code)
EOF
```

### 4. Add to config

Edit `configs/golden_episodes.py`:

```python
NEW_EPISODE = GoldenEpisodeConfig(
    episode_id="new-episode-id",
    description="Description of the clip",
    video_path="videos/new-episode-id/episode.mp4",
    expected_metrics=ExpectedMetrics(
        tracks_per_minute=(min, max),
        short_track_fraction=(min, max),
        id_switch_rate=(min, max),
        identities_count=(min, max),
        faces_count=(min, max),
    ),
    baseline_config={
        "stride": 6,
        "det_thresh": 0.65,
        "cluster_thresh": 0.70,
    },
)

# Add to registry
GOLDEN_EPISODES["new-episode-id"] = NEW_EPISODE
```

### 5. Verify the test passes

```bash
python -m tests.test_golden_screen_time new-episode-id
```

## Metric Definitions

| Metric | Formula | Meaning |
|--------|---------|---------|
| `tracks_per_minute` | total_tracks / duration_minutes | Track creation rate |
| `short_track_fraction` | tracks_with_<5_frames / total_tracks | Fragmentation indicator |
| `id_switch_rate` | id_switches / total_tracks | Identity consistency |
| `identities_count` | Number of identity clusters | Grouping quality |
| `faces_count` | Total face detections across all tracks | Detection coverage |

## Troubleshooting

### Test fails with "Video not found"

Ensure the video file exists at the expected path:
```bash
ls -la data/videos/rhobh-s05e17/episode.mp4
```

### Test fails with "Missing artifacts"

The test requires processed artifacts. Run the pipeline first:
```bash
python -m tests.test_golden_screen_time --run-pipeline rhobh-s05e17
```

### Metrics drift outside range

If metrics are consistently outside the expected range after a known-good change:

1. Verify the change is intentional and improves results
2. Update the expected ranges in `configs/golden_episodes.py`
3. Document the change in the config's `notes` field

## Files

| File | Purpose |
|------|---------|
| `configs/golden_episodes.py` | Golden episode configuration and expected metrics |
| `tests/test_golden_screen_time.py` | Regression test script |
| `docs/reference/golden-episodes.md` | This documentation |
