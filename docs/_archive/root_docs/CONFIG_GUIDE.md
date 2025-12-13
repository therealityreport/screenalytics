# Configuration Guide — Screenalytics

**Quick Reference** | [Full Documentation →](docs/reference/config/pipeline_configs.md)

---

## Overview

All pipeline behavior is **config-driven** (no hardcoded thresholds). Configs live in `config/pipeline/*.yaml`.

---

## Key Configuration Files

| File | Purpose | Key Parameters |
|------|---------|----------------|
| **detection.yaml** | RetinaFace detection | `min_size`, `confidence_th`, `iou_th`, `nms_mode` |
| **tracking.yaml** | ByteTrack association | `track_thresh`, `match_thresh`, `track_buffer`, `gate_enabled` |
| **faces_embed_sampling.yaml** | Face quality gating | `min_quality`, `max_crops_per_track`, `sampling_mode` |
| **performance_profiles.yaml** | Device-aware profiles | `low_power`, `balanced`, `high_accuracy` |
| **screen_time_v2.yaml** | Screentime aggregation | `quality_min`, `gap_tolerance_s`, `track_coverage_min` |

---

## Performance Profiles

Pre-configured profiles for common hardware (used by the API to choose defaults). `tools/episode_run.py`
does **not** accept `--profile`; pass explicit `--stride`/`--fps` or call the API with `profile`
to apply these presets.

```yaml
# config/pipeline/performance_profiles.yaml

low_power:
  # For fanless devices (MacBook Air, low-power)
  frame_stride: 8
  detection_fps_limit: 8
  min_size: 120
  cpu_threads: 2

balanced:
  # Standard local dev
  frame_stride: 5
  detection_fps_limit: 24
  min_size: 90

high_accuracy:
  # GPU production
  frame_stride: 1
  detection_fps_limit: 30
  min_size: 64

# Compatibility: API maps profile="fast_cpu" to low_power for legacy clients.
```

**Usage:**
```bash
# CLI: pass explicit stride/FPS (mirrors "balanced")
python tools/episode_run.py --ep-id <ep_id> --video <path> --stride 5 --fps 24

# API: include a profile name to apply defaults server-side
POST /jobs/detect_track { "ep_id": "...", "profile": "balanced" }
```

---

## Common Knobs

### Detection
- **`min_size`**: Minimum face size (pixels). Lower = more faces, slower.
  - CPU safe: `120`
  - Balanced: `90`
  - Max recall: `64`

- **`confidence_th`**: Detection threshold. Lower = more faces, more false positives.
  - Safe: `0.8`
  - Balanced: `0.7`
  - Aggressive: `0.6`

### Tracking
- **`track_thresh`**: Min confidence to track. Lower = more tracks, more noise.
  - Safe: `0.75`
  - Balanced: `0.70`
  - Permissive: `0.65`

- **`track_buffer`**: Frames to keep track alive. Higher = fewer ID switches, longer tracks.
  - Short: `60` (~2 sec)
  - Balanced: `90` (~3 sec)
  - Long: `120` (~4 sec)

### Embedding
- **`min_quality`**: Combined quality threshold. Higher = fewer but better faces.
  - Permissive: `0.6`
  - Balanced: `0.7`
  - Strict: `0.8`

- **`max_crops_per_track`**: Limit crops per track.
  - Fast: `20`
  - Balanced: `50`
  - Max recall: `100`

### Clustering
- **`cluster_thresh`**: Cosine similarity threshold. Lower = more clusters.
  - Loose: `0.50`
  - Balanced: `0.58`
  - Strict: `0.70`

---

## Configuration Override Precedence

1. **CLI args** (highest): `--stride 3`
2. **Environment variables**: `TRACK_THRESH=0.75`
3. **Profile-specific**: `performance_profiles.yaml:balanced`
4. **Stage-specific**: `tracking.yaml`, `detection.yaml`
5. **Default values** (lowest): Hardcoded

---

## Documentation

**For complete details, see:**

- **[Pipeline Configs Reference](docs/reference/config/pipeline_configs.md)** — Key-by-key documentation
- **[Performance Tuning](docs/ops/performance_tuning_faces_pipeline.md)** — Speed vs accuracy tuning
- **[Pipeline Overview](docs/pipeline/overview.md)** — How configs affect each stage

---

**Maintained by:** Screenalytics Engineering
