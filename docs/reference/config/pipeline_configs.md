# Pipeline Configuration Reference

**Last Updated:** 2025-12-13

This document describes the **active** pipeline YAML files under `config/pipeline/` and the knobs that matter most in practice. The YAML files are the source of truth; treat this doc as an index + tuning guide to reduce doc drift.

---

## 1) Quick Map (`config/pipeline/*.yaml`)

| File | Purpose | High-impact knobs |
|------|---------|-------------------|
| `detection.yaml` | Face detection (RetinaFace) | `min_size`, `confidence_th`, `wide_shot_*` |
| `tracking.yaml` | Face tracking (ByteTrack) | `match_thresh`, `track_buffer`, `new_track_thresh` |
| `faces_embed_sampling.yaml` | Face sampling + quality gating for embedding/export | `quality_gating.*`, `limits.*` |
| `embedding.yaml` | Embedding engine selection + TRT cache | `embedding.backend`, `tensorrt.batch_size`, `storage.cache_dir` |
| `face_alignment.yaml` | Face alignment (feature sandbox) | `face_alignment.enabled`, `processing.batch_size` |
| `clustering.yaml` | Identity clustering + singleton merge | `cluster_thresh`, `min_identity_sim`, `singleton_merge.*` |
| `audio.yaml` | Audio pipeline (diarization + ASR) | `diarization.*`, `asr.*`, `voice_clustering.*` |
| `screen_time_v2.yaml` | Screentime presets + smoothing | `gap_tolerance_s`, `screen_time_mode`, `track_coverage_min` |
| `body_detection.yaml` | Body tracking (feature sandbox) | `body_tracking.enabled`, `person_reid.enabled` |
| `track_fusion.yaml` | Face↔body fusion rules | `iou_association.*`, `reid_handoff.*` |
| `performance_profiles.yaml` | System-level speed/thermal presets | `frame_stride`, `detection_fps_limit` |

---

## 2) Detection (`detection.yaml`)

Core keys (see `config/pipeline/detection.yaml` for full context):
```yaml
model_id: retinaface_r50
min_size: 16
confidence_th: 0.50
wide_shot_mode: true
wide_shot_input_size: 960
wide_shot_confidence_th: 0.40
```

Notes:
- `min_size` and `confidence_th` are the main recall/perf tradeoffs.
- `wide_shot_*` controls a higher-resolution pass for distant faces.
- `enable_person_fallback` enables a person/body fallback path for “face-missing” scenes.

---

## 3) Tracking (`tracking.yaml`)

ByteTrack thresholds:
```yaml
track_thresh: 0.55
match_thresh: 0.65
track_buffer: 90
new_track_thresh: 0.60
gate_enabled: true
```

Additional knobs are environment-driven (see comments in `config/pipeline/tracking.yaml`):
- `TRACK_GATE_APPEAR_HARD`, `TRACK_GATE_APPEAR_SOFT`, `TRACK_GATE_APPEAR_STREAK`
- `TRACK_GATE_IOU`, `TRACK_GATE_EMB_EVERY`
- `SCREENALYTICS_GMC_METHOD` (global motion compensation)
- `TRACK_MAX_GAP_SEC` (time-based max gap for linking)

---

## 4) Face Sampling for Embeddings (`faces_embed_sampling.yaml`)

Quality gating (relaxed defaults for reality TV):
```yaml
quality_gating:
  min_quality_score: 1.5
  min_confidence: 0.45
  min_blur_score: 18.0
  min_std: 0.5
  max_yaw_angle: 60.0
  max_pitch_angle: 45.0
```

Volume/sampling limits are profile-aware:
```yaml
limits:
  default:
    max_samples_per_track: 5
    max_faces_per_episode: 10000
  profiles:
    low_power:
      max_faces_per_episode: 6000
```

---

## 5) Embedding Engine (`embedding.yaml`)

Backend selection:
```yaml
embedding:
  backend: pytorch  # tensorrt | pytorch
  tensorrt_config: config/pipeline/arcface_tensorrt.yaml
```

Performance knobs:
- `tensorrt.batch_size`, `tensorrt.max_batch_size`
- TRT engine cache: `storage.cache_dir` (local) and `storage.bucket/prefix` (S3)

Alignment gating before embedding:
- `face_alignment.enabled`, `face_alignment.min_alignment_quality`

---

## 6) Face Alignment (`face_alignment.yaml`)

This is a feature sandbox stage that can produce aligned crops and per-face alignment quality.
Key knobs:
- `face_alignment.enabled`
- `processing.batch_size`, `processing.device`
- `quality.min_face_size`, `quality.min_confidence`

---

## 7) Clustering (`clustering.yaml`)

Core thresholds:
```yaml
cluster_thresh: 0.52
min_cluster_size: 1
min_identity_sim: 0.45
```

Optional singleton merge pass (used when singleton fraction is high):
```yaml
singleton_merge:
  enabled: true
  trigger_singleton_frac: 0.40
  similarity_thresh: 0.55
```

---

## 8) Audio Pipeline (`audio.yaml`)

Audio pipeline stages (see `config/pipeline/audio.yaml` for defaults):
- `separation` (MDX-Extra)
- `enhance` (Resemble)
- `diarization` (**NeMo MSDD**, overlap-aware)
- `asr` (**OpenAI Whisper**)
- `voice_clustering` (often `use_diarization_labels: true` to skip embedding clustering)

Artifacts produced include:
- `audio_diarization.jsonl`, `audio_asr_raw.jsonl`
- `audio_voice_mapping.json`, `episode_transcript.jsonl`

---

## 9) Body Tracking + Fusion (`body_detection.yaml`, `track_fusion.yaml`)

Body tracking is a feature sandbox that can compute body-only visibility and face↔body fusion metrics.
Key toggles:
- `body_tracking.enabled`
- `person_reid.enabled`
- `track_fusion.enabled`

---

## 10) Screentime (`screen_time_v2.yaml`)

Screentime uses named presets (default `preset: bravo_default`) to control smoothing and interval derivation:
```yaml
screen_time_presets:
  bravo_default:
    gap_tolerance_s: 1.2
    screen_time_mode: tracks
    edge_padding_s: 0.2
    track_coverage_min: 0.35
```

Docs: `docs/pipeline/screentime_analytics_optimization.md`

---

## 11) Performance Profiles (`performance_profiles.yaml`)

Profiles are applied by the API/job layer to choose stride/FPS defaults for different hardware.
The CLI expects explicit flags (no `--profile` switch).

Profile names (API-facing):
- `fast_cpu` (legacy alias)
- `low_power`
- `balanced`
- `high_accuracy`

Docs: `docs/ops/performance_tuning_faces_pipeline.md`

---

## 12) Override Precedence

Highest → lowest:
1. CLI args (e.g., `--stride 5 --fps 24`)
2. Environment variables (where supported; see file comments)
3. API profile presets (`performance_profiles.yaml`)
4. Stage YAML config (`config/pipeline/*.yaml`)
5. Code defaults

---

## 13) References

- `docs/pipeline/overview.md`
- `docs/pipeline/audio_pipeline.md`
- `docs/pipeline/screentime_analytics_optimization.md`
- `docs/ops/performance_tuning_faces_pipeline.md`
