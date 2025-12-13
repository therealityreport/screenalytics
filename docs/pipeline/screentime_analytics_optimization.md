# Screentime Analytics: Meaning, Optimization, and Model Integration

This document is the canonical reference for how Screenalytics computes “screentime” and how to tune/debug it.

---

## 1) What “Screentime” Means In This System

Screenalytics produces **per-cast, per-episode** visibility and speaking metrics by aggregating pipeline artifacts:

- **Visual presence (faces/tracks):** “How long was this cast member visible?”
- **Speaking time (audio):** “How long was this cast member speaking?” (optional; requires audio artifacts)
- **Body metrics (body tracking):** “How much time was visible by body when the face is missing?” (optional; requires body tracking artifacts)

### 1.1 Inputs

**Required (vision + identity assignment)**
- `data/manifests/{ep_id}/faces.jsonl`
- `data/manifests/{ep_id}/tracks.jsonl`
- `data/manifests/{ep_id}/identities.json`
- `data/shows/{SHOW_ID}/people.json` (identity → person → cast_id mapping)

**Optional (audio speaking time)**
- `data/manifests/{ep_id}/episode_transcript.jsonl`
- `data/manifests/{ep_id}/audio_voice_mapping.json`

**Optional (body tracking metrics)**
- `data/manifests/{ep_id}/body_tracking/screentime_comparison.json`

### 1.2 Outputs and Schema

Outputs are written to:
- `data/analytics/{ep_id}/screentime.json`
- `data/analytics/{ep_id}/screentime.csv`

Top-level keys:
- `episode_id`, `generated_at`, `metrics`, `metadata` (+ `diagnostics`, `timeline`)

Per-cast metrics (current schema; legacy fields kept for backward compatibility):
- `face_visible_seconds` (alias: `visual_s`)
- `speaking_s`
- `body_visible_seconds` (optional)
- `body_only_seconds` (optional)
- `gap_bridged_seconds` (optional)
- `metadata.body_tracking_enabled`, `metadata.body_metrics_available`

Authoritative field definitions live in `docs/reference/data_schema.md`.

---

## 2) Gap Handling / Smoothing

### 2.1 Face/Track Gap Bridging

The screentime analyzer merges consecutive face/track intervals when the gap between them is below `gap_tolerance_s`.

Key controls (see `config/pipeline/screen_time_v2.yaml` presets):
- `gap_tolerance_s`: maximum “silent” gap to merge (seconds)
- `screen_time_mode`:
  - `faces`: intervals inferred from face timestamps (more conservative for sparse detections)
  - `tracks`: intervals inferred from track spans (more stable for cut-heavy footage)
- `edge_padding_s`: expands each merged interval slightly to better match perceived on-screen presence
- `track_coverage_min`: when `screen_time_mode=tracks`, drops tracks whose detections cover too little of the span

### 2.2 Audio Overlap Policy

Transcript segments can include overlap (multiple speakers active). Speaking time supports an overlap policy:
- `shared`: split segment duration across active speakers
- `full`: credit full duration to each active speaker
- `primary_only`: only credit the primary speaker

### 2.3 `gap_bridged_seconds` (Body Tracking)

When body tracking is enabled, `gap_bridged_seconds` is sourced from:
- `body_tracking/screentime_comparison.json` (`delta.duration_gain`)

It represents additional visible time gained when bodies are used to bridge face-missing gaps.

---

## 3) New Models and Where They Plug In

### 3.1 Face Detection and Tracking
- **Detection:** RetinaFace (`config/pipeline/detection.yaml`)
- **Tracking:** ByteTrack (`config/pipeline/tracking.yaml`)

These artifacts drive face visibility aggregation and identity formation.

### 3.2 Face Alignment + Embeddings
- **Alignment (optional):** `config/pipeline/face_alignment.yaml` (feature sandbox: `FEATURES/face_alignment/`)
- **Embeddings:** ArcFace 512-d (`config/pipeline/embedding.yaml`)
  - Optional TensorRT backend configured via `config/pipeline/arcface_tensorrt.yaml`

Alignment quality can be used as a gating signal before embedding (`embedding.yaml: face_alignment.*`).

### 3.3 Body Tracking (Optional)
Body tracking can supplement face-only visibility:
- Config: `config/pipeline/body_detection.yaml`, `config/pipeline/track_fusion.yaml`
- Feature sandbox entrypoint: `python -m FEATURES.body_tracking` (see `FEATURES/body-tracking/`)

Outputs used by screentime are summarized in `body_tracking/screentime_comparison.json`.

### 3.4 Audio Pipeline (Optional)
Speaking time requires:
- `episode_transcript.jsonl`: speaker-labeled transcript rows
- `audio_voice_mapping.json`: voice_cluster → voice bank match results

Config: `config/pipeline/audio.yaml` (NeMo MSDD diarization + OpenAI Whisper ASR).

---

## 4) Performance Tuning

High-impact levers (prefer these before changing algorithms):
- **Profiles/stride/FPS:** `config/pipeline/performance_profiles.yaml`
- **Detection cost vs recall:** `config/pipeline/detection.yaml` (`min_size`, `wide_shot_*`, `confidence_th`)
- **Tracking fragmentation vs ID switches:** `config/pipeline/tracking.yaml` (`match_thresh`, `track_buffer`, `new_track_thresh`)
- **Embedding throughput:** `config/pipeline/embedding.yaml` (`tensorrt.batch_size`, engine cache under `storage.cache_dir`)
- **Embedding volume:** `config/pipeline/faces_embed_sampling.yaml` (`limits.*`, `sample_every_n_frames`)

For “screentime feels too choppy”, tune `config/pipeline/screen_time_v2.yaml` first (`gap_tolerance_s`, `screen_time_mode=tracks`, `edge_padding_s`).

---

## 5) Troubleshooting Checklist

Symptoms → likely causes → fixes:

- **`speaking_s` always 0**
  - Missing `episode_transcript.jsonl` or `audio_voice_mapping.json`
  - Voice bank entries not labeled (no cast IDs), or mapping can’t resolve to cast IDs
  - Fix: run audio pipeline; ensure voice bank for the show exists under `data/voice_bank/{show}.json`

- **No cast metrics despite tracks/faces existing**
  - `identities.json` missing or no `people.json` mapping for the episode’s clusters
  - Fix: run clustering + cast assignment workflow; verify `people.json` contains `cluster_ids` for the episode

- **Body metrics missing in UI**
  - Body tracking disabled or comparison artifact not present
  - Fix: enable `body_tracking.enabled` and ensure `body_tracking/screentime_comparison.json` is generated

- **Wildly inflated visibility**
  - `screen_time_mode=tracks` with low `track_coverage_min`, or overly permissive detection thresholds
  - Fix: increase `track_coverage_min`, reduce `edge_padding_s`, or tighten detection/tracking settings

---

## 6) Acceptance Criteria (What CI/Smoke Validates)

The screentime validator (CI + `tools/smoke/smoke_run.py`) enforces:
- Top-level keys: `episode_id`, `generated_at`, `metrics`, `metadata`
- Each metric includes: `name`, (`face_visible_seconds` or `visual_s`), `confidence`
- Seconds are **non-negative** (`face_visible_seconds`, `speaking_s`, and body fields when present)
- `confidence` is within `[0, 1]`
- If `metadata.body_tracking_enabled && metadata.body_metrics_available`, then `body_visible_seconds` and `body_only_seconds` must be present and non-null

See `tests/api/test_screentime_schema.py` for the contract tests.

