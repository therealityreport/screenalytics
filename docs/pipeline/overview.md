# Pipeline Overview — Screenalytics

**Version:** 2.0
**Last Updated:** 2025-11-18

---

## 1. Introduction

The Screenalytics pipeline transforms raw video episodes into structured, per-person screentime analytics through a series of modular stages. Each stage consumes well-defined inputs, produces versioned artifacts, and exposes configurable parameters to balance speed, accuracy, and resource usage.

This document provides a high-level overview of the complete pipeline. For detailed stage-specific documentation, see the individual pipeline docs linked below.

---

## 2. Pipeline Stages

```
┌──────────────────────────────────────────────────────────────────┐
│                   SCREENALYTICS PIPELINE                         │
└──────────────────────────────────────────────────────────────────┘

1. DETECT → 2. TRACK → 3. EMBED → 4. CLUSTER → 5. CLEANUP (optional)
                 ↓          ↓          ↓              ↓
          tracks.jsonl  faces.jsonl  identities.json  cleanup_report.json

                              ↓

6. AUDIO PIPELINE (optional)     7. BODY TRACKING (optional)
        ↓                                  ↓
 episode_transcript.jsonl       body_tracking/screentime_comparison.json
                 \                /
                  ▼              ▼
                 8. SCREENTIME ANALYZE
                           ↓
                screentime.json (+ screentime.csv)
```

### Stage Summary

| Stage | Input | Output | Models/Tools | Purpose |
|-------|-------|--------|--------------|---------|
| **1. Detect** | episode.mp4 | detections.jsonl | RetinaFace | Find face bounding boxes and landmarks per frame |
| **2. Track** | detections.jsonl | tracks.jsonl, track_metrics.json | ByteTrack + AppearanceGate | Associate detections across frames into temporal tracks |
| **3. Embed** | episode.mp4, tracks.jsonl | faces.jsonl, faces.npy, crops/*.jpg | ArcFace ONNX | Extract 512-d embeddings from face crops with quality gating |
| **4. Cluster** | faces.jsonl, faces.npy | identities.json, thumbs/*.jpg | Agglomerative Clustering | Group tracks by identity using track-level embeddings |
| **5. Cleanup** | tracks.jsonl, faces.jsonl, identities.json | Updated artifacts, cleanup_report.json | Detect/Track/Embed/Cluster (re-run) | Split long tracks, re-embed, re-cluster, group identities |
| **6. Audio Pipeline** | episode.mp4 (audio) | episode_transcript.jsonl, audio_voice_mapping.json | NeMo MSDD + OpenAI Whisper | Produce speaker-labeled transcript + voice→cast mapping inputs for speaking time |
| **7. Body Tracking** | episode.mp4 | body_tracking/*, screentime_comparison.json | YOLO + ByteTrack + OSNet Re-ID | Preserve identity through face loss; compute optional body visibility metrics |
| **8. Screentime Analyze** | faces/tracks/identities (+ optional audio/body inputs) | screentime.json, screentime.csv | ScreenTimeAnalyzer | Compute per-cast face/speaking (and optional body) seconds |

---

## 3. Data Flow

### 3.1 Visual Pipeline (Detect → Track → Embed → Cluster)

```
episode.mp4 (input video)
    ├─► [1. DETECT] RetinaFace
    │   └─► detections.jsonl (det_v1 schema)
    │       - ep_id, frame_idx, ts_s, bbox [x1,y1,x2,y2], landmarks [x,y]*5, conf
    │
    ├─► [2. TRACK] ByteTrack + AppearanceGate
    │   └─► tracks.jsonl (track_v1 schema)
    │       - track_id, ep_id, start_s, end_s, frame_span, sample_thumbs, stats
    │   └─► track_metrics.json
    │       - tracks_born, tracks_lost, id_switches, longest_tracks, avg_tracks_per_frame
    │
    ├─► [3. EMBED] ArcFace ONNX
    │   └─► faces.jsonl
    │       - face_id, track_id, frame_idx, bbox, landmarks, embedding, quality_score
    │   └─► faces.npy (Nx512 float32 embeddings)
    │   └─► crops/{track_id}/*.jpg (optional)
    │
    └─► [4. CLUSTER] Agglomerative
        └─► identities.json
            - identity_id, cluster_id, track_ids, canonical_face_path, embedding_stats, labels, locked
        └─► thumbs/{identity_id}/rep.jpg
```

### 3.2 Audio Pipeline (Diarize → Transcribe)

```
episode.mp4 (audio stream)
    ├─► [6. DIARIZE] NeMo MSDD (default)
    │   └─► audio_diarization.jsonl
    │       - start, end, speaker_label, (optional) confidence / overlap fields
    │
    └─► [6. ASR] OpenAI Whisper (default)
        └─► audio_asr_raw.jsonl
            - start, end, text, (optional) words[], confidence
```

### 3.3 Fusion & Aggregation

```
faces.jsonl + tracks.jsonl + identities.json + shows/{SHOW}/people.json
    ├─► (optional) episode_transcript.jsonl + audio_voice_mapping.json
    ├─► (optional) body_tracking/screentime_comparison.json
    └─► [8. SCREENTIME ANALYZE] ScreenTimeAnalyzer
        └─► screentime.json
            - face_visible_seconds (+ legacy visual_s)
            - speaking_s
            - body_visible_seconds / body_only_seconds / gap_bridged_seconds (optional)
        └─► screentime.csv
```

---

## 4. Stage Details

### 4.1 Detect (RetinaFace)
- **CLI:** `python tools/episode_run.py --ep-id <ep_id> --video <path> --stride <N>`
- **API:** `POST /jobs/detect_track` (sync) or `POST /jobs/detect_track_async`
- **Config:** `config/pipeline/detection.yaml`
- **Key Params:**
  - `stride`: Process every Nth frame (higher = faster, lower recall)
  - `min_size`: Minimum face size in pixels (see `config/pipeline/detection.yaml`)
  - `confidence_th`: Detection confidence threshold (see `config/pipeline/detection.yaml`)
  - `device`: `auto`, `cpu`, `cuda`, `mps` (Apple Silicon)
- **Artifacts:**
  - `data/manifests/{ep_id}/detections.jsonl`
  - `data/manifests/{ep_id}/progress.json` (live progress updates)
- **Docs:** [detect_track_faces.md](detect_track_faces.md)

### 4.2 Track (ByteTrack + AppearanceGate)
- **CLI:** Runs automatically after detect (same `episode_run.py` invocation)
- **Config:** `config/pipeline/tracking.yaml`
- **Key Params:**
  - `track_thresh`: Min confidence to continue a track (see `config/pipeline/tracking.yaml`)
  - `match_thresh`: IoU threshold for bbox matching (see `config/pipeline/tracking.yaml`)
  - `track_buffer`: Frames to keep track alive (see `config/pipeline/tracking.yaml`)
  - `new_track_thresh`: Threshold for starting new tracks (see `config/pipeline/tracking.yaml`)
  - Appearance gate: `TRACK_GATE_APPEAR_HARD`, `TRACK_GATE_APPEAR_SOFT`, `TRACK_GATE_APPEAR_STREAK`
- **Artifacts:**
  - `data/manifests/{ep_id}/tracks.jsonl`
  - `data/manifests/{ep_id}/track_metrics.json`
- **Docs:** [detect_track_faces.md](detect_track_faces.md)

### 4.3 Embed (ArcFace)
- **CLI:** `python tools/episode_run.py --ep-id <ep_id> --faces-embed --save-crops`
- **API:** `POST /jobs/faces_embed`
- **Config:** `config/pipeline/faces_embed_sampling.yaml`
- **Key Params:**
  - `min_quality`: Minimum face quality score (confidence + size + blur)
  - `max_crops_per_track`: Limit crops per track to avoid embedding thousands of near-identical faces
  - Sampling strategy: `uniform`, `stratified`, `quality-weighted`
- **Artifacts:**
  - `data/manifests/{ep_id}/faces.jsonl`
  - `data/embeds/{ep_id}/faces.npy`
  - `data/frames/{ep_id}/crops/{track_id}/*.jpg` (optional)
- **Docs:** [faces_harvest.md](faces_harvest.md)

### 4.4 Cluster (Agglomerative)
- **CLI:** `python tools/episode_run.py --ep-id <ep_id> --cluster`
- **API:** `POST /jobs/cluster`
- **Config:** `config/pipeline/recognition.yaml` (TBD) or inline params
- **Key Params:**
  - `cluster_thresh`: Cosine similarity threshold (see `config/pipeline/clustering.yaml`)
  - `min_cluster_size`: Minimum tracks per cluster (see `config/pipeline/clustering.yaml`)
  - Outlier handling: Singletons with `cluster_id = null` or noise label
- **Artifacts:**
  - `data/manifests/{ep_id}/identities.json`
  - `data/frames/{ep_id}/thumbs/{identity_id}/rep.jpg`
- **Docs:** [cluster_identities.md](cluster_identities.md)

### 4.5 Cleanup (Optional)
- **CLI:** `python tools/episode_cleanup.py --ep-id <ep_id> --actions split_tracks reembed recluster group_clusters`
- **API:** `POST /jobs/episode_cleanup_async`
- **Config:** Reuses configs from detect, track, embed, cluster stages
- **Key Params:**
  - `actions`: List of phases to run (`split_tracks`, `reembed`, `recluster`, `group_clusters`)
  - `write_back`: Whether to overwrite original artifacts (default: false)
- **Artifacts:**
  - `data/manifests/{ep_id}/cleanup_report.json` (before/after metrics)
  - Updated `tracks.jsonl`, `faces.jsonl`, `identities.json`, `track_metrics.json`
- **Docs:** [episode_cleanup.md](episode_cleanup.md)

### 4.6 Audio Pipeline (Implemented)
- **API:** `POST /jobs/episode_audio_pipeline`
- **Config:** `config/pipeline/audio.yaml`
- **Models:**
  - MDX-Extra/Demucs for stem separation
  - Resemble Enhance for audio denoising
  - NeMo MSDD for overlap-aware speaker diarization
  - OpenAI Whisper (primary) with optional Gemini cleanup/enrichment
- **Artifacts:**
  - `data/audio/{ep_id}/episode_original.wav`
  - `data/audio/{ep_id}/episode_vocals.wav`
  - `data/audio/{ep_id}/episode_vocals_enhanced.wav`
  - `data/manifests/{ep_id}/audio_diarization.jsonl`
  - `data/manifests/{ep_id}/audio_asr_raw.jsonl`
  - `data/manifests/{ep_id}/audio_voice_clusters.json`
  - `data/manifests/{ep_id}/audio_voice_mapping.json`
  - `data/manifests/{ep_id}/episode_transcript.jsonl`
  - `data/manifests/{ep_id}/episode_transcript.vtt`
  - `data/manifests/{ep_id}/audio_qc.json`
- **Docs:** [audio_pipeline.md](audio_pipeline.md)

### 4.7 Screentime (Implemented)
- **CLI:** `python -m tools.analyze_screen_time --ep-id <ep_id>`
- **API:** `POST /jobs/screen_time/analyze`
- **Config:** `config/pipeline/screen_time_v2.yaml`
- **Artifacts:**
  - `data/analytics/{ep_id}/screentime.json`
  - `data/analytics/{ep_id}/screentime.csv`
- **Notes:**
  - If `episode_transcript.jsonl` + `audio_voice_mapping.json` exist, screentime includes `speaking_s`.
  - If `body_tracking/screentime_comparison.json` exists, screentime includes body metrics and `gap_bridged_seconds`.
- **Docs:** [screentime_analytics_optimization.md](screentime_analytics_optimization.md)

---

## 5. Performance Profiles

To prevent CPU overheating and manage resource usage, the pipeline supports device-aware **performance profiles**.
The API applies these presets to choose stride/FPS defaults; `tools/episode_run.py` expects explicit flags (no
`--profile` switch).

| Profile | Device | Stride | FPS | Batch Size | Exporters | Use Case |
|---------|--------|--------|-----|------------|-----------|----------|
| **low_power** | CPU/CoreML | 8 | ≤ 8 | 1 | Off | Fanless devices, exploratory passes |
| **balanced** | CPU/MPS | 5 | ≤ 24 | 2 | Frames only | Standard local dev |
| **high_accuracy** | GPU (CUDA) | 1 | 30 | 4 | Frames + Crops | Production, full recall |

**Config:** `config/pipeline/performance_profiles.yaml`

**CLI Usage:**
```bash
python tools/episode_run.py --ep-id <ep_id> --video <path> --stride 8 --fps 8 --device auto --coreml-only
```

**API Usage:**
```bash
POST /jobs/detect_track_async
{
  "ep_id": "rhobh-s05e02",
  "profile": "balanced"
}
```

See [docs/ops/performance_tuning_faces_pipeline.md](../ops/performance_tuning_faces_pipeline.md) for detailed tuning guidance.

---

## 6. Artifact Schemas

All artifacts use versioned schemas for backward compatibility and validation.

### 6.1 `detections.jsonl` (det_v1)
```jsonl
{
  "ep_id": "rhobh-s05e02",
  "frame_idx": 42,
  "ts_s": 1.75,
  "bbox": [0.1, 0.2, 0.3, 0.4],
  "landmarks": [0.15, 0.25, 0.25, 0.25, 0.2, 0.3, 0.15, 0.35, 0.25, 0.35],
  "conf": 0.95,
  "model_id": "retinaface_r50_v1",
  "schema_version": "det_v1"
}
```

### 6.2 `tracks.jsonl` (track_v1)
```jsonl
{
  "track_id": "track-00001",
  "ep_id": "rhobh-s05e02",
  "start_s": 1.5,
  "end_s": 12.3,
  "frame_span": [30, 246],
  "sample_thumbs": ["crops/track-00001/frame_0030.jpg", "..."],
  "stats": {"detections": 216, "avg_conf": 0.92},
  "schema_version": "track_v1"
}
```

### 6.3 `faces.jsonl`
```jsonl
{
  "face_id": "face-00001",
  "track_id": "track-00001",
  "frame_idx": 42,
  "bbox": [0.1, 0.2, 0.3, 0.4],
  "landmarks": [0.15, 0.25, 0.25, 0.25, 0.2, 0.3, 0.15, 0.35, 0.25, 0.35],
  "embedding": [0.012, -0.045, ...],  // 512-d unit-norm vector
  "quality_score": 0.87,
  "crop_path": "crops/track-00001/frame_0042.jpg"
}
```

### 6.4 `identities.json`
```json
{
  "identities": [
    {
      "identity_id": "identity-00001",
      "cluster_id": 1,
      "track_ids": ["track-00001", "track-00005", "track-00012"],
      "canonical_face_path": "thumbs/identity-00001/rep.jpg",
      "embedding_stats": {
        "centroid_norm": 1.0,
        "variance": 0.023,
        "num_tracks": 3
      },
      "labels": {"person_id": "lisa-vanderpump", "name": "Lisa Vanderpump"},
      "locked": true
    }
  ]
}
```

See [docs/reference/artifacts_faces_tracks_identities.md](../reference/artifacts_faces_tracks_identities.md) for complete schema reference.

---

## 7. Quality Metrics

The pipeline emits quality metrics at each stage to detect issues early:

### 7.1 Detect/Track Metrics (`track_metrics.json`)
- `tracks_born`: Total tracks created
- `tracks_lost`: Tracks terminated
- `id_switches`: Track ID reassignments (lower is better)
- `longest_tracks`: Top 5 longest track lengths (frames)
- `avg_tracks_per_frame`: Average concurrent tracks
- `tracks_per_minute`: Derived metric (higher may indicate ghost tracks)
- `short_track_fraction`: Fraction of tracks < N frames (configurable threshold)

### 7.2 Embedding Metrics
- `total_faces`: Faces extracted
- `faces_per_track_avg`: Average crops per track
- `quality_mean`: Average quality score
- `rejection_count`: Faces rejected due to low quality

### 7.3 Clustering Metrics
- `num_tracks`: Total tracks clustered
- `num_clusters`: Clusters formed
- `avg_cluster_size`: Mean tracks per cluster
- `singleton_fraction`: Fraction of tracks in singleton clusters
- `largest_cluster_fraction`: Fraction of tracks in largest cluster (high may indicate over-merging)

See [ACCEPTANCE_MATRIX.md](../../ACCEPTANCE_MATRIX.md) for acceptance thresholds.

---

## 8. Error Handling & Recovery

### 8.1 Graceful Degradation
- If RetinaFace fails to load, fall back to `detector="simulated"` (warn user)
- If ArcFace fails, skip embedding stage (allow detect/track to complete)
- If clustering fails, preserve `faces.jsonl` and `faces.npy` (allow manual retry)

### 8.2 Progress Tracking
- All stages write `progress.json` with:
  - `phase`: Current stage (`detect`, `track`, `embed`, `cluster`, `done`)
  - `frames_done`, `frames_total`: Frame-level progress
  - `elapsed_sec`: Wall-clock time
  - `fps_detected`: Actual video FPS
  - `analyzed_fps`: Processing FPS (effective throughput)
  - `eta_sec`: Estimated time remaining

### 8.3 Job Cancellation
- All async jobs support `POST /jobs/{job_id}/cancel`
- Clean SIGTERM to worker process
- Persist `state=canceled` in `data/jobs/{job_id}.json`

---

## 9. Common Workflows

### 9.1 Quick Local Run (CPU-safe)
```bash
# Detect + Track with CPU-safe defaults
python tools/episode_run.py \
  --ep-id rhobh-s05e02 \
  --video data/videos/rhobh-s05e02/episode.mp4 \
  --stride 8 --fps 8 \
  --device auto --coreml-only \
  --no-save-frames --no-save-crops

# Embed faces
python tools/episode_run.py --ep-id rhobh-s05e02 --faces-embed --save-crops

# Cluster identities
python tools/episode_run.py --ep-id rhobh-s05e02 --cluster
```

### 9.2 Full GPU Run (Max Quality)
```bash
python tools/episode_run.py \
  --ep-id rhobh-s05e02 \
  --video data/videos/rhobh-s05e02/episode.mp4 \
  --stride 1 --fps 30 \
  --device cuda \
  --save-frames --save-crops
```

### 9.3 Episode Cleanup (Post-Processing)
```bash
python tools/episode_cleanup.py \
  --ep-id rhobh-s05e02 \
  --video data/videos/rhobh-s05e02/episode.mp4 \
  --actions split_tracks reembed recluster group_clusters \
  --write-back
```

### 9.4 API-Driven Workflow
```bash
# Submit async detect/track job
curl -X POST http://localhost:8000/jobs/detect_track_async \
  -H "Content-Type: application/json" \
  -d '{"ep_id": "rhobh-s05e02", "profile": "balanced"}'
# => {"job_id": "job-abc123"}

# Poll progress
curl http://localhost:8000/jobs/job-abc123/progress
# => {"phase": "track", "frames_done": 1200, "frames_total": 2400, "eta_sec": 45.2}

# Cancel if needed
curl -X POST http://localhost:8000/jobs/job-abc123/cancel
```

---

## 10. Next Steps

- **New to the pipeline?** Start with [detect_track_faces.md](detect_track_faces.md) to understand detection and tracking.
- **Embedding issues?** See [faces_harvest.md](faces_harvest.md) for crop quality and sampling.
- **Clustering problems?** See [cluster_identities.md](cluster_identities.md) for identity grouping.
- **Performance tuning?** See [docs/ops/performance_tuning_faces_pipeline.md](../ops/performance_tuning_faces_pipeline.md).
- **Troubleshooting?** See [docs/ops/troubleshooting_faces_pipeline.md](../ops/troubleshooting_faces_pipeline.md).

---

**Maintained by:** Screenalytics Engineering
**Next Review:** After Phase 2 completion
