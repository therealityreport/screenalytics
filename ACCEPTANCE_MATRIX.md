# ACCEPTANCE_MATRIX.md — Screenalytics

Version: 2.0
Status: Live QA Standard
Last Updated: 2025-11-18

---

## 1. Purpose

To define measurable acceptance criteria for every pipeline module in Screenalytics.
A module is considered **Accepted** only when all associated checkpoints below are satisfied and verified by CI or QA review.

---

## 2. Acceptance Columns

| Field | Description |
|-------|-------------|
| **Module** | Pipeline stage or system component |
| **Scope** | What it's responsible for |
| **Acceptance Criteria** | Quantifiable pass/fail conditions |
| **Verification Method** | Unit, Integration, Manual, or CI |
| **Status** | ✅ Accepted / ⚠ Pending / ❌ Blocked |
| **Owner** | Team or engineer responsible |

---

## 3. Pipeline Modules

### 3.1 Detection & Tracking (`detect_track`)

| Metric | Target (CPU) | Target (GPU) | Warning Threshold | Verification |
|--------|--------------|--------------|-------------------|--------------|
| **tracks_per_minute** | 10–30 | 10–30 | > 50 | Integration test |
| **short_track_fraction** | < 0.20 | < 0.15 | > 0.30 | Integration test |
| **id_switch_rate** | < 0.05 | < 0.03 | > 0.10 | Integration test |
| **Runtime (1hr episode)** | ≤ 3× realtime (~3hrs) | ≤ 10 min | > 5× realtime (CPU), > 15 min (GPU) | Benchmark |
| **Missing faces (recall)** | ≥ 85% | ≥ 90% | < 80% | Manual QA on validation clips |

**Tests:**
- `tests/ml/test_detect_track_real.py`

**Config Dependencies:**
- `config/pipeline/detection.yaml`
- `config/pipeline/tracking.yaml`
- `config/pipeline/performance_profiles.yaml`

**Docs:**
- [docs/pipeline/detect_track_faces.md](docs/pipeline/detect_track_faces.md)
- [docs/reference/artifacts_faces_tracks_identities.md](docs/reference/artifacts_faces_tracks_identities.md)

**Status:** ⚠ Pending

---

### 3.2 Faces Embedding (`faces_embed`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **faces_per_track_avg** | 20–50 | > 100 | Integration test |
| **quality_mean** | ≥ 0.75 | < 0.60 | Integration test |
| **rejection_rate** | < 0.30 | > 0.50 | Integration test |
| **embedding_dimension** | 512 | ≠ 512 | Unit test |
| **embedding_norm** | ≈ 1.0 (±0.01) | > 1.05 or < 0.95 | Unit test |
| **Runtime (1hr episode, CPU)** | ≤ 2 min | > 5 min | Benchmark |
| **Runtime (1hr episode, GPU)** | ≤ 30 sec | > 1 min | Benchmark |

**Tests:**
- `tests/ml/test_faces_embed.py`

**Config Dependencies:**
- `config/pipeline/faces_embed_sampling.yaml`

**Docs:**
- [docs/pipeline/faces_harvest.md](docs/pipeline/faces_harvest.md)

**Status:** ⚠ Pending

---

### 3.3 Identity Clustering (`cluster`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **num_clusters** | 5–15 (typical TV episode) | > 30 | Integration test |
| **singleton_fraction** | < 0.30 | > 0.50 | Integration test |
| **largest_cluster_fraction** | < 0.40 | > 0.60 | Integration test |
| **avg_cluster_size** | 2–5 | > 10 | Integration test |
| **cluster_centroid_norm** | ≈ 1.0 (±0.01) | > 1.05 or < 0.95 | Unit test |
| **Runtime (1hr episode)** | ≤ 10 sec | > 30 sec | Benchmark |

**Tests:**
- `tests/ml/test_cluster.py`

**Config Dependencies:**
- Clustering params (TBD: `config/pipeline/recognition.yaml` or inline)

**Docs:**
- [docs/pipeline/cluster_identities.md](docs/pipeline/cluster_identities.md)

**Status:** ⚠ Pending

---

### 3.4 Episode Cleanup (`episode_cleanup`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **short_track_fraction_improvement** | ≥ 30% reduction | < 10% reduction | Integration test |
| **singleton_fraction_improvement** | ≥ 20% reduction | < 5% reduction | Integration test |
| **tracks_split** | ≥ 5% of long tracks | 0 (no splits performed) | Integration test |
| **Runtime (1hr episode)** | ≤ 5 min | > 15 min | Benchmark |

**Tests:**
- `tests/ml/test_episode_cleanup.py`

**Config Dependencies:**
- Reuses all pipeline configs

**Docs:**
- [docs/pipeline/episode_cleanup.md](docs/pipeline/episode_cleanup.md)

**Status:** ⚠ Pending

---

### 3.5 Audio + Voice Identity (`audio_pipeline`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **duration_drift_pct** | < 1.0% | > 2.0% | Integration test |
| **snr_db** | ≥ 18 dB | < 14 dB | Integration test |
| **mean_diarization_conf** | ≥ 0.75 | < 0.65 | Integration test |
| **mean_asr_conf** | ≥ 0.80 | < 0.70 | Integration test |
| **voice_cluster_count** | 3–15 (typical TV episode) | > 20 | Integration test |
| **transcript_speaker_fields** | 100% rows have all fields | < 95% | Unit test |
| **voice_cluster_consistency** | All transcript IDs in clusters.json | Missing IDs | Unit test |
| **voice_mapping_consistency** | All clusters mapped | Missing mappings | Unit test |
| **Runtime (1hr episode)** | ≤ 15 min | > 30 min | Benchmark |

**Required Transcript Fields (per row):**
- `speaker_id` (e.g., `SPK_LISA_BARLOW`)
- `speaker_display_name` (e.g., `"Lisa Barlow"`)
- `voice_cluster_id` (e.g., `VC_01`)
- `voice_bank_id` (e.g., `voice_lisa_barlow`)

**Artifacts Validated:**
- `audio_voice_clusters.json` - Cluster IDs and segments
- `audio_voice_mapping.json` - Cluster-to-speaker mapping
- `episode_transcript.jsonl` - Speaker-labeled transcript
- `episode_transcript.vtt` - WebVTT with speaker metadata
- `audio_qc.json` - QC report with voice stats

**Tests:**
- `tests/audio/test_audio_models.py`
- `tests/audio/test_audio_qc.py`
- `tests/audio/test_voice_clusters.py`
- `tests/audio/test_voice_bank.py`
- `tests/audio/test_fuse_diarization_asr.py`
- `tests/audio/test_episode_audio_pipeline_e2e.py`

**Config Dependencies:**
- `config/pipeline/audio.yaml`

**Docs:**
- [docs/pipeline/audio_pipeline.md](docs/pipeline/audio_pipeline.md)

**Status:** ⚠ Pending

---

### 3.6 Facebank Management (`facebank`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **Duplicate faces** | < 2% | > 5% | Integration test |
| **Embedding drift** | < 3% per month | > 10% | Manual QA |
| **Cross-episode match accuracy** | ≥ 85% | < 75% | Manual QA on validation episodes |

**Tests:**
- `tests/ml/test_facebank.py`

**Docs:**
- [docs/reference/facebank.md](docs/reference/facebank.md)

**Status:** ⚠ Pending

---

### 3.7 Face Alignment (`face_alignment`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **landmark_jitter_px** | < 2.0 px | > 5.0 px | Integration test |
| **alignment_quality_mean** | ≥ 0.75 | < 0.60 | Integration test |
| **pose_accuracy_degrees** | ≤ 5° MAE | > 10° | Manual QA / Eval set |
| **track_fragmentation_delta** | ≥ 10% reduction | < 5% reduction | Integration test |
| **Runtime (1hr episode)** | ≤ 5 min | > 10 min | Benchmark |

**Definition of Done:**
- [x] FAN aligner scaffold in `FEATURES/face_alignment/src/`
- [x] 68-point landmark extraction interface (`FANAligner` class)
- [x] 5-point extraction from 68-point for ArcFace alignment
- [x] Aligned crop generation with similarity transform
- [x] Config-driven thresholds (`config/pipeline/face_alignment.yaml`)
- [x] CLI entrypoint (`python -m FEATURES.face_alignment`)
- [x] Test fixtures with synthetic data
- [x] Unit tests for data structures and math utilities
- [x] Alignment quality heuristic wired into pipeline (populates `alignment_quality` field)
- [ ] Install `face-alignment` package and validate with real model
- [x] Main pipeline consumes `alignment_quality` for optional embedding gating (when `face_alignment/aligned_faces.jsonl` exists)
- [ ] Auto-run face alignment in the main pipeline (requires promotion out of `FEATURES/`; CI blocks production imports from `FEATURES/**`)
- [ ] Metrics validated on eval set

**Tests:**
- `FEATURES/face_alignment/tests/test_face_alignment.py` - Unit + integration tests
- `FEATURES/face_alignment/tests/fixtures.py` - Synthetic test data

**Config Dependencies:**
- `config/pipeline/face_alignment.yaml`

**Rollback Levers:**
- `config/pipeline/face_alignment.yaml` → `face_alignment.enabled: false` - Disable producing alignment artifacts
- Disable embedding gating: `config/pipeline/embedding.yaml` → `face_alignment.enabled: false` (or set `min_alignment_quality: 0.0`)

**Docs:**
- [docs/todo/feature_face_alignment_fan_luvli_3ddfa.md](docs/todo/feature_face_alignment_fan_luvli_3ddfa.md)
- [docs/features/vision_alignment_and_body_tracking.md](docs/features/vision_alignment_and_body_tracking.md)
- [FEATURES/face_alignment/TODO.md](FEATURES/face_alignment/TODO.md)

**Feature Sandbox:** `FEATURES/face_alignment/`

**Note:** The `alignment_quality` field is currently heuristic-based. There is no hard numeric acceptance bound for this phase; the field is primarily diagnostic. Model-based thresholds will be added with LUVLi integration.

**Evaluation (via `tools/experiments/face_alignment_eval.py`):**

| Metric | Target | Verification |
|--------|--------|--------------|
| `embedding_jitter_mean` | No regression (≤ baseline + 0.005) | Eval script |
| `id_switch_rate_per_minute` | No increase (≤ baseline + 0.1/min) | Eval script |
| `avg_track_length` | No decrease (≥ baseline - 0.5) | Eval script |
| `alignment_quality_mean` | ≥ 0.60 | Eval script |

**Status:** ✅ Scaffold Implemented (pending eval)

---

### 3.8 Alignment Quality Gate (`alignment_quality`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **alignment_quality_threshold** | 0.30 (configurable) | N/A | Unit test |
| **faces_gated_pct** | 10–30% | > 50% | Integration test |
| **false_rejection_rate** | < 5% | > 10% | Manual QA / Eval set |
| **recognition_accuracy_delta** | ≥ 2% improvement | < 0% (regression) | Eval set |

**Definition of Done:**
- [x] Alignment quality heuristic (`FEATURES/face_alignment/src/alignment_quality.py`)
- [x] `alignment_quality` field populated in pipeline
- [x] Gating integrated in embedding stage (`_run_faces_embed_stage`)
- [x] Config-driven threshold (`config/pipeline/embedding.yaml` → `face_alignment.min_alignment_quality`)
- [x] Skip reason logging (`low_alignment_quality:{score}`)
- [x] Eval harness gating mode (`--gating on`)
- [ ] Gating impact validated on eval set (embedding jitter reduction)

**Gating Configuration:**
```yaml
# config/pipeline/embedding.yaml
face_alignment:
  enabled: true
  use_for_embedding: true
  min_alignment_quality: 0.3    # Skip faces below this threshold
```

**Eval Harness:**
```bash
# Evaluate with gating impact
python -m tools.experiments.face_alignment_eval --episode-id ep1 --gating on
```

**Tests:**
- `tests/integration/test_face_alignment_pipeline.py`
- `tools/experiments/tests/test_face_alignment_eval.py`

**Config Dependencies:**
- `config/pipeline/embedding.yaml` (face_alignment section)

**Docs:**
- [FEATURES/face_alignment/docs/README.md](FEATURES/face_alignment/docs/README.md)
- [docs/todo/feature_face_alignment_fan_luvli_3ddfa.md](docs/todo/feature_face_alignment_fan_luvli_3ddfa.md)

**Feature Sandbox:** `FEATURES/face_alignment/`

**Current State:** `alignment_quality` is heuristic-based (see `FEATURES/face_alignment/src/alignment_quality.py`). LUVLi-style scaffolding exists, but a faithful LUVLi uncertainty/visibility model is not implemented yet; treat any per-landmark uncertainty/visibility fields as non-acceptance-grade until the model is integrated and validated.

**Status:** ✅ Heuristic Gating Integrated (pending eval validation)

---

### 3.9 3D Head Pose (`head_pose_3d`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **pose_yaw_accuracy** | ≤ 5° MAE | > 10° | Eval set |
| **pose_pitch_accuracy** | ≤ 5° MAE | > 10° | Eval set |
| **pose_roll_accuracy** | ≤ 3° MAE | > 7° | Eval set |
| **pose_jump_detection** | ≥ 95% | < 85% | Integration test |
| **Runtime (per face)** | ≤ 10ms (GPU) | > 20ms | Benchmark |

**Tests:**
- (None yet; feature not implemented)

**Config Dependencies:**
- `config/pipeline/face_alignment.yaml` (`head_pose_3d` section)

**Docs:**
- [docs/todo/feature_face_alignment_fan_luvli_3ddfa.md](docs/todo/feature_face_alignment_fan_luvli_3ddfa.md)

**Feature Sandbox:** `FEATURES/face_alignment/`

**Status:** ⚠ Pending

---

### 3.10 Body Detection & Tracking (`body_tracking`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **person_recall** | ≥ 90% | < 80% | Eval set |
| **person_precision** | ≥ 85% | < 75% | Eval set |
| **body_track_fragmentation** | < 0.15 | > 0.25 | Integration test |
| **body_id_switch_rate** | < 0.05 | > 0.10 | Integration test |
| **body_tracks_per_minute** | 5–20 | > 30 | Integration test |
| **Runtime (1hr episode)** | ≤ 15 min (GPU) | > 25 min | Benchmark |

**Definition of Done:**
- [x] YOLO person detection runs on video frames
- [x] ByteTrack associates detections into tracks (with SimpleIoU fallback)
- [x] OSNet Re-ID embeddings computed for representative frames
- [x] Artifacts written to `data/manifests/{ep_id}/body_tracking/`
- [x] Config-driven thresholds (no hardcoded values)
- [ ] Metrics validated on eval set

**Tests:**
- `FEATURES/body_tracking/tests/test_body_tracking.py` - Unit + integration tests
- `FEATURES/body_tracking/tests/fixtures.py` - Synthetic test data

**Config Dependencies:**
- `config/pipeline/body_detection.yaml`

**Rollback Levers:**
- `body_tracking.enabled: false` - Disable entire feature
- `person_detection.device: cpu` - Force CPU mode

**Docs:**
- [FEATURES/body_tracking/docs/README.md](FEATURES/body_tracking/docs/README.md)
- [docs/todo/feature_body_tracking_reid_fusion.md](docs/todo/feature_body_tracking_reid_fusion.md)
- [docs/features/vision_alignment_and_body_tracking.md](docs/features/vision_alignment_and_body_tracking.md)

**Feature Sandbox:** `FEATURES/body_tracking/`

**Status:** ⚠ Implemented (pending eval)

---

### 3.11 Person Re-ID (`person_reid`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **reid_mAP** | ≥ 0.80 | < 0.70 | Eval set |
| **reid_rank1_accuracy** | ≥ 0.90 | < 0.80 | Eval set |
| **embedding_dimension** | 512 (OSNet x1.0) | varies by model | Unit test |
| **embedding_norm** | ≈ 1.0 (±0.01) | > 1.05 or < 0.95 | Unit test |
| **Runtime (per crop)** | ≤ 5ms (GPU) | > 10ms | Benchmark |

**Definition of Done:**
- [x] OSNet model loads via torchreid
- [x] Body crops extracted with configurable margin
- [x] Embeddings computed in batches
- [x] Embeddings saved to `.npy` with metadata JSON
- [ ] Metrics validated on Re-ID benchmark

**Design Note:** Re-ID is pluggable - can swap OSNet for heavier models (MGN/BoT) later.

**Tests:**
- `FEATURES/body_tracking/tests/test_body_tracking.py` (TestConfigLoading)

**Config Dependencies:**
- `config/pipeline/body_detection.yaml` (person_reid section)

**Rollback Levers:**
- `person_reid.enabled: false` - Disable Re-ID embeddings

**Docs:**
- [docs/todo/feature_body_tracking_reid_fusion.md](docs/todo/feature_body_tracking_reid_fusion.md)

**Feature Sandbox:** `FEATURES/body_tracking/`

**Status:** ⚠ Implemented (pending eval)

---

### 3.12 Face↔Body Track Fusion (`track_fusion`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **association_accuracy** | ≥ 95% | < 90% | Manual QA / Annotated set |
| **false_association_rate** | < 2% | > 5% | Manual QA |
| **screen_time_gap_reduction** | ≥ 30% | < 15% | Integration test |
| **face_vs_body_screentime_gap_fraction** | Report only | N/A | Integration test |
| **handoff_latency_frames** | ≤ 3 | > 10 | Integration test |
| **reid_handoff_accuracy** | ≥ 90% | < 80% | Eval set |

**Definition of Done:**
- [x] IoU-based face↔body association when face visible
- [x] Re-ID handoff when face disappears
- [x] Union-find grouping of fused identities
- [x] Screen-time comparison (face-only vs combined)
- [x] Artifacts written to `track_fusion.json` and `screentime_comparison.json`
- [ ] Metrics validated on annotated episodes

**Tests:**
- `FEATURES/body_tracking/tests/test_body_tracking.py` (TestTrackFusion, TestScreenTimeComparison)

**Config Dependencies:**
- `config/pipeline/track_fusion.yaml`

**Rollback Levers:**
- `track_fusion.enabled: false` - Disable fusion (body tracks kept separate)
- `reid_handoff.enabled: false` - IoU-only association

**Analytics Integration:**
- `analytics_integration.py` computes `EpisodeAnalytics` with:
  - `face_visible_seconds` - Screen time where face is detected
  - `body_only_seconds` - Screen time with body but no face
  - `total_screen_time_seconds` - Combined face + body
  - `body_contribution_pct` - Percentage from body-only tracking
  - `gap_bridged_seconds` - Time recovered by body tracking
- Validation: `validate_acceptance_metrics()` checks thresholds

**Docs:**
- [FEATURES/body_tracking/docs/README.md](FEATURES/body_tracking/docs/README.md)
- [docs/todo/feature_body_tracking_reid_fusion.md](docs/todo/feature_body_tracking_reid_fusion.md)

**Feature Sandbox:** `FEATURES/body_tracking/`

**Status:** ⚠ Implemented (pending eval)

---

### 3.13 Face Alignment Future Features

#### 3.13.1 Alignment Quality Gate (`alignment_quality`) - FUTURE

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **alignment_quality_threshold** | 0.60 (configurable) | N/A | Unit test |
| **faces_gated_pct** | 10–30% | > 50% | Integration test |
| **false_rejection_rate** | < 5% | > 10% | Manual QA / Eval set |
| **recognition_accuracy_delta** | ≥ 2% improvement | < 0% (regression) | Eval set |

**Status:** ⚠ Not yet implemented - design ready (see [TODO.md](FEATURES/face_alignment/TODO.md))

#### 3.13.2 3D Head Pose Consistency (`head_pose_consistency`) - FUTURE

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **pose_temporal_smoothness** | < 5° jitter frame-to-frame | > 10° | Integration test |
| **profile_detection_accuracy** | ≥ 95% | < 85% | Eval set |
| **visibility_estimation_accuracy** | ≥ 90% | < 80% | Manual QA |

**Status:** ⚠ Not yet implemented - design ready (see [TODO.md](FEATURES/face_alignment/TODO.md))

---

### 3.14 TensorRT Embedding (`tensorrt_embedding`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **speedup_vs_pytorch** | ≥ 5× @ batch=32 | < 3× | Benchmark |
| **cosine_sim_mean** | ≥ 0.995 | < 0.990 | Parity test |
| **cosine_sim_min** | ≥ 0.990 | < 0.980 | Parity test |
| **embedding_norm** | ≈ 1.0 (±0.01) | > 1.05 or < 0.95 | Unit test |
| **vram_usage** | ≤ 2GB | > 4GB | Benchmark |
| **fp16_accuracy_delta** | < 0.1% | > 0.5% | Eval set |
| **engine_build_time** | ≤ 5 min | > 15 min | CI Benchmark |

**Definition of Done:**
- [x] ONNX export utility (`tools/models/export_arcface_onnx.py`)
- [x] TensorRT engine builder (`FEATURES/arcface_tensorrt/src/tensorrt_builder.py`)
- [x] TensorRT inference wrapper (`FEATURES/arcface_tensorrt/src/tensorrt_inference.py`)
- [x] Embedding parity comparison (`FEATURES/arcface_tensorrt/src/embedding_compare.py`)
- [x] Backend abstraction in main pipeline (`EmbeddingBackend` protocol)
- [x] Config-driven backend selection (`embedding.backend: pytorch|tensorrt`)
- [ ] Parity validated on eval set (cosine_sim_mean ≥ 0.995)
- [ ] Throughput validated on benchmark (speedup ≥ 3×)

**Backend Selection:**
```yaml
# config/pipeline/embedding.yaml
embedding:
  backend: pytorch    # pytorch | tensorrt
  tensorrt_config: config/pipeline/arcface_tensorrt.yaml
```

**Tests:**
- `FEATURES/arcface_tensorrt/tests/test_tensorrt_embedding.py`
- `tests/ml/test_arcface_embeddings.py` (ML-gated; baseline embedding invariants)

**Config Dependencies:**
- `config/pipeline/embedding.yaml`
- `config/pipeline/arcface_tensorrt.yaml`

**Docs:**
- [docs/todo/feature_arcface_tensorrt_onnxruntime.md](docs/todo/feature_arcface_tensorrt_onnxruntime.md)

**Feature Sandbox:** `FEATURES/arcface_tensorrt/`

**Status:** ✅ Backend Integrated (pending parity validation)

---

### 3.15 Face Mesh Analytics (`face_mesh`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **mesh_stability_px** | < 3.0 px | > 6.0 px | Integration test |
| **visibility_fraction_accuracy** | ≥ 90% | < 80% | Manual QA / Labeled set |
| **gaze_category_accuracy** | ≥ 85% | < 75% | Labeled set |
| **runtime_per_face** | ≤ 15ms | > 30ms | Benchmark |
| **selective_execution_savings** | ≥ 80% compute reduction | < 60% | Integration test |

**Tests:**
- (None yet; feature not implemented)

**Config Dependencies:**
- `config/pipeline/analytics.yaml`

**Docs:**
- [docs/todo/feature_mesh_and_advanced_visibility.md](docs/todo/feature_mesh_and_advanced_visibility.md)

**Feature Sandbox:** `FEATURES/vision_analytics/`

**Status:** ⚠ Pending

---

### 3.16 CenterFace Detector (`centerface`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **precision** | ≥ 0.90 | < 0.85 | Eval set |
| **recall** | ≥ 0.85 | < 0.80 | Eval set |
| **agreement_with_retinaface** | ≥ 90% | < 85% | Integration test |
| **runtime_cpu** | ≤ 50ms/frame | > 100ms | Benchmark |
| **runtime_gpu** | ≤ 10ms/frame | > 25ms | Benchmark |

**Tests:**
- (None yet; feature not implemented)

**Config Dependencies:**
- `config/pipeline/analytics.yaml` (centerface section)

**Docs:**
- [docs/todo/feature_mesh_and_advanced_visibility.md](docs/todo/feature_mesh_and_advanced_visibility.md)

**Feature Sandbox:** `FEATURES/vision_analytics/`

**Status:** ⚠ Pending (Future Work)

---

## 4. API & UI Modules

| Module | Scope | Acceptance Criteria | Verification | Status |
|--------|-------|---------------------|--------------|--------|
| **Jobs API** | Async job orchestration | Jobs complete successfully; progress.json updated; state transitions correct | Integration test | ⚠ Pending |
| **Facebank UI** | Identity moderation | Merge/split/move/delete operations persist; thumbnails load; presigned URLs valid | UI smoke test | ⚠ Pending |
| **Episode Workspace** | Multi-tab interface | All tabs load; detect/track/cluster jobs launch; artifacts downloadable | UI smoke test | ⚠ Pending |

---

## 5. Non-Functional Criteria

| Category | Target | Verification |
|----------|--------|--------------|
| **Performance (GPU)** | ≤ 10 min / hour episode | Benchmark CI |
| **Performance (CPU)** | ≤ 3× realtime | Benchmark CI |
| **Scalability** | Horizontal workers supported (4+ parallel) | Stress Test |
| **Reliability** | No data loss on crash; graceful degradation | Restart Simulation |
| **Security** | Signed URLs; no secrets in repo | Code Scan |
| **Docs / Config** | Updated README, PRD, Pipeline docs, Config reference | CI doc-check |
| **CI Stability** | 100% green builds on `main` | GitHub Actions |
| **Promotion Policy** | TTL ≤ 30 days; tests+docs required | CI Gate |

---

## 6. Agents & Automation

| Task | Expected Behavior | Verification | Status |
|------|-------------------|--------------|--------|
| **update-docs-on-change** | Auto-syncs README, PRD, Directory, Architecture on file changes | Codex CI log | ⚠ Pending |
| **promotion-check** | Validates TODO → PROMOTED, tests, docs | CI | ⚠ Pending |
| **feature-expiry** | Flags features older than 30 days | CI | ⚠ Pending |
| **config-validator** | Ensures YAML schemas valid | CI | ⚠ Pending |

---

## 7. Sign-off Procedure

1. Feature owner verifies all applicable criteria
2. CI passes automated checks (tests, lint, schema validation)
3. Manual QA on validation clips (if applicable)
4. PR labeled `promotion` merges into `main`
5. `ACCEPTANCE_MATRIX.md` row updated to ✅ **Accepted**
6. Agents post summary under release tag

---

## 8. Change Control

- All edits tracked via PR review
- Agents automatically append new rows when features are added in `FEATURES/`
- CI compares feature folders against matrix entries; warns if missing
- CI blocks promotion if production feature lacks ✅ entry

---

## 9. Module Status Summary

### Core Pipeline Modules (3.1-3.6)

| Module | Status | Blocker | Next Action |
|--------|--------|---------|-------------|
| **detect_track** | ⚠ Pending | Integration tests not passing thresholds | Tune thresholds, add test assertions |
| **faces_embed** | ⚠ Pending | Tests not implemented | Write integration tests |
| **cluster** | ⚠ Pending | Tests not implemented | Write integration tests |
| **episode_cleanup** | ⚠ Pending | Tests not implemented | Write integration tests |
| **audio_pipeline** | ⚠ Pending | E2E tests incomplete | Add voice clustering + bank tests |
| **facebank** | ⚠ Pending | Cross-episode matching not validated | Manual QA on validation episodes |

### Vision Enhancement Modules (3.7-3.15) — NEW

| Module | Status | Blocker | Next Action |
|--------|--------|---------|-------------|
| **face_alignment** | ✅ Scaffold Implemented | Not auto-run by main pipeline | Validate with real model + decide promotion/integration path |
| **alignment_quality** | ✅ Heuristic Integrated | Pending eval validation | Run gating impact eval |
| **head_pose_3d** | ⚠ Pending | Depends on face_alignment | Implement 3DDFA_V2 integration |
| **body_tracking** | ✅ Implemented | Pending eval metrics | Run eval on validation episodes |
| **person_reid** | ✅ Implemented | Pending eval metrics | Run Re-ID benchmark |
| **track_fusion** | ✅ Implemented | Pending eval metrics | Manual QA on fused identities |
| **tensorrt_embedding** | ✅ Integrated | Pending parity validation | Run parity + throughput tests |
| **face_mesh** | ⚠ Pending | Implementation not started | Implement MediaPipe mesh |
| **centerface** | ⚠ Pending (Future) | Deferred | Stub interface only |

### API & UI Modules

| Module | Status | Blocker | Next Action |
|--------|--------|---------|-------------|
| **jobs_api** | ⚠ Pending | Integration tests incomplete | Extend API test coverage |
| **facebank_ui** | ⚠ Pending | UI smoke tests not automated | Add Playwright/Cypress tests |
| **agents** | ⚠ Pending | Playbooks not validated | Test doc-sync automation |

---

**Next Review:** After Vision Enhancement Phase 1 (FAN + TensorRT)
**Maintainers:** Screenalytics Engineering + QA
