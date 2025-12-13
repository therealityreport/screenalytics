# Audit: Vision Alignment / Body Tracking / Embedding Engines — Status

Last updated: 2025-12-13  
Scope: `FEATURES/face_alignment`, `FEATURES/body_tracking`, `FEATURES/arcface_tensorrt`, `FEATURES/embedding_engines`, `FEATURES/vision_analytics`

This audit is the “single source of truth” for what exists **in code today** vs what is **planned-only** (docs/config scaffolds).

## Quick Naming / Canonical References

- **Feature sandbox paths use underscores** (examples): `FEATURES/face_alignment/`, `FEATURES/body_tracking/`, `FEATURES/arcface_tensorrt/`, `FEATURES/embedding_engines/`, `FEATURES/vision_analytics/`.
- **Canonical face-alignment config:** `config/pipeline/face_alignment.yaml` (used by `FEATURES.face_alignment`).
- `config/pipeline/alignment.yaml` exists but is **legacy / not wired** to the current implementation (keep only as reference unless/until re-integrated).
- `config/pipeline/analytics.yaml` contains planned knobs for mesh/visibility/centerface, but there is **no runnable implementation** yet.

## Status Summary

| Building Block | Status | Needed Now? | Primary “Where” | Primary Config |
|---|---|---:|---|---|
| FAN face alignment (2D/3D landmarks) | Implemented (sandbox) | Core | `FEATURES/face_alignment/src/run_fan_alignment.py` | `config/pipeline/face_alignment.yaml` |
| Body tracking + Re-ID + fusion | Implemented (sandbox) | Scale | `FEATURES/body_tracking/src/body_tracking_runner.py` | `config/pipeline/body_detection.yaml`, `config/pipeline/track_fusion.yaml` |
| ArcFace TensorRT embeddings | Implemented (sandbox + pipeline backend) | Scale | `FEATURES/arcface_tensorrt/src/` + `tools/episode_run.py` | `config/pipeline/embedding.yaml`, `config/pipeline/arcface_tensorrt.yaml` |
| LUVLi-style quality + visibility gating | Partial (heuristic) | Core | `FEATURES/face_alignment/src/run_luvli_quality.py` | `config/pipeline/embedding.yaml` (gating) |
| 3DDFA_V2 dense 3D + head pose | Not found (planned-only) | Future | — | `config/pipeline/face_alignment.yaml` (stub) |
| InsightFace_Pytorch reference ArcFace + eval suite | Not found (planned-only) | Future | — | — |
| ONNXRuntime C++ embedding service plan | Planned-only | Future | `FEATURES/embedding_engines/docs/` | `docs/todo/feature_arcface_tensorrt_onnxruntime.md` |
| MediaPipe face mesh / advanced visibility | Planned-only | Advanced | `FEATURES/vision_analytics/docs/` | `config/pipeline/analytics.yaml` |
| CenterFace detector (CPU fallback) | Planned-only | Future | `FEATURES/vision_analytics/docs/` | `config/pipeline/analytics.yaml` |

---

## 1) FAN / Face Alignment (2D/3D landmarks)

- **Status:** Implemented (Feature sandbox; manual run)
- **Where:**
  - `FEATURES/face_alignment/src/run_fan_alignment.py` (`FANAligner`, `AlignedFace`)
  - `FEATURES/face_alignment/src/face_alignment_runner.py` (`FaceAlignmentRunner`)
  - `FEATURES/face_alignment/src/export_aligned_faces.py` (writes `aligned_faces.jsonl`, optional crops)
- **Pipeline wiring:**
  - CLI: `python -m FEATURES.face_alignment --episode-id <EP_ID>`
  - Current pipeline consumption is **artifact-based** (embedding stage reads alignment quality from `aligned_faces.jsonl` when present; see `_load_alignment_quality_index()` in `tools/episode_run.py`).
  - Smoke runner validates presence (does not execute sandbox): `python -m tools.smoke.smoke_run --episode-id <EP_ID> --alignment on`
- **Artifacts:**
  - `data/manifests/{ep_id}/face_alignment/aligned_faces.jsonl`
  - `data/manifests/{ep_id}/face_alignment/aligned_crops/` (optional)
- **Config flags / rollback:**
  - `config/pipeline/face_alignment.yaml`: `face_alignment.enabled: false` disables producing alignment artifacts.
  - If you want **no embedding gating** (regardless of alignment artifacts), set `config/pipeline/embedding.yaml` → `face_alignment.enabled: false` or `min_alignment_quality: 0.0`.
- **Tests:**
  - `pytest FEATURES/face_alignment/tests/test_face_alignment.py -v`
  - `pytest tools/experiments/tests/test_face_alignment_eval.py -v`
- **Observability:**
  - Logs: `face_alignment` logger emits `[STAGE]` and output paths.
  - Metrics: `data/manifests/{ep_id}/face_alignment/alignment_metrics.json` (if export stage runs).
- **Gaps / risks:**
  - Not automatically run by the “main” detect→embed pipeline; must be run separately today.
- **Needed now?** Core — improves diagnosability and enables embedding quality gating.

## 2) Body Tracking + Person Re-ID + Face↔Body Fusion

- **Status:** Implemented (Feature sandbox; optional/manual run)
- **Where:**
  - `FEATURES/body_tracking/src/body_tracking_runner.py` (`BodyTrackingRunner`)
  - `FEATURES/body_tracking/src/detect_bodies.py` (YOLO person detections)
  - `FEATURES/body_tracking/src/track_bodies.py` (ByteTrack-style tracking + fallback)
  - `FEATURES/body_tracking/src/body_embeddings.py` (OSNet via `torchreid`)
  - `FEATURES/body_tracking/src/track_fusion.py` (IoU + Re-ID associations)
  - `FEATURES/body_tracking/src/screentime_compare.py` (face-only vs face+body diffs)
- **Pipeline wiring:**
  - CLI: `python -m FEATURES.body_tracking --episode-id <EP_ID> [--stage detect|track|fuse|compare]`
  - Screentime service uses the comparison artifact when present:
    - `apps/api/services/screentime.py` reads `data/manifests/{ep_id}/body_tracking/screentime_comparison.json`
  - Smoke runner validates presence (does not execute sandbox): `python -m tools.smoke.smoke_run --episode-id <EP_ID> --body-tracking`
- **Artifacts:**
  - `data/manifests/{ep_id}/body_tracking/body_detections.jsonl`
  - `data/manifests/{ep_id}/body_tracking/body_tracks.jsonl`
  - `data/manifests/{ep_id}/body_tracking/body_embeddings.npy`
  - `data/manifests/{ep_id}/body_tracking/body_embeddings_meta.json`
  - `data/manifests/{ep_id}/body_tracking/track_fusion.json`
  - `data/manifests/{ep_id}/body_tracking/screentime_comparison.json`
- **Config flags / rollback:**
  - `config/pipeline/body_detection.yaml`: `body_tracking.enabled: false`, `person_reid.enabled: false`
  - `config/pipeline/track_fusion.yaml`: `track_fusion.enabled: false`, `reid_handoff.enabled: false`
- **Tests:**
  - `pytest FEATURES/body_tracking/tests/test_body_tracking.py -v`
- **Observability:**
  - Logs: `body_tracking` logger emits stage summaries and artifact paths.
  - UI: `apps/workspace-ui/pages/4_Screentime.py` shows body metrics if present.
- **Gaps / risks:**
  - External deps (`ultralytics`, `torchreid`) may not be installed in minimal environments; treat as optional.
- **Needed now?** Scale — improves continuity and fills “face missing” gaps.

## 3) TensorRT ArcFace via tensorrtx (embedding engine)

- **Status:** Implemented (Feature sandbox + selectable backend in embedding pipeline)
- **Where:**
  - `FEATURES/arcface_tensorrt/src/tensorrt_builder.py` (engine build/cache)
  - `FEATURES/arcface_tensorrt/src/tensorrt_inference.py` (inference wrapper)
  - `FEATURES/arcface_tensorrt/src/embedding_compare.py` (parity utilities)
  - `tools/episode_run.py` (backend selection + runtime)
- **Pipeline wiring:**
  - Config selects backend:
    - `config/pipeline/embedding.yaml`: `embedding.backend: pytorch|tensorrt`
    - `config/pipeline/embedding.yaml`: `embedding.tensorrt_config: config/pipeline/arcface_tensorrt.yaml`
  - CLI for building/benchmarking sandbox: `python -m FEATURES.arcface_tensorrt --mode build|compare|benchmark`
- **Artifacts:**
  - Engine cache defaults under `data/engines/` (from `config/pipeline/arcface_tensorrt.yaml`) and/or user cache dirs (see `FEATURES/arcface_tensorrt/src/tensorrt_builder.py`).
- **Config flags / rollback:**
  - Set `embedding.backend: pytorch` to disable TensorRT.
  - `config/pipeline/arcface_tensorrt.yaml`: `arcface_tensorrt.enabled` (feature flag for sandbox usage; pipeline uses backend selection).
- **Tests:**
  - `pytest FEATURES/arcface_tensorrt/tests/test_tensorrt_embedding.py -v`
- **Observability:**
  - Logs: builder/inference logs report engine path, resolved device, throughput.
- **Gaps / risks:**
  - Requires NVIDIA + TensorRT; keep PyTorch as default fallback.
- **Needed now?** Scale — throughput/perf lever once parity is validated on target hardware.

## 4) LUVLi-style alignment quality + visibility gating

- **Status:** Partial (heuristic + scaffolding)
- **Where:**
  - `FEATURES/face_alignment/src/alignment_quality.py` (heuristic)
  - `FEATURES/face_alignment/src/run_luvli_quality.py` (`LUVLiQualityEstimator` heuristic fallback; uncertainty/visibility fields are placeholders)
  - `tools/episode_run.py` consumes `alignment_quality` for optional embedding gating (`_load_alignment_quality_index`)
- **Pipeline wiring:**
  - Gating config lives in `config/pipeline/embedding.yaml` → `face_alignment.min_alignment_quality`
- **Artifacts:**
  - `alignment_quality` stored per aligned face in `data/manifests/{ep_id}/face_alignment/aligned_faces.jsonl`
- **Gaps / risks:**
  - Not true LUVLi; do not treat uncertainty/visibility fields as reliable yet.
- **Needed now?** Core — gating is already plumbed; model-based scoring is the missing piece.

## 5) 3DDFA_V2 (3D dense alignment + head pose)

- **Status:** Not found (planned-only)
- **Where:** No implementation under `FEATURES/` at time of audit.
- **Pipeline wiring:** Stub config exists in `config/pipeline/face_alignment.yaml` (`head_pose_3d.*`) and legacy `config/pipeline/alignment.yaml` (`ddfa_v2.*`).
- **Needed now?** Future — useful for visibility analytics and pose-based embedding gating, but higher complexity.

## 6) InsightFace_Pytorch (reference ArcFace + eval suite)

- **Status:** Not found (planned-only)
- **Where:** No vendored training/eval code in-repo; current runtime uses the `insightface` Python package (models fetched via `scripts/fetch_models.py`).
- **Needed now?** Future — consider only if training/eval becomes a first-class workflow.

## 7) ONNXRuntime C++ embedding service plan

- **Status:** Planned-only
- **Where:**
  - `FEATURES/embedding_engines/docs/README.md`
  - `docs/todo/feature_arcface_tensorrt_onnxruntime.md`
- **Notes:** ONNXRuntime is used **in-process** today (provider selection in `tools/episode_run.py`); the separate C++ service is a future architecture idea.
- **Needed now?** Future — only needed at higher scale / service boundaries.

## 8) MediaPipe Face Mesh / Attention Mesh

- **Status:** Planned-only
- **Where:**
  - `FEATURES/vision_analytics/docs/README.md`
  - `docs/todo/feature_mesh_and_advanced_visibility.md`
  - `config/pipeline/analytics.yaml`
- **Needed now?** Advanced — valuable for visibility analytics, but requires new implementation + labeled QA.

## 9) CenterFace detector (CPU fallback)

- **Status:** Planned-only
- **Where:**
  - `config/pipeline/analytics.yaml` (`centerface.*`)
  - `docs/todo/feature_mesh_and_advanced_visibility.md`
- **Needed now?** Future — revisit only if a CPU-only detector fallback is required.

