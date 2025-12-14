# Product Requirements Document — Screenalytics

Screenalytics delivers automated face, voice, and screen-time intelligence for unscripted television, powering catalog metadata, commerce, and fan engagement flows. This PRD captures the CI/promotion process, phased roadmap, risks, acceptance criteria, future enhancements, and key references.

---

## 11. CI / Promotion Flow
1. Develop in `FEATURES/<name>/` (`src/`, `tests/`, `docs/`, `TODO.md`).
2. CI enforces:
   - 30-day TTL
   - Tests + docs present
   - No imports from `FEATURES/**` in production
3. Promote via PR: move code/tests/docs out of `FEATURES/` into production paths (`apps/`, `web/`, `packages/`).
4. Post-promotion CI re-runs end-to-end sample clip.

---

## 12. Phased Roadmap

| Phase | Scope | Key Deliverables |
|-------|-------|------------------|
| **0 – Init** | Repo scaffold | MANIFEST, README, PRD, Architecture, Directory docs |
| **1 – Vision Pipeline** | Detection → Tracking → Embeddings | Accurate track files + metrics |
| **2 – Identity & Facebank** | ArcFace + override UI | Reliable per-cast ID |
| **3 – Audio Fusion** | Diarization + ASR | Speaking-time metrics |
| **4 – UI Integration** | SHOWS + PEOPLE tabs | Unified catalog + featured thumbnails |
| **5 – Agents & Automation** | Codex + MCP | Auto relabel + exports |
| **6 – Production Scale** | Multi-GPU pipeline + dashboard | 100 % episode coverage |

---

## 13. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|---------|-------------|
| **Occlusion / lighting variance** | Misidentifications | Use RetinaFace + quality gating; augment facebank |
| **Storage cost** | ↑ S3 fees | Lifecycle expiry for frames/thumbnails |
| **Audio desync** | Misaligned speech | Sync via ffmpeg timestamps + cross-correlation |
| **Facebank drift** | Identity errors | Periodic re-embedding + QA reports |
| **Agent miswrites** | Doc sprawl | Configured write-policy + CI checks |
| **Human bottleneck** | Slow QA | Cluster/track propagate + batch assign |
| **Legacy imports** | Fragile builds | No `.bak`; pre-commit denylist |

---

## 14. Acceptance Criteria

All acceptance criteria are **defined and enforced** in [ACCEPTANCE_MATRIX.md](../../ACCEPTANCE_MATRIX.md). High-level requirements:

### Pipeline Metrics (see ACCEPTANCE_MATRIX.md for full thresholds)

**Detection & Tracking:**
- tracks_per_minute: 10–30 (CPU/GPU)
- short_track_fraction: < 0.20 (CPU), < 0.15 (GPU)
- id_switch_rate: < 0.05 (CPU), < 0.03 (GPU)
- Runtime: ≤ 10 min GPU for 1hr episode

**Face Embedding:**
- faces_per_track_avg: 20–50
- quality_mean: ≥ 0.75
- rejection_rate: < 0.30
- Runtime: ≤ 30 sec GPU for 1hr episode

**Clustering:**
- singleton_fraction: < 0.30
- largest_cluster_fraction: < 0.40
- num_clusters: 5–15 (typical TV episode)
- Runtime: ≤ 10 sec for 1hr episode

### Quality Gates
- ✅ Artifacts follow documented formats ([docs/reference/artifacts_faces_tracks_identities.md](../reference/artifacts_faces_tracks_identities.md))
- ✅ Integration tests pass with metric assertions ([ACCEPTANCE_MATRIX.md](../../ACCEPTANCE_MATRIX.md) section 3)
- ✅ All configs documented ([docs/reference/config/pipeline_configs.md](../reference/config/pipeline_configs.md))
- ✅ Performance profiles verified on CPU and GPU
- ✅ Guardrails trigger warnings when metrics exceed thresholds

**For complete acceptance criteria, thresholds, and verification methods**, see [ACCEPTANCE_MATRIX.md](../../ACCEPTANCE_MATRIX.md).

---

## 15. Future Enhancements
- Scene-type classification via VLM (e.g., LLaVA / Qwen-VL).
- Emotion / sentiment tagging.
- Multi-camera or split-screen handling.
- Real-time episode analytics stream.
- Dashboard for cross-season trends.

---

## 16. References

### Top-Level Documentation
- [README.md](../../README.md) — Project overview and quick start
- [SETUP.md](../../SETUP.md) — Installation, artifact pipeline, hardware requirements
- [docs/README.md](../README.md) — Canonical docs index
- [docs/reference/api.md](../reference/api.md) — API endpoint reference
- [docs/reference/config/pipeline_configs.md](../reference/config/pipeline_configs.md) — Configuration reference
- [ACCEPTANCE_MATRIX.md](../../ACCEPTANCE_MATRIX.md) — QA checklist with metrics & thresholds
- [docs/architecture/solution_architecture.md](../architecture/solution_architecture.md) — System architecture diagram
- [docs/architecture/directory_structure.md](../architecture/directory_structure.md) — Repository layout
- [docs/features/feature_sandboxes.md](../features/feature_sandboxes.md) — Feature sandbox workflow

### Pipeline Documentation
- [docs/pipeline/overview.md](../pipeline/overview.md) — End-to-end pipeline architecture
- [docs/pipeline/detect_track_faces.md](../pipeline/detect_track_faces.md) — Detection & tracking
- [docs/pipeline/faces_harvest.md](../pipeline/faces_harvest.md) — Face sampling & embedding
- [docs/pipeline/cluster_identities.md](../pipeline/cluster_identities.md) — Identity clustering
- [docs/pipeline/episode_cleanup.md](../pipeline/episode_cleanup.md) — Post-processing & cleanup
- [docs/pipeline/audio_pipeline.md](../pipeline/audio_pipeline.md) — Audio pipeline and A/V fusion inputs

### Configuration & Tuning
- [docs/reference/config/pipeline_configs.md](../reference/config/pipeline_configs.md) — Complete config parameters
- [docs/ops/performance_tuning_faces_pipeline.md](../ops/performance_tuning_faces_pipeline.md) — Speed vs accuracy tuning
- [docs/ops/troubleshooting_faces_pipeline.md](../ops/troubleshooting_faces_pipeline.md) — Debugging & common fixes

### Schemas & Metrics
- [docs/reference/artifacts_faces_tracks_identities.md](../reference/artifacts_faces_tracks_identities.md) — Vision artifact schemas
- [docs/audio/diarization_manifest.md](../audio/diarization_manifest.md) — Audio diarization/ASR manifests
- [ACCEPTANCE_MATRIX.md](../../ACCEPTANCE_MATRIX.md) — Metric thresholds and guardrails

### Operations
- [docs/ops/ARTIFACTS_STORE.md](../ops/ARTIFACTS_STORE.md) — Storage layout and artifact handling

---

**Approved by:** TRR Engineering / Analytics / Design / Automation leads
**Next review:** after Phase 1 completion
