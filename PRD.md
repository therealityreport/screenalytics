# Product Requirements Document — Screenalytics

Screenalytics delivers automated face, voice, and screen-time intelligence for unscripted television, powering catalog metadata, commerce, and fan engagement flows. This PRD captures the CI/promotion process, phased roadmap, risks, acceptance criteria, future enhancements, and key references.

---

## 11. CI / Promotion Flow
1. Develop in `FEATURES/<name>/` (`src/`, `tests/`, `docs/`, `TODO.md`).
2. CI enforces:
   - 30-day TTL
   - Tests + docs present
   - No imports from `FEATURES/**` in production
3. Promote via `tools/promote-feature.py`.
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

All acceptance criteria are **defined and enforced** in [ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md). High-level requirements:

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
- ✅ All artifacts validate against schemas ([docs/reference/schemas/artifacts_schemas.md](docs/reference/schemas/artifacts_schemas.md))
- ✅ Integration tests pass with metric assertions ([ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md) section 3)
- ✅ All configs documented ([CONFIG_GUIDE.md](CONFIG_GUIDE.md))
- ✅ Performance profiles verified on CPU and GPU
- ✅ Guardrails trigger warnings when metrics exceed thresholds

**For complete acceptance criteria, thresholds, and verification methods**, see [ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md).

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
- [README.md](README.md) — Project overview and quick start
- [SETUP.md](SETUP.md) — Installation, artifact pipeline, hardware requirements
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) — Configuration quick reference
- [ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md) — QA checklist with metrics & thresholds
- [SOLUTION_ARCHITECTURE.md](SOLUTION_ARCHITECTURE.md) — System architecture diagram
- [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md) — Repository layout
- [FEATURES_GUIDE.md](FEATURES_GUIDE.md) — Feature promotion policy

### Pipeline Documentation
- [docs/pipeline/overview.md](docs/pipeline/overview.md) — End-to-end pipeline architecture
- [docs/pipeline/detect_track_stage.md](docs/pipeline/detect_track_stage.md) — Detection & tracking
- [docs/pipeline/faces_embed_stage.md](docs/pipeline/faces_embed_stage.md) — Face sampling & embedding
- [docs/pipeline/clustering_stage.md](docs/pipeline/clustering_stage.md) — Identity clustering
- [docs/pipeline/cleanup_stage.md](docs/pipeline/cleanup_stage.md) — Outlier removal & post-processing

### Configuration & Tuning
- [docs/reference/config/pipeline_configs.md](docs/reference/config/pipeline_configs.md) — Complete config parameters
- [docs/ops/performance_tuning_faces_pipeline.md](docs/ops/performance_tuning_faces_pipeline.md) — Speed vs accuracy tuning
- [docs/ops/monitoring_logging_faces_pipeline.md](docs/ops/monitoring_logging_faces_pipeline.md) — Logging & debugging

### Schemas & Metrics
- [docs/reference/schemas/artifacts_schemas.md](docs/reference/schemas/artifacts_schemas.md) — All artifact schemas
- [docs/reference/schemas/identities_v1_spec.md](docs/reference/schemas/identities_v1_spec.md) — Identity cluster schema
- [docs/reference/metrics/derived_metrics.md](docs/reference/metrics/derived_metrics.md) — Calculated metrics & guardrails

### Operations
- [docs/ops/episode_cleanup.md](docs/ops/episode_cleanup.md) — Cleanup workflow and configuration

---

**Approved by:** TRR Engineering / Analytics / Design / Automation leads
**Next review:** after Phase 1 completion
