# Screenalytics Documentation

> **This is the authoritative documentation entry point.** Canonical docs live under `docs/`.
> Historical/low-signal docs live under `docs/plans/` (by status) and `docs/_archive/`.

## Start Here

- **Getting started:** [SETUP.md](../SETUP.md)
- **Pipeline overview:** [docs/pipeline/overview.md](pipeline/overview.md)
- **Workspace UI (Streamlit):** [apps/workspace-ui/pages/](../apps/workspace-ui/pages/)
- **API reference:** [docs/reference/api.md](reference/api.md)

## Docs Layout (high signal → low signal)

- `docs/_meta/` — doc status taxonomy + triage report + machine-readable catalog (drives UI)
- `docs/architecture/` — architecture and repo layout
- `docs/pipeline/` — pipeline phases and manifests
- `docs/ops/` — runbooks and troubleshooting
- `docs/reference/` — stable reference docs (branching, configs, schemas, developer notes)
- `docs/features/` — feature overviews + sandbox expectations
- `docs/plans/` — non-canonical docs by status (`in_progress`, `draft`, `complete`, `outdated`, `superseded`)
- `docs/_archive/` — intentionally archived legacy docs (keep for history only)

## Key Docs

- **Architecture**
  - [docs/architecture/solution_architecture.md](architecture/solution_architecture.md)
  - [docs/architecture/directory_structure.md](architecture/directory_structure.md)
- **Pipeline**
  - [docs/pipeline/detect_track_faces.md](pipeline/detect_track_faces.md)
  - [docs/pipeline/faces_harvest.md](pipeline/faces_harvest.md)
  - [docs/pipeline/cluster_identities.md](pipeline/cluster_identities.md)
  - [docs/pipeline/episode_cleanup.md](pipeline/episode_cleanup.md)
  - [docs/pipeline/audio_pipeline.md](pipeline/audio_pipeline.md)
  - [docs/pipeline/screentime_analytics_optimization.md](pipeline/screentime_analytics_optimization.md)
- **Operations**
  - [docs/ops/performance_tuning_faces_pipeline.md](ops/performance_tuning_faces_pipeline.md)
  - [docs/ops/troubleshooting_faces_pipeline.md](ops/troubleshooting_faces_pipeline.md)
  - [docs/ops/hardware_sizing.md](ops/hardware_sizing.md)
  - [docs/ops/deployment/DEPLOYMENT_RENDER.md](ops/deployment/DEPLOYMENT_RENDER.md)
- **Reference**
  - [docs/reference/branching/BRANCHING_STRATEGY.md](reference/branching/BRANCHING_STRATEGY.md)
  - [docs/reference/dev/detect_track_call_chain.md](reference/dev/detect_track_call_chain.md)
  - [docs/reference/config/pipeline_configs.md](reference/config/pipeline_configs.md)
  - [docs/reference/artifacts_faces_tracks_identities.md](reference/artifacts_faces_tracks_identities.md)
  - [docs/reference/facebank.md](reference/facebank.md)
  - [docs/reference/golden-episodes.md](reference/golden-episodes.md)
  - [docs/reference/similarity-scores-guide.md](reference/similarity-scores-guide.md)

## In Progress (TODO / plans)

- [Face Alignment roadmap](plans/in_progress/feature_face_alignment_fan_luvli_3ddfa.md)
- [ArcFace TensorRT / ONNXRuntime roadmap](plans/in_progress/feature_arcface_tensorrt_onnxruntime.md)
- [Body Tracking + Re-ID + Fusion roadmap](plans/in_progress/feature_body_tracking_reid_fusion.md)
- [Mesh + advanced visibility roadmap](plans/in_progress/feature_mesh_and_advanced_visibility.md)

## Status + Catalog

- Doc status meanings live in [docs/_meta/STATUS.md](_meta/STATUS.md).
- Doc triage decisions live in [docs/_meta/docs_triage_report.md](_meta/docs_triage_report.md).
- The machine-readable doc index lives in `docs/_meta/docs_catalog.json` (used by the Streamlit docs dashboard).

## CI/Quality Gates

- [ACCEPTANCE_MATRIX.md](../ACCEPTANCE_MATRIX.md) remains at the repo root because CI/tests reference it directly.

## Archived / Non‑Canonical

- [docs/plans/](./plans/) contains drafts, in-progress docs, and historical change notes grouped by status.
- [docs/_archive/](./_archive/) contains intentionally archived documents kept for historical context.
