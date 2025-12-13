# Screenalytics Documentation

> **This is the authoritative documentation entry point.** All canonical docs live under `docs/`.
> Archived/superseded docs are kept in `docs/_archive/` for historical reference only.

## Start Here

- **Getting started:** [SETUP.md](../SETUP.md)
- **Pipeline overview:** [docs/pipeline/overview.md](pipeline/overview.md)
- **Workspace UI (Streamlit):** [apps/workspace-ui/pages/](../apps/workspace-ui/pages/)
- **API reference:** [docs/reference/api.md](reference/api.md)

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
- **Reference**
  - [docs/reference/config/pipeline_configs.md](reference/config/pipeline_configs.md)
  - [docs/reference/artifacts_faces_tracks_identities.md](reference/artifacts_faces_tracks_identities.md)
  - [docs/reference/facebank.md](reference/facebank.md)

## CI/Quality Gates

- [ACCEPTANCE_MATRIX.md](../ACCEPTANCE_MATRIX.md) remains at the repo root because CI/tests reference it directly.

## Archived / Non‑Canonical

- [docs/_archive/](./_archive/) contains deprecated or superseded documents kept for historical context.
