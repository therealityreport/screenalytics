# Docs Triage Report

This report inventories tracked Markdown docs and records the intended action to reduce noise and keep docs operationally useful.
See `docs/_meta/STATUS.md` for status meanings.

## Known Feature Implementation Status (current)

- `face_alignment`: **partial** — Phase A (FAN 2D) complete; Phase B (LUVLi) heuristic stub; Phase C (3DDFA_V2) not started; pending pipeline integration + model-based quality.
- `arcface_tensorrt`: **scaffold only** — engine/inference/comparison scaffolds; integration future; pending real GPU testing + eval.
- `vision_analytics`: **not started** — docs only; no implementation wiring yet.

## Status Counts

- `complete`: 98
- `in_progress`: 28
- `draft`: 2
- `outdated`: 1
- `archive`: 8

## Inventory

| Path | Type | Status | Action | Replacement Path | Notes |
|---|---|---|---|---|---|
| `.claude/commands/episode-status.md` | `reference` | `complete` | `keep` | `` |  |
| `.claude/commands/test.md` | `reference` | `complete` | `keep` | `` |  |
| `.claude/skills/body-tracking/SKILL.md` | `reference` | `complete` | `keep` | `` |  |
| `.claude/skills/cluster-quality/SKILL.md` | `reference` | `complete` | `keep` | `` |  |
| `.claude/skills/embedding-engine/SKILL.md` | `reference` | `complete` | `keep` | `` |  |
| `.claude/skills/face-alignment/SKILL.md` | `reference` | `complete` | `keep` | `` |  |
| `.claude/skills/faces-review-ux/SKILL.md` | `reference` | `complete` | `keep` | `` |  |
| `.claude/skills/pipeline-debug/SKILL.md` | `reference` | `complete` | `keep` | `` |  |
| `.claude/skills/pipeline-insights/SKILL.md` | `reference` | `complete` | `keep` | `` |  |
| `.claude/skills/qa-acceptance/SKILL.md` | `reference` | `complete` | `keep` | `` |  |
| `.claude/skills/storage-health/SKILL.md` | `reference` | `complete` | `keep` | `` |  |
| `.claude/skills/vision-analytics/SKILL.md` | `reference` | `complete` | `keep` | `` |  |
| `.github/CONTRIBUTING.md` | `reference` | `complete` | `keep` | `` |  |
| `.github/pull_request_template.md` | `reference` | `complete` | `keep` | `` |  |
| `ACCEPTANCE_MATRIX.md` | `reference` | `complete` | `keep` | `` |  |
| `AGENTS.md` | `reference` | `complete` | `keep` | `` |  |
| `CLAUDE.md` | `howto` | `complete` | `keep` | `` |  |
| `CONTRIBUTING.md` | `howto` | `complete` | `keep` | `` |  |
| `FEATURES/agents-mcps/TODO.md` | `plan` | `in_progress` | `keep` | `` |  |
| `FEATURES/agents-mcps/docs/PROGRESS.md` | `features` | `in_progress` | `keep` | `` |  |
| `FEATURES/arcface_tensorrt/TODO.md` | `plan` | `in_progress` | `keep` | `` |  |
| `FEATURES/arcface_tensorrt/docs/README.md` | `features` | `in_progress` | `keep` | `` |  |
| `FEATURES/av-fusion/TODO.md` | `plan` | `in_progress` | `keep` | `` |  |
| `FEATURES/body_tracking/TODO.md` | `plan` | `in_progress` | `keep` | `` |  |
| `FEATURES/body_tracking/docs/README.md` | `features` | `in_progress` | `keep` | `` |  |
| `FEATURES/detection/TODO.md` | `plan` | `in_progress` | `keep` | `` |  |
| `FEATURES/detection/docs/PROGRESS.md` | `features` | `in_progress` | `keep` | `` |  |
| `FEATURES/embedding_engines/TODO.md` | `plan` | `in_progress` | `keep` | `` |  |
| `FEATURES/embedding_engines/docs/README.md` | `features` | `in_progress` | `keep` | `` |  |
| `FEATURES/export-results/TODO.md` | `plan` | `in_progress` | `keep` | `` |  |
| `FEATURES/face_alignment/TODO.md` | `plan` | `in_progress` | `keep` | `` |  |
| `FEATURES/face_alignment/docs/README.md` | `features` | `in_progress` | `keep` | `` |  |
| `FEATURES/identity/TODO.md` | `plan` | `in_progress` | `keep` | `` |  |
| `FEATURES/shows-people-ui/TODO.md` | `plan` | `in_progress` | `keep` | `` |  |
| `FEATURES/tracking/TODO.md` | `plan` | `in_progress` | `keep` | `` |  |
| `FEATURES/tracking/docs/PROGRESS.md` | `features` | `in_progress` | `keep` | `` |  |
| `FEATURES/vision_analytics/TODO.md` | `plan` | `in_progress` | `keep` | `` |  |
| `FEATURES/vision_analytics/docs/README.md` | `features` | `in_progress` | `keep` | `` |  |
| `README.md` | `canonical` | `complete` | `keep` | `` |  |
| `SETUP.md` | `howto` | `complete` | `keep` | `` |  |
| `agents/AGENTS.md` | `reference` | `complete` | `keep` | `` |  |
| `apps/api/CLAUDE.md` | `reference` | `complete` | `keep` | `` |  |
| `apps/workspace-ui/CLAUDE.md` | `reference` | `complete` | `keep` | `` |  |
| `apps/workspace-ui/assets/fonts/README.md` | `reference` | `complete` | `keep` | `` |  |
| `docs/BRANCHING_STRATEGY.md` | `reference` | `complete` | `move` | `docs/reference/branching/BRANCHING_STRATEGY.md` |  |
| `docs/CLAUDE-SETUP.md` | `reference` | `complete` | `move` | `docs/reference/assistants/CLAUDE-SETUP.md` |  |
| `docs/CODEX_SLACK.md` | `reference` | `complete` | `move` | `docs/reference/assistants/CODEX_SLACK.md` |  |
| `docs/DEPLOYMENT_RENDER.md` | `ops` | `complete` | `move` | `docs/ops/deployment/DEPLOYMENT_RENDER.md` |  |
| `docs/README.md` | `canonical` | `complete` | `update` | `` | Update index to match new docs layout + UI surfacing. |
| `docs/WEB_APP_MIGRATION_PLAN.md` | `plan` | `in_progress` | `move` | `docs/plans/in_progress/WEB_APP_MIGRATION_PLAN.md` |  |
| `docs/_archive/new-features/masterWEBtodolist.md` | `legacy` | `archive` | `keep` | `` |  |
| `docs/_archive/pipeline/audio_pipeline_pyannote_legacy.md` | `legacy` | `archive` | `keep` | `` |  |
| `docs/_archive/root_docs/CONFIG_GUIDE.md` | `legacy` | `archive` | `keep` | `` |  |
| `docs/_archive/root_docs/DIRECTORY_STRUCTURE.md` | `legacy` | `archive` | `keep` | `` |  |
| `docs/_archive/root_docs/MANIFEST.md` | `legacy` | `archive` | `keep` | `` |  |
| `docs/_archive/root_docs/QUICK_START_GUIDE.md` | `legacy` | `archive` | `keep` | `` |  |
| `docs/_archive/root_docs/SOLUTION_ARCHITECTURE.md` | `legacy` | `archive` | `keep` | `` |  |
| `docs/_archive/root_docs/migration_plan.md` | `legacy` | `archive` | `keep` | `` |  |
| `docs/architecture/directory_structure.md` | `architecture` | `complete` | `keep` | `` |  |
| `docs/architecture/solution_architecture.md` | `architecture` | `complete` | `keep` | `` |  |
| `docs/audio/diarization_manifest.md` | `pipeline` | `complete` | `move` | `docs/pipeline/audio/diarization_manifest.md` |  |
| `docs/audio/overview.md` | `pipeline` | `complete` | `move` | `docs/pipeline/audio/overview.md` |  |
| `docs/audit/vision_alignment_body_tracking_status.md` | `audit` | `complete` | `move` | `docs/plans/complete/audit/vision_alignment_body_tracking_status.md` | Ensure feature status claims match current implementation. |
| `docs/branching/2025-12-main-promotion-plan.md` | `plan` | `complete` | `move` | `docs/plans/complete/branching/2025-12-main-promotion-plan.md` |  |
| `docs/changes/2025-11-11-api-health-and-ui-preflight.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-api-health-and-ui-preflight.md` |  |
| `docs/changes/2025-11-11-auto-fps.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-auto-fps.md` |  |
| `docs/changes/2025-11-11-deps-split-and-storage-import.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-deps-split-and-storage-import.md` |  |
| `docs/changes/2025-11-11-detect-track-device-selection.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-detect-track-device-selection.md` |  |
| `docs/changes/2025-11-11-detect-track-real.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-detect-track-real.md` |  |
| `docs/changes/2025-11-11-existing-episode-tools.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-existing-episode-tools.md` |  |
| `docs/changes/2025-11-11-face-tracking-reid.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-face-tracking-reid.md` |  |
| `docs/changes/2025-11-11-faces-pipeline-and-facebank.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-faces-pipeline-and-facebank.md` |  |
| `docs/changes/2025-11-11-jobs-async-progress.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-jobs-async-progress.md` |  |
| `docs/changes/2025-11-11-phase-3A-acceptance.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-phase-3A-acceptance.md` |  |
| `docs/changes/2025-11-11-phase-3A-summary.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-phase-3A-summary.md` |  |
| `docs/changes/2025-11-11-progress-and-artifacts.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-progress-and-artifacts.md` |  |
| `docs/changes/2025-11-11-retinaface-arcface-facebank.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-retinaface-arcface-facebank.md` |  |
| `docs/changes/2025-11-11-s3-bucket-structure.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-s3-bucket-structure.md` |  |
| `docs/changes/2025-11-11-s3-layout-v2.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-s3-layout-v2.md` |  |
| `docs/changes/2025-11-11-s3-single-bucket.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-s3-single-bucket.md` |  |
| `docs/changes/2025-11-11-scene-detect.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-scene-detect.md` |  |
| `docs/changes/2025-11-11-tests-backend-aware-presign.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-tests-backend-aware-presign.md` |  |
| `docs/changes/2025-11-11-tests-green-local.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-tests-green-local.md` |  |
| `docs/changes/2025-11-11-ui-detect-resp-guard.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-ui-detect-resp-guard.md` |  |
| `docs/changes/2025-11-11-ui-multipage.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-ui-multipage.md` |  |
| `docs/changes/2025-11-11-ui-s3-picker.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-ui-s3-picker.md` |  |
| `docs/changes/2025-11-11-ui-shared-state.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-ui-shared-state.md` |  |
| `docs/changes/2025-11-11-ui-syntax-guard-and-run-cmd.md` | `legacy` | `complete` | `move` | `docs/plans/complete/changes/2025-11-11-ui-syntax-guard-and-run-cmd.md` |  |
| `docs/code-updates/IMPLEMENTATION_SUMMARY_nov-18-track-reps.md` | `legacy` | `complete` | `move` | `docs/plans/complete/code-updates/IMPLEMENTATION_SUMMARY_nov-18-track-reps.md` |  |
| `docs/code-updates/dec-03-screen-time-engine-refactor.md` | `legacy` | `complete` | `move` | `docs/plans/complete/code-updates/dec-03-screen-time-engine-refactor.md` |  |
| `docs/code-updates/dec-2025-engine-api-jobs.md` | `legacy` | `complete` | `move` | `docs/plans/complete/code-updates/dec-2025-engine-api-jobs.md` |  |
| `docs/code-updates/nov-17-cluster-min-sim-and-thumbs.md` | `legacy` | `complete` | `move` | `docs/plans/complete/code-updates/nov-17-cluster-min-sim-and-thumbs.md` |  |
| `docs/code-updates/nov-17-detect-track-none-bbox-fix.md` | `legacy` | `complete` | `move` | `docs/plans/complete/code-updates/nov-17-detect-track-none-bbox-fix.md` |  |
| `docs/code-updates/nov-17-episode-detail-detect-track-fallback.md` | `legacy` | `complete` | `move` | `docs/plans/complete/code-updates/nov-17-episode-detail-detect-track-fallback.md` |  |
| `docs/code-updates/nov-17-zero-tracks-and-bytetrack-thresholds.md` | `legacy` | `complete` | `move` | `docs/plans/complete/code-updates/nov-17-zero-tracks-and-bytetrack-thresholds.md` |  |
| `docs/code-updates/nov-18-detect-track-crop-none-guards.md` | `legacy` | `complete` | `move` | `docs/plans/complete/code-updates/nov-18-detect-track-crop-none-guards.md` |  |
| `docs/code-updates/nov-18-health-and-detect-track-diagnostics.md` | `legacy` | `complete` | `move` | `docs/plans/complete/code-updates/nov-18-health-and-detect-track-diagnostics.md` |  |
| `docs/debugging/faces-manifest-crops-mismatch.md` | `ops` | `complete` | `move` | `docs/ops/debugging/faces-manifest-crops-mismatch.md` |  |
| `docs/detect_harvest_findings.md` | `audit` | `outdated` | `move` | `docs/plans/outdated/detect_harvest_findings.md` |  |
| `docs/dev/detect_track_call_chain.md` | `reference` | `complete` | `move` | `docs/reference/dev/detect_track_call_chain.md` |  |
| `docs/dev/detect_track_regression.md` | `reference` | `complete` | `move` | `docs/reference/dev/detect_track_regression.md` |  |
| `docs/dev/episode_detail_detect_faces_cluster_audit.md` | `reference` | `complete` | `move` | `docs/reference/dev/episode_detail_detect_faces_cluster_audit.md` |  |
| `docs/features/feature_sandboxes.md` | `features` | `complete` | `keep` | `` |  |
| `docs/features/vision_alignment_and_body_tracking.md` | `features` | `in_progress` | `update` | `` | Ensure LUVLi/3DDFA status is described as heuristic stub / not started. |
| `docs/golden-episodes.md` | `reference` | `complete` | `move` | `docs/reference/golden-episodes.md` |  |
| `docs/ideas/FACE_REVIEW_SUGGESTIONS.md` | `plan` | `draft` | `move` | `docs/plans/draft/ideas/FACE_REVIEW_SUGGESTIONS.md` |  |
| `docs/ideas/SINGLETONS_PLAN.md` | `plan` | `draft` | `move` | `docs/plans/draft/ideas/SINGLETONS_PLAN.md` |  |
| `docs/infra/screenalytics_deploy_plan.md` | `plan` | `in_progress` | `move` | `docs/plans/in_progress/infra/screenalytics_deploy_plan.md` |  |
| `docs/ops/ARTIFACTS_STORE.md` | `ops` | `complete` | `keep` | `` |  |
| `docs/ops/TENSORRT_ENGINES.md` | `ops` | `complete` | `keep` | `` |  |
| `docs/ops/execution_mode.md` | `ops` | `complete` | `keep` | `` |  |
| `docs/ops/faces_review_guide.md` | `ops` | `complete` | `keep` | `` |  |
| `docs/ops/hardware_sizing.md` | `ops` | `complete` | `keep` | `` |  |
| `docs/ops/performance_tuning_faces_pipeline.md` | `ops` | `complete` | `keep` | `` |  |
| `docs/ops/troubleshooting_faces_pipeline.md` | `ops` | `complete` | `keep` | `` |  |
| `docs/pipeline/audio_pipeline.md` | `pipeline` | `complete` | `keep` | `` |  |
| `docs/pipeline/cluster_identities.md` | `pipeline` | `complete` | `keep` | `` |  |
| `docs/pipeline/detect_track_faces.md` | `pipeline` | `complete` | `keep` | `` |  |
| `docs/pipeline/episode_cleanup.md` | `pipeline` | `complete` | `keep` | `` |  |
| `docs/pipeline/faces_harvest.md` | `pipeline` | `complete` | `keep` | `` |  |
| `docs/pipeline/overview.md` | `pipeline` | `complete` | `keep` | `` |  |
| `docs/pipeline/screentime_analytics_optimization.md` | `pipeline` | `complete` | `keep` | `` |  |
| `docs/product/prd.md` | `plan` | `in_progress` | `move` | `docs/plans/in_progress/product/prd.md` |  |
| `docs/reference/api.md` | `reference` | `complete` | `keep` | `` |  |
| `docs/reference/artifacts_faces_tracks_identities.md` | `reference` | `complete` | `keep` | `` |  |
| `docs/reference/config/pipeline_configs.md` | `reference` | `complete` | `keep` | `` |  |
| `docs/reference/data_schema.md` | `reference` | `complete` | `keep` | `` |  |
| `docs/reference/facebank.md` | `reference` | `complete` | `keep` | `` |  |
| `docs/similarity-scores-guide.md` | `reference` | `complete` | `move` | `docs/reference/similarity-scores-guide.md` |  |
| `docs/todo/feature_arcface_tensorrt_onnxruntime.md` | `plan` | `in_progress` | `move` | `docs/plans/in_progress/feature_arcface_tensorrt_onnxruntime.md` |  |
| `docs/todo/feature_body_tracking_reid_fusion.md` | `plan` | `in_progress` | `move` | `docs/plans/in_progress/feature_body_tracking_reid_fusion.md` |  |
| `docs/todo/feature_face_alignment_fan_luvli_3ddfa.md` | `plan` | `in_progress` | `move` | `docs/plans/in_progress/feature_face_alignment_fan_luvli_3ddfa.md` |  |
| `docs/todo/feature_mesh_and_advanced_visibility.md` | `plan` | `in_progress` | `move` | `docs/plans/in_progress/feature_mesh_and_advanced_visibility.md` |  |
| `docs/tracking-accuracy-tuning.md` | `pipeline` | `complete` | `move` | `docs/pipeline/tracking/tracking-accuracy-tuning.md` |  |
| `docs/tracking-strict-config.md` | `pipeline` | `complete` | `move` | `docs/pipeline/tracking/tracking-strict-config.md` |  |
| `infra/README.md` | `ops` | `complete` | `keep` | `` |  |
| `tests/README.md` | `reference` | `complete` | `keep` | `` |  |
