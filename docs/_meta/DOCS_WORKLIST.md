# Docs Worklist

Consolidated index of unfinished documentation across `docs/plans/` and `docs/pipeline/`.

Last updated: 2025-12-15

---

## Status Legend

| Status | Meaning |
|--------|---------|
| `in_progress` | Active work, not yet complete |
| `draft` | Ideas/proposals not yet approved or started |
| `outdated` | Superseded or no longer accurate |
| `planned` | Documented but no implementation yet |
| `partial` | Partially implemented |

---

## docs/plans/ - Incomplete Items

### In Progress

| Path | Description | Next Steps | Feature Tags |
|------|-------------|------------|--------------|
| `docs/plans/in_progress/feature_face_alignment_fan_luvli_3ddfa.md` | Face Alignment (FAN, LUVLi, 3DDFA_V2) roadmap | Phase B: LUVLi integration; Phase C: 3DDFA_V2 | `face_alignment` |
| `docs/plans/in_progress/feature_arcface_tensorrt_onnxruntime.md` | ArcFace TensorRT + ONNXRuntime engines | Real GPU testing; registry/service integration | `arcface_tensorrt` |
| `docs/plans/in_progress/feature_body_tracking_reid_fusion.md` | Body Tracking + Person Re-ID + Track Fusion | Dependency hardening; main pipeline integration | `body_tracking` |
| `docs/plans/in_progress/feature_mesh_and_advanced_visibility.md` | Face Mesh + Advanced Visibility Analytics | Create src/; wire visibility labeling; UI surfacing | `vision_analytics` |
| `docs/plans/in_progress/WEB_APP_MIGRATION_PLAN.md` | React/Next.js web app migration | Continue phase implementation | `web_migration` |
| `docs/plans/in_progress/infra/screenalytics_deploy_plan.md` | Deployment infrastructure plan | Finalize staging/prod setup | `infra` |
| `docs/plans/in_progress/product/prd.md` | Product Requirements Document | Ongoing feature prioritization | `product` |

### Draft

| Path | Description | Next Steps | Feature Tags |
|------|-------------|------------|--------------|
| `docs/plans/draft/ideas/FACE_REVIEW_SUGGESTIONS.md` | Face review suggestions feature ideas | Review and approve or archive | `ui` |
| `docs/plans/draft/ideas/SINGLETONS_PLAN.md` | Singleton handling improvements | Review and approve or archive | `clustering` |

### Outdated

| Path | Description | Replacement | Notes |
|------|-------------|-------------|-------|
| `docs/plans/outdated/detect_harvest_findings.md` | Early detect/harvest review notes | `docs/pipeline/detect_track_faces.md`, `docs/pipeline/faces_harvest.md` | Historical only |

---

## docs/pipeline/ - Incomplete Items

### TBD/TODO Markers

| Path | Issue | Next Steps |
|------|-------|------------|
| `docs/pipeline/cluster_identities.md:90` | "TBD: `config/pipeline/recognition.yaml`" | Create recognition.yaml or update doc to reflect actual config location |
| `docs/pipeline/overview.md:158` | "TBD" for recognition config reference | Update to point to actual config file |

### Documents Needing Review

| Path | Status | Notes |
|------|--------|-------|
| `docs/pipeline/screentime_analytics_optimization.md` | Complete | Contains vision_analytics references - verify alignment with feature status |
| `docs/pipeline/tracking/tracking-accuracy-tuning.md` | Complete | Verify thresholds match current config |
| `docs/pipeline/tracking/tracking-strict-config.md` | Complete | Verify config file references |

---

## FEATURES/ - In Progress Items

| Path | Description | Status |
|------|-------------|--------|
| `FEATURES/face_alignment/TODO.md` | FAN/LUVLi/3DDFA tracking | in_progress |
| `FEATURES/body_tracking/TODO.md` | Body tracking feature | in_progress |
| `FEATURES/arcface_tensorrt/TODO.md` | TensorRT embedding engine | in_progress |
| `FEATURES/vision_analytics/TODO.md` | Advanced visibility analytics | in_progress |
| `FEATURES/embedding_engines/TODO.md` | Embedding engine abstraction | in_progress |
| `FEATURES/tracking/TODO.md` | Tracking improvements | in_progress |
| `FEATURES/detection/TODO.md` | Detection enhancements | in_progress |
| `FEATURES/identity/TODO.md` | Identity/clustering | in_progress |
| `FEATURES/shows-people-ui/TODO.md` | Shows/people UI | in_progress |
| `FEATURES/export-results/TODO.md` | Results export feature | in_progress |
| `FEATURES/av-fusion/TODO.md` | Audio-visual fusion | in_progress |
| `FEATURES/agents-mcps/TODO.md` | Agent MCP integration | in_progress |

---

## Deferred Work (Follow-up Tasks)

These items were identified but not addressed in this PR:

1. **Recognition config file**: Create `config/pipeline/recognition.yaml` or update docs to point to actual location
2. **Feature status alignment**: Verify all docs/pipeline references to features match actual implementation status
3. **Directory moves**: Some docs are flagged for relocation in `docs_triage_report.md` - execute moves when safe

---

## Related Resources

- [docs/_meta/STATUS.md](STATUS.md) - Status taxonomy definitions
- [docs/_meta/docs_catalog.json](docs_catalog.json) - Machine-readable docs inventory
- [docs/_meta/docs_triage_report.md](docs_triage_report.md) - Full triage with actions

---

## How to Update This File

When working on documentation:

1. Mark items as `complete` when finished
2. Add new incomplete items as discovered
3. Update "Last updated" date
4. Keep feature tags consistent with docs_catalog.json
