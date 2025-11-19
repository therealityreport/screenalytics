Status: PLANNING
Owner: TBA
Goal: Redefine tracks from "detection sequences" to "scene-scoped person appearances"
TTL: 30 days from start
Created: 2025-11-19

## Overview
Transform the track concept from continuous detection sequences to semantic scene-scoped person appearances. This reduces track fragmentation from ~4,352 to ~100 per episode.

## Implementation Checklist

### Phase 1: Semantic Scene Detection (Weeks 1-2)
- [ ] Implement `detect_semantic_scenes()` in episode_run.py
  - [ ] Shot boundary detection (existing PySceneDetect)
  - [ ] Temporal clustering heuristics (time gaps, rapid cuts)
  - [ ] Confessional detection logic
- [ ] Create `config/pipeline/scene_segmentation.yaml`
- [ ] Add scene table to Postgres schema
  - [ ] Write migration: `db/migrations/003_add_scene_table.sql`
  - [ ] Test migration up/down
- [ ] Write tests
  - [ ] `test_detect_semantic_scenes.py`
  - [ ] `test_confessional_detection.py`
  - [ ] `test_scene_boundary_heuristics.py`
- [ ] Add API endpoint: `GET /api/episodes/{ep_id}/scenes`

**Success Criteria**:
- Scene count ≈ 15-30 per episode (vs. 200+ shot cuts)
- 90%+ confessionals correctly identified
- Dialogue scenes grouped despite camera cuts

---

### Phase 2: Shot-Level Track Annotation (Week 3)
- [ ] Update `TrackRecorder` class to accept scene metadata
- [ ] Modify `track_v1` schema to include `scene_id`
- [ ] Rename internal artifact: `tracks.jsonl` → `shot_tracks.jsonl`
- [ ] Update artifact resolver paths in `packages/py-screenalytics/artifacts.py`
- [ ] Add backwards compatibility handling
- [ ] Write tests
  - [ ] `test_track_scene_annotation.py`
  - [ ] `test_backwards_compat_no_scenes.py`

**Success Criteria**:
- All tracks have valid `scene_id`
- Track termination at scene boundaries
- No regression in tracking quality

---

### Phase 3: Scene-Scoped Track Merging (Weeks 4-5)
- [ ] Implement appearance clustering module
  - [ ] `cluster_by_appearance()` - hierarchical clustering
  - [ ] `compute_similarity_matrix()` - pairwise ArcFace cosine similarity
  - [ ] `get_representative_embedding()` - median/mean strategy
- [ ] Implement track merging module
  - [ ] `merge_cluster_into_track()` - aggregate shot tracks
  - [ ] `compute_appearance_segments()` - visible intervals
  - [ ] `aggregate_quality_stats()` - quality metrics
- [ ] Create `tools/merge_scene_tracks.py` script
- [ ] Create `config/pipeline/track_merging.yaml`
- [ ] Update `track` table schema
  - [ ] Migration: `db/migrations/004_scene_scoped_track_columns.sql`
  - [ ] Add columns: `scene_id`, `appearance_segments`, `shot_track_ids`
- [ ] Write tests
  - [ ] `test_cluster_by_appearance.py`
  - [ ] `test_merge_cluster_into_track.py`
  - [ ] `test_edge_cases.py` (twins, leave/return, confessionals)
  - [ ] `test_track_merging_pipeline.py`

**Success Criteria**:
- 95%+ track reduction (4,352 → ~100)
- < 2% false merge rate
- < 5% identity drift rate
- Handle edge cases correctly

---

### Phase 4: API & UI Updates (Week 6)
- [ ] Update API routes
  - [ ] Return SSTs by default in `GET /api/episodes/{ep_id}/tracks`
  - [ ] Add `GET /api/scenes/{scene_id}/tracks`
  - [ ] Add query param: `?track_version=v1` for backwards compat
  - [ ] Version API: `/api/v2/tracks`
- [ ] Update screen time calculation to use SSTs
- [ ] Update UI components
  - [ ] Scene boundary visualization in video player
  - [ ] Scene-grouped track display
  - [ ] Scene metadata (location, type, duration)
- [ ] Write API migration guide
- [ ] Update OpenAPI schema

**Success Criteria**:
- API returns SSTs by default
- UI shows scene-grouped tracks
- Backwards compatibility maintained
- Migration guide published

---

### Phase 5: Testing & Validation (Week 7)
- [ ] Integration tests
  - [ ] `test_end_to_end_scene_tracks.py`
  - [ ] `test_scene_api_integration.py`
  - [ ] `test_ui_scene_display.py`
- [ ] Regression tests
  - [ ] `test_existing_episodes_process.py`
  - [ ] `test_screen_time_consistency.py`
  - [ ] `test_backwards_compat_full.py`
- [ ] Performance benchmarks
  - [ ] Measure processing time increase
  - [ ] Memory usage profiling
  - [ ] Database query performance
- [ ] Quality metrics
  - [ ] Manual review: 10 episodes, verify scene boundaries
  - [ ] False merge rate measurement
  - [ ] Identity drift rate measurement
- [ ] Documentation
  - [ ] Update DATA_SCHEMA.md
  - [ ] Update API.md
  - [ ] Write MIGRATION_GUIDE.md
  - [ ] Update README.md

**Success Criteria**:
- All tests pass
- < 20% processing time increase
- < 2% false merge rate
- Documentation complete

---

## Promotion Checklist
- [ ] All Phase 1-5 tasks complete
- [ ] Test coverage > 85%
- [ ] Performance benchmarks meet targets
- [ ] API migration guide published
- [ ] CI green
- [ ] Code review approved
- [ ] Update ACCEPTANCE_MATRIX.md
- [ ] Run promotion script: `tools/promote-feature.py scene-scoped-tracks`

---

## Key Files & Artifacts

### Configuration
- `config/pipeline/scene_segmentation.yaml` - Scene detection config
- `config/pipeline/track_merging.yaml` - Track merging config

### Database
- `db/migrations/003_add_scene_table.sql`
- `db/migrations/004_scene_scoped_track_columns.sql`

### Core Implementation
- `tools/episode_run.py` - Modified for scene detection
- `tools/merge_scene_tracks.py` - NEW: Track merging script
- `FEATURES/scene-scoped-tracks/src/scene_detector.py` - NEW: Scene detection module
- `FEATURES/scene-scoped-tracks/src/track_merger.py` - NEW: Track merging module

### Tests
- `tests/ml/test_semantic_scene_detection.py`
- `tests/ml/test_appearance_clustering.py`
- `tests/api/test_scene_tracks_api.py`
- `tests/FEATURES/scene-scoped-tracks/`

### Documentation
- `FEATURES/scene-scoped-tracks/FEATURE.md` - This document
- `FEATURES/scene-scoped-tracks/docs/MIGRATION_GUIDE.md`
- `FEATURES/scene-scoped-tracks/docs/PROGRESS.md`

---

## Risk Register

| Risk | Severity | Mitigation Status |
|------|----------|-------------------|
| Semantic scene detection inaccurate | HIGH | Tunable thresholds, manual override UI |
| False merges (different people) | CRITICAL | Conservative thresholds, spatial constraints |
| Processing time increase > 20% | MEDIUM | Optimize clustering, parallel processing |
| API breaking changes impact users | HIGH | Versioned API, 3-month deprecation |
| Confessional misclassification | MEDIUM | Heuristic tuning, training data |
| Backwards compatibility bugs | MEDIUM | Comprehensive testing, gradual rollout |

---

## Questions for Review
1. Scene definition priority: Location vs. activity vs. narrative?
2. API versioning strategy: Hard cutover or permanent dual support?
3. Manual override UI for scene boundaries?
4. Performance budget: Max acceptable processing time increase?
5. Quality thresholds: Acceptable false merge rate for v1?

---

## Notes & Decisions
- **2025-11-19**: Feature proposal drafted
- Decision needed: Confessional handling (separate vs. merge)
- Decision needed: API versioning strategy
