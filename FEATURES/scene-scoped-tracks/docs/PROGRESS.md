# Progress Tracking: Scene-Scoped Tracks

**Feature**: Scene-Scoped Tracks  
**Status**: PLANNING  
**Owner**: TBA  
**Started**: 2025-11-19  
**Target Completion**: TBD (7 weeks from start)

---

## Progress Overview

| Phase | Status | Start Date | End Date | Progress |
|-------|--------|------------|----------|----------|
| Phase 1: Semantic Scene Detection | 🔵 Not Started | TBD | TBD | 0% |
| Phase 2: Shot-Level Track Annotation | 🔵 Not Started | TBD | TBD | 0% |
| Phase 3: Scene-Scoped Track Merging | 🔵 Not Started | TBD | TBD | 0% |
| Phase 4: API & UI Updates | 🔵 Not Started | TBD | TBD | 0% |
| Phase 5: Testing & Validation | 🔵 Not Started | TBD | TBD | 0% |

**Legend**: 🔵 Not Started | 🟡 In Progress | 🟢 Complete | 🔴 Blocked

---

## Phase 1: Semantic Scene Detection (Weeks 1-2)

### Status: 🔵 Not Started
**Target**: Produce `scenes.jsonl` for episodes

### Tasks
- [ ] Implement `SemanticSceneDetector` class
  - [ ] `_cuts_to_shots()` method
  - [ ] `_cluster_shots_into_scenes()` method
  - [ ] `_classify_scene_types()` method
- [ ] Integrate with `episode_run.py`
  - [ ] Import scene detector
  - [ ] Call after PySceneDetect
  - [ ] Write scenes artifact
- [ ] Create `config/pipeline/scene_segmentation.yaml`
- [ ] Write database migration `003_add_scene_table.sql`
  - [ ] Test migration up
  - [ ] Test migration down
- [ ] Add API endpoint `GET /api/episodes/{ep_id}/scenes`
- [ ] Write tests
  - [ ] `test_cluster_shots_rapid_cuts()`
  - [ ] `test_cluster_shots_temporal_gap()`
  - [ ] `test_classify_confessional()`
  - [ ] `test_scene_detection_integration()`

### Blockers
- None

### Notes
- Need to decide: scene classification heuristics vs. ML model?
- Confessional detection may need tuning for different shows

---

## Phase 2: Shot-Level Track Annotation (Week 3)

### Status: 🔵 Not Started
**Target**: Add `scene_id` to tracks

### Tasks
- [ ] Modify `TrackRecorder` class
  - [ ] Accept scenes in constructor
  - [ ] Implement `get_current_scene_id()` method
  - [ ] Add `scene_id` to track records
- [ ] Update artifact resolver
  - [ ] Add `shot_tracks` path
  - [ ] Keep `tracks` for scene-scoped tracks
- [ ] Define `ShotTrackV1` schema
- [ ] Write tests
  - [ ] `test_track_has_scene_id()`
  - [ ] `test_track_termination_at_scene_boundary()`
  - [ ] `test_backwards_compat_no_scenes()`

### Blockers
- Depends on Phase 1 completion

### Notes
- Minimal changes to existing tracking logic
- Focus on backwards compatibility

---

## Phase 3: Scene-Scoped Track Merging (Weeks 4-5)

### Status: 🔵 Not Started
**Target**: Merge shot tracks into scene tracks

### Tasks
- [ ] Implement `appearance_clustering.py`
  - [ ] `cluster_by_appearance()` function
  - [ ] `compute_cosine_similarity()` function
  - [ ] `get_representative_embedding()` function
- [ ] Implement `track_merger.py`
  - [ ] `merge_cluster_into_track()` function
  - [ ] `compute_appearance_segments()` helper
  - [ ] `aggregate_quality_stats()` helper
- [ ] Create `tools/merge_scene_tracks.py` script
- [ ] Create `config/pipeline/track_merging.yaml`
- [ ] Write database migration `004_scene_scoped_track_columns.sql`
  - [ ] Test migration up
  - [ ] Test migration down
- [ ] Write tests
  - [ ] `test_cluster_by_appearance()`
  - [ ] `test_merge_cluster_into_track()`
  - [ ] `test_edge_case_person_leaves_returns()`
  - [ ] `test_edge_case_twins()`
  - [ ] `test_edge_case_confessionals()`
  - [ ] `test_track_merging_pipeline()`

### Blockers
- Depends on Phase 2 completion
- Requires ArcFace embeddings to be available

### Notes
- This is the most complex phase
- Need to carefully tune similarity threshold
- Edge cases require extensive testing

---

## Phase 4: API & UI Updates (Week 6)

### Status: 🔵 Not Started
**Target**: Expose SSTs in API/UI

### Tasks
- [ ] Update API routes
  - [ ] Modify `GET /api/episodes/{ep_id}/tracks` (default to v2)
  - [ ] Add `?track_version=v1` query param
  - [ ] Add `GET /api/scenes/{scene_id}/tracks`
  - [ ] Add `GET /api/v2/tracks` explicit endpoint
- [ ] Update screen time calculation
  - [ ] Use `total_visible_s` instead of `end_s - start_s`
  - [ ] Handle appearance segments
- [ ] Update UI components
  - [ ] Scene boundary visualization in video player
  - [ ] Scene-grouped track display
  - [ ] Scene metadata panel
- [ ] Write API migration guide
- [ ] Update OpenAPI schema

### Blockers
- Depends on Phase 3 completion

### Notes
- API versioning strategy needs product decision
- UI mockups needed for scene visualization

---

## Phase 5: Testing & Validation (Week 7)

### Status: 🔵 Not Started
**Target**: Comprehensive testing and QA

### Tasks
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
  - [ ] Manual review: 10 episodes
  - [ ] False merge rate measurement
  - [ ] Identity drift rate measurement
- [ ] Documentation
  - [ ] Update DATA_SCHEMA.md
  - [ ] Update API.md
  - [ ] Write MIGRATION_GUIDE.md
  - [ ] Update README.md
- [ ] Promotion
  - [ ] Update ACCEPTANCE_MATRIX.md
  - [ ] Run `tools/promote-feature.py scene-scoped-tracks`
  - [ ] Code review
  - [ ] CI green

### Blockers
- Depends on Phase 4 completion

### Notes
- Quality thresholds must be met before promotion
- Manual review is critical for catching edge cases

---

## Metrics Tracking

### Quantitative Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Track reduction | 95%+ | TBD | 🔵 Not measured |
| False merge rate | < 2% | TBD | 🔵 Not measured |
| Identity drift rate | < 5% | TBD | 🔵 Not measured |
| Processing time increase | < 20% | TBD | 🔵 Not measured |
| Scene detection accuracy | 90%+ | TBD | 🔵 Not measured |
| Test coverage | > 85% | TBD | 🔵 Not measured |

### Qualitative Metrics
- [ ] Reviewers report easier scene navigation
- [ ] Tracks align with human perception of scenes
- [ ] Better facebank diversity per identity

---

## Risks & Issues

| Risk | Status | Mitigation | Owner |
|------|--------|------------|-------|
| Scene detection inaccurate | 🟡 Open | Tunable thresholds, manual override | TBA |
| False merges | 🟡 Open | Conservative thresholds, spatial constraints | TBA |
| Processing time increase | 🟡 Open | Optimize clustering, parallel processing | TBA |
| API breaking changes | 🟡 Open | Versioned API, 3-month deprecation | TBA |

---

## Decisions Log

| Date | Decision | Rationale | Owner |
|------|----------|-----------|-------|
| 2025-11-19 | Feature proposed | Reduce track fragmentation | TBA |
| TBD | Scene classification: heuristics vs. ML | Pending discussion | TBA |
| TBD | API versioning strategy | Pending product input | TBA |
| TBD | Confessional handling (separate vs. merge) | Pending testing | TBA |

---

## Weekly Updates

### Week of 2025-11-19
**Status**: PLANNING  
**Progress**: Feature specification and planning complete

**Completed**:
- Feature specification (FEATURE.md)
- Terminology documentation (TERMINOLOGY.md)
- Implementation outline (IMPLEMENTATION_OUTLINE.md)
- Feature directory structure created
- TODO checklist created

**Next Week**:
- Assign owner
- Begin Phase 1: Semantic Scene Detection
- Set up development environment

**Blockers**:
- Need to decide scene classification approach
- Need product input on API versioning

---

## Contact & Communication

**Feature Owner**: TBA  
**Stakeholders**: TBA  
**Slack Channel**: TBA  
**Standup Time**: TBA

---

## Links

- **Feature Spec**: [FEATURE.md](../FEATURE.md)
- **TODO**: [TODO.md](../TODO.md)
- **Terminology**: [TERMINOLOGY.md](./TERMINOLOGY.md)
- **Implementation**: [IMPLEMENTATION_OUTLINE.md](./IMPLEMENTATION_OUTLINE.md)

---

**Last Updated**: 2025-11-19  
**Next Review**: TBD (start of Phase 1)
