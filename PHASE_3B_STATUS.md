# Phase 3b Status Report — Screenalytics

**Date:** 2025-11-18
**Branch:** `nov-18`
**Latest Commit:** `4e7bd75` - API metrics exposure

---

## Progress Summary

**Phase 3a (COMPLETE)** — Top-level docs + metrics persistence
**Phase 3b (IN PROGRESS)** — API integration, tests, cleanup alignment

---

## ✅ Completed (Phase 3b)

### 1. API Integration — Profile Support (HIGH Priority)

**Status:** ✅ **COMPLETE**

**What was done:**
- Added `PROFILE_LITERAL` type to `apps/api/routers/jobs.py`
  - Valid profiles: `fast_cpu`, `low_power`, `balanced`, `high_accuracy`
- Added `profile` field to ALL job request schemas:
  - `DetectTrackRequest`
  - `FacesEmbedRequest`
  - `ClusterRequest`
  - `CleanupJobRequest`
- Updated ALL command builders to pass `--profile` to CLI:
  - `_build_detect_track_command()`
  - `_build_faces_command()`
  - `_build_cluster_command()`
- Updated ALL 4 service methods in `apps/api/services/jobs.py`:
  - `start_detect_track_job()` - accepts `profile` parameter, adds `--profile` to command
  - `start_faces_embed_job()` - accepts `profile` parameter, adds `--profile` to command
  - `start_cluster_job()` - accepts `profile` parameter, adds `--profile` to command
  - `start_episode_cleanup_job()` - accepts `profile` parameter, adds `--profile` to command
- Updated ALL 4 router endpoints to pass `profile=req.profile` to service:
  - `POST /detect_track_async`
  - `POST /faces_embed_async`
  - `POST /cluster_async`
  - `POST /episode_cleanup_async`

**How to use:**
```bash
curl -X POST http://localhost:8000/jobs/detect_track_async \
  -H "Content-Type: application/json" \
  -d '{
    "ep_id": "demo",
    "profile": "balanced",
    "stride": 6,
    "device": "auto"
  }'
```

**Files Modified:**
- `apps/api/routers/jobs.py` (+23 lines)
- `apps/api/services/jobs.py` (+20 lines)

**Commit:** `dd1f022`

---

## ⏳ Remaining Work (Phase 3b)

### 2. API Integration — Metrics Exposure (HIGH Priority)

**Status:** ✅ **COMPLETE**

**What was done:**
- Added `_read_track_metrics()` helper method to `apps/api/services/jobs.py`
  - Reads `track_metrics.json` for a given episode
  - Returns full metrics payload including detect/track, cluster, scene, and crop stats
  - Gracefully handles missing metrics file (returns None)
- Updated `JobService.get()` to include metrics in JobRecord response
  - Adds `track_metrics` field to record if available
- Updated `JobService.get_progress()` to include metrics in progress response
  - Merges metrics into progress.json response
  - No schema changes required; metrics added as optional top-level field

**Metrics Exposed:**
- **Detect/Track Metrics:**
  - `tracks_per_minute` - Track generation rate
  - `short_track_fraction` - Fraction of tracks below minimum length
  - `id_switch_rate` - Identity switching/fragmentation rate
  - `total_tracks` - Total number of tracks
  - `total_detections` - Total number of detections
  - `duration_minutes` - Episode duration
- **Cluster Metrics:**
  - `singleton_fraction` - Fraction of single-track clusters
  - `largest_cluster_fraction` - Size of largest cluster relative to total
  - `total_clusters` - Number of identity clusters
  - `outlier_tracks` - Number of tracks flagged as outliers
  - `singleton_count` - Count of singleton clusters
  - `largest_cluster_size` - Size of largest cluster
- **Additional Metrics:**
  - `scene_cuts` - Scene detection summary
  - `crop_stats` - Crop quality statistics

**How to use:**
```bash
# Get job with metrics
curl http://localhost:8000/jobs/{job_id}

# Get progress with metrics
curl http://localhost:8000/jobs/{job_id}/progress
```

**Files Modified:**
- `apps/api/services/jobs.py` (+26 lines)

**Commit:** `4e7bd75`

---

### 3. API Integration — Update API.md (MEDIUM Priority)

**Status:** ⏳ **TODO**

**Requirements:**
1. Document `profile` field in all job request schemas
2. Show example requests with profile
3. Document metrics in job status responses
4. Link to CONFIG_GUIDE.md for profile details

---

### 4. Tests & CI — Integration Tests (HIGH Priority)

**Status:** ⏳ **TODO**

**Required Tests:**

#### 4.1 Detect/Track Integration Test
- **File:** `tests/ml/test_detect_track_real.py`
- **Requirements:**
  - Run detect/track on small fixture video (10-30 sec)
  - Assert `detections.jsonl` and `tracks.jsonl` exist
  - Assert `track_metrics.json` exists
  - Assert `tracks_per_minute` < threshold (e.g., 50)
  - Assert `short_track_fraction` < threshold (e.g., 0.30)
  - Assert `id_switch_rate` < threshold (e.g., 0.10)
  - Test with `--profile balanced`

#### 4.2 Faces Embed Integration Test
- **File:** `tests/ml/test_faces_embed_real.py`
- **Requirements:**
  - Run faces_embed on episode with existing tracks
  - Assert `faces.jsonl` and `faces.npy` exist
  - Assert embeddings are unit-norm (||e|| ≈ 1.0)
  - Assert `max_samples_per_track` is respected
  - Assert quality gating works (not all detections pass)
  - Test with `--profile balanced`

#### 4.3 Cluster Integration Test
- **File:** `tests/ml/test_cluster_identities_real.py`
- **Requirements:**
  - Run cluster on episode with faces
  - Assert `identities.json` exists
  - Assert `cluster_metrics` in `track_metrics.json`
  - Assert `singleton_fraction`, `largest_cluster_fraction` within expected ranges
  - Test with synthetic embeddings (known clusters)
  - Test with `--profile balanced`

#### 4.4 Episode Cleanup Tests
- **File:** `tests/ml/test_episode_cleanup.py`
- **Requirements:**
  - Test each cleanup action (split_tracks, reembed, recluster, group_clusters)
  - Assert `cleanup_report.json` exists
  - Assert before/after metrics show improvement
  - Assert no dangling track_id or identity_id references

#### 4.5 API Smoke Test
- **File:** `tests/api/test_jobs_smoke.py`
- **Requirements:**
  - Submit detect_track → faces_embed → cluster jobs via API
  - Poll until completion
  - Verify progress.json updates
  - Verify artifacts exist
  - Verify metrics are returned in responses

#### 4.6 Acceptance Matrix Wiring
- **Requirements:**
  - Update ACCEPTANCE_MATRIX.md with test file references
  - Configure CI to run tests
  - Gate CI on metric thresholds

---

### 5. Episode Cleanup Alignment (MEDIUM Priority)

**Status:** ⏳ **TODO**

**Requirements:**

#### 5.1 Config & Profile Reuse
- Update `tools/episode_cleanup.py` to:
  - Accept `--profile` argument
  - Load configs from `config/pipeline/*.yaml` (same as episode_run.py)
  - Remove hardcoded thresholds

#### 5.2 Cleanup Report Enhancement
- Update `cleanup_report.json` schema to include:
  - Before/after track metrics (count, short_track_fraction)
  - Before/after face metrics (count, per-track averages)
  - Before/after cluster metrics (singleton_fraction, largest_cluster_fraction, total_clusters)
- Update `track_metrics.json` after cleanup to reflect "after" state

#### 5.3 Progress Reporting
- Ensure episode_cleanup emits progress events for each phase:
  - Phase name (split_tracks, reembed, recluster, group_clusters)
  - Phase progress (0.0-1.0)
  - Phase runtime
  - Summary of changes when phase completes

---

### 6. Docs Reconciliation (LOW Priority)

**Status:** ⏳ **TODO**

**Requirements:**

#### 6.1 FEATURES Cleanup
- Enumerate all `FEATURES/*/docs/*.md`
- For each doc:
  - If current: Update references to new configs, metrics, docs
  - If legacy/experimental: Move to `docs/archive` or delete
- Update `FEATURES_GUIDE.md` to reflect current features

#### 6.2 Naming Consistency Pass
- Audit all docs for consistent terminology:
  - Profile names: `fast_cpu` vs `low_power` (currently both exist)
  - Metric names: `tracks_per_minute`, `short_track_fraction`, etc.
  - Config keys: Ensure consistency across all YAML files
- Files to check:
  - `docs/pipeline/*`
  - `docs/reference/*`
  - `docs/ops/*`
  - All top-level docs (README, SETUP, PRD, CONFIG_GUIDE, ACCEPTANCE_MATRIX)

---

## Summary Statistics

**Files Modified (Phase 3a + 3b):**
- Documentation: 3 files (README, SETUP, PRD)
- Code: 3 files (episode_run.py, jobs.py routers, jobs.py services)

**Lines Changed:**
- Phase 3a: ~600 lines (docs + metrics)
- Phase 3b: ~43 lines (API profile integration)

**Commits:**
- Phase 3a: 2 commits (`c629e51`, `6e62325`)
- Phase 3b: 1 commit (`dd1f022`)

**Test Coverage:**
- Integration tests: 0/6 implemented
- Unit tests: 0/4 implemented
- API smoke test: 0/1 implemented

**Priority Breakdown:**
- ✅ HIGH: API profile integration (DONE)
- ⏳ HIGH: API metrics exposure (TODO)
- ⏳ HIGH: Integration tests (TODO)
- ⏳ MEDIUM: Episode cleanup alignment (TODO)
- ⏳ MEDIUM: API.md update (TODO)
- ⏳ LOW: FEATURES cleanup (TODO)
- ⏳ LOW: Naming consistency (TODO)

---

## Next Steps

### Immediate (Session Continuation)
1. ✅ **DONE:** API metrics exposure
2. **NEXT:** Expose metrics in API responses
3. **THEN:** Create 1-2 integration tests (detect/track, cluster)

### Future Sessions
4. Complete remaining integration tests
5. Align episode_cleanup.py with configs
6. Update API.md documentation
7. FEATURES cleanup and naming consistency pass
8. Update PHASE_3_IMPLEMENTATION_SUMMARY.md with all Phase 3b work

---

## References

- [PHASE_3_ROADMAP.md](PHASE_3_ROADMAP.md) — Original Phase 3 requirements
- [PHASE_3_IMPLEMENTATION_SUMMARY.md](PHASE_3_IMPLEMENTATION_SUMMARY.md) — Phase 3a summary
- [ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md) — Test requirements and thresholds
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) — Profile configuration reference

---

**Status:** Phase 3b API integration complete. Remaining HIGH priority: metrics exposure + tests.
**Maintainer:** Claude Code (Anthropic)
**Last Updated:** 2025-11-18
