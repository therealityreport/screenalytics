# Phase 3: Hardening & Cleanup - Implementation Roadmap

**Status:** 🚧 IN PROGRESS
**Started:** 2025-11-18
**Completion Target:** TBD

---

## Overview

Phase 3 closes remaining gaps from the original specification with **concrete implementation** (not just documentation or recommendations). This document tracks all required work items.

---

## Progress Summary

### ✅ Completed (Phase 3)
- [x] README.md updated as concise entry point with performance warnings
- [x] Added comprehensive "Performance Profiles & Limits" section to README

### 🚧 In Progress
- [ ] SETUP.md update (started)

### ⏳ Remaining Work
- 11 major work items across docs, code, API, and tests

---

## 1. Top-Level Docs (3 items)

### 1.1 ✅ README.md - COMPLETE
**Status:** ✅ Done
**Location:** `/README.md`

**Completed:**
- Reduced from 626 lines to 337 lines (concise entry point)
- Added Quick Links section with all new docs
- Added "Performance Profiles & Limits" section with:
  - Profile comparison table
  - ⚠️ Thermal warnings for CPU-only laptops
  - Safe defaults and GPU recommendations
- Added "Pipeline Metrics & Guardrails" section
- Links to all Phase 1/2 docs

### 1.2 ⏳ SETUP.md - TODO
**Status:** ⏳ Pending
**Location:** `/SETUP.md`

**Required Changes:**
1. **Artifact Pipeline Section**
   - Explain the full pipeline artifact flow:
     ```
     detect/track → detections.jsonl, tracks.jsonl, track_metrics.json
     faces_embed → faces.jsonl, faces.npy, data/embeds/{ep_id}/tracks.npy
     cluster → identities.json (with stats: singleton_fraction, etc.)
     cleanup → cleanup_report.json (before/after metrics)
     ```
   - Document relationships:
     - `track_id` links tracks.jsonl → faces.jsonl → identities.json
     - `identity_id` groups tracks together
     - Track embeddings: `data/embeds/{ep_id}/tracks.npy` + `track_ids.json`

2. **Hardware Requirements**
   - **CPU-only (MacBook Air, etc.):**
     - Use `--profile fast_cpu`
     - 5-min clip: ~30 seconds
     - 1-hour episode: ~2-3 hours
     - Export: Avoid `--save-frames --save-crops` together
   - **GPU (RTX 3080, T4, etc.):**
     - Use `--profile high_accuracy --device cuda`
     - 1-hour episode: ~5-10 minutes
     - Can export frames+crops without thermal issues

3. **Links to Detailed Docs**
   - Point to all Phase 1 docs:
     - `docs/pipeline/detect_track_faces.md`
     - `docs/pipeline/faces_harvest.md`
     - `docs/pipeline/cluster_identities.md`
     - `docs/pipeline/episode_cleanup.md`
     - `docs/reference/artifacts_faces_tracks_identities.md`
     - `docs/ops/hardware_sizing.md`
     - `docs/ops/performance_tuning_faces_pipeline.md`

### 1.3 ⏳ PRD.md - TODO
**Status:** ⏳ Pending
**Location:** `/PRD.md`

**Required Changes:**
1. **Update All Doc References**
   - Replace any broken or outdated links with actual files created in Phase 1/2
   - Ensure all references point to:
     - `SOLUTION_ARCHITECTURE.md`
     - `DIRECTORY_STRUCTURE.md`
     - `CONFIG_GUIDE.md`
     - `ACCEPTANCE_MATRIX.md`
     - `FEATURES_GUIDE.md`
     - Docs under `docs/architecture/`, `docs/pipeline/`, `docs/reference/`, `docs/ops/`

2. **Align Acceptance Criteria with Actual Thresholds**
   - **Detect/Track:**
     - Max `tracks_per_minute`: 50 (CPU), 50 (GPU)
     - Max `short_track_fraction`: 0.30
     - Max `id_switch_rate`: 0.10
     - Runtime: ≤ 3× realtime (CPU), ≤ 10 min (GPU for 1hr episode)
   - **Faces:**
     - `min_quality_score`: 3.0
     - `max_crops_per_track`: 16 (configurable)
     - Reasonable total faces per episode
   - **Clustering:**
     - `singleton_fraction`: 0.20-0.50 (acceptable range)
     - `largest_cluster_fraction`: < 0.60
     - Expected clusters: 3-50 for typical episode

3. **Delegate to ACCEPTANCE_MATRIX.md**
   - Add statement: "Specific acceptance thresholds and test assertions are maintained in ACCEPTANCE_MATRIX.md"

---

## 2. Code Gaps - Faces Harvest (2 items)

### 2.1 ⏳ Fix Blank/Gray Crops - TODO
**Status:** ⏳ Pending
**Location:** `tools/episode_run.py` (faces_embed section)

**Required Implementation:**
1. **Bbox Clamping**
   - Audit `_prepare_face_crop()` function
   - Ensure bboxes are clamped to image bounds before slicing:
     ```python
     x1 = max(0, min(x1, img_width))
     y1 = max(0, min(y1, img_height))
     x2 = max(0, min(x2, img_width))
     y2 = max(0, min(y2, img_height))
     ```
   - Handle degenerate boxes (zero or negative size):
     ```python
     if x2 <= x1 or y2 <= y1:
         return None, "degenerate_box"
     ```

2. **Runtime Checks**
   - Track blank crop count during run
   - If blank_count / total_crops > 0.10, log warning
   - Include in `track_metrics.json`:
     ```json
     {
       "crop_stats": {
         "total_crops": 1000,
         "blank_crops": 5,
         "blank_fraction": 0.005
       }
     }
     ```

3. **Update Docs**
   - Document fix in `docs/pipeline/faces_harvest.md`
   - Add troubleshooting entry in `docs/ops/troubleshooting_faces_pipeline.md`

### 2.2 ⏳ Hard Volume Control - TODO
**Status:** ⏳ Pending
**Location:** `tools/episode_run.py`, `config/pipeline/faces_embed_sampling.yaml`

**Required Implementation:**
1. **Per-Track Cap**
   - Enforce `max_crops_per_track` in code (currently exists but verify enforcement)
   - Implement sampling strategy when cap is hit:
     - Uniform: Sample evenly across track timeline
     - Quality-weighted: Prioritize high-quality faces
     - Stratified: Ensure coverage of track segments

2. **Per-Episode Cap (Optional)**
   - Add `max_crops_per_episode` config option
   - When exceeded, emit warning:
     ```python
     if total_crops > max_crops_per_episode:
         LOGGER.warning("[VOLUME] Episode has %d crops (max: %d). Consider increasing min_quality or decreasing max_crops_per_track.", total_crops, max_crops_per_episode)
     ```

3. **Config Update**
   - Add to `config/pipeline/faces_embed_sampling.yaml`:
     ```yaml
     volume_control:
       max_crops_per_track: 16    # Hard cap per track (existing)
       max_crops_per_episode: 10000  # Optional episode-wide cap
       sampling_strategy: "quality_weighted"  # uniform | quality_weighted | stratified
     ```

4. **Schema Update**
   - Verify `faces.jsonl` includes:
     - `track_id`
     - `frame_index` or `timestamp`
     - `quality_score`
     - `rejection_reason` (for skipped faces)
   - Update `docs/reference/artifacts_faces_tracks_identities.md`

---

## 3. Code Gaps - Clustering (2 items)

### 3.1 ⏳ Explicit Outlier Handling - TODO
**Status:** ⏳ Pending
**Location:** `tools/episode_run.py` (_run_cluster_stage)

**Required Implementation:**
1. **Outlier Treatment Config**
   - Add to `config/pipeline/clustering.yaml`:
     ```yaml
     outlier_handling:
       treat_as_singletons: true   # false = mark as noise/null
       mark_low_cohesion: true      # Add low_cohesion flag
     ```

2. **Code Implementation**
   - When `treat_as_singletons: true`:
     - Outliers become identity_id with single track
   - When `treat_as_singletons: false`:
     - Set `identity_id: null` or `cluster_id: "noise"`
   - Always include `outlier: true` flag in identities.json

3. **Identities.json Schema**
   - Ensure each identity has:
     ```json
     {
       "identity_id": "ident_001",
       "track_ids": [5, 12, 23],
       "outlier": false,
       "low_cohesion": false,
       "cohesion_score": 0.85,
       "min_identity_sim": 0.72
     }
     ```

### 3.2 ⏳ Metrics Persistence - TODO
**Status:** ⏳ Pending
**Location:** `tools/episode_run.py`, cluster metrics writing

**Required Implementation:**
1. **Write to track_metrics.json**
   - Currently cluster metrics only go to `identities.json`
   - Also write to `track_metrics.json`:
     ```json
     {
       "ep_id": "rhobh-s05e02",
       "detect_track_metrics": { ... },
       "cluster_metrics": {
         "total_clusters": 12,
         "total_tracks": 45,
         "singleton_count": 8,
         "singleton_fraction": 0.667,
         "largest_cluster_size": 15,
         "largest_cluster_fraction": 0.333,
         "outlier_count": 3,
         "low_cohesion_count": 1
       }
     }
     ```

2. **Write to cleanup_report.json**
   - When cleanup runs, include before/after cluster metrics
   - Enable comparison: "singleton_fraction went from 0.6 to 0.4"

3. **UI Integration**
   - Expose in Episode Detail / Facebank views
   - Show cluster health at a glance

---

## 4. Code Gaps - Episode Cleanup (1 item)

### 4.1 ⏳ Align with Configs & Add Metrics - TODO
**Status:** ⏳ Pending
**Location:** `tools/episode_cleanup.py`

**Required Implementation:**
1. **Config Alignment**
   - Ensure cleanup uses the same config sources as main pipeline:
     - `detection.yaml`
     - `tracking.yaml`
     - `faces_embed_sampling.yaml`
     - `clustering.yaml`
     - `performance_profiles.yaml`
   - Remove any duplicated magic thresholds

2. **Cleanup Report Enhancement**
   - Implement or extend `cleanup_report.json`:
     ```json
     {
       "ep_id": "rhobh-s05e02",
       "cleanup_actions": ["split_tracks", "reembed", "recluster"],
       "before": {
         "tracks": 120,
         "tracks_per_minute": 65.3,
         "short_track_fraction": 0.42,
         "id_switch_rate": 0.15,
         "clusters": 45,
         "singleton_fraction": 0.68,
         "largest_cluster_fraction": 0.22
       },
       "after": {
         "tracks": 95,
         "tracks_per_minute": 48.2,
         "short_track_fraction": 0.18,
         "id_switch_rate": 0.06,
         "clusters": 18,
         "singleton_fraction": 0.38,
         "largest_cluster_fraction": 0.31
       },
       "improvement": {
         "tracks_per_minute": -17.1,
         "short_track_fraction": -0.24,
         "id_switch_rate": -0.09,
         "singleton_fraction": -0.30
       }
     }
     ```

3. **Progress Streaming**
   - For `/jobs/episode_cleanup_async`, ensure `progress.json` tracks:
     - Current phase (split_tracks, reembed, recluster, group_clusters)
     - Per-phase progress (frames_done/total for relevant phases)
     - Key before/after stats at end

---

## 5. API Integration (1 item)

### 5.1 ⏳ Add Profile Field to API Endpoints - TODO
**Status:** ⏳ Pending
**Location:** `apps/api/main.py` or equivalent job endpoints

**Required Implementation:**
1. **Add `profile` Field to Payloads**
   - `POST /jobs/detect_track`:
     ```python
     class DetectTrackRequest(BaseModel):
         ep_id: str
         profile: Optional[str] = "balanced"  # fast_cpu | balanced | high_accuracy
         stride: Optional[int] = None
         fps: Optional[float] = None
         ...
     ```
   - `POST /jobs/detect_track_async`: Same
   - `POST /jobs/faces_embed`: Optional profile for embed-specific settings
   - `POST /jobs/cluster`: Use profile to set cluster_thresh defaults
   - `POST /jobs/episode_cleanup_async`: Use profile for all phases

2. **Profile Application Logic**
   - When `profile` is provided:
     - Load profile from `config/pipeline/performance_profiles.yaml`
     - Apply defaults (stride, fps, min_size, etc.)
     - CLI/request overrides take precedence
   - Same precedence as CLI: Request params > Profile > Stage config > Defaults

3. **Progress & Metrics Exposure**
   - Ensure `progress.json` includes:
     - For detect/track: `tracks_per_minute`, `short_track_fraction`, `id_switch_rate`
     - For cluster: `singleton_fraction`, `largest_cluster_fraction`
   - Include in final summary response

4. **Update API.md**
   - Document new `profile` field
   - Show example requests
   - Document metrics in responses

---

## 6. Tests & CI (4 items)

### 6.1 ⏳ Detect/Track Integration Test - TODO
**Status:** ⏳ Pending
**Location:** `tests/ml/test_detect_track_real.py` (create or extend)

**Required Implementation:**
```python
import pytest
import os

@pytest.mark.skipif(
    os.environ.get("RUN_ML_TESTS") != "1",
    reason="ML tests require RUN_ML_TESTS=1"
)
def test_detect_track_metrics_within_bounds():
    """Test that detect/track produces acceptable metrics."""
    # Run detect/track on small test clip
    result = run_detect_track("test_clip.mp4", profile="balanced")

    metrics = result["metrics"]

    # Assert metrics within acceptance bounds
    assert metrics["tracks_per_minute"] < 50, \
        f"tracks_per_minute={metrics['tracks_per_minute']} exceeds threshold 50"

    assert metrics["short_track_fraction"] < 0.30, \
        f"short_track_fraction={metrics['short_track_fraction']} exceeds threshold 0.30"

    assert metrics["id_switch_rate"] < 0.10, \
        f"id_switch_rate={metrics['id_switch_rate']} exceeds threshold 0.10"
```

**Acceptance Thresholds:**
- `tracks_per_minute < 50`
- `short_track_fraction < 0.30`
- `id_switch_rate < 0.10`
- Runtime ≤ 3× realtime (CPU) or ≤ 10 min (GPU for 1hr)

### 6.2 ⏳ Faces/Clustering Integration Tests - TODO
**Status:** ⏳ Pending
**Location:** `tests/ml/test_faces_embed.py`, `tests/ml/test_cluster.py`

**Required Implementation:**
1. **test_faces_embed.py**
   ```python
   def test_faces_volume_control():
       """Test that max_crops_per_track is enforced."""
       result = run_faces_embed("test_ep", max_crops_per_track=16)

       faces = load_faces_jsonl(result["faces_path"])
       crops_per_track = defaultdict(int)
       for face in faces:
           crops_per_track[face["track_id"]] += 1

       for track_id, count in crops_per_track.items():
           assert count <= 16, \
               f"Track {track_id} has {count} crops (max: 16)"

   def test_quality_gating():
       """Test that quality_score filters work."""
       result = run_faces_embed("test_ep", min_quality_score=5.0)

       faces = load_faces_jsonl(result["faces_path"])
       for face in faces:
           if "quality_score" in face:
               assert face["quality_score"] >= 5.0
   ```

2. **test_cluster.py**
   ```python
   def test_cluster_metrics_within_bounds():
       """Test cluster metrics are reasonable."""
       result = run_cluster("test_ep", cluster_thresh=0.58)

       identities = load_identities_json(result["identities_path"])
       stats = identities["stats"]

       assert 0.20 <= stats["singleton_fraction"] <= 0.50, \
           f"singleton_fraction={stats['singleton_fraction']} outside acceptable range"

       assert stats["largest_cluster_fraction"] < 0.60, \
           f"largest_cluster_fraction={stats['largest_cluster_fraction']} exceeds 0.60"

       assert 3 <= stats["clusters"] <= 50, \
           f"clusters={stats['clusters']} outside expected range [3, 50]"
   ```

### 6.3 ⏳ Quality Gating Unit Tests - TODO
**Status:** ⏳ Pending
**Location:** `tests/ml/test_quality_gating.py` (new file)

**Required Implementation:**
```python
def test_quality_gating_config_loading():
    """Test that quality gating config loads correctly."""
    from tools.episode_run import _load_quality_gating_config

    config = _load_quality_gating_config()

    assert "min_quality_score" in config
    assert "max_yaw_angle" in config
    assert "max_pitch_angle" in config
    assert "allowed_expressions" in config

def test_quality_score_calculation():
    """Test quality score combines metrics correctly."""
    # Mock crop with known metrics
    crop = create_test_crop(blur=50.0, size=120, confidence=0.8)

    quality_score = calculate_quality_score(crop)

    assert quality_score > 0.0
    assert quality_score <= 10.0

def test_pose_filtering():
    """Test that extreme poses are filtered."""
    # Create face with extreme yaw
    face_extreme_yaw = {"landmarks": [...], "yaw": 60.0}  # > 45 degrees

    should_skip, reason = check_pose_quality(face_extreme_yaw, max_yaw=45.0)

    assert should_skip is True
    assert "yaw" in reason
```

### 6.4 ⏳ Performance Profile Unit Tests - TODO
**Status:** ⏳ Pending
**Location:** `tests/ml/test_performance_profiles.py` (new file)

**Required Implementation:**
```python
def test_profile_loading():
    """Test that all profiles load correctly."""
    from tools.episode_run import _load_performance_profile

    for profile_name in ["fast_cpu", "balanced", "high_accuracy"]:
        profile = _load_performance_profile(profile_name)

        assert "frame_stride" in profile
        assert "detection_fps_limit" in profile
        assert "description" in profile

def test_profile_precedence():
    """Test that CLI args override profile settings."""
    args = create_test_args(profile="fast_cpu", stride=3)  # Explicit stride

    apply_profile_settings(args)

    # Explicit stride=3 should override profile's stride=10
    assert args.stride == 3

def test_env_var_overrides_profile():
    """Test that env vars override profile settings."""
    os.environ["SCREENALYTICS_TRACK_BUFFER"] = "150"

    args = create_test_args(profile="balanced")
    apply_profile_settings(args)

    # Env var should win
    assert args.track_buffer == 150
```

---

## 7. Docs Reconciliation (2 items)

### 7.1 ⏳ FEATURES Docs Cleanup - TODO
**Status:** ⏳ Pending
**Location:** `FEATURES/*/docs/`

**Required Actions:**
1. **Enumerate All Feature Docs**
   ```bash
   find FEATURES/ -name "*.md" -type f
   ```

2. **For Each Feature Doc:**
   - If it documents **current pipeline**:
     - Update to reference new configs/metrics/docs
     - Link to Phase 1/2 docs
   - If it describes **old/experimental** pipelines:
     - Move to `docs/archive/` or delete
     - Update `FEATURES_GUIDE.md` to mark as archived

3. **Update FEATURES_GUIDE.md**
   - Mark archived features
   - Document cleanup actions

### 7.2 ⏳ Doc Consistency Pass - TODO
**Status:** ⏳ Pending
**Location:** All docs

**Required Actions:**
1. **Metric Name Consistency**
   - Ensure these are spelled consistently everywhere:
     - `tracks_per_minute` (not tracks_per_min)
     - `short_track_fraction` (not ghost_track_fraction)
     - `id_switch_rate` (not id_switches_rate)
     - `singleton_fraction`
     - `largest_cluster_fraction`

2. **Config Key Consistency**
   - `cluster_thresh` (not cluster_threshold)
   - `min_cluster_size`
   - `min_quality_score`
   - `max_yaw_angle`, `max_pitch_angle`
   - `frame_stride`, `detection_fps_limit`

3. **Profile Name Consistency**
   - `fast_cpu` (not fast-cpu or fastcpu)
   - `balanced`
   - `high_accuracy` (not high-accuracy)

4. **Files to Audit:**
   - `docs/pipeline/*`
   - `docs/reference/*`
   - `docs/ops/*`
   - `README.md`, `SETUP.md`, `PRD.md`, `API.md`, `CONFIG_GUIDE.md`, `ACCEPTANCE_MATRIX.md`

---

## 8. Phase 3 Summary Creation (1 item)

### 8.1 ⏳ Create PHASE_3_IMPLEMENTATION_SUMMARY.md - TODO
**Status:** ⏳ Pending
**Location:** `/PHASE_3_IMPLEMENTATION_SUMMARY.md`

**Required Content:**
1. **List All Files Changed**
   - Top-level docs updated
   - Code changes (with line numbers)
   - Config changes
   - Test files created

2. **Map Changes to Phase 3 Requirements**
   - For each gap identified in Phase 3 spec, show how it was addressed

3. **Document Thresholds/Defaults Set**
   - All acceptance thresholds with rationale
   - Easy to tune later

4. **Before/After Metrics**
   - Any performance improvements
   - Test coverage added

---

## Acceptance Criteria

Phase 3 is complete when:

- [ ] All top-level docs (README, SETUP, PRD) are updated and link to Phase 1/2 docs
- [ ] Blank/gray crops are fixed with bbox clamping and validation
- [ ] Hard volume control enforced (max_crops_per_track, optional max_crops_per_episode)
- [ ] Outlier handling is explicit and configurable
- [ ] Cluster metrics persisted to track_metrics.json and cleanup_report.json
- [ ] episode_cleanup.py uses same configs as main pipeline
- [ ] API endpoints accept `profile` parameter
- [ ] All 4 test suites implemented and passing in CI
- [ ] FEATURES docs cleaned up (current vs archived)
- [ ] Consistency pass complete across all docs
- [ ] PHASE_3_IMPLEMENTATION_SUMMARY.md created

---

## Implementation Priority

**High Priority** (blocking production use):
1. Fix blank/gray crops (security/quality issue)
2. API profile integration (UX improvement)
3. Cluster metrics persistence (observability)
4. Detect/track integration tests (CI gate)

**Medium Priority** (polish):
5. SETUP.md, PRD.md updates (documentation completeness)
6. Hard volume control (resource management)
7. Outlier handling refinement (quality improvement)
8. Faces/cluster integration tests (coverage)

**Low Priority** (nice-to-have):
9. Quality gating unit tests (test coverage)
10. Performance profile unit tests (test coverage)
11. FEATURES docs cleanup (maintenance)
12. Doc consistency pass (polish)

---

## Next Steps

1. **Resume from where Phase 3 was paused** (currently: SETUP.md update)
2. **Implement in priority order** (see above)
3. **Create PR for each major work item** (easier to review than one giant PR)
4. **Update this roadmap** as items are completed

---

**Maintained by:** Screenalytics Engineering
**Last Updated:** 2025-11-18
