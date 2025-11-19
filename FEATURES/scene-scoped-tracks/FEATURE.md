# Scene-Scoped Tracks Feature

**Status**: PLANNING  
**Owner**: TBA  
**Version**: 1.0  
**Created**: 2025-11-19

## Executive Summary

Redefine the fundamental unit of **track** from "continuous detection sequence" to "all appearances of ONE person within ONE semantic scene." This paradigm shift aligns tracking with human perception of reality TV content and dramatically reduces track fragmentation.

### Current Problem
- **Track** = continuous sequence where person is visible without occlusion
- Reality TV has frequent camera cuts between subjects → massive fragmentation
- Example: 4,352 tracks for a single episode (avg ~0.5s per track)
- Poor clustering: short tracks lack embedding diversity
- UX friction: reviewing "all Lisa faces from pool party" spans 50+ tracks

### Proposed Solution
- **Track** = all appearances of ONE person within ONE semantic scene
- **Semantic Scene** = location/activity unit (e.g., "Kitchen argument between Heather and Shannon")
- Scene boundaries (location/activity changes) are the ONLY track terminators
- Expected reduction: 20 scenes × 5 cast = ~100 tracks (vs. 4,352 current)

---

## Terminology Changes

### NEW: Scene-Scoped Track (SST)
**Definition**: All appearances of one person within one semantic scene, regardless of camera cuts or temporal gaps.

**Example**: "Kitchen argument scene"
- **Track #1**: All Heather frames (frames 1200-1850, even if she's only visible in frames 1200-1300, 1400-1500, 1700-1850)
- **Track #2**: All Shannon frames in same scene
- **Track #3**: All Tamra reaction shots in same scene

**Properties**:
- `scene_id`: Links to semantic scene
- `person_id`: Identity within scene (can be null if unidentified)
- `scene_start_s`, `scene_end_s`: Scene temporal boundaries
- `appearance_segments`: List of [start, end] frame ranges where person actually appears
- `total_visible_s`: Sum of visible time (not scene duration)

### Renamed: Detection Sequence → Shot-Level Track (SLT)
**Definition**: Current "track" concept - continuous detection sequence without occlusion.

**Purpose**: Internal intermediate representation for tracking algorithm output.

**Fate**: These become building blocks that get merged into SSTs. Not exposed in final API/UI.

---

## Architecture

### Phase 1: Semantic Scene Segmentation

**Input**: Video file  
**Output**: `scenes.jsonl` with scene boundaries and metadata

```json
{
  "scene_id": "scene_005",
  "ep_id": "ep_12345",
  "start_frame": 1200,
  "end_frame": 1850,
  "start_s": 40.0,
  "end_s": 61.67,
  "scene_type": "dialogue",
  "location_hint": "kitchen",
  "shot_cuts": [1200, 1245, 1298, 1350, 1405, 1487, 1650, 1701, 1785],
  "confidence": 0.92,
  "schema_version": "scene_v1"
}
```

#### Semantic Scene Detection Algorithm

**Hybrid approach**: Shot boundary detection + temporal clustering

1. **Shot boundary detection** (existing PySceneDetect):
   - Detects hard cuts (camera angle changes, dissolves)
   - Current threshold: 27.0 (configurable)
   
2. **Semantic clustering** (NEW):
   - Group shots into scenes using temporal + visual heuristics:
     - **Time gap heuristic**: 3+ second black frames → scene boundary
     - **Shot duration pattern**: Rapid back-and-forth (< 2s shots) → same dialogue scene
     - **Visual similarity**: Compare first/last frames of adjacent shots (optional, Phase 2)
     - **Audio cues**: Ambient sound changes, music transitions (optional, Phase 3)

3. **Confessional detection** (special case):
   - Fixed camera angle + single person + characteristic background
   - Treat as separate scenes even if intercut with main narrative
   - Heuristic: single-person shots with duration > 5s + similar background across cuts

**Configuration** (`config/pipeline/scene_segmentation.yaml`):
```yaml
detector: pyscenedetect  # or: internal, off
shot_threshold: 27.0
min_shot_len: 5

# Semantic scene grouping
temporal_gap_threshold_s: 3.0  # Black screen / fade duration
rapid_cut_window_s: 2.0        # Max shot duration for dialogue scenes
confessional_min_duration_s: 5.0
confessional_person_count: 1

# Advanced (Phase 2+)
visual_similarity_threshold: 0.85  # SSIM between shots
enable_audio_scene_detection: false
```

---

### Phase 2: Shot-Level Tracking (Current System)

**Input**: Video frames + scene boundaries  
**Output**: `shot_tracks.jsonl` (detection sequences)

**Changes from current**:
- Add `scene_id` to each track record
- Track termination ON scene boundaries (already implemented via scene cuts)
- Rename output artifact: `tracks.jsonl` → `shot_tracks.jsonl` (internal only)

```json
{
  "shot_track_id": 42,
  "scene_id": "scene_005",
  "ep_id": "ep_12345",
  "start_s": 40.5,
  "end_s": 43.2,
  "frame_span": [1215, 1296],
  "stats": {"detections": 82, "avg_conf": 0.93},
  "schema_version": "shot_track_v1"
}
```

**Implementation**: Minimal changes to existing `episode_run.py` tracking loop.

---

### Phase 3: Scene-Scoped Track Merging (NEW)

**Input**: `shot_tracks.jsonl` + `chips_manifest.parquet` + `embeddings`  
**Output**: `tracks.jsonl` (scene-scoped tracks)

**Algorithm**: Appearance-based clustering within scene boundaries

```python
def merge_shot_tracks_into_scene_tracks(shot_tracks, embeddings, scenes):
    """Merge shot-level tracks into scene-scoped tracks."""
    scene_tracks = []
    
    for scene in scenes:
        # Get all shot tracks within this scene
        scene_shot_tracks = [
            st for st in shot_tracks 
            if st["scene_id"] == scene["scene_id"]
        ]
        
        if not scene_shot_tracks:
            continue
        
        # Cluster by appearance similarity (ArcFace embeddings)
        identity_clusters = cluster_by_appearance(
            shot_tracks=scene_shot_tracks,
            embeddings=embeddings,
            similarity_threshold=0.80,  # Configurable
            linkage="average"
        )
        
        # Create one scene-scoped track per identity cluster
        for cluster_idx, cluster in enumerate(identity_clusters):
            sst = merge_cluster_into_track(
                cluster=cluster,
                scene=scene,
                track_id=f"{scene['scene_id']}_p{cluster_idx}"
            )
            scene_tracks.append(sst)
    
    return scene_tracks


def cluster_by_appearance(shot_tracks, embeddings, similarity_threshold, linkage):
    """Hierarchical clustering using ArcFace embeddings."""
    # 1. Get representative embedding for each shot track (median or centroid)
    track_embeddings = []
    for st in shot_tracks:
        track_embs = get_embeddings_for_track(st["shot_track_id"], embeddings)
        if len(track_embs) > 0:
            representative = np.median(track_embs, axis=0)  # Or mean
            track_embeddings.append((st, representative))
    
    # 2. Compute pairwise cosine similarity
    similarity_matrix = compute_similarity_matrix(
        [emb for _, emb in track_embeddings]
    )
    
    # 3. Agglomerative clustering
    distance_matrix = 1.0 - similarity_matrix
    clusters = scipy.cluster.hierarchy.fclusterdata(
        distance_matrix,
        t=1.0 - similarity_threshold,  # Distance threshold
        criterion='distance',
        method=linkage  # 'average', 'complete', 'single'
    )
    
    # 4. Group shot tracks by cluster ID
    cluster_groups = defaultdict(list)
    for (st, _), cluster_id in zip(track_embeddings, clusters):
        cluster_groups[cluster_id].append(st)
    
    return list(cluster_groups.values())


def merge_cluster_into_track(cluster, scene, track_id):
    """Merge shot tracks in cluster into single scene-scoped track."""
    # Sort by temporal order
    cluster = sorted(cluster, key=lambda st: st["start_s"])
    
    # Compute appearance segments (visible intervals)
    appearance_segments = [
        [st["frame_span"][0], st["frame_span"][1]]
        for st in cluster
    ]
    
    # Compute total visible time
    total_visible_s = sum(st["end_s"] - st["start_s"] for st in cluster)
    
    # Aggregate quality stats
    quality_stats = aggregate_quality_stats(cluster)
    
    return {
        "track_id": track_id,
        "scene_id": scene["scene_id"],
        "ep_id": scene["ep_id"],
        "scene_start_s": scene["start_s"],
        "scene_end_s": scene["end_s"],
        "appearance_segments": appearance_segments,
        "total_visible_s": total_visible_s,
        "shot_track_ids": [st["shot_track_id"] for st in cluster],
        "quality_stats": quality_stats,
        "schema_version": "track_v2"
    }
```

**New Script**: `tools/merge_scene_tracks.py`

**Configuration** (`config/pipeline/track_merging.yaml`):
```yaml
# Appearance clustering
similarity_threshold: 0.80      # ArcFace cosine similarity
linkage_method: average         # average|complete|single
min_cluster_size: 1             # Allow single-shot tracks

# Embedding extraction
embedding_sample_strategy: median  # median|mean|max_quality
min_embedding_quality: 0.5         # Filter low-quality chips

# Edge cases
max_temporal_gap_s: 30.0           # Max gap within scene before splitting
handle_confessionals: separate     # separate|merge
```

---

### Phase 4: Database Schema Updates

#### New Table: `scene`
```sql
CREATE TABLE scene (
    scene_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ep_id UUID NOT NULL REFERENCES episode(ep_id) ON DELETE CASCADE,
    start_s FLOAT NOT NULL,
    end_s FLOAT NOT NULL,
    start_frame INT NOT NULL,
    end_frame INT NOT NULL,
    scene_type TEXT,  -- dialogue, action, confessional, montage
    location_hint TEXT,  -- kitchen, pool, restaurant (optional)
    shot_cuts INT[],  -- Frame indices of shot boundaries within scene
    confidence FLOAT,
    meta JSONB,
    schema_version TEXT DEFAULT 'scene_v1',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_scene_ep_time ON scene(ep_id, start_s, end_s);
CREATE INDEX idx_scene_type ON scene(ep_id, scene_type);
```

#### Updated Table: `track`
```sql
-- Add scene_id and appearance_segments
ALTER TABLE track ADD COLUMN scene_id UUID REFERENCES scene(scene_id) ON DELETE CASCADE;
ALTER TABLE track ADD COLUMN appearance_segments JSONB;  -- [[start_frame, end_frame], ...]
ALTER TABLE track ADD COLUMN shot_track_ids UUID[];  -- IDs of merged shot tracks
ALTER TABLE track ADD COLUMN schema_version TEXT DEFAULT 'track_v2';

-- Update indices
CREATE INDEX idx_track_scene ON track(scene_id);
CREATE INDEX idx_track_ep_scene ON track(ep_id, scene_id);
```

#### Migration Script: `db/migrations/003_scene_scoped_tracks.sql`
```sql
-- Phase 1: Add scene table
-- Phase 2: Add columns to track table (nullable for backwards compat)
-- Phase 3: Backfill scene data for existing episodes (optional)
-- Phase 4: Update constraints and indices
```

---

## Implementation Phases

### Phase 1: Semantic Scene Detection (Weeks 1-2)
**Goal**: Produce `scenes.jsonl` for episodes

- [ ] Implement `detect_semantic_scenes()` in `episode_run.py`
- [ ] Add scene segmentation config (`config/pipeline/scene_segmentation.yaml`)
- [ ] Write tests for scene detection heuristics
- [ ] Add scene table to Postgres schema
- [ ] Update API to expose scene boundaries (`GET /api/episodes/{ep_id}/scenes`)

**Success Criteria**:
- Scene count ≈ 15-30 per reality TV episode (vs. 200+ shot cuts)
- Confessionals correctly identified as separate scenes
- Dialogues correctly grouped despite camera cuts

---

### Phase 2: Shot-Level Track Annotation (Week 3)
**Goal**: Add `scene_id` to existing tracks

- [ ] Update `TrackRecorder` to accept scene metadata
- [ ] Modify `tracks.jsonl` schema to include `scene_id`
- [ ] Rename output: `tracks.jsonl` → `shot_tracks.jsonl` (internal)
- [ ] Update artifact resolver paths
- [ ] Backwards compatibility: handle episodes without scenes

**Success Criteria**:
- All tracks have valid `scene_id` reference
- Track termination happens at scene boundaries
- No regression in existing tracking quality

---

### Phase 3: Scene-Scoped Track Merging (Weeks 4-5)
**Goal**: Merge shot tracks into scene tracks

- [ ] Implement `cluster_by_appearance()` using ArcFace embeddings
- [ ] Implement `merge_cluster_into_track()`
- [ ] Create `tools/merge_scene_tracks.py` script
- [ ] Add track merging config (`config/pipeline/track_merging.yaml`)
- [ ] Update `track` table with scene-scoped columns
- [ ] Write tests for merge logic and edge cases

**Success Criteria**:
- Track count reduced by ~95% (4,352 → ~100)
- Correct identity preservation within scenes
- No false merges (different people merged)
- Handle edge cases (person leaves/returns to scene)

---

### Phase 4: API & UI Updates (Week 6)
**Goal**: Expose scene-scoped tracks in API/UI

- [ ] Update API routes to return SSTs instead of SLTs
- [ ] Add `GET /api/scenes/{scene_id}/tracks` endpoint
- [ ] Update UI to show scene-grouped tracks
- [ ] Add scene boundary visualization in video player
- [ ] Update screen time calculation to use SSTs
- [ ] Migration guide for API consumers

**Success Criteria**:
- API returns scene-scoped tracks by default
- UI shows "Kitchen scene: Heather (45s), Shannon (38s), Tamra (12s)"
- Backwards compatibility flag: `?track_version=v1` for old behavior

---

### Phase 5: Testing & Validation (Week 7)
**Goal**: Comprehensive testing and quality assurance

- [ ] Integration tests: end-to-end scene detection → merging
- [ ] Regression tests: existing episodes produce valid SSTs
- [ ] Performance benchmarks: processing time impact
- [ ] Quality metrics: false merge rate, identity accuracy
- [ ] Documentation: migration guide, API changes, config options

**Success Criteria**:
- < 2% false merge rate (different people merged)
- < 5% identity drift rate (same person split)
- < 20% processing time increase
- All existing tests pass

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: Scene Detection                                        │
├─────────────────────────────────────────────────────────────────┤
│ video.mp4 → PySceneDetect → shot_cuts[]                        │
│                           → temporal_clustering()                │
│                           → scenes.jsonl                         │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: Shot-Level Tracking (current system + scene_id)       │
├─────────────────────────────────────────────────────────────────┤
│ video.mp4 + scenes.jsonl → detector (RetinaFace)               │
│                          → tracker (ByteTrack)                   │
│                          → shot_tracks.jsonl (scene_id added)   │
│                          → chips_manifest.parquet                │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: Embedding Extraction (unchanged)                       │
├─────────────────────────────────────────────────────────────────┤
│ chips/ → ArcFace → embeddings (stored in pgvector)             │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: Scene-Scoped Track Merging (NEW)                      │
├─────────────────────────────────────────────────────────────────┤
│ shot_tracks.jsonl + embeddings → cluster_by_appearance()       │
│                                 → merge_cluster_into_track()    │
│                                 → tracks.jsonl (scene-scoped)   │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: Identity Assignment (updated for SSTs)                │
├─────────────────────────────────────────────────────────────────┤
│ tracks.jsonl + facebank → assignment table (person_id)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Edge Cases & Handling

### 1. Person Leaves & Returns to Scene
**Scenario**: Heather exits kitchen at 00:42, returns at 00:58 (still same scene)

**Handling**:
- Two shot tracks: `st_042` (00:40-00:42), `st_058` (00:58-01:05)
- Clustering merges them into one SST (high appearance similarity)
- `appearance_segments`: `[[1200, 1260], [1740, 1890]]`
- **Config option**: `max_temporal_gap_s: 30.0` - if gap > threshold, split into separate SSTs

### 2. Identical Twins / Lookalikes
**Scenario**: Two cast members with similar facial features

**Handling**:
- Appearance clustering may incorrectly merge
- **Mitigation**: Lower `similarity_threshold` (0.80 → 0.75)
- **Fallback**: Spatial constraints (two people can't occupy same frame)
- **UI**: Flag low-confidence merges for manual review

### 3. Confessionals Intercut with Main Scene
**Scenario**: Kitchen argument intercut with Heather's confessional

**Handling**:
- Confessional detection: single-person + fixed angle + duration > 5s
- Treat confessionals as **separate scenes** (even if temporally intercut)
- Config: `handle_confessionals: separate` (vs. `merge`)

### 4. Montage Scenes (Rapid Cuts)
**Scenario**: Pool party montage with 2-3 second clips of different locations

**Handling**:
- Heuristic: If shots < 3s and no temporal gap, treat as one "montage scene"
- Label: `scene_type: montage`
- Config: `rapid_cut_window_s: 2.0`

### 5. Flashbacks
**Scenario**: Narrative cuts to flashback of earlier event

**Handling**:
- Treat flashbacks as **new scenes** (different temporal context)
- **Future**: OCR/subtitle analysis to detect "Three weeks earlier..." cues

### 6. Empty Scenes (No Faces)
**Scenario**: Establishing shot (building exterior, landscape)

**Handling**:
- Scene exists in `scene` table
- No tracks generated (zero shot tracks)
- Not an error condition

### 7. Occlusion & Partial Visibility
**Scenario**: Person visible but face occluded (back of head, profile)

**Handling**:
- Shot-level tracking handles occlusion (existing logic)
- Embedding quality filters remove low-quality chips
- Config: `min_embedding_quality: 0.5`

---

## Success Metrics

### Quantitative
- **Track reduction**: 95%+ (4,352 → ~100 tracks)
- **False merge rate**: < 2% (different people merged)
- **Identity drift rate**: < 5% (same person split into multiple SSTs)
- **Processing time**: < 20% increase over current pipeline
- **Scene detection accuracy**: 90%+ scenes correctly bounded

### Qualitative
- **UX improvement**: Reviewers report "easier to find scenes"
- **Semantic coherence**: Tracks align with human perception of scenes
- **Clustering improvement**: Better facebank diversity per identity

### Performance Benchmarks
- **Existing**: 45-min episode → 12 min processing (4,352 tracks)
- **Target**: 45-min episode → 14 min processing (100 scene-scoped tracks)
- **Breakdown**:
  - Scene detection: +1 min
  - Shot tracking: 7 min (unchanged)
  - Merging: +1 min
  - Embedding: 4 min (unchanged)
  - Total: 14 min

---

## Configuration

### `config/pipeline/scene_segmentation.yaml`
```yaml
# Scene segmentation configuration
detector: pyscenedetect  # pyscenedetect | internal | off
shot_threshold: 27.0
min_shot_len: 5

# Semantic scene grouping
enable_semantic_scenes: true
temporal_gap_threshold_s: 3.0
rapid_cut_window_s: 2.0
confessional_min_duration_s: 5.0
confessional_person_count: 1
scene_type_detection: heuristic  # heuristic | ml (future)

# Advanced (future)
visual_similarity_threshold: 0.85
enable_audio_scene_detection: false
```

### `config/pipeline/track_merging.yaml`
```yaml
# Track merging configuration
enable_scene_scoped_tracks: true  # Feature flag
similarity_threshold: 0.80
linkage_method: average
min_cluster_size: 1

# Embedding extraction
embedding_sample_strategy: median
min_embedding_quality: 0.5

# Edge cases
max_temporal_gap_s: 30.0
handle_confessionals: separate
split_on_large_gaps: true

# Quality thresholds
max_intra_cluster_variance: 0.25
warn_on_large_clusters: 10  # > 10 shot tracks per SST
```

---

## Migration Strategy

### Backwards Compatibility
1. **Dual mode operation**: Support both track_v1 (SLT) and track_v2 (SST)
2. **API versioning**: `/api/v2/tracks` returns SSTs, `/api/v1/tracks` returns SLTs
3. **Feature flag**: `enable_scene_scoped_tracks` in config (default: `false` initially)

### Migration Steps
1. **Phase 1-2 (non-breaking)**: Add scene detection + scene_id to tracks
2. **Phase 3 (additive)**: Introduce SST merging (optional, enabled via flag)
3. **Phase 4 (breaking)**: Default API to track_v2, deprecate track_v1
4. **Phase 5 (cleanup)**: Remove track_v1 support after 3-month transition

### Reprocessing Existing Episodes
```bash
# Backfill scenes for existing episodes
python tools/backfill_scenes.py --show rhoc --season 18

# Regenerate scene-scoped tracks
python tools/merge_scene_tracks.py --ep-id ep_12345

# Bulk reprocessing
python tools/bulk_reprocess.py --show rhoc --season 18 --track-version v2
```

---

## Testing Strategy

### Unit Tests
- [ ] `test_detect_semantic_scenes.py`: Scene boundary detection
- [ ] `test_cluster_by_appearance.py`: Embedding clustering
- [ ] `test_merge_cluster_into_track.py`: Track merging logic
- [ ] `test_scene_track_schema.py`: Schema validation

### Integration Tests
- [ ] `test_end_to_end_scene_tracks.py`: Full pipeline
- [ ] `test_scene_api.py`: API endpoint behavior
- [ ] `test_backwards_compat.py`: track_v1 vs track_v2 consistency

### Regression Tests
- [ ] `test_existing_episodes.py`: Existing episodes still process correctly
- [ ] `test_screen_time_consistency.py`: Screen time matches between v1 and v2

### Quality Assurance
- [ ] Manual review: Sample 10 episodes, verify scene boundaries
- [ ] Clustering validation: Check false merge/split rates
- [ ] Performance profiling: Memory usage, processing time

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Semantic scene detection inaccurate | High | Extensive testing, tunable thresholds, manual override UI |
| False merges (different people) | Critical | Conservative similarity threshold, spatial constraints, QA review |
| Processing time increase | Medium | Optimize clustering algorithm, parallel processing, caching |
| API breaking changes | High | Versioned API, 3-month deprecation period, migration tooling |
| Confessional misclassification | Medium | Heuristic tuning, confessional training data, manual labels |
| Backwards compatibility bugs | Medium | Comprehensive testing, feature flag, gradual rollout |

---

## Future Enhancements (Post-v1)

### Phase 6+: ML-Based Scene Segmentation
- Train scene classifier on labeled reality TV data
- Features: visual (place embeddings), audio (ambient sound), OCR (location text)
- Model: Transformer-based scene boundary detector

### Phase 7: Hierarchical Scene Structure
- **Scene groups**: Multi-scene sequences (e.g., "Dinner party" = kitchen + dining room + patio)
- **Subscenes**: Within-scene segments (e.g., "argument" vs. "reconciliation")
- Nested track representation

### Phase 8: Cross-Episode Scene Linking
- Detect recurring locations (e.g., "Lisa's kitchen")
- Link tracks across episodes in same location
- "Show me all scenes in Lisa's kitchen across Season 18"

### Phase 9: Audio-Visual Fusion for Scenes
- Speaker diarization to refine scene boundaries
- Music transitions as scene boundary cues
- Ambient sound classification (indoor/outdoor, party/quiet)

---

## Acceptance Criteria (ACCEPTANCE_MATRIX.md)

| Component | Status | Test Coverage | Owner | Notes |
|-----------|--------|---------------|-------|-------|
| Scene Detection | 🟡 PLANNED | TBD | TBA | PySceneDetect + temporal clustering |
| Shot Track Annotation | 🟡 PLANNED | TBD | TBA | Add scene_id to existing tracks |
| Appearance Clustering | 🟡 PLANNED | TBD | TBA | ArcFace-based merging |
| Scene-Scoped Track Merging | 🟡 PLANNED | TBD | TBA | Main merge logic |
| Database Schema Migration | 🟡 PLANNED | TBD | TBA | scene table + track columns |
| API v2 Endpoints | 🟡 PLANNED | TBD | TBA | /api/v2/tracks, /api/scenes |
| UI Scene Visualization | 🟡 PLANNED | TBD | TBA | Scene timeline, grouped tracks |
| Migration Tooling | 🟡 PLANNED | TBD | TBA | Backfill scripts |

---

## References

- **DATA_SCHEMA.md**: Track v1 schema, embedding schema
- **FEATURES/tracking/**: Existing ByteTrack implementation
- **tools/episode_run.py**: Current tracking pipeline (lines 3200-3350)
- **config/pipeline/tracking.yaml**: ByteTrack config
- **PySceneDetect docs**: https://scenedetect.com/

---

## Questions for Product/Engineering Review

1. **Scene definition priority**: Location-based vs. activity-based vs. narrative-based?
2. **API versioning**: Hard cutover date for track_v2 or permanent dual support?
3. **Manual override UI**: Should users be able to manually split/merge scenes?
4. **Confessional handling**: Always separate or allow merge via config?
5. **Performance budget**: Max acceptable processing time increase?
6. **Quality thresholds**: Acceptable false merge rate for v1 release?
