# Scene-Scoped Tracks Feature Plan

**Status:** PLANNING  
**Goal:** Redefine tracks to be scene-scoped rather than continuous detection sequences  
**Target:** Track = All appearances of ONE person within ONE scene/location

---

## Executive Summary

### Current Definition
- **Track**: Continuous sequence of detections for one person, terminated by:
  - Max gap threshold (frames without detection)
  - Scene cuts (hard visual cuts detected by PySceneDetect)
  - Appearance gate splits (identity mismatch)

### Proposed Definition
- **Track**: All appearances of ONE person within ONE semantic scene
- **Scene**: Filming location/activity (e.g., "Kitchen argument", "Pool party")
- **Termination**: ONLY scene boundaries (location/activity changes)

### Example Transformation

**Before (Current):**
```
Episode: 45 minutes, 4,352 tracks
- Track 1: Heather frames 120-145 (kitchen)
- Track 2: Heather frames 147-162 (kitchen, after cut)
- Track 3: Heather frames 165-180 (kitchen, after cut)
- Track 4: Lisa frames 120-138 (kitchen)
- Track 5: Lisa frames 140-155 (kitchen, after cut)
...
```

**After (Scene-Scoped):**
```
Episode: 45 minutes, ~100 tracks (20 scenes × 5 cast avg)
Scene: "Kitchen argument between Heather and Shannon"
- Track 1: All Heather frames (120-180, regardless of cuts)
- Track 2: All Shannon frames (120-195, regardless of cuts)
- Track 3: All Tamra reaction shots (125-170, regardless of cuts)
```

---

## Benefits Analysis

### ✅ Pros

1. **Semantic Coherence**
   - Tracks represent "appearances in a scene" rather than "continuous detection sequences"
   - Matches human mental model: "all Lisa faces from the pool party"

2. **Better for Reality TV**
   - Scenes are the natural unit of narrative
   - Easier to review "the dinner party scene" as cohesive units

3. **Reduced Fragmentation**
   - Current: 4,352 tracks for one episode
   - Expected: ~100 tracks (20 scenes × 5 cast members)
   - 40x reduction in track count

4. **Clustering Benefits**
   - Fewer, longer tracks = better embedding diversity per identity
   - More samples per person per scene = more robust prototypes
   - Reduced noise from fragmented tracks

5. **UI/UX Improvement**
   - Easier to review "all Lisa faces from the pool party" as one track
   - Scene-based navigation in UI
   - Better organization for manual review

6. **Assignment Accuracy**
   - Single track per person per scene = clearer identity assignments
   - Less ambiguity when linking to cast members

### ⚠️ Challenges

1. **Scene Boundary Detection**
   - Current PySceneDetect finds visual cuts (shot changes), not semantic scenes
   - Need to distinguish:
     - "Cut to different camera angle" (same scene)
     - "Cut to different location" (new scene)
   - Example: Kitchen argument might have 30 shot changes but 1 semantic scene

2. **Track Merging Logic**
   - Need to merge tracks WITHIN a scene that belong to same person
   - Requires appearance matching (ArcFace) across temporal gaps
   - "Heather at 00:12" + "Heather at 00:47" = same track if same scene

3. **Scene Definition Ambiguity**
   - What defines a scene?
     - Location change? (kitchen → pool)
     - Activity change? (cooking → eating)
     - Time jump? (flashback)
   - Edge cases:
     - Confessionals: separate scene or part of narrative scene?
     - Flashbacks: new scene or continuation?
     - Montages: one scene or multiple?

4. **Implementation Effort**
   - Need semantic scene segmentation (not just shot detection)
   - Need track merging post-processor
   - Need to handle edge cases (person leaves/returns to scene)
   - Schema changes required (add scene_id to tracks)

5. **Backward Compatibility**
   - Existing tracks don't have scene_id
   - Migration path needed
   - API changes required

---

## Technical Architecture

### Phase 1: Semantic Scene Detection

**Location:** `tools/episode_run.py` (enhance `detect_scene_cuts`)

**Current Implementation:**
```python
def detect_scene_cuts_pyscenedetect(video_path, threshold=27.0, min_len=1):
    # Returns list of frame indices where visual cuts occur
    # Example: [120, 145, 162, 180, 250, ...]
```

**Proposed Enhancement:**
```python
def detect_semantic_scenes(
    video_path: str | Path,
    fps_detected: float,
    stride: int = 1,
    *,
    shot_threshold: float = 27.0,
    shot_min_len: int = 1,
    scene_gap_threshold_s: float = 3.0,
    scene_min_duration_s: float = 5.0,
) -> list[dict]:
    """
    Detect semantic scenes by grouping shot cuts.
    
    Returns:
        List of scene dicts:
        {
            "scene_id": "scene_001",
            "start_frame": 120,
            "end_frame": 1850,
            "start_s": 4.0,
            "end_s": 61.7,
            "shot_cuts": [120, 145, 162, 180, ...],  # All cuts within scene
        }
    """
    # Step 1: Detect visual shot cuts (existing PySceneDetect)
    shot_cuts = detect_scene_cuts_pyscenedetect(
        video_path,
        threshold=shot_threshold,
        min_len=shot_min_len,
    )
    
    if not shot_cuts:
        # No cuts detected = single scene
        return [{
            "scene_id": "scene_001",
            "start_frame": 0,
            "end_frame": total_frames - 1,
            "shot_cuts": [],
        }]
    
    # Step 2: Group shots into semantic scenes
    scenes = []
    current_scene_shots = []
    current_start_frame = 0
    
    for i, cut_frame in enumerate(shot_cuts):
        current_scene_shots.append(cut_frame)
        
        # Check if next cut represents scene boundary
        next_cut_frame = shot_cuts[i + 1] if i + 1 < len(shot_cuts) else None
        
        if next_cut_frame:
            time_gap_s = (next_cut_frame - cut_frame) / fps_detected
            
            # Heuristic: Large temporal gap = likely scene change
            if time_gap_s > scene_gap_threshold_s:
                scene_end_frame = cut_frame
                scene_duration_s = (scene_end_frame - current_start_frame) / fps_detected
                
                # Only create scene if meets minimum duration
                if scene_duration_s >= scene_min_duration_s:
                    scenes.append({
                        "scene_id": f"scene_{len(scenes) + 1:03d}",
                        "start_frame": current_start_frame,
                        "end_frame": scene_end_frame,
                        "start_s": current_start_frame / fps_detected,
                        "end_s": scene_end_frame / fps_detected,
                        "shot_cuts": current_scene_shots.copy(),
                    })
                    current_start_frame = next_cut_frame
                    current_scene_shots = []
    
    # Final scene
    if current_start_frame < total_frames:
        scenes.append({
            "scene_id": f"scene_{len(scenes) + 1:03d}",
            "start_frame": current_start_frame,
            "end_frame": total_frames - 1,
            "shot_cuts": current_scene_shots,
        })
    
    return scenes
```

**Heuristics for Scene Boundaries:**
1. **Temporal gap**: >3 seconds between shot cuts = likely scene change
2. **Visual similarity**: (Future) Compare histograms/frames across gaps
3. **Audio cues**: (Future) Silence or audio change detection
4. **Location cues**: (Future) Object detection or scene classification

**Configuration:**
```yaml
# config/pipeline/tracking.yaml
scene_detection:
  shot_threshold: 27.0        # PySceneDetect threshold
  shot_min_len: 1             # Minimum shot length (frames)
  scene_gap_threshold_s: 3.0  # Seconds between cuts to consider scene boundary
  scene_min_duration_s: 5.0   # Minimum scene duration (seconds)
```

---

### Phase 2: Track Recording with Scene Context

**Location:** `tools/episode_run.py` (`TrackRecorder` class)

**Current Behavior:**
- `recorder.on_cut(frame_idx)` forces all tracks to terminate at scene cuts
- Tracks are fragmented across shot boundaries

**Proposed Behavior:**
- Track recording continues across shot cuts WITHIN same scene
- Only terminate tracks at semantic scene boundaries
- Add `scene_id` metadata to each track

**Changes:**
```python
class TrackRecorder:
    def __init__(self, *, max_gap: int, remap_ids: bool, scenes: list[dict]):
        self.max_gap = max(1, int(max_gap))
        self.remap_ids = remap_ids
        self.scenes = scenes  # List of scene dicts from detect_semantic_scenes
        self._scene_index = self._build_scene_index(scenes)
        # ... rest of init
    
    def _build_scene_index(self, scenes: list[dict]) -> dict[int, str]:
        """Map frame_idx -> scene_id"""
        index = {}
        for scene in scenes:
            for frame in range(scene["start_frame"], scene["end_frame"] + 1):
                index[frame] = scene["scene_id"]
        return index
    
    def _get_scene_id(self, frame_idx: int) -> str | None:
        """Get scene_id for given frame"""
        return self._scene_index.get(frame_idx)
    
    def record(
        self,
        *,
        tracker_track_id: int,
        frame_idx: int,
        ts: float,
        bbox: list[float] | np.ndarray,
        class_label: int | str,
        landmarks: list[float] | None = None,
        confidence: float | None = None,
        force_new_track: bool = False,
    ) -> int:
        current_scene_id = self._get_scene_id(frame_idx)
        
        # Check if we've crossed a scene boundary
        mapping = self._mapping.get(tracker_track_id)
        if mapping:
            previous_scene_id = mapping.get("scene_id")
            if previous_scene_id and previous_scene_id != current_scene_id:
                # Scene boundary crossed - force new track
                self._complete_track(mapping["export_id"])
                force_new_track = True
        
        # ... rest of record logic (existing max_gap handling)
        
        # Store scene_id in mapping
        if mapping:
            mapping["scene_id"] = current_scene_id
        
        # ... rest of logic
    
    def rows(self) -> list[dict]:
        """Export tracks with scene_id metadata"""
        payload: list[dict] = []
        for track in sorted(self._accumulators.values(), key=lambda item: item.track_id):
            row = track.to_row()
            # Add scene_id from first frame
            first_frame = track.first_frame_idx
            scene_id = self._get_scene_id(first_frame)
            row["scene_id"] = scene_id
            payload.append(row)
        return payload
```

**Track Schema Update:**
```json
{
  "track_id": 42,
  "ep_id": "ep_123",
  "scene_id": "scene_005",
  "start_s": 12.5,
  "end_s": 47.3,
  "frame_span": [375, 1419],
  "sample_thumbs": ["track_0042/thumb_0375.jpg", ...],
  "schema_version": "track_v2"  // Bump version
}
```

---

### Phase 3: Post-Processing Track Merger

**Location:** New file `tools/merge_scene_tracks.py`

**Purpose:** Merge tracks within same scene that belong to same person

**Algorithm:**
1. Load existing tracks and faces
2. Group tracks by scene_id
3. For each scene:
   - Extract embeddings from all tracks
   - Cluster by identity (ArcFace similarity)
   - Merge tracks within same cluster
   - Assign new track_ids

**Implementation:**
```python
def merge_tracks_by_scene(
    tracks_path: Path,
    faces_path: Path,
    *,
    similarity_threshold: float = 0.85,
    min_track_frames: int = 5,
) -> tuple[list[dict], list[dict]]:
    """
    Merge tracks within scenes that belong to same person.
    
    Returns:
        (merged_tracks, track_mapping)
        track_mapping: {old_track_id: new_track_id}
    """
    tracks = load_tracks(tracks_path)
    faces = load_faces(faces_path)
    
    # Group tracks by scene_id
    tracks_by_scene: dict[str, list[dict]] = defaultdict(list)
    for track in tracks:
        scene_id = track.get("scene_id")
        if scene_id:
            tracks_by_scene[scene_id].append(track)
    
    merged_tracks = []
    track_mapping: dict[int, int] = {}
    next_track_id = max(t["track_id"] for t in tracks) + 1
    
    for scene_id, scene_tracks in tracks_by_scene.items():
        if len(scene_tracks) < 2:
            # Single track in scene - no merging needed
            merged_tracks.extend(scene_tracks)
            continue
        
        # Extract embeddings for all tracks in scene
        track_embeddings: dict[int, np.ndarray] = {}
        for track in scene_tracks:
            track_id = track["track_id"]
            # Get best embedding from track (from faces.jsonl)
            track_faces = [f for f in faces if f.get("track_id") == track_id]
            if track_faces:
                # Use highest quality embedding
                best_face = max(track_faces, key=lambda f: f.get("quality", 0.0))
                if "embedding" in best_face:
                    track_embeddings[track_id] = np.array(best_face["embedding"])
        
        if len(track_embeddings) < 2:
            # Not enough embeddings to cluster
            merged_tracks.extend(scene_tracks)
            continue
        
        # Cluster tracks by identity
        identity_groups = cluster_tracks_by_similarity(
            track_embeddings,
            threshold=similarity_threshold,
        )
        
        # Merge tracks within each identity group
        for group_track_ids in identity_groups:
            if len(group_track_ids) == 1:
                # Single track - no merge needed
                merged_tracks.append(scene_tracks[0])
                continue
            
            # Merge multiple tracks
            group_tracks = [t for t in scene_tracks if t["track_id"] in group_track_ids]
            merged = merge_track_group(group_tracks)
            merged["track_id"] = next_track_id
            merged["scene_id"] = scene_id
            
            # Update mapping
            for old_track_id in group_track_ids:
                track_mapping[old_track_id] = next_track_id
            
            merged_tracks.append(merged)
            next_track_id += 1
    
    return merged_tracks, track_mapping


def cluster_tracks_by_similarity(
    track_embeddings: dict[int, np.ndarray],
    threshold: float = 0.85,
) -> list[list[int]]:
    """
    Cluster tracks by ArcFace embedding similarity.
    
    Returns:
        List of clusters, each cluster is list of track_ids
    """
    track_ids = list(track_embeddings.keys())
    if len(track_ids) < 2:
        return [[tid] for tid in track_ids]
    
    # Compute similarity matrix
    similarities = {}
    for i, tid1 in enumerate(track_ids):
        for tid2 in track_ids[i+1:]:
            emb1 = track_embeddings[tid1]
            emb2 = track_embeddings[tid2]
            sim = cosine_similarity(emb1, emb2)
            similarities[(tid1, tid2)] = sim
    
    # Simple agglomerative clustering
    clusters = [[tid] for tid in track_ids]
    
    while True:
        # Find closest pair of clusters
        max_sim = -1.0
        merge_pair = None
        
        for i, cluster1 in enumerate(clusters):
            for j, cluster2 in enumerate(clusters[i+1:], i+1):
                # Average linkage: max similarity between any pair
                cluster_sim = max(
                    similarities.get((tid1, tid2), 0.0)
                    for tid1 in cluster1
                    for tid2 in cluster2
                    if (tid1, tid2) in similarities or (tid2, tid1) in similarities
                )
                
                if cluster_sim > max_sim:
                    max_sim = cluster_sim
                    merge_pair = (i, j)
        
        if max_sim < threshold or merge_pair is None:
            break
        
        # Merge clusters
        i, j = merge_pair
        clusters[i].extend(clusters[j])
        clusters.pop(j)
    
    return clusters


def merge_track_group(tracks: list[dict]) -> dict:
    """Merge multiple tracks into single track."""
    if len(tracks) == 1:
        return tracks[0].copy()
    
    # Combine frame spans
    all_frames = []
    for track in tracks:
        frame_span = track.get("frame_span", [])
        if isinstance(frame_span, list) and len(frame_span) == 2:
            all_frames.extend(range(frame_span[0], frame_span[1] + 1))
    
    if not all_frames:
        # Fallback to time-based
        start_s = min(t.get("start_s", 0.0) for t in tracks)
        end_s = max(t.get("end_s", 0.0) for t in tracks)
        return {
            "start_s": start_s,
            "end_s": end_s,
            "frame_span": [],  # Will be filled later
        }
    
    merged = {
        "start_s": min(t.get("start_s", 0.0) for t in tracks),
        "end_s": max(t.get("end_s", 0.0) for t in tracks),
        "frame_span": [min(all_frames), max(all_frames)],
        "sample_thumbs": [],  # Combine thumbnails from all tracks
    }
    
    # Combine sample thumbnails
    for track in tracks:
        thumbs = track.get("sample_thumbs", [])
        if isinstance(thumbs, list):
            merged["sample_thumbs"].extend(thumbs)
    
    # Deduplicate and limit thumbnails
    merged["sample_thumbs"] = list(set(merged["sample_thumbs"]))[:10]
    
    return merged
```

**CLI Integration:**
```bash
# Merge tracks after detection/tracking
python tools/merge_scene_tracks.py \
  --ep-id EP123 \
  --similarity-threshold 0.85 \
  --min-track-frames 5
```

---

### Phase 4: Schema Updates

**Artifact Schema:** `tracks.jsonl`

**Current Schema (track_v1):**
```json
{
  "track_id": 42,
  "ep_id": "ep_123",
  "start_s": 12.5,
  "end_s": 47.3,
  "frame_span": [375, 1419],
  "sample_thumbs": ["track_0042/thumb_0375.jpg"],
  "schema_version": "track_v1"
}
```

**New Schema (track_v2):**
```json
{
  "track_id": 42,
  "ep_id": "ep_123",
  "scene_id": "scene_005",
  "start_s": 12.5,
  "end_s": 47.3,
  "frame_span": [375, 1419],
  "sample_thumbs": ["track_0042/thumb_0375.jpg"],
  "schema_version": "track_v2"
}
```

**Database Schema:** `track` table

**Current:**
```sql
CREATE TABLE track (
    track_id uuid PRIMARY KEY,
    ep_id uuid REFERENCES episode(ep_id),
    start_s float,
    end_s float,
    occlusion_rate float,
    quality_stats_json jsonb,
    ...
);
```

**Proposed:**
```sql
ALTER TABLE track ADD COLUMN scene_id text;
CREATE INDEX idx_track_scene ON track(ep_id, scene_id);
```

**Migration Script:**
```python
# db/migrations/004_add_scene_id_to_tracks.sql
ALTER TABLE track ADD COLUMN scene_id text;
CREATE INDEX idx_track_scene ON track(ep_id, scene_id);

-- Backfill: Set scene_id to NULL for existing tracks (v1 tracks)
-- They can be reprocessed with scene detection to populate scene_id
```

---

### Phase 5: API Updates

**Location:** `apps/api/routers/` and `apps/api/services/`

**Endpoints Affected:**
1. `GET /api/episodes/{ep_id}/tracks` - Filter by scene_id
2. `GET /api/episodes/{ep_id}/scenes` - New endpoint
3. `GET /api/episodes/{ep_id}/tracks/{track_id}` - Include scene_id

**New Endpoint:**
```python
@router.get("/episodes/{ep_id}/scenes")
async def get_episode_scenes(ep_id: str):
    """Get all scenes for an episode."""
    scenes = storage_service.get_scenes(ep_id)
    return {"scenes": scenes}
```

**Updated Endpoint:**
```python
@router.get("/episodes/{ep_id}/tracks")
async def get_episode_tracks(
    ep_id: str,
    scene_id: str | None = None,  # New filter
    person_id: str | None = None,
):
    """Get tracks for an episode, optionally filtered by scene."""
    tracks = storage_service.get_tracks(ep_id, scene_id=scene_id, person_id=person_id)
    return {"tracks": tracks}
```

---

### Phase 6: UI Updates

**Location:** `apps/workspace-ui/`

**Changes:**
1. Scene-based navigation
2. Filter tracks by scene
3. Display scene_id in track metadata
4. Scene timeline visualization

**New UI Components:**
- Scene list sidebar
- Scene timeline view
- Track grouping by scene

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Implement `detect_semantic_scenes()` function
- [ ] Add scene detection configuration to `config/pipeline/tracking.yaml`
- [ ] Update `episode_run.py` to call semantic scene detection
- [ ] Add unit tests for scene detection
- [ ] Document scene detection heuristics

### Phase 2: Track Recording (Week 2-3)
- [ ] Update `TrackRecorder` to accept scenes
- [ ] Modify `record()` to check scene boundaries
- [ ] Add `scene_id` to track output
- [ ] Update track schema to `track_v2`
- [ ] Add integration tests

### Phase 3: Post-Processing (Week 3-4)
- [ ] Create `tools/merge_scene_tracks.py`
- [ ] Implement track clustering by identity
- [ ] Implement track merging logic
- [ ] Add CLI interface
- [ ] Add tests for merging

### Phase 4: Schema Migration (Week 4)
- [ ] Create database migration for `scene_id` column
- [ ] Update artifact schema documentation
- [ ] Add backward compatibility handling
- [ ] Migration script for existing tracks

### Phase 5: API Updates (Week 5)
- [ ] Add `scene_id` to track endpoints
- [ ] Create `/scenes` endpoint
- [ ] Update API documentation
- [ ] Add API tests

### Phase 6: UI Updates (Week 5-6)
- [ ] Add scene navigation
- [ ] Filter tracks by scene
- [ ] Display scene metadata
- [ ] Scene timeline visualization

### Phase 7: Testing & Validation (Week 6-7)
- [ ] End-to-end tests
- [ ] Performance benchmarks
- [ ] Accuracy validation
- [ ] User acceptance testing

---

## Edge Cases & Considerations

### 1. Scene Boundary Ambiguity

**Problem:** What if a person appears in two adjacent scenes?

**Solution:** Create separate tracks per scene. If same person appears in scene_001 and scene_002, they get two tracks.

**Example:**
```
Scene 001: Kitchen argument (frames 0-500)
  Track 1: Heather (frames 50-480)

Scene 002: Pool party (frames 501-1000)
  Track 2: Heather (frames 520-950)  # Different track, same person
```

### 2. Person Leaves and Returns to Scene

**Problem:** Heather leaves kitchen (frame 200), returns later (frame 400). Same scene.

**Solution:** Track merger will combine these into single track if similarity is high enough.

**Example:**
```
Scene 001: Kitchen argument
  Track 1: Heather (frames 50-200)   # Initial appearance
  Track 2: Heather (frames 400-480)  # Return appearance
  → Merged: Track 1 (frames 50-480)  # Combined by post-processor
```

### 3. Confessionals

**Problem:** Confessionals are separate from narrative scenes but may appear interleaved.

**Options:**
- **Option A:** Treat confessionals as separate scenes
- **Option B:** Include confessionals in narrative scene if temporally close

**Recommendation:** Option A (separate scenes) for semantic clarity.

### 4. Flashbacks

**Problem:** Flashback to earlier scene - new scene or continuation?

**Solution:** Treat as new scene if location/context changes significantly.

### 5. Montages

**Problem:** Rapid cuts between multiple locations/activities.

**Solution:** Group by dominant location or treat as single "montage" scene.

### 6. Empty Scenes

**Problem:** Scene with no face detections.

**Solution:** Still create scene record, but no tracks. Useful for scene navigation.

### 7. Very Long Scenes

**Problem:** Scene lasts 10+ minutes with many people.

**Solution:** Still single track per person per scene. Post-processor handles merging.

### 8. Scene Detection Failures

**Problem:** Scene detection misses a boundary or creates false boundaries.

**Solution:**
- Provide manual scene editing in UI
- Allow reprocessing with different thresholds
- Store scene detection confidence scores

---

## Configuration

### Scene Detection Parameters

```yaml
# config/pipeline/tracking.yaml
scene_detection:
  enabled: true
  method: "semantic"  # or "shot_only" for backward compatibility
  
  # Shot detection (PySceneDetect)
  shot_threshold: 27.0
  shot_min_len: 1
  
  # Semantic grouping
  scene_gap_threshold_s: 3.0    # Seconds between cuts = scene boundary
  scene_min_duration_s: 5.0      # Minimum scene duration (seconds)
  
  # Future enhancements
  use_visual_similarity: false   # Compare histograms across gaps
  use_audio_cues: false          # Detect audio changes
```

### Track Merging Parameters

```yaml
# config/pipeline/tracking.yaml
track_merging:
  enabled: true
  similarity_threshold: 0.85      # ArcFace cosine similarity
  min_track_frames: 5             # Minimum frames to merge
  max_gap_frames: 300             # Max frames between tracks to merge (10s at 30fps)
```

---

## Migration Strategy

### For New Episodes
- Use scene-scoped tracks by default
- No migration needed

### For Existing Episodes

**Option 1: Reprocess**
- Re-run `detect track` with scene detection enabled
- Tracks will be regenerated with scene_id

**Option 2: Post-Process**
- Run scene detection on existing video
- Run track merger to assign scene_id to existing tracks
- Update database and artifacts

**Option 3: Hybrid**
- Keep existing tracks (scene_id = NULL)
- New processing uses scene-scoped tracks
- UI handles both v1 and v2 tracks

**Recommendation:** Option 3 (hybrid) for backward compatibility.

---

## Testing Strategy

### Unit Tests
- `test_scene_detection.py`: Test semantic scene detection
- `test_track_merging.py`: Test track clustering and merging
- `test_track_recorder_scenes.py`: Test scene-aware track recording

### Integration Tests
- End-to-end: `detect track` → scene detection → track recording → merging
- Validate track counts: Should be ~40x fewer tracks
- Validate track quality: Single person per track per scene

### Performance Tests
- Scene detection overhead: <5% of total processing time
- Track merging overhead: <10% of total processing time
- Memory usage: Track merging may require loading all embeddings

### Accuracy Tests
- Scene boundary detection: Compare manual annotations vs. automatic
- Track merging: Verify same-person tracks are merged correctly
- Track splitting: Verify different-person tracks are NOT merged

---

## Success Metrics

### Quantitative
- **Track count reduction**: Target 40x reduction (4,352 → ~100)
- **Scene detection accuracy**: >90% agreement with manual annotations
- **Track merging accuracy**: >95% correct merges (no false positives)
- **Performance impact**: <15% slowdown vs. current implementation

### Qualitative
- **Semantic coherence**: Tracks represent "person in scene" not "continuous detection"
- **UI usability**: Easier to navigate and review tracks by scene
- **Clustering quality**: Better identity assignments due to longer tracks

---

## Future Enhancements

### Phase 2 Features (Post-MVP)

1. **Visual Scene Classification**
   - Use object detection to identify locations (kitchen, pool, etc.)
   - Use scene classification models (Indoor/Outdoor, etc.)

2. **Audio-Based Scene Detection**
   - Detect audio changes (music, silence, dialogue)
   - Use audio features to refine scene boundaries

3. **Manual Scene Editing**
   - UI to manually adjust scene boundaries
   - Merge/split scenes based on user feedback

4. **Scene Metadata**
   - Store scene descriptions ("Kitchen argument")
   - Link scenes to cast members present
   - Scene-level screen time calculations

5. **Advanced Merging**
   - Temporal ordering: Merge tracks in chronological order
   - Quality weighting: Prefer high-quality embeddings for merging
   - Confidence scores: Track merging confidence

---

## Related Documentation

- `docs/tracking-accuracy-tuning.md` - Current tracking configuration
- `FEATURES/tracking/docs/PROGRESS.md` - Tracking feature progress
- `DATA_SCHEMA.md` - Current data schema
- `tools/episode_run.py` - Main detection/tracking pipeline

---

## Open Questions

1. **Scene naming**: Auto-generate scene names or require manual labeling?
2. **Scene duration**: What's the minimum/maximum scene duration?
3. **Overlapping scenes**: Can scenes overlap (e.g., split-screen)?
4. **Scene hierarchy**: Support sub-scenes (e.g., "Kitchen argument" → "Heather confronts Shannon")?
5. **Performance**: Is track merging fast enough for real-time processing?

---

## Appendix: Example Implementation

See `tools/episode_run.py` lines 3230-3278 for current scene detection integration point.

See `tools/merge_scene_tracks.py` (to be created) for track merging implementation.
