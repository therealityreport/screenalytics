# Scene-Scoped Tracks Feature

**Redefining tracks from detection sequences to scene-scoped person appearances**

---

## Quick Links

- **[Feature Specification](FEATURE.md)** - Full technical specification
- **[Implementation Outline](docs/IMPLEMENTATION_OUTLINE.md)** - Detailed implementation plan
- **[Terminology Changes](docs/TERMINOLOGY.md)** - Naming conventions and migration
- **[TODO Checklist](TODO.md)** - Implementation tasks and status
- **[Progress Tracking](docs/PROGRESS.md)** - Current progress and metrics

---

## TL;DR

**Problem**: Current tracks are continuous detection sequences → 4,352 tracks/episode (fragmented, poor UX)

**Solution**: Redefine tracks as scene-scoped person appearances → ~100 tracks/episode (semantic, better UX)

**Impact**:
- ✅ 95% track reduction (4,352 → 100)
- ✅ Tracks align with human perception of scenes
- ✅ Better clustering for identity assignment
- ✅ Easier review: "Show all Heather faces from kitchen scene"

---

## Concept

### Current: Track = Continuous Detection Sequence
```
[Frame 100-120: Heather visible]  → Track #42
[Frame 121-140: Cut to Shannon]
[Frame 141-160: Cut back to Heather]  → Track #87 (NEW TRACK!)
[Frame 161-180: Heather visible]
```
**Result**: 2 tracks for same person in same scene 😞

### NEW: Track = Scene-Scoped Person Appearance
```
Scene: "Kitchen argument between Heather and Shannon"
├─ Track #1: ALL Heather frames (100-120, 141-180) ✅
└─ Track #2: ALL Shannon frames (121-140) ✅
```
**Result**: 1 track per person per scene, regardless of cuts 🎉

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│ Phase 1: Semantic Scene Detection              │
│ video.mp4 → PySceneDetect → scenes.jsonl       │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ Phase 2: Shot-Level Tracking                   │
│ video + scenes → ByteTrack → shot_tracks.jsonl │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ Phase 3: Appearance Clustering & Merging       │
│ shot_tracks + embeddings → tracks.jsonl (SSTs) │
└─────────────────────────────────────────────────┘
```

---

## Key Terminology

| Term | Definition | Example |
|------|------------|---------|
| **Scene-Scoped Track (SST)** | All appearances of ONE person in ONE scene | "All Heather frames in kitchen argument" |
| **Semantic Scene** | Location/activity unit (human perception) | "Kitchen argument" (20-120s) |
| **Shot-Level Track (SLT)** | Continuous detection sequence (old "track") | Single camera angle appearance (0.5-5s) |
| **Appearance Segment** | Visible interval within SST | `[frame_100, frame_120]` |

---

## Implementation Timeline

| Phase | Duration | Description | Status |
|-------|----------|-------------|--------|
| **Phase 1** | Weeks 1-2 | Semantic scene detection | 🔵 Not Started |
| **Phase 2** | Week 3 | Add scene_id to tracks | 🔵 Not Started |
| **Phase 3** | Weeks 4-5 | Appearance clustering & merging | 🔵 Not Started |
| **Phase 4** | Week 6 | API & UI updates | 🔵 Not Started |
| **Phase 5** | Week 7 | Testing & validation | 🔵 Not Started |

**Total**: ~7 weeks

---

## Example Output

### Scene Record (`scenes.jsonl`)
```json
{
  "scene_id": "scene_005",
  "ep_id": "ep_12345",
  "start_s": 40.0,
  "end_s": 61.67,
  "scene_type": "dialogue",
  "location_hint": "kitchen",
  "shot_cuts": [1200, 1245, 1298, 1350, 1405],
  "schema_version": "scene_v1"
}
```

### Scene-Scoped Track Record (`tracks.jsonl`)
```json
{
  "track_id": "scene_005_p0",
  "scene_id": "scene_005",
  "person_id": "heather_dubrow",
  "scene_start_s": 40.0,
  "scene_end_s": 61.67,
  "appearance_segments": [
    [1200, 1300],  // First appearance
    [1400, 1500],  // Returns after cut
    [1700, 1850]   // Final appearance
  ],
  "total_visible_s": 15.0,
  "shot_track_ids": [42, 87, 103],
  "schema_version": "track_v2"
}
```

---

## Success Metrics

### Quantitative
- ✅ **Track reduction**: 95%+ (4,352 → ~100)
- ✅ **False merge rate**: < 2%
- ✅ **Identity drift rate**: < 5%
- ✅ **Processing time**: < 20% increase
- ✅ **Scene accuracy**: 90%+ correct boundaries

### Qualitative
- ✅ Easier scene navigation in UI
- ✅ Tracks align with human perception
- ✅ Better embedding diversity for clustering

---

## Configuration

### Scene Segmentation (`config/pipeline/scene_segmentation.yaml`)
```yaml
detector: pyscenedetect
temporal_gap_threshold_s: 3.0
rapid_cut_window_s: 2.0
confessional_min_duration_s: 5.0
```

### Track Merging (`config/pipeline/track_merging.yaml`)
```yaml
enable_scene_scoped_tracks: true
similarity_threshold: 0.80
linkage_method: average
max_temporal_gap_s: 30.0
```

---

## API Changes

### New Endpoints
- `GET /api/episodes/{ep_id}/scenes` - List scenes
- `GET /api/scenes/{scene_id}/tracks` - List tracks in scene

### Modified Endpoints
- `GET /api/episodes/{ep_id}/tracks` - Returns SSTs by default
  - Add `?track_version=v1` for old behavior (SLTs)

---

## Database Schema

### New Table: `scene`
```sql
CREATE TABLE scene (
    scene_id UUID PRIMARY KEY,
    ep_id UUID REFERENCES episode(ep_id),
    start_s FLOAT,
    end_s FLOAT,
    scene_type TEXT,
    location_hint TEXT,
    shot_cuts INT[],
    ...
);
```

### Updated Table: `track`
```sql
ALTER TABLE track 
    ADD COLUMN scene_id UUID REFERENCES scene(scene_id),
    ADD COLUMN appearance_segments JSONB,
    ADD COLUMN shot_track_ids UUID[];
```

---

## Edge Cases Handled

| Edge Case | Handling |
|-----------|----------|
| Person leaves & returns | Appearance segments track gaps |
| Identical twins | Lower similarity threshold, spatial constraints |
| Confessionals | Detected as separate scenes |
| Montage scenes | Grouped by rapid cut heuristic |
| Flashbacks | Treated as new scenes |

---

## Testing Strategy

### Unit Tests
- Scene detection heuristics
- Appearance clustering
- Track merging logic

### Integration Tests
- End-to-end pipeline
- API endpoint behavior
- UI scene visualization

### Regression Tests
- Existing episodes still process
- Screen time consistency

### Performance Tests
- Processing time benchmarks
- Memory usage profiling

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Scene detection inaccurate | Tunable thresholds, manual override UI |
| False merges (different people) | Conservative similarity threshold |
| Processing time increase | Optimize clustering, parallel processing |
| API breaking changes | Versioned API, 3-month deprecation |

---

## Questions for Review

1. **Scene definition**: Location-based vs. activity-based vs. narrative-based?
2. **API versioning**: Hard cutover or permanent dual support?
3. **Manual override**: UI for manual scene boundary editing?
4. **Performance budget**: Max acceptable processing time increase?
5. **Quality thresholds**: Acceptable false merge rate for v1?

---

## Getting Started (For Contributors)

### Prerequisites
- Existing Screenalytics setup
- Python 3.10+
- PostgreSQL with pgvector
- PySceneDetect installed

### Development Setup
```bash
# Clone feature branch
git checkout -b scene-scoped-tracks

# Install dependencies
pip install -r requirements.txt

# Run scene detection tests
pytest tests/ml/test_semantic_scene_detection.py
```

### Running the Pipeline
```bash
# Phase 1: Detect scenes
python tools/episode_run.py --ep-id ep_12345 --enable-scene-detection

# Phase 3: Merge tracks
python tools/merge_scene_tracks.py --ep-id ep_12345
```

---

## Related Features

- **FEATURES/tracking/** - ByteTrack implementation (shot-level tracking)
- **FEATURES/detection/** - RetinaFace detection
- **FEATURES/identity/** - ArcFace embedding and identity assignment

---

## Status: PLANNING

**Created**: 2025-11-19  
**Owner**: TBA  
**Target**: 7 weeks from start  
**Next Milestone**: Complete Phase 1 (Semantic Scene Detection)

---

## Contact

- **Feature Owner**: TBA
- **Reviewers**: TBA
- **Slack**: TBA

---

## License

See LICENSE file at repository root.
