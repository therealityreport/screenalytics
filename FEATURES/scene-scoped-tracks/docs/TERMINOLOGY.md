# Terminology Changes: Scene-Scoped Tracks

**Version**: 1.0  
**Status**: DRAFT  
**Last Updated**: 2025-11-19

---

## Overview

This document defines the terminology changes required for the Scene-Scoped Tracks feature. It provides clear definitions, renaming mappings, and migration guidance.

---

## NEW Terminology

### 1. Scene-Scoped Track (SST)
**Definition**: All appearances of **one person** within **one semantic scene**, regardless of camera cuts, temporal gaps, or occlusions.

**Key Properties**:
- Bounded by semantic scene boundaries (location/activity changes)
- May contain temporal gaps (person leaves and returns)
- Represents all visible intervals for one identity within a scene
- Typical duration: 10-90 seconds
- Typical count per episode: ~100 tracks (20 scenes × 5 people)

**Example**:
```json
{
  "track_id": "scene_005_p0",
  "scene_id": "scene_005",
  "ep_id": "ep_12345",
  "person_id": "heather_dubrow",
  "scene_start_s": 40.0,
  "scene_end_s": 61.67,
  "appearance_segments": [
    [1200, 1300],  // 40.0s - 43.3s (visible)
    [1400, 1500],  // 46.7s - 50.0s (returns after cut)
    [1700, 1850]   // 56.7s - 61.7s (final appearance)
  ],
  "total_visible_s": 15.0,  // Sum of visible time
  "shot_track_ids": [42, 87, 103],  // Merged shot tracks
  "schema_version": "track_v2"
}
```

**Use Cases**:
- UI: "Show me all of Heather's appearances in the kitchen argument scene"
- Screen time: Calculate per-scene screen time
- Review: "Review all faces from the pool party scene"
- Clustering: Better embedding diversity for identity assignment

---

### 2. Semantic Scene
**Definition**: A contiguous video segment representing a single location/activity unit, regardless of internal camera cuts.

**Boundaries**: Location change, activity change, time jump, or confessional transition.

**Key Properties**:
- Contains multiple shots (camera angles)
- Represents human-perceived scene unity
- Typical duration: 20-120 seconds
- Typical count per episode: 15-30 scenes

**Scene Types**:
- `dialogue`: Conversation between 2+ people (e.g., kitchen argument)
- `confessional`: Single person, fixed camera, commentary
- `action`: Physical activity (e.g., pool party, shopping)
- `montage`: Rapid cuts across multiple locations
- `establishing`: Location-only shot (no people)

**Example**:
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

**Scene Boundary Heuristics**:
1. **Location change**: Kitchen → pool (hard boundary)
2. **Activity change**: Cooking → argument (soft boundary)
3. **Time jump**: Black screen > 3 seconds (hard boundary)
4. **Confessional**: Single person, fixed camera (hard boundary)
5. **Montage end**: Rapid cuts → sustained shot (soft boundary)

---

### 3. Shot-Level Track (SLT)
**Definition**: Continuous detection sequence where a person is visible without occlusion or temporal gap (current "track" concept).

**Renamed From**: Track (track_v1)

**Key Properties**:
- Bounded by occlusion, scene cuts, or tracking loss
- No temporal gaps allowed
- Represents uninterrupted detection sequence
- Typical duration: 0.5-5 seconds
- Typical count per episode: 4,000-6,000 tracks

**Status**: **Internal intermediate representation**. Not exposed in final API/UI.

**Purpose**: Building block for Scene-Scoped Tracks. ByteTrack produces SLTs, which are then merged into SSTs.

**Example**:
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

---

### 4. Appearance Segment
**Definition**: A continuous interval `[start_frame, end_frame]` where a person is visible within a scene.

**Purpose**: Represent temporal gaps within Scene-Scoped Tracks.

**Example**:
A person appears in frames 100-200, disappears (cut to another person), then reappears in frames 350-450.

Scene-Scoped Track:
```json
{
  "track_id": "scene_002_p1",
  "appearance_segments": [
    [100, 200],  // First appearance
    [350, 450]   // Second appearance
  ],
  "total_visible_s": 6.67  // (100 + 100) frames / 30 fps
}
```

---

## RENAMED Terminology

| Old Term (v1) | New Term (v2) | Status | Notes |
|---------------|---------------|--------|-------|
| **Track** | **Shot-Level Track (SLT)** | Internal only | Not exposed in API/UI |
| `tracks.jsonl` | `shot_tracks.jsonl` | Internal artifact | Intermediate representation |
| `track_id` | `shot_track_id` | Internal only | Renamed for clarity |
| `track_v1` schema | `shot_track_v1` schema | Internal only | Backward compat maintained |

---

## DEPRECATED Terminology

### 1. "Track" (unqualified)
**Status**: AMBIGUOUS after this feature. Always specify:
- "Scene-Scoped Track" (SST) for user-facing contexts
- "Shot-Level Track" (SLT) for internal contexts

**Migration**:
- API responses: Return SSTs, label as `track` (default)
- API query param: `?track_version=v1` returns SLTs (labeled as `shot_track`)
- Internal code: Use `sst` or `slt` variable names for clarity

---

## Schema Version Evolution

### track_v1 (Current)
```json
{
  "track_id": 42,
  "ep_id": "ep_12345",
  "start_s": 40.5,
  "end_s": 43.2,
  "frame_span": [1215, 1296],
  "sample_thumbs": ["chips/track_42_frame_1250.jpg"],
  "schema_version": "track_v1"
}
```

**Characteristics**:
- No scene reference
- Continuous sequence only (no gaps)
- No appearance segments

---

### shot_track_v1 (Phase 2: Add scene_id)
```json
{
  "shot_track_id": 42,
  "scene_id": "scene_005",  // NEW
  "ep_id": "ep_12345",
  "start_s": 40.5,
  "end_s": 43.2,
  "frame_span": [1215, 1296],
  "stats": {"detections": 82, "avg_conf": 0.93},
  "schema_version": "shot_track_v1"
}
```

**Changes**:
- Added `scene_id` reference
- Renamed `track_id` → `shot_track_id`
- Removed `sample_thumbs` (redundant with chips_manifest)

---

### track_v2 (Phase 3: Scene-Scoped Tracks)
```json
{
  "track_id": "scene_005_p0",
  "scene_id": "scene_005",
  "ep_id": "ep_12345",
  "person_id": "heather_dubrow",  // Optional (null if unidentified)
  "scene_start_s": 40.0,
  "scene_end_s": 61.67,
  "appearance_segments": [  // NEW
    [1200, 1300],
    [1400, 1500],
    [1700, 1850]
  ],
  "total_visible_s": 15.0,  // NEW
  "shot_track_ids": [42, 87, 103],  // NEW: merged SLTs
  "quality_stats": {  // NEW
    "avg_conf": 0.91,
    "total_detections": 450,
    "avg_occlusion_rate": 0.12
  },
  "schema_version": "track_v2"
}
```

**Changes**:
- Added `appearance_segments` (temporal gaps supported)
- Added `total_visible_s` (sum of visible time)
- Added `shot_track_ids` (traceability to SLTs)
- Added `quality_stats` (aggregated from SLTs)
- Changed `track_id` format: `scene_XXX_pN` (human-readable)

---

## Database Schema Changes

### NEW Table: `scene`
```sql
CREATE TABLE scene (
    scene_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ep_id UUID NOT NULL REFERENCES episode(ep_id) ON DELETE CASCADE,
    start_s FLOAT NOT NULL,
    end_s FLOAT NOT NULL,
    start_frame INT NOT NULL,
    end_frame INT NOT NULL,
    scene_type TEXT,  -- dialogue, action, confessional, montage, establishing
    location_hint TEXT,
    shot_cuts INT[],
    confidence FLOAT,
    meta JSONB,
    schema_version TEXT DEFAULT 'scene_v1',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_scene_ep_time ON scene(ep_id, start_s, end_s);
CREATE INDEX idx_scene_type ON scene(ep_id, scene_type);
```

---

### UPDATED Table: `track`
```sql
-- Add columns for scene-scoped tracks
ALTER TABLE track ADD COLUMN scene_id UUID REFERENCES scene(scene_id) ON DELETE CASCADE;
ALTER TABLE track ADD COLUMN appearance_segments JSONB;  -- [[start_frame, end_frame], ...]
ALTER TABLE track ADD COLUMN shot_track_ids UUID[];      -- IDs of merged shot tracks
ALTER TABLE track ADD COLUMN schema_version TEXT DEFAULT 'track_v2';

-- Update indices
CREATE INDEX idx_track_scene ON track(scene_id);
CREATE INDEX idx_track_ep_scene ON track(ep_id, scene_id);
```

**Backwards Compatibility**:
- All new columns are NULLABLE
- Existing rows (track_v1) have NULL `scene_id`, `appearance_segments`
- `schema_version` distinguishes track_v1 vs. track_v2

---

## API Changes

### Endpoint Changes

| Endpoint | v1 Behavior | v2 Behavior | Migration |
|----------|-------------|-------------|-----------|
| `GET /api/episodes/{ep_id}/tracks` | Returns SLTs | Returns SSTs | Add `?track_version=v1` for old behavior |
| `GET /api/tracks/{track_id}` | Returns SLT | Returns SST | Check `schema_version` field |
| `GET /api/episodes/{ep_id}/screen-time` | Calculated from SLTs | Calculated from SSTs | No change (internal) |

---

### NEW Endpoints

| Endpoint | Description | Response |
|----------|-------------|----------|
| `GET /api/episodes/{ep_id}/scenes` | List semantic scenes | `[{scene_id, start_s, end_s, scene_type, ...}]` |
| `GET /api/scenes/{scene_id}` | Get scene details | `{scene_id, tracks: [...], ...}` |
| `GET /api/scenes/{scene_id}/tracks` | List tracks in scene | `[{track_id, person_id, total_visible_s, ...}]` |
| `GET /api/v2/tracks` | Explicitly request v2 tracks | Same as `GET /api/episodes/{ep_id}/tracks` (v2) |

---

### Response Schema Changes

#### v1 Response (Shot-Level Track)
```json
GET /api/episodes/ep_12345/tracks?track_version=v1

[
  {
    "track_id": 42,
    "ep_id": "ep_12345",
    "start_s": 40.5,
    "end_s": 43.2,
    "person_id": "heather_dubrow",
    "schema_version": "track_v1"
  }
]
```

#### v2 Response (Scene-Scoped Track)
```json
GET /api/episodes/ep_12345/tracks

[
  {
    "track_id": "scene_005_p0",
    "scene_id": "scene_005",
    "ep_id": "ep_12345",
    "person_id": "heather_dubrow",
    "scene_start_s": 40.0,
    "scene_end_s": 61.67,
    "appearance_segments": [[1200, 1300], [1400, 1500], [1700, 1850]],
    "total_visible_s": 15.0,
    "schema_version": "track_v2"
  }
]
```

---

## Code Terminology Guidelines

### Variable Naming Conventions

```python
# GOOD: Clear distinction
sst = get_scene_scoped_track(track_id)
slt = get_shot_level_track(shot_track_id)
scene = get_scene(scene_id)

# BAD: Ambiguous
track = get_track(id)  # Which type?

# GOOD: Type-prefixed IDs
scene_track_id = "scene_005_p0"  # SST
shot_track_id = 42               # SLT
scene_id = "scene_005"

# BAD: Unclear ID type
track_id = 42  # Is this SST or SLT?
```

---

### Function Naming Conventions

```python
# Scene-Scoped Tracks
def merge_shot_tracks_into_scene_tracks(...)
def get_scene_tracks_for_episode(...)
def compute_scene_track_embeddings(...)

# Shot-Level Tracks (internal)
def detect_and_track_faces(...)  # Produces SLTs
def write_shot_tracks(...)       # Internal artifact

# Scenes
def detect_semantic_scenes(...)
def classify_scene_type(...)
def get_scene_boundaries(...)
```

---

## Documentation Updates Required

### Files to Update

1. **DATA_SCHEMA.md**
   - Add `scene` table definition
   - Update `track` table with new columns
   - Document track_v2 schema
   - Add "Schema Evolution" section

2. **API.md**
   - Document new `/api/scenes/*` endpoints
   - Update `/api/tracks` response schema
   - Add `?track_version` query param documentation
   - Add deprecation timeline for track_v1

3. **README.md**
   - Update "How It Works" section
   - Change "Track" definition
   - Add "Semantic Scenes" section

4. **DIRECTORY_STRUCTURE.md**
   - Update artifact paths (`shot_tracks.jsonl` vs. `tracks.jsonl`)
   - Document new config files

5. **CONFIG_GUIDE.md**
   - Add scene segmentation config
   - Add track merging config

---

## Migration Timeline

### Phase 1: Additive (Non-Breaking)
**Duration**: Weeks 1-3  
**Changes**:
- Add `scene` table (no impact on existing queries)
- Add `scene_id` column to `track` table (nullable)
- Add `/api/scenes` endpoints (new, no breaking changes)

**User Impact**: NONE (additive only)

---

### Phase 2: Dual Support (Deprecation Warning)
**Duration**: Weeks 4-6  
**Changes**:
- `/api/tracks` returns SSTs by default
- Add `?track_version=v1` for SLT behavior
- Emit deprecation warnings for track_v1 usage
- Update UI to use SSTs

**User Impact**: LOW (backwards compat maintained)

---

### Phase 3: Default Switchover
**Duration**: Weeks 7-9  
**Changes**:
- API defaults to track_v2 (SSTs)
- track_v1 still available via query param
- Update documentation to feature track_v2

**User Impact**: MEDIUM (API consumers must adapt or use `?track_version=v1`)

---

### Phase 4: Deprecation (3 Months After Phase 3)
**Changes**:
- Remove `?track_version=v1` support
- Remove shot_track_v1 schema support
- Clean up legacy code paths

**User Impact**: HIGH (breaking change for users not migrated)

---

## Migration Checklist for API Consumers

- [ ] Review API changes in this document
- [ ] Identify all API calls to `/api/tracks` endpoints
- [ ] Update code to handle `track_v2` schema:
  - [ ] Handle `appearance_segments` (multiple intervals)
  - [ ] Use `total_visible_s` instead of `end_s - start_s`
  - [ ] Handle new `track_id` format (`scene_XXX_pN`)
- [ ] Test with `?track_version=v1` query param (temporary)
- [ ] Migrate to track_v2 before Phase 4 deprecation
- [ ] Update internal documentation and team training

---

## FAQ

### Q1: Why rename "track" to "shot-level track"?
**A**: The term "track" becomes ambiguous. "Shot-level track" explicitly indicates the scope (single continuous sequence), distinguishing it from "scene-scoped track" (all appearances in a scene).

### Q2: Will existing track IDs break?
**A**: No. Shot-level track IDs remain numeric (e.g., `42`). Scene-scoped track IDs use a new format (e.g., `scene_005_p0`). The `schema_version` field distinguishes them.

### Q3: How do I calculate screen time with SSTs?
**A**: Use `total_visible_s` instead of `end_s - start_s`. SSTs explicitly track visible time excluding gaps.

```python
# v1 (SLT): Assumes continuous visibility
screen_time = track["end_s"] - track["start_s"]

# v2 (SST): Uses explicit visible time
screen_time = track["total_visible_s"]
```

### Q4: What happens to tracks without scene assignments?
**A**: Tracks with `scene_id = NULL` are treated as track_v1 (shot-level tracks). They remain valid and queryable via `?track_version=v1`.

### Q5: Can I manually override scene boundaries?
**A**: Future feature (Phase 6+). Initial release uses automatic scene detection only.

---

## Glossary

| Term | Definition |
|------|------------|
| **SST** | Scene-Scoped Track - all appearances of one person in one scene |
| **SLT** | Shot-Level Track - continuous detection sequence (old "track") |
| **Semantic Scene** | Location/activity unit perceived as coherent by humans |
| **Shot** | Single camera angle (terminated by cut/dissolve) |
| **Appearance Segment** | Continuous interval where person is visible within SST |
| **Confessional** | Single-person commentary shot (reality TV convention) |
| **Scene Boundary** | Location/activity change marking end of semantic scene |

---

## Related Documents

- **FEATURE.md**: Full feature specification
- **TODO.md**: Implementation checklist
- **MIGRATION_GUIDE.md**: Step-by-step migration guide (to be written)
- **DATA_SCHEMA.md**: Database schema reference
- **API.md**: API endpoint documentation

---

**Document Owner**: TBA  
**Last Review**: 2025-11-19  
**Next Review**: After Phase 1 completion
