# Implementation Outline: Scene-Scoped Tracks

**Version**: 1.0  
**Status**: PLANNING  
**Last Updated**: 2025-11-19

---

## Executive Summary

This document provides a detailed implementation outline for transforming the tracking system from detection-sequence-based tracks to scene-scoped person appearances. The implementation is divided into 5 phases spanning ~7 weeks.

---

## Phase 1: Semantic Scene Detection (Weeks 1-2)

### Goal
Produce `scenes.jsonl` artifact for each episode, containing semantic scene boundaries.

### Components

#### 1.1 Scene Detection Module
**File**: `FEATURES/scene-scoped-tracks/src/scene_detector.py`

```python
"""Semantic scene detection for reality TV content."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class Shot:
    """Single camera shot (bounded by cuts)."""
    start_frame: int
    end_frame: int
    duration_s: float

@dataclass
class Scene:
    """Semantic scene (group of shots)."""
    scene_id: str
    ep_id: str
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    scene_type: str  # dialogue, action, confessional, montage, establishing
    location_hint: str | None
    shot_cuts: List[int]
    confidence: float


class SemanticSceneDetector:
    """Detects semantic scene boundaries using shot detection + heuristics."""
    
    def __init__(
        self,
        temporal_gap_threshold_s: float = 3.0,
        rapid_cut_window_s: float = 2.0,
        confessional_min_duration_s: float = 5.0,
    ):
        self.temporal_gap_threshold_s = temporal_gap_threshold_s
        self.rapid_cut_window_s = rapid_cut_window_s
        self.confessional_min_duration_s = confessional_min_duration_s
    
    def detect_scenes(
        self, 
        shot_cuts: List[int], 
        fps: float,
        frame_count: int
    ) -> List[Scene]:
        """
        Detect semantic scenes from shot boundaries.
        
        Args:
            shot_cuts: Frame indices of shot boundaries (from PySceneDetect)
            fps: Video frame rate
            frame_count: Total frames in video
            
        Returns:
            List of Scene objects
        """
        shots = self._cuts_to_shots(shot_cuts, frame_count, fps)
        scenes = self._cluster_shots_into_scenes(shots, fps)
        return self._classify_scene_types(scenes)
    
    def _cuts_to_shots(
        self, 
        cuts: List[int], 
        frame_count: int,
        fps: float
    ) -> List[Shot]:
        """Convert cut frame indices to Shot objects."""
        if not cuts:
            return [Shot(0, frame_count, frame_count / fps)]
        
        shots = []
        cuts_with_bookends = [0] + cuts + [frame_count]
        
        for i in range(len(cuts_with_bookends) - 1):
            start = cuts_with_bookends[i]
            end = cuts_with_bookends[i + 1]
            duration = (end - start) / fps
            shots.append(Shot(start, end, duration))
        
        return shots
    
    def _cluster_shots_into_scenes(
        self, 
        shots: List[Shot],
        fps: float
    ) -> List[Scene]:
        """
        Cluster shots into semantic scenes using heuristics.
        
        Heuristics:
        1. Temporal gap > threshold → scene boundary
        2. Rapid cuts (< 2s) → same dialogue scene
        3. Long shot (> 5s) + single person → confessional
        """
        scenes = []
        current_scene_shots = []
        
        for i, shot in enumerate(shots):
            current_scene_shots.append(shot)
            
            # Check if next shot starts a new scene
            is_last_shot = (i == len(shots) - 1)
            if is_last_shot:
                scenes.append(self._create_scene(current_scene_shots))
                break
            
            next_shot = shots[i + 1]
            
            # Heuristic 1: Temporal gap (black frames, fades)
            gap_frames = next_shot.start_frame - shot.end_frame
            gap_s = gap_frames / fps
            if gap_s > self.temporal_gap_threshold_s:
                scenes.append(self._create_scene(current_scene_shots))
                current_scene_shots = []
                continue
            
            # Heuristic 2: Rapid cut pattern indicates dialogue
            if shot.duration_s < self.rapid_cut_window_s:
                continue  # Keep in same scene
            
            # Heuristic 3: Long shot may indicate scene change
            # (Conservative: only split on temporal gaps for now)
            
        return scenes
    
    def _create_scene(self, shots: List[Shot]) -> Scene:
        """Create Scene object from list of shots."""
        start_frame = shots[0].start_frame
        end_frame = shots[-1].end_frame
        shot_cuts = [s.start_frame for s in shots]
        
        return Scene(
            scene_id=f"scene_{start_frame:06d}",  # Temporary ID
            ep_id="",  # Filled in by caller
            start_frame=start_frame,
            end_frame=end_frame,
            start_s=0.0,  # Filled in by caller
            end_s=0.0,    # Filled in by caller
            scene_type="unknown",  # Classified later
            location_hint=None,
            shot_cuts=shot_cuts,
            confidence=1.0,  # TODO: Compute confidence score
        )
    
    def _classify_scene_types(self, scenes: List[Scene]) -> List[Scene]:
        """Classify scene types (dialogue, confessional, etc.)."""
        for scene in scenes:
            # Heuristic: Single long shot = likely confessional
            shot_count = len(scene.shot_cuts)
            duration = scene.end_s - scene.start_s
            
            if shot_count == 1 and duration > self.confessional_min_duration_s:
                scene.scene_type = "confessional"
            elif shot_count > 5 and duration < 10.0:
                scene.scene_type = "montage"
            else:
                scene.scene_type = "dialogue"  # Default
        
        return scenes


def write_scenes_jsonl(scenes: List[Scene], output_path: Path) -> None:
    """Write scenes to JSONL artifact."""
    import json
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for scene in scenes:
            record = {
                "scene_id": scene.scene_id,
                "ep_id": scene.ep_id,
                "start_frame": scene.start_frame,
                "end_frame": scene.end_frame,
                "start_s": scene.start_s,
                "end_s": scene.end_s,
                "scene_type": scene.scene_type,
                "location_hint": scene.location_hint,
                "shot_cuts": scene.shot_cuts,
                "confidence": scene.confidence,
                "schema_version": "scene_v1",
            }
            f.write(json.dumps(record) + "\n")
```

#### 1.2 Integration with episode_run.py
**File**: `tools/episode_run.py` (modifications)

**Changes**:
1. Import SemanticSceneDetector
2. After PySceneDetect shot detection, call semantic scene detection
3. Write `scenes.jsonl` artifact
4. Pass scene boundaries to tracking loop

**Pseudocode**:
```python
# Around line 3200 in episode_run.py

from FEATURES.scene_scoped_tracks.src.scene_detector import (
    SemanticSceneDetector,
    write_scenes_jsonl,
)

# After shot detection
if scene_detector_choice != "off":
    shot_cuts = detect_scene_cuts(...)  # Existing code
    
    # NEW: Semantic scene detection
    scene_detector = SemanticSceneDetector(
        temporal_gap_threshold_s=config.get("temporal_gap_threshold_s", 3.0),
        rapid_cut_window_s=config.get("rapid_cut_window_s", 2.0),
    )
    scenes = scene_detector.detect_scenes(shot_cuts, analyzed_fps, total_frames)
    
    # Fill in ep_id and timestamps
    for scene in scenes:
        scene.ep_id = args.ep_id
        scene.start_s = scene.start_frame / analyzed_fps
        scene.end_s = scene.end_frame / analyzed_fps
    
    # Write scenes artifact
    scenes_path = get_path(args.ep_id, "scenes")
    write_scenes_jsonl(scenes, scenes_path)
    
    LOGGER.info(
        "Detected %d semantic scenes (from %d shot cuts)",
        len(scenes),
        len(shot_cuts),
    )
```

#### 1.3 Configuration
**File**: `config/pipeline/scene_segmentation.yaml`

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

# Scene type detection
classify_scene_types: true
default_scene_type: dialogue

# Advanced (Phase 2+)
enable_visual_similarity: false
visual_similarity_threshold: 0.85
enable_audio_scene_detection: false
```

#### 1.4 Database Schema
**File**: `db/migrations/003_add_scene_table.sql`

```sql
-- Migration: Add scene table
-- Version: scene_v1
-- Author: TBA
-- Date: 2025-11-19

BEGIN;

-- Create scene table
CREATE TABLE IF NOT EXISTS scene (
    scene_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ep_id UUID NOT NULL REFERENCES episode(ep_id) ON DELETE CASCADE,
    start_s DOUBLE PRECISION NOT NULL,
    end_s DOUBLE PRECISION NOT NULL,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    scene_type TEXT,
    location_hint TEXT,
    shot_cuts INTEGER[],
    confidence REAL,
    meta JSONB,
    schema_version TEXT DEFAULT 'scene_v1',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT scene_time_order CHECK (start_s <= end_s),
    CONSTRAINT scene_frame_order CHECK (start_frame <= end_frame)
);

-- Create indices
CREATE INDEX idx_scene_ep_time ON scene(ep_id, start_s, end_s);
CREATE INDEX idx_scene_type ON scene(ep_id, scene_type) WHERE scene_type IS NOT NULL;
CREATE INDEX idx_scene_ep_id ON scene(ep_id);

-- Add trigger for updated_at
CREATE TRIGGER update_scene_updated_at
    BEFORE UPDATE ON scene
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

COMMIT;
```

#### 1.5 API Endpoint
**File**: `apps/api/routers/episodes.py` (modifications)

```python
@router.get("/{ep_id}/scenes", response_model=List[SceneResponse])
async def list_episode_scenes(
    ep_id: str,
    db: AsyncSession = Depends(get_db),
) -> List[SceneResponse]:
    """List semantic scenes for an episode."""
    stmt = (
        select(Scene)
        .where(Scene.ep_id == ep_id)
        .order_by(Scene.start_s)
    )
    result = await db.execute(stmt)
    scenes = result.scalars().all()
    
    return [
        SceneResponse(
            scene_id=str(scene.scene_id),
            ep_id=str(scene.ep_id),
            start_s=scene.start_s,
            end_s=scene.end_s,
            start_frame=scene.start_frame,
            end_frame=scene.end_frame,
            scene_type=scene.scene_type,
            location_hint=scene.location_hint,
            shot_count=len(scene.shot_cuts) if scene.shot_cuts else 0,
            confidence=scene.confidence,
        )
        for scene in scenes
    ]
```

#### 1.6 Tests
**File**: `tests/ml/test_semantic_scene_detection.py`

```python
import pytest
from FEATURES.scene_scoped_tracks.src.scene_detector import (
    SemanticSceneDetector,
    Shot,
)


def test_cluster_shots_rapid_cuts():
    """Rapid cuts (< 2s) should be grouped into same scene."""
    detector = SemanticSceneDetector(rapid_cut_window_s=2.0)
    
    # 5 shots, each 1s (rapid dialogue)
    shots = [
        Shot(0, 30, 1.0),
        Shot(30, 60, 1.0),
        Shot(60, 90, 1.0),
        Shot(90, 120, 1.0),
        Shot(120, 150, 1.0),
    ]
    
    scenes = detector._cluster_shots_into_scenes(shots, fps=30.0)
    
    assert len(scenes) == 1  # All grouped into one scene
    assert scenes[0].start_frame == 0
    assert scenes[0].end_frame == 150


def test_cluster_shots_temporal_gap():
    """Temporal gap > threshold should split scenes."""
    detector = SemanticSceneDetector(temporal_gap_threshold_s=3.0)
    
    # Shot 1: frames 0-100
    # Gap: frames 100-200 (100 frames = 3.33s gap at 30fps)
    # Shot 2: frames 200-300
    shots = [
        Shot(0, 100, 3.33),
        Shot(200, 300, 3.33),  # 100-frame gap before this
    ]
    
    scenes = detector._cluster_shots_into_scenes(shots, fps=30.0)
    
    assert len(scenes) == 2  # Gap causes split


def test_classify_confessional():
    """Single long shot should be classified as confessional."""
    detector = SemanticSceneDetector(confessional_min_duration_s=5.0)
    
    scene = Scene(
        scene_id="scene_001",
        ep_id="ep_123",
        start_frame=0,
        end_frame=300,
        start_s=0.0,
        end_s=10.0,  # 10 seconds
        scene_type="unknown",
        location_hint=None,
        shot_cuts=[0],  # Single shot
        confidence=1.0,
    )
    
    scenes = detector._classify_scene_types([scene])
    
    assert scenes[0].scene_type == "confessional"
```

---

## Phase 2: Shot-Level Track Annotation (Week 3)

### Goal
Add `scene_id` to existing track records (now called "shot tracks").

### Components

#### 2.1 TrackRecorder Modifications
**File**: `tools/episode_run.py` (TrackRecorder class)

**Changes**:
- Accept scene metadata in constructor
- Add `scene_id` field to track records
- Update `on_cut()` to handle scene boundaries

**Pseudocode**:
```python
class TrackRecorder:
    def __init__(self, max_gap: int, remap_ids: bool, scenes: List[Scene] = None):
        self.scenes = scenes or []
        self.scene_index = 0
        # ... existing fields
    
    def get_current_scene_id(self, frame_idx: int) -> str | None:
        """Get scene_id for current frame."""
        if not self.scenes:
            return None
        
        # Find scene containing this frame
        for scene in self.scenes:
            if scene.start_frame <= frame_idx <= scene.end_frame:
                return scene.scene_id
        
        return None
    
    def update(self, frame_idx: int, detections: List[Detection]) -> None:
        # ... existing tracking logic
        
        # Add scene_id to track record
        scene_id = self.get_current_scene_id(frame_idx)
        for track in self.active_tracks:
            track.scene_id = scene_id
```

#### 2.2 Shot Track Schema
**File**: `FEATURES/scene-scoped-tracks/src/schemas.py`

```python
from pydantic import BaseModel
from typing import List, Optional

class ShotTrackV1(BaseModel):
    """Shot-level track schema (track_v1 + scene_id)."""
    shot_track_id: int
    scene_id: Optional[str]
    ep_id: str
    start_s: float
    end_s: float
    frame_span: List[int]  # [start_frame, end_frame]
    stats: dict  # {"detections": int, "avg_conf": float}
    schema_version: str = "shot_track_v1"
```

#### 2.3 Artifact Resolver Updates
**File**: `packages/py-screenalytics/artifacts.py`

```python
# Add new artifact type
ARTIFACT_PATHS = {
    "detections": "{ep_id}/detections.jsonl",
    "tracks": "{ep_id}/tracks.jsonl",           # SSTs (v2)
    "shot_tracks": "{ep_id}/shot_tracks.jsonl", # SLTs (internal)
    "scenes": "{ep_id}/scenes.jsonl",           # NEW
    # ... other artifacts
}
```

#### 2.4 Tests
**File**: `tests/ml/test_track_scene_annotation.py`

```python
def test_track_has_scene_id(tmp_path):
    """Tracks should be annotated with scene_id."""
    # Setup: Create video with 2 scenes
    # Run: episode_run.py with scene detection enabled
    # Assert: All tracks have valid scene_id
    pass
```

---

## Phase 3: Scene-Scoped Track Merging (Weeks 4-5)

### Goal
Merge shot-level tracks into scene-scoped tracks using appearance clustering.

### Components

#### 3.1 Appearance Clustering Module
**File**: `FEATURES/scene-scoped-tracks/src/appearance_clustering.py`

```python
"""Appearance-based clustering for track merging."""

import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from typing import List, Tuple

def cluster_by_appearance(
    shot_tracks: List[dict],
    embeddings: dict,  # {shot_track_id: np.ndarray}
    similarity_threshold: float = 0.80,
    linkage: str = "average",
) -> List[List[dict]]:
    """
    Cluster shot tracks by appearance similarity.
    
    Args:
        shot_tracks: List of shot track records
        embeddings: Mapping from shot_track_id to embedding vector
        similarity_threshold: Minimum cosine similarity for same cluster
        linkage: Hierarchical clustering linkage method
        
    Returns:
        List of clusters (each cluster is a list of shot tracks)
    """
    if not shot_tracks:
        return []
    
    # 1. Get representative embeddings
    track_embeddings = []
    valid_tracks = []
    
    for st in shot_tracks:
        track_id = st["shot_track_id"]
        if track_id in embeddings:
            emb = embeddings[track_id]
            track_embeddings.append(emb)
            valid_tracks.append(st)
    
    if not track_embeddings:
        return [[st] for st in shot_tracks]  # Singleton clusters
    
    # 2. Compute similarity matrix
    embeddings_matrix = np.array(track_embeddings)
    similarity_matrix = compute_cosine_similarity(embeddings_matrix)
    
    # 3. Hierarchical clustering
    distance_matrix = 1.0 - similarity_matrix
    cluster_labels = fclusterdata(
        distance_matrix,
        t=1.0 - similarity_threshold,
        criterion="distance",
        method=linkage,
    )
    
    # 4. Group tracks by cluster label
    clusters = {}
    for st, label in zip(valid_tracks, cluster_labels):
        clusters.setdefault(label, []).append(st)
    
    return list(clusters.values())


def compute_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    
    # Compute similarity
    similarity = np.dot(normalized, normalized.T)
    
    return similarity


def get_representative_embedding(
    track_id: int,
    embeddings_db: dict,
    strategy: str = "median",
) -> np.ndarray:
    """
    Get representative embedding for a track.
    
    Args:
        track_id: Shot track ID
        embeddings_db: Embedding database
        strategy: "median", "mean", or "max_quality"
        
    Returns:
        Representative embedding vector
    """
    track_embeddings = embeddings_db.get(track_id, [])
    
    if not track_embeddings:
        return None
    
    if strategy == "median":
        return np.median(track_embeddings, axis=0)
    elif strategy == "mean":
        return np.mean(track_embeddings, axis=0)
    elif strategy == "max_quality":
        # Select embedding with highest quality score
        qualities = [e["quality"] for e in track_embeddings]
        best_idx = np.argmax(qualities)
        return track_embeddings[best_idx]["vector"]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
```

#### 3.2 Track Merging Module
**File**: `FEATURES/scene-scoped-tracks/src/track_merger.py`

```python
"""Merge shot-level tracks into scene-scoped tracks."""

from typing import List, Dict
import uuid

def merge_cluster_into_track(
    cluster: List[dict],
    scene: dict,
    person_id: str | None = None,
) -> dict:
    """
    Merge a cluster of shot tracks into a single scene-scoped track.
    
    Args:
        cluster: List of shot track records (same identity)
        scene: Scene metadata
        person_id: Optional person ID (if identified)
        
    Returns:
        Scene-scoped track record
    """
    # Sort by temporal order
    cluster = sorted(cluster, key=lambda st: st["start_s"])
    
    # Compute appearance segments
    appearance_segments = [
        [st["frame_span"][0], st["frame_span"][1]]
        for st in cluster
    ]
    
    # Compute total visible time
    total_visible_s = sum(st["end_s"] - st["start_s"] for st in cluster)
    
    # Aggregate quality stats
    total_detections = sum(st["stats"]["detections"] for st in cluster)
    avg_conf = sum(
        st["stats"]["avg_conf"] * st["stats"]["detections"]
        for st in cluster
    ) / total_detections if total_detections > 0 else 0.0
    
    # Generate track ID
    track_id = str(uuid.uuid4())
    
    return {
        "track_id": track_id,
        "scene_id": scene["scene_id"],
        "ep_id": scene["ep_id"],
        "person_id": person_id,
        "scene_start_s": scene["start_s"],
        "scene_end_s": scene["end_s"],
        "appearance_segments": appearance_segments,
        "total_visible_s": total_visible_s,
        "shot_track_ids": [st["shot_track_id"] for st in cluster],
        "quality_stats": {
            "avg_conf": avg_conf,
            "total_detections": total_detections,
        },
        "schema_version": "track_v2",
    }
```

#### 3.3 Merge Script
**File**: `tools/merge_scene_tracks.py`

```python
#!/usr/bin/env python3
"""Merge shot-level tracks into scene-scoped tracks."""

import argparse
import json
from pathlib import Path
from py_screenalytics.artifacts import get_path

from FEATURES.scene_scoped_tracks.src.appearance_clustering import cluster_by_appearance
from FEATURES.scene_scoped_tracks.src.track_merger import merge_cluster_into_track


def load_jsonl(path: Path) -> list:
    """Load JSONL file."""
    records = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-id", required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.80)
    parser.add_argument("--linkage", default="average")
    args = parser.parse_args()
    
    # Load artifacts
    shot_tracks_path = get_path(args.ep_id, "shot_tracks")
    scenes_path = get_path(args.ep_id, "scenes")
    
    shot_tracks = load_jsonl(shot_tracks_path)
    scenes = load_jsonl(scenes_path)
    
    # TODO: Load embeddings from database
    embeddings = {}  # Placeholder
    
    # Merge tracks
    scene_tracks = []
    
    for scene in scenes:
        # Get shot tracks for this scene
        scene_shot_tracks = [
            st for st in shot_tracks
            if st.get("scene_id") == scene["scene_id"]
        ]
        
        if not scene_shot_tracks:
            continue
        
        # Cluster by appearance
        clusters = cluster_by_appearance(
            scene_shot_tracks,
            embeddings,
            similarity_threshold=args.similarity_threshold,
            linkage=args.linkage,
        )
        
        # Merge each cluster into a scene-scoped track
        for cluster in clusters:
            sst = merge_cluster_into_track(cluster, scene)
            scene_tracks.append(sst)
    
    # Write output
    output_path = get_path(args.ep_id, "tracks")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w") as f:
        for sst in scene_tracks:
            f.write(json.dumps(sst) + "\n")
    
    print(f"Wrote {len(scene_tracks)} scene-scoped tracks")


if __name__ == "__main__":
    main()
```

#### 3.4 Database Migration
**File**: `db/migrations/004_scene_scoped_track_columns.sql`

```sql
-- Migration: Add scene-scoped track columns
-- Version: track_v2
-- Author: TBA
-- Date: 2025-11-19

BEGIN;

-- Add columns for scene-scoped tracks
ALTER TABLE track 
    ADD COLUMN IF NOT EXISTS scene_id UUID REFERENCES scene(scene_id) ON DELETE CASCADE,
    ADD COLUMN IF NOT EXISTS appearance_segments JSONB,
    ADD COLUMN IF NOT EXISTS shot_track_ids INTEGER[],
    ADD COLUMN IF NOT EXISTS schema_version TEXT DEFAULT 'track_v1';

-- Create indices
CREATE INDEX IF NOT EXISTS idx_track_scene ON track(scene_id);
CREATE INDEX IF NOT EXISTS idx_track_ep_scene ON track(ep_id, scene_id);

-- Add constraint for track_v2 records
ALTER TABLE track ADD CONSTRAINT track_v2_requires_scene_id 
    CHECK (
        schema_version != 'track_v2' OR scene_id IS NOT NULL
    );

COMMIT;
```

---

## Phase 4: API & UI Updates (Week 6)

### Goal
Expose scene-scoped tracks in API and update UI to use them.

### Components

#### 4.1 API Updates
**File**: `apps/api/routers/episodes.py`

```python
@router.get("/{ep_id}/tracks", response_model=List[TrackResponse])
async def list_episode_tracks(
    ep_id: str,
    track_version: str = "v2",  # Default to scene-scoped tracks
    db: AsyncSession = Depends(get_db),
) -> List[TrackResponse]:
    """List tracks for an episode."""
    if track_version == "v1":
        # Return shot-level tracks (backwards compat)
        stmt = (
            select(Track)
            .where(Track.ep_id == ep_id)
            .where(Track.schema_version == "track_v1")
            .order_by(Track.start_s)
        )
    else:  # v2 (default)
        # Return scene-scoped tracks
        stmt = (
            select(Track)
            .where(Track.ep_id == ep_id)
            .where(Track.schema_version == "track_v2")
            .order_by(Track.scene_start_s, Track.total_visible_s.desc())
        )
    
    result = await db.execute(stmt)
    tracks = result.scalars().all()
    
    return [track_to_response(t) for t in tracks]
```

---

## Phase 5: Testing & Validation (Week 7)

### Goal
Comprehensive testing and quality assurance.

### Test Coverage

#### Integration Tests
- End-to-end scene detection → merging
- API endpoint behavior
- UI scene visualization

#### Regression Tests
- Existing episodes still process
- Screen time consistency

#### Performance Tests
- Processing time benchmarks
- Memory usage profiling

---

## Configuration Files Summary

### `config/pipeline/scene_segmentation.yaml`
Scene detection parameters (temporal gaps, confessional detection)

### `config/pipeline/track_merging.yaml`
Appearance clustering parameters (similarity threshold, linkage method)

---

## Artifact Flow Summary

```
video.mp4
  ↓ (PySceneDetect)
shot_cuts: [120, 245, 298, ...]
  ↓ (SemanticSceneDetector)
scenes.jsonl (20 scenes)
  ↓ (ByteTrack + scene annotation)
shot_tracks.jsonl (4,352 shot tracks)
  ↓ (ArcFace embeddings)
embeddings (pgvector)
  ↓ (appearance clustering + merging)
tracks.jsonl (100 scene-scoped tracks)
```

---

## Success Metrics Checklist

- [ ] Scene count: 15-30 per episode
- [ ] Track reduction: 95%+ (4,352 → ~100)
- [ ] False merge rate: < 2%
- [ ] Identity drift rate: < 5%
- [ ] Processing time increase: < 20%
- [ ] Test coverage: > 85%

---

## Related Documents

- **FEATURE.md**: Full feature specification
- **TERMINOLOGY.md**: Terminology changes and migration
- **TODO.md**: Implementation checklist
- **PROGRESS.md**: Current progress tracking

---

**Last Updated**: 2025-11-19
