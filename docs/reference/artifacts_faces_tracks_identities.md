# Artifacts: Faces, Tracks, Identities — Schema Reference

**Version:** 2.0
**Last Updated:** 2025-11-18

---

## 1. Overview

This document defines the canonical schemas for all pipeline artifacts produced by the detect/track/embed/cluster stages. All schemas are versioned for backward compatibility and validation.

### Storage layout (faces + tracks)
- `artifacts/frames/<show>/s##/e##/frames/…` — optional full-frame exports (heavy; disable via Faces Harvest toggle on laptops).
- `artifacts/crops/<show>/s##/e##/tracks/…` — **canonical track-level images**. UI + ML both read these; spacing is enforced per-track when sampling.
- `artifacts/thumbs/<show>/s##/e##/identities/…` — one square thumbnail per identity (derived from best crops).
- `artifacts/thumbs/<show>/s##/e##/tracks/…` — **legacy** track thumbnails; new runs skip creation/uploads and UIs fall back to crops when thumbs are absent.

---

## 2. `detections.jsonl` (det_v1)

**Format:** JSON Lines (one object per line)
**Location:** `data/manifests/{ep_id}/detections.jsonl`

### Schema
```jsonl
{
  "ep_id": "rhobh-s05e02",
  "frame_idx": 42,
  "ts_s": 1.75,
  "bbox": [0.1, 0.2, 0.3, 0.4],
  "landmarks": [0.15, 0.25, 0.25, 0.25, 0.2, 0.3, 0.15, 0.35, 0.25, 0.35],
  "conf": 0.95,
  "model_id": "retinaface_r50_v1",
  "schema_version": "det_v1"
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ep_id` | string | ✅ | Episode identifier |
| `frame_idx` | int | ✅ | Zero-based frame number |
| `ts_s` | float | ✅ | Timestamp in seconds |
| `bbox` | array[float] | ✅ | `[x1, y1, x2, y2]` normalized (0–1) coordinates |
| `landmarks` | array[float] | ✅ | Flattened `[x, y] * 5` facial landmarks (left eye, right eye, nose, left mouth, right mouth) |
| `conf` | float | ✅ | Detector confidence (0–1) |
| `model_id` | string | ✅ | Model identifier (e.g., `"retinaface_r50_v1"`) |
| `schema_version` | string | ✅ | `"det_v1"` |

### Invariants
- `bbox` coordinates normalized to [0, 1]
- `landmarks` length = 10 (5 points × 2 coords)
- `conf` in range [0, 1]

---

## 3. `tracks.jsonl` (track_v1)

**Format:** JSON Lines (one object per line)
**Location:** `data/manifests/{ep_id}/tracks.jsonl`

### Schema
```jsonl
{
  "track_id": "track-00001",
  "ep_id": "rhobh-s05e02",
  "start_s": 1.5,
  "end_s": 12.3,
  "frame_span": [30, 246],
  "sample_thumbs": ["crops/track-00001/frame_0030.jpg", "crops/track-00001/frame_0138.jpg"],
  "stats": {
    "detections": 216,
    "avg_conf": 0.92
  },
  "schema_version": "track_v1"
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `track_id` | string | ✅ | Deterministic track identifier (e.g., `"track-00001"`) |
| `ep_id` | string | ✅ | Episode identifier |
| `start_s` | float | ✅ | Track start timestamp (seconds) |
| `end_s` | float | ✅ | Track end timestamp (seconds) |
| `frame_span` | array[int] | ✅ | `[start_frame, end_frame]` (inclusive) |
| `sample_thumbs` | array[string] | ⚠️ | List of thumbnail paths (relative to artifacts root); empty if crops not saved |
| `stats.detections` | int | ✅ | Number of detections in this track |
| `stats.avg_conf` | float | ✅ | Average detection confidence |
| `schema_version` | string | ✅ | `"track_v1"` |

### Invariants
- `track_id` format: `track-{N:05d}` (zero-padded 5 digits)
- `start_s <= end_s`
- `frame_span[0] <= frame_span[1]`
- `stats.detections >= 1`

---

## 4. `track_metrics.json`

**Format:** Single JSON object
**Location:** `data/manifests/{ep_id}/track_metrics.json`

### Schema
```json
{
  "ep_id": "rhobh-s05e02",
  "tracks_born": 42,
  "tracks_lost": 40,
  "id_switches": 2,
  "longest_tracks": [
    {"track_id": "track-00005", "length": 512},
    {"track_id": "track-00012", "length": 384}
  ],
  "avg_tracks_per_frame": 2.3,
  "tracks_per_minute": 18.5,
  "short_track_fraction": 0.12,
  "total_frames": 2400,
  "analyzed_fps": 8.0,
  "elapsed_sec": 120.5,
  "device": "cpu",
  "detector": "retinaface",
  "tracker": "bytetrack",
  "scene_cuts": 5
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `ep_id` | string | Episode identifier |
| `tracks_born` | int | Total tracks created |
| `tracks_lost` | int | Tracks terminated |
| `id_switches` | int | Track ID reassignments (lower is better) |
| `longest_tracks` | array[object] | Top 5 longest tracks (by frame count) |
| `avg_tracks_per_frame` | float | Average concurrent tracks per frame |
| `tracks_per_minute` | float | Derived metric (tracks_born / episode_duration_min) |
| `short_track_fraction` | float | Fraction of tracks < threshold (e.g., < 30 frames or < 1 sec) |
| `total_frames` | int | Total frames analyzed |
| `analyzed_fps` | float | Processing FPS (effective throughput) |
| `elapsed_sec` | float | Wall-clock runtime |
| `device` | string | Device used (`"cpu"`, `"cuda"`, `"mps"`) |
| `detector` | string | Detector model (`"retinaface"`) |
| `tracker` | string | Tracker algorithm (`"bytetrack"`) |
| `scene_cuts` | int | Number of scene changes detected |

---

## 5. `faces.jsonl`

**Format:** JSON Lines (one object per line)
**Location:** `data/manifests/{ep_id}/faces.jsonl`

### Schema
```jsonl
{
  "face_id": "face-00001",
  "track_id": "track-00001",
  "frame_idx": 42,
  "bbox": [0.1, 0.2, 0.3, 0.4],
  "landmarks": [0.15, 0.25, 0.25, 0.25, 0.2, 0.3, 0.15, 0.35, 0.25, 0.35],
  "embedding": [0.012, -0.045, ...],
  "quality_score": 0.87,
  "crop_path": "crops/track-00001/frame_000042.jpg"
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `face_id` | string | ✅ | Unique face identifier (e.g., `"face-00001"`) |
| `track_id` | string | ✅ | Links back to `tracks.jsonl` |
| `frame_idx` | int | ✅ | Zero-based frame number |
| `bbox` | array[float] | ✅ | `[x1, y1, x2, y2]` normalized (0–1) |
| `landmarks` | array[float] | ✅ | Flattened `[x, y] * 5` facial landmarks |
| `embedding` | array[float] | ✅ | 512-d float32 unit-norm vector |
| `quality_score` | float | ✅ | Combined quality metric (0–1) |
| `crop_path` | string | ⚠️ | Relative path to face crop (empty if crops not saved) |
| `rejection_reason` | string | ❌ | Optional: Reason if face rejected (`"low_quality"`, `"pose_too_extreme"`, etc.) |

### Invariants
- `embedding` length = 512
- `embedding` unit-norm: `np.linalg.norm(embedding) ≈ 1.0`
- `quality_score` in range [0, 1]

---

## 6. `faces.npy`

**Format:** NumPy binary array
**Location:** `data/embeds/{ep_id}/faces.npy`

### Schema
- **Shape:** `(N, 512)` where N = number of faces in `faces.jsonl`
- **Dtype:** `float32`
- **Invariant:** Each row is a unit-norm embedding

### Usage
```python
import numpy as np
embeddings = np.load("data/embeds/{ep_id}/faces.npy")
print(embeddings.shape)  # (N, 512)
print(np.linalg.norm(embeddings[0]))  # ≈ 1.0
```

---

## 7. `identities.json`

**Format:** Single JSON object
**Location:** `data/manifests/{ep_id}/identities.json`

### Schema
```json
{
  "identities": [
    {
      "identity_id": "identity-00001",
      "cluster_id": 1,
      "track_ids": ["track-00001", "track-00005", "track-00012"],
      "canonical_face_path": "thumbs/identity-00001/rep.jpg",
      "embedding_stats": {
        "centroid_norm": 1.0,
        "variance": 0.023,
        "num_tracks": 3,
        "num_faces_total": 142
      },
      "labels": {
        "person_id": "lisa-vanderpump",
        "name": "Lisa Vanderpump"
      },
      "locked": true,
      "created_at": "2025-11-18T12:34:56Z",
      "updated_at": "2025-11-18T12:34:56Z"
    }
  ],
  "metadata": {
    "ep_id": "rhobh-s05e02",
    "num_tracks": 42,
    "num_clusters": 8,
    "num_singletons": 6,
    "singleton_fraction": 0.14,
    "largest_cluster_size": 12,
    "largest_cluster_fraction": 0.29,
    "cluster_thresh": 0.58,
    "min_cluster_size": 2,
    "algorithm": "agglomerative",
    "linkage": "average",
    "created_at": "2025-11-18T12:34:56Z"
  }
}
```

### Field Definitions (per identity)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `identity_id` | string | ✅ | Unique identity identifier (e.g., `"identity-00001"`) |
| `cluster_id` | int or null | ✅ | Cluster number (null for singletons/outliers) |
| `track_ids` | array[string] | ✅ | List of track IDs assigned to this identity |
| `canonical_face_path` | string | ✅ | Path to representative thumbnail |
| `embedding_stats.centroid_norm` | float | ✅ | Norm of cluster centroid (≈ 1.0) |
| `embedding_stats.variance` | float | ✅ | Intra-cluster variance (lower = tighter) |
| `embedding_stats.num_tracks` | int | ✅ | Number of tracks in this cluster |
| `embedding_stats.num_faces_total` | int | ✅ | Total faces across all tracks |
| `labels` | object | ⚠️ | Human labels (person_id, name, etc.); empty for unlabeled |
| `locked` | bool | ✅ | `true` = confirmed by human, protected from auto-merge/split |
| `created_at` | string (ISO 8601) | ✅ | Creation timestamp |
| `updated_at` | string (ISO 8601) | ✅ | Last modification timestamp |

### Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `ep_id` | string | Episode identifier |
| `num_tracks` | int | Total tracks clustered |
| `num_clusters` | int | Number of clusters formed |
| `num_singletons` | int | Tracks in singleton clusters (cluster_id=null) |
| `singleton_fraction` | float | `num_singletons / num_tracks` |
| `largest_cluster_size` | int | Tracks in largest cluster |
| `largest_cluster_fraction` | float | `largest_cluster_size / num_tracks` |
| `cluster_thresh` | float | Cosine similarity threshold used |
| `min_cluster_size` | int | Minimum tracks per cluster |
| `algorithm` | string | Clustering algorithm (`"agglomerative"`) |
| `linkage` | string | Linkage method (`"ward"`, `"average"`, `"complete"`) |
| `created_at` | string (ISO 8601) | Clustering run timestamp |

---

## 8. `cleanup_report.json`

**Format:** Single JSON object
**Location:** `data/manifests/{ep_id}/cleanup_report.json`

### Schema
```json
{
  "ep_id": "rhobh-s05e02",
  "actions_run": ["split_tracks", "reembed", "recluster"],
  "before": {
    "num_tracks": 42,
    "num_faces": 2048,
    "num_clusters": 18,
    "singleton_fraction": 0.38
  },
  "after": {
    "num_tracks": 58,
    "num_faces": 1876,
    "num_clusters": 12,
    "singleton_fraction": 0.15
  },
  "split_tracks_summary": {
    "tracks_split": 8,
    "new_tracks_created": 16
  },
  "reembed_summary": {
    "faces_rejected": 172,
    "mean_quality_before": 0.72,
    "mean_quality_after": 0.84
  },
  "recluster_summary": {
    "clusters_merged": 6
  },
  "elapsed_sec": 245.3,
  "created_at": "2025-11-18T12:34:56Z"
}
```

---

## 9. Relationships

```
detections.jsonl
    └─► (grouped by spatial + temporal proximity) ────► tracks.jsonl
                                                           ↓
                                                        track_id
                                                           ↓
faces.jsonl ←─────────────────────────────────────────────┘
    ↓
embedding (512-d)
    ↓
(pooled per track) ────► track_embeddings.npy
    ↓
(clustered) ────► identities.json
```

---

## 10. Validation

All artifacts should be validated on write and read:

### 10.1 Schema Validation
- Check `schema_version` field matches expected version
- Validate required fields present
- Validate field types (int, float, string, array)

### 10.2 Invariant Validation
- `bbox` coordinates in [0, 1]
- `embedding` unit-norm (tolerance ± 0.01)
- `track_id` format matches `track-{N:05d}`
- `start_s <= end_s`, `frame_span[0] <= frame_span[1]`

### 10.3 Referential Integrity
- Every `track_id` in `faces.jsonl` exists in `tracks.jsonl`
- Every `track_id` in `identities.json` exists in `tracks.jsonl`
- Row count in `faces.npy` matches line count in `faces.jsonl`

---

## 11. References

- [Pipeline Overview](../pipeline/overview.md)
- [Detect & Track](../pipeline/detect_track_faces.md)
- [Faces Harvest](../pipeline/faces_harvest.md)
- [Cluster Identities](../pipeline/cluster_identities.md)
- [Episode Cleanup](../pipeline/episode_cleanup.md)

---

**Maintained by:** Screenalytics Engineering
