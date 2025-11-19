# Episode Cleanup — Screenalytics Pipeline

**Version:** 2.0
**Last Updated:** 2025-11-18

---

## 1. Overview

The **Episode Cleanup** workflow refines initial detect/track/embed/cluster results through iterative post-processing. It addresses common issues like:

- **Long tracks contaminated by multiple people** (split via appearance gate)
- **Low-quality embeddings** (re-embed with stricter quality gating)
- **Over-/under-segmented clusters** (re-cluster with tuned thresholds)
- **Fragmented identities** (group related clusters)

**Entry point:** `tools/episode_cleanup.py` or `POST /jobs/episode_cleanup_async`

---

## 2. Cleanup Phases

```
1. SPLIT_TRACKS
   - Apply AppearanceGate to long tracks
   - Split tracks where appearance changes significantly
   - Update tracks.jsonl

2. REEMBED
   - Re-run faces_embed with stricter quality gating
   - Filter low-quality crops
   - Update faces.jsonl, faces.npy

3. RECLUSTER
   - Re-cluster with adjusted thresholds
   - Update identities.json

4. GROUP_CLUSTERS
   - Merge related clusters based on cross-episode Facebank matching
   - Update identities.json with person_id labels
```

---

## 3. CLI Usage

```bash
python tools/episode_cleanup.py \
  --ep-id rhobh-s05e02 \
  --video data/videos/rhobh-s05e02/episode.mp4 \
  --actions split_tracks reembed recluster group_clusters \
  --cluster-thresh 0.58 \
  --min-cluster-size 2 \
  --write-back
```

**API:**
```bash
POST /jobs/episode_cleanup_async
{
  "ep_id": "rhobh-s05e02",
  "actions": ["split_tracks", "reembed", "recluster", "group_clusters"],
  "write_back": true
}
```

---

## 4. Configuration

Cleanup reuses configs from detect, track, embed, cluster stages:
- `config/pipeline/detection.yaml`
- `config/pipeline/tracking.yaml`
- `config/pipeline/faces_embed_sampling.yaml`
- Clustering params (inline or future `recognition.yaml`)

**No new config file**—intentionally shares configs to ensure consistency.

---

## 5. Artifacts

### 5.1 `cleanup_report.json`
```json
{
  "ep_id": "rhobh-s05e02",
  "actions_run": ["split_tracks", "reembed", "recluster"],
  "before": {
    "num_tracks": 42,
    "num_faces": 2048,
    "num_clusters": 18,
    "singleton_fraction": 0.38,
    "short_track_fraction": 0.25
  },
  "after": {
    "num_tracks": 58,
    "num_faces": 1876,
    "num_clusters": 12,
    "singleton_fraction": 0.15,
    "short_track_fraction": 0.08
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
    "clusters_merged": 6,
    "largest_cluster_fraction_before": 0.45,
    "largest_cluster_fraction_after": 0.32
  },
  "elapsed_sec": 245.3,
  "created_at": "2025-11-18T12:34:56Z"
}
```

### 5.2 Updated Artifacts
- `tracks.jsonl` (with split tracks)
- `faces.jsonl`, `faces.npy` (with re-embedded faces)
- `identities.json` (with re-clustered identities)
- `track_metrics.json` (updated metrics)

---

## 6. Phase Details

### 6.1 Split Tracks
**Purpose:** Break contaminated tracks (multiple people in one track)

**Algorithm:**
1. Load existing tracks
2. For each track, extract frame embeddings
3. Compute pairwise cosine similarity
4. Split track where similarity drops below `TRACK_GATE_APPEAR_HARD` threshold
5. Write updated `tracks.jsonl`

**Metrics:**
- `tracks_split`: Number of original tracks split
- `new_tracks_created`: Number of new tracks created

### 6.2 Re-Embed
**Purpose:** Re-extract embeddings with stricter quality gating

**Algorithm:**
1. Load existing `faces.jsonl`
2. Re-run faces_embed with higher `min_quality` threshold
3. Reject low-quality crops (blur, occlusion, extreme pose)
4. Update `faces.jsonl`, `faces.npy`

**Metrics:**
- `faces_rejected`: Number of faces dropped
- `mean_quality_before`, `mean_quality_after`

### 6.3 Re-Cluster
**Purpose:** Re-cluster tracks with adjusted thresholds

**Algorithm:**
1. Load track embeddings (pooled from updated `faces.npy`)
2. Re-run agglomerative clustering with new `cluster_thresh`
3. Update `identities.json`

**Metrics:**
- `clusters_merged`: Number of clusters combined
- `singleton_fraction_before`, `singleton_fraction_after`
- `largest_cluster_fraction_before`, `largest_cluster_fraction_after`

### 6.4 Group Clusters
**Purpose:** Merge clusters across episodes using Facebank

**Algorithm:**
1. For each cluster, compute centroid embedding
2. Query pgvector for nearest Facebank seed embeddings
3. If similarity > threshold, assign `person_id` label
4. Merge clusters with same `person_id`
5. Update `identities.json`

**Metrics:**
- `clusters_grouped`: Number of clusters merged via Facebank matching
- `auto_assigned_identities`: Number of identities auto-labeled

---

## 7. Common Workflows

### 7.1 Fix Over-Segmented Clusters
```bash
python tools/episode_cleanup.py \
  --ep-id <ep_id> \
  --actions recluster \
  --cluster-thresh 0.50 \
  --write-back
```

### 7.2 Fix Low-Quality Embeddings
```bash
python tools/episode_cleanup.py \
  --ep-id <ep_id> \
  --actions reembed recluster \
  --min-quality 0.8 \
  --write-back
```

### 7.3 Full Cleanup (All Phases)
```bash
python tools/episode_cleanup.py \
  --ep-id <ep_id> \
  --video <path> \
  --actions split_tracks reembed recluster group_clusters \
  --write-back
```

---

## 8. References

- [Pipeline Overview](overview.md)
- [Detect & Track](detect_track_faces.md)
- [Faces Harvest](faces_harvest.md)
- [Cluster Identities](cluster_identities.md)
- [Artifact Schemas](../reference/artifacts_faces_tracks_identities.md)

---

**Maintained by:** Screenalytics Engineering
