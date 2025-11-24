# Troubleshooting — Faces Pipeline

**Version:** 2.0
**Last Updated:** 2025-11-18

---

## 1. Overview

This guide provides **symptom → cause → fix** tables for common issues in the detect/track/embed/cluster pipeline.

---

## 2. Detection & Tracking Issues

### 2.1 Too Many Tracks (Exploding Track Count)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `tracks_per_minute > 50` | `track_thresh` too low | Increase to 0.75–0.85 |
| `short_track_fraction > 0.5` | Background motion creating ghost tracks | Increase `new_track_thresh` to 0.85–0.90 |
| Thousands of fleeting tracks | `min_box_area` too small | Set `min_box_area: 400` to filter tiny faces |
| Tracker resets every frame | Scene detection too sensitive | Increase `scene_threshold` or disable scene detection |

**Quick Fix:**
```bash
python tools/episode_run.py \
  --ep-id <ep_id> --video <path> \
  --profile balanced  # Uses saner defaults
```

---

### 2.2 Missed Faces (Low Recall)

| Symptom | Cause | Fix |
|---------|-------|-----|
| Known faces not detected | `stride` too high | Decrease to 3 or 1 |
| Small/distant faces missing | `min_size` too large | Decrease to 64 or 48 |
| Dark scenes missing faces | `confidence_th` too high | Decrease to 0.6, enable `adaptive_confidence: true` |
| Side profiles missing | `check_pose_quality: true` filtering them | Disable pose check or increase `max_yaw_angle` |

**Quick Fix:**
```bash
python tools/episode_run.py \
  --stride 1 \
  --min-face-size 64 \
  --min-face-conf 0.6 \
  --profile high_accuracy
```

---

### 2.3 ID Switches (Tracker Instability)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `id_switch_rate > 0.1` | `match_thresh` too low | Increase to 0.85–0.90 |
| Tracks dying and reviving | `track_buffer` too short | Increase to 120–180 |
| Appearance gate splitting aggressively | `TRACK_GATE_APPEAR_HARD` too high | Decrease to 0.60 |
| Camera pans causing splits | GMC disabled | Enable GMC: `export SCREENALYTICS_GMC_METHOD=sparseOptFlow` |

**Quick Fix:**
```bash
export TRACK_GATE_APPEAR_HARD=0.60
export TRACK_GATE_APPEAR_STREAK=5
python tools/episode_run.py ...
```

---

### 2.4 Performance Issues (Overheating, Slow Processing)

| Symptom | Cause | Fix |
|---------|-------|-----|
| CPU thermal throttling, fans at max | Too many threads | Limit: `export OMP_NUM_THREADS=2` |
| Processing < 1 FPS on CPU | Profile too aggressive | Use `--profile fast_cpu` |
| Disk I/O bottleneck | Exporters writing thousands of files | Disable: `--no-save-frames --no-save-crops` |
| GPU not utilized | Wrong device selected | Force: `--device cuda` |

**Quick Fix (Fanless Devices):**
```bash
export OMP_NUM_THREADS=2
python tools/episode_run.py \
  --profile fast_cpu \
  --no-save-frames --no-save-crops
```

---

## 3. Embedding & Crop Issues

### 3.1 Blank/Gray Crops

| Symptom | Cause | Fix |
|---------|-------|-----|
| Face crops are blank rectangles | Out-of-bounds bbox | Enable `check_pose_quality: true` to discard unreliable landmarks |
| Crops are gray | Invalid landmarks | Review `crops_debug.jsonl` (with `DEBUG_THUMBS=1`) |
| Crops have wrong aspect ratio | Crop generation bug | Check RetinaFace bbox clamping logic |

**Debugging:**
```bash
DEBUG_THUMBS=1 python tools/episode_run.py --ep-id <ep_id> --faces-embed --save-crops
python tools/debug_thumbs.py data/frames/<ep_id>/crops_debug.jsonl
```

---

### 3.2 Too Many Faces (Memory Exhaustion)

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM error during embedding | No volume control | Set `max_crops_per_track: 50` |
| Embedding stage takes hours | Too many crops | Decrease to `max_crops_per_track: 20`, use quality-weighted sampling |

**Quick Fix:**
```yaml
# config/pipeline/faces_embed_sampling.yaml
max_crops_per_track: 20
min_quality: 0.8
```

---

### 3.3 Embedding Dimension Mismatch

| Symptom | Cause | Fix |
|---------|-------|-----|
| `faces.npy` shape is not `(N, 512)` | Wrong ArcFace model | Confirm model ID: `arcface_r100_v1` |
| Embeddings not unit-norm | Normalization bug | Verify: `np.linalg.norm(embeddings[i]) ≈ 1.0` |

---

## 4. Clustering Issues

### 4.1 Too Many Clusters (Over-Segmentation)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `num_clusters > 30` | `cluster_thresh` too high | Decrease to 0.50–0.55 |
| Many small clusters | Poor embedding quality | Increase `min_quality` in faces_embed (0.8+) |
| Clusters per person | Under-sampling (too few crops per track) | Increase `max_crops_per_track` to 50–100 |

**Quick Fix:**
```bash
python tools/episode_cleanup.py \
  --ep-id <ep_id> \
  --actions recluster \
  --cluster-thresh 0.50 \
  --write-back
```

---

### 4.2 Over-Merged Cluster (One Mega-Cluster)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `largest_cluster_fraction > 0.6` | `cluster_thresh` too low | Increase to 0.65–0.70 |
| All faces look similar | Poor ArcFace discrimination | Review embedding distribution (t-SNE/UMAP plot) |
| Duplicate tracks | Same person tracked multiple times | Increase `track_buffer`, run cleanup `split_tracks` |

**Quick Fix:**
```bash
python tools/episode_cleanup.py \
  --ep-id <ep_id> \
  --actions split_tracks recluster \
  --cluster-thresh 0.65 \
  --write-back
```

---

### 4.3 High Singleton Fraction

| Symptom | Cause | Fix |
|---------|-------|-----|
| `singleton_fraction > 0.5` | `min_cluster_size` too high | Decrease to 1 (allow singleton clusters) |
| Many unidentified tracks | Poor embedding quality | Re-embed with higher `min_quality` |
| Short tracks with few faces | Filtering issue | Review track lengths, filter short tracks in cleanup |

---

### 4.4 Canonical Thumbnail is Poor Quality

| Symptom | Cause | Fix |
|---------|-------|-----|
| Representative thumbnail blurry/occluded | Selection logic prioritizes wrong metric | Adjust to prioritize `quality_score` > `face_size` |
| Thumbnail from extreme pose | Pose filtering disabled | Enable `check_pose_quality: true` |

**Manual Fix:** Replace thumbnail via Facebank UI

---

## 5. Episode Cleanup Issues

### 5.1 Cleanup Report Shows No Improvement

| Symptom | Cause | Fix |
|---------|-------|-----|
| `before` == `after` metrics | No tracks needed splitting | Skip `split_tracks` action |
| Singleton fraction unchanged | Threshold unchanged | Adjust `cluster_thresh` before re-clustering |

---

### 5.2 Cleanup Takes Too Long

| Symptom | Cause | Fix |
|---------|-------|-----|
| Cleanup runtime > 1 hour | Re-running full detect/track | Use `actions: [recluster]` only if detect/track already good |
| Re-embedding all faces | Too many crops | Limit `max_crops_per_track` before cleanup |

---

## 6. API & Job Issues

### 6.1 Job Stuck in "running" State

| Symptom | Cause | Fix |
|---------|-------|-----|
| Job state never updates | Worker crashed | Check worker logs: `docker logs worker` |
| `progress.json` not updating | Process killed (OOM) | Reduce batch size, increase memory |

**Check job status:**
```bash
curl http://localhost:8000/jobs/{job_id}/progress
```

---

### 6.2 SSE Stream Disconnects

| Symptom | Cause | Fix |
|---------|-------|-----|
| SSE connection drops after 30s | Proxy timeout | Fallback to `detect_track_async` + polling |
| No progress events | API not streaming | Use polling: `GET /jobs/{job_id}/progress` |

---

## 7. Artifact Issues

### 7.1 Missing Artifacts

| Symptom | Cause | Fix |
|---------|-------|-----|
| `tracks.jsonl` not found | Job failed midway | Check logs for errors, re-run |
| `faces.npy` empty | No faces extracted | Review quality gating thresholds |
| `identities.json` missing | Clustering failed | Check `cluster_thresh` and `min_cluster_size` |

---

### 7.2 Corrupted Artifacts

| Symptom | Cause | Fix |
|---------|-------|-----|
| JSONL malformed | Write interrupted | Re-run stage with `--write-back` |
| NPY wrong shape | Concatenation bug | Delete and re-run embedding |

---

## 8. Quick Diagnostic Commands

### 8.1 Check Track Metrics
```bash
cat data/manifests/{ep_id}/track_metrics.json | jq .
```

Look for:
- `tracks_per_minute > 50` → Too many tracks
- `short_track_fraction > 0.3` → Too many ghost tracks
- `id_switch_rate > 0.1` → Tracker unstable

### 8.2 Check Cluster Metrics
```bash
cat data/manifests/{ep_id}/identities.json | jq '.stats.singleton_stats'
```

Look for:
- `before.singleton_fraction` vs `after.singleton_fraction` to confirm singleton-merge impact
- `after.cluster_count` and `after.merge_count` to see how many identities were merged
- `largest_cluster_fraction > 0.6` (in `.stats.largest_cluster_fraction`) → Over-merged

#### High singleton fraction (merge disabled) {#high-singleton-fraction}
- Guardrail fires when `singleton_fraction_before > 0.50`.
- Tuning: decrease `cluster_thresh` to `0.65–0.62` or raise `faces_embed.min_quality` (e.g., 0.35 → 0.45).
- If most clusters are singletons because tracks are noisy, revisit detection thresholds and tracker IOU gates before tightening clustering.

#### High singleton fraction with singleton merge {#high-singleton-fraction-with-singleton-merge}
- Guardrail uses `after.singleton_fraction` when `singleton_merge.enabled: true`.
- Logs/UI show `Singletons: {before} → {after} (threshold {t})` plus merge config (`similarity_thresh`, `neighbor_top_k`, `merge_count`).
- If `after` is still above threshold:
  - Tighten `cluster_thresh` by +0.03 to +0.07.
  - Raise `faces_embed.min_quality` to filter low-quality faces.
  - Increase `singleton_merge.similarity_thresh` (e.g., 0.60 → 0.64) or reduce `neighbor_top_k` (10–30 works well).
  - Re-run and compare `before` vs `after` to confirm the merge is actually being applied.

### 8.3 Check Embedding Quality
```bash
python - <<'PY'
import numpy as np
faces = np.load("data/embeds/{ep_id}/faces.npy")
print(f"Shape: {faces.shape}")
print(f"Norm: {np.linalg.norm(faces[0]):.4f}")  # Should be ≈ 1.0
PY
```

---

## 9. Getting Help

### 9.1 Logs
- **API logs:** `docker logs api`
- **Worker logs:** `docker logs worker`
- **Episode run logs:** Stdout from `python tools/episode_run.py`

### 9.2 Debug Mode
```bash
DEBUG=1 python tools/episode_run.py ...
```

### 9.3 GitHub Issues
Report issues at: `https://github.com/<your-org>/screenalytics/issues`

Include:
- `track_metrics.json`
- `cleanup_report.json` (if applicable)
- `progress.json`
- Logs (sanitized)

---

## 10. References

- [Pipeline Overview](../pipeline/overview.md)
- [Performance Tuning](performance_tuning_faces_pipeline.md)
- [Config Reference](../reference/config/pipeline_configs.md)
- [Artifact Schemas](../reference/artifacts_faces_tracks_identities.md)

---

**Maintained by:** Screenalytics Engineering
