# Performance Tuning — Faces Pipeline

**Version:** 2.0
**Last Updated:** 2025-11-18

---

## 1. Overview

This guide explains how to tune the Screenalytics pipeline for **speed vs accuracy** trade-offs across different hardware profiles.

**Key Principles:**
- **CPU-only:** Conservative defaults to prevent overheating and thermal throttling
- **GPU:** Aggressive settings for maximum throughput
- **Profiles:** Pre-configured performance profiles for common use cases

---

## 2. Performance Profiles

### 2.1 Pre-Configured Profiles

| Profile | Device | Stride | FPS | Batch | Exporters | Use Case | Expected Runtime (1hr episode) |
|---------|--------|--------|-----|-------|-----------|----------|--------------------------------|
| **fast_cpu** | CPU (fanless) | 10 | ≤4 | 1 | Off | Exploratory, thermal-constrained | ~5× realtime (~5 hours) |
| **balanced** | CPU/MPS | 5 | 8 | 2 | Frames only | Standard local dev | ~3× realtime (~3 hours) |
| **high_accuracy** | GPU (CUDA) | 1 | 30 | 8 | Frames + Crops | Production, max recall | ≤10 minutes |

### 2.2 Using Profiles
```bash
# CLI
python tools/episode_run.py --ep-id <ep_id> --video <path> --profile fast_cpu

# API
POST /jobs/detect_track_async
{
  "ep_id": "rhobh-s05e02",
  "profile": "balanced"
}
```

**Profile config:** `config/pipeline/performance_profiles.yaml`

---

## 3. Key Performance Knobs

### 3.1 Stride (Frame Sampling)
**What:** Process every Nth frame

| Stride | FPS Effective | Speed | Recall | Use Case |
|--------|---------------|-------|--------|----------|
| 1 | 100% | Slow | Max | Production, GPU |
| 3 | 33% | Fast | Good | Standard CPU |
| 5 | 20% | Faster | OK | Exploratory CPU |
| 10 | 10% | Fastest | Low | Quick preview |

**Recommendation:**
- GPU: `stride=1` (max recall)
- CPU balanced: `stride=3–5`
- CPU fast: `stride=10`

**CLI:**
```bash
python tools/episode_run.py --stride 3
```

### 3.2 Analyzed FPS (Video Resampling)
**What:** Resample video to target FPS before processing

| FPS | Speed | Recall | Use Case |
|-----|-------|--------|----------|
| 30 | Slow | Max | GPU, production |
| 24 | Medium | Good | Standard |
| 8 | Fast | OK | CPU exploratory |
| 4 | Fastest | Low | Quick preview |

**Recommendation:**
- GPU: `fps=30` (or auto-detect)
- CPU balanced: `fps=8`
- CPU fast: `fps=4`

**CLI:**
```bash
python tools/episode_run.py --fps 8
```

### 3.3 Device Selection
**Order (auto):** CUDA GPU → Apple CoreML (MPS) → CPU

| Device | Speed | Use Case |
|--------|-------|----------|
| `cuda` | 10–20× faster than CPU | Production, GPU servers |
| `mps` | 3–5× faster than CPU | Apple Silicon (M1/M2/M3) |
| `cpu` | Baseline | Fallback, no GPU |

**CLI:**
```bash
# Auto-detect (recommended)
python tools/episode_run.py --device auto

# Force CPU (testing)
python tools/episode_run.py --device cpu

# Force GPU
python tools/episode_run.py --device cuda
```

### 3.4 Exporters (Save Frames/Crops)
**What:** Write frames and face crops to disk

| Exporter | Disk I/O | Speed Impact | Use Case |
|----------|----------|--------------|----------|
| None | Minimal | None | Fast runs, no downstream needs |
| Frames only | Moderate | ~10–20% slower | UI preview, debugging |
| Frames + Crops | High | ~30–50% slower | Full pipeline, Facebank |

**Recommendation:**
- Fast preview: Disable both
- Standard: Frames only
- Production: Frames + Crops

**CLI:**
```bash
# Disable exporters
python tools/episode_run.py --no-save-frames --no-save-crops

# Enable both
python tools/episode_run.py --save-frames --save-crops --jpeg-quality 90
```

---

## 4. Threading & Parallelism

### 4.1 Thread Limits (CPU)
**Problem:** PyTorch, ONNX, OpenCV default to using all CPU cores → overheating, thermal throttling

**Solution:** Limit thread counts via environment variables

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

python tools/episode_run.py ...
```

**Recommended values:**
- Fanless devices (MacBook Air): `2`
- Standard laptop: `4`
- Desktop (8+ cores): `8`

### 4.2 Batch Size (GPU)
**What:** Number of faces processed per batch during embedding

| Batch Size | GPU Memory | Speed | Use Case |
|------------|------------|-------|----------|
| 1 | Low | Slow | CPU, low VRAM GPU |
| 4 | Medium | Good | Balanced |
| 8 | High | Fast | High-end GPU (8GB+ VRAM) |
| 16+ | Very High | Fastest | A100, H100 |

**CLI (future):**
```bash
python tools/episode_run.py --embed-batch-size 8
```

**Currently:** Hardcoded to 1 (CPU) or 4 (GPU); future config option.

---

## 5. Quality vs Speed Trade-offs

### 5.1 Detection Thresholds
```yaml
# config/pipeline/detection.yaml
min_size: 90           # Lower = more faces, slower (e.g., 64)
confidence_th: 0.8     # Lower = more faces, more false positives (e.g., 0.6)
```

**Speed gain (lower thresholds):** ~10–20%
**Recall gain (lower thresholds):** +15–30%
**Precision loss:** +5–10% false positives

### 5.2 Tracking Thresholds
```yaml
# config/pipeline/tracking.yaml
track_thresh: 0.70     # Lower = more tracks, more noise
track_buffer: 90       # Lower = faster (fewer tracks kept alive)
```

**Speed gain (lower buffer):** ~5–10%
**Quality loss:** More ID switches, fragmented tracks

### 5.3 Embedding Quality Gating
```yaml
# config/pipeline/faces_embed_sampling.yaml
min_quality: 0.7       # Higher = fewer faces, faster embedding
max_crops_per_track: 50  # Lower = faster (e.g., 20)
```

**Speed gain (fewer crops):** ~20–40% on embedding stage
**Quality loss:** Looser clusters, more singletons

---

## 6. Hardware-Specific Recommendations

### 6.1 MacBook Air M1/M2 (Fanless)
**Profile:** `fast_cpu`

**Settings:**
```bash
export OMP_NUM_THREADS=2
python tools/episode_run.py \
  --profile fast_cpu \
  --device mps \
  --coreml-det-size 384 \
  --no-save-frames --no-save-crops
```

**Expected runtime (1hr episode):** ~4–6 hours

### 6.2 MacBook Pro M1/M2/M3 (Active Cooling)
**Profile:** `balanced`

**Settings:**
```bash
export OMP_NUM_THREADS=4
python tools/episode_run.py \
  --profile balanced \
  --device mps \
  --save-frames
```

**Expected runtime (1hr episode):** ~2–3 hours

### 6.3 Desktop CPU (8+ cores)
**Profile:** `balanced`

**Settings:**
```bash
export OMP_NUM_THREADS=8
python tools/episode_run.py \
  --profile balanced \
  --device cpu \
  --save-frames --save-crops
```

**Expected runtime (1hr episode):** ~2–3 hours

### 6.4 GPU Server (CUDA)
**Profile:** `high_accuracy`

**Settings:**
```bash
python tools/episode_run.py \
  --profile high_accuracy \
  --device cuda \
  --save-frames --save-crops \
  --embed-batch-size 8
```

**Expected runtime (1hr episode):** ~5–10 minutes

---

## 7. Profiling & Monitoring

### 7.1 Progress Tracking
All stages write `progress.json` with:
```json
{
  "phase": "track",
  "frames_done": 1200,
  "frames_total": 2400,
  "elapsed_sec": 45.2,
  "fps_detected": 24.0,
  "analyzed_fps": 8.0,
  "eta_sec": 47.8
}
```

**Monitor:**
```bash
watch -n 1 cat data/manifests/{ep_id}/progress.json
```

### 7.2 Timing Breakdown
`track_metrics.json` includes:
```json
{
  "elapsed_sec": 120.5,
  "total_frames": 2400,
  "analyzed_fps": 8.0,
  "device": "cpu"
}
```

**Stages (typical 1hr episode, CPU):**
- Video decode: ~10 sec
- Detection: ~90 sec
- Tracking: ~15 sec
- Export (if enabled): ~20 sec
- **Total:** ~135 sec (~2 minutes)

**GPU (same episode):**
- Detection: ~15 sec
- Tracking: ~5 sec
- **Total:** ~25 sec

---

## 8. Optimization Checklist

### 8.1 For Speed
- [ ] Use GPU if available (`--device cuda`)
- [ ] Increase stride (`--stride 5` or `10`)
- [ ] Decrease analyzed FPS (`--fps 4` or `8`)
- [ ] Disable exporters (`--no-save-frames --no-save-crops`)
- [ ] Limit threads (`export OMP_NUM_THREADS=4`)
- [ ] Increase `min_size` to filter small faces (`min_size: 120`)
- [ ] Decrease `max_crops_per_track` for embedding (`20` instead of `50`)

### 8.2 For Recall
- [ ] Decrease stride (`--stride 1`)
- [ ] Increase analyzed FPS (`--fps 30`)
- [ ] Decrease `min_size` (`min_size: 64`)
- [ ] Decrease `confidence_th` (`0.6` instead of `0.8`)
- [ ] Enable adaptive confidence (`adaptive_confidence: true`)
- [ ] Increase `track_buffer` (`120` instead of `90`)
- [ ] Increase `max_crops_per_track` (`50` or `100`)

### 8.3 For Thermal Management (CPU)
- [ ] Use `fast_cpu` profile
- [ ] Limit threads (`export OMP_NUM_THREADS=2`)
- [ ] Increase stride (`--stride 10`)
- [ ] Disable exporters
- [ ] Decrease CoreML input size (`--coreml-det-size 384` on macOS)
- [ ] Run in low-power mode (macOS: System Preferences → Battery → Low Power Mode)

---

## 9. References

- [Pipeline Overview](../pipeline/overview.md)
- [Detect & Track](../pipeline/detect_track_faces.md)
- [Config Reference](../reference/config/pipeline_configs.md)
- [Troubleshooting](troubleshooting_faces_pipeline.md)
- [Hardware Sizing](hardware_sizing.md)

---

**Maintained by:** Screenalytics Engineering
