# Hardware Sizing — Screenalytics

**Version:** 2.0
**Last Updated:** 2025-11-18

---

## 1. Overview

This guide provides hardware recommendations for different Screenalytics deployment scenarios, from local dev to production season backfills.

---

## 2. Deployment Scenarios

### 2.1 Local Dev (Small Clips, Exploratory)

**Use Case:** Developers testing pipeline on 5–10 minute clips

**Recommended Hardware:**
- **CPU:** 4+ cores (Intel i5/i7, AMD Ryzen 5, Apple M1/M2)
- **RAM:** 8 GB
- **GPU:** Optional (integrated graphics OK)
- **Disk:** 50 GB SSD
- **OS:** macOS, Linux, Windows (WSL2)

**Performance Profile:** `fast_cpu` or `balanced`

**Expected Runtime:**
- 5-minute clip: ~30 seconds (CPU) or ~10 seconds (GPU)

**Cost:** Laptop/desktop you already have

---

### 2.2 Full-Episode Processing (Single Episodes)

**Use Case:** Processing individual 1-hour episodes (QA, manual moderation)

**Recommended Hardware:**
- **CPU:** 8+ cores (Intel i7/i9, AMD Ryzen 7, Apple M1 Pro/Max)
- **RAM:** 16 GB
- **GPU:** NVIDIA GTX 1660+ (6 GB VRAM) or Apple M1 Pro/Max
- **Disk:** 500 GB SSD
- **OS:** Linux (Ubuntu 22.04), macOS

**Performance Profile:** `balanced` (CPU) or `high_accuracy` (GPU)

**Expected Runtime:**
- 1-hour episode: ~2–3 hours (CPU) or ~10 minutes (GPU)

**Cost:** $1,500–$3,000 (desktop) or $2,000–$4,000 (laptop)

---

### 2.3 Season Backfills (Batch Processing)

**Use Case:** Processing 10–20 episodes per season (backfill, analytics pipeline)

**Recommended Hardware:**
- **CPU:** 16+ cores (AMD Ryzen 9, Intel Xeon, AMD Threadripper)
- **RAM:** 32 GB
- **GPU:** NVIDIA RTX 3080+ (10 GB VRAM) or NVIDIA A4000 (16 GB VRAM)
- **Disk:** 2 TB NVMe SSD
- **OS:** Linux (Ubuntu 22.04)

**Performance Profile:** `high_accuracy` (GPU)

**Expected Runtime:**
- 1-hour episode: ~5–10 minutes
- 20-episode season: ~2–3 hours (parallelized across 4 workers)

**Cost:** $3,000–$6,000 (desktop workstation) or $500–$1,000/month (cloud instance)

---

### 2.4 Production Cloud (Multi-Season, Multi-Show)

**Use Case:** Continuous processing of 100s of episodes, production API

**Recommended Hardware:**
- **Instance Type:** AWS `g4dn.2xlarge` or equivalent
  - CPU: 8 vCPUs (Intel Cascade Lake)
  - RAM: 32 GB
  - GPU: NVIDIA T4 (16 GB VRAM)
  - Disk: 500 GB EBS SSD
- **Workers:** 4–8 parallel workers
- **S3:** 10 TB storage
- **RDS:** Postgres 15+ with pgvector (db.r5.large)

**Performance Profile:** `high_accuracy` (GPU)

**Expected Runtime:**
- 1-hour episode: ~5 minutes
- 100-episode backfill: ~8 hours (8 workers in parallel)

**Cost:** ~$1,500–$3,000/month (AWS, 24/7)

---

## 3. Hardware Component Breakdown

### 3.1 CPU

| Cores | Use Case | Performance | Cost |
|-------|----------|-------------|------|
| 4 | Local dev, exploratory | Slow (5× realtime) | $200–$400 |
| 8 | Single episodes, balanced | Medium (3× realtime) | $400–$800 |
| 16+ | Season backfills, batch processing | Fast (with GPU) | $800–$2,000 |

**Recommendation:**
- **Apple Silicon (M1/M2/M3):** Best performance/watt for CPU-only
- **AMD Ryzen:** Best value for desktop (7000 series)
- **Intel Xeon:** Production servers (scalability)

---

### 3.2 GPU

| GPU | VRAM | Use Case | Speed vs CPU | Cost |
|-----|------|----------|--------------|------|
| Integrated (Intel/AMD) | Shared | Local dev only | 1× (no benefit) | Included |
| NVIDIA GTX 1660 | 6 GB | Single episodes | 5–10× | $200–$300 |
| NVIDIA RTX 3080 | 10 GB | Season backfills | 10–15× | $700–$1,200 |
| NVIDIA A4000 | 16 GB | Production | 15–20× | $1,500–$2,500 |
| NVIDIA A100 | 40 GB | Large-scale production | 20–30× | $10,000+ |
| Apple M1 Pro/Max | Unified | MacBook, single episodes | 3–5× | $2,000–$4,000 (laptop) |

**Recommendation:**
- **Budget:** NVIDIA GTX 1660 or RTX 3060
- **Production:** NVIDIA RTX 3080 or A4000
- **Cloud:** AWS g4dn (T4), g5 (A10G), or p3 (V100)

---

### 3.3 RAM

| RAM | Use Case | Headroom |
|-----|----------|----------|
| 8 GB | Local dev (small clips) | Tight |
| 16 GB | Single episodes | Comfortable |
| 32 GB | Season backfills, multiple workers | Plenty |
| 64 GB+ | Production, pgvector in-memory | Overkill (unless DB on same host) |

**Recommendation:** 16 GB minimum, 32 GB for production

---

### 3.4 Disk

| Disk | Size | Use Case | IOPS |
|------|------|----------|------|
| SSD (SATA) | 500 GB | Local dev | 500 MB/s |
| NVMe SSD | 1–2 TB | Production | 3,500 MB/s |
| EBS gp3 (AWS) | 500 GB–10 TB | Cloud | 3,000–16,000 IOPS |
| S3 | Unlimited | Archive, artifacts | N/A (network-bound) |

**Recommendation:** NVMe SSD for local processing, S3 for long-term storage

---

### 3.5 Network

| Bandwidth | Use Case | Latency |
|-----------|----------|---------|
| 1 Gbps | Local dev, single episodes | Low |
| 10 Gbps | Production, multi-worker | Very low |
| AWS VPC | Cloud, S3 access | Low (same region) |

**Recommendation:** 1 Gbps minimum, 10 Gbps for production (if local storage)

---

## 4. Cloud Instance Recommendations

### 4.1 AWS

| Instance Type | vCPU | RAM | GPU | Use Case | Cost (us-east-1) |
|---------------|------|-----|-----|----------|------------------|
| `c5.2xlarge` | 8 | 16 GB | None | CPU-only exploratory | $0.34/hr (~$250/mo) |
| `g4dn.xlarge` | 4 | 16 GB | T4 (16 GB) | Single episodes | $0.526/hr (~$380/mo) |
| `g4dn.2xlarge` | 8 | 32 GB | T4 (16 GB) | Production | $0.752/hr (~$550/mo) |
| `g5.2xlarge` | 8 | 32 GB | A10G (24 GB) | High throughput | $1.212/hr (~$880/mo) |
| `p3.2xlarge` | 8 | 61 GB | V100 (16 GB) | Large-scale | $3.06/hr (~$2,200/mo) |

**Recommendation:** `g4dn.2xlarge` for production (best $/performance)

### 4.2 GCP

| Instance Type | vCPU | RAM | GPU | Cost (us-central1) |
|---------------|------|-----|-----|---------------------|
| `n1-standard-8` | 8 | 30 GB | None | $0.38/hr (~$275/mo) |
| `n1-standard-8` + T4 | 8 | 30 GB | T4 (16 GB) | $0.73/hr (~$530/mo) |
| `a2-highgpu-1g` | 12 | 85 GB | A100 (40 GB) | $3.67/hr (~$2,660/mo) |

### 4.3 Azure

| Instance Type | vCPU | RAM | GPU | Cost (East US) |
|---------------|------|-----|-----|----------------|
| `Standard_D8s_v3` | 8 | 32 GB | None | $0.384/hr (~$280/mo) |
| `Standard_NC6s_v3` | 6 | 112 GB | V100 (16 GB) | $3.06/hr (~$2,220/mo) |

---

## 5. Storage Sizing

### 5.1 Artifacts per Episode (1 hour, 24 FPS)

| Artifact | Size (Frames OFF) | Size (Frames ON) | Size (Frames + Crops) |
|----------|-------------------|------------------|-----------------------|
| `detections.jsonl` | ~1 MB | ~1 MB | ~1 MB |
| `tracks.jsonl` | ~100 KB | ~100 KB | ~100 KB |
| `faces.jsonl` | ~5 MB | ~5 MB | ~5 MB |
| `faces.npy` | ~20 MB | ~20 MB | ~20 MB |
| `identities.json` | ~50 KB | ~50 KB | ~50 KB |
| Frames JPG | 0 | ~500 MB | ~500 MB |
| Crops JPG | 0 | 0 | ~200 MB |
| **Total** | **~26 MB** | **~526 MB** | **~726 MB** |

### 5.2 Season Storage (20 episodes)

| Exporters | Per Episode | Per Season (20 ep) |
|-----------|-------------|---------------------|
| None | ~26 MB | ~520 MB |
| Frames only | ~526 MB | ~10.5 GB |
| Frames + Crops | ~726 MB | ~14.5 GB |

**Recommendation:**
- **Local dev:** No exporters (manifests only)
- **Production:** Frames + Crops for first pass, expire after N days

---

## 6. Cost Analysis

### 6.1 Local Hardware (One-Time)

| Configuration | Hardware Cost | Use Case |
|---------------|---------------|----------|
| Laptop (M1 Pro) | $2,000–$3,000 | Dev, single episodes |
| Desktop (Ryzen 9 + RTX 3080) | $3,000–$4,000 | Season backfills |
| Workstation (Threadripper + A4000) | $6,000–$10,000 | Production |

### 6.2 Cloud (Monthly)

| Configuration | Monthly Cost | Throughput (episodes/day) |
|---------------|--------------|---------------------------|
| 1× `g4dn.xlarge` (T4) | ~$380 | ~20 (1hr episodes) |
| 4× `g4dn.xlarge` (T4) | ~$1,520 | ~80 |
| 2× `g4dn.2xlarge` (T4) | ~$1,100 | ~60 |

**Break-Even:** ~12–18 months (cloud vs local workstation)

---

## 7. Scaling Strategies

### 7.1 Horizontal Scaling (Multiple Workers)
- **Redis queue:** Add more worker instances
- **S3:** Shared artifact storage
- **Postgres + pgvector:** Shared DB (scale reads with replicas)

**Bottlenecks:**
- Redis queue throughput (~10,000 jobs/sec)
- S3 write throughput (~3,500 requests/sec per prefix)
- Postgres connections (~100 concurrent connections default)

### 7.2 Vertical Scaling (Bigger Instances)
- **GPU:** Larger VRAM for bigger batch sizes
- **CPU:** More cores for parallel decode/encode
- **RAM:** Larger batches in-memory

---

## 8. References

- [Performance Tuning](performance_tuning_faces_pipeline.md)
- [Pipeline Overview](../pipeline/overview.md)
- [Config Reference](../reference/config/pipeline_configs.md)

---

**Maintained by:** Screenalytics Engineering
