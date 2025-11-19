# Solution Architecture — Screenalytics

**Quick Reference** | [Full Documentation →](docs/architecture/solution_architecture.md)

---

## Overview

Screenalytics transforms raw video episodes into structured, per-person screentime analytics through an automated ML pipeline.

**Pipeline:** Detect → Track → Embed → Cluster → Cleanup → Audio Diarize → A/V Fusion → Aggregate → Export

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      WORKSPACE UI (Next.js)                      │
│  SHOWS Directory | PEOPLE Directory | Episode Workspace | Facebank│
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/REST + SSE
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        API (FastAPI)                             │
│  Episodes CRUD | Jobs Orchestrator | Identities | Facebank      │
└────┬───────────────┬──────────────────────────────┬─────────────┘
     │               │                              │
     ▼               ▼                              ▼
┌─────────┐   ┌──────────────┐    ┌────────────────────────┐
│ Redis   │   │ Postgres     │    │ Object Storage         │
│ Queue   │   │ + pgvector   │    │ (S3/R2/GCS/MinIO)      │
└────┬────┘   └──────────────┘    └────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WORKERS (Pipeline Stages)                     │
│  Detect | Track | Embed | Cluster | Cleanup | Audio | Aggregate │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **UI** | Upload episodes, run jobs, moderate identities | Next.js, TypeScript |
| **API** | CRUD operations, job orchestration, signed URLs | FastAPI, Python 3.11+ |
| **Workers** | ML pipeline stages (detect/track/embed/cluster) | RetinaFace, ByteTrack, ArcFace ONNX |
| **Storage** | Video, frames, crops, manifests, analytics | S3-compatible (AWS S3, R2, GCS, MinIO) |
| **Database** | Metadata, embeddings (pgvector), audit trails | Postgres 15+ with pgvector |
| **Agents** | Auto-documentation, low-confidence relabeling | Codex SDK, Claude API, MCP |

---

## Performance & Limits

**Safe Defaults (CPU-only):**
- Stride: 3–10 (process every Nth frame)
- Analyzed FPS: 4–8
- Exporters: Off by default (frames/crops optional)
- Threading: Limited (`OMP_NUM_THREADS=2` for fanless devices)

**GPU Recommendations:**
- Stride: 1 (max recall)
- Analyzed FPS: 24–30
- Exporters: Frames + Crops
- Expected runtime (1hr episode): ~5–10 minutes

**⚠️ Warning:** Aggressive settings (stride=1, high FPS, exporters ON) on CPU will be slow and hot. Use performance profiles:
- `fast_cpu` — Fanless devices, exploratory
- `balanced` — Standard local dev
- `high_accuracy` — GPU, production

---

## Documentation

**For complete details, see:**

- **[Full Solution Architecture](docs/architecture/solution_architecture.md)** — System design, data model, configuration layers
- **[Directory Structure](docs/architecture/directory_structure.md)** — Repo layout, promotion policy
- **[Pipeline Overview](docs/pipeline/overview.md)** — Stage-by-stage pipeline guide
- **[Performance Tuning](docs/ops/performance_tuning_faces_pipeline.md)** — Speed vs accuracy tuning
- **[Troubleshooting](docs/ops/troubleshooting_faces_pipeline.md)** — Common issues and fixes

---

**Maintained by:** Screenalytics Engineering
