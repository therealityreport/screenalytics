# SOLUTION_ARCHITECTURE.md — Screenalytics

Version: 1.1 (Enhanced)  
Status: Planning  

---

## 1. Purpose
Screenalytics transforms raw video and audio into structured cast-level analytics. It quantifies screen presence, speaking time, and visual engagement for every episode, powering TRR’s analytics, design, and merch systems.

---

## 2. Architecture Overview
```

[Workspace UI (Next.js)]
│
▼
[API (FastAPI)] ──► [Redis Queue] ──► [Workers (Python)]
│                               │
│                               ├─► Object Storage (S3/R2/GCS)
│                               └─► Postgres + pgvector
│
├─► MCP Servers (screenalytics / storage / postgres)
│        ▲
│        └─ Codex + Claude Agents
└─► Webhooks (Zapier/n8n) → Exports, Alerts

```

---

## 3. Core Components
| Component | Function | Key Tech |
|------------|-----------|----------|
| **UI** | Upload, workspace, SHOWS/PEOPLE catalog | Next.js, TypeScript |
| **API** | CRUD, orchestration, signed URLs | FastAPI |
| **Workers** | Pipeline stages (Detect → Track → ID → Fuse → Aggregate) | Python (uv), Redis |
| **Storage** | Object storage for frames, chips, reports | S3/R2/GCS |
| **Database** | Structured + vector data | Postgres + pgvector |
| **Agents** | Autonomous updates + documentation | Codex + Claude via MCP |
| **Automation** | Notifications and exports | Zapier / n8n |

---

## 4. Pipeline
| Stage | Input | Output | Tools |
|--------|--------|---------|-------|
| **Ingest** | MP4 | frames/, audio.wav | FFmpeg, OpenCV |
| **Shot Split** | video | shots.csv | PySceneDetect |
| **Detect** | frames | detections.jsonl | RetinaFace |
| **Align / Quality** | faces | chips/, manifest | InsightFace, MagFace |
| **Track** | detections | tracks.jsonl | ByteTrack |
| **Embed / ID** | chips | embeddings.parquet, assignments | ArcFace ONNX, pgvector |
| **Audio Fuse** | audio.wav | speech_segments, transcripts | Pyannote, Faster-Whisper |
| **Aggregate** | assignments + speech | screen_time.csv/json | pandas |
| **QA / Review** | artifacts | overrides + history | Streamlit / Next.js UI |

---

## 5. Data Model (Simplified)
```

show → season → episode → track → embedding → assignment → screen_time
person → cast_membership → facebank → featured_thumbnail
speech_segment → av_link (track_id + seg_id)

```
Stored in Postgres, with vectors indexed via HNSW in pgvector.

---

## 6. Storage Layout
```

videos/{show}/{SxxEyy}/episode.mp4
audio/{show}/{SxxEyy}/episode.wav
frames/{show}/{SxxEyy}/{shot}/{ts}.jpg
chips/{show}/{SxxEyy}/{track}/{n}.jpg
facebank/{person}/ref_{hash}.jpg
reports/{show}/{SxxEyy}/screen_time.csv

```

Lifecycle rules:
- Expire `frames/` after N days
- Retain `chips/`, `facebank/`, `reports/`

---

## 7. Configuration Layers
- **Pipeline configs** (`config/pipeline/*.yaml`): thresholds, model IDs, stage toggles  
- **Service configs** (`config/services.yaml`): timeouts, endpoints  
- **Storage** (`config/storage.yaml`): bucket names, lifecycle  
- **Agents** (`config/codex.config.toml`, `config/claude.policies.yaml`): write rules and MCP servers  

---

## 8. Observability
- Structured logs → stdout (JSON)
- Progress events → Redis PubSub
- Metrics: throughput, error count, avg runtime per stage
- QA view: aggregated low-confidence tracks

---

## 9. Security
- Signed URLs (temporary)
- RLS by show
- Audit trail via `assignment_history`
- No PII; embeddings only

---

## 10. Performance Targets
- 1-hour episode ≤ 10 min on single GPU  
- ID accuracy ≥ 90%  
- Speaking match ≥ 85%  
- CI integration test: ≤ 15 min full pipeline run

---

## 11. Promotion & CI Flow
1. Build feature in `FEATURES/<name>/`  
2. Test, document, and config-drive it  
3. Promote via PR → CI verifies  
4. Agents update docs automatically (see configuration below)

---

## 12. Agents & Automation Hooks
- **Trigger:** when files are added or removed  
- **Action:** Codex & Claude agents update  
  - `SOLUTION_ARCHITECTURE.md`
  - `DIRECTORY_STRUCTURE.md`
  - `PRD.md`
  - `README.md`
- **Location:** `/agents/playbooks/update-docs-on-change.yaml`
