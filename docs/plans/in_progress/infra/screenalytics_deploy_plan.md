# SCREENALYTICS Infrastructure Deployment Plan

**Version:** 1.0
**Status:** Draft
**Last Updated:** 2025-12-02

---

## 1. Overview

SCREENALYTICS runs as a **Vercel-hosted Next.js web application** communicating with an **EC2-hosted FastAPI backend** that orchestrates Celery workers, Redis, and connects to S3 for object storage and Postgres for metadata. The frontend uses typed API clients with Server-Sent Events (SSE) for real-time job progress. Heavy compute (face detection, tracking, clustering, audio processing) runs on EC2, keeping Vercel functions lightweight.

> **Note:** This plan documents infrastructure and hosting only. It does **not** change the audio pipeline implementation, config files, or any audio-related code.

---

## 2. Goals

- **Move heavy jobs off local dev machines** — Pipeline jobs (detect, track, faces, cluster) run on persistent EC2 infrastructure
- **Keep Next.js UI on Vercel** — Leverage Vercel's edge network, zero-config deployments, and preview environments
- **Keep S3 + Postgres on AWS** — Minimize latency between compute and storage
- **Single EC2 host for API + workers + Redis** — Simple initial architecture, scale later
- **Background jobs without laptop dependency** — Users can submit jobs and close their laptops

---

## 3. Non-goals

- **No changes to audio pipeline implementation** — Audio code, configs, and workflows remain unchanged
- **No Kubernetes or multi-node cluster design** — Start simple with a single EC2 instance
- **No full cost optimization** — This is a "sane, production-capable starter architecture"
- **No breaking changes to existing API contracts** — Frontend/backend interfaces remain stable
- **No IaC implementation in this task** — Document only; Terraform/CDK is future work

---

## 4. Current State

Today, SCREENALYTICS development runs across multiple local components:

| Component | Current Setup |
|-----------|---------------|
| **UI** | Streamlit (`apps/workspace-ui/`) and Next.js (`web/`) running locally |
| **API** | FastAPI (`apps/api/main.py`) on local machine, port 8000 |
| **Workers** | Celery workers on local machine when needed |
| **Redis** | Local redis-server or Docker |
| **Storage** | S3 bucket (`screenalytics`) already in production use |
| **Database** | Postgres (local or RDS) for metadata |

**Pain points:**
- Jobs stop when laptop sleeps or disconnects
- Local thermal throttling affects long-running ML jobs
- No persistent environment for testing multi-hour episodes
- Streamlit UI not suitable for production deployment

---

## 5. Proposed Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              USERS                                          │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         VERCEL (Frontend)                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Next.js App (web/)                                                   │  │
│  │  • /screenalytics/* routes                                            │  │
│  │  • Typed API client (api/client.ts)                                   │  │
│  │  • SSE EventSource for job progress                                   │  │
│  │  • Presigned URLs for media thumbnails                                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│  Environment: NEXT_PUBLIC_API_BASE=https://api.<screenalytics-domain>      │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │ HTTPS / SSE
                                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         EC2 INSTANCE (Backend)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  nginx (TLS termination, reverse proxy)                              │   │
│  │  • https://api.<screenalytics-domain>:443 → localhost:8000           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  FastAPI (uvicorn, port 8000)                                        │   │
│  │  • REST endpoints: /episodes, /jobs, /identities, /facebank, etc.    │   │
│  │  • SSE endpoint: /episodes/{id}/events                               │   │
│  │  • Presigned URL generation for S3                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Celery Workers (--concurrency=2-4)                                  │   │
│  │  • detect_track, faces_embed, cluster queues                         │   │
│  │  • Audio queues (existing, unchanged)                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Redis (localhost:6379)                                              │   │
│  │  • Celery broker + result backend                                    │   │
│  │  • Job locking (screenalytics:job_lock:*)                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
            ▼                        ▼                        ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│   S3 Bucket       │    │   Postgres (RDS)   │    │   External APIs   │
│   (screenalytics) │    │   or Managed PG    │    │   (Resemble, etc) │
│                   │    │                    │    │                   │
│ • raw/videos/     │    │ • episodes         │    │ • Audio enhance   │
│ • artifacts/      │    │ • identities       │    │ • Diarization     │
│ • facebank/       │    │ • assignments      │    │ • ASR             │
└───────────────────┘    └────────────────────┘    └───────────────────┘
```

**Key patterns:**
- **SSE only** — No WebSockets; frontend uses native `EventSource` API
- **Presigned URLs** — Media served directly from S3, not proxied through API
- **Typed API client** — `web/api/client.ts` with OpenAPI-generated types

---

## 6. Components

### 6.1 Frontend (Vercel, Next.js)

| Aspect | Details |
|--------|---------|
| **Framework** | Next.js 14, App Router |
| **Location** | `web/` directory |
| **Deployment** | Git push → Vercel build → Deploy |
| **Domain** | `app.<screenalytics-domain>` or Vercel subdomain |

**Environment Variables (Vercel Dashboard):**

| Variable | Value | Description |
|----------|-------|-------------|
| `NEXT_PUBLIC_API_BASE` | `https://api.<screenalytics-domain>` | Backend API URL (required) |
| `NEXT_PUBLIC_WS_BASE` | (unused currently) | Reserved for future WebSocket |
| `NEXT_PUBLIC_MSW` | `0` | Disable mock service worker in production |

`NEXT_PUBLIC_API_BASE` must always point at the EC2 FastAPI domain (https://api.<domain>); the typed API client and hooks rely on it for every request.

**Deployment flow:**
1. Push to `main` branch (or PR for preview)
2. Vercel builds Next.js app
3. Vercel deploys to edge network
4. API requests rewritten to EC2 backend via `next.config.mjs` rewrites

---

### 6.2 Backend API (FastAPI on EC2)

| Aspect | Details |
|--------|---------|
| **Framework** | FastAPI + Uvicorn |
| **Location** | `apps/api/main.py` |
| **Internal Port** | 8000 |
| **External Port** | 443 (via nginx) |
| **Process Manager** | systemd service or Docker |

**Key routers:**
- `/episodes/*` — Episode CRUD, status, SSE events
- `/jobs/*` — Local subprocess job management
- `/celery_jobs/*` — Background Celery job management
- `/identities/*` — Cluster/identity management
- `/facebank/*` — Face seed management
- `/audio/*` — Audio pipeline endpoints (unchanged)

**systemd service example (`/etc/systemd/system/screenalytics-api.service`):**
```ini
[Unit]
Description=Screenalytics FastAPI
After=network.target redis.service

[Service]
User=screenalytics
WorkingDirectory=/opt/screenalytics
EnvironmentFile=/opt/screenalytics/.env
ExecStart=/opt/screenalytics/.venv/bin/uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

---

### 6.3 Workers (Celery)

| Aspect | Details |
|--------|---------|
| **App** | `apps.api.celery_app:celery_app` |
| **Broker** | Redis (localhost:6379) |
| **Default Concurrency** | 2 workers |
| **Time Limits** | 7200s hard, 6000s soft |

**Queues:**

| Queue | Purpose |
|-------|---------|
| `celery` | Default queue for grouping/ML tasks |
| `SCREENALYTICS_AUDIO_*` | Audio pipeline queues (10 total, unchanged) |

**Video/faces pipeline tasks (in default queue):**
- `run_detect_track_task` — Face detection + tracking
- `run_faces_embed_task` — Face embedding extraction
- `run_cluster_task` — Clustering
- `run_auto_group_task` — Auto-grouping
- `run_manual_assign_task` — Manual assignments
- `run_refresh_similarity_task` — Similarity refresh

**Audio pipeline tasks (existing, leave as-is):**
- `audio.ingest`, `audio.separate`, `audio.enhance`, `audio.diarize`
- `audio.voices`, `audio.transcribe`, `audio.align`, `audio.qc`, `audio.export`

**systemd service example (`/etc/systemd/system/screenalytics-worker.service`):**
```ini
[Unit]
Description=Screenalytics Celery Worker
After=network.target redis.service

[Service]
User=screenalytics
WorkingDirectory=/opt/screenalytics
EnvironmentFile=/opt/screenalytics/.env
ExecStart=/opt/screenalytics/.venv/bin/celery -A apps.api.celery_app:celery_app worker -l info --concurrency=2
Restart=always

[Install]
WantedBy=multi-user.target
```

---

### 6.4 Redis

| Aspect | Details |
|--------|---------|
| **Installation** | `apt install redis-server` or Docker |
| **Port** | 6379 (localhost only) |
| **Persistence** | RDB snapshots (default) |
| **Memory** | 256MB-1GB sufficient for job queue |

**Connection string:** `redis://localhost:6379/0`

**Key patterns:**
- `screenalytics:job_lock:{episode_id}:{operation}` — Prevents duplicate jobs
- `screenalytics:job_history:{user_id}` — Job history (sorted set)
- Celery task results stored with TTL

**Security:** Bind to 127.0.0.1 only; no public exposure needed.

---

### 6.5 Database

| Aspect | Details |
|--------|---------|
| **Engine** | PostgreSQL 15+ with pgvector |
| **Hosting** | AWS RDS, Neon, Supabase, or self-hosted |
| **Connection** | Via `DATABASE_URL` environment variable |

**Current state:** Episode metadata stored in JSON files (`data/meta/episodes.json`). Future work may migrate to Postgres for richer querying.

**If using existing RDS:** No migration needed; configure connection string in EC2 `.env`.

---

### 6.6 Storage (S3)

| Aspect | Details |
|--------|---------|
| **Bucket** | `screenalytics` (existing) |
| **Region** | `us-east-1` |
| **Access** | IAM role on EC2, or access keys |

**Layout (v2 format):**
```
raw/videos/{show}/s{season}/e{episode}/episode.mp4
artifacts/frames/{show}/s{season}/e{episode}/frames/
artifacts/crops/{show}/s{season}/e{episode}/tracks/
artifacts/manifests/{show}/s{season}/e{episode}/
artifacts/thumbs/{show}/s{season}/e{episode}/
facebank/{person_id}/{seed_id}_d.png
facebank/{person_id}/{seed_id}_e.png
```

**Presigned URLs:** 15-minute expiry for GET/POST operations.

---

## 7. Environments

### 7.1 Local Development

```bash
# Terminal 1: Next.js frontend
cd web && npm run dev
# Runs at http://localhost:3000, proxies /api to localhost:8000

# Terminal 2: FastAPI
source .venv/bin/activate
uvicorn apps.api.main:app --reload --port 8000

# Terminal 3: Celery worker
celery -A apps.api.celery_app:celery_app worker -l info

# Terminal 4: Redis (if not running)
redis-server
# Or: docker run -d -p 6379:6379 redis:7-alpine
```

**Local `.env` overrides:**
```bash
STORAGE_BACKEND=local  # or s3 with AWS credentials
REDIS_URL=redis://localhost:6379/0
API_BASE_URL=http://localhost:8000
```

### 7.2 Staging / Production

| Component | Staging | Production |
|-----------|---------|------------|
| **Frontend** | Vercel preview deployments | Vercel production |
| **API Domain** | `api-staging.<domain>` | `api.<domain>` |
| **EC2** | t3.medium (testing) | t3.xlarge or c6i.xlarge |
| **Redis** | On EC2 | On EC2 (or ElastiCache later) |
| **S3 Bucket** | `screenalytics-staging` | `screenalytics` |

**Environment variables on EC2 (`/opt/screenalytics/.env`):**
```bash
# API
API_BASE_URL=https://api.<domain>
UI_ORIGIN=https://app.<domain>

# Storage
STORAGE_BACKEND=s3
AWS_S3_BUCKET=screenalytics
AWS_DEFAULT_REGION=us-east-1

# Redis/Celery
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Database (if using RDS)
DATABASE_URL=postgresql://user:pass@rds-host:5432/screenalytics

# External APIs (audio pipeline)
RESEMBLE_API_KEY=xxx
OPENAI_API_KEY=xxx
```

---

## 8. Deployment Steps (High-Level)

### 8.1 EC2 Setup

1. **Provision EC2 instance**
   - AMI: Ubuntu 22.04 LTS
   - Instance type: t3.xlarge (4 vCPU, 16GB RAM) or c6i.xlarge
   - Storage: 100GB+ EBS (gp3)
   - Security group: 22 (SSH), 80, 443 inbound

2. **Install system dependencies**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y python3.11 python3.11-venv python3-pip \
       redis-server nginx certbot python3-certbot-nginx \
       git build-essential
   ```

3. **Create application user**
   ```bash
   sudo useradd -m -s /bin/bash screenalytics
   sudo mkdir -p /opt/screenalytics
   sudo chown screenalytics:screenalytics /opt/screenalytics
   ```

4. **Clone repository and set up virtualenv**
   ```bash
   sudo -u screenalytics git clone <repo-url> /opt/screenalytics
   cd /opt/screenalytics
   sudo -u screenalytics python3.11 -m venv .venv
   sudo -u screenalytics .venv/bin/pip install -r requirements.txt
   ```

5. **Configure environment**
   ```bash
   sudo -u screenalytics cp .env.example /opt/screenalytics/.env
   # Edit .env with production values
   ```

6. **Configure Redis**
   ```bash
   # Edit /etc/redis/redis.conf
   # bind 127.0.0.1
   # maxmemory 512mb
   # maxmemory-policy allkeys-lru
   sudo systemctl enable redis-server
   sudo systemctl start redis-server
   ```

7. **Install systemd services**
   ```bash
   sudo cp infra/systemd/screenalytics-api.service /etc/systemd/system/
   sudo cp infra/systemd/screenalytics-worker.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable screenalytics-api screenalytics-worker
   sudo systemctl start screenalytics-api screenalytics-worker
   ```

8. **Configure nginx + TLS**
   ```bash
   # Create /etc/nginx/sites-available/screenalytics-api
   sudo certbot --nginx -d api.<domain>
   sudo systemctl reload nginx
   ```
   Use `infra/nginx/screenalytics-api.conf` as the drop-in config (copy to `/etc/nginx/sites-available/` and symlink from `sites-enabled/`). Keep the SSE-friendly directives (`proxy_buffering off`, long `proxy_read_timeout`) intact; certbot will append the TLS server block.

   **nginx config example:**
   ```nginx
   server {
       server_name api.<domain>;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_http_version 1.1;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;

           # SSE support
           proxy_buffering off;
           proxy_cache off;
           proxy_read_timeout 86400;
       }
   }
   ```

### 8.2 Vercel Setup

1. **Create Vercel project**
   - Import from Git repository
   - Root directory: `web`
   - Framework preset: Next.js

2. **Configure environment variables**
   - `NEXT_PUBLIC_API_BASE` (required) = `https://api.<domain>` — must match the EC2 API domain
   - (Optional) `NEXT_PUBLIC_MSW` = `0` to keep mocks off in production

3. **Configure domain** (optional)
   - Add custom domain in Vercel dashboard
   - Update DNS records

### 8.3 Verification

#### TLS / HTTPS Verification
- `curl -v https://api.<domain>/healthz` — TLS handshake should succeed; response JSON includes `status`, `version`, `redis`, `storage`, `db`. Healthy = HTTP 200 with all values `ok` (db may be `unconfigured` when TRR_DB_URL is unset). Dependency failures return HTTP 503 with a `details` field describing the error.
- If handshake or cert issues occur, check nginx logs: `/var/log/nginx/access.log` and `/var/log/nginx/error.log`.

#### End-to-End Smoke Test
- **Frontend → API (local dev):** `NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev` in `web/`, open `/screenalytics/upload` or `/screenalytics/episodes` and confirm data loads.
- **Frontend → API (Vercel):** Set `NEXT_PUBLIC_API_BASE=https://api.<domain>` in Vercel env vars, visit the deployed site, ensure `/screenalytics/upload` loads and episode pages render status indicators (driven by the API client using `NEXT_PUBLIC_API_BASE`).
- **Backend checks:** `curl https://api.<domain>/healthz` → 200 JSON with `status: "ok"`; `curl -N https://api.<domain>/episodes/test-ep/events` → SSE stream/heartbeat.
- **Pipeline (non-audio):** Upload a tiny test episode, trigger `detect_track`, watch SSE updates for queue → running → complete, and confirm faces/episode status updates. Do not exercise audio pipelines in this phase.

---

## 9. Scaling & Future Work

### 9.1 When to Split Workers

**Trigger:** Worker queue depth consistently >10, or job latency >10 minutes for simple operations.

**Action:**
- Launch second EC2 for workers only
- Both connect to same Redis (need Redis accessible from second host)
- Consider moving Redis to ElastiCache at this point

### 9.2 When to Move Redis to ElastiCache

**Trigger:** Multi-host architecture, or Redis memory >2GB.

**Action:**
- Provision ElastiCache Redis cluster
- Update `REDIS_URL` on all hosts
- Remove local redis-server

### 9.3 When to Add GPU Workers

**Trigger:** Want faster detection/embedding, or processing >50 episodes/week.

**Action:**
- Launch GPU EC2 (g4dn.xlarge or similar)
- Configure Celery queue routing for GPU-accelerated tasks
- Keep CPU workers for non-ML tasks

### 9.4 Audio Pipeline Dedicated Worker

The audio pipeline has 10 dedicated queues. When audio processing volume increases:
- Launch dedicated EC2 for audio workers
- Configure worker to only consume `SCREENALYTICS_AUDIO_*` queues
- No code changes needed; just queue routing

---

## 10. References

- [Solution Architecture](../../../architecture/solution_architecture.md) — Overall system design
- [Render Deployment Guide](../../../ops/deployment/DEPLOYMENT_RENDER.md) — Alternative deployment option
- [Hardware Sizing](../../../ops/hardware_sizing.md) — Resource recommendations
- [.env.example](../../../../.env.example) — Environment variable template

---

**Maintained by:** Screenalytics Engineering
**Next Review:** After initial EC2 deployment
