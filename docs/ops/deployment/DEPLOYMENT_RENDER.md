# Render Deployment Guide

This guide covers deploying SCREANALYTICS to Render with the Celery background job system.

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Streamlit  │────▶│  FastAPI    │────▶│   Celery    │
│     UI      │     │    API      │     │   Worker    │
└─────────────┘     └──────┬──────┘     └──────┬──────┘
                           │                    │
                           └────────┬───────────┘
                                    ▼
                           ┌─────────────┐
                           │    Redis    │
                           │ (broker +   │
                           │  backend)   │
                           └─────────────┘
```

## Services

### 1. API (Web Service)

**Type:** Web Service
**Runtime:** Python 3
**Build Command:** `pip install -r requirements.txt`
**Start Command:** `uvicorn apps.api.main:app --host 0.0.0.0 --port $PORT`

### 2. Celery Worker (Background Worker)

**Type:** Background Worker
**Runtime:** Python 3
**Build Command:** `pip install -r requirements.txt -r requirements-ml.txt`
**Start Command:** `celery -A apps.api.celery_app:celery_app worker -l info --concurrency 2`

### 3. Redis (Key-Value Store)

**Type:** Redis
**Plan:** Free tier or Starter
**Use:** Celery broker and result backend

### 4. Streamlit UI (Optional separate service)

**Type:** Web Service
**Runtime:** Python 3
**Build Command:** `pip install -r requirements.txt`
**Start Command:** `streamlit run apps/workspace-ui/streamlit_app.py --server.address 0.0.0.0 --server.port $PORT`

## Environment Variables

Configure these for all services:

| Variable | Description | Example |
|----------|-------------|---------|
| `REDIS_URL` | Redis connection URL | `redis://red-xxx:6379` |
| `DATABASE_URL` | PostgreSQL connection (optional) | `postgresql://...` |
| `STORAGE_BACKEND` | Storage backend type | `s3` or `local` |
| `S3_BUCKET` | S3 bucket for artifacts | `screenalytics-prod` |
| `AWS_ACCESS_KEY_ID` | AWS credentials | |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials | |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `UI_ORIGIN` | CORS origin for Streamlit UI | `https://your-ui.onrender.com` |
| `API_BASE` | API URL for Streamlit (internal) | `https://your-api.onrender.com` |

## Render Configuration Files

### render.yaml (Blueprint)

```yaml
services:
  # API Service
  - type: web
    name: screanalytics-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn apps.api.main:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /healthz
    envVars:
      - key: REDIS_URL
        fromService:
          type: redis
          name: screanalytics-redis
          property: connectionString
      - key: STORAGE_BACKEND
        value: s3
      - key: S3_BUCKET
        sync: false
      - key: AWS_ACCESS_KEY_ID
        sync: false
      - key: AWS_SECRET_ACCESS_KEY
        sync: false
    autoDeploy: true

  # Celery Worker
  - type: worker
    name: screanalytics-worker
    runtime: python
    buildCommand: pip install -r requirements.txt -r requirements-ml.txt
    startCommand: celery -A apps.api.celery_app:celery_app worker -l info --concurrency 2
    envVars:
      - key: REDIS_URL
        fromService:
          type: redis
          name: screanalytics-redis
          property: connectionString
      - key: STORAGE_BACKEND
        value: s3
      - key: S3_BUCKET
        sync: false
      - key: AWS_ACCESS_KEY_ID
        sync: false
      - key: AWS_SECRET_ACCESS_KEY
        sync: false
    autoDeploy: true

  # Redis
  - type: redis
    name: screanalytics-redis
    plan: free
    maxmemoryPolicy: allkeys-lru
```

## Local Development

### Prerequisites

```bash
# Install Redis (macOS)
brew install redis

# Start Redis
brew services start redis

# Or run Redis in Docker
docker run -d -p 6379:6379 redis:7-alpine
```

### Running Services

```bash
# Terminal 1: API
uvicorn apps.api.main:app --reload --port 8000

# Terminal 2: Celery Worker
celery -A apps.api.celery_app:celery_app worker -l info

# Terminal 3: Streamlit UI
streamlit run apps/workspace-ui/streamlit_app.py
```

### Environment Variables for Local Dev

```bash
export REDIS_URL=redis://localhost:6379/0
export STORAGE_BACKEND=local
export API_BASE=http://localhost:8000
```

## Celery Task Configuration

Tasks are defined in `apps/api/tasks.py`:

- `run_manual_assign_task` - Batch cluster assignments
- `run_auto_group_task` - Auto-grouping clusters
- `run_reembed_task` - Re-embed faces (ML pipeline)
- `run_recluster_task` - Recluster faces (ML pipeline)
- `run_split_tracks_task` - Split tracks (ML pipeline)

### Task Timeouts

- **Soft limit:** 20 minutes (1200 seconds)
- **Hard limit:** 30 minutes (1800 seconds)

### Concurrency

Workers use Redis-based locking to prevent duplicate jobs per episode/operation.

## API Endpoints

### Async Job Submission

```
POST /episodes/{ep_id}/clusters/batch_assign_async
POST /episodes/{ep_id}/clusters/group_async
```

Returns HTTP 202 Accepted with:
```json
{
  "job_id": "abc123...",
  "status": "queued",
  "async": true
}
```

### Job Status

```
GET /celery_jobs/{job_id}
```

Returns:
```json
{
  "job_id": "abc123...",
  "state": "in_progress",  // queued, in_progress, success, failed, cancelled
  "raw_state": "STARTED",  // Celery state
  "progress": {
    "step": "group_within_episode",
    "progress": 0.7,
    "message": "Merged 5 cluster groups"
  },
  "result": null  // Populated when finished
}
```

### Cancel Job

```
POST /celery_jobs/{job_id}/cancel
```

## Monitoring

### Celery Flower (Optional)

```bash
pip install flower
celery -A apps.api.celery_app:celery_app flower --port=5555
```

### Redis CLI

```bash
redis-cli
> KEYS screenalytics:*
> GET screenalytics:job_lock:rhobh-s01e01:auto_group
```

## Troubleshooting

### Common Issues

1. **Jobs stuck in PENDING**
   - Check Redis connectivity
   - Verify Celery worker is running
   - Check for task import errors in worker logs

2. **409 Conflict: Job already in progress**
   - A job is already running for this episode/operation
   - Wait for completion or cancel via API

3. **Celery unavailable - falling back to sync**
   - Redis is not reachable
   - Endpoints will work but block until complete

### Logs

- **API logs:** Check Render dashboard or `uvicorn` output
- **Worker logs:** Check Render dashboard or `celery worker` output
- **Redis:** `redis-cli MONITOR`

## Scaling

### Horizontal Scaling

- **API:** Auto-scales based on traffic
- **Worker:** Manually scale or use multiple worker instances
- **Redis:** Upgrade plan for more memory/connections

### Worker Concurrency

Adjust `--concurrency` flag based on available memory:

```bash
# Low memory (512MB)
celery ... worker --concurrency 1

# Medium (1-2GB)
celery ... worker --concurrency 2

# High (4GB+)
celery ... worker --concurrency 4
```
