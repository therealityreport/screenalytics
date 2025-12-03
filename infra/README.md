# Screenalytics Infrastructure

This directory contains infrastructure configuration templates for deploying Screenalytics on EC2.

These files implement the architecture described in [`docs/infra/screenalytics_deploy_plan.md`](../docs/infra/screenalytics_deploy_plan.md).

## Directory Structure

```
infra/
├── README.md                           # This file
├── systemd/
│   ├── screenalytics-api.service       # FastAPI API server unit
│   └── screenalytics-worker.service    # Celery worker unit
└── nginx/
    └── screenalytics-api.conf          # Reverse proxy with SSE support
```

## Quick Start

### 1. Prerequisites

On your EC2 instance (Ubuntu 22.04 recommended):

```bash
# Install system packages
sudo apt update
sudo apt install -y python3.11 python3.11-venv redis-server nginx certbot python3-certbot-nginx

# Create application user
sudo useradd -m -s /bin/bash screenalytics
sudo mkdir -p /opt/screenalytics
sudo chown screenalytics:screenalytics /opt/screenalytics

# Clone repo and setup virtualenv
sudo -u screenalytics git clone <repo-url> /opt/screenalytics
cd /opt/screenalytics
sudo -u screenalytics python3.11 -m venv .venv
sudo -u screenalytics .venv/bin/pip install -r requirements.txt

# Configure environment
sudo -u screenalytics cp .env.example /opt/screenalytics/.env
# Edit .env with production values
```

### 2. Install systemd Services

```bash
# Copy service files
sudo cp infra/systemd/screenalytics-api.service /etc/systemd/system/
sudo cp infra/systemd/screenalytics-worker.service /etc/systemd/system/

# Reload systemd and enable services
sudo systemctl daemon-reload
sudo systemctl enable screenalytics-api screenalytics-worker
sudo systemctl start screenalytics-api screenalytics-worker

# Check status
sudo systemctl status screenalytics-api
sudo systemctl status screenalytics-worker
```

### 3. Configure nginx

```bash
# Copy nginx config
sudo cp infra/nginx/screenalytics-api.conf /etc/nginx/sites-available/

# Edit to replace api.example.com with your domain
sudo nano /etc/nginx/sites-available/screenalytics-api.conf

# Enable the site
sudo ln -s /etc/nginx/sites-available/screenalytics-api.conf /etc/nginx/sites-enabled/

# Test and reload
sudo nginx -t
sudo systemctl reload nginx

# Add TLS with certbot
sudo certbot --nginx -d api.yourdomain.com
```

`infra/nginx/screenalytics-api.conf` is a drop-in for `/etc/nginx/sites-available/`; symlink it from `sites-enabled/` and let certbot append the TLS server block or companion file. Keep the SSE-friendly directives (`proxy_buffering off`, long `proxy_read_timeout`, etc.) unchanged so job event streams stay responsive.

### 4. Verify

```bash
# Check API health
curl http://localhost:8000/healthz

# After TLS setup
curl https://api.yourdomain.com/healthz
```

## Component Details

### systemd/screenalytics-api.service

Runs the FastAPI backend:
- **Binds to:** `127.0.0.1:8000` (localhost only; nginx handles external traffic)
- **Workers:** 1 (uvicorn worker; use nginx for load balancing if needed)
- **Environment:** Loaded from `/opt/screenalytics/.env`
- **User:** `screenalytics`

### systemd/screenalytics-worker.service

Runs the Celery worker for background jobs:
- **Queues:** Default queue (`celery`) for detect_track, faces, cluster, grouping tasks
- **Concurrency:** 2 (conservative for t3.xlarge; increase for larger instances)
- **Broker:** Redis (configured via `CELERY_BROKER_URL` in `.env`)
- **Shutdown:** Graceful with 5-minute timeout for in-progress tasks

### nginx/screenalytics-api.conf

Reverse proxy configuration:
- **Upstream:** `127.0.0.1:8000`
- **SSE Support:** `proxy_buffering off`, `proxy_cache off`, long timeouts
- **TLS:** Managed by certbot (run after copying config)
- **Endpoints:** Special handling for `/episodes/*/events` and `/celery_jobs/stream/*`

## Important Notes

### What's In Scope

This infrastructure supports:
- FastAPI backend (`apps/api/`)
- Celery workers for video/faces pipeline (detect, track, embed, cluster)
- Redis as Celery broker
- S3 for artifact storage
- SSE for real-time progress updates

### What's Out of Scope

**Audio pipeline components are not modified by this infrastructure setup.**

The audio pipeline (`py_screenalytics/audio/`, `apps/api/jobs_audio.py`, `apps/api/routers/audio.py`) has dedicated Celery queues (`SCREENALYTICS_AUDIO_*`) that will run on the same worker. No changes are made to audio code, configs, or queue routing in this phase.

### Scaling

When ready to scale:
1. **Split workers:** Launch a second EC2 for workers only
2. **Move Redis:** Migrate to ElastiCache for multi-host access
3. **Add GPU:** Launch GPU instance for accelerated ML tasks
4. **Audio workers:** Dedicate separate workers to `SCREENALYTICS_AUDIO_*` queues

See `docs/infra/screenalytics_deploy_plan.md` Section 9 for detailed scaling triggers and actions.

## Troubleshooting

### Services won't start

```bash
# Check logs
sudo journalctl -u screenalytics-api -f
sudo journalctl -u screenalytics-worker -f

# Common issues:
# - Missing .env file
# - Redis not running
# - Virtualenv not created
# - Wrong file permissions
```

### nginx returns 502

```bash
# Check if API is running
curl http://127.0.0.1:8000/healthz

# Check nginx error log
sudo tail -f /var/log/nginx/screenalytics-api.error.log
```

### SSE not streaming

Ensure nginx config has:
- `proxy_buffering off;`
- `proxy_cache off;`
- Long `proxy_read_timeout`

## Related Documentation

- [Infrastructure Deploy Plan](../docs/infra/screenalytics_deploy_plan.md) — Full deployment blueprint
- [Solution Architecture](../docs/architecture/solution_architecture.md) — System design overview
- [.env.example](../.env.example) — Environment variable reference
