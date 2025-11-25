#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
API_LOG="$ROOT/api_server.log"
CELERY_LOG="$ROOT/celery_worker.log"

# Cleanup function to kill all background processes on exit
cleanup() {
    echo ""
    echo "[dev_auto] Shutting down services..."
    kill "$API_PID" 2>/dev/null || true
    kill "$CELERY_PID" 2>/dev/null || true
    echo "[dev_auto] Done."
}
trap cleanup EXIT

# Kill existing processes first
echo "[dev_auto] Killing existing processes..."
pkill -f "uvicorn apps.api.main:app" 2>/dev/null || true
pkill -f "streamlit run" 2>/dev/null || true
pkill -f "celery.*apps.api.celery_app" 2>/dev/null || true
sleep 1

# ============================================================================
# Redis Setup (Phase 2 - Background Jobs)
# ============================================================================
echo "[dev_auto] Checking Redis..."
if command -v redis-cli >/dev/null 2>&1; then
    if redis-cli ping >/dev/null 2>&1; then
        echo "[dev_auto] Redis is already running"
    else
        echo "[dev_auto] Starting Redis via brew..."
        if command -v brew >/dev/null 2>&1; then
            brew services start redis 2>/dev/null || true
            sleep 2
            if redis-cli ping >/dev/null 2>&1; then
                echo "[dev_auto] Redis started successfully"
            else
                echo "[dev_auto] WARNING: Redis failed to start. Celery jobs will fall back to sync mode."
            fi
        else
            echo "[dev_auto] WARNING: brew not found. Install Redis manually or jobs will run synchronously."
        fi
    fi
else
    echo "[dev_auto] WARNING: redis-cli not found. Install Redis for background job support."
    echo "[dev_auto] Run: brew install redis && brew services start redis"
fi

# ============================================================================
# Celery Worker (Phase 2 - Background Jobs)
# ============================================================================
echo "[dev_auto] Starting Celery worker..."
"$PYTHON" -m celery -A apps.api.celery_app:celery_app worker \
    --loglevel=info \
    --concurrency=2 \
    >> "$CELERY_LOG" 2>&1 &
CELERY_PID=$!
echo "[dev_auto] Celery worker started (PID: $CELERY_PID, log: $CELERY_LOG)"

# ============================================================================
# FastAPI Server
# ============================================================================
echo "[dev_auto] Starting API (auto-reload)..."
"$PYTHON" -m uvicorn apps.api.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --reload \
    --reload-dir apps \
    --reload-dir tools \
    --reload-dir config \
    --reload-exclude data \
    --reload-exclude .venv \
    --log-level info >> "$API_LOG" 2>&1 &
API_PID=$!
echo "[dev_auto] API started (PID: $API_PID, log: $API_LOG)"

# ============================================================================
# Open Browser
# ============================================================================
BROWSER_URL="http://127.0.0.1:8505"
if command -v open >/dev/null 2>&1; then
    (sleep 3; open "$BROWSER_URL") &
elif command -v xdg-open >/dev/null 2>&1; then
    (sleep 3; xdg-open "$BROWSER_URL") &
fi

# ============================================================================
# Streamlit UI (foreground - blocking)
# ============================================================================
echo "[dev_auto] Starting Streamlit UI (auto-rerun)..."
echo ""
echo "=============================================="
echo "  SCREENALYTICS Dev Environment"
echo "=============================================="
echo "  API:      http://127.0.0.1:8000"
echo "  UI:       http://127.0.0.1:8505"
echo "  API Log:  $API_LOG"
echo "  Celery:   $CELERY_LOG"
echo "=============================================="
echo "  Press Ctrl+C to stop all services"
echo "=============================================="
echo ""

"$PYTHON" -m streamlit run apps/workspace-ui/streamlit_app.py \
    --server.port 8505 \
    --server.address 127.0.0.1 || true
