#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
API_LOG="$ROOT/api_server.log"

# Kill existing processes first
echo "[dev_auto] Killing existing API and Streamlit processes..."
pkill -f "uvicorn apps.api.main:app" 2>/dev/null || true
pkill -f "streamlit run" 2>/dev/null || true
sleep 1

echo "[dev_auto] Starting API (auto-reload)"
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

echo "[dev_auto] Starting Streamlit UI (auto-rerun)"
"$PYTHON" -m streamlit run apps/workspace-ui/streamlit_app.py \
  --server.port 8505 \
  --server.address 127.0.0.1 || true

echo "[dev_auto] Stopping API"
kill "$API_PID" 2>/dev/null || true
