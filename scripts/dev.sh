#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export STORAGE_BACKEND="${STORAGE_BACKEND:-local}"
export API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
export UI_ORIGIN="${UI_ORIGIN:-http://localhost:8501}"
python -m uvicorn apps.api.main:app --port 8000 & API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT
for _ in {1..50}; do
  curl -fsS "${API_BASE_URL%/}/healthz" >/dev/null && break
  sleep 0.2
done
streamlit run apps/workspace-ui/streamlit_app.py
