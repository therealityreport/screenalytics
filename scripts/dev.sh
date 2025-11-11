#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${STORAGE_BACKEND:=s3}"
: "${AWS_S3_BUCKET:=screenalytics}"
: "${AWS_DEFAULT_REGION:=us-east-1}"
export STORAGE_BACKEND
export AWS_S3_BUCKET
export AWS_DEFAULT_REGION
export API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
export UI_ORIGIN="${UI_ORIGIN:-http://localhost:8501}"

if [[ "$STORAGE_BACKEND" == "s3" && -z "$AWS_S3_BUCKET" ]]; then
  echo "AWS_S3_BUCKET must be set when STORAGE_BACKEND=s3" >&2
  exit 1
fi

echo "API STORAGE_BACKEND=${STORAGE_BACKEND}  BUCKET=${AWS_S3_BUCKET:-local}"

python -m uvicorn apps.api.main:app --port 8000 & API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

HEALTH_URL="${API_BASE_URL%/}/healthz"
echo "Waiting for ${HEALTH_URL} …"
for _ in {1..50}; do
  if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    echo "API ready → ${API_BASE_URL%/}"
    break
  fi
  sleep 0.2
done

streamlit run apps/workspace-ui/streamlit_app.py
