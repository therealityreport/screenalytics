#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${STORAGE_BACKEND:=s3}"
: "${AWS_S3_BUCKET:=screenalytics}"
: "${AWS_DEFAULT_REGION:=us-east-1}"
: "${FACEBANK_S3_BUCKET:=$AWS_S3_BUCKET}"
: "${AWS_PROFILE:=default}"
export STORAGE_BACKEND
export AWS_S3_BUCKET
export AWS_DEFAULT_REGION
export FACEBANK_S3_BUCKET
export AWS_PROFILE
export API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
export UI_ORIGIN="${UI_ORIGIN:-http://localhost:8501}"
# Default CPU cap for local services; set SCREENALYTICS_CPULIMIT_PERCENT=0 to disable
CPULIMIT_PERCENT=${SCREENALYTICS_CPULIMIT_PERCENT:-250}
if command -v cpulimit >/dev/null 2>&1 && [[ "${CPULIMIT_PERCENT}" =~ ^[0-9]+$ ]] && [[ $CPULIMIT_PERCENT -gt 0 ]]; then
  CPULIMIT_PREFIX=(cpulimit -l "$CPULIMIT_PERCENT" -i --)
else
  CPULIMIT_PREFIX=()
fi
# CPU thread limits are now managed by apps/common/cpu_limits.py
# Override the default (3 threads = ~300% CPU) by setting SCREANALYTICS_MAX_CPU_THREADS
# Example: export SCREENALYTICS_MAX_CPU_THREADS=5  # for ~500% CPU usage
export SCREENALYTICS_MAX_CPU_THREADS="${SCREANALYTICS_MAX_CPU_THREADS:-3}"

# =============================================================================
# macOS Low-Noise Profile (Apple Silicon)
# =============================================================================
# These environment variables optimize MPS (Metal Performance Shaders) for
# quiet laptop operation. They're only effective on macOS but harmless elsewhere.
#
# See: apps/common/macos_profile.py for full documentation
# =============================================================================

# Mark this as a dev environment for auto-profile activation
export SCREENALYTICS_ENV="${SCREENALYTICS_ENV:-dev}"

# MPS memory management - prevent unified memory pressure
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.6}"
export PYTORCH_MPS_BLOCK_SIZE="${PYTORCH_MPS_BLOCK_SIZE:-262144}"
export PYTORCH_MPS_ALLOCATOR_MAX_SHARE="${PYTORCH_MPS_ALLOCATOR_MAX_SHARE:-0.8}"
export PYTORCH_MPS_LOGS="${PYTORCH_MPS_LOGS:-0}"

# CoreML/ONNX Runtime settings
export ORT_USE_COREML="${ORT_USE_COREML:-1}"

if [[ "$STORAGE_BACKEND" == "s3" && -z "$AWS_S3_BUCKET" ]]; then
  echo "AWS_S3_BUCKET must be set when STORAGE_BACKEND=s3" >&2
  exit 1
fi

echo "[dev.sh] STORAGE_BACKEND=${STORAGE_BACKEND}  BUCKET=${AWS_S3_BUCKET:-local}  API=${API_BASE_URL}"

# Activate virtual environment
source .venv/bin/activate

# Verify venv is active
if [[ "$VIRTUAL_ENV" != *"/SCREENALYTICS/.venv" ]]; then
  echo "[dev.sh] ERROR: Virtual environment not activated correctly" >&2
  echo "[dev.sh] VIRTUAL_ENV=$VIRTUAL_ENV" >&2
  exit 1
fi

# Check and install requirements if needed
echo "[dev.sh] Checking Python dependencies..."
REQUIREMENTS_HASH="${ROOT}/.requirements_hash"
CURRENT_HASH=$(cat requirements-ml.txt requirements-core.txt 2>/dev/null | shasum -a 256 | cut -d' ' -f1)

if [[ ! -f "$REQUIREMENTS_HASH" ]] || [[ "$(cat "$REQUIREMENTS_HASH" 2>/dev/null)" != "$CURRENT_HASH" ]]; then
  echo "[dev.sh] Installing/updating requirements..."
  pip install -q -r requirements.txt || {
    echo "[dev.sh] WARNING: Some packages failed to install. Continuing anyway..." >&2
  }
  echo "$CURRENT_HASH" > "$REQUIREMENTS_HASH"
  echo "[dev.sh] ✅ Requirements up to date"
else
  echo "[dev.sh] ✅ Requirements already satisfied"
fi


# Start API server with output captured to log file
API_LOG="${ROOT}/api_server.log"
echo "[dev.sh] Starting API server (logs: ${API_LOG})"
echo "[dev.sh] Using Python: $(which python)"
"${CPULIMIT_PREFIX[@]}" python -m uvicorn apps.api.main:app --port 8000 > "$API_LOG" 2>&1 & API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

HEALTH_URL="${API_BASE_URL%/}/healthz"
echo "Waiting for ${HEALTH_URL} (this may take 30-60s for ML models to load)…"
health_ok=false
for i in {1..150}; do
  if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    echo "API ready → ${API_BASE_URL%/}"
    health_ok=true
    break
  fi
  # Show progress every 5 seconds
  if (( i % 25 == 0 )); then
    echo "  Still waiting... ($((i / 5))s elapsed)"
  fi
  sleep 0.2
done
if [[ "$health_ok" != "true" ]]; then
  echo "[dev.sh] ERROR: API health check failed after 30 seconds" >&2
  echo "[dev.sh] API server may still be loading models or crashed during startup" >&2
  echo "[dev.sh] Last 100 lines of API log:" >&2
  tail -n 100 "$API_LOG" >&2
  echo "" >&2
  echo "[dev.sh] The API server is still running in the background (PID $API_PID)" >&2
  echo "[dev.sh] You can:" >&2
  echo "  1. Wait longer and check manually: curl $HEALTH_URL" >&2
  echo "  2. Kill it: kill $API_PID" >&2
  echo "  3. Continue anyway - Streamlit will start but API calls may fail" >&2
  read -p "Continue to Streamlit? [y/N] " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting. Kill API server with: kill $API_PID"
    exit 1
  fi
fi

"${CPULIMIT_PREFIX[@]}" streamlit run apps/workspace-ui/Upload_Video.py
