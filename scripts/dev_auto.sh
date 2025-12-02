#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
API_LOG="$ROOT/api_server.log"
CELERY_LOG="$ROOT/celery_worker.log"

# ============================================================================
# Detect VS Code and offer to use VS Code Tasks instead
# ============================================================================
if [[ "${TERM_PROGRAM:-}" == "vscode" ]]; then
    echo ""
    echo "=============================================="
    echo "  Running inside VS Code Terminal"
    echo "=============================================="
    echo ""
    echo "For the best experience with separate terminal panels,"
    echo "use VS Code Tasks instead:"
    echo ""
    echo "  1. Press Cmd+Shift+P (or Ctrl+Shift+P)"
    echo "  2. Type 'Tasks: Run Task'"
    echo "  3. Select 'Dev: Start All Services'"
    echo ""
    echo "This will open Redis, Celery, API, and Streamlit"
    echo "in separate terminal panels within VS Code."
    echo ""
    echo "To kill all services: Run task 'Dev: Kill All Services'"
    echo ""
    read -p "Continue with background mode anyway? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Use VS Code Tasks for the best experience."
        exit 0
    fi
    # Force background mode when in VS Code
    OPEN_IN_TERMINALS="false"
    echo ""
    echo "Running in background mode. Logs will be written to files."
    echo "Use 'tail -f api_server.log' or 'tail -f celery_worker.log' to view."
    echo ""
fi

# Option: Open Redis and Celery in separate Terminal windows (macOS only)
# Set to "true" to open new windows, "false" to run in background with log files
OPEN_IN_TERMINALS="${OPEN_IN_TERMINALS:-true}"

# Initialize PIDs to empty (prevents unbound variable errors on early exit)
API_PID=""
CELERY_PID=""
STREAMLIT_PID=""

# Cleanup function to kill all background processes and close Terminal windows on exit
cleanup() {
    echo ""
    echo "[dev_auto] Shutting down services..."

    # Kill background processes
    [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null || true
    [ -n "$CELERY_PID" ] && kill "$CELERY_PID" 2>/dev/null || true
    [ -n "$STREAMLIT_PID" ] && kill "$STREAMLIT_PID" 2>/dev/null || true

    # Kill any remaining processes
    pkill -f "uvicorn apps.api.main:app" 2>/dev/null || true
    pkill -f "streamlit run" 2>/dev/null || true
    pkill -f "celery.*apps.api.celery_app" 2>/dev/null || true
    pkill -f "redis-server" 2>/dev/null || true

    # Close Terminal windows opened by this script (macOS)
    if [[ "$OSTYPE" == "darwin"* ]] && command -v osascript >/dev/null 2>&1; then
        echo "[dev_auto] Closing Terminal windows..."
        osascript <<'CLEANUP_EOF' 2>/dev/null || true
tell application "Terminal"
    set windowList to every window
    repeat with w in windowList
        try
            set winName to name of w
            if winName contains "Redis Server" or winName contains "Celery Worker" then
                close w
            end if
        end try
    end repeat
end tell
CLEANUP_EOF
    fi

    echo "[dev_auto] Done."
}
trap cleanup EXIT

# ============================================================================
# Helper: Close Terminal windows by title (macOS)
# ============================================================================
close_terminal_by_title() {
    local title="$1"
    if [[ "$OSTYPE" == "darwin"* ]] && command -v osascript >/dev/null 2>&1; then
        osascript <<EOF 2>/dev/null || true
tell application "Terminal"
    set windowList to every window
    repeat with w in windowList
        try
            if name of w contains "$title" then
                close w
            end if
        end try
    end repeat
end tell
EOF
    fi
}

# ============================================================================
# Kill existing processes and close Terminal windows
# ============================================================================
echo "[dev_auto] Closing previous Terminal windows..."
close_terminal_by_title "Redis Server"
close_terminal_by_title "Celery Worker"

echo "[dev_auto] Killing existing processes..."
pkill -f "uvicorn apps.api.main:app" 2>/dev/null || true
pkill -f "streamlit run" 2>/dev/null || true
pkill -f "celery.*apps.api.celery_app" 2>/dev/null || true
pkill -f "redis-server" 2>/dev/null || true
sleep 1

# ============================================================================
# Helper: Open command in new Terminal window (macOS)
# ============================================================================
open_in_terminal() {
    local title="$1"
    local cmd="$2"

    if [[ "$OSTYPE" == "darwin"* ]] && command -v osascript >/dev/null 2>&1; then
        osascript <<EOF
tell application "Terminal"
    activate
    set newTab to do script "cd '$ROOT' && echo '=== $title ===' && $cmd"
    set custom title of front window to "$title"
end tell
EOF
        return 0
    else
        return 1
    fi
}

# ============================================================================
# Redis Setup (Phase 2 - Background Jobs)
# ============================================================================
echo "[dev_auto] Checking Redis..."
if command -v redis-cli >/dev/null 2>&1; then
    if redis-cli ping >/dev/null 2>&1; then
        echo "[dev_auto] Redis is already running"
    else
        if [[ "$OPEN_IN_TERMINALS" == "true" ]]; then
            echo "[dev_auto] Starting Redis in new Terminal window..."
            if open_in_terminal "Redis Server" "redis-server"; then
                sleep 2
                if redis-cli ping >/dev/null 2>&1; then
                    echo "[dev_auto] Redis started successfully (in separate Terminal)"
                else
                    echo "[dev_auto] WARNING: Redis may still be starting. Check the Redis terminal."
                fi
            else
                echo "[dev_auto] Could not open Terminal, starting Redis via brew..."
                brew services start redis 2>/dev/null || true
            fi
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
    fi
else
    echo "[dev_auto] WARNING: redis-cli not found. Install Redis for background job support."
    echo "[dev_auto] Run: brew install redis && brew services start redis"
fi

# ============================================================================
# Celery Worker (Phase 2 - Background Jobs)
# Use explicit queues so audio tasks are consumed; allow override via env
# ============================================================================
CELERY_QUEUES="${CELERY_QUEUES:-SCREENALYTICS_AUDIO_PIPELINE,SCREENALYTICS_AUDIO_INGEST,SCREENALYTICS_AUDIO_SEPARATE,SCREENALYTICS_AUDIO_ENHANCE,SCREENALYTICS_AUDIO_DIARIZE,SCREENALYTICS_AUDIO_TRANSCRIBE,SCREENALYTICS_AUDIO_VOICES,SCREENALYTICS_AUDIO_ALIGN,SCREENALYTICS_AUDIO_QC,SCREENALYTICS_AUDIO_EXPORT,celery}"
CELERY_CONCURRENCY="${CELERY_CONCURRENCY:-2}"

# Use unique hostname to avoid Celery naming collisions with stale Redis entries
CELERY_HOSTNAME="celery_audio@%h"

if [[ "$OPEN_IN_TERMINALS" == "true" ]]; then
    echo "[dev_auto] Starting Celery worker in new Terminal window..."
    CELERY_CMD="source '$ROOT/.venv/bin/activate' && python -m celery -A apps.api.celery_app:celery_app worker --loglevel=info --concurrency=$CELERY_CONCURRENCY --queues=$CELERY_QUEUES --hostname=$CELERY_HOSTNAME"
    if open_in_terminal "Celery Worker" "$CELERY_CMD"; then
        echo "[dev_auto] Celery worker started (in separate Terminal)"
        # We don't have a PID for the external terminal process
        CELERY_PID=""
    else
        echo "[dev_auto] Could not open Terminal, starting Celery in background..."
        "$PYTHON" -m celery -A apps.api.celery_app:celery_app worker \
            --loglevel=info \
            --concurrency="$CELERY_CONCURRENCY" \
            --queues="$CELERY_QUEUES" \
            --hostname="$CELERY_HOSTNAME" \
            >> "$CELERY_LOG" 2>&1 &
        CELERY_PID=$!
        echo "[dev_auto] Celery worker started (PID: $CELERY_PID, log: $CELERY_LOG)"
    fi
else
    echo "[dev_auto] Starting Celery worker..."
    "$PYTHON" -m celery -A apps.api.celery_app:celery_app worker \
        --loglevel=info \
        --concurrency="$CELERY_CONCURRENCY" \
        --queues="$CELERY_QUEUES" \
        --hostname="$CELERY_HOSTNAME" \
        >> "$CELERY_LOG" 2>&1 &
    CELERY_PID=$!
    echo "[dev_auto] Celery worker started (PID: $CELERY_PID, log: $CELERY_LOG)"
fi

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
# Streamlit UI (background)
# ============================================================================
echo "[dev_auto] Starting Streamlit UI (auto-rerun)..."
"$PYTHON" -m streamlit run apps/workspace-ui/streamlit_app.py \
    --server.port 8505 \
    --server.address 127.0.0.1 &
STREAMLIT_PID=$!
echo "[dev_auto] Streamlit started (PID: $STREAMLIT_PID)"

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

# Wait for Ctrl+C - don't exit if a single service crashes
while true; do
    sleep 5
    # Optional: check if services are still running and report
    if [ -n "$API_PID" ] && ! kill -0 "$API_PID" 2>/dev/null; then
        echo "[dev_auto] WARNING: API died. Check $API_LOG"
        API_PID=""
    fi
    if [ -n "$STREAMLIT_PID" ] && ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        echo "[dev_auto] WARNING: Streamlit died. Restarting..."
        "$PYTHON" -m streamlit run apps/workspace-ui/streamlit_app.py \
            --server.port 8505 \
            --server.address 127.0.0.1 &
        STREAMLIT_PID=$!
    fi
done
