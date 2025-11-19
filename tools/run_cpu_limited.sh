#!/usr/bin/env bash
# CPU-limited wrapper for episode_run.py
# Ensures SCREENALYTICS never uses more than 300% CPU (3 cores)

set -euo pipefail

MAX_CPU="${CPU_LIMIT:-300}"  # Default 300%, override with CPU_LIMIT env var

if ! command -v cpulimit &>/dev/null; then
    echo "⚠️  WARNING: cpulimit not found - running WITHOUT CPU cap"
    echo "   CPU usage may exceed ${MAX_CPU}%"
    echo "   Install with: brew install cpulimit (macOS) or apt install cpulimit (Linux)"
    echo ""
    exec "$@"
fi

echo "⚙️  Running with CPU limit: ${MAX_CPU}% ($(echo "scale=1; $MAX_CPU / 100" | bc) cores)"
echo "   Command: $*"
echo ""

# Run with cpulimit
# -l: CPU limit percentage (300 = 3 cores)
# --: Everything after is the command to run
exec cpulimit -l "$MAX_CPU" -- "$@"
