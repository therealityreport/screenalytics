#!/usr/bin/env bash
# ULTRA-STRICT detect/track: Extract embeddings EVERY FRAME for maximum accuracy
#
# Usage:
#   ./scripts/detect_track_strict.sh <ep_id>
#
# Note: The default config is already strict (emb every 5 frames).
# This script is ULTRA-STRICT (emb EVERY frame) for cases where you need
# absolute maximum accuracy and can tolerate 2-3x slower processing.
#
# Ultra-strict vs Default:
# - Embedding extraction: EVERY frame (default: every 5 frames)
# - All other settings same as default (0.65 det thresh, 0.75/0.82 appear gates, etc.)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <ep_id>" >&2
    exit 1
fi

EP_ID="$1"

# Force embedding extraction every frame (only difference from default)
export TRACK_GATE_EMB_EVERY=1        # Extract embeddings EVERY frame (default is 5)

# Run detect/track with default config (which is already strict)
python3 tools/episode_run.py \
    --ep-id "$EP_ID" \
    detect track

echo ""
echo "✅ ULTRA-STRICT detect/track completed for ${EP_ID}"
echo ""
echo "Settings used:"
echo "  Embedding extraction: EVERY frame (vs default every 5 frames)"
echo "  Detection threshold: 0.65"
echo "  Appearance hard split: < 75% similarity"
echo "  Appearance soft split: < 82% similarity"
echo "  Track buffer: 15 frames (0.5s at 30fps)"
echo "  ByteTrack IoU: 0.85"
echo ""
echo "Note: This is 2-3x slower than default due to embedding every frame."
echo "      Use default mode for most cases (already strict enough)."
