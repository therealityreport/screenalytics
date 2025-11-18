#!/usr/bin/env bash
# Strict detect/track configuration for single-person tracks
#
# Usage:
#   ./scripts/detect_track_strict.sh <ep_id>
#
# Changes from default:
# - Higher detection threshold (0.65 vs 0.5) - filters low-quality detections
# - Stricter appearance gate (0.75 hard, 0.82 soft vs 0.60/0.70)
# - Extract embeddings every frame (--gate-emb-every 1)
# - Shorter track buffer (15 vs 30 frames)
# - Higher ByteTrack IoU matching (0.85 vs 0.8)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <ep_id>" >&2
    exit 1
fi

EP_ID="$1"

# Set appearance gate thresholds (env vars)
export TRACK_GATE_APPEAR_HARD=0.75   # Hard split if similarity < 75% (was 60%)
export TRACK_GATE_APPEAR_SOFT=0.82   # Soft split if similarity < 82% (was 70%)
export TRACK_GATE_APPEAR_STREAK=2    # Split after 2 consecutive low-sim frames
export TRACK_GATE_IOU=0.40           # Split if spatial jump (IoU < 40%)
export TRACK_GATE_EMB_EVERY=1        # Extract embeddings EVERY frame (critical!)

# Run detect/track with strict config
python3 tools/episode_run.py \
    --ep-id "$EP_ID" \
    --config config/pipeline/tracking-strict.yaml \
    --det-thresh 0.65 \
    --gate-emb-every 1 \
    detect track

echo ""
echo "✅ Strict detect/track completed for ${EP_ID}"
echo ""
echo "Settings used:"
echo "  Detection threshold: 0.65 (filters low-quality detections)"
echo "  Appearance hard split: < 75% similarity"
echo "  Appearance soft split: < 82% similarity"
echo "  Embedding extraction: every frame (ensures gate can reject mismatches)"
echo "  Track buffer: 15 frames (0.5s at 30fps)"
echo "  ByteTrack IoU: 0.85 (tighter spatial matching)"
