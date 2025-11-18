#!/usr/bin/env bash
# Set environment variables for strict tracking configuration
# Source this file before running episode_run.py: source scripts/set_strict_tracking.sh

# ByteTrack spatial matching - strict mode
export SCREENALYTICS_TRACK_BUFFER=25
export BYTE_TRACK_BUFFER=25
export BYTE_TRACK_MATCH_THRESH=0.90
export SCREANALYTICS_TRACK_HIGH_THRESH=0.65
export BYTE_TRACK_HIGH_THRESH=0.65
export SCREANALYTICS_NEW_TRACK_THRESH=0.65
export BYTE_TRACK_NEW_TRACK_THRESH=0.65

# Appearance gate - strict thresholds
export TRACK_GATE_APPEAR_HARD=0.75
export TRACK_GATE_APPEAR_SOFT=0.82
export TRACK_GATE_APPEAR_STREAK=2
export TRACK_GATE_IOU=0.50
export TRACK_GATE_PROTO_MOM=0.90
export TRACK_GATE_EMB_EVERY=10  # Reduced from 5 for better thermal performance

echo "✓ Strict tracking configuration loaded:"
echo "  - track_buffer: $SCREANALYTICS_TRACK_BUFFER (was 15, prevents fragmentation)"
echo "  - match_thresh: $BYTE_TRACK_MATCH_THRESH (was 0.85, prevents jumping)"
echo "  - track_thresh: $SCREANALYTICS_TRACK_HIGH_THRESH (was 0.5)"
echo "  - gate_iou: $TRACK_GATE_IOU (was 0.40, tighter spatial constraint)"
echo "  - gate_appear_hard: $TRACK_GATE_APPEAR_HARD"
echo "  - gate_appear_soft: $TRACK_GATE_APPEAR_SOFT"
