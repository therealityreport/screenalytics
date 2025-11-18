#!/usr/bin/env bash
# Quick reference: How to run tracking with different configurations
# Location: /Volumes/HardDrive/SCREENALYTICS/scripts/tracking_quick_reference.sh

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════════════╗
║                     TRACKING CONFIGURATION QUICK REFERENCE                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. DEFAULT (Strict) - Recommended for most use cases                         │
└──────────────────────────────────────────────────────────────────────────────┘

  Strict settings are now the default. Just run episode_run.py normally:

    python tools/episode_run.py --ep-id rhobh-s05e01

  Settings:
    • track_buffer: 15 frames
    • match_thresh: 0.85 (IoU)
    • track_thresh: 0.65
    • gate_appear_hard: 0.75
    • gate_appear_soft: 0.82
    • emb_every: 5 frames

┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. ULTRA-STRICT - Maximum accuracy (2-3x slower)                             │
└──────────────────────────────────────────────────────────────────────────────┘

  Extracts embeddings EVERY frame instead of every 5 frames:

    ./scripts/detect_track_strict.sh rhobh-s05e01

  Settings:
    • All strict defaults PLUS
    • emb_every: 1 frame (vs 5)

┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. CUSTOM - Override specific parameters                                     │
└──────────────────────────────────────────────────────────────────────────────┘

  Set environment variables before running:

    export BYTE_TRACK_MATCH_THRESH=0.90    # Even stricter IoU
    export SCREANALYTICS_TRACK_BUFFER=10   # Shorter track lifetime
    python tools/episode_run.py --ep-id rhobh-s05e01

  Available environment variables:
    • SCREANALYTICS_TRACK_BUFFER      - Track buffer frames (default: 15)
    • BYTE_TRACK_MATCH_THRESH         - IoU threshold (default: 0.85)
    • SCREANALYTICS_TRACK_HIGH_THRESH - Detection confidence (default: 0.65)
    • TRACK_GATE_APPEAR_HARD          - Hard similarity split (default: 0.75)
    • TRACK_GATE_APPEAR_SOFT          - Soft similarity split (default: 0.82)
    • TRACK_GATE_EMB_EVERY            - Embedding interval (default: 5)

┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. BATCH LOAD - Load all strict settings at once                             │
└──────────────────────────────────────────────────────────────────────────────┘

  Source the strict tracking script (useful for custom modifications):

    source scripts/set_strict_tracking.sh
    python tools/episode_run.py --ep-id rhobh-s05e01

┌──────────────────────────────────────────────────────────────────────────────┐
│ VERIFICATION                                                                  │
└──────────────────────────────────────────────────────────────────────────────┘

  Check current configuration:

    python -c "
    from tools.episode_run import TRACK_BUFFER_BASE_DEFAULT, BYTE_TRACK_MATCH_THRESH_DEFAULT, TRACK_HIGH_THRESH_DEFAULT
    print(f'track_buffer: {TRACK_BUFFER_BASE_DEFAULT}')
    print(f'match_thresh: {BYTE_TRACK_MATCH_THRESH_DEFAULT}')
    print(f'track_thresh: {TRACK_HIGH_THRESH_DEFAULT}')
    "

┌──────────────────────────────────────────────────────────────────────────────┐
│ CONFIGURATION FILES                                                           │
└──────────────────────────────────────────────────────────────────────────────┘

  Hardcoded defaults:      tools/episode_run.py (lines 152-169)
  YAML config:             config/pipeline/tracking.yaml
  Environment loader:      scripts/set_strict_tracking.sh
  Ultra-strict runner:     scripts/detect_track_strict.sh
  Documentation:           docs/tracking-strict-config.md

EOF
