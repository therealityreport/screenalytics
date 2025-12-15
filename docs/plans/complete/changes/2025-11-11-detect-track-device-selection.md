# 2025-11-11 — Detect/track device selection

## Highlights
- `tools/episode_run.py` now picks the safest device automatically (CUDA → MPS → CPU) via `--device auto`, with explicit overrides for `cpu`, `mps`, or `cuda`. The resolved device is logged in the run summary.
- `/jobs/detect_track` accepts a `device` value that flows from the Streamlit UI (Upload, Episodes, Episode Detail pages) down to the CLI, so operators can choose Auto vs. CPU/MPS/CUDA per run.
- Added unit/integration coverage ensuring the picker falls back to CPU when no accelerators exist.

## Why it matters
Most contributors don’t have GPUs. Making CPU the safe baseline while still supporting MPS/CUDA when available prevents Ultralytics from crashing and keeps the UI consistent across machines.

## How to use it
- UI: use the **Device** dropdown (Auto/CPU/MPS/CUDA). Auto picks CUDA if present, then MPS, else CPU.
- CLI: `python tools/episode_run.py ... --device auto` (or `cpu` / `mps` / `cuda`).
- API: POST `/jobs/detect_track` with `{ "device": "cpu" }` (default `auto`).
