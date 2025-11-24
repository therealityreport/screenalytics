# MASTER_TODO.md — Screenalytics

Status: Live tracker
Owner: TRR Analytics

## Legend
[ ] open   [~] in progress   [x] done   (!) blocked

## Program Status (Roll-up)
- detection: PLANNED
- tracking: PLANNED
- identity: PLANNED
- av-fusion: PLANNED
- facebank: PLANNED
- shows-people-ui: PLANNED
- export-results: PLANNED
- agents-mcps: PLANNED
- infra+db: IN_PROGRESS
- ci+policies: IN_PROGRESS

## Critical Path (V2 GA)
1. [ ] Detection (RetinaFace + fallback) → detections.jsonl
2. [ ] Tracking (ByteTrack) → tracks.jsonl
3. [ ] Embedding/Identity (ArcFace + pgvector + hysteresis) → embeddings + assignments
4. [ ] Aggregation → screen_time.csv/json + DB rows
5. [ ] Shows/People UI + featured thumbnails → shared catalog
6. [ ] A/V Fusion (pyannote + Whisper) → speech_segment + av_link
7. [ ] Agents/MCPs (doc-sync, relabel, exports)
8. [ ] CI smoke pipeline on sample clip

## Cross-Cutting / Foundations
- [ ] Config schemas frozen (config/pipeline/*.yaml)
- [ ] Artifact resolver & schema versions
- [ ] Progress events + metrics
- [ ] Pre-commit hooks (black/ruff)
- [ ] Branch protection + required checks
- [ ] .env.example verified

## Feature Backlog (staged in FEATURES/)
### detection
- [ ] Wrapper for RetinaFace; MediaPipe fallback
- [ ] Config: min_size, conf_th, iou_th
- [ ] Save 5-point landmarks; artifact schema v1
- [ ] Unit test: detector config round-trip
- [ ] Integration: sample clip → detections.jsonl

### tracking
- [ ] ByteTrack integration; config track_thresh/match/buffer
- [ ] Scene-boundary safe termination
- [ ] Integration: stable IDs across cuts on sample clip

### identity
- [ ] ArcFace ONNX embedding; pooled track embedding
- [ ] pgvector index + hysteresis policy (enter/exit)
- [ ] Occlusion gate (obstruction classifier)
- [ ] Assignment audit+undo (assignment_history)

### av-fusion
- [ ] Diarization (pyannote) + ASR (faster-whisper)
- [ ] Timestamp alignment + av_link join
- [ ] Metric: speaking overlap precision ≥ 85%

### facebank
- [ ] Self-facebank from episode clusters (HDBSCAN)
- [ ] Curator UI; de-dup + promote chips
- [ ] Reindex FAISS/pgvector service

### shows-people-ui
- [ ] SHOWS CRUD (show/season/episode)
- [ ] PEOPLE grid + featured thumbnails
- [ ] Presence per episode view

### export-results
- [ ] screen_time exporters (CSV/JSON)
- [ ] API endpoints + pagination
- [ ] Zapier/n8n webhook integration

### agents-mcps
- [ ] screenalytics MCP: list_low_confidence, assign_identity, export_screen_time
- [ ] storage MCP: sign_url, list, purge
- [ ] postgres MCP: presence_by_person, episodes_by_show
- [ ] update-docs-on-change playbook wired in CI

## Milestones
- Alpha: end-to-end on 1 sample episode (no AV Fusion)
- Beta: AV Fusion + UI overrides + exports
- GA: agents wired + CI stable + 3 shows processed

## Risks / Watch
- Occlusion edge cases → stricter quality gating
- Audio desync → FFmpeg/PTS verified
- Storage growth → lifecycle rules applied
