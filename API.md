# API.md — Screenalytics

Version: 0.1 (skeleton)

## Auth
TBD (key or session). All write routes audited.

## Routes (high-level)
- `POST /shows`, `GET /shows/:id`, `PATCH /shows/:id`
- `POST /shows/:id/seasons`
- `POST /seasons/:id/episodes`
- `POST /episodes/:id/assets` → signed URLs
- `POST /jobs/{stage}` (ingest, detect, track, embed, assign, diarize, transcribe, fuse, aggregate)
- `GET /jobs/:id/status`
- `GET /episodes/:id/screen_time`
- `GET /people/:id/presence?show_id=...`
- `POST /tracks/:id/assign`  (manual)
- `POST /clusters/:id/split`  (manual)
- `POST /frames/:id/exclude`  (manual)

## Response envelope
```json
{ "ok": true, "data": {...}, "meta": {"request_id":"..."} }
```
