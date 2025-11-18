# DATA_SCHEMA.md — Screenalytics

Version: 1.0
Status: Draft (ready to implement)

## Principles
- Explicit artifact schemas (JSONL/Parquet) with schema_version.
- Postgres is the source of truth for relationships.
- pgvector holds face embeddings; HNSW index for k-NN.

## ER Overview (text)
show(1) ──< season(1) ──< episode(1) ──< track(1) ──< assignment
                    └─< shot
person ──< cast_membership (to show)
person ──< media_asset (featured)
episode ──< detection
track ──< embedding (owner_type='track')
person ──< embedding (owner_type='facebank')
episode ──< speech_segment
track ──< av_link >── speech_segment
episode+person ──< screen_time

## Tables (Postgres)

### show
- show_id (uuid, pk)
- slug (text, unique)
- title (text)
- network (text, null)
- status (text)
- created_at, updated_at (timestamptz)

### season
- season_id (uuid, pk)
- show_id (fk show)
- number (int)
- year (int, null)
- meta (jsonb)
- unique(show_id, number)
- created_at, updated_at

### episode
- ep_id (uuid, pk)
- season_id (fk season)
- number (int)
- air_date (date, null)
- title (text, null)
- duration_s (int, null)
- meta (jsonb)
- unique(season_id, number)
- created_at, updated_at

### shot
- shot_id (uuid, pk)
- ep_id (fk episode)
- start_s (float)
- end_s (float)
- index btree(ep_id, start_s, end_s)

### detection
- det_id (uuid, pk)
- ep_id (fk)
- ts_s (float)
- bbox (float[4])   -- [x1,y1,x2,y2]
- landmarks (float[10]) -- 5 points
- conf (float)
- schema_version (text)
- index btree(ep_id, ts_s)

### track
- track_id (uuid, pk)
- ep_id (fk)
- start_s (float)
- end_s (float)
- occlusion_rate (float, null)
- quality_stats_json (jsonb, null)
- index btree(ep_id, start_s, end_s)

### embedding
- emb_id (uuid, pk)
- owner_type (text check in ['facebank','track'])
- owner_id (uuid)  -- track_id or person_id
- vec (vector(512))
- model (text)     -- e.g., arcface_r100
- created_at (timestamptz)
- index hnsw(vec) with (m=16, ef_construction=200)
- index btree(owner_type, owner_id)

### assignment
- track_id (uuid fk track)
- person_id (uuid fk person)
- method (text)    -- arcface_v1, fusion_v1, user
- score (float)
- threshold (float)
- decided_at (timestamptz)
- locked (bool default false)
- label_source (text)  -- auto|user|ocr|av_fusion
- primary key(track_id)

### assignment_history
- id (uuid, pk)
- target_type (text check in ['cluster','track','frame'])
- target_id (uuid)
- old_person_id (uuid, null)
- new_person_id (uuid, null)
- actor (text) -- user/agent
- reason (text)
- ts (timestamptz)
- index btree(target_type, target_id, ts desc)

### person
- person_id (uuid, pk)
- canonical_name (text)
- display_name (text)
- aliases (text[])
- created_at, updated_at

### cast_membership
- id (uuid, pk)
- show_id (fk show)
- person_id (fk person)
- first_season (int)
- last_season (int)
- roles (text[])
- active (bool default true)
- index btree(show_id, person_id)

### media_asset
- asset_id (uuid, pk)
- owner_type (text check in ['person','show','season','episode'])
- owner_id (uuid)
- kind (text check in ['featured','chip','poster','thumb'])
- url (text)
- width (int), height (int), hash (text)
- created_at, updated_at
- unique partial: (owner_type, owner_id, kind) where kind='featured'

### person_featured
- person_id (fk person, pk)
- asset_id (fk media_asset, unique)

### speech_segment
- seg_id (uuid, pk)
- ep_id (fk)
- start_s (float)
- end_s (float)
- speaker_label (text)     -- cluster id
- speaker_conf (float)
- index btree(ep_id, start_s, end_s)

### transcript
- token_id (uuid, pk)
- ep_id (fk)
- start_s (float)
- end_s (float)
- text (text)
- asr_conf (float)
- speaker_label (text, null)

### av_link
- track_id (fk track)
- seg_id (fk speech_segment)
- overlap_s (float)
- label (text check in ['visual','speaking','both'])
- primary key(track_id, seg_id)

### screen_time
- ep_id (fk)
- person_id (fk)
- visual_s (float)
- speaking_s (float)
- both_s (float)
- confidence (float)
- primary key(ep_id, person_id)

## Artifact Schemas (files in object storage)

### detections.jsonl (schema_version: 'det_v1')
- ep_id, ts_s, bbox [x1,y1,x2,y2], landmarks [10], conf, model_id

### chips_manifest.parquet (chip_v1)
- track_id, frame_ts, yaw, pitch, roll, magface, serfiq, sharpness, occluded(bool), path

### tracks.jsonl (track_v1)
- track_id, ep_id, start_s, end_s, frame_span [start,end], sample_thumbs [paths]

## Indices & Performance
- HNSW(vec) for embedding.vec (dim 512).
- btree on time ranges: (ep_id, start_s, end_s).
- Recommended partition by ep_id or by show/season if tables grow.

## Schema Versioning
- Every artifact/table row carries schema_version where applicable.
- Migrations bump version and include compatibility views.
