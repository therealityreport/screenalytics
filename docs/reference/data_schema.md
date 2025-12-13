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

### audio_diarization.jsonl (audio_diar_v1)
- start (float): Segment start time in seconds
- end (float): Segment end time in seconds
- speaker (text): Raw speaker label from diarization (e.g., "SPEAKER_00")
- confidence (float, null): Diarization confidence score
- Notes: source-specific manifests are also written as `audio_diarization_pyannote.jsonl` and `audio_diarization_gpt4o.jsonl` for comparison.

### audio_speaker_groups.json (audio_sg_v1)
```json
{
  "ep_id": "show-s01e01",
  "schema_version": "audio_sg_v1",
  "sources": [
    {
      "source": "pyannote",
      "summary": {"speakers": 2, "segments": 25, "speech_seconds": 83.8},
      "speakers": [
        {
          "speaker_label": "PY_SPK_00",
          "speaker_group_id": "pyannote:PY_SPK_00",
          "total_duration": 52.4,
          "segment_count": 13,
          "segments": [
            {"segment_id": "py_0001", "start": 0.5, "end": 4.2},
            {"segment_id": "py_0002", "start": 5.0, "end": 7.8}
          ]
        }
      ]
    }
  ]
}
```
- Primary surface for UI; groups remain stable even when clusters change.

### audio_asr_raw.jsonl (audio_asr_v1)
- start (float): Segment start time
- end (float): Segment end time
- text (text): Transcribed text
- confidence (float, null): ASR confidence
- words (array, null): Word-level timings [{w, t0, t1}]
- language (text, null): Detected language

### audio_voice_clusters.json (audio_vc_v2)
```json
[
  {
    "voice_cluster_id": "VC_01",
    "speaker_group_ids": ["pyannote:PY_SPK_00", "gpt4o:LLM_SPK_02"],
    "sources": [
      {"source": "pyannote", "speaker_group_id": "pyannote:PY_SPK_00", "speaker_label": "PY_SPK_00"},
      {"source": "gpt4o", "speaker_group_id": "gpt4o:LLM_SPK_02", "speaker_label": "LLM_SPK_02"}
    ],
    "segments": [{"start": 0.0, "end": 5.0, "diar_speaker": "PY_SPK_00", "speaker_group_id": "pyannote:PY_SPK_00"}],
    "total_duration": 30.0,
    "segment_count": 5,
    "centroid": [0.1, 0.2, ...]
  }
]
```
- Clusters are built on top of speaker groups (not raw micro-segments) and may merge similar groups across sources.

### audio_voice_mapping.json (audio_vm_v1)
```json
[
  {
    "voice_cluster_id": "VC_01",
    "voice_bank_id": "voice_lisa_barlow",
    "speaker_id": "SPK_LISA_BARLOW",
    "speaker_display_name": "Lisa Barlow",
    "similarity": 0.89,
    "is_new_entry": false
  }
]
```

### episode_transcript.jsonl (audio_tx_v1)
- start (float): Segment start time
- end (float): Segment end time
- text (text): Transcribed text
- speaker_id (text): Speaker ID (e.g., "SPK_LISA_BARLOW" or "SPK_UNLABELED_01")
- speaker_display_name (text): Human-readable speaker name
- voice_cluster_id (text): Episode voice cluster ID (e.g., "VC_01")
- voice_bank_id (text): Voice bank entry ID
- conf (float, null): Confidence score
- words (array, null): Word-level timings

### episode_transcript.vtt (audio_vtt_v1)
WebVTT with speaker metadata in NOTE lines:
```
NOTE speaker_id=SPK_LISA_BARLOW speaker_display_name="Lisa Barlow" voice_cluster_id=VC_01 voice_bank_id=voice_lisa_barlow
1
00:00:00.000 --> 00:00:05.000
<v Lisa Barlow>I said what I said.</v>
```

### audio_qc.json (audio_qc_v1)
```json
{
  "ep_id": "rhoslc-s06e02",
  "status": "ok",  // ok | warn | needs_review
  "metrics": [
    {"name": "duration_drift_pct", "value": 0.14, "threshold": 1.0, "passed": true},
    {"name": "snr_db", "value": 22.5, "threshold": 14.0, "passed": true}
  ],
  "voice_cluster_count": 5,
  "labeled_voices": 4,
  "unlabeled_voices": 1,
  "transcript_row_count": 245,
  "warnings": [],
  "errors": []
}
```

## Indices & Performance
- HNSW(vec) for embedding.vec (dim 512).
- btree on time ranges: (ep_id, start_s, end_s).
- Recommended partition by ep_id or by show/season if tables grow.

## Schema Versioning
- Every artifact/table row carries schema_version where applicable.
- Migrations bump version and include compatibility views.
