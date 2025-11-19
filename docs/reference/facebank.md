# Facebank Reference — Screenalytics

**Version:** 2.0
**Last Updated:** 2025-11-18

---

## 1. Overview

The **Facebank** is a persistent repository of known cast member face embeddings used for:

1. **Cross-episode identity recognition:** Automatically label tracks across episodes
2. **Seed-based matching:** Reference faces uploaded outside of episode detection
3. **Manual moderation:** UI for managing identity assignments, merges, splits

**Storage:** S3/local filesystem + Postgres (pgvector) for embedding search

---

## 2. Facebank Structure

### 2.1 Filesystem Layout (S3/FS)
```
facebank/
  {person_id}/
    {seed_id}_d.png       # Display derivative (512x512, high quality)
    {seed_id}_e.png       # Embed derivative (112x112, ArcFace input size)
    {seed_id}_orig.png    # Original upload (if FACEBANK_KEEP_ORIG=1)

artifacts/thumbs/
  {show_slug}/s{season}/e{episode}/identities/
    {identity_id}/
      rep.jpg             # Representative thumbnail for episode identity
```

### 2.2 Database (Postgres + pgvector)
```sql
CREATE TABLE facebank_seed (
  seed_id UUID PRIMARY KEY,
  person_id VARCHAR,
  display_path VARCHAR,
  embed_path VARCHAR,
  embedding VECTOR(512),  -- ArcFace 512-d unit-norm embedding
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON facebank_seed USING HNSW (embedding vector_cosine_ops);
```

---

## 3. Seed Upload Workflow

### 3.1 API
```bash
POST /cast/{cast_id}/seeds/upload
Content-Type: multipart/form-data

Form Data:
  - file: <image.jpg>
  - person_id: "lisa-vanderpump"
```

### 3.2 Processing Steps
1. **Upload:** Receive image file
2. **EXIF transpose:** Rotate based on EXIF orientation
3. **Face detection:** RetinaFace detects face, extracts landmarks
4. **Crop & align:** Align face to canonical pose using landmarks
5. **Generate derivatives:**
   - Display derivative (`{seed_id}_d.png`): Resize to `SEED_DISPLAY_SIZE` (default: 512x512)
   - Embed derivative (`{seed_id}_e.png`): Resize to `SEED_EMBED_SIZE` (default: 112x112)
   - Original (optional): Save original post-EXIF-transpose if `FACEBANK_KEEP_ORIG=1`
6. **Extract embedding:** ArcFace ONNX on embed derivative → 512-d unit-norm vector
7. **Store:** Write derivatives to S3/FS, insert embedding into Postgres (pgvector)

### 3.3 Configuration
```bash
# config/pipeline/facebank.yaml (or environment variables)
SEED_DISPLAY_SIZE: 512        # Display derivative size (px)
SEED_EMBED_SIZE: 112          # Embed derivative size (px, ArcFace input)
SEED_DISPLAY_FORMAT: png      # png | jpg
SEED_EMBED_FORMAT: png        # png | jpg
SEED_JPEG_QUALITY: 92         # JPEG quality if format=jpg
SEED_DISPLAY_MIN: 256         # Minimum display size
SEED_DISPLAY_MAX: 1024        # Maximum display size
FACEBANK_KEEP_ORIG: 1         # Keep original upload for re-cropping
```

---

## 4. Cross-Episode Matching

### 4.1 Algorithm
For each episode identity (cluster):

1. **Compute cluster centroid:** Mean of track-level embeddings
2. **Query pgvector:** Find top-K nearest Facebank seed embeddings
   ```sql
   SELECT person_id, seed_id, embedding <=> $1::VECTOR AS distance
   FROM facebank_seed
   ORDER BY embedding <=> $1::VECTOR
   LIMIT 5;
   ```
3. **Match threshold:** If `distance < (1 - cluster_thresh)` (e.g., distance < 0.42 for `cluster_thresh=0.58`), assign `person_id` label
4. **Ambiguity handling:**
   - If multiple seeds > threshold from different people → flag for manual review
   - If no seeds > threshold → leave unlabeled (singleton cluster)

### 4.2 Auto-Assignment
```bash
POST /jobs/cluster
{
  "ep_id": "rhobh-s05e02",
  "facebank_matching": true,
  "match_threshold": 0.70
}
```

**Response:** `identities.json` with `labels.person_id` populated

---

## 5. Facebank UI

### 5.1 Identity Grid
- Displays all identities for selected episode
- Presigned thumbnail URLs from S3
- Inline controls: Rename, Delete, Merge into…

### 5.2 Cluster Drill-Down
- Click identity → view tracks
- Track thumbnails in 4:5 aspect ratio filmstrip
- Pagination: Load more (40 crops per page)
- Moderation actions: View track, Move to identity, Remove from identity, Delete track

### 5.3 Moderation Endpoints

| Action | Endpoint | Effect |
|--------|----------|--------|
| **Merge identities** | `POST /episodes/{ep_id}/identities/merge` | Combine source identities into target |
| **Split identity** | `POST /episodes/{ep_id}/identities/split` | Extract tracks into new identity |
| **Move track** | `POST /episodes/{ep_id}/tracks/{track_id}/move` | Move track between identities |
| **Delete track** | `DELETE /episodes/{ep_id}/tracks/{track_id}` | Soft-delete track |
| **Lock identity** | `PATCH /episodes/{ep_id}/identities/{identity_id}` | Protect from auto-merge/split |

---

## 6. Backfill & Maintenance

### 6.1 Backfill Display Derivatives
If display derivatives are missing or low-res (≤128px):

```bash
POST /jobs/facebank/backfill_display
{
  "person_id": "lisa-vanderpump"
}
```

**Behavior:**
1. For each seed, check if display derivative exists and meets size requirements
2. If original exists, regenerate display from original
3. If no original, upscale from embed derivative (mark `low_res=true`)

### 6.2 Re-Embed Seeds
If ArcFace model is upgraded or embedding dimension changes:

```bash
POST /jobs/facebank/reembed_all
```

**Behavior:**
1. For each seed, load embed derivative
2. Re-extract embedding with new ArcFace model
3. Update `facebank_seed.embedding` in Postgres

---

## 7. Storage Costs & Lifecycle

### 7.1 Derivative Sizes
Typical sizes per seed:
- Display derivative (`512x512 PNG`): ~400 KB
- Embed derivative (`112x112 PNG`): ~15 KB
- Original (`variable, JPEG`): ~200 KB

**Total per seed:** ~615 KB

### 7.2 Lifecycle Rules
Facebank seeds are **permanent** (no expiration):
- Display derivatives: kept indefinitely for UI
- Embed derivatives: kept indefinitely for re-embedding
- Originals: kept indefinitely if `FACEBANK_KEEP_ORIG=1`

Episode identity thumbnails:
- Expire after N days (configurable)
- Regenerate on-demand from `faces.jsonl` + crops

---

## 8. Migration & Backup

### 8.1 Export Facebank
```bash
python tools/facebank_export.py --output facebank_backup.tar.gz
```

**Contents:**
- All derivative images
- `facebank_manifest.jsonl` (person_id, seed_id, embedding)

### 8.2 Import Facebank
```bash
python tools/facebank_import.py --input facebank_backup.tar.gz
```

**Behavior:**
- Upload derivatives to S3/FS
- Insert embeddings into Postgres (pgvector)
- Validate embedding norms and integrity

---

## 9. References

- [Cluster Identities](../pipeline/cluster_identities.md)
- [Artifact Schemas](artifacts_faces_tracks_identities.md)
- [Config Reference](config/pipeline_configs.md)

---

**Maintained by:** Screenalytics Engineering
