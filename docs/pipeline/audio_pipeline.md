# Audio Pipeline — Screenalytics

**Version:** 1.0
**Last Updated:** 2025-11-29

---

## 1. Overview

The Audio Pipeline transforms video episodes into structured transcripts with speaker identification. It extracts audio, separates speech from background noise, identifies speakers through diarization, transcribes speech via ASR, and maps voice clusters to known cast members through the voice bank.

```
┌───────────────────────────────────────────────────────────────────────┐
│                        AUDIO PIPELINE                                   │
└───────────────────────────────────────────────────────────────────────┘

1. INGEST → 2. SEPARATE → 3. ENHANCE → 4. DIARIZE → 5. ASR → 6. FUSE → 7. QC
     ↓           ↓            ↓           ↓          ↓        ↓        ↓
  .wav      vocals.wav   enhanced.wav  segments  asr.jsonl  transcript  qc.json
```

---

## 2. Pipeline Stages

| Stage | Input | Output | Models/Tools | Purpose |
|-------|-------|--------|--------------|---------|
| **1. Ingest** | episode.mp4 | episode_original.wav | FFmpeg | Extract audio stream |
| **2. Separate** | episode_original.wav | episode_vocals.wav | MDX-Extra/Demucs | Isolate speech from music/SFX |
| **3. Enhance** | episode_vocals.wav | episode_vocals_enhanced.wav | Resemble Enhance | Denoise and clean audio |
| **4. Diarize** | episode_vocals_enhanced.wav | audio_diarization.jsonl | Pyannote 3.1 | Segment by speaker |
| **5. ASR** | episode_vocals_enhanced.wav + segments | audio_asr_raw.jsonl | OpenAI Whisper / Gemini | Transcribe with word timestamps |
| **6. Fuse** | diarization + ASR + voice bank | episode_transcript.jsonl/vtt | Custom | Merge and label speakers |
| **7. QC** | All artifacts | audio_qc.json | Custom | Validate quality metrics |

---

## 3. Data Layout

### 3.1 Audio Artifacts

```
data/audio/{ep_id}/
├── episode_original.wav          # Full audio from video
├── episode_vocals.wav            # Speech-only (separated)
├── episode_vocals_enhanced.wav   # Denoised vocals
└── episode_final_voice_only.wav  # Final cleaned audio
```

### 3.2 Manifest Files

```
data/manifests/{ep_id}/
├── audio_diarization.jsonl       # Speaker segments
├── audio_asr_raw.jsonl           # Raw ASR output
├── audio_voice_clusters.json     # Clustered speakers
├── audio_voice_mapping.json      # Voice → Cast mapping
├── episode_transcript.jsonl      # Final transcript
├── episode_transcript.vtt        # WebVTT format
└── audio_qc.json                 # Quality report
```

---

## 4. Configuration

Configuration is defined in `config/pipeline/audio.yaml`:

```yaml
audio_pipeline:
  separation:
    provider: "mdx_extra"
    model_name: "mdx_extra_q"
    chunk_seconds: 15
    overlap_seconds: 2
    device: "auto"

  enhance:
    provider: "resemble"
    mode: "studio"
    batch_seconds: 20

  diarization:
    provider: "pyannote"
    model_name: "pyannote/speaker-diarization-3.1"
    min_speech: 0.2
    merge_gap_ms: 300
    min_speakers: 1
    max_speakers: 10

  asr:
    provider: "openai_whisper"  # or "gemini"
    model: "whisper-1"
    language: "en"
    timestamp_granularity: "word"

  voice_clustering:
    similarity_threshold: 0.78
    min_samples_per_cluster: 2
    merge_gap_ms: 500

  qc:
    max_duration_drift_pct: 1.0
    min_snr_db: 14.0
    warn_snr_db: 18.0
    min_diarization_conf: 0.65
    min_asr_conf: 0.70

  export:
    sample_rate: 48000
    bit_depth: 24
    format: "wav"
```

---

## 5. API Endpoints

### 5.1 Start Audio Pipeline

```bash
POST /jobs/episode_audio_pipeline
Content-Type: application/json

{
  "ep_id": "rhobh-s05e02",
  "overwrite": false,
  "asr_provider": "openai_whisper"  # or "gemini"
}
```

Response:
```json
{
  "job_id": "audio-job-abc123",
  "ep_id": "rhobh-s05e02",
  "job_type": "audio_pipeline",
  "status": "started"
}
```

### 5.2 Query Job Status

```bash
GET /jobs/episode_audio_status?ep_id=rhobh-s05e02
```

Response:
```json
{
  "ep_id": "rhobh-s05e02",
  "status": "in_progress",
  "progress_pct": 45.0,
  "current_step": "diarization",
  "job_id": "audio-job-abc123"
}
```

### 5.3 Download Transcript (VTT)

```bash
GET /episodes/{ep_id}/audio/transcript.vtt
```

Returns WebVTT file with speaker labels.

### 5.4 Download Transcript (JSONL)

```bash
GET /episodes/{ep_id}/audio/transcript.jsonl
```

Returns JSONL with full metadata per line.

### 5.5 Check Prerequisites

```bash
GET /audio/prerequisites
```

Response:
```json
{
  "pyannote_available": true,
  "resemble_api_key_set": true,
  "openai_api_key_set": true,
  "gemini_api_key_set": false
}
```

---

## 6. Celery Queues

The audio pipeline uses dedicated Celery queues for load balancing:

| Queue Name | Task | Description |
|------------|------|-------------|
| `SCREENALYTICS_AUDIO_INGEST` | `audio.ingest` | Audio extraction |
| `SCREENALYTICS_AUDIO_SEPARATE` | `audio.separate` | Vocal separation |
| `SCREANALYTICS_AUDIO_ENHANCE` | `audio.enhance` | Denoise/enhance |
| `SCREENALYTICS_AUDIO_DIARIZE` | `audio.diarize` | Speaker diarization |
| `SCREENALYTICS_AUDIO_VOICES` | `audio.voices` | Voice clustering |
| `SCREENALYTICS_AUDIO_TRANSCRIBE` | `audio.transcribe` | ASR transcription |
| `SCREANALYTICS_AUDIO_ALIGN` | `audio.align` | Word alignment |
| `SCREENALYTICS_AUDIO_QC` | `audio.qc` | Quality checks |
| `SCREENALYTICS_AUDIO_EXPORT` | `audio.export` | Export formats |
| `SCREENALYTICS_AUDIO_PIPELINE` | `audio.pipeline` | Full pipeline orchestrator |

### Worker Command

```bash
celery -A apps.api.celery_app:celery_app worker \
  -Q SCREENALYTICS_AUDIO_PIPELINE,SCREENALYTICS_AUDIO_INGEST,SCREENALYTICS_AUDIO_SEPARATE,SCREENALYTICS_AUDIO_ENHANCE,SCREANALYTICS_AUDIO_DIARIZE,SCREENALYTICS_AUDIO_TRANSCRIBE,SCREANALYTICS_AUDIO_QC \
  -l info
```

---

## 7. Artifact Schemas

### 7.1 `audio_diarization.jsonl`

```jsonl
{
  "speaker_label": "SPEAKER_00",
  "start_ms": 0,
  "end_ms": 5000,
  "confidence": 0.95
}
```

### 7.2 `audio_asr_raw.jsonl`

```jsonl
{
  "text": "Hello, how are you?",
  "start_ms": 0,
  "end_ms": 2500,
  "confidence": 0.97,
  "words": [
    {"word": "Hello", "start_ms": 0, "end_ms": 400},
    {"word": "how", "start_ms": 450, "end_ms": 650},
    {"word": "are", "start_ms": 700, "end_ms": 850},
    {"word": "you", "start_ms": 900, "end_ms": 1100}
  ]
}
```

### 7.3 `audio_voice_clusters.json`

```json
{
  "clusters": [
    {
      "voice_cluster_id": "vc_001",
      "total_duration": 120.5,
      "segment_count": 15,
      "centroid": [0.1, 0.2, ...]  // 256-d embedding
    }
  ]
}
```

### 7.4 `audio_voice_mapping.json`

```json
{
  "mappings": [
    {
      "voice_cluster_id": "vc_001",
      "voice_bank_id": "vb_lisa_vanderpump",
      "similarity": 0.89,
      "display_name": "Lisa Vanderpump",
      "cast_member_id": "cast_001"
    }
  ]
}
```

### 7.5 `episode_transcript.jsonl`

```jsonl
{
  "idx": 0,
  "start_ms": 0,
  "end_ms": 5000,
  "text": "Hello, how are you?",
  "speaker_id": "cast_001",
  "speaker_display_name": "Lisa Vanderpump",
  "voice_cluster_id": "vc_001",
  "voice_bank_id": "vb_lisa_vanderpump",
  "diarization_confidence": 0.95,
  "asr_confidence": 0.97
}
```

### 7.6 `episode_transcript.vtt`

```vtt
WEBVTT

00:00:00.000 --> 00:00:05.000
<v Lisa Vanderpump>Hello, how are you?</v>

00:00:05.500 --> 00:00:10.000
<v Kyle Richards>I'm doing great, thanks!</v>
```

### 7.7 `audio_qc.json`

```json
{
  "ep_id": "rhobh-s05e02",
  "status": "ok",
  "metrics": [
    {"name": "duration_drift_pct", "value": 0.14, "threshold": 1.0, "passed": true},
    {"name": "snr_db", "value": 22.5, "threshold": 14.0, "passed": true},
    {"name": "mean_diarization_conf", "value": 0.88, "threshold": 0.65, "passed": true},
    {"name": "mean_asr_conf", "value": 0.92, "threshold": 0.70, "passed": true}
  ],
  "voice_cluster_count": 5,
  "labeled_voices": 4,
  "unlabeled_voices": 1,
  "transcript_row_count": 245,
  "warnings": [],
  "errors": []
}
```

---

## 8. Voice Bank Integration

The audio pipeline integrates with the existing voice bank for speaker identification:

### 8.1 Voice Bank Location

```
data/voice_bank/{show_slug}/
├── {voice_bank_id}/
│   ├── embedding.npy           # 256-d speaker embedding
│   ├── samples/                # Reference audio clips
│   │   ├── sample_001.wav
│   │   └── sample_002.wav
│   └── metadata.json           # Name, cast member ID, etc.
```

### 8.2 Matching Process

1. Extract speaker embeddings from diarization segments
2. Cluster similar voices into `voice_clusters`
3. Compute cosine similarity between cluster centroids and voice bank entries
4. Map clusters to cast members if similarity > threshold (default 0.78)
5. Assign "Unlabeled Voice N" for unmatched clusters

---

## 9. QC Thresholds

| Metric | Error Threshold | Warning Threshold | Description |
|--------|-----------------|-------------------|-------------|
| Duration Drift | > 2% | > 1% | Final audio vs original duration |
| SNR | < 14 dB | < 18 dB | Signal-to-noise ratio |
| Diarization Confidence | < 0.50 | < 0.65 | Mean segment confidence |
| ASR Confidence | < 0.60 | < 0.70 | Mean word confidence |
| Voice Cluster Duration | < 5s | < 10s | Minimum cluster duration |

---

## 10. Environment Variables

Add to `.env`:

```bash
# Resemble AI (audio enhancement)
RESEMBLE_API_KEY=your_resemble_api_key

# Pyannote (speaker diarization)
PYANNOTE_AUTH_TOKEN=your_huggingface_token

# OpenAI (primary ASR)
OPENAI_API_KEY=your_openai_api_key

# Gemini (secondary ASR, optional)
GEMINI_API_KEY=your_gemini_api_key

# Celery/Redis
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

---

## 11. Streamlit UI Integration

### 11.1 Upload Page

The Upload Video page includes options to:
- **Run audio pipeline after upload**: Checkbox to automatically start audio processing
- **ASR Provider**: Dropdown to select Whisper (default) or Gemini

### 11.2 Episode Detail Page

The Episode Detail page includes an "Audio & Transcript" section with:
- **Status**: Pipeline completion status with QC badge
- **Voice Stats**: Cluster count, labeled vs unlabeled voices
- **Generate Button**: Start audio pipeline on-demand
- **Download Links**: VTT and JSONL transcript downloads

---

## 12. Common Workflows

### 12.1 Process Single Episode (API)

```bash
# Start audio pipeline
curl -X POST http://localhost:8000/jobs/episode_audio_pipeline \
  -H "Content-Type: application/json" \
  -d '{"ep_id": "rhobh-s05e02"}'

# Poll status
curl "http://localhost:8000/jobs/episode_audio_status?ep_id=rhobh-s05e02"

# Download transcript when complete
curl "http://localhost:8000/episodes/rhobh-s05e02/audio/transcript.vtt" -o transcript.vtt
```

### 12.2 Reprocess with Overwrite

```bash
curl -X POST http://localhost:8000/jobs/episode_audio_pipeline \
  -H "Content-Type: application/json" \
  -d '{"ep_id": "rhobh-s05e02", "overwrite": true, "asr_provider": "gemini"}'
```

---

## 13. Troubleshooting

### 13.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| No audio output | Video has no audio track | Check video with `ffprobe` |
| Low SNR warning | Noisy source audio | Consider manual audio cleanup |
| Missing speakers | Diarization threshold too high | Lower `min_speech` config |
| Poor transcription | Wrong language setting | Set correct `language` in config |
| Voice bank mismatch | Low similarity threshold | Add more voice bank samples |

### 13.2 Debug Mode

Set `AUDIO_PIPELINE_DEBUG=1` to preserve intermediate artifacts and enable verbose logging.

---

**Maintained by:** Screenalytics Engineering
**Next Review:** After Phase 3 completion
