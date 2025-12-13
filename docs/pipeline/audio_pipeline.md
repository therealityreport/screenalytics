# Audio Pipeline — Screenalytics

**Version:** 2.0  
**Last Updated:** 2025-12-13

---

## 1. Overview

The Audio Pipeline transforms an episode’s audio into speaker-labeled transcript artifacts that can be used for:

- **Speaking time** (`speaking_s`) per cast member
- **Voice↔cast mapping** (voice clusters / voiceprints)
- **UI review/editing** (speaker groups, segment assignment, archival/splits)

The current default pipeline is **NeMo MSDD diarization + OpenAI Whisper ASR**, with optional Gemini cleanup.

---

## 2. Current Stack (Canonical)

| Stage | Purpose | Default Provider |
|------:|---------|------------------|
| Ingest | Extract audio from video | FFmpeg |
| Separate | Isolate speech from music/SFX | MDX‑Extra (Demucs) |
| Enhance | Denoise/enhance vocals | Resemble Enhance |
| Diarize | Speaker segmentation (overlap-aware) | **NeMo MSDD** |
| ASR | Speech-to-text | OpenAI Whisper (API) |
| Fuse | Align diarization + ASR; build transcript | Screenalytics |
| QC / Export | Validate + write outputs | Screenalytics |

### 2.1 Retired / Legacy Backends

Legacy diarization sources are supported for **read-only manifest compatibility** (and for comparison tooling), but the default pipeline does not use them:

- `pyannote` (deprecated)
- `gpt4o` (deprecated)

---

## 3. Output Artifacts

### 3.1 Audio Files

Stored under `data/audio/{ep_id}/`:

- `episode_original.wav`
- `episode_vocals.wav`
- `episode_vocals_enhanced.wav`
- `episode_final_voice_only.wav`

### 3.2 Manifests

Stored under `data/manifests/{ep_id}/`:

- `audio_diarization.jsonl` — primary diarization segments (NeMo by default)
- `audio_asr_raw.jsonl` — raw ASR output (word timestamps when enabled)
- `audio_diarization_combined.jsonl` — merged/normalized diarization stream (when present)
- `audio_voice_clusters.json` — clustered speakers / voice groups
- `audio_voice_mapping.json` — voice→cast mapping (when available)
- `audio_speaker_groups.json` — UI-facing speaker groups (stable IDs)
- `audio_speaker_assignments.json` — operator assignments (if edited)
- `episode_transcript.jsonl` — fused transcript (speaker labels + text)
- `episode_transcript.vtt` — WebVTT export (for playback)
- `audio_qc.json` — quality report / warnings

---

## 4. Configuration (`config/pipeline/audio.yaml`)

The pipeline is config-driven.

Key defaults (abridged):

```yaml
audio_pipeline:
  diarization:
    provider: "nemo"
    backend: "msdd"
    model_name: "diar_msdd_telephonic"
    min_speakers: 1
    max_speakers: 8
    overlap_threshold: 0.5
    embedding_model: "titanet_large"

  asr:
    provider: "openai_whisper"
    model: "whisper-1"
    language: "en"
    enable_word_timestamps: true

  voice_clustering:
    use_diarization_labels: true
    embedding_model: "titanet_large"
```

See the full file at `config/pipeline/audio.yaml`.

---

## 5. Running (API)

Primary entry points:

- `POST /jobs/episode_audio_pipeline` — run the full audio pipeline
- `GET /jobs/episode_audio_status?ep_id=...` — status/progress
- `GET /jobs/audio/prerequisites` — dependency/API-key checks

Common read endpoints:

- `GET /episodes/{ep_id}/audio/transcript.jsonl`
- `GET /episodes/{ep_id}/audio/transcript.vtt`
- `GET /episodes/{ep_id}/audio/qc.json`

Incremental / operator workflows are exposed under `apps/api/routers/audio.py` (segment moves/splits, speaker assignment, cluster merge, diarize-only/transcribe-only, comparison views).

---

## 6. Speaking Time Semantics (Overlap Policy)

When transcripts are present, speaking time is aggregated with an explicit overlap policy (see `apps/api/services/screentime.py`):

- `shared` — split overlapping seconds across active speakers (default)
- `full` — each active speaker gets full credit
- `primary` — only primary speaker gets credit

---

## 7. Troubleshooting Checklist

- **`/jobs/audio/prerequisites` shows `ffmpeg=false`** → install ffmpeg and ensure it’s on PATH
- **`nemo` missing** → install NeMo deps (GPU strongly recommended)
- **`openai=false`** → set `OPENAI_API_KEY`
- **`resemble=false`** → set `RESEMBLE_API_KEY` (or disable enhancement if supported in your run mode)
- **Empty diarization** → tune `min_speakers/max_speakers`, validate audio separation output, and review `audio_qc.json`

---

## 8. Related Docs

- `docs/audio/overview.md`
- `docs/audio/diarization_manifest.md`
- `docs/reference/api.md`

