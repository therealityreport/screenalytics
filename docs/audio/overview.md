# Audio Pipeline Overview

This document provides an overview of the SCREENALYTICS audio pipeline for episode transcription and voice identification.

## Architecture

The audio pipeline processes episode audio through several stages:

```
Video → Audio Extraction → Stem Separation → Enhancement → Diarization → ASR → Transcript Fusion
```

### Stack Components

| Component | Provider | Purpose |
|-----------|----------|---------|
| Stem Separation | MDX-Extra | Isolate speech from music/SFX |
| Enhancement | Resemble AI | Denoise and enhance vocals |
| Diarization | **NeMo MSDD** | Overlap-aware speaker segmentation |
| ASR | OpenAI Whisper | Speech-to-text transcription |
| Embeddings | NeMo TitaNet | 192-dim speaker embeddings |

## NeMo MSDD Diarization

The audio pipeline uses **NeMo MSDD (Multi-Scale Diarization Decoder)** as the sole diarization backend.

### Key Features

- **Overlap-aware**: Detects when multiple speakers talk simultaneously
- **High accuracy**: State-of-the-art performance on telephone/broadcast speech
- **Speaker embeddings**: Uses TitaNet for 192-dimensional speaker vectors

### Retired Backends

The following diarization backends have been retired:

- **Pyannote** - Legacy, read-only manifest support preserved
- **GPT-4o** - Legacy, read-only manifest support preserved

Legacy manifests (`audio_diarization_pyannote.jsonl`, `audio_diarization_gpt4o.jsonl`) can still be read for backward compatibility, but all new diarization runs use NeMo MSDD.

## Configuration

Audio pipeline settings are in `config/pipeline/audio.yaml`:

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
```

### Speaker Count Hints

- `min_speakers` / `max_speakers`: Provide hints for expected speaker range
- `num_speakers`: Force exact count (overrides min/max)
- Cast count from show is used to auto-tune these values

## Overlap Detection

NeMo MSDD provides overlap-aware diarization with these fields:

| Field | Description |
|-------|-------------|
| `overlap` | Boolean flag for multi-speaker segments |
| `active_speakers` | List of all speakers active in segment |
| `speaker_probs` | Per-speaker probability scores |

### Overlap in Transcripts

Overlap information flows through to transcripts:

```json
{
  "start": 12.34,
  "end": 15.67,
  "speaker_id": "SPK_LISA",
  "text": "I know what you mean",
  "overlap": true,
  "secondary_speakers": ["speaker_1"]
}
```

## Speaking Time Analytics

The screentime service computes `speaking_s` per cast member with configurable overlap handling:

### Overlap Policies

| Policy | Behavior |
|--------|----------|
| `SHARED` (default) | Split duration among all active speakers |
| `FULL` | Each speaker gets full credit |
| `PRIMARY_ONLY` | Only primary speaker gets credit |

## Running the Pipeline

### Via CLI

```bash
python tools/episode_run.py rhobh-s05e17 --stages audio
```

### Via API

```bash
curl -X POST "http://localhost:8000/jobs/episodes/rhobh-s05e17/audio/run"
```

### Incremental Re-runs

Re-run specific stages without redoing earlier stages:

```bash
# Re-run diarization only
curl -X POST "http://localhost:8000/jobs/episodes/rhobh-s05e17/audio/rediarize" \
  -H "Content-Type: application/json" \
  -d '{"min_speakers": 4, "max_speakers": 8}'
```

## GPU Requirements

NeMo MSDD benefits significantly from GPU acceleration:

- **CUDA**: Recommended for production workloads
- **CPU**: Supported but ~10x slower
- **MPS (Apple Silicon)**: Experimental support

## File Artifacts

| File | Description |
|------|-------------|
| `audio_diarization.jsonl` | NeMo diarization segments |
| `audio_diarization.embeddings.json` | Speaker embeddings |
| `audio_voice_clusters.json` | Grouped voice clusters |
| `audio_voice_mapping.json` | Voice-to-cast assignments |
| `episode_transcript.jsonl` | Fused transcript with speakers |
| `episode_transcript.vtt` | WebVTT subtitle format |

## Related Documentation

- [Diarization Manifest Format](diarization_manifest.md)
- [Audio Pipeline Config](../../config/pipeline/audio.yaml)
