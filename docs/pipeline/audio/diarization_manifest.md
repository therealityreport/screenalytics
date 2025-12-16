# NeMo Diarization Manifest Format

This document specifies the canonical format for diarization manifests in SCREENALYTICS.

## Overview

The diarization manifest stores speaker segmentation results from NeMo MSDD (Multi-Scale Diarization Decoder). Each segment represents a time range where a speaker is detected, with optional overlap information when multiple speakers are active simultaneously.

## File Location

```
data/manifests/{ep_id}/audio_diarization.jsonl
```

## Format: JSONL (JSON Lines)

Each line is a self-contained JSON object representing one diarization segment.

## Schema: `NeMoDiarizationSegment`

```json
{
  "segment_id": "diar_12.340_15.670",
  "start": 12.34,
  "end": 15.67,
  "speaker": "speaker_0",
  "confidence": 0.92,
  "overlap": false,
  "active_speakers": ["speaker_0"],
  "speaker_probs": {"speaker_0": 0.92, "speaker_1": 0.05}
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `start` | `float` | Start time in seconds |
| `end` | `float` | End time in seconds |
| `speaker` | `string` | Primary speaker label (e.g., `speaker_0`) |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `segment_id` | `string` | auto-generated | Stable identifier for the segment |
| `confidence` | `float` | `null` | Diarization confidence (0-1) |
| `overlap` | `bool` | `false` | Whether multiple speakers are active |
| `active_speakers` | `list[str]` | `[]` | All speakers active in this segment |
| `speaker_probs` | `dict[str, float]` | `{}` | Per-speaker probability scores |

## Overlap Detection

NeMo MSDD provides overlap-aware diarization. When multiple speakers are detected:

1. `overlap` is set to `true`
2. `active_speakers` contains all detected speakers
3. `speaker_probs` contains probability scores for each speaker
4. `speaker` contains the primary (highest probability) speaker

### Example: Overlapping Segment

```json
{
  "segment_id": "diar_45.200_47.800",
  "start": 45.2,
  "end": 47.8,
  "speaker": "speaker_0",
  "confidence": 0.78,
  "overlap": true,
  "active_speakers": ["speaker_0", "speaker_2"],
  "speaker_probs": {"speaker_0": 0.78, "speaker_2": 0.65}
}
```

## Example Manifest

```jsonl
{"segment_id": "diar_0.000_3.450", "start": 0.0, "end": 3.45, "speaker": "speaker_0", "confidence": 0.95, "overlap": false, "active_speakers": ["speaker_0"], "speaker_probs": {"speaker_0": 0.95}}
{"segment_id": "diar_3.500_8.200", "start": 3.5, "end": 8.2, "speaker": "speaker_1", "confidence": 0.88, "overlap": false, "active_speakers": ["speaker_1"], "speaker_probs": {"speaker_1": 0.88}}
{"segment_id": "diar_8.100_10.500", "start": 8.1, "end": 10.5, "speaker": "speaker_0", "confidence": 0.72, "overlap": true, "active_speakers": ["speaker_0", "speaker_1"], "speaker_probs": {"speaker_0": 0.72, "speaker_1": 0.68}}
```

## Embeddings File

Speaker embeddings are stored in a companion file:

```
data/manifests/{ep_id}/audio_diarization.embeddings.json
```

### Embeddings Schema

```json
{
  "embeddings": {
    "speaker_0": [0.123, -0.456, ...],
    "speaker_1": [0.789, 0.012, ...]
  },
  "speaker_count": 2,
  "total_speech_duration": 245.67,
  "overlap_duration": 12.34,
  "metadata": {
    "model": "titanet_large",
    "embedding_dim": 192,
    "diarization_model": "diar_msdd_telephonic"
  }
}
```

## Legacy Manifests (Read-Only)

For backward compatibility, the system can read legacy manifests:

| Legacy File | Format | Status |
|-------------|--------|--------|
| `audio_diarization_pyannote.jsonl` | Pyannote 3.x | Deprecated |
| `audio_diarization_gpt4o.jsonl` | GPT-4o | Deprecated |

Legacy manifests lack overlap information and use simplified schemas.

## API Functions

### Loading Manifests

```python
from py_screenalytics.audio.diarization_nemo import load_diarization_manifest

segments = load_diarization_manifest(Path("data/manifests/ep-s01e01/audio_diarization.jsonl"))
```

### Saving Manifests

```python
from py_screenalytics.audio.diarization_nemo import save_diarization_manifest

save_diarization_manifest(segments, Path("data/manifests/ep-s01e01/audio_diarization.jsonl"))
```

## Related Configuration

See `config/pipeline/audio.yaml` for diarization settings:

```yaml
diarization:
  provider: "nemo"
  backend: "msdd"
  model_name: "diar_msdd_telephonic"
  overlap_threshold: 0.5
  embedding_model: "titanet_large"
```
