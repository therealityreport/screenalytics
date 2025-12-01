"""OpenAI ASR integration.

Handles:
- Speech-to-text using OpenAI Whisper API (whisper-1)
- Speech-to-text using GPT-4o Transcribe models (gpt-4o-transcribe, gpt-4o-mini-transcribe)
- Speech-to-text with diarization (gpt-4o-transcribe-diarize)
- Audio chunking for long files
- Word-level timestamp extraction (whisper-1 only)
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from .models import ASRConfig, ASRSegment, WordTiming

LOGGER = logging.getLogger(__name__)


def _get_openai_client():
    """Get OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for Whisper ASR. "
            "Get your API key from https://platform.openai.com/api-keys"
        )

    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError as e:
        raise ImportError(
            "openai package is required for Whisper ASR. "
            "Install with: pip install openai"
        ) from e


def transcribe_audio(
    audio_path: Path,
    output_path: Path,
    config: Optional[ASRConfig] = None,
    overwrite: bool = False,
) -> List[ASRSegment]:
    """Transcribe audio using OpenAI Whisper API.

    Args:
        audio_path: Path to input audio file
        output_path: Path for ASR manifest (JSONL)
        config: ASR configuration
        overwrite: Whether to overwrite existing results

    Returns:
        List of ASRSegment objects
    """
    if output_path.exists() and not overwrite:
        LOGGER.info(f"ASR results already exist: {output_path}")
        return _load_asr_manifest(output_path)

    config = config or ASRConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = _get_openai_client()

    LOGGER.info(f"Transcribing audio: {audio_path}")

    # Get audio duration
    from .io import get_audio_duration
    total_duration = get_audio_duration(audio_path)

    # Check if we need to chunk
    chunk_duration = config.chunk_duration_seconds
    if total_duration <= chunk_duration:
        # Process whole file at once
        segments = _transcribe_file(client, audio_path, config, total_duration)
    else:
        # Process in chunks
        LOGGER.info(f"Processing {total_duration:.1f}s audio in {chunk_duration}s chunks")
        segments = _transcribe_chunked(client, audio_path, config, total_duration)

    # Save manifest
    _save_asr_manifest(segments, output_path)

    total_words = sum(len(s.words or []) for s in segments)
    LOGGER.info(f"Transcription complete: {len(segments)} segments, {total_words} words")

    return segments


def _is_gpt4o_model(model: str) -> bool:
    """Check if this is a GPT-4o transcription model."""
    return model.startswith("gpt-4o")


def _is_diarize_model(model: str) -> bool:
    """Check if this is a diarization model."""
    return "diarize" in model


def _transcribe_file(
    client,
    audio_path: Path,
    config: ASRConfig,
    total_duration: Optional[float] = None,
) -> List[ASRSegment]:
    """Transcribe a single audio file."""
    model = config.model

    with audio_path.open("rb") as f:
        if _is_diarize_model(model):
            # gpt-4o-transcribe-diarize: use diarized_json format
            # Requires chunking_strategy for audio > 30s
            response = client.audio.transcriptions.create(
                model=model,
                file=f,
                response_format="diarized_json",
                chunking_strategy="auto",  # Required for audio > 30s
            )
            return _parse_diarized_response(response)

        elif _is_gpt4o_model(model):
            # gpt-4o-transcribe / gpt-4o-mini-transcribe: only json or text
            # No timestamp_granularities support
            response = client.audio.transcriptions.create(
                model=model,
                file=f,
                response_format="json",
            )
            return _parse_gpt4o_response(response, offset=0.0, duration=total_duration)

        else:
            # whisper-1: supports verbose_json with timestamps
            response = client.audio.transcriptions.create(
                model=model,
                file=f,
                language=config.language,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"] if config.enable_word_timestamps else ["segment"],
                temperature=config.temperature,
            )
            return _parse_response(response, config.enable_word_timestamps)


def _transcribe_chunked(
    client,
    audio_path: Path,
    config: ASRConfig,
    total_duration: float,
) -> List[ASRSegment]:
    """Transcribe audio in chunks."""
    model = config.model

    # GPT-4o diarize model handles chunking internally, no need to chunk manually
    if _is_diarize_model(model):
        LOGGER.info(f"Using {model} with auto chunking for {total_duration:.1f}s audio")
        return _transcribe_file(client, audio_path, config)

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required for chunked transcription")

    data, sample_rate = sf.read(audio_path)

    # Ensure mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    chunk_samples = int(config.chunk_duration_seconds * sample_rate)
    total_samples = len(data)

    all_segments = []
    offset = 0.0

    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = data[start:end]

        # Convert chunk to file-like object
        buffer = io.BytesIO()
        sf.write(buffer, chunk, sample_rate, format="WAV")
        buffer.seek(0)
        buffer.name = "chunk.wav"

        # Transcribe chunk
        try:
            if _is_gpt4o_model(model):
                # gpt-4o-transcribe / gpt-4o-mini-transcribe
                response = client.audio.transcriptions.create(
                    model=model,
                    file=buffer,
                    response_format="json",
                )
                chunk_duration = (end - start) / sample_rate
                chunk_segments = _parse_gpt4o_response(response, offset, duration=chunk_duration)
            else:
                # whisper-1
                response = client.audio.transcriptions.create(
                    model=model,
                    file=buffer,
                    language=config.language,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"] if config.enable_word_timestamps else ["segment"],
                    temperature=config.temperature,
                )
                chunk_segments = _parse_response(response, config.enable_word_timestamps)

                # Adjust timestamps with offset for whisper-1
                for segment in chunk_segments:
                    segment.start += offset
                    segment.end += offset
                    if segment.words:
                        for word in segment.words:
                            word.t0 += offset
                            word.t1 += offset

            all_segments.extend(chunk_segments)

        except Exception as e:
            LOGGER.warning(f"Failed to transcribe chunk at {offset:.1f}s: {e}")

        offset = end / sample_rate
        progress = offset / total_duration * 100
        LOGGER.debug(f"Transcription progress: {progress:.1f}%")

    return all_segments


def _parse_response(response, include_words: bool) -> List[ASRSegment]:
    """Parse OpenAI API response into ASRSegment objects."""
    segments = []

    # Handle different response formats
    if hasattr(response, "segments"):
        raw_segments = response.segments
    elif isinstance(response, dict) and "segments" in response:
        raw_segments = response["segments"]
    else:
        # Single segment fallback
        raw_segments = [{
            "start": 0,
            "end": getattr(response, "duration", 0) or 0,
            "text": getattr(response, "text", "") or "",
        }]

    # Get word timings if available
    words_by_segment = {}
    if include_words:
        if hasattr(response, "words"):
            raw_words = response.words
        elif isinstance(response, dict) and "words" in response:
            raw_words = response["words"]
        else:
            raw_words = []

        # Map words to segments based on timing
        for word in raw_words:
            # Handle both object attributes and dict access
            if isinstance(word, dict):
                word_start = word.get("start", 0)
                word_end = word.get("end", 0)
                word_text = word.get("word", "")
            else:
                word_start = getattr(word, "start", 0)
                word_end = getattr(word, "end", 0)
                word_text = getattr(word, "word", "")

            # Find which segment this word belongs to
            for i, seg in enumerate(raw_segments):
                seg_start = getattr(seg, "start", None) or seg.get("start", 0)
                seg_end = getattr(seg, "end", None) or seg.get("end", 0)

                if seg_start <= word_start < seg_end:
                    if i not in words_by_segment:
                        words_by_segment[i] = []
                    words_by_segment[i].append(WordTiming(
                        w=word_text,
                        t0=word_start,
                        t1=word_end,
                    ))
                    break

    for i, seg in enumerate(raw_segments):
        start = getattr(seg, "start", None) or seg.get("start", 0)
        end = getattr(seg, "end", None) or seg.get("end", 0)
        text = getattr(seg, "text", None) or seg.get("text", "")

        # Get confidence if available
        confidence = None
        if hasattr(seg, "avg_logprob"):
            # Convert log probability to confidence (0-1)
            import math
            confidence = math.exp(seg.avg_logprob)
        elif isinstance(seg, dict) and "avg_logprob" in seg:
            import math
            confidence = math.exp(seg["avg_logprob"])

        segment = ASRSegment(
            start=start,
            end=end,
            text=text.strip(),
            confidence=confidence,
            words=words_by_segment.get(i),
        )
        segments.append(segment)

    return segments


def _parse_gpt4o_response(
    response,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> List[ASRSegment]:
    """Parse GPT-4o transcribe response into ASRSegment objects.

    GPT-4o models only return text, no timestamps or segments.
    We create a single segment for the whole chunk.
    """
    text = getattr(response, "text", "") or ""
    if isinstance(response, dict):
        text = response.get("text", "")

    if not text.strip():
        return []

    # For chunked audio, we estimate timing based on offset/duration
    # This is approximate since GPT-4o doesn't provide timestamps
    if duration is None or duration <= 0:
        # Roughly 150 words/minute -> ~2.5 words/sec
        word_count = len(text.split())
        duration = max(1.5, word_count / 2.5)

    end_time = offset + duration

    segment = ASRSegment(
        start=offset,
        end=end_time,
        text=text.strip(),
        confidence=None,
        words=None,
    )
    return [segment]


def _parse_diarized_response(response) -> List[ASRSegment]:
    """Parse GPT-4o transcribe-diarize response into ASRSegment objects.

    The diarized_json format returns segments with speaker labels:
    {
        "segments": [
            {"speaker": "SPEAKER_01", "text": "...", "start": 0.0, "end": 5.0},
            ...
        ]
    }
    """
    segments = []

    # Handle different response formats
    if hasattr(response, "segments"):
        raw_segments = response.segments
    elif isinstance(response, dict) and "segments" in response:
        raw_segments = response["segments"]
    else:
        # Fallback to text only
        text = getattr(response, "text", "") or ""
        if isinstance(response, dict):
            text = response.get("text", "")
        if text:
            return [ASRSegment(start=0, end=0, text=text.strip())]
        return []

    for seg in raw_segments:
        start = getattr(seg, "start", None) or seg.get("start", 0)
        end = getattr(seg, "end", None) or seg.get("end", 0)
        text = getattr(seg, "text", None) or seg.get("text", "")
        speaker = getattr(seg, "speaker", None) or seg.get("speaker", "")

        segment = ASRSegment(
            start=start,
            end=end,
            text=text.strip(),
            confidence=None,
            words=None,
            speaker=speaker or None,
        )
        segments.append(segment)

    return segments


def _save_asr_manifest(segments: List[ASRSegment], output_path: Path):
    """Save ASR segments to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for segment in segments:
            f.write(segment.model_dump_json() + "\n")


def _load_asr_manifest(manifest_path: Path) -> List[ASRSegment]:
    """Load ASR segments from JSONL file."""
    segments = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                segments.append(ASRSegment(**data))
    return segments


def check_api_available() -> bool:
    """Check if OpenAI API is available."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def transcribe_with_diarization(
    audio_path: Path,
    output_path: Path,
    known_speaker_names: Optional[List[str]] = None,
    known_speaker_audio_paths: Optional[List[Path]] = None,
    overwrite: bool = False,
) -> List[ASRSegment]:
    """Transcribe audio with speaker diarization using gpt-4o-transcribe-diarize.

    This is a unified transcription+diarization function that can replace
    both pyannote diarization and whisper transcription in one API call.

    Args:
        audio_path: Path to input audio file
        output_path: Path for ASR manifest (JSONL)
        known_speaker_names: Optional list of known speaker names (up to 4)
        known_speaker_audio_paths: Optional list of audio reference files for known speakers
        overwrite: Whether to overwrite existing results

    Returns:
        List of ASRSegment objects with speaker labels
    """
    import base64

    if output_path.exists() and not overwrite:
        LOGGER.info(f"ASR+diarization results already exist: {output_path}")
        return _load_asr_manifest(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = _get_openai_client()

    LOGGER.info(f"Transcribing audio with diarization: {audio_path}")

    # Prepare known speaker references if provided
    extra_body = {}
    if known_speaker_names and known_speaker_audio_paths:
        # Limit to 4 speakers as per OpenAI docs
        names = known_speaker_names[:4]
        refs = []
        for path in known_speaker_audio_paths[:4]:
            if Path(path).exists():
                with open(path, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode("utf-8")
                    # Determine MIME type
                    suffix = Path(path).suffix.lower()
                    mime = {
                        ".wav": "audio/wav",
                        ".mp3": "audio/mpeg",
                        ".m4a": "audio/mp4",
                    }.get(suffix, "audio/wav")
                    refs.append(f"data:{mime};base64,{audio_data}")

        if refs:
            extra_body = {
                "known_speaker_names": names,
                "known_speaker_references": refs,
            }
            LOGGER.info(f"Using {len(refs)} known speaker references")

    with audio_path.open("rb") as f:
        if extra_body:
            response = client.audio.transcriptions.create(
                model="gpt-4o-transcribe-diarize",
                file=f,
                response_format="diarized_json",
                chunking_strategy="auto",
                extra_body=extra_body,
            )
        else:
            response = client.audio.transcriptions.create(
                model="gpt-4o-transcribe-diarize",
                file=f,
                response_format="diarized_json",
                chunking_strategy="auto",
            )

    segments = _parse_diarized_response(response)

    # Save manifest
    _save_asr_manifest(segments, output_path)

    speaker_count = len(set(s.speaker for s in segments if s.speaker))
    LOGGER.info(f"Transcription+diarization complete: {len(segments)} segments, {speaker_count} speakers")

    return segments
