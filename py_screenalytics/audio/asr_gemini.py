"""Gemini ASR integration.

Handles:
- Speech-to-text using Gemini API
- Transcript cleanup and enrichment
- Alternative ASR provider
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from .models import ASRConfig, ASRSegment, WordTiming

LOGGER = logging.getLogger(__name__)


def _get_gemini_client():
    """Get Gemini client."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required for Gemini ASR. "
            "Get your API key from https://makersuite.google.com/app/apikey"
        )

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai
    except ImportError as e:
        raise ImportError(
            "google-generativeai package is required for Gemini ASR. "
            "Install with: pip install google-generativeai"
        ) from e


def transcribe_audio(
    audio_path: Path,
    output_path: Path,
    config: Optional[ASRConfig] = None,
    overwrite: bool = False,
) -> List[ASRSegment]:
    """Transcribe audio using Gemini API.

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

    genai = _get_gemini_client()

    LOGGER.info(f"Transcribing audio with Gemini: {audio_path}")

    # Read and encode audio
    with audio_path.open("rb") as f:
        audio_bytes = f.read()

    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    # Determine MIME type
    suffix = audio_path.suffix.lower()
    mime_type = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
    }.get(suffix, "audio/wav")

    # Create prompt for transcription
    prompt = f"""Transcribe this audio file. Output the transcription in JSON format with the following structure:
{{
    "segments": [
        {{
            "start": <start_time_in_seconds>,
            "end": <end_time_in_seconds>,
            "text": "<transcribed_text>",
            "words": [
                {{"w": "<word>", "t0": <start_time>, "t1": <end_time>}},
                ...
            ]
        }},
        ...
    ]
}}

Language: {config.language}
Include word-level timestamps: {config.enable_word_timestamps}

Please provide accurate timestamps and transcription."""

    # Call Gemini API
    model = genai.GenerativeModel(config.gemini_model)

    response = model.generate_content([
        {
            "mime_type": mime_type,
            "data": audio_base64,
        },
        prompt,
    ])

    # Parse response
    segments = _parse_gemini_response(response.text)

    # Save manifest
    _save_asr_manifest(segments, output_path)

    total_words = sum(len(s.words or []) for s in segments)
    LOGGER.info(f"Gemini transcription complete: {len(segments)} segments, {total_words} words")

    return segments


def cleanup_transcript(
    raw_segments: List[ASRSegment],
    config: Optional[ASRConfig] = None,
) -> List[ASRSegment]:
    """Use Gemini to clean up and repair a transcript.

    Args:
        raw_segments: Raw ASR segments (e.g., from Whisper)
        config: ASR configuration

    Returns:
        Cleaned ASR segments
    """
    config = config or ASRConfig()

    if not config.gemini_use_for_cleanup:
        return raw_segments

    genai = _get_gemini_client()

    LOGGER.info("Cleaning transcript with Gemini")

    # Prepare transcript text
    transcript_json = json.dumps([s.model_dump() for s in raw_segments], indent=2)

    prompt = f"""Review and clean up this transcript. Fix any:
- Spelling errors
- Grammar issues
- Speaker attribution errors
- Repeated phrases or stutters that should be cleaned
- Punctuation issues

Maintain the original timing information. Output in the same JSON format.

Original transcript:
{transcript_json}

Cleaned transcript:"""

    model = genai.GenerativeModel(config.gemini_model)
    response = model.generate_content(prompt)

    # Parse cleaned response
    try:
        cleaned_segments = _parse_gemini_response(response.text)
        if cleaned_segments:
            LOGGER.info(f"Transcript cleanup complete: {len(cleaned_segments)} segments")
            return cleaned_segments
    except Exception as e:
        LOGGER.warning(f"Failed to parse Gemini cleanup response: {e}")

    # Return original if cleanup fails
    return raw_segments


def _parse_gemini_response(response_text: str) -> List[ASRSegment]:
    """Parse Gemini API response into ASRSegment objects."""
    import re

    # Try to extract JSON from response
    text = response_text.strip()

    # Handle markdown code blocks - use regex for robustness
    # Match ```json or ``` followed by content and ending ```
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    code_matches = re.findall(code_block_pattern, text)
    if code_matches:
        # Use the first code block that contains valid JSON
        for match in code_matches:
            try:
                data = json.loads(match.strip())
                break
            except json.JSONDecodeError:
                continue
        else:
            # None of the code blocks were valid JSON, try the full text
            data = None
    else:
        data = None

    if data is None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object/array in response (non-greedy for nested objects)
            # Look for both object {...} and array [...]
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested object
                r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # Simple nested array
            ]
            for pattern in json_patterns:
                match = re.search(pattern, text)
                if match:
                    try:
                        data = json.loads(match.group())
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                LOGGER.error(f"Could not parse Gemini response: {text[:200]}...")
                return []

    segments = []
    raw_segments = data.get("segments", [data]) if isinstance(data, dict) else data

    for seg in raw_segments:
        words = None
        if seg.get("words"):
            words = [
                WordTiming(w=w.get("w", ""), t0=w.get("t0", 0), t1=w.get("t1", 0))
                for w in seg["words"]
            ]

        segment = ASRSegment(
            start=seg.get("start", 0),
            end=seg.get("end", 0),
            text=seg.get("text", "").strip(),
            confidence=seg.get("confidence"),
            words=words,
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
    """Check if Gemini API is available."""
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
