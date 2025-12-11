"""AI-powered pipeline diagnostics.

Uses OpenAI or Anthropic to analyze pipeline failures and generate
human-readable explanations with actionable recommendations.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

# Model and provider configuration via environment variables
# Set these in your .env or shell:
#   DIAGNOSTIC_PROVIDER=anthropic|openai
#   ANTHROPIC_DIAGNOSTIC_MODEL=claude-opus-4-5
#   OPENAI_DIAGNOSTIC_MODEL=gpt-4o
#   ANTHROPIC_API_KEY=sk-ant-...
#   OPENAI_API_KEY=sk-...

def _get_provider() -> str:
    """Get preferred AI provider from env."""
    return os.environ.get("DIAGNOSTIC_PROVIDER", "anthropic")

def _get_anthropic_model() -> str:
    """Get Anthropic model from env."""
    return os.environ.get("ANTHROPIC_DIAGNOSTIC_MODEL", "claude-sonnet-4-20250514")

def _get_openai_model() -> str:
    """Get OpenAI model from env."""
    return os.environ.get("OPENAI_DIAGNOSTIC_MODEL", "gpt-4o")


def _get_openai_client():
    """Get OpenAI client.

    Requires OPENAI_API_KEY environment variable.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for OpenAI diagnostics. "
            "Get your API key from https://platform.openai.com/api-keys"
        )

    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError as e:
        raise ImportError(
            "openai package is required for OpenAI diagnostics. "
            "Install with: pip install openai"
        ) from e


def _get_anthropic_client():
    """Get Anthropic client.

    Requires ANTHROPIC_API_KEY environment variable.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required for Anthropic diagnostics. "
            "Get your API key from https://console.anthropic.com/"
        )

    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError as e:
        raise ImportError(
            "anthropic package is required for Claude diagnostics. "
            "Install with: pip install anthropic"
        ) from e


def _build_diagnostic_prompt(
    diagnostics: Dict[str, Any],
    config_thresholds: Dict[str, Any],
) -> str:
    """Build the prompt for GPT-4o to analyze pipeline issues."""

    # Determine expected stage based on what failed
    stage_failed = diagnostics.get("stage_failed", "unknown")
    stage_reached = diagnostics.get("stage_reached", "unknown")

    prompt = f"""You are an expert in face detection and recognition ML pipelines.
Analyze this pipeline failure and explain why the track wasn't processed correctly.

## Track Diagnostic Data
```json
{json.dumps(diagnostics, indent=2)}
```

## Pipeline Configuration Thresholds
```json
{json.dumps(config_thresholds, indent=2)}
```

## Pipeline Stages
1. **Detection**: RetinaFace detects faces in video frames
2. **Tracking**: ByteTrack links detections across frames into tracks
3. **Embedding**: ArcFace generates 512-d embeddings for high-quality faces
4. **Clustering**: Agglomerative clustering groups tracks by face similarity

## Analysis Task
The track reached stage "{stage_reached}" but failed at "{stage_failed}".

1. **Explain WHY** this track wasn't successfully processed to completion
2. **Identify the specific threshold** that blocked it (if applicable)
3. **Recommend config changes** with specific values
4. **Suggest manual fixes** if config changes won't help

## Response Format
Respond ONLY with a valid JSON object (no markdown, no explanation outside JSON):
{{
  "explanation": "Clear 2-3 sentence explanation a non-technical user could understand",
  "root_cause": "Brief technical cause in 5-10 words",
  "blocked_by": "stage:threshold_name (e.g., embedding:min_blur_score)",
  "suggested_fixes": [
    "First fix option",
    "Second fix option"
  ],
  "config_changes": [
    {{
      "file": "config/pipeline/filename.yaml",
      "key": "threshold_name",
      "current": 18.0,
      "suggested": 10.0,
      "reason": "Why this change helps"
    }}
  ]
}}
"""
    return prompt


def _call_anthropic(client, model: str, prompt: str) -> Optional[str]:
    """Call Anthropic Claude API."""
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt},
        ],
        system="You are a pipeline diagnostics expert. Always respond with valid JSON only.",
    )

    if response.content and len(response.content) > 0:
        return response.content[0].text
    return None


def _call_openai(client, model: str, prompt: str) -> Optional[str]:
    """Call OpenAI API."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a pipeline diagnostics expert. Always respond with valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    if response.choices and len(response.choices) > 0:
        return response.choices[0].message.content
    return None


def analyze_pipeline_issues(
    diagnostics: Dict[str, Any],
    config_thresholds: Dict[str, Any],
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Call AI to analyze pipeline issues and generate explanation.

    Args:
        diagnostics: Track diagnostic data (from collect_track_diagnostics)
        config_thresholds: Pipeline configuration thresholds
        model: Model to use (overrides env var if provided)
        provider: "anthropic" or "openai" (overrides env var if provided)

    Returns:
        AI analysis dict with explanation, root_cause, suggested_fixes, config_changes
    """
    provider = provider or _get_provider()

    prompt = _build_diagnostic_prompt(diagnostics, config_thresholds)
    content = None

    # Try Anthropic first if preferred
    if provider == "anthropic":
        anthropic_model = model or _get_anthropic_model()
        try:
            client = _get_anthropic_client()
            LOGGER.info(f"Using Anthropic {anthropic_model} for diagnostics")
            content = _call_anthropic(client, anthropic_model, prompt)
        except (ValueError, ImportError) as e:
            LOGGER.warning(f"Anthropic not available: {e}, trying OpenAI...")
            provider = "openai"  # Fallback to OpenAI

    # Try OpenAI
    if provider == "openai" or content is None:
        openai_model = model or _get_openai_model()
        try:
            client = _get_openai_client()
            LOGGER.info(f"Using OpenAI {openai_model} for diagnostics")
            content = _call_openai(client, openai_model, prompt)
        except (ValueError, ImportError) as e:
            LOGGER.warning(f"OpenAI not available: {e}")
            return _generate_fallback_analysis(diagnostics, config_thresholds)

    if not content:
        LOGGER.warning("Empty response from AI provider")
        return _generate_fallback_analysis(diagnostics, config_thresholds)

    try:

        # Parse JSON response
        # Strip markdown code blocks if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        analysis = json.loads(content)

        LOGGER.info(
            f"AI analysis complete for track {diagnostics.get('track_id')}: "
            f"blocked_by={analysis.get('blocked_by')}"
        )

        return analysis

    except json.JSONDecodeError as e:
        LOGGER.warning(f"Failed to parse OpenAI response as JSON: {e}")
        return _generate_fallback_analysis(diagnostics, config_thresholds)
    except Exception as e:
        LOGGER.error(f"OpenAI API error: {e}")
        return _generate_fallback_analysis(diagnostics, config_thresholds)


def _generate_fallback_analysis(
    diagnostics: Dict[str, Any],
    config_thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate rule-based fallback analysis when OpenAI is unavailable.

    This provides basic diagnostics without AI, based on threshold comparisons.
    """
    stage_failed = diagnostics.get("stage_failed", "unknown")
    stage_reached = diagnostics.get("stage_reached", "unknown")
    raw_data = diagnostics.get("raw_data", {})

    # Extract relevant data
    faces_count = raw_data.get("faces_count", 0)
    faces_skipped = raw_data.get("faces_skipped", 0)
    faces_with_embeddings = raw_data.get("faces_with_embeddings", 0)
    max_blur = raw_data.get("max_blur_score")
    skip_reasons = raw_data.get("skip_reasons", [])

    # Get thresholds
    embed_config = config_thresholds.get("embedding", {})
    min_blur_threshold = embed_config.get("min_blur_score", 18.0)
    cluster_config = config_thresholds.get("clustering", {})
    cluster_thresh = cluster_config.get("cluster_thresh", 0.52)

    # Determine cause based on stage
    if stage_failed == "embedded":
        # Failed at embedding stage
        if faces_count == 0:
            explanation = f"Track has no faces recorded. This usually means detection or tracking failed for this region."
            root_cause = "No faces detected in track"
            blocked_by = "detection:confidence"
            suggested_fixes = [
                "Check if the face was visible and not occluded",
                "Lower detection confidence threshold if face is partial",
            ]
            config_changes = []
        elif faces_skipped == faces_count:
            # All faces were skipped
            primary_reason = "quality"
            if skip_reasons:
                first_reason = skip_reasons[0].split(":")[0] if ":" in skip_reasons[0] else skip_reasons[0]
                primary_reason = first_reason

            if "blurry" in primary_reason.lower() or "blur" in primary_reason.lower():
                blur_info = f" (max blur: {max_blur:.1f}, threshold: {min_blur_threshold})" if max_blur else ""
                explanation = f"All {faces_count} face(s) were too blurry to generate embeddings{blur_info}. The face quality didn't meet the minimum blur score threshold."
                root_cause = "All faces below blur threshold"
                blocked_by = "embedding:min_blur_score"
                suggested = max(5.0, (max_blur - 2.0)) if max_blur else 10.0
                config_changes = [{
                    "file": "config/pipeline/faces_embed_sampling.yaml",
                    "key": "quality_gating.min_blur_score",
                    "current": min_blur_threshold,
                    "suggested": suggested,
                    "reason": "Allow blurrier faces for short tracks with motion blur",
                }]
            else:
                explanation = f"All {faces_count} face(s) were skipped due to quality filtering. Primary reason: {primary_reason}."
                root_cause = f"Quality gate: {primary_reason}"
                blocked_by = f"embedding:{primary_reason}"
                config_changes = []

            suggested_fixes = [
                "Lower quality thresholds in faces_embed_sampling.yaml",
                "Manually assign track in Faces Review UI",
                "Accept that very low quality faces may not be processable",
            ]
        else:
            explanation = f"Track has {faces_count} face(s) but no embeddings were generated. This may be a pipeline error or all faces failed quality checks."
            root_cause = "Embedding generation failed"
            blocked_by = "embedding:unknown"
            suggested_fixes = [
                "Check pipeline logs for embedding errors",
                "Re-run embedding stage for this episode",
            ]
            config_changes = []

    elif stage_failed == "clustered":
        # Failed at clustering stage
        if faces_with_embeddings == 0:
            explanation = f"Track has embeddings marked but clustering couldn't process it. The track may have been an outlier."
            root_cause = "No valid embeddings for clustering"
            blocked_by = "clustering:no_embeddings"
        else:
            explanation = f"Track has {faces_with_embeddings} embedding(s) but wasn't assigned to any cluster. It may be an outlier below the similarity threshold, or a new person not matching any existing cluster."
            root_cause = "Below cluster similarity threshold"
            blocked_by = "clustering:min_identity_sim"

        suggested_fixes = [
            "Lower clustering threshold to accept looser matches",
            "Manually assign track to existing identity in Faces Review",
            "Create new identity for this track if it's a new person",
        ]
        config_changes = [{
            "file": "config/pipeline/clustering.yaml",
            "key": "cluster_thresh",
            "current": cluster_thresh,
            "suggested": max(0.40, cluster_thresh - 0.08),
            "reason": "Lower threshold allows looser cluster membership",
        }]

    elif stage_failed == "tracked":
        explanation = f"Face was detected but not assigned to a track. This usually happens with single-frame detections or when tracking loses the face quickly."
        root_cause = "Tracking lost face across frames"
        blocked_by = "tracking:track_buffer"
        suggested_fixes = [
            "Increase track buffer to maintain tracks longer",
            "Lower tracking threshold for new tracks",
        ]
        config_changes = [{
            "file": "config/pipeline/tracking.yaml",
            "key": "track_buffer",
            "current": 90,
            "suggested": 180,
            "reason": "Maintain tracks longer across occlusions",
        }]

    else:
        explanation = f"Track reached stage '{stage_reached}' but failed at '{stage_failed}'. Unable to determine specific cause."
        root_cause = "Unknown pipeline failure"
        blocked_by = "unknown:unknown"
        suggested_fixes = [
            "Check pipeline logs for errors",
            "Re-run pipeline stages for this episode",
        ]
        config_changes = []

    return {
        "explanation": explanation,
        "root_cause": root_cause,
        "blocked_by": blocked_by,
        "suggested_fixes": suggested_fixes,
        "config_changes": config_changes,
        "_fallback": True,  # Indicates this was rule-based, not AI
    }


def is_openai_available() -> bool:
    """Check if OpenAI API is available and configured."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False

    try:
        from openai import OpenAI
        return True
    except ImportError:
        return False
