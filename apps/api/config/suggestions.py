"""Smart Suggestions threshold configuration.

Single source of truth for all similarity thresholds used in Smart Suggestions,
grouping, and related UI components.

Environment Variables:
    SUGGESTION_CAST_HIGH: High confidence threshold for cast suggestions (default: 0.68)
    SUGGESTION_CAST_MEDIUM: Medium confidence threshold (default: 0.50)
    SUGGESTION_AUTO_ASSIGN: Auto-assign threshold (default: 0.85)
    CLUSTER_MERGE_HIGH: High confidence for cluster merging (default: 0.90)
    CLUSTER_MERGE_CANDIDATE: Candidate threshold for merging (default: 0.85)
    CLUSTER_SIMILAR: Similarity threshold for grouping (default: 0.70)
    SEED_MATCH_THRESHOLD: Seed matching threshold (default: 0.50)
    GROUP_WITHIN_EP_DISTANCE: Within-episode grouping distance (default: 0.42)
    PEOPLE_MATCH_DISTANCE: Cross-episode people matching distance (default: 0.40)
    SEED_CLUSTER_DELTA: Seed cluster delta bonus (default: 0.08)
    API_TIMEOUT_DEFAULT: Default API timeout in seconds (default: 30)
    API_TIMEOUT_HEAVY: Heavy operation timeout (default: 60)
    API_TIMEOUT_FAST: Fast operation timeout (default: 15)
    API_BASE_URL: Base URL for API (default: http://localhost:8000)

Quality Gate Environment Variables:
    QUALITY_SHARPNESS_STRICT: Strict quality gate threshold (default: 100.0)
    QUALITY_SHARPNESS_BALANCED: Balanced quality gate threshold (default: 50.0)
    QUALITY_SHARPNESS_INCLUSIVE: Inclusive/rescue quality gate threshold (default: 15.0)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Cast/Facebank Similarity Thresholds
# =============================================================================

SUGGESTION_THRESHOLDS = {
    # Confidence tiers for cast suggestions:
    # - Auto-accept: ≥95% - very strong match, can auto-assign
    # - HIGH: 65-94% - strong match, likely correct
    # - MEDIUM: 35-64% - reasonable match, needs review
    # - LOW: 15-34% - weak match, review carefully
    # - REST: <15% - very weak, shown separately
    "cast_auto_assign": float(os.getenv("SUGGESTION_AUTO_ASSIGN", "0.95")),
    "cast_high": float(os.getenv("SUGGESTION_CAST_HIGH", "0.65")),
    "cast_medium": float(os.getenv("SUGGESTION_CAST_MEDIUM", "0.35")),
    "cast_low": float(os.getenv("SUGGESTION_CAST_LOW", "0.15")),

    # Cluster grouping thresholds
    # High confidence for auto-merging similar clusters
    "cluster_merge_high": float(os.getenv("CLUSTER_MERGE_HIGH", "0.90")),
    # Candidate threshold for potential duplicate detection
    "cluster_merge_candidate": float(os.getenv("CLUSTER_MERGE_CANDIDATE", "0.85")),
    # General similarity threshold for grouping unassigned clusters
    "cluster_similar": float(os.getenv("CLUSTER_SIMILAR", "0.70")),

    # Seed confidence threshold - % of faces that must match seed for seed-based matching
    "seed_match": float(os.getenv("SEED_MATCH_THRESHOLD", "0.50")),
}


# =============================================================================
# Grouping Distance Thresholds
# =============================================================================

GROUPING_THRESHOLDS = {
    # Maximum cosine distance for grouping clusters within the same episode
    # Lower = stricter matching, higher = more lenient
    "within_episode": float(os.getenv("GROUP_WITHIN_EP_DISTANCE", "0.42")),

    # Maximum cosine distance for matching clusters across episodes to existing people
    # Kept strict for precision in cross-episode identity matching
    "cross_episode": float(os.getenv("PEOPLE_MATCH_DISTANCE", "0.40")),

    # Bonus delta applied to seed matches - reduces effective distance threshold
    # when cluster has a seed match, making it easier to match to the right person
    "seed_delta": float(os.getenv("SEED_CLUSTER_DELTA", "0.08")),

    # Prototype momentum - how much weight to give new data vs existing prototype
    # 0.8 = 80% existing, 20% new data
    "prototype_momentum": float(os.getenv("PEOPLE_PROTO_MOMENTUM", "0.8")),

    # Minimum embedding dimension for validation
    "min_embedding_dim": int(os.getenv("MIN_EMBEDDING_DIM", "128")),

    # Minimum faces for reliable matching (small clusters get distance penalty)
    "min_faces_reliable": max(2, int(os.getenv("MIN_FACES_FOR_RELIABLE_MATCH", "3"))),

    # Distance penalty applied to small clusters (fewer faces = less reliable)
    "small_cluster_penalty": float(os.getenv("SMALL_CLUSTER_DISTANCE_PENALTY", "0.05")),

    # Maximum cohesion bonus for high-cohesion clusters
    "cohesion_bonus_max": float(os.getenv("COHESION_BONUS_MAX", "0.05")),
}


# =============================================================================
# Quality Gate Profiles (Face Sharpness Thresholds)
# =============================================================================
# Laplacian variance threshold for face quality gate
# Higher = stricter (more faces skipped), Lower = more permissive (more faces pass)

QUALITY_PROFILES = {
    # Strict: Higher threshold, only crisp faces pass
    # Use when you want maximum embedding quality at cost of coverage
    "strict": {
        "sharpness_threshold": float(os.getenv("QUALITY_SHARPNESS_STRICT", "100.0")),
        "description": "Only crisp faces - maximum embedding quality",
        "warning": "Many faces may be skipped",
    },
    # Balanced: Default threshold, good balance of quality and coverage
    "balanced": {
        "sharpness_threshold": float(os.getenv("QUALITY_SHARPNESS_BALANCED", "50.0")),
        "description": "Default - balance of quality and coverage",
        "warning": None,
    },
    # Inclusive: Lower threshold, allows borderline blurry faces through
    # Use for "rescue" operations on quality-only clusters
    "inclusive": {
        "sharpness_threshold": float(os.getenv("QUALITY_SHARPNESS_INCLUSIVE", "15.0")),
        "description": "Rescue mode - allows borderline blurry faces",
        "warning": "Embeddings may be less reliable for matching",
    },
    # Bypass: No quality gate - use with extreme caution
    "bypass": {
        "sharpness_threshold": 0.0,
        "description": "No quality filtering - all faces pass",
        "warning": "Very noisy embeddings - use only for desperate rescue",
    },
}

# Default profile used for normal operations
DEFAULT_QUALITY_PROFILE = os.getenv("DEFAULT_QUALITY_PROFILE", "balanced")


def get_quality_profile(profile_name: str | None = None) -> Dict[str, Any]:
    """Get quality gate profile by name.

    Args:
        profile_name: Profile name ("strict", "balanced", "inclusive", "bypass")
                      If None, uses DEFAULT_QUALITY_PROFILE

    Returns:
        Profile dict with sharpness_threshold, description, warning
    """
    name = profile_name or DEFAULT_QUALITY_PROFILE
    profile = QUALITY_PROFILES.get(name)
    if not profile:
        LOGGER.warning(f"Unknown quality profile '{name}', using balanced")
        profile = QUALITY_PROFILES["balanced"]
    return {"name": name, **profile}


def get_sharpness_threshold(profile_name: str | None = None) -> float:
    """Get sharpness threshold for a quality profile.

    Args:
        profile_name: Profile name or None for default

    Returns:
        Sharpness threshold value
    """
    return get_quality_profile(profile_name)["sharpness_threshold"]


# =============================================================================
# Timeout Configuration
# =============================================================================

TIMEOUTS = {
    # Default timeout for most API calls
    "api_default": int(os.getenv("API_TIMEOUT_DEFAULT", "30")),

    # Timeout for heavy/slow operations (large episodes, bulk operations)
    "api_heavy": int(os.getenv("API_TIMEOUT_HEAVY", "60")),

    # Timeout for fast operations (simple lookups, health checks)
    "api_fast": int(os.getenv("API_TIMEOUT_FAST", "15")),
}


# =============================================================================
# API Configuration
# =============================================================================

# Base URL for API - used by UI components
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    os.getenv("SMART_SUGGESTIONS_API_BASE", "http://localhost:8000")
)


# =============================================================================
# Helper Functions
# =============================================================================

def get_confidence_label(similarity: float) -> str:
    """Get human-readable confidence label for a similarity score.

    Args:
        similarity: Cosine similarity value (0.0 to 1.0)

    Returns:
        "auto", "high", "medium", "low", or "rest" based on thresholds
        - auto: ≥95% - can auto-assign
        - high: 65-94% - strong match
        - medium: 35-64% - reasonable match
        - low: 15-34% - weak match
        - rest: <15% - very weak
    """
    if similarity >= SUGGESTION_THRESHOLDS["cast_auto_assign"]:
        return "auto"
    elif similarity >= SUGGESTION_THRESHOLDS["cast_high"]:
        return "high"
    elif similarity >= SUGGESTION_THRESHOLDS["cast_medium"]:
        return "medium"
    elif similarity >= SUGGESTION_THRESHOLDS["cast_low"]:
        return "low"
    return "rest"


def get_confidence_color(similarity: float) -> str:
    """Get color for confidence level.

    Args:
        similarity: Cosine similarity value (0.0 to 1.0)

    Returns:
        Hex color code
    """
    label = get_confidence_label(similarity)
    colors = {
        "high": "#4CAF50",    # Green
        "medium": "#FF9800",  # Orange
        "low": "#F44336",     # Red
    }
    return colors.get(label, "#9E9E9E")


def get_threshold_description(key: str) -> str:
    """Get human-readable description of a threshold.

    Args:
        key: Threshold key (e.g., "cast_high", "cast_medium")

    Returns:
        Human-readable string like "≥68%"
    """
    value = SUGGESTION_THRESHOLDS.get(key)
    if value is None:
        value = GROUPING_THRESHOLDS.get(key)
    if value is None:
        return "Unknown"
    return f"≥{int(value * 100)}%"


def get_all_thresholds() -> Dict[str, Any]:
    """Get all thresholds as a single dictionary for API exposure.

    Returns:
        Dictionary containing all threshold values and metadata
    """
    return {
        "suggestion": {
            "cast_auto_assign": SUGGESTION_THRESHOLDS["cast_auto_assign"],
            "cast_high": SUGGESTION_THRESHOLDS["cast_high"],
            "cast_high_label": get_threshold_description("cast_high"),
            "cast_medium": SUGGESTION_THRESHOLDS["cast_medium"],
            "cast_medium_label": get_threshold_description("cast_medium"),
            "cast_low": SUGGESTION_THRESHOLDS["cast_low"],
            "cast_low_label": get_threshold_description("cast_low"),
            "cluster_merge_high": SUGGESTION_THRESHOLDS["cluster_merge_high"],
            "cluster_merge_candidate": SUGGESTION_THRESHOLDS["cluster_merge_candidate"],
            "cluster_similar": SUGGESTION_THRESHOLDS["cluster_similar"],
            "seed_match": SUGGESTION_THRESHOLDS["seed_match"],
        },
        "grouping": {
            "within_episode": GROUPING_THRESHOLDS["within_episode"],
            "cross_episode": GROUPING_THRESHOLDS["cross_episode"],
            "seed_delta": GROUPING_THRESHOLDS["seed_delta"],
            "prototype_momentum": GROUPING_THRESHOLDS["prototype_momentum"],
            "min_faces_reliable": GROUPING_THRESHOLDS["min_faces_reliable"],
            "small_cluster_penalty": GROUPING_THRESHOLDS["small_cluster_penalty"],
        },
        "timeouts": TIMEOUTS,
        "api_base_url": API_BASE_URL,
    }


# =============================================================================
# Threshold Override Persistence
# =============================================================================

def _get_overrides_path() -> Path:
    """Get path to threshold overrides config file."""
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
    config_dir = data_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "threshold_overrides.json"


def load_threshold_overrides() -> Dict[str, Any]:
    """Load threshold overrides from disk.

    Returns:
        Dict with any user-configured threshold overrides
    """
    path = _get_overrides_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("[thresholds] Failed to load overrides from %s: %s", path, exc)
        return {}


def save_threshold_overrides(overrides: Dict[str, Any]) -> bool:
    """Save threshold overrides to disk.

    Args:
        overrides: Dict with threshold values to override

    Returns:
        True if saved successfully
    """
    path = _get_overrides_path()
    try:
        path.write_text(json.dumps(overrides, indent=2), encoding="utf-8")
        LOGGER.info("[thresholds] Saved overrides to %s", path)
        return True
    except OSError as exc:
        LOGGER.error("[thresholds] Failed to save overrides to %s: %s", path, exc)
        return False


def get_effective_thresholds() -> Dict[str, Any]:
    """Get thresholds with any user overrides applied.

    Overrides take precedence over environment variables,
    which take precedence over defaults.

    Returns:
        Dict with all effective threshold values
    """
    overrides = load_threshold_overrides()

    # Build effective thresholds - overrides > env > defaults
    suggestion = overrides.get("suggestion", {})

    return {
        "suggestion": {
            "cast_auto_assign": suggestion.get("cast_auto_assign", SUGGESTION_THRESHOLDS["cast_auto_assign"]),
            "cast_high": suggestion.get("cast_high", SUGGESTION_THRESHOLDS["cast_high"]),
            "cast_high_label": f"≥{int(suggestion.get('cast_high', SUGGESTION_THRESHOLDS['cast_high']) * 100)}%",
            "cast_medium": suggestion.get("cast_medium", SUGGESTION_THRESHOLDS["cast_medium"]),
            "cast_medium_label": f"≥{int(suggestion.get('cast_medium', SUGGESTION_THRESHOLDS['cast_medium']) * 100)}%",
            "cast_low": suggestion.get("cast_low", SUGGESTION_THRESHOLDS["cast_low"]),
            "cast_low_label": f"≥{int(suggestion.get('cast_low', SUGGESTION_THRESHOLDS['cast_low']) * 100)}%",
            "cluster_merge_high": SUGGESTION_THRESHOLDS["cluster_merge_high"],
            "cluster_merge_candidate": SUGGESTION_THRESHOLDS["cluster_merge_candidate"],
            "cluster_similar": SUGGESTION_THRESHOLDS["cluster_similar"],
            "seed_match": SUGGESTION_THRESHOLDS["seed_match"],
        },
        "grouping": {
            "within_episode": GROUPING_THRESHOLDS["within_episode"],
            "cross_episode": GROUPING_THRESHOLDS["cross_episode"],
            "seed_delta": GROUPING_THRESHOLDS["seed_delta"],
            "prototype_momentum": GROUPING_THRESHOLDS["prototype_momentum"],
            "min_faces_reliable": GROUPING_THRESHOLDS["min_faces_reliable"],
            "small_cluster_penalty": GROUPING_THRESHOLDS["small_cluster_penalty"],
        },
        "timeouts": TIMEOUTS,
        "api_base_url": API_BASE_URL,
        "has_overrides": bool(overrides),
    }


def update_suggestion_thresholds(
    cast_high: Optional[float] = None,
    cast_medium: Optional[float] = None,
    cast_auto_assign: Optional[float] = None,
) -> Dict[str, Any]:
    """Update suggestion thresholds with validation.

    Args:
        cast_high: High confidence threshold (0.5-1.0)
        cast_medium: Medium confidence threshold (0.3-cast_high)
        cast_auto_assign: Auto-assign threshold (cast_high-1.0)

    Returns:
        Dict with updated effective thresholds

    Raises:
        ValueError: If thresholds are invalid
    """
    overrides = load_threshold_overrides()
    suggestion = overrides.get("suggestion", {})

    # Get current effective values
    current_high = suggestion.get("cast_high", SUGGESTION_THRESHOLDS["cast_high"])
    current_medium = suggestion.get("cast_medium", SUGGESTION_THRESHOLDS["cast_medium"])
    current_auto = suggestion.get("cast_auto_assign", SUGGESTION_THRESHOLDS["cast_auto_assign"])

    # Apply updates
    new_high = cast_high if cast_high is not None else current_high
    new_medium = cast_medium if cast_medium is not None else current_medium
    new_auto = cast_auto_assign if cast_auto_assign is not None else current_auto

    # Validate ranges
    if not (0.5 <= new_high <= 1.0):
        raise ValueError(f"cast_high must be between 0.5 and 1.0, got {new_high}")
    if not (0.3 <= new_medium <= new_high):
        raise ValueError(f"cast_medium must be between 0.3 and cast_high ({new_high}), got {new_medium}")
    if not (new_high <= new_auto <= 1.0):
        raise ValueError(f"cast_auto_assign must be between cast_high ({new_high}) and 1.0, got {new_auto}")

    # Update overrides
    suggestion["cast_high"] = new_high
    suggestion["cast_medium"] = new_medium
    suggestion["cast_auto_assign"] = new_auto
    overrides["suggestion"] = suggestion

    if not save_threshold_overrides(overrides):
        raise ValueError("Failed to save threshold overrides")

    return get_effective_thresholds()


def reset_suggestion_thresholds() -> Dict[str, Any]:
    """Reset suggestion thresholds to defaults (remove overrides).

    Returns:
        Dict with default thresholds
    """
    overrides = load_threshold_overrides()
    if "suggestion" in overrides:
        del overrides["suggestion"]
    save_threshold_overrides(overrides)
    return get_effective_thresholds()
