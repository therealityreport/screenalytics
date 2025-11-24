"""Filesystem layout helpers for Screenalytics artifacts.

The resolver mirrors the object-storage hierarchy locally so CLI tools and
pipelines can agree on canonical paths. Paths are rooted at
`SCREENALYTICS_DATA_ROOT` (defaults to `./data`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

_DEFAULT_ROOT = Path("data")


def _data_root() -> Path:
    raw = os.environ.get("SCREENALYTICS_DATA_ROOT")
    base = Path(raw).expanduser() if raw else _DEFAULT_ROOT
    return base


def get_path(ep_id: str, kind: str) -> Path:
    """Return the canonical path for a given artifact kind."""
    ep = str(ep_id)
    root = _data_root()
    kind_map: Dict[str, Path] = {
        "video": root / "videos" / ep / "episode.mp4",
        "detections": root / "manifests" / ep / "detections.jsonl",
        "tracks": root / "manifests" / ep / "tracks.jsonl",
        "frames_root": root / "frames" / ep,
    }
    key = kind.lower()
    if key not in kind_map:
        raise ValueError(f"Unknown artifact kind '{kind}'")
    return kind_map[key]


def ensure_dirs(ep_id: str) -> None:
    """Create parent directories for all artifacts associated with an episode."""
    video_parent = get_path(ep_id, "video").parent
    manifests_parent = get_path(ep_id, "detections").parent
    tracks_parent = get_path(ep_id, "tracks").parent
    frames_root = get_path(ep_id, "frames_root")

    for path in {video_parent, manifests_parent, tracks_parent, frames_root}:
        path.mkdir(parents=True, exist_ok=True)
