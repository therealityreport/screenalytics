from __future__ import annotations

import pytest

from apps.api.services.storage import artifact_prefixes, episode_context_from_id


def test_artifact_prefixes_v2_paths() -> None:
    ctx = episode_context_from_id("show-slug-s02e05")
    prefixes = artifact_prefixes(ctx)
    assert prefixes["frames"] == "artifacts/frames/show-slug/s02/e05/frames/"
    assert prefixes["crops"] == "artifacts/crops/show-slug/s02/e05/tracks/"
    assert prefixes["manifests"] == "artifacts/manifests/show-slug/s02/e05/"


def test_episode_context_requires_valid_format() -> None:
    with pytest.raises(ValueError):
        episode_context_from_id("invalid-format")
