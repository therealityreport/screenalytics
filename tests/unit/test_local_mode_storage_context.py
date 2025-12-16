from __future__ import annotations

import os

from tools import episode_run


def test_storage_context_defaults_to_local_in_local_mode(monkeypatch) -> None:
    """Local-mode runs should not default to S3 uploads unless explicitly configured."""
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)
    monkeypatch.setattr(episode_run, "LOCAL_MODE_INSTRUMENTATION", True)

    storage, ep_ctx, prefixes = episode_run._storage_context("ep123")

    assert storage is None
    assert ep_ctx is None
    assert prefixes is None

    # Ensure we didn't mutate global env in the helper.
    assert os.environ.get("STORAGE_BACKEND") is None

