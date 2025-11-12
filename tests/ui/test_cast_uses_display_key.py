from __future__ import annotations

from tests.ui._helpers_loader import load_ui_helpers_module


def test_seed_display_source_prefers_display_key():
    helpers = load_ui_helpers_module()
    seed = {
        "display_s3_key": "artifacts/facebank/RHOBH/cast/seeds/seed_d.png",
        "image_s3_key": "artifacts/facebank/RHOBH/cast/legacy.jpg",
        "image_uri": "https://cdn.example/legacy.jpg",
    }
    assert helpers.seed_display_source(seed) == seed["display_s3_key"]


def test_seed_display_source_falls_back_to_image_uri():
    helpers = load_ui_helpers_module()
    seed = {"orig_s3_key": "artifacts/facebank/RHOBH/cast/seeds/seed_o.png"}
    assert helpers.seed_display_source(seed) == seed["orig_s3_key"]


def test_seed_display_source_last_resort_image_uri():
    helpers = load_ui_helpers_module()
    seed = {"image_uri": "https://cdn.example/fallback.png"}
    assert helpers.seed_display_source(seed) == seed["image_uri"]
