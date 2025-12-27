from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from apps.api.services import faces_review_bundle as bundle_mod
from py_screenalytics import run_layout

RUN_ID = "Attempt1_2025-01-01_000000EST"
RUN_ID_2 = "Attempt2_2025-01-02_000000EST"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_run_artifacts(
    ep_id: str,
    run_id: str,
    identities: List[Dict[str, Any]],
    *,
    tracks: List[Dict[str, Any]] | None = None,
    faces: List[Dict[str, Any]] | None = None,
) -> None:
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "identities.json", {"ep_id": ep_id, "identities": identities})
    if tracks is not None:
        _write_jsonl(run_root / "tracks.jsonl", tracks)
    if faces is not None:
        _write_jsonl(run_root / "faces.jsonl", faces)


def _write_people(data_root: Path, show_id: str, people: List[Dict[str, Any]]) -> None:
    path = data_root / "shows" / show_id.upper() / "people.json"
    _write_json(path, {"show_id": show_id.upper(), "people": people})


def _write_cast(data_root: Path, show_id: str, cast: List[Dict[str, Any]]) -> None:
    path = data_root / "cast" / show_id.upper() / "cast.json"
    _write_json(path, {"show_id": show_id.upper(), "cast": cast})


def _write_archive(
    data_root: Path,
    show_id: str,
    *,
    clusters: List[Dict[str, Any]] | None = None,
    tracks: List[Dict[str, Any]] | None = None,
) -> None:
    payload = {
        "show_id": show_id.upper(),
        "archived_people": [],
        "archived_clusters": clusters or [],
        "archived_tracks": tracks or [],
        "stats": {},
    }
    path = data_root / "shows" / show_id.upper() / "archived.json"
    _write_json(path, payload)


def test_bundle_filters_by_cast_id_not_person_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "demo-show-s01e01"
    show_id = "DEMO-SHOW"
    _write_run_artifacts(
        ep_id,
        RUN_ID,
        identities=[{"identity_id": "c1", "track_ids": [1]}],
        tracks=[{"track_id": 1, "faces_count": 2}],
        faces=[{"track_id": 1, "frame_idx": 1}],
    )
    _write_people(
        tmp_path,
        show_id,
        [
            {
                "person_id": "p_0001",
                "name": "Person One",
                "cast_id": "cast_1",
                "cluster_ids": [f"{ep_id}:{RUN_ID}:c1"],
            }
        ],
    )
    _write_cast(tmp_path, show_id, [{"cast_id": "cast_1", "name": "Person One"}])

    def _fake_unlinked(self: bundle_mod.GroupingService, ep_id: str) -> Dict[str, Any]:
        return {
            "entities": [
                {
                    "entity_id": "p_0001",
                    "entity_type": "person",
                    "person": {"person_id": "p_0001", "cast_id": "cast_1"},
                    "cluster_ids": ["c1"],
                }
            ]
        }

    monkeypatch.setattr(bundle_mod.GroupingService, "list_unlinked_entities", _fake_unlinked)

    bundle = bundle_mod.build_faces_review_bundle(
        ep_id,
        RUN_ID,
        filter_cast_id="cast_1",
        data_root=tmp_path,
    )
    assert bundle["unlinked_entities"]
    assert bundle["unlinked_entities"][0]["entity_id"] == "p_0001"


def test_bundle_unlinked_entities_present_when_people_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "demo-show-s01e02"
    show_id = "DEMO-SHOW"
    _write_run_artifacts(
        ep_id,
        RUN_ID,
        identities=[{"identity_id": "c1", "track_ids": [1]}],
        tracks=[{"track_id": 1, "faces_count": 1}],
        faces=[{"track_id": 1, "frame_idx": 1}],
    )
    _write_people(tmp_path, show_id, [])

    bundle = bundle_mod.build_faces_review_bundle(ep_id, RUN_ID, data_root=tmp_path)
    assert bundle["unlinked_entities"]


def test_bundle_archived_parses_hyphenated_show_slug(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "real-housewives-ny-s01e01"
    show_id = "REAL-HOUSEWIVES-NY"
    _write_archive(
        tmp_path,
        show_id,
        clusters=[
            {"type": "cluster", "cluster_id": "c9", "episode_id": ep_id},
        ],
        tracks=[
            {"type": "track", "track_id": 7, "episode_id": ep_id},
        ],
    )

    bundle = bundle_mod.build_faces_review_bundle(ep_id, RUN_ID, data_root=tmp_path)
    archived = bundle["archived_ids"]
    assert "c9" in archived.get("clusters", [])
    assert 7 in archived.get("tracks", [])


def test_bundle_archived_updates_between_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "demo-show-s01e03"
    show_id = "DEMO-SHOW"
    _write_archive(
        tmp_path,
        show_id,
        clusters=[{"type": "cluster", "cluster_id": "c1", "episode_id": ep_id}],
    )
    bundle_one = bundle_mod.build_faces_review_bundle(ep_id, RUN_ID, data_root=tmp_path)
    assert "c1" in bundle_one["archived_ids"].get("clusters", [])

    _write_archive(
        tmp_path,
        show_id,
        clusters=[
            {"type": "cluster", "cluster_id": "c1", "episode_id": ep_id},
            {"type": "cluster", "cluster_id": "c2", "episode_id": ep_id},
        ],
    )
    bundle_two = bundle_mod.build_faces_review_bundle(ep_id, RUN_ID_2, data_root=tmp_path)
    assert "c2" in bundle_two["archived_ids"].get("clusters", [])


def test_bundle_counts_reflect_archived_tracks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "demo-show-s01e04"
    show_id = "DEMO-SHOW"
    _write_run_artifacts(
        ep_id,
        RUN_ID,
        identities=[{"identity_id": "c1", "track_ids": [1, 2]}],
        tracks=[
            {"track_id": 1, "faces_count": 2},
            {"track_id": 2, "faces_count": 1},
        ],
        faces=[
            {"track_id": 1, "frame_idx": 1},
            {"track_id": 1, "frame_idx": 2},
            {"track_id": 2, "frame_idx": 1},
        ],
    )
    _write_archive(
        tmp_path,
        show_id,
        tracks=[{"type": "track", "track_id": 2, "episode_id": ep_id}],
    )

    bundle = bundle_mod.build_faces_review_bundle(ep_id, RUN_ID, data_root=tmp_path)
    clusters = bundle["cluster_payload"].get("clusters", [])
    assert clusters
    cluster = clusters[0]
    counts = cluster.get("counts", {})
    assert counts.get("tracks") == 1
    assert counts.get("faces") == 2
    assert len(cluster.get("tracks", [])) == 1


def test_bundle_tracks_field_type_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "demo-show-s01e05"

    def _fake_summary(ep_id: str, *, run_id: str | None = None) -> Dict[str, Any]:
        return {
            "clusters": [
                {
                    "identity_id": "c1",
                    "tracks": 5,
                    "counts": {"tracks": 5, "faces": 10},
                }
            ]
        }

    monkeypatch.setattr(bundle_mod, "cluster_track_summary", _fake_summary)
    monkeypatch.setattr(bundle_mod, "load_identities", lambda *args, **kwargs: {"identities": []})
    monkeypatch.setattr(bundle_mod.GroupingService, "list_unlinked_entities", lambda *args, **kwargs: {"entities": []})

    bundle = bundle_mod.build_faces_review_bundle(ep_id, RUN_ID, data_root=tmp_path)
    cluster = bundle["cluster_payload"]["clusters"][0]
    assert isinstance(cluster.get("tracks"), list)


def test_bundle_thumbnail_fallback_uses_rep_media_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "demo-show-s01e06"
    show_id = "DEMO-SHOW"
    _write_people(
        tmp_path,
        show_id,
        [
            {
                "person_id": "p_0001",
                "name": "Person One",
                "cast_id": "cast_1",
                "cluster_ids": [f"{ep_id}:{RUN_ID}:c1"],
            }
        ],
    )
    _write_cast(tmp_path, show_id, [{"cast_id": "cast_1", "name": "Person One"}])

    def _fake_summary(ep_id: str, *, run_id: str | None = None) -> Dict[str, Any]:
        return {
            "clusters": [
                {
                    "identity_id": "c1",
                    "tracks": [
                        {
                            "track_id": 1,
                            "rep_media_url": "https://example.com/media.jpg",
                        }
                    ],
                }
            ]
        }

    monkeypatch.setattr(bundle_mod, "cluster_track_summary", _fake_summary)
    monkeypatch.setattr(bundle_mod, "load_identities", lambda *args, **kwargs: {"identities": []})
    monkeypatch.setattr(bundle_mod.GroupingService, "list_unlinked_entities", lambda *args, **kwargs: {"entities": []})

    bundle = bundle_mod.build_faces_review_bundle(ep_id, RUN_ID, data_root=tmp_path)
    cluster = bundle["cluster_payload"]["clusters"][0]
    track = cluster["tracks"][0]
    assert track.get("crop_url") == "https://example.com/media.jpg"
    assert track.get("thumb_url") == "https://example.com/media.jpg"
    assert bundle["cast_gallery_cards"][0]["featured_thumbnail"] == "https://example.com/media.jpg"


def test_bundle_cast_options_include_legacy_people(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "demo-show-s01e07"
    show_id = "DEMO-SHOW"
    _write_people(
        tmp_path,
        show_id,
        [
            {
                "person_id": "p_0009",
                "name": "Legacy Person",
                "cast_id": "legacy_1",
                "cluster_ids": [f"{ep_id}:{RUN_ID}:c9"],
            }
        ],
    )

    bundle = bundle_mod.build_faces_review_bundle(ep_id, RUN_ID, data_root=tmp_path)
    assert bundle["cast_options"].get("legacy_1") == "Legacy Person"


def test_bundle_cluster_payload_respects_run_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "demo-show-s01e08"
    _write_run_artifacts(
        ep_id,
        RUN_ID,
        identities=[{"identity_id": "c1", "track_ids": [1]}],
        tracks=[{"track_id": 1, "faces_count": 1}],
        faces=[{"track_id": 1, "frame_idx": 1}],
    )
    _write_run_artifacts(
        ep_id,
        RUN_ID_2,
        identities=[{"identity_id": "c2", "track_ids": [2]}],
        tracks=[{"track_id": 2, "faces_count": 1}],
        faces=[{"track_id": 2, "frame_idx": 1}],
    )

    bundle_one = bundle_mod.build_faces_review_bundle(ep_id, RUN_ID, data_root=tmp_path)
    bundle_two = bundle_mod.build_faces_review_bundle(ep_id, RUN_ID_2, data_root=tmp_path)

    clusters_one = {c.get("identity_id") for c in bundle_one["cluster_payload"].get("clusters", [])}
    clusters_two = {c.get("identity_id") for c in bundle_two["cluster_payload"].get("clusters", [])}
    assert clusters_one == {"c1"}
    assert clusters_two == {"c2"}


def test_bundle_cast_gallery_run_scoped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "demo-show-s01e09"
    show_id = "DEMO-SHOW"
    _write_people(
        tmp_path,
        show_id,
        [
            {
                "person_id": "p_0010",
                "name": "Cast One",
                "cast_id": "cast_10",
                "cluster_ids": [
                    f"{ep_id}:{RUN_ID}:c1",
                    f"{ep_id}:{RUN_ID_2}:c2",
                ],
            }
        ],
    )
    _write_cast(tmp_path, show_id, [{"cast_id": "cast_10", "name": "Cast One"}])

    bundle_one = bundle_mod.build_faces_review_bundle(ep_id, RUN_ID, data_root=tmp_path)
    bundle_two = bundle_mod.build_faces_review_bundle(ep_id, RUN_ID_2, data_root=tmp_path)

    assert bundle_one["cast_gallery_cards"][0]["episode_clusters"] == ["c1"]
    assert bundle_two["cast_gallery_cards"][0]["episode_clusters"] == ["c2"]
