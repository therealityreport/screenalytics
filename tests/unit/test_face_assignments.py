from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.api.services import assignments as assignment_service
from apps.api.services.assignment_resolver import (
    resolve_face_assignment,
    resolve_track_assignment,
)
from apps.api.services.people import PeopleService
from py_screenalytics import run_layout

RUN_ID = "Attempt1_2025-02-01_000000EST"
RUN_ID_2 = "Attempt2_2025-02-02_000000EST"


def _write_identities(ep_id: str, run_id: str, identities: list[dict[str, object]], root: Path) -> None:
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    payload = {"ep_id": ep_id, "identities": identities, "stats": {}}
    (run_root / "identities.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_resolver_prefers_track_override() -> None:
    cluster_assignments = {"c1": {"cast_id": "cast_a", "source": "manual"}}
    track_overrides = {"123": {"cast_id": "cast_b", "source": "manual"}}

    resolved = resolve_track_assignment(123, "c1", cluster_assignments, track_overrides)
    assert resolved["cast_id"] == "cast_b"
    assert resolved["assignment_type"] == "track_override"


def test_resolver_face_exclusion_blocks_cast() -> None:
    cluster_assignments = {"c1": {"cast_id": "cast_a", "source": "manual"}}
    track_overrides = {}
    face_exclusions = {"face_001": {"excluded": True, "reason": "blur"}}

    resolved = resolve_face_assignment(
        "face_001",
        123,
        "c1",
        cluster_assignments,
        track_overrides,
        face_exclusions,
    )
    assert resolved["cast_id"] is None
    assert resolved["excluded"] is True
    assert resolved["assignment_type"] == "face_exclusion"


def test_assignments_run_scoped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "demo-show-s01e10"
    _write_identities(ep_id, RUN_ID, [{"identity_id": "c1"}], tmp_path)
    _write_identities(ep_id, RUN_ID_2, [{"identity_id": "c1"}], tmp_path)

    assignment_service.set_cluster_assignment(
        ep_id,
        RUN_ID,
        cluster_id="c1",
        cast_id="cast_a",
    )

    state_one = assignment_service.load_assignment_state(ep_id, RUN_ID, include_inferred=False)
    state_two = assignment_service.load_assignment_state(ep_id, RUN_ID_2, include_inferred=False)

    assert state_one["cluster_assignments_raw"]["c1"]["cast_id"] == "cast_a"
    assert "c1" not in state_two["cluster_assignments_raw"]


def test_cluster_assignment_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "demo-show-s01e11"
    _write_identities(ep_id, RUN_ID, [{"identity_id": "c1"}], tmp_path)

    assignment_service.set_cluster_assignment(
        ep_id,
        RUN_ID,
        cluster_id="c1",
        cast_id="cast_a",
    )
    state_one = assignment_service.load_assignment_state(ep_id, RUN_ID, include_inferred=False)
    first_timestamp = state_one["cluster_assignments_raw"]["c1"]["updated_at"]

    assignment_service.set_cluster_assignment(
        ep_id,
        RUN_ID,
        cluster_id="c1",
        cast_id="cast_a",
    )
    state_two = assignment_service.load_assignment_state(ep_id, RUN_ID, include_inferred=False)
    second_timestamp = state_two["cluster_assignments_raw"]["c1"]["updated_at"]

    assert first_timestamp == second_timestamp


def test_unassign_overrides_inferred_cast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "demo-show-s01e12"
    people_service = PeopleService(tmp_path)
    person = people_service.create_person("demo-show", name="Test Person", cast_id="cast_a")
    _write_identities(ep_id, RUN_ID, [{"identity_id": "c1", "person_id": person["person_id"]}], tmp_path)

    assignment_service.set_cluster_assignment(
        ep_id,
        RUN_ID,
        cluster_id="c1",
        cast_id=None,
    )

    state = assignment_service.load_assignment_state(
        ep_id,
        RUN_ID,
        include_inferred=True,
        data_root=tmp_path,
    )
    assert state["cluster_assignments"]["c1"]["cast_id"] is None
    assert state["cluster_assignments"]["c1"].get("unassigned") is True
