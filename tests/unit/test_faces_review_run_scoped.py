from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "apps" / "workspace-ui" / "faces_review_run_scoped.py"
    spec = importlib.util.spec_from_file_location("faces_review_run_scoped", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load faces_review_run_scoped module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def test_split_cluster_ids_handles_run_scope() -> None:
    module = _load_module()
    ep_id = "ep-test"
    run_id = "Attempt1_2025-12-24_000000EST"
    run_ids, legacy_ids = module.split_cluster_ids(
        [f"{ep_id}:{run_id}:cluster_1", f"{ep_id}:cluster_2"],
        ep_id,
        run_id,
    )
    assert run_ids == ["cluster_1"]
    assert legacy_ids == ["cluster_2"]


def test_filter_people_for_run_scoped() -> None:
    module = _load_module()
    ep_id = "ep-test"
    run_id = "Attempt1_2025-12-24_000000EST"
    people = [
        {"person_id": "p1", "cluster_ids": [f"{ep_id}:{run_id}:c1"]},
        {"person_id": "p2", "cluster_ids": [f"{ep_id}:c2"]},
    ]
    run_people, legacy_people = module.filter_people_for_run(people, ep_id, run_id)
    assert [p["person_id"] for p in run_people] == ["p1"]
    assert [p["person_id"] for p in legacy_people] == ["p2"]


def test_should_offer_group_clusters_is_idempotent() -> None:
    module = _load_module()
    ep_id = "ep-test"
    run_id = "Attempt1_2025-12-24_000000EST"
    people = [{"person_id": "p1", "cluster_ids": [f"{ep_id}:{run_id}:c1"]}]
    assert not module.should_offer_group_clusters(people, ep_id, run_id, already_ran=False)
    assert not module.should_offer_group_clusters(people, ep_id, run_id, already_ran=True)
