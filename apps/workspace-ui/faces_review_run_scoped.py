from __future__ import annotations

from typing import Iterable, Sequence


def split_cluster_ids(
    cluster_ids: Iterable[str],
    ep_id: str,
    run_id: str | None,
) -> tuple[list[str], list[str]]:
    run_cluster_ids: list[str] = []
    legacy_cluster_ids: list[str] = []
    run_prefix = f"{ep_id}:{run_id}:" if run_id else None
    episode_prefix = f"{ep_id}:"
    for raw in cluster_ids:
        if not isinstance(raw, str):
            continue
        if run_prefix and raw.startswith(run_prefix):
            run_cluster_ids.append(raw[len(run_prefix):])
            continue
        if raw.startswith(episode_prefix):
            legacy_cluster_ids.append(raw[len(episode_prefix):])
    return run_cluster_ids, legacy_cluster_ids


def filter_people_for_run(
    people: Sequence[dict],
    ep_id: str,
    run_id: str | None,
) -> tuple[list[dict], list[dict]]:
    run_people: list[dict] = []
    legacy_people: list[dict] = []
    for person in people:
        cluster_ids = person.get("cluster_ids", []) if isinstance(person, dict) else []
        run_clusters, legacy_clusters = split_cluster_ids(cluster_ids, ep_id, run_id)
        if run_clusters:
            run_people.append(person)
        elif legacy_clusters:
            legacy_people.append(person)
    return run_people, legacy_people


def should_offer_group_clusters(
    people: Sequence[dict],
    ep_id: str,
    run_id: str | None,
    already_ran: bool,
) -> bool:
    if not run_id or already_ran:
        return False
    run_people, _ = filter_people_for_run(people, ep_id, run_id)
    return len(run_people) == 0
