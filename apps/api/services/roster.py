from __future__ import annotations

from typing import Dict, List

from apps.shared import storage

__all__ = ["add_if_missing", "load_roster", "save_roster"]


def _normalize_show(show: str) -> str:
    slug = (show or "").strip().lower()
    if not slug:
        raise ValueError("show is required")
    return slug


def _local_path(show: str) -> str:
    slug = _normalize_show(show)
    return f"data/rosters/{slug}.json"


def _s3_path(show: str) -> str:
    slug = _normalize_show(show)
    return f"artifacts/manifests/{slug}/roster.json"


def _empty_payload(show: str) -> Dict[str, object]:
    slug = _normalize_show(show)
    return {"show": slug, "names": [], "updated_at": None}


def _dedupe(names: List[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for name in names:
        cleaned = (name or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    ordered.sort(key=str.lower)
    return ordered


def load_roster(show: str) -> Dict[str, object]:
    slug = _normalize_show(show)
    if storage.use_s3():
        s3_path = _s3_path(slug)
        if storage.exists(s3_path):
            return storage.read_json(s3_path)
    local_path = _local_path(slug)
    if storage.exists(local_path):
        return storage.read_json(local_path)
    return _empty_payload(slug)


def save_roster(show: str, roster: Dict[str, object]) -> Dict[str, object]:
    slug = _normalize_show(show)
    payload = {
        "show": slug,
        "names": _dedupe(roster.get("names", []) if isinstance(roster, dict) else []),
        "updated_at": storage.now_iso(),
    }
    storage.write_json(_local_path(slug), payload)
    if storage.use_s3():
        storage.s3_write_json(_s3_path(slug), payload)
    return payload


def add_if_missing(show: str, name: str) -> Dict[str, object]:
    slug = _normalize_show(show)
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("name required")
    roster = load_roster(slug)
    names = roster.get("names", []) or []
    lowered = {n.lower() for n in names if isinstance(n, str)}
    if cleaned.lower() in lowered:
        return roster
    roster["names"] = [*names, cleaned]
    return save_roster(slug, roster)
