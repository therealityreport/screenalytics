#!/usr/bin/env python3
"""Generate a run-scoped endpoint inventory from FastAPI routes.

Outputs docs/reference/run_scoped_endpoints.md.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone


def _stub_modules() -> None:
    sys.modules.setdefault(
        "redis",
        types.SimpleNamespace(Redis=object, from_url=lambda *args, **kwargs: None),
    )

    class _DummyConf(dict):
        def __getattr__(self, name: str) -> object:
            return self.get(name)

        def __setattr__(self, name: str, value: object) -> None:
            self[name] = value

    class _DummyCelery:
        def __init__(self, *_args, **_kwargs) -> None:
            self.conf = _DummyConf()

        def autodiscover_tasks(self, *_args, **_kwargs) -> None:
            return None

        def task(self, *_args, **_kwargs):
            def _decorator(func):
                if not hasattr(func, "delay"):
                    func.delay = lambda *d_args, **d_kwargs: None  # type: ignore[attr-defined]
                return func

            return _decorator

    sys.modules.setdefault(
        "celery",
        types.SimpleNamespace(
            Task=object,
            states=types.SimpleNamespace(PROGRESS="PROGRESS"),
            group=lambda *args, **kwargs: None,
            chord=lambda *args, **kwargs: None,
            Celery=_DummyCelery,
        ),
    )
    sys.modules.setdefault(
        "celery.result",
        types.SimpleNamespace(AsyncResult=object, GroupResult=object),
    )
    sys.modules.setdefault("kombu", types.SimpleNamespace(Queue=lambda *args, **kwargs: object()))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    _stub_modules()
    from fastapi.routing import APIRoute
    from apps.api.main import app

    canonical_routes = [
        "POST /episodes/{ep_id}/runs/{run_id}/jobs/{stage}",
        "GET /episodes/{ep_id}/runs/{run_id}/state",
        "GET /episodes/{ep_id}/runs/{run_id}/integrity",
        "GET /episodes/{ep_id}/runs/{run_id}/screentime",
        "GET /episodes/{ep_id}/faces_review_bundle",
    ]
    deprecated_routes = [
        "POST /jobs/detect_track",
        "POST /jobs/detect_track_async",
        "POST /jobs/cluster",
        "POST /jobs/cluster_async",
        "POST /jobs/faces_embed",
        "POST /celery_jobs/detect_track",
        "POST /celery_jobs/cluster",
        "POST /celery_jobs/faces_embed",
    ]
    include_prefixes = ("/episodes", "/jobs", "/celery_jobs")
    id_fields = ("ep_id", "run_id", "track_id", "cluster_id", "identity_id", "face_id", "cast_id", "job_id")
    rows: list[dict[str, str]] = []

    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        path = route.path
        if not path.startswith(include_prefixes):
            continue
        endpoint = route.endpoint
        endpoint_name = f"{endpoint.__module__}.{endpoint.__name__}"
        methods = sorted(route.methods or [])
        path_params = [p.name for p in route.dependant.path_params]
        required_query = [p.name for p in route.dependant.query_params if getattr(p, "required", False)]
        required_all = set(path_params + required_query)
        required_ids = [name for name in id_fields if name in required_all]

        writes = "yes" if any(m in methods for m in ("POST", "PUT", "PATCH", "DELETE")) else "no"
        reads = "yes" if "GET" in methods else "no"
        artifact_pointers = "unknown"
        if "/runs/{run_id}/jobs/" in path or path.startswith("/jobs") or path.startswith("/celery_jobs"):
            artifact_pointers = "job-trigger"
        if path.endswith("/runs/{run_id}/state") or path.endswith("/runs/{run_id}/integrity"):
            artifact_pointers = "run_state"

        rows.append(
            {
                "method": ",".join(methods),
                "route": path,
                "handler": endpoint_name,
                "required_ids": ", ".join(required_ids) or "-",
                "required_query": ", ".join(required_query) or "-",
                "reads": reads,
                "writes": writes,
                "artifact_pointers": artifact_pointers,
            }
        )

    rows.sort(key=lambda r: (r["route"], r["method"]))

    output_path = "docs/reference/run_scoped_endpoints.md"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("# Run-Scoped Endpoint Inventory (Auto-Generated)\n\n")
        handle.write(f"Generated: {_now_iso()}\n\n")
        handle.write(
            "This table is generated from FastAPI routes at runtime. "
            "Columns `reads/writes/artifact_pointers` are heuristic.\n\n"
        )
        handle.write("## Canonical vs Deprecated\n\n")
        handle.write("Canonical (run-scoped):\n")
        for entry in canonical_routes:
            handle.write(f"- {entry}\n")
        handle.write("\nDeprecated wrappers (use canonical paths):\n")
        for entry in deprecated_routes:
            handle.write(f"- {entry}\n")
        handle.write("\n")
        handle.write("| Method | Route | Handler | Required IDs | Required Query Params | Reads | Writes | Artifact Pointers |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row['method']} | {row['route']} | {row['handler']} | "
                f"{row['required_ids']} | {row['required_query']} | {row['reads']} | "
                f"{row['writes']} | {row['artifact_pointers']} |\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
