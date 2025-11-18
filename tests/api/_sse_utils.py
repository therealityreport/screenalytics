from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from py_screenalytics.artifacts import ensure_dirs, get_path


def write_sample_tracks(ep_id: str, sample_count: int = 5) -> None:
    """Create a minimal tracks.jsonl with a handful of sampled faces."""
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    tracks_path = manifests_dir / "tracks.jsonl"
    tracks_path.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    for idx in range(max(sample_count, 1)):
        samples.append(
            {
                "frame_idx": idx * 5,
                "ts": round(idx * 0.5, 4),
                "bbox_xyxy": [10 + idx, 20 + idx, 110 + idx, 160 + idx],
            }
        )
    row = {
        "ep_id": ep_id,
        "track_id": 1,
        "bboxes_sampled": samples,
    }
    tracks_path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def write_sample_faces(ep_id: str, face_count: int = 5) -> None:
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    faces_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(max(face_count, 1)):
        rows.append(
            {
                "ep_id": ep_id,
                "face_id": f"face_{idx:04d}",
                "track_id": 1,
                "frame_idx": idx,
                "ts": round(idx * 0.5, 4),
                "bbox_xyxy": [10, 20, 110, 160],
            }
        )
    faces_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def collect_sse_events(response) -> List[Tuple[str, Dict[str, Any]]]:
    events: List[Tuple[str, Dict[str, Any]]] = []
    data_lines: List[str] = []
    event_name = "message"
    for raw_line in response.iter_lines():
        if raw_line is None:
            continue
        line = (
            raw_line.decode() if isinstance(raw_line, (bytes, bytearray)) else raw_line
        )
        line = line.strip()
        if not line:
            if data_lines:
                payload = json.loads("\n".join(data_lines))
                events.append((event_name or "message", payload))
            data_lines = []
            event_name = "message"
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())
    if data_lines:
        payload = json.loads("\n".join(data_lines))
        events.append((event_name or "message", payload))
    return events
