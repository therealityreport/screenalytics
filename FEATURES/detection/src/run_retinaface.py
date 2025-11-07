"""Detection scaffolding for Screenalytics RetinaFace stage.

This implementation is a lightweight stub that reads a frame manifest (JSONL)
and emits detections.jsonl records that follow schema `det_v1`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Any

SCHEMA_VERSION = "det_v1"


def load_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a mock model handle for pipeline wiring tests."""
    return {
        "model_id": cfg.get("model_id", "retinaface_stub"),
        "min_size": cfg.get("min_size", 0),
        "confidence_th": cfg.get("confidence_th", 0.0),
    }


def _ensure_bbox(raw: Iterable[float]) -> List[float]:
    bbox = list(raw)[:4]
    while len(bbox) < 4:
        bbox.append(0.0)
    return bbox


def _ensure_landmarks(raw: Iterable[float]) -> List[float]:
    pts = list(raw)[:10]
    while len(pts) < 10:
        pts.append(0.0)
    return pts


def detect_frames(manifest_path: Path, out_path: Path, cfg: Dict[str, Any]) -> int:
    """Emit detections for every frame listed in the manifest JSONL file."""
    manifest_path = Path(manifest_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_count = 0
    with manifest_path.open("r", encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            detection = {
                "ep_id": frame.get("ep_id", "unknown-episode"),
                "ts_s": frame.get("ts_s", 0.0),
                "bbox": _ensure_bbox(frame.get("bbox", [0.0, 0.0, 1.0, 1.0])),
                "landmarks": _ensure_landmarks(frame.get("landmarks", [])),
                "conf": cfg.get("confidence_th", 0.0),
                "model_id": cfg.get("model_id", "retinaface_stub"),
                "schema_version": SCHEMA_VERSION,
            }
            dst.write(json.dumps(detection) + "\n")
            write_count += 1
    return write_count


if __name__ == "__main__":
    raise SystemExit("Use detect_frames() via orchestration or tests.")
