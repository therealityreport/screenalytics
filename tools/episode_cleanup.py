#!/usr/bin/env python
"""Orchestrate full detect → embed → cluster → grouping cleanup for an episode."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_screenalytics.artifacts import ensure_dirs, get_path

from apps.api.services.grouping import GroupingService

LOGGER = logging.getLogger("episode_cleanup")


DEFAULT_ACTIONS = ("split_tracks", "reembed", "recluster", "group_clusters")


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_command(command: Sequence[str]) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} exited with {result.returncode}")


def _build_detect_command(args, video_path: Path, progress_path: Path) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(REPO_ROOT / "tools" / "episode_run.py"),
        "--ep-id",
        args.ep_id,
        "--video",
        str(video_path),
        "--stride",
        str(args.stride),
        "--device",
        args.device,
        "--progress-file",
        str(progress_path),
        "--scene-detector",
        args.scene_detector,
        "--scene-threshold",
        str(args.scene_threshold),
        "--scene-min-len",
        str(args.scene_min_len),
        "--scene-warmup-dets",
        str(args.scene_warmup_dets),
        "--detector",
        args.detector,
        "--tracker",
        args.tracker,
        "--max-gap",
        str(args.max_gap),
    ]
    if args.fps and args.fps > 0:
        cmd += ["--fps", str(args.fps)]
    if args.save_frames:
        cmd.append("--save-frames")
    if args.save_crops:
        cmd.append("--save-crops")
    if args.jpeg_quality != 85:
        cmd += ["--jpeg-quality", str(args.jpeg_quality)]
    if args.det_thresh is not None:
        cmd += ["--det-thresh", str(args.det_thresh)]
    return cmd


def _build_faces_command(args, progress_path: Path) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(REPO_ROOT / "tools" / "episode_run.py"),
        "--ep-id",
        args.ep_id,
        "--faces-embed",
        "--device",
        args.embed_device,
        "--progress-file",
        str(progress_path),
        "--thumb-size",
        str(args.thumb_size),
    ]
    if args.save_frames:
        cmd.append("--save-frames")
    if args.save_crops:
        cmd.append("--save-crops")
    if args.jpeg_quality != 85:
        cmd += ["--jpeg-quality", str(args.jpeg_quality)]
    return cmd


def _build_cluster_command(args, progress_path: Path) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "tools" / "episode_run.py"),
        "--ep-id",
        args.ep_id,
        "--cluster",
        "--device",
        args.device,
        "--cluster-thresh",
        str(args.cluster_thresh),
        "--min-cluster-size",
        str(args.min_cluster_size),
        "--progress-file",
        str(progress_path),
    ]


@dataclass
class CleanupResult:
    actions: List[str]
    tracks_before: int
    tracks_after: int
    clusters_before: int
    clusters_after: int
    faces_after: int
    splits: Dict[str, int]
    report_path: Path
    grouping: dict | None


def _normalize_actions(actions: Iterable[str] | None) -> List[str]:
    if not actions:
        return list(DEFAULT_ACTIONS)
    selected: List[str] = []
    for action in actions:
        if action not in DEFAULT_ACTIONS:
            continue
        if action not in selected:
            selected.append(action)
    return selected or list(DEFAULT_ACTIONS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full cleanup pipeline for an episode.")
    parser.add_argument("--ep-id", required=True, help="Episode identifier (slug-sXXeYY)")
    parser.add_argument("--video", help="Override path to mirrored episode video")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--embed-device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--detector", default="retinaface")
    parser.add_argument("--tracker", default="bytetrack")
    parser.add_argument("--det-thresh", type=float, default=None)
    parser.add_argument("--scene-detector", default="pyscenedetect")
    parser.add_argument("--scene-threshold", type=float, default=27.0)
    parser.add_argument("--scene-min-len", type=int, default=12)
    parser.add_argument("--scene-warmup-dets", type=int, default=3)
    parser.add_argument("--max-gap", type=int, default=30)
    parser.add_argument("--cluster-thresh", type=float, default=0.6)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--thumb-size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--progress-file", help="Path to write aggregated progress JSON")
    parser.add_argument("--actions", nargs="+", choices=DEFAULT_ACTIONS, help="Subset of cleanup actions to run")
    parser.add_argument("--save-frames", dest="save_frames", action="store_true", default=False)
    parser.add_argument("--no-save-frames", dest="save_frames", action="store_false")
    parser.add_argument("--save-crops", dest="save_crops", action="store_true", default=False)
    parser.add_argument("--no-save-crops", dest="save_crops", action="store_false")
    parser.add_argument("--write-back", dest="write_back", action="store_true", default=True)
    parser.add_argument("--no-write-back", dest="write_back", action="store_false")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_dirs(args.ep_id)
    manifests_dir = get_path(args.ep_id, "detections").parent
    data_actions = _normalize_actions(args.actions)
    LOGGER.info("[cleanup] starting run ep_id=%s actions=%s", args.ep_id, data_actions)
    progress_path = Path(args.progress_file) if args.progress_file else manifests_dir / "cleanup_progress.json"
    video_path = Path(args.video) if args.video else get_path(args.ep_id, "video")
    if not video_path.exists():
        raise FileNotFoundError(f"Episode video not found at {video_path}")

    tracks_path = get_path(args.ep_id, "tracks")
    identities_path = manifests_dir / "identities.json"
    faces_path = manifests_dir / "faces.jsonl"

    tracks_before = _count_lines(tracks_path)
    clusters_before = 0
    identities_doc = _read_json(identities_path)
    if identities_doc:
        clusters_before = len(identities_doc.get("identities", []))

    if "split_tracks" in data_actions:
        _run_command(_build_detect_command(args, video_path, progress_path))
        tracks_after = _count_lines(tracks_path)
        LOGGER.info(
            "[cleanup] split_tracks done: tracks %s → %s (Δ %+d)",
            tracks_before,
            tracks_after,
            tracks_after - tracks_before,
        )
    else:
        tracks_after = tracks_before

    if "reembed" in data_actions:
        _run_command(_build_faces_command(args, progress_path))
        LOGGER.info("[cleanup] reembed done: faces now %s", _count_lines(faces_path))
    if "recluster" in data_actions:
        _run_command(_build_cluster_command(args, progress_path))
        LOGGER.info("[cleanup] recluster done (identities will refresh).")
    faces_after = _count_lines(faces_path)
    identities_doc = _read_json(identities_path)
    clusters_after = len(identities_doc.get("identities", [])) if identities_doc else 0
    LOGGER.info(
        "[cleanup] cluster stats: clusters %s → %s (Δ %+d) · faces=%s",
        clusters_before,
        clusters_after,
        clusters_after - clusters_before,
        faces_after,
    )

    metrics_path = manifests_dir / "track_metrics.json"
    metrics_payload = _read_json(metrics_path) or {}
    splits = metrics_payload.get("metrics", {}).get("appearance_gate", {}).get("splits", {})
    if splits:
        LOGGER.info(
            "[cleanup] gate splits breakdown: hard=%s streak=%s iou=%s total=%s",
            splits.get("hard"),
            splits.get("streak"),
            splits.get("iou"),
            splits.get("total"),
        )

    grouping_result = None
    if "group_clusters" in data_actions:
        LOGGER.info("[cleanup] Starting cluster grouping...")
        grouping = GroupingService()

        # Compute centroids with progress logging
        def log_progress(current, total, status):
            LOGGER.info(f"[cleanup] centroid progress: {current}/{total} - {status}")

        centroids = grouping.compute_cluster_centroids(args.ep_id, progress_callback=log_progress)

        # Group within episode
        def log_within_progress(current, total, status):
            LOGGER.info(f"[cleanup] within-episode grouping: {current}/{total} - {status}")

        within = grouping.group_within_episode(args.ep_id, progress_callback=log_within_progress)

        # Group across episodes
        across = None
        if args.write_back:
            LOGGER.info("[cleanup] Matching clusters to show-level people...")
            across = grouping.group_across_episodes(args.ep_id)

        grouping_result = {
            "centroids": centroids,
            "within_episode": within,
            "across_episodes": across,
        }
        centroids_count = len(centroids.get("centroids", []))
        merged = within.get("merged_count", 0) if isinstance(within, dict) else 0
        assigned = len(across.get("assigned", [])) if isinstance(across, dict) else 0
        new_people = across.get("new_people_count", 0) if isinstance(across, dict) else 0
        LOGGER.info(
            "[cleanup] grouping done: centroids=%s merged=%s assigned=%s new_people=%s",
            centroids_count,
            merged,
            assigned,
            new_people,
        )

    report = {
        "ep_id": args.ep_id,
        "actions": data_actions,
        "tracks_before": tracks_before,
        "tracks_after": tracks_after,
        "clusters_before": clusters_before,
        "clusters_after": clusters_after,
        "faces_after": faces_after,
        "splits": splits,
        "grouping": grouping_result,
    }
    report_path = manifests_dir / "cleanup_report.json"
    _write_json(report_path, report)
    _write_json(progress_path, {"stage": "episode_cleanup", "summary": report, "ep_id": args.ep_id})

    LOGGER.info("[cleanup] completed run for %s; report → %s", args.ep_id, report_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
