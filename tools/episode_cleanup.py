#!/usr/bin/env python
"""Orchestrate full detect → embed → cluster → grouping cleanup for an episode."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_screenalytics.artifacts import ensure_dirs, get_path

from apps.api.services.grouping import GroupingService

LOGGER = logging.getLogger("episode_cleanup")


def _load_performance_profile(profile_name: str | None = None) -> dict[str, Any]:
    """
    Load performance profile configuration.

    Args:
        profile_name: Profile to load ("fast_cpu", "low_power", "balanced", "high_accuracy")
                     If None, uses SCREANALYTICS_PERF_PROFILE env var or "balanced"

    Returns:
        Dictionary of profile settings
    """
    if profile_name is None:
        profile_name = os.environ.get("SCREENALYTICS_PERF_PROFILE", "balanced")

    profile_name = profile_name.lower().strip()

    config_path = REPO_ROOT / "config" / "pipeline" / "performance_profiles.yaml"
    if not config_path.exists():
        LOGGER.debug("Performance profiles YAML not found at %s", config_path)
        return {}

    try:
        import yaml

        with open(config_path, "r") as f:
            all_profiles = yaml.safe_load(f)

        if not all_profiles or profile_name not in all_profiles:
            LOGGER.warning("Profile '%s' not found, using defaults", profile_name)
            return {}

        profile = all_profiles[profile_name]
        LOGGER.info("Loaded performance profile '%s': %s", profile_name, profile.get("description", ""))
        return profile
    except Exception as exc:
        LOGGER.warning("Failed to load performance profile: %s", exc)
        return {}


def _load_clustering_config() -> dict[str, Any]:
    """
    Load clustering configuration from clustering.yaml.

    Returns:
        Dictionary of clustering settings
    """
    config_path = REPO_ROOT / "config" / "pipeline" / "clustering.yaml"
    if not config_path.exists():
        LOGGER.debug("Clustering config YAML not found at %s, using defaults", config_path)
        return {}

    try:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if config:
            LOGGER.info("Loaded clustering config from %s", config_path)
            return config
    except Exception as exc:
        LOGGER.warning("Failed to load clustering config YAML: %s", exc)

    return {}


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


def _write_progress(
    progress_path: Path,
    phase: str,
    phase_index: int,
    phase_total: int,
    phase_progress: float,
    start_time: float,
    ep_id: str,
) -> None:
    """Write progress JSON with phase information."""
    elapsed = time.time() - start_time
    progress_payload = {
        "stage": "episode_cleanup",
        "ep_id": ep_id,
        "phase": phase,
        "phase_index": phase_index,
        "phase_total": phase_total,
        "phase_progress": round(phase_progress, 3),
        "total_elapsed_seconds": round(elapsed, 2),
    }
    _write_json(progress_path, progress_payload)


def _build_detect_command(args, video_path: Path, progress_path: Path) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(REPO_ROOT / "tools" / "episode_run.py"),
        "--ep-id",
        args.ep_id,
        "--video",
        str(video_path),
        "--device",
        args.device,
        "--progress-file",
        str(progress_path),
    ]
    # Pass profile if set
    if hasattr(args, "profile") and args.profile:
        cmd += ["--profile", args.profile]

    # Explicit overrides (only if not using profile or if explicitly set)
    if hasattr(args, "stride"):
        cmd += ["--stride", str(args.stride)]
    if args.fps and args.fps > 0:
        cmd += ["--fps", str(args.fps)]
    if hasattr(args, "scene_detector"):
        cmd += ["--scene-detector", args.scene_detector]
    if hasattr(args, "scene_threshold"):
        cmd += ["--scene-threshold", str(args.scene_threshold)]
    if hasattr(args, "scene_min_len"):
        cmd += ["--scene-min-len", str(args.scene_min_len)]
    if hasattr(args, "scene_warmup_dets"):
        cmd += ["--scene-warmup-dets", str(args.scene_warmup_dets)]
    if hasattr(args, "detector"):
        cmd += ["--detector", args.detector]
    if hasattr(args, "tracker"):
        cmd += ["--tracker", args.tracker]
    if hasattr(args, "max_gap"):
        cmd += ["--max-gap", str(args.max_gap)]
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
    ]
    # Pass profile if set
    if hasattr(args, "profile") and args.profile:
        cmd += ["--profile", args.profile]

    if hasattr(args, "thumb_size"):
        cmd += ["--thumb-size", str(args.thumb_size)]
    if args.save_frames:
        cmd.append("--save-frames")
    if args.save_crops:
        cmd.append("--save-crops")
    if args.jpeg_quality != 85:
        cmd += ["--jpeg-quality", str(args.jpeg_quality)]
    return cmd


def _build_cluster_command(args, progress_path: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "episode_run.py"),
        "--ep-id",
        args.ep_id,
        "--cluster",
        "--device",
        args.device,
        "--progress-file",
        str(progress_path),
    ]
    # Pass profile if set
    if hasattr(args, "profile") and args.profile:
        cmd += ["--profile", args.profile]

    if hasattr(args, "cluster_thresh"):
        cmd += ["--cluster-thresh", str(args.cluster_thresh)]
    if hasattr(args, "min_cluster_size"):
        cmd += ["--min-cluster-size", str(args.min_cluster_size)]
    if hasattr(args, "min_identity_sim"):
        cmd += ["--min-identity-sim", str(args.min_identity_sim)]
    return cmd


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
    parser.add_argument(
        "--profile",
        choices=["fast_cpu", "low_power", "balanced", "high_accuracy"],
        default=None,
        help=(
            "Performance profile preset. Overrides stride, fps, and other detection/clustering defaults. "
            "Profiles: fast_cpu/low_power (lower quality, faster), balanced (default), high_accuracy (slower, higher quality)."
        ),
    )
    parser.add_argument("--stride", type=int, default=None, help="Frame stride for detection (default from profile or 3)")
    parser.add_argument("--fps", type=float, default=None, help="FPS limit for detection (default from profile or no limit)")
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
    parser.add_argument("--cluster-thresh", type=float, default=None, help="Clustering threshold (default from config or 0.6)")
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--min-identity-sim", type=float, default=0.5)
    parser.add_argument("--thumb-size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--progress-file", help="Path to write aggregated progress JSON")
    parser.add_argument(
        "--actions",
        nargs="+",
        choices=DEFAULT_ACTIONS,
        help="Subset of cleanup actions to run",
    )
    parser.add_argument("--out-root", dest="out_root", help="Override data root (sets SCREENALYTICS_DATA_ROOT)")
    parser.add_argument("--save-frames", dest="save_frames", action="store_true", default=False)
    parser.add_argument("--no-save-frames", dest="save_frames", action="store_false")
    parser.add_argument("--save-crops", dest="save_crops", action="store_true", default=False)
    parser.add_argument("--no-save-crops", dest="save_crops", action="store_false")
    parser.add_argument("--write-back", dest="write_back", action="store_true", default=True)
    parser.add_argument("--no-write-back", dest="write_back", action="store_false")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    start_time = time.time()
    args = parse_args(argv)
    if getattr(args, "out_root", None):
        os.environ["SCREENALYTICS_DATA_ROOT"] = str(Path(args.out_root).expanduser())

    # Apply performance profile settings (if not overridden by CLI flags)
    if hasattr(args, "profile") and args.profile:
        profile = _load_performance_profile(args.profile)
        if profile:
            # Apply profile defaults only if not explicitly set via CLI
            if args.stride is None:
                args.stride = profile.get("frame_stride", 3)
                LOGGER.info("[PROFILE] Applied frame_stride=%d from %s profile", args.stride, args.profile)
            if args.fps is None:
                detection_fps_limit = profile.get("detection_fps_limit")
                if detection_fps_limit:
                    args.fps = float(detection_fps_limit)
                    LOGGER.info("[PROFILE] Applied fps=%.1f from %s profile", args.fps, args.profile)
            LOGGER.info("[PROFILE] Loaded %s profile: %s", args.profile, profile.get("description", ""))
    else:
        # No profile, apply defaults
        if args.stride is None:
            args.stride = 3
        if args.fps is None:
            args.fps = 0.0

    # Load clustering config for cluster_thresh default
    clustering_config = _load_clustering_config()
    if args.cluster_thresh is None:
        args.cluster_thresh = clustering_config.get("cluster_thresh", 0.6)
        LOGGER.info("[CONFIG] Applied cluster_thresh=%.2f from clustering config", args.cluster_thresh)

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
    metrics_path = manifests_dir / "track_metrics.json"

    # Capture BEFORE state
    tracks_before = _count_lines(tracks_path)
    faces_before = _count_lines(faces_path)
    clusters_before = 0
    identities_doc = _read_json(identities_path)
    if identities_doc:
        clusters_before = len(identities_doc.get("identities", []))

    # Capture BEFORE metrics (key metrics from track_metrics.json)
    metrics_before_payload = _read_json(metrics_path) or {}
    metrics_before = {}
    if "metrics" in metrics_before_payload:
        m = metrics_before_payload["metrics"]
        metrics_before = {
            "tracks_per_minute": m.get("tracks_per_minute"),
            "short_track_fraction": m.get("short_track_fraction"),
            "id_switch_rate": m.get("id_switch_rate"),
        }
    if "cluster_metrics" in metrics_before_payload:
        cm = metrics_before_payload["cluster_metrics"]
        metrics_before.update({
            "singleton_fraction": cm.get("singleton_fraction"),
            "largest_cluster_fraction": cm.get("largest_cluster_fraction"),
        })

    phase_total = len(data_actions)

    if "split_tracks" in data_actions:
        phase_index = data_actions.index("split_tracks") + 1
        _write_progress(progress_path, "split_tracks", phase_index, phase_total, 0.0, start_time, args.ep_id)
        _run_command(_build_detect_command(args, video_path, progress_path))
        _write_progress(progress_path, "split_tracks", phase_index, phase_total, 1.0, start_time, args.ep_id)
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
        phase_index = data_actions.index("reembed") + 1
        _write_progress(progress_path, "reembed", phase_index, phase_total, 0.0, start_time, args.ep_id)
        _run_command(_build_faces_command(args, progress_path))
        _write_progress(progress_path, "reembed", phase_index, phase_total, 1.0, start_time, args.ep_id)
        LOGGER.info("[cleanup] reembed done: faces now %s", _count_lines(faces_path))
    if "recluster" in data_actions:
        phase_index = data_actions.index("recluster") + 1
        _write_progress(progress_path, "recluster", phase_index, phase_total, 0.0, start_time, args.ep_id)
        _run_command(_build_cluster_command(args, progress_path))
        _write_progress(progress_path, "recluster", phase_index, phase_total, 1.0, start_time, args.ep_id)
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

    # Capture AFTER metrics
    metrics_after_payload = _read_json(metrics_path) or {}
    metrics_after = {}
    if "metrics" in metrics_after_payload:
        m = metrics_after_payload["metrics"]
        metrics_after = {
            "tracks_per_minute": m.get("tracks_per_minute"),
            "short_track_fraction": m.get("short_track_fraction"),
            "id_switch_rate": m.get("id_switch_rate"),
        }
    if "cluster_metrics" in metrics_after_payload:
        cm = metrics_after_payload["cluster_metrics"]
        metrics_after.update({
            "singleton_fraction": cm.get("singleton_fraction"),
            "largest_cluster_fraction": cm.get("largest_cluster_fraction"),
        })

    splits = metrics_after_payload.get("metrics", {}).get("appearance_gate", {}).get("splits", {})
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
        phase_index = data_actions.index("group_clusters") + 1
        _write_progress(progress_path, "group_clusters", phase_index, phase_total, 0.0, start_time, args.ep_id)
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
            try:
                across = grouping.group_across_episodes(args.ep_id)
            except ValueError as exc:
                LOGGER.warning("Skipping cross-episode grouping: %s", exc)
                across = {"skipped": True, "reason": str(exc)}

        grouping_result = {
            "centroids": centroids,
            "within_episode": within,
            "across_episodes": across,
        }
        centroids_count = len(centroids.get("centroids", []))
        merged = within.get("merged_count", 0) if isinstance(within, dict) else 0
        assigned = len(across.get("assigned", [])) if isinstance(across, dict) else 0
        new_people = across.get("new_people_count", 0) if isinstance(across, dict) else 0
        _write_progress(progress_path, "group_clusters", phase_index, phase_total, 1.0, start_time, args.ep_id)
        LOGGER.info(
            "[cleanup] grouping done: centroids=%s merged=%s assigned=%s new_people=%s",
            centroids_count,
            merged,
            assigned,
            new_people,
        )

    runtime_sec = time.time() - start_time

    report = {
        "ep_id": args.ep_id,
        "actions_completed": data_actions,
        "runtime_sec": round(runtime_sec, 2),
        # Counts (before/after)
        "tracks_before": tracks_before,
        "tracks_after": tracks_after,
        "faces_before": faces_before,
        "faces_after": faces_after,
        "clusters_before": clusters_before,
        "clusters_after": clusters_after,
        # Key metrics (before/after)
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        # Legacy fields for compatibility
        "actions": data_actions,
        "splits": splits,
        "grouping": grouping_result,
    }
    report_path = manifests_dir / "cleanup_report.json"
    _write_json(report_path, report)
    final_progress = {
        "stage": "episode_cleanup",
        "ep_id": args.ep_id,
        "phase": "done",
        "phase_index": phase_total,
        "phase_total": phase_total,
        "phase_progress": 1.0,
        "total_elapsed_seconds": round(runtime_sec, 2),
        "summary": report,
    }
    _write_json(progress_path, final_progress)

    LOGGER.info("[cleanup] completed run for %s in %.1fs; report → %s", args.ep_id, runtime_sec, report_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
