"""Async job orchestration helpers for detect/track runs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from py_screenalytics.artifacts import ensure_dirs, get_path

try:  # pragma: no cover - optional ML stack
    from tools import episode_run  # type: ignore
except ModuleNotFoundError:
    episode_run = None  # type: ignore[assignment]
DEFAULT_DATA_ROOT = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
DETECTOR_CHOICES = {"retinaface"}
TRACKER_CHOICES = {"bytetrack", "strongsort"}
_DEFAULT_DETECTOR_RAW = os.getenv("DEFAULT_DETECTOR", "retinaface").lower()
DEFAULT_DETECTOR_ENV = (
    _DEFAULT_DETECTOR_RAW if _DEFAULT_DETECTOR_RAW in DETECTOR_CHOICES else "retinaface"
)
DEFAULT_TRACKER_ENV = os.getenv("DEFAULT_TRACKER", "bytetrack").lower()
SCENE_DETECTOR_CHOICES = tuple(
    getattr(episode_run, "SCENE_DETECTOR_CHOICES", ("pyscenedetect", "internal", "off"))
)
_SCENE_DETECTOR_ENV = os.getenv("SCENE_DETECTOR", "pyscenedetect").strip().lower()
if _SCENE_DETECTOR_ENV not in SCENE_DETECTOR_CHOICES:
    _SCENE_DETECTOR_ENV = "pyscenedetect"
SCENE_DETECTOR_DEFAULT = getattr(
    episode_run, "SCENE_DETECTOR_DEFAULT", _SCENE_DETECTOR_ENV
)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float_multi(names: tuple[str, ...], default: float) -> float:
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return default


def _env_int_multi(names: tuple[str, ...], default: int) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return default


SCENE_THRESHOLD_DEFAULT = getattr(
    episode_run, "SCENE_THRESHOLD_DEFAULT", _env_float("SCENE_THRESHOLD", 27.0)
)
SCENE_MIN_LEN_DEFAULT = getattr(
    episode_run, "SCENE_MIN_LEN_DEFAULT", max(_env_int("SCENE_MIN_LEN", 12), 1)
)
SCENE_WARMUP_DETS_DEFAULT = getattr(
    episode_run,
    "SCENE_WARMUP_DETS_DEFAULT",
    max(_env_int("SCENE_WARMUP_DETS", 3), 0),
)
CLEANUP_ACTIONS = ("split_tracks", "reembed", "recluster", "group_clusters")
TRACK_HIGH_THRESH_DEFAULT = getattr(episode_run, "TRACK_HIGH_THRESH_DEFAULT", 0.5)
TRACK_NEW_THRESH_DEFAULT = getattr(episode_run, "TRACK_NEW_THRESH_DEFAULT", 0.5)
TRACK_BUFFER_BASE_DEFAULT = getattr(episode_run, "TRACK_BUFFER_BASE_DEFAULT", 30)
TRACK_MIN_BOX_AREA_DEFAULT = getattr(
    episode_run, "BYTE_TRACK_MIN_BOX_AREA_DEFAULT", 20.0
)


def _resolve_track_high_thresh(value: float | None) -> float:
    if value is not None:
        return min(max(float(value), 0.0), 1.0)
    return _env_float_multi(
        ("SCREENALYTICS_TRACK_HIGH_THRESH", "BYTE_TRACK_HIGH_THRESH"),
        TRACK_HIGH_THRESH_DEFAULT,
    )


def _resolve_new_track_thresh(value: float | None) -> float:
    if value is not None:
        return min(max(float(value), 0.0), 1.0)
    return _env_float_multi(
        ("SCREENALYTICS_NEW_TRACK_THRESH", "BYTE_TRACK_NEW_TRACK_THRESH"),
        TRACK_NEW_THRESH_DEFAULT,
    )


def _resolve_track_buffer(value: int | None) -> int:
    if value is not None:
        try:
            return max(int(value), 1)
        except (TypeError, ValueError):
            return TRACK_BUFFER_BASE_DEFAULT
    return max(
        _env_int_multi(
            ("SCREENALYTICS_TRACK_BUFFER", "BYTE_TRACK_BUFFER"),
            TRACK_BUFFER_BASE_DEFAULT,
        ),
        1,
    )


def _resolve_min_box_area(value: float | None) -> float:
    if value is not None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = TRACK_MIN_BOX_AREA_DEFAULT
        return max(numeric, 0.0)
    return max(
        _env_float_multi(
            ("SCREENALYTICS_MIN_BOX_AREA", "BYTE_TRACK_MIN_BOX_AREA"),
            TRACK_MIN_BOX_AREA_DEFAULT,
        ),
        0.0,
    )


JobRecord = Dict[str, Any]


class JobNotFoundError(FileNotFoundError):
    """Raised when attempting to operate on a job that is unknown."""


class JobService:
    """Minimal filesystem-backed job tracker."""

    def __init__(self, data_root: Path | str | None = None) -> None:
        self.data_root = (
            Path(data_root).expanduser() if data_root else DEFAULT_DATA_ROOT
        )
        self.jobs_dir = self.data_root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._monitors: Dict[str, threading.Thread] = {}

    # ------------------------------------------------------------------
    def _now(self) -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _read_job(self, job_id: str) -> JobRecord:
        path = self._job_path(job_id)
        if not path.exists():
            raise JobNotFoundError(job_id)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupt file
            raise JobNotFoundError(job_id) from exc

    def _write_job(self, record: JobRecord) -> None:
        path = self._job_path(record["job_id"])
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
        tmp_path.replace(path)

    def _mutate_job(
        self, job_id: str, mutator: Callable[[JobRecord], None]
    ) -> JobRecord:
        with self._lock:
            record = self._read_job(job_id)
            mutator(record)
            self._write_job(record)
            return record

    def _progress_path(self, ep_id: str) -> Path:
        manifests_dir = get_path(ep_id, "detections").parent
        return manifests_dir / "progress.json"

    def _read_progress(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _launch_job(
        self,
        *,
        job_type: str,
        ep_id: str,
        command: list[str],
        progress_path: Path,
        requested: Dict[str, Any],
    ) -> JobRecord:
        ensure_dirs(ep_id)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            progress_path.unlink()
        except FileNotFoundError:
            # Old progress files are optional; ignore if missing.
            pass

        job_id = uuid.uuid4().hex

        # Create stderr log file to capture early failures (model init, video access, etc.)
        stderr_log_dir = progress_path.parent / "logs"
        stderr_log_dir.mkdir(parents=True, exist_ok=True)
        stderr_log_path = stderr_log_dir / f"job-{job_id}.stderr.log"

        env = os.environ.copy()
        # Open stderr log for subprocess (will be closed automatically when process exits)
        stderr_file = open(stderr_log_path, "w", encoding="utf-8")  # noqa: SIM115
        proc = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stderr=stderr_file,
            stdout=subprocess.DEVNULL,  # Stdout goes to progress.json via ProgressEmitter
        )  # noqa: S603

        record: JobRecord = {
            "job_id": job_id,
            "job_type": job_type,
            "ep_id": ep_id,
            "pid": proc.pid,
            "state": "running",
            "started_at": self._now(),
            "ended_at": None,
            "progress_file": str(progress_path),
            "stderr_log": str(stderr_log_path),
            "command": command,
            "requested": requested,
            "summary": None,
            "error": None,
            "return_code": None,
            "data_root": str(self.data_root),
        }
        with self._lock:
            self._write_job(record)

        monitor = threading.Thread(
            target=self._monitor_process,
            args=(job_id, proc),
            name=f"job-monitor-{job_id}",
            daemon=True,
        )
        monitor.start()
        self._monitors[job_id] = monitor
        return record

    # ------------------------------------------------------------------
    def start_detect_track_job(
        self,
        *,
        ep_id: str,
        stride: int,
        fps: float | None,
        device: str,
        video_path: Path,
        save_frames: bool,
        save_crops: bool,
        jpeg_quality: int,
        detector: str,
        tracker: str,
        max_gap: int | None,
        det_thresh: float | None,
        scene_detector: str,
        scene_threshold: float,
        scene_min_len: int,
        scene_warmup_dets: int,
        track_high_thresh: float | None = None,
        new_track_thresh: float | None = None,
        track_buffer: int | None = None,
        min_box_area: float | None = None,
    ) -> JobRecord:
        if not video_path.exists():
            raise FileNotFoundError(f"Episode video not found: {video_path}")
        detector_value = self._normalize_detector(detector)
        tracker_value = self._normalize_tracker(tracker)
        self.ensure_retinaface_ready(detector_value, device, det_thresh)
        progress_path = self._progress_path(ep_id)
        scene_detector_value = self._normalize_scene_detector(scene_detector)
        scene_min_len = max(int(scene_min_len), 1)
        scene_warmup_dets = max(int(scene_warmup_dets), 0)
        scene_threshold = max(float(scene_threshold), 0.0)
        track_high_value = _resolve_track_high_thresh(track_high_thresh)
        new_track_value = _resolve_new_track_thresh(new_track_thresh)
        track_buffer_value = _resolve_track_buffer(track_buffer)
        min_box_area_value = _resolve_min_box_area(min_box_area)
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "episode_run.py"),
            "--ep-id",
            ep_id,
            "--video",
            str(video_path),
            "--stride",
            str(stride),
            "--device",
            device,
            "--progress-file",
            str(progress_path),
        ]
        if fps and fps > 0:
            command += ["--fps", str(fps)]
        if save_frames:
            command.append("--save-frames")
        if save_crops:
            command.append("--save-crops")
        jpeg_quality = max(1, min(int(jpeg_quality), 100))
        if jpeg_quality != 85:
            command += ["--jpeg-quality", str(jpeg_quality)]
        command += ["--detector", detector_value]
        command += ["--tracker", tracker_value]
        command += ["--track-high-thresh", str(track_high_value)]
        command += ["--new-track-thresh", str(new_track_value)]
        command += ["--track-buffer", str(track_buffer_value)]
        command += ["--min-box-area", str(min_box_area_value)]
        if max_gap is not None:
            command += ["--max-gap", str(max_gap)]
        if det_thresh is not None:
            command += ["--det-thresh", str(det_thresh)]
        command += ["--scene-detector", scene_detector_value]
        command += ["--scene-threshold", str(scene_threshold)]
        command += ["--scene-min-len", str(scene_min_len)]
        command += ["--scene-warmup-dets", str(scene_warmup_dets)]
        requested = {
            "stride": stride,
            "fps": fps,
            "device": device,
            "save_frames": save_frames,
            "save_crops": save_crops,
            "jpeg_quality": jpeg_quality,
            "detector": detector_value,
            "tracker": tracker_value,
            "max_gap": max_gap,
            "det_thresh": det_thresh,
            "scene_detector": scene_detector_value,
            "scene_threshold": scene_threshold,
            "scene_min_len": scene_min_len,
            "scene_warmup_dets": scene_warmup_dets,
            "track_high_thresh": track_high_value,
            "new_track_thresh": new_track_value,
            "track_buffer": track_buffer_value,
            "min_box_area": min_box_area_value,
        }
        return self._launch_job(
            job_type="detect_track",
            ep_id=ep_id,
            command=command,
            progress_path=progress_path,
            requested=requested,
        )

    def start_faces_embed_job(
        self,
        *,
        ep_id: str,
        device: str,
        save_frames: bool,
        save_crops: bool,
        jpeg_quality: int,
        thumb_size: int,
    ) -> JobRecord:
        track_path = get_path(ep_id, "tracks")
        if not track_path.exists():
            raise FileNotFoundError("tracks.jsonl not found; run detect/track first")
        device_value = device or "auto"
        self.ensure_arcface_ready(device_value)
        progress_path = self._progress_path(ep_id)
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "episode_run.py"),
            "--ep-id",
            ep_id,
            "--faces-embed",
            "--device",
            device_value,
            "--progress-file",
            str(progress_path),
        ]
        if save_frames:
            command.append("--save-frames")
        if save_crops:
            command.append("--save-crops")
        jpeg_quality = max(1, min(int(jpeg_quality), 100))
        if jpeg_quality != 85:
            command += ["--jpeg-quality", str(jpeg_quality)]
        command += ["--thumb-size", str(thumb_size)]
        requested = {
            "device": device_value,
            "save_frames": save_frames,
            "save_crops": save_crops,
            "jpeg_quality": jpeg_quality,
            "thumb_size": thumb_size,
        }
        return self._launch_job(
            job_type="faces_embed",
            ep_id=ep_id,
            command=command,
            progress_path=progress_path,
            requested=requested,
        )

    def start_cluster_job(
        self,
        *,
        ep_id: str,
        device: str,
        cluster_thresh: float,
        min_cluster_size: int,
        min_identity_sim: float,
    ) -> JobRecord:
        manifests_dir = get_path(ep_id, "detections").parent
        faces_path = manifests_dir / "faces.jsonl"
        if not faces_path.exists():
            raise FileNotFoundError("faces.jsonl not found; run faces_embed first")
        progress_path = self._progress_path(ep_id)
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "episode_run.py"),
            "--ep-id",
            ep_id,
            "--cluster",
            "--device",
            device,
            "--progress-file",
            str(progress_path),
        ]
        command += ["--cluster-thresh", str(cluster_thresh)]
        command += ["--min-cluster-size", str(min_cluster_size)]
        command += ["--min-identity-sim", str(min_identity_sim)]
        requested = {
            "device": device,
            "cluster_thresh": cluster_thresh,
            "min_cluster_size": min_cluster_size,
            "min_identity_sim": min_identity_sim,
        }
        return self._launch_job(
            job_type="cluster",
            ep_id=ep_id,
            command=command,
            progress_path=progress_path,
            requested=requested,
        )

    def start_episode_cleanup_job(
        self,
        *,
        ep_id: str,
        video_path: Path,
        stride: int,
        fps: float | None,
        device: str,
        embed_device: str,
        save_frames: bool,
        save_crops: bool,
        jpeg_quality: int,
        detector: str,
        tracker: str,
        max_gap: int,
        det_thresh: float | None,
        scene_detector: str,
        scene_threshold: float,
        scene_min_len: int,
        scene_warmup_dets: int,
        cluster_thresh: float,
        min_cluster_size: int,
        min_identity_sim: float,
        thumb_size: int,
        actions: List[str],
        write_back: bool,
    ) -> JobRecord:
        if not video_path.exists():
            raise FileNotFoundError(f"Episode video not found: {video_path}")
        detector_value = self._normalize_detector(detector)
        tracker_value = self._normalize_tracker(tracker)
        scene_detector_value = self._normalize_scene_detector(scene_detector)
        progress_path = self._progress_path(ep_id)
        command: List[str] = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "episode_cleanup.py"),
            "--ep-id",
            ep_id,
            "--video",
            str(video_path),
            "--stride",
            str(stride),
            "--max-gap",
            str(max_gap),
            "--device",
            device,
            "--embed-device",
            embed_device or device,
            "--detector",
            detector_value,
            "--tracker",
            tracker_value,
            "--scene-detector",
            scene_detector_value,
            "--scene-threshold",
            str(scene_threshold),
            "--scene-min-len",
            str(scene_min_len),
            "--scene-warmup-dets",
            str(scene_warmup_dets),
            "--cluster-thresh",
            str(cluster_thresh),
            "--min-cluster-size",
            str(min_cluster_size),
            "--min-identity-sim",
            str(min_identity_sim),
            "--thumb-size",
            str(thumb_size),
            "--jpeg-quality",
            str(jpeg_quality),
            "--progress-file",
            str(progress_path),
        ]
        if fps and fps > 0:
            command += ["--fps", str(fps)]
        if det_thresh is not None:
            command += ["--det-thresh", str(det_thresh)]
        if save_frames:
            command.append("--save-frames")
        else:
            command.append("--no-save-frames")
        if save_crops:
            command.append("--save-crops")
        else:
            command.append("--no-save-crops")
        if not write_back:
            command.append("--no-write-back")
        normalized_actions = [
            action for action in actions if action in CLEANUP_ACTIONS
        ] or list(CLEANUP_ACTIONS)
        command += ["--actions", *normalized_actions]
        requested = {
            "stride": stride,
            "fps": fps,
            "device": device,
            "embed_device": embed_device,
            "save_frames": save_frames,
            "save_crops": save_crops,
            "jpeg_quality": jpeg_quality,
            "detector": detector_value,
            "tracker": tracker_value,
            "max_gap": max_gap,
            "det_thresh": det_thresh,
            "scene_detector": scene_detector_value,
            "scene_threshold": scene_threshold,
            "scene_min_len": scene_min_len,
            "scene_warmup_dets": scene_warmup_dets,
            "cluster_thresh": cluster_thresh,
            "min_cluster_size": min_cluster_size,
            "min_identity_sim": min_identity_sim,
            "thumb_size": thumb_size,
            "actions": normalized_actions,
            "write_back": write_back,
        }
        return self._launch_job(
            job_type="episode_cleanup",
            ep_id=ep_id,
            command=command,
            progress_path=progress_path,
            requested=requested,
        )

    def start_screen_time_job(
        self,
        *,
        ep_id: str,
        quality_min: float | None = None,
        gap_tolerance_s: float | None = None,
        use_video_decode: bool | None = None,
        screen_time_mode: Literal["faces", "tracks"] | None = None,
        edge_padding_s: float | None = None,
        track_coverage_min: float | None = None,
        preset: str | None = None,
    ) -> JobRecord:
        """Start a screen time analysis job for an episode.

        Args:
            ep_id: Episode identifier
            quality_min: Optional minimum face quality threshold (0.0-1.0)
            gap_tolerance_s: Optional gap tolerance in seconds
            use_video_decode: Optional flag to use video decode for timestamps
            screen_time_mode: Optional override for interval calculation
            edge_padding_s: Optional edge padding override for intervals
            track_coverage_min: Optional coverage gate when using track mode
            preset: Optional named preset from the pipeline config

        Returns:
            JobRecord with job metadata

        Raises:
            FileNotFoundError: If required artifacts are missing
        """
        # Validate that required artifacts exist
        manifests_dir = get_path(ep_id, "detections").parent
        faces_path = manifests_dir / "faces.jsonl"
        tracks_path = get_path(ep_id, "tracks")
        identities_path = manifests_dir / "identities.json"

        missing = []
        if not faces_path.exists():
            missing.append("faces.jsonl")
        if not tracks_path.exists():
            missing.append("tracks.jsonl")
        if not identities_path.exists():
            missing.append("identities.json")

        if missing:
            raise FileNotFoundError(
                f"Required artifacts missing for screen time analysis: {', '.join(missing)}"
            )

        # Validate people.json exists for the show
        parts = ep_id.split("-")
        if len(parts) < 2:
            raise ValueError(f"Invalid episode ID format: {ep_id}")
        show_id = parts[0].upper()
        people_path = self.data_root / "shows" / show_id / "people.json"

        if not people_path.exists():
            raise FileNotFoundError(
                f"people.json not found for show {show_id}: {people_path}"
            )

        # Build command
        progress_path = self._progress_path(ep_id)
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "analyze_screen_time.py"),
            "--ep-id",
            ep_id,
            "--progress-file",
            str(progress_path),
        ]

        # Add optional overrides
        requested = {}
        if quality_min is not None:
            command += ["--quality-min", str(quality_min)]
            requested["quality_min"] = quality_min
        if gap_tolerance_s is not None:
            command += ["--gap-tolerance-s", str(gap_tolerance_s)]
            requested["gap_tolerance_s"] = gap_tolerance_s
        if use_video_decode is not None:
            command += ["--use-video-decode", str(use_video_decode).lower()]
            requested["use_video_decode"] = use_video_decode
        if screen_time_mode is not None:
            command += ["--screen-time-mode", screen_time_mode]
            requested["screen_time_mode"] = screen_time_mode
        if edge_padding_s is not None:
            command += ["--edge-padding-s", str(edge_padding_s)]
            requested["edge_padding_s"] = edge_padding_s
        if track_coverage_min is not None:
            command += ["--track-coverage-min", str(track_coverage_min)]
            requested["track_coverage_min"] = track_coverage_min
        if preset:
            command += ["--preset", preset]
            requested["preset"] = preset

        return self._launch_job(
            job_type="screen_time_analyze",
            ep_id=ep_id,
            command=command,
            progress_path=progress_path,
            requested=requested,
        )

    def _monitor_process(self, job_id: str, proc: subprocess.Popen) -> None:
        error_msg: str | None = None
        try:
            return_code = proc.wait()
        except Exception as exc:  # pragma: no cover - rare failure
            return_code = -1
            error_msg = str(exc)
        state = "succeeded" if return_code == 0 and error_msg is None else "failed"
        self._finalize_job(job_id, state, return_code, error_msg)

    def _finalize_job(
        self, job_id: str, state: str, return_code: int, error_msg: str | None
    ) -> None:
        progress_data = None

        def _apply(record: JobRecord) -> None:
            nonlocal progress_data
            if record.get("state") == "canceled":
                return
            record["state"] = state
            record["ended_at"] = self._now()
            record["return_code"] = return_code
            if progress_data is None:
                progress_path = Path(record["progress_file"])
                progress_data = self._read_progress(progress_path)
            if progress_data:
                record["summary"] = progress_data
            if error_msg:
                record["error"] = error_msg
            elif state == "failed" and not record.get("error"):
                record["error"] = f"episode_run exited with code {return_code}"

        try:
            self._mutate_job(job_id, _apply)
        except JobNotFoundError:
            return

    # ------------------------------------------------------------------
    def get(self, job_id: str) -> JobRecord:
        return self._read_job(job_id)

    def get_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        record = self._read_job(job_id)
        return self._read_progress(Path(record["progress_file"]))

    def cancel(self, job_id: str) -> JobRecord:
        def _apply(record: JobRecord) -> None:
            if record.get("state") != "running":
                return
            pid = record.get("pid")
            if isinstance(pid, int) and pid > 0:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    # Process already exited on its own.
                    pass
                except PermissionError:
                    # PID belongs to another user; nothing else to do.
                    pass
            record["state"] = "canceled"
            record["ended_at"] = self._now()
            record["error"] = record.get("error") or "Job canceled by user"
            progress_data = self._read_progress(Path(record["progress_file"]))
            if progress_data:
                record["summary"] = progress_data

        return self._mutate_job(job_id, _apply)

    def list_jobs(
        self,
        *,
        ep_id: str | None = None,
        job_type: str | None = None,
        limit: int = 50,
    ) -> list[JobRecord]:
        """List all jobs, optionally filtered by episode and/or job type."""
        jobs_dir = self.data_root / "jobs"
        if not jobs_dir.exists():
            return []

        all_jobs: list[JobRecord] = []
        for job_file in jobs_dir.glob("*.json"):
            try:
                job = self._read_job(job_file.stem)
                # Apply filters
                if ep_id and job.get("ep_id") != ep_id:
                    continue
                if job_type and job.get("job_type") != job_type:
                    continue
                all_jobs.append(job)
            except Exception:
                # Skip corrupted job files
                continue

        # Sort by started_at descending (most recent first)
        all_jobs.sort(key=lambda j: j.get("started_at", ""), reverse=True)

        return all_jobs[:limit]

    def _normalize_detector(self, detector: str | None) -> str:
        fallback = DEFAULT_DETECTOR_ENV or "retinaface"
        value = (detector or fallback).strip().lower()
        if value not in DETECTOR_CHOICES:
            raise ValueError(f"Unsupported detector '{detector}'")
        return value

    def _normalize_tracker(self, tracker: str | None) -> str:
        fallback = DEFAULT_TRACKER_ENV or "bytetrack"
        value = (tracker or fallback).strip().lower()
        if value not in TRACKER_CHOICES:
            raise ValueError(f"Unsupported tracker '{tracker}'")
        return value

    def _normalize_scene_detector(self, scene_detector: str | None) -> str:
        fallback = SCENE_DETECTOR_DEFAULT or "pyscenedetect"
        value = (scene_detector or fallback).strip().lower()
        if value not in SCENE_DETECTOR_CHOICES:
            raise ValueError(f"Unsupported scene detector '{scene_detector}'")
        return value

    def ensure_retinaface_ready(
        self,
        detector: str,
        device: str,
        det_thresh: float | None,
    ) -> None:
        if detector != "retinaface":
            return
        if episode_run is None:
            raise ValueError(
                "RetinaFace validation unavailable: install the ML stack (pip install -r requirements-ml.txt) "
                "before running RetinaFace."
            )
        ok, error_detail, _ = episode_run.ensure_retinaface_ready(
            device,
            det_thresh if det_thresh is not None else None,
        )
        if ok:
            return
        message = episode_run.RETINAFACE_HELP
        if error_detail:
            message = f"{message} ({error_detail})"
        raise ValueError(message)

    def ensure_arcface_ready(self, device: str) -> None:
        if episode_run is None:
            raise ValueError(
                "ArcFace validation unavailable: install the ML stack (pip install -r requirements-ml.txt) "
                "before running face embedding."
            )
        ok, error_detail, _ = episode_run.ensure_arcface_ready(device)
        if ok:
            return
        message = getattr(
            episode_run,
            "ARC_FACE_HELP",
            "ArcFace weights missing or could not initialize. See README 'Models' or run scripts/fetch_models.py.",
        )
        if error_detail:
            message = f"{message} ({error_detail})"
        raise ValueError(message)

    def emit_facebank_refresh(
        self,
        show_id: str,
        cast_id: str,
        *,
        action: str,
        seed_ids: list[str] | None = None,
    ) -> JobRecord:
        payload = {
            "action": action,
            "show_id": show_id,
            "cast_id": cast_id,
            "seed_ids": list(seed_ids or []),
        }
        job_id = f"facebank-refresh-{uuid.uuid4().hex[:10]}"
        now = self._now()
        record: JobRecord = {
            "job_id": job_id,
            "ep_id": f"{show_id}:{cast_id}",
            "job_type": "facebank_refresh",
            "pid": None,
            "state": "succeeded",
            "command": [],
            "started_at": now,
            "ended_at": now,
            "progress_file": None,
            "requested": payload,
            "summary": payload,
            "error": None,
            "return_code": 0,
            "data_root": str(self.data_root),
        }
        self._write_job(record)
        return record


__all__ = ["JobService", "JobNotFoundError", "JobRecord"]
