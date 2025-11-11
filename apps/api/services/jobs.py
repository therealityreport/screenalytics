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
from typing import Any, Callable, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from py_screenalytics.artifacts import ensure_dirs, get_path

try:  # pragma: no cover - optional ML stack
    from tools import episode_run  # type: ignore
except ModuleNotFoundError:
    episode_run = None  # type: ignore[assignment]
DEFAULT_DATA_ROOT = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
DETECTOR_CHOICES = {"retinaface", "yolov8face"}
TRACKER_CHOICES = {"bytetrack", "strongsort"}
DEFAULT_DETECTOR_ENV = os.getenv("DEFAULT_DETECTOR", "retinaface").lower()
DEFAULT_TRACKER_ENV = os.getenv("DEFAULT_TRACKER", "bytetrack").lower()

JobRecord = Dict[str, Any]


class JobNotFoundError(FileNotFoundError):
    """Raised when attempting to operate on a job that is unknown."""


class JobService:
    """Minimal filesystem-backed job tracker."""

    def __init__(self, data_root: Path | str | None = None) -> None:
        self.data_root = Path(data_root).expanduser() if data_root else DEFAULT_DATA_ROOT
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
        tmp_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)

    def _mutate_job(self, job_id: str, mutator: Callable[[JobRecord], None]) -> JobRecord:
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
            pass

        job_id = uuid.uuid4().hex
        env = os.environ.copy()
        proc = subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=env)  # noqa: S603

        record: JobRecord = {
            "job_id": job_id,
            "job_type": job_type,
            "ep_id": ep_id,
            "pid": proc.pid,
            "state": "running",
            "started_at": self._now(),
            "ended_at": None,
            "progress_file": str(progress_path),
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
        stub: bool,
        device: str,
        video_path: Path,
        save_frames: bool,
        save_crops: bool,
        jpeg_quality: int,
        detector: str,
        tracker: str,
        max_gap: int | None,
        det_thresh: float | None,
    ) -> JobRecord:
        if not video_path.exists():
            raise FileNotFoundError(f"Episode video not found: {video_path}")
        detector_value = self._normalize_detector(detector)
        tracker_value = self._normalize_tracker(tracker)
        self.ensure_retinaface_ready(detector_value, stub, device, det_thresh)
        progress_path = self._progress_path(ep_id)
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
        if stub:
            command.append("--stub")
        if save_frames:
            command.append("--save-frames")
        if save_crops:
            command.append("--save-crops")
        jpeg_quality = max(1, min(int(jpeg_quality), 100))
        if jpeg_quality != 85:
            command += ["--jpeg-quality", str(jpeg_quality)]
        command += ["--detector", detector_value]
        command += ["--tracker", tracker_value]
        if max_gap is not None:
            command += ["--max-gap", str(max_gap)]
        if det_thresh is not None:
            command += ["--det-thresh", str(det_thresh)]
        requested = {
            "stride": stride,
            "fps": fps,
            "stub": stub,
            "device": device,
            "save_frames": save_frames,
            "save_crops": save_crops,
            "jpeg_quality": jpeg_quality,
            "detector": detector_value,
            "tracker": tracker_value,
            "max_gap": max_gap,
            "det_thresh": det_thresh,
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
        stub: bool,
        device: str,
        save_frames: bool,
        save_crops: bool,
        jpeg_quality: int,
        thumb_size: int,
    ) -> JobRecord:
        track_path = get_path(ep_id, "tracks")
        if not track_path.exists():
            raise FileNotFoundError("tracks.jsonl not found; run detect/track first")
        progress_path = self._progress_path(ep_id)
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "episode_run.py"),
            "--ep-id",
            ep_id,
            "--faces-embed",
            "--device",
            device,
            "--progress-file",
            str(progress_path),
        ]
        if stub:
            command.append("--stub")
        if save_frames:
            command.append("--save-frames")
        if save_crops:
            command.append("--save-crops")
        jpeg_quality = max(1, min(int(jpeg_quality), 100))
        if jpeg_quality != 85:
            command += ["--jpeg-quality", str(jpeg_quality)]
        command += ["--thumb-size", str(thumb_size)]
        requested = {
            "stub": stub,
            "device": device,
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
        stub: bool,
        device: str,
        cluster_thresh: float,
        min_cluster_size: int,
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
        if stub:
            command.append("--stub")
        command += ["--cluster-thresh", str(cluster_thresh)]
        command += ["--min-cluster-size", str(min_cluster_size)]
        requested = {
            "stub": stub,
            "device": device,
            "cluster_thresh": cluster_thresh,
            "min_cluster_size": min_cluster_size,
        }
        return self._launch_job(
            job_type="cluster",
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

    def _finalize_job(self, job_id: str, state: str, return_code: int, error_msg: str | None) -> None:
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
                    pass
                except PermissionError:
                    pass
            record["state"] = "canceled"
            record["ended_at"] = self._now()
            record["error"] = record.get("error") or "Job canceled by user"
            progress_data = self._read_progress(Path(record["progress_file"]))
            if progress_data:
                record["summary"] = progress_data

        return self._mutate_job(job_id, _apply)

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

    def ensure_retinaface_ready(
        self,
        detector: str,
        stub: bool,
        device: str,
        det_thresh: float | None,
    ) -> None:
        if stub or detector != "retinaface":
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


__all__ = ["JobService", "JobNotFoundError", "JobRecord"]
