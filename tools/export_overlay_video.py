#!/usr/bin/env python3
"""Export full episode video with face overlay annotations.

This script generates a full video with bounding boxes and name labels
for all detected and identified faces throughout the episode.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from py_screenalytics.artifacts import get_path  # noqa: E402

LOGGER = logging.getLogger(__name__)


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in PATH."""
    return shutil.which("ffmpeg") is not None


def mux_audio_with_ffmpeg(
    video_path: Path,
    source_video_path: Path,
    output_path: Path,
    output_fps: float,
) -> bool:
    """Mux audio from source video into the overlay video using ffmpeg.

    Args:
        video_path: Path to video-only overlay file
        source_video_path: Path to original video with audio
        output_path: Path to write final video with audio

    Returns:
        True if successful, False otherwise
    """
    if not check_ffmpeg_available():
        LOGGER.warning("[export] ffmpeg not found, skipping audio mux")
        return False

    try:
        # Build ffmpeg command to combine video from overlay with audio from source
        # -i video_path: input video (overlay, no audio)
        # -i source_video_path: input source (for audio)
        # -c:v copy: copy video stream without re-encoding
        # -c:a aac: encode audio as AAC for compatibility
        # -map 0:v:0: take video from first input
        # -map 1:a:0?: take audio from second input (? = optional, don't fail if no audio)
        # -shortest: end when shortest stream ends
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", str(video_path),  # Video-only overlay
            "-i", str(source_video_path),  # Original with audio
            "-c:v", "copy",  # Copy video stream (no re-encode)
            "-c:a", "aac",  # Encode audio as AAC
            "-b:a", "192k",  # Audio bitrate
            "-map", "0:v:0",  # Video from first input
            "-map", "1:a:0?",  # Audio from second input (optional)
            "-shortest",  # End when shortest stream ends
            str(output_path),
        ]

        LOGGER.info(f"[export] Running ffmpeg to mux audio...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            LOGGER.warning(f"[export] ffmpeg failed: {result.stderr[:500]}")
            return False

        LOGGER.info("[export] Audio mux complete")
        return True

    except subprocess.TimeoutExpired:
        LOGGER.warning("[export] ffmpeg timeout during audio mux")
        return False
    except Exception as e:
        LOGGER.warning(f"[export] ffmpeg error: {e}")
        return False


def update_progress(progress_file: Path, data: dict) -> None:
    """Write progress update to file."""
    try:
        with progress_file.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    except IOError as e:
        LOGGER.warning(f"Failed to write progress: {e}")


def load_faces(ep_id: str) -> List[Dict[str, Any]]:
    """Load faces.jsonl for an episode."""
    faces_path = get_path(ep_id, "detections").parent / "faces.jsonl"
    if not faces_path.exists():
        raise FileNotFoundError(f"faces.jsonl not found: {faces_path}")

    faces = []
    with faces_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                face = json.loads(line)
                faces.append(face)
            except json.JSONDecodeError:
                continue
    return faces


def load_identities(ep_id: str) -> Dict[str, Any]:
    """Load identities.json for an episode."""
    identities_path = get_path(ep_id, "detections").parent / "identities.json"
    if not identities_path.exists():
        return {"identities": []}

    with identities_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_people(show_id: str) -> List[Dict[str, Any]]:
    """Load people.json for a show."""
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
    people_path = data_root / "shows" / show_id / "people.json"

    if not people_path.exists():
        return []

    with people_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("people", [])


def load_cast_members(show_id: str) -> List[Dict[str, Any]]:
    """Load cast members for name fallback."""
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
    cast_path = data_root / "shows" / show_id / "cast.json"

    if not cast_path.exists():
        return []

    with cast_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cast", [])


def export_overlay_video(
    ep_id: str,
    output_path: Path,
    progress_file: Optional[Path] = None,
    include_unidentified: bool = True,
    output_fps: Optional[float] = None,
) -> Dict[str, Any]:
    """Export full episode video with face overlays.

    Args:
        ep_id: Episode identifier
        output_path: Path to write output video
        progress_file: Optional path for progress updates
        include_unidentified: Include faces without cast assignment
        output_fps: Output FPS (default: source FPS / 2 for smaller file)

    Returns:
        Dict with export metadata
    """
    LOGGER.info(f"[export] Starting overlay video export for {ep_id}")

    if progress_file:
        update_progress(progress_file, {"phase": "init", "message": "Initializing..."})

    # Get video path
    video_path = get_path(ep_id, "video")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video file")

    try:
        # Get video metadata
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0:
            source_fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / source_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Use source FPS if not specified
        if output_fps is None:
            output_fps = source_fps

        LOGGER.info(
            f"[export] Video: {total_frames} frames, {source_fps:.1f} fps, "
            f"{video_duration:.1f}s, {width}x{height}"
        )

        if progress_file:
            update_progress(progress_file, {
                "phase": "loading",
                "message": "Loading face data...",
                "total_frames": total_frames,
            })

        # Load faces
        faces = load_faces(ep_id)
        LOGGER.info(f"[export] Loaded {len(faces)} face detections")

        # Index faces by frame
        faces_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for face in faces:
            frame_idx = face.get("frame_idx")
            if frame_idx is not None:
                faces_by_frame.setdefault(frame_idx, []).append(face)

        # Build track -> identity -> person -> name mapping
        identities_data = load_identities(ep_id)
        identities_list = identities_data.get("identities", [])

        track_to_identity: Dict[int, Dict[str, Any]] = {}
        for identity in identities_list:
            identity_id = identity.get("identity_id")
            for track_id in identity.get("track_ids", []) or []:
                try:
                    track_to_identity[int(track_id)] = identity
                except (TypeError, ValueError):
                    continue

        # Load people and cast for name mapping
        parts = ep_id.split("-")
        show_id = parts[0].upper() if len(parts) >= 2 else ""
        people = load_people(show_id)
        cast_members = load_cast_members(show_id)

        # Build person_id -> name map with cast fallback
        cast_name_lookup = {c.get("cast_id"): c.get("name") for c in cast_members if c.get("cast_id")}
        person_to_name: Dict[str, str] = {}
        for person in people:
            person_id = person.get("person_id")
            name = person.get("name")
            cast_id = person.get("cast_id")
            if person_id:
                if name and name != "None":
                    person_to_name[person_id] = name
                elif cast_id and cast_id in cast_name_lookup:
                    person_to_name[person_id] = cast_name_lookup[cast_id]

        LOGGER.info(f"[export] Mapped {len(person_to_name)} people to names")

        # Colors for different tracks
        colors = [
            (66, 133, 244),   # Blue
            (52, 168, 83),    # Green
            (251, 188, 4),    # Yellow
            (234, 67, 53),    # Red
            (154, 0, 255),    # Purple
            (0, 188, 212),    # Cyan
            (255, 152, 0),    # Orange
            (156, 39, 176),   # Deep Purple
        ]
        gray_color = (128, 128, 128)

        if progress_file:
            update_progress(progress_file, {
                "phase": "encoding",
                "message": "Encoding video...",
                "current_frame": 0,
                "total_frames": total_frames,
                "percent": 0,
            })

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temp file first, then mux audio from source
        temp_video_path = output_path.with_suffix(".temp.mp4")

        # Initialize video writer - write to temp file (video only)
        # Try avc1 (H.264) first for browser compatibility, fallback to mp4v
        frame_step = max(1, int(source_fps / output_fps))
        writer = None
        for codec in ["avc1", "H264", "mp4v"]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(temp_video_path), fourcc, output_fps, (width, height))
            if writer.isOpened():
                LOGGER.info(f"[export] Using codec: {codec}")
                break
            writer.release()
            writer = None

        if writer is None or not writer.isOpened():
            raise RuntimeError("Could not create video writer with any codec")

        try:
            frames_written = 0
            total_faces_rendered = 0
            stride = 8  # Detection stride from tracker
            missing_label_count = 0

            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Skip frames for output FPS
                if frame_idx % frame_step != 0:
                    continue

                # Update progress every 100 frames
                if frame_idx % 100 == 0:
                    percent = int((frame_idx / total_frames) * 100)
                    if progress_file:
                        update_progress(progress_file, {
                            "phase": "encoding",
                            "message": f"Encoding frame {frame_idx}/{total_frames}...",
                            "current_frame": frame_idx,
                            "total_frames": total_frames,
                            "percent": percent,
                        })

                # Find faces for this frame (check nearby frames within stride)
                frame_faces = []
                for check_frame in range(frame_idx - stride, frame_idx + stride + 1):
                    if check_frame in faces_by_frame:
                        frame_faces = faces_by_frame[check_frame]
                        break

                track_ids_seen = set()
                for face in frame_faces:
                    track_id = face.get("track_id")
                    bbox = face.get("bbox_xyxy")
                    if track_id is None or not bbox:
                        continue
                    if track_id in track_ids_seen:
                        continue
                    track_ids_seen.add(track_id)

                    # Resolve identity chain
                    identity = track_to_identity.get(track_id)
                    identity_id = identity.get("identity_id") if identity else None
                    identity_label = identity.get("label") if identity else None
                    person_id = identity.get("person_id") if identity else None
                    name = person_to_name.get(person_id) if person_id else None
                    face_name = face.get("name") or face.get("display_name")
                    face_cast_id = face.get("cast_id")
                    face_cast_name = cast_name_lookup.get(face_cast_id) if face_cast_id else None

                    # Determine whether this face should be treated as identified
                    is_identified = bool(name or identity_label or face_cast_name or face_name or person_id)

                    # Skip unidentified if not requested
                    if not include_unidentified and not is_identified:
                        continue

                    total_faces_rendered += 1

                    # Choose color
                    color = colors[track_id % len(colors)] if is_identified else gray_color

                    # Draw bbox
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Create label
                        label = (
                            name
                            or identity_label
                            or face_cast_name
                            or face_name
                        )
                        if not label and person_id:
                            label = f"Person {person_id}"
                        if not label and identity_id:
                            label = f"[{identity_id[:8]}]"
                        if not label:
                            missing_label_count += 1
                            label = f"T{track_id}"

                        font_scale = 0.6
                        thickness = 2
                        (label_w, label_h), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                        )

                        # Draw label background
                        cv2.rectangle(
                            frame, (x1, y1 - label_h - 6), (x1 + label_w + 6, y1), color, -1
                        )
                        cv2.putText(
                            frame, label, (x1 + 3, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
                        )
                    except (TypeError, ValueError):
                        continue

                # Add timestamp overlay
                current_time = frame_idx / source_fps
                mins = int(current_time // 60)
                secs = current_time % 60
                timestamp_text = f"{mins}:{secs:05.2f}"
                cv2.putText(
                    frame, timestamp_text, (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

                writer.write(frame)
                frames_written += 1

        finally:
            writer.release()

        if frames_written == 0:
            temp_video_path.unlink(missing_ok=True)
            raise RuntimeError("No frames written to output")

        LOGGER.info(f"[export] Wrote {frames_written} frames, {total_faces_rendered} face annotations")
        if missing_label_count:
            LOGGER.info(f"[export] {missing_label_count} faces missing cast/person names (used IDs/tracks)")

        # Mux audio from source video into final output
        has_audio = False
        if progress_file:
            update_progress(progress_file, {
                "phase": "muxing",
                "message": "Adding audio...",
                "percent": 95,
            })

        if mux_audio_with_ffmpeg(temp_video_path, video_path, output_path, output_fps):
            has_audio = True
            # Clean up temp file
            temp_video_path.unlink(missing_ok=True)
            LOGGER.info("[export] Audio muxed successfully")
        else:
            # If audio mux fails, just rename temp file to output
            LOGGER.warning("[export] Audio mux failed, using video-only output")
            if output_path.exists():
                output_path.unlink()
            temp_video_path.rename(output_path)

        if progress_file:
            update_progress(progress_file, {
                "phase": "done",
                "message": "Export complete" + (" with audio" if has_audio else " (no audio)"),
                "percent": 100,
                "output_path": str(output_path),
            })

        return {
            "output_path": str(output_path),
            "duration_s": round(video_duration, 2),
            "frames_written": frames_written,
            "faces_rendered": total_faces_rendered,
            "output_fps": output_fps,
            "width": width,
            "height": height,
            "has_audio": has_audio,
        }

    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description="Export video with face overlays")
    parser.add_argument("--ep-id", required=True, help="Episode identifier")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--progress-file", help="Path to write progress updates")
    parser.add_argument("--include-unidentified", type=bool, default=True)
    parser.add_argument("--output-fps", type=float, default=None)
    parser.add_argument("--upload-s3", action="store_true", help="Upload to S3 after export")
    parser.add_argument("--s3-key", help="S3 key for upload")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    progress_file = Path(args.progress_file) if args.progress_file else None
    output_path = Path(args.output)

    try:
        result = export_overlay_video(
            ep_id=args.ep_id,
            output_path=output_path,
            progress_file=progress_file,
            include_unidentified=args.include_unidentified,
            output_fps=args.output_fps,
        )

        # Upload to S3 if requested
        if args.upload_s3 and args.s3_key and output_path.exists():
            LOGGER.info(f"[export] Uploading to S3: {args.s3_key}")
            if progress_file:
                update_progress(progress_file, {
                    "phase": "uploading",
                    "message": "Uploading to S3...",
                    "percent": 98,
                })

            from apps.api.services.storage import StorageService

            storage = StorageService()
            if storage.backend in {"s3", "minio"} and storage._client is not None:
                extra_args = {"ContentType": "video/mp4"}
                storage._client.upload_file(
                    str(output_path),
                    storage.bucket,
                    args.s3_key,
                    ExtraArgs=extra_args,
                )
                url = storage.presign_get(args.s3_key, expires_in=86400)  # 24 hour URL
                result["s3_key"] = args.s3_key
                result["url"] = url
                LOGGER.info(f"[export] Uploaded to S3: {url}")

                if progress_file:
                    update_progress(progress_file, {
                        "phase": "done",
                        "message": "Export and upload complete",
                        "percent": 100,
                        "s3_key": args.s3_key,
                        "url": url,
                    })
            else:
                # No upload client available; keep progress in done state with local path
                if progress_file:
                    update_progress(progress_file, {
                        "phase": "done",
                        "message": "Export complete (local storage)",
                        "percent": 100,
                        "output_path": str(output_path),
                    })
        elif progress_file:
            # Ensure we leave progress in a done state if upload not requested
            update_progress(progress_file, {
                "phase": "done",
                "message": "Export complete",
                "percent": 100,
                "output_path": str(output_path),
            })

        print(json.dumps(result, indent=2))

    except Exception as e:
        LOGGER.error(f"[export] Failed: {e}")
        if progress_file:
            update_progress(progress_file, {
                "phase": "error",
                "message": str(e),
            })
        sys.exit(1)


if __name__ == "__main__":
    main()
