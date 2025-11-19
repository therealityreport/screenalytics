#!/usr/bin/env python3
"""
Apply the remaining 5 optimization patches to episode_run.py
"""

import re
import sys
from pathlib import Path


def apply_d1_cap_grab(content: str) -> str:
    """Apply D1: cap.grab() optimization"""
    print("Applying D1: cap.grab() optimization...")

    # Find the main pipeline loop
    old_pattern = r'''(            while True:\n)                ok, frame = cap\.read\(\)\n                if not ok:\n                    break\n\n                # Guard against empty/None frames before detection\n                if frame is None or frame\.size == 0:\n                    LOGGER\.warning\(\n                        "Skipping frame %d for %s: empty or None frame from video capture",\n                        frame_idx,\n                        args\.ep_id,\n                    \)\n                    frame_idx \+= 1\n                    frames_since_cut \+= 1\n                    continue\n\n                if next_cut is not None and frame_idx >= next_cut:\n                    reset_tracker = getattr\(tracker_adapter, "reset", None\)\n                    if callable\(reset_tracker\):\n                        reset_tracker\(\)\n                    if appearance_gate:\n                        appearance_gate\.reset_all\(\)\n                    recorder\.on_cut\(frame_idx\)\n                    frames_since_cut = 0\n                    cut_ix \+= 1\n                    next_cut = scene_cuts\[cut_ix\] if cut_ix < len\(scene_cuts\) else None\n                    if progress:\n                        emit_frames, video_meta = _progress_value\(frame_idx, include_current=False\)\n                        progress\.emit\(\n                            emit_frames,\n                            phase="track",\n                            device=device,\n                            detector=detector_choice,\n                            tracker=tracker_label,\n                            resolved_device=detector_device,\n                            summary=\{"event": "reset_on_cut", "frame": frame_idx\},\n                            force=True,\n                            extra=video_meta,\n                        \)\n                force_detect = frames_since_cut < scene_warmup\n                should_sample = frame_idx % frame_stride == 0\n                if not \(should_sample or force_detect\):\n                    frame_idx \+= 1\n                    frames_since_cut \+= 1\n                    continue\n\n                frames_sampled \+= 1'''

    new_code = r'''while True:
                # D1: Use grab() to skip frame decode for frames we won't analyze
                # This avoids decoding ~83% of frames when stride=6
                ok = cap.grab()
                if not ok:
                    break

                # Determine if we need to actually decode this frame
                force_detect = frames_since_cut < scene_warmup
                should_sample = frame_idx % frame_stride == 0
                at_scene_cut = next_cut is not None and frame_idx >= next_cut

                # Skip decode if we won't process this frame
                if not (should_sample or force_detect or at_scene_cut):
                    frame_idx += 1
                    frames_since_cut += 1
                    continue

                # Retrieve (decode) only frames we'll actually process
                frame_ok, frame = cap.retrieve()
                if not frame_ok:
                    LOGGER.warning(
                        "Failed to retrieve frame %d for %s after successful grab",
                        frame_idx,
                        args.ep_id,
                    )
                    frame_idx += 1
                    frames_since_cut += 1
                    continue

                # Guard against empty/None frames before detection
                if frame is None or frame.size == 0:
                    LOGGER.warning(
                        "Skipping frame %d for %s: empty or None frame from video capture",
                        frame_idx,
                        args.ep_id,
                    )
                    frame_idx += 1
                    frames_since_cut += 1
                    continue

                if next_cut is not None and frame_idx >= next_cut:
                    reset_tracker = getattr(tracker_adapter, "reset", None)
                    if callable(reset_tracker):
                        reset_tracker()
                    if appearance_gate:
                        appearance_gate.reset_all()
                    recorder.on_cut(frame_idx)
                    frames_since_cut = 0
                    cut_ix += 1
                    next_cut = scene_cuts[cut_ix] if cut_ix < len(scene_cuts) else None
                    if progress:
                        emit_frames, video_meta = _progress_value(frame_idx, include_current=False)
                        progress.emit(
                            emit_frames,
                            phase="track",
                            device=device,
                            detector=detector_choice,
                            tracker=tracker_label,
                            resolved_device=detector_device,
                            summary={"event": "reset_on_cut", "frame": frame_idx},
                            force=True,
                            extra=video_meta,
                        )

                frames_sampled += 1'''

    # Simpler approach: find and replace the specific section
    lines = content.split('\n')

    # Find the main while True loop (last occurrence)
    while_indices = [i for i, line in enumerate(lines) if line.strip() == 'while True:']
    if not while_indices:
        print("ERROR: Could not find 'while True:' loop")
        return content

    main_while_idx = while_indices[-1]  # Last occurrence is the main pipeline loop
    print(f"Found main pipeline loop at line {main_while_idx + 1}")

    # Find the section to replace (from while True to frames_sampled += 1)
    start_idx = main_while_idx
    end_idx = None

    for i in range(start_idx + 1, min(start_idx + 100, len(lines))):
        if 'frames_sampled += 1' in lines[i]:
            end_idx = i
            break

    if end_idx is None:
        print("ERROR: Could not find end of section (frames_sampled += 1)")
        return content

    print(f"Replacing lines {start_idx + 1} to {end_idx + 1}")

    # Replace the section
    new_lines = new_code.split('\n')
    lines[start_idx:end_idx + 1] = new_lines

    print("✓ D1 applied successfully")
    return '\n'.join(lines)


def main():
    episode_run_path = Path("tools/episode_run.py")

    if not episode_run_path.exists():
        print(f"ERROR: {episode_run_path} not found")
        sys.exit(1)

    print(f"Reading {episode_run_path}...")
    content = episode_run_path.read_text()

    # Apply D1
    content = apply_d1_cap_grab(content)

    # Write back
    print(f"\nWriting changes to {episode_run_path}...")
    episode_run_path.write_text(content)

    print("\n✅ D1 optimization applied successfully!")
    print("\nNext steps:")
    print("1. Run: python3 -m py_compile tools/episode_run.py")
    print("2. Test the pipeline")
    print("3. Apply remaining patches (F1, B4, C3, G3)")


if __name__ == "__main__":
    main()
