#!/usr/bin/env python3
"""Apply D1: cap.grab() optimization with correct indentation"""

from pathlib import Path


def main():
    episode_run = Path("tools/episode_run.py")
    content = episode_run.read_text()
    lines = content.split('\n')

    # Find the line with "while True:" after "with det_path.open"
    det_path_line = None
    while_true_line = None

    for i, line in enumerate(lines):
        if 'with det_path.open("w", encoding="utf-8") as det_handle:' in line:
            det_path_line = i
        if det_path_line and i > det_path_line and line.strip() == 'while True:':
            while_true_line = i
            break

    if while_true_line is None:
        print("ERROR: Could not find while True loop")
        return

    print(f"Found while True at line {while_true_line + 1}")

    # Find the end (frames_sampled += 1)
    end_line = None
    for i in range(while_true_line + 1, min(while_true_line + 100, len(lines))):
        if 'frames_sampled += 1' in lines[i]:
            end_line = i
            break

    if end_line is None:
        print("ERROR: Could not find frames_sampled += 1")
        return

    print(f"Replacing lines {while_true_line + 1} to {end_line + 1}")

    # New code with proper indentation (12 spaces for while, 16 for content)
    new_code = [
        "            while True:",
        "                # D1: Use grab() to skip frame decode for frames we won't analyze",
        "                # This avoids decoding ~83% of frames when stride=6",
        "                ok = cap.grab()",
        "                if not ok:",
        "                    break",
        "",
        "                # Determine if we need to actually decode this frame",
        "                force_detect = frames_since_cut < scene_warmup",
        "                should_sample = frame_idx % frame_stride == 0",
        "                at_scene_cut = next_cut is not None and frame_idx >= next_cut",
        "",
        "                # Skip decode if we won't process this frame",
        "                if not (should_sample or force_detect or at_scene_cut):",
        "                    frame_idx += 1",
        "                    frames_since_cut += 1",
        "                    continue",
        "",
        "                # Retrieve (decode) only frames we'll actually process",
        "                frame_ok, frame = cap.retrieve()",
        "                if not frame_ok:",
        "                    LOGGER.warning(",
        "                        \"Failed to retrieve frame %d for %s after successful grab\",",
        "                        frame_idx,",
        "                        args.ep_id,",
        "                    )",
        "                    frame_idx += 1",
        "                    frames_since_cut += 1",
        "                    continue",
        "",
        "                # Guard against empty/None frames before detection",
        "                if frame is None or frame.size == 0:",
        "                    LOGGER.warning(",
        "                        \"Skipping frame %d for %s: empty or None frame from video capture\",",
        "                        frame_idx,",
        "                        args.ep_id,",
        "                    )",
        "                    frame_idx += 1",
        "                    frames_since_cut += 1",
        "                    continue",
        "",
        "                if next_cut is not None and frame_idx >= next_cut:",
        "                    reset_tracker = getattr(tracker_adapter, \"reset\", None)",
        "                    if callable(reset_tracker):",
        "                        reset_tracker()",
        "                    if appearance_gate:",
        "                        appearance_gate.reset_all()",
        "                    recorder.on_cut(frame_idx)",
        "                    frames_since_cut = 0",
        "                    cut_ix += 1",
        "                    next_cut = scene_cuts[cut_ix] if cut_ix < len(scene_cuts) else None",
        "                    if progress:",
        "                        emit_frames, video_meta = _progress_value(frame_idx, include_current=False)",
        "                        progress.emit(",
        "                            emit_frames,",
        "                            phase=\"track\",",
        "                            device=device,",
        "                            detector=detector_choice,",
        "                            tracker=tracker_label,",
        "                            resolved_device=detector_device,",
        "                            summary={\"event\": \"reset_on_cut\", \"frame\": frame_idx},",
        "                            force=True,",
        "                            extra=video_meta,",
        "                        )",
        "",
        "                frames_sampled += 1",
    ]

    # Replace the section
    lines[while_true_line:end_line + 1] = new_code

    # Write back
    episode_run.write_text('\n'.join(lines))
    print("✓ D1 optimization applied successfully!")


if __name__ == "__main__":
    main()
