#!/usr/bin/env python3
"""Apply F1: Scene cut cooldown optimization"""

from pathlib import Path


def main():
    episode_run = Path("tools/episode_run.py")
    content = episode_run.read_text()
    lines = content.split('\n')

    print("Applying F1: Scene cut cooldown optimization...")

    # STEP 1: Add CLI argument after --scene-warmup-dets
    for i, line in enumerate(lines):
        if '"--scene-warmup-dets",' in line:
            # Find the closing parenthesis of this argument block
            for j in range(i, i + 10):
                if ')' in lines[j] and 'parser.add_argument' in lines[j + 1]:
                    # Insert new argument here
                    new_arg = [
                        "    parser.add_argument(",
                        "        \"--scene-cut-cooldown\",",
                        "        type=int,",
                        "        default=24,",
                        "        help=\"Minimum frames between scene cut resets (default: 24, prevents reset thrashing)\",",
                        "    )",
                    ]
                    lines[j + 1:j + 1] = new_arg
                    print(f"✓ Added --scene-cut-cooldown CLI argument at line {j + 2}")
                    break
            break

    # STEP 2: Add tracking variables
    # Find the line with "frames_since_cut = 10**9"
    for i, line in enumerate(lines):
        if 'frames_since_cut = 10**9' in line:
            # Insert after this line
            new_vars = [
                "    last_cut_reset = -999  # F1: Track last reset to prevent thrashing",
                "    scene_cut_cooldown = getattr(args, \"scene_cut_cooldown\", 24)",
            ]
            lines[i + 1:i + 1] = new_vars
            print(f"✓ Added tracking variables at line {i + 2}")
            break

    # STEP 3: Update scene cut logic
    # Find "if next_cut is not None and frame_idx >= next_cut:" in the main loop
    in_scene_cut_block = False
    scene_cut_start = None
    scene_cut_end = None

    for i, line in enumerate(lines):
        if 'if next_cut is not None and frame_idx >= next_cut:' in line:
            # Make sure we're in the main loop (after det_path.open)
            # Check if we've seen det_path.open before this
            found_det_path = False
            for j in range(i - 100, i):
                if j >= 0 and 'with det_path.open' in lines[j]:
                    found_det_path = True
                    break

            if found_det_path:
                scene_cut_start = i
                # Find the end of this block (before force_detect or frames_sampled)
                for j in range(i + 1, i + 50):
                    if 'frames_sampled += 1' in lines[j]:
                        scene_cut_end = j - 1
                        break

                if scene_cut_end:
                    print(f"✓ Found scene cut block at lines {scene_cut_start + 1} to {scene_cut_end + 1}")
                    break

    if scene_cut_start and scene_cut_end:
        # Replace the scene cut logic
        indent = "                "  # 16 spaces
        new_scene_logic = [
            f"{indent}if next_cut is not None and frame_idx >= next_cut:",
            f"{indent}    # F1: Only reset if we're past cooldown period",
            f"{indent}    if frame_idx - last_cut_reset >= scene_cut_cooldown:",
            f"{indent}        reset_tracker = getattr(tracker_adapter, \"reset\", None)",
            f"{indent}        if callable(reset_tracker):",
            f"{indent}            reset_tracker()",
            f"{indent}        if appearance_gate:",
            f"{indent}            appearance_gate.reset_all()",
            f"{indent}        recorder.on_cut(frame_idx)",
            f"{indent}        frames_since_cut = 0",
            f"{indent}        last_cut_reset = frame_idx  # F1: Record reset time",
            f"{indent}        if progress:",
            f"{indent}            emit_frames, video_meta = _progress_value(frame_idx, include_current=False)",
            f"{indent}            progress.emit(",
            f"{indent}                emit_frames,",
            f"{indent}                phase=\"track\",",
            f"{indent}                device=device,",
            f"{indent}                detector=detector_choice,",
            f"{indent}                tracker=tracker_label,",
            f"{indent}                resolved_device=detector_device,",
            f"{indent}                summary={{\"event\": \"reset_on_cut\", \"frame\": frame_idx}},",
            f"{indent}                force=True,",
            f"{indent}                extra=video_meta,",
            f"{indent}            )",
            f"{indent}    else:",
            f"{indent}        # F1: Cut detected but within cooldown - skip reset",
            f"{indent}        LOGGER.debug(",
            f"{indent}            \"Skipping scene cut reset at frame %d (last reset at %d, cooldown=%d)\",",
            f"{indent}            frame_idx,",
            f"{indent}            last_cut_reset,",
            f"{indent}            scene_cut_cooldown,",
            f"{indent}        )",
            f"{indent}",
            f"{indent}    # Always advance to next cut (even if we skipped reset)",
            f"{indent}    cut_ix += 1",
            f"{indent}    next_cut = scene_cuts[cut_ix] if cut_ix < len(scene_cuts) else None",
        ]

        # Find how many lines the old scene cut logic takes
        old_lines_count = 0
        for j in range(scene_cut_start, scene_cut_end + 1):
            if 'next_cut = scene_cuts[cut_ix] if cut_ix < len(scene_cuts) else None' in lines[j]:
                old_lines_count = j - scene_cut_start + 1
                break

        if old_lines_count > 0:
            lines[scene_cut_start:scene_cut_start + old_lines_count] = new_scene_logic
            print(f"✓ Updated scene cut logic ({old_lines_count} lines replaced with {len(new_scene_logic)} lines)")

    # Write back
    episode_run.write_text('\n'.join(lines))
    print("\n✅ F1 optimization applied successfully!")


if __name__ == "__main__":
    main()
