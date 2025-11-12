"""Detect and optionally drop facebank seeds whose stored crops are nearly uniform."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from apps.api.services.facebank import FacebankService


def _iter_cast_dirs(fs: FacebankService, show_id: str, cast_id: str | None) -> Iterable[Path]:
    show_dir = fs.facebank_dir / show_id
    if cast_id:
        yield show_dir / cast_id
        return
    if not show_dir.exists():
        return
    for path in sorted(show_dir.iterdir()):
        if path.is_dir():
            yield path


def _image_std(path: Path) -> float | None:
    try:
        arr = np.asarray(Image.open(path))
    except Exception:
        return None
    if arr.size == 0:
        return 0.0
    return float(arr.std())


def prune_facebank_seeds(
    show_id: str,
    *,
    cast_id: str | None = None,
    threshold: float = 5.0,
    delete: bool = False,
) -> dict[str, int]:
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
    fs = FacebankService(data_root)
    stats = {"inspected": 0, "flagged": 0, "removed": 0, "errors": 0}

    for cast_path in _iter_cast_dirs(fs, show_id, cast_id):
        cid = cast_path.name
        data = fs._load_facebank(show_id, cid)
        seeds = data.get("seeds", [])
        keep = []
        changed = False
        for seed in seeds:
            stats["inspected"] += 1
            image_uri = seed.get("image_uri")
            if not image_uri:
                stats["errors"] += 1
                continue
            path = Path(image_uri)
            if not path.is_absolute():
                path = (cast_path / Path(image_uri).name).resolve()
            std = _image_std(path)
            if std is None:
                stats["errors"] += 1
                keep.append(seed)
                continue
            if std < threshold:
                stats["flagged"] += 1
                if delete:
                    stats["removed"] += 1
                    changed = True
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    continue
            keep.append(seed)
        if delete and changed:
            data["seeds"] = keep
            fs._save_facebank(show_id, cid, data)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect or delete blank facebank crops.")
    parser.add_argument("show_id", help="Show identifier (e.g. RHOBH)")
    parser.add_argument("--cast-id", help="Limit to a single cast id")
    parser.add_argument("--threshold", type=float, default=5.0, help="Std-dev threshold for flagging (default: 5.0)")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Remove flagged seeds (otherwise only report counts)",
    )
    args = parser.parse_args(argv)
    stats = prune_facebank_seeds(
        args.show_id,
        cast_id=args.cast_id,
        threshold=max(float(args.threshold), 0.0),
        delete=args.delete,
    )
    print(
        "inspected={inspected} flagged={flagged} removed={removed} errors={errors}".format(**stats)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
