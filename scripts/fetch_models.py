#!/usr/bin/env python3
"""
D8: Model fetcher with manifest generation and checksum validation.

Downloads InsightFace models and generates a manifest file with checksums
for integrity verification.
"""

import hashlib
import json
import os
from pathlib import Path

from insightface.app import FaceAnalysis


def compute_file_checksum(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(65536)  # 64KB chunks
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def generate_manifest(models_dir: Path, output_path: Path) -> dict:
    """
    Generate manifest with checksums for all model files.

    Args:
        models_dir: Directory containing model files
        output_path: Path to write manifest JSON

    Returns:
        Dictionary of manifest data
    """
    manifest = {
        "version": "1.0",
        "pack": "buffalo_l",
        "models": {}
    }

    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return manifest

    # Find all .onnx and .npy files
    for model_file in models_dir.rglob("*.onnx"):
        rel_path = model_file.relative_to(models_dir)
        checksum = compute_file_checksum(model_file)
        manifest["models"][str(rel_path)] = {
            "checksum": checksum,
            "size": model_file.stat().st_size,
            "algorithm": "sha256"
        }
        print(f"  {rel_path}: {checksum[:16]}...")

    for model_file in models_dir.rglob("*.npy"):
        rel_path = model_file.relative_to(models_dir)
        checksum = compute_file_checksum(model_file)
        manifest["models"][str(rel_path)] = {
            "checksum": checksum,
            "size": model_file.stat().st_size,
            "algorithm": "sha256"
        }
        print(f"  {rel_path}: {checksum[:16]}...")

    # Write manifest
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to: {output_path}")
    return manifest


def main():
    root = os.path.expanduser(os.getenv("INSIGHTFACE_HOME", "~/.insightface"))
    pack = os.getenv("INSIGHTFACE_PACK", "buffalo_l")

    print(f"Downloading InsightFace pack: {pack}")
    print(f"Root directory: {root}")

    app = FaceAnalysis(name=pack, root=root, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    models_dir = Path(root) / "models" / pack
    print(f"\nModels ready at: {models_dir}")

    # Generate manifest
    print("\nGenerating model manifest with checksums...")
    manifest_path = Path(root) / "models" / f"{pack}_manifest.json"
    generate_manifest(models_dir, manifest_path)

    print("\n✓ Model fetch complete!")
    print(f"  Models: {models_dir}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
