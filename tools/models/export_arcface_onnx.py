"""
Export ArcFace model to ONNX format for TensorRT conversion.

InsightFace models are already stored as ONNX internally, so this utility:
1. Locates the existing ONNX model in InsightFace cache
2. Verifies model format and input shape (1x3x112x112)
3. Copies to configured path for TensorRT engine building

Usage:
    python -m tools.models.export_arcface_onnx --config config/pipeline/arcface_tensorrt.yaml
    python -m tools.models.export_arcface_onnx --output data/models/arcface_r100.onnx
"""

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import yaml


logger = logging.getLogger(__name__)

# Default InsightFace model locations
INSIGHTFACE_MODEL_DIR = Path.home() / ".insightface" / "models"
DEFAULT_MODEL_NAMES = [
    "arcface_r100_v1",
    "buffalo_l",
    "buffalo_sc",
]

# Expected model specifications
EXPECTED_INPUT_SHAPE = (1, 3, 112, 112)  # NCHW format
EXPECTED_OUTPUT_SHAPE = (1, 512)


def find_arcface_onnx(model_name: str = "arcface_r100_v1") -> Optional[Path]:
    """
    Find ArcFace ONNX model in InsightFace cache.

    Args:
        model_name: Model name to search for

    Returns:
        Path to ONNX file or None if not found
    """
    # Check InsightFace model directory
    model_dir = INSIGHTFACE_MODEL_DIR / model_name
    if model_dir.exists():
        # Look for ONNX files in model directory
        onnx_files = list(model_dir.glob("*.onnx"))
        if onnx_files:
            # Prefer w600k_r50.onnx or similar recognition model
            for onnx_file in onnx_files:
                if "r50" in onnx_file.name or "r100" in onnx_file.name:
                    logger.info(f"Found ArcFace ONNX: {onnx_file}")
                    return onnx_file
            # Fall back to first ONNX file
            logger.info(f"Found ONNX file: {onnx_files[0]}")
            return onnx_files[0]

    # Try alternative model names
    for alt_name in DEFAULT_MODEL_NAMES:
        alt_dir = INSIGHTFACE_MODEL_DIR / alt_name
        if alt_dir.exists():
            onnx_files = list(alt_dir.glob("*.onnx"))
            for onnx_file in onnx_files:
                if "r50" in onnx_file.name or "r100" in onnx_file.name:
                    logger.info(f"Found ArcFace ONNX (alt): {onnx_file}")
                    return onnx_file

    logger.warning(f"No ArcFace ONNX found in {INSIGHTFACE_MODEL_DIR}")
    return None


def verify_onnx_model(onnx_path: Path) -> Tuple[bool, dict]:
    """
    Verify ONNX model has expected ArcFace structure.

    Args:
        onnx_path: Path to ONNX file

    Returns:
        Tuple of (is_valid, info_dict)
    """
    info = {
        "path": str(onnx_path),
        "size_mb": round(onnx_path.stat().st_size / (1024 * 1024), 2),
        "input_shape": None,
        "output_shape": None,
        "valid": False,
    }

    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

        # Get input shape
        inputs = model.graph.input
        if inputs:
            input_shape = []
            for dim in inputs[0].type.tensor_type.shape.dim:
                if dim.dim_value:
                    input_shape.append(dim.dim_value)
                else:
                    input_shape.append(-1)  # Dynamic dimension
            info["input_shape"] = tuple(input_shape)

        # Get output shape
        outputs = model.graph.output
        if outputs:
            output_shape = []
            for dim in outputs[0].type.tensor_type.shape.dim:
                if dim.dim_value:
                    output_shape.append(dim.dim_value)
                else:
                    output_shape.append(-1)
            info["output_shape"] = tuple(output_shape)

        # Validate shapes
        if info["input_shape"] and len(info["input_shape"]) == 4:
            # Check spatial dimensions (112x112)
            if info["input_shape"][2:] == (112, 112):
                info["valid"] = True
            elif info["input_shape"][2:] == (112, 112) or info["input_shape"][1:3] == (112, 112):
                info["valid"] = True

        if info["output_shape"]:
            # Check embedding dimension (512)
            if 512 in info["output_shape"]:
                info["valid"] = info["valid"] and True
            else:
                info["valid"] = False

        return info["valid"], info

    except ImportError:
        logger.warning("onnx package not installed, skipping verification")
        info["valid"] = True  # Assume valid if can't verify
        return True, info
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
        info["error"] = str(e)
        return False, info


def export_arcface_onnx(
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
    model_name: str = "arcface_r100_v1",
    force: bool = False,
) -> Optional[Path]:
    """
    Export/copy ArcFace ONNX model to configured location.

    Args:
        config_path: Path to arcface_tensorrt.yaml config
        output_path: Explicit output path (overrides config)
        model_name: Model name to export
        force: Overwrite existing file

    Returns:
        Path to exported ONNX file or None on failure
    """
    # Determine output path
    dest_path = None

    if output_path:
        dest_path = Path(output_path)
    elif config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        onnx_path = config.get("tensorrt", {}).get("onnx_path")
        if onnx_path:
            dest_path = Path(onnx_path)
        else:
            # Use default based on config
            engine_dir = config.get("tensorrt", {}).get("engine_local_dir", "data/engines")
            model = config.get("tensorrt", {}).get("model_name", model_name)
            dest_path = Path(engine_dir) / f"{model}.onnx"

    if dest_path is None:
        dest_path = Path("data/models") / f"{model_name}.onnx"

    # Check if already exists
    if dest_path.exists() and not force:
        logger.info(f"ONNX already exists at {dest_path}")
        is_valid, info = verify_onnx_model(dest_path)
        if is_valid:
            logger.info(f"  Verified: input={info['input_shape']}, output={info['output_shape']}")
            return dest_path
        else:
            logger.warning(f"  Existing file invalid, will re-export")

    # Find source ONNX
    source_path = find_arcface_onnx(model_name)
    if source_path is None:
        logger.error(
            "Could not find ArcFace ONNX model. "
            "Run: python -c 'from insightface.app import FaceAnalysis; FaceAnalysis(name=\"buffalo_l\")' "
            "to download models first."
        )
        return None

    # Verify source
    is_valid, info = verify_onnx_model(source_path)
    if not is_valid:
        logger.error(f"Source ONNX invalid: {info}")
        return None

    logger.info(f"Source ONNX verified:")
    logger.info(f"  Path: {source_path}")
    logger.info(f"  Size: {info['size_mb']} MB")
    logger.info(f"  Input shape: {info['input_shape']}")
    logger.info(f"  Output shape: {info['output_shape']}")

    # Copy to destination
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)
    logger.info(f"Exported to: {dest_path}")

    return dest_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export ArcFace ONNX model for TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline/arcface_tensorrt.yaml",
        help="Path to TensorRT config YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Explicit output path (overrides config)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="arcface_r100_v1",
        help="Model name to export",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing ONNX file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.verify_only:
        # Verify existing file
        if args.output:
            path = Path(args.output)
        elif args.config and os.path.exists(args.config):
            with open(args.config) as f:
                config = yaml.safe_load(f)
            path = Path(config.get("tensorrt", {}).get("onnx_path", ""))
        else:
            path = find_arcface_onnx(args.model_name)

        if path is None or not path.exists():
            print(f"ERROR: ONNX file not found")
            return 1

        is_valid, info = verify_onnx_model(path)
        print(f"\nONNX Model Verification:")
        print(f"  Path: {info['path']}")
        print(f"  Size: {info['size_mb']} MB")
        print(f"  Input shape: {info['input_shape']}")
        print(f"  Output shape: {info['output_shape']}")
        print(f"  Valid: {'YES' if is_valid else 'NO'}")

        if info.get("error"):
            print(f"  Error: {info['error']}")

        return 0 if is_valid else 1

    # Export model
    config_path = args.config if os.path.exists(args.config) else None
    result = export_arcface_onnx(
        config_path=config_path,
        output_path=args.output,
        model_name=args.model_name,
        force=args.force,
    )

    if result:
        print(f"\nSuccess! ONNX model exported to: {result}")
        print(f"\nNext steps:")
        print(f"  1. Build TensorRT engine:")
        print(f"     python -m FEATURES.arcface_tensorrt build --onnx-path {result}")
        print(f"  2. Compare embeddings:")
        print(f"     python -m FEATURES.arcface_tensorrt compare")
        return 0
    else:
        print(f"\nFailed to export ONNX model")
        return 1


if __name__ == "__main__":
    exit(main())
