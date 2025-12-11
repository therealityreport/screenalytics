"""
TensorRT Engine Builder for ArcFace.

Builds and manages TensorRT engines from ONNX models.
Supports local caching and S3/MinIO storage.
"""

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class TensorRTConfig:
    """Configuration for TensorRT engine building."""

    # Model
    model_name: str = "arcface_r100"
    onnx_path: Optional[Path] = None

    # Engine storage
    engine_local_dir: Path = Path("data/engines")
    engine_s3_bucket: Optional[str] = None
    engine_s3_key_prefix: str = "engines/arcface"

    # Build settings
    precision: str = "fp16"  # fp32, fp16, int8
    max_batch_size: int = 32
    workspace_size_mb: int = 1024
    min_batch_size: int = 1
    opt_batch_size: int = 16

    # Input shape (ArcFace standard)
    input_height: int = 112
    input_width: int = 112
    input_channels: int = 3

    @classmethod
    def from_yaml(cls, path: Path) -> "TensorRTConfig":
        """Load config from YAML file."""
        import yaml

        if not path.exists():
            logger.warning(f"Config not found at {path}, using defaults")
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f)

        trt_config = data.get("tensorrt", {})

        return cls(
            model_name=trt_config.get("model_name", cls.model_name),
            onnx_path=Path(trt_config["onnx_path"]) if "onnx_path" in trt_config else None,
            engine_local_dir=Path(trt_config.get("engine_local_dir", cls.engine_local_dir)),
            engine_s3_bucket=trt_config.get("engine_s3_bucket"),
            engine_s3_key_prefix=trt_config.get("engine_s3_key_prefix", cls.engine_s3_key_prefix),
            precision=trt_config.get("precision", cls.precision),
            max_batch_size=trt_config.get("max_batch_size", cls.max_batch_size),
            workspace_size_mb=trt_config.get("workspace_size_mb", cls.workspace_size_mb),
        )


def get_sm_arch() -> str:
    """
    Get CUDA SM architecture for current GPU.

    Returns string like "sm75" or "sm86".
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"

        # Get compute capability
        major, minor = torch.cuda.get_device_capability(0)
        return f"sm{major}{minor}"
    except ImportError:
        logger.warning("PyTorch not available, cannot detect SM arch")
        return "unknown"


def get_engine_filename(config: TensorRTConfig, sm_arch: str) -> str:
    """Generate engine filename with model and architecture info."""
    return f"{config.model_name}-{config.precision}-{sm_arch}.plan"


def get_local_engine_path(config: TensorRTConfig) -> Path:
    """Get local path for TensorRT engine."""
    sm_arch = get_sm_arch()
    filename = get_engine_filename(config, sm_arch)
    return config.engine_local_dir / filename


def get_s3_engine_key(config: TensorRTConfig) -> Optional[str]:
    """Get S3 key for TensorRT engine."""
    if not config.engine_s3_bucket:
        return None

    sm_arch = get_sm_arch()
    filename = get_engine_filename(config, sm_arch)
    return f"{config.engine_s3_key_prefix}/{filename}"


def download_engine_from_s3(config: TensorRTConfig, local_path: Path) -> bool:
    """
    Download TensorRT engine from S3/MinIO.

    Returns True if download successful.
    """
    if not config.engine_s3_bucket:
        return False

    s3_key = get_s3_engine_key(config)
    if not s3_key:
        return False

    try:
        import boto3
        from botocore.exceptions import ClientError

        s3 = boto3.client("s3")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading engine from s3://{config.engine_s3_bucket}/{s3_key}")
        s3.download_file(config.engine_s3_bucket, s3_key, str(local_path))

        logger.info(f"Downloaded engine to {local_path}")
        return True

    except ImportError:
        logger.warning("boto3 not available for S3 download")
        return False
    except Exception as e:
        logger.warning(f"Failed to download engine from S3: {e}")
        return False


def upload_engine_to_s3(config: TensorRTConfig, local_path: Path) -> bool:
    """
    Upload TensorRT engine to S3/MinIO.

    Returns True if upload successful.
    """
    if not config.engine_s3_bucket:
        return False

    s3_key = get_s3_engine_key(config)
    if not s3_key:
        return False

    try:
        import boto3

        s3 = boto3.client("s3")

        logger.info(f"Uploading engine to s3://{config.engine_s3_bucket}/{s3_key}")
        s3.upload_file(str(local_path), config.engine_s3_bucket, s3_key)

        logger.info("Upload complete")
        return True

    except ImportError:
        logger.warning("boto3 not available for S3 upload")
        return False
    except Exception as e:
        logger.warning(f"Failed to upload engine to S3: {e}")
        return False


def find_onnx_model(config: TensorRTConfig) -> Optional[Path]:
    """Find ONNX model file."""
    if config.onnx_path and config.onnx_path.exists():
        return config.onnx_path

    # Search common locations
    search_paths = [
        Path(f"models/{config.model_name}.onnx"),
        Path(f"~/.insightface/models/{config.model_name}/{config.model_name}.onnx").expanduser(),
        Path(f"data/models/{config.model_name}.onnx"),
    ]

    for path in search_paths:
        if path.exists():
            logger.info(f"Found ONNX model at {path}")
            return path

    return None


def build_engine_from_onnx(
    config: TensorRTConfig,
    onnx_path: Path,
    engine_path: Path,
) -> bool:
    """
    Build TensorRT engine from ONNX model.

    Uses trtexec CLI tool for reliable conversion.
    """
    try:
        import tensorrt as trt

        logger.info(f"Building TensorRT engine from {onnx_path}")
        logger.info(f"  Precision: {config.precision}")
        logger.info(f"  Max batch size: {config.max_batch_size}")

        # Create builder
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)

        # Create network with explicit batch
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)

        # Parse ONNX
        parser = trt.OnnxParser(network, trt_logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(i)}")
                return False

        # Configure builder
        config_builder = builder.create_builder_config()
        config_builder.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            config.workspace_size_mb * 1024 * 1024,
        )

        # Set precision
        if config.precision == "fp16" and builder.platform_has_fast_fp16:
            config_builder.set_flag(trt.BuilderFlag.FP16)
            logger.info("  Using FP16 precision")
        elif config.precision == "int8" and builder.platform_has_fast_int8:
            config_builder.set_flag(trt.BuilderFlag.INT8)
            logger.info("  Using INT8 precision")
        else:
            logger.info("  Using FP32 precision")

        # Set optimization profile for dynamic batch
        profile = builder.create_optimization_profile()
        input_shape = (
            config.input_channels,
            config.input_height,
            config.input_width,
        )

        # Get input name
        input_layer = network.get_input(0)
        input_name = input_layer.name

        profile.set_shape(
            input_name,
            min=(config.min_batch_size,) + input_shape,
            opt=(config.opt_batch_size,) + input_shape,
            max=(config.max_batch_size,) + input_shape,
        )
        config_builder.add_optimization_profile(profile)

        # Build engine
        logger.info("Building engine (this may take several minutes)...")
        serialized_engine = builder.build_serialized_network(network, config_builder)

        if serialized_engine is None:
            logger.error("Failed to build TensorRT engine")
            return False

        # Save engine
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        logger.info(f"Engine saved to {engine_path}")
        logger.info(f"  Size: {engine_path.stat().st_size / 1024 / 1024:.1f} MB")

        return True

    except ImportError:
        logger.error("TensorRT not available. Install with: pip install tensorrt")
        return False
    except Exception as e:
        logger.exception(f"Failed to build TensorRT engine: {e}")
        return False


def build_or_load_engine(
    config: Optional[TensorRTConfig] = None,
    force_rebuild: bool = False,
) -> Tuple[Optional[Path], bool]:
    """
    Build or load TensorRT engine.

    Priority:
    1. Local cache
    2. S3/MinIO download
    3. Build from ONNX

    Args:
        config: TensorRT configuration
        force_rebuild: Force rebuild even if cached

    Returns:
        Tuple of (engine_path, was_built)
        engine_path is None if build failed
    """
    config = config or TensorRTConfig()
    engine_path = get_local_engine_path(config)

    # Check local cache
    if not force_rebuild and engine_path.exists():
        logger.info(f"Using cached engine: {engine_path}")
        return engine_path, False

    # Try S3 download
    if not force_rebuild and download_engine_from_s3(config, engine_path):
        return engine_path, False

    # Build from ONNX
    onnx_path = find_onnx_model(config)
    if not onnx_path:
        logger.error(
            f"ONNX model not found for {config.model_name}. "
            f"Please provide --onnx-path or place model in models/ directory."
        )
        return None, False

    success = build_engine_from_onnx(config, onnx_path, engine_path)

    if success:
        # Upload to S3 for future use
        upload_engine_to_s3(config, engine_path)
        return engine_path, True

    return None, False


def get_engine_info(engine_path: Path) -> dict:
    """Get information about a TensorRT engine."""
    try:
        import tensorrt as trt

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            return {"error": "Failed to load engine"}

        info = {
            "path": str(engine_path),
            "size_mb": engine_path.stat().st_size / 1024 / 1024,
            "num_bindings": engine.num_bindings,
            "bindings": [],
        }

        for i in range(engine.num_bindings):
            binding_info = {
                "name": engine.get_binding_name(i),
                "is_input": engine.binding_is_input(i),
                "shape": list(engine.get_binding_shape(i)),
                "dtype": str(engine.get_binding_dtype(i)),
            }
            info["bindings"].append(binding_info)

        return info

    except ImportError:
        return {"error": "TensorRT not available"}
    except Exception as e:
        return {"error": str(e)}
