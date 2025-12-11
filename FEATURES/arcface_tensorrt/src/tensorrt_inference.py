"""
TensorRT Inference for ArcFace Embeddings.

Runs face embedding inference using TensorRT-accelerated ArcFace model.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


logger = logging.getLogger(__name__)


# ArcFace preprocessing constants (same as PyTorch reference)
ARCFACE_MEAN = np.array([127.5, 127.5, 127.5], dtype=np.float32)
ARCFACE_STD = np.array([127.5, 127.5, 127.5], dtype=np.float32)


class TensorRTArcFace:
    """TensorRT-accelerated ArcFace embedder."""

    def __init__(
        self,
        engine_path: Optional[Path] = None,
        config: Optional["TensorRTConfig"] = None,
    ):
        """
        Initialize TensorRT ArcFace embedder.

        Args:
            engine_path: Path to TensorRT engine file
            config: TensorRT configuration (used if engine_path not provided)
        """
        self.engine_path = engine_path
        self.config = config

        self._engine = None
        self._context = None
        self._stream = None

        # Bindings
        self._input_binding_idx = None
        self._output_binding_idx = None
        self._input_shape = None
        self._output_shape = None

        # Device memory
        self._d_input = None
        self._d_output = None
        self._h_output = None

    def _load_engine(self):
        """Load TensorRT engine."""
        if self._engine is not None:
            return

        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401

        except ImportError as e:
            raise ImportError(
                f"TensorRT or PyCUDA not available: {e}. "
                "Install with: pip install tensorrt pycuda"
            )

        engine_path = self.engine_path
        if engine_path is None:
            from .tensorrt_builder import build_or_load_engine

            engine_path, _ = build_or_load_engine(self.config)
            if engine_path is None:
                raise RuntimeError("Failed to build or load TensorRT engine")

        logger.info(f"Loading TensorRT engine: {engine_path}")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError(f"Failed to load engine from {engine_path}")

        # Create execution context
        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()

        # Get binding information
        for i in range(self._engine.num_bindings):
            if self._engine.binding_is_input(i):
                self._input_binding_idx = i
                self._input_shape = self._engine.get_binding_shape(i)
            else:
                self._output_binding_idx = i
                self._output_shape = self._engine.get_binding_shape(i)

        logger.info(f"  Input shape: {self._input_shape}")
        logger.info(f"  Output shape: {self._output_shape}")

    def _allocate_buffers(self, batch_size: int):
        """Allocate device buffers for inference."""
        import pycuda.driver as cuda

        # Calculate sizes
        input_size = batch_size * int(np.prod(self._input_shape[1:]))
        output_size = batch_size * int(np.prod(self._output_shape[1:]))

        # Allocate device memory
        self._d_input = cuda.mem_alloc(input_size * np.float32().nbytes)
        self._d_output = cuda.mem_alloc(output_size * np.float32().nbytes)

        # Allocate host output buffer
        self._h_output = np.empty(
            (batch_size, int(np.prod(self._output_shape[1:]))),
            dtype=np.float32,
        )

    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocess images for ArcFace.

        Args:
            images: BGR images of shape (N, H, W, 3) or (H, W, 3)
                   Expected size: 112x112

        Returns:
            Preprocessed tensor of shape (N, 3, 112, 112)
        """
        if images.ndim == 3:
            images = images[np.newaxis, ...]

        # Convert BGR to RGB
        images = images[..., ::-1].copy()

        # Normalize: (x - mean) / std
        images = images.astype(np.float32)
        images = (images - ARCFACE_MEAN) / ARCFACE_STD

        # Transpose to NCHW
        images = images.transpose(0, 3, 1, 2)

        return images.astype(np.float32)

    def embed(self, images: np.ndarray) -> np.ndarray:
        """
        Compute embeddings for face images.

        Args:
            images: BGR images of shape (N, 112, 112, 3) or (112, 112, 3)

        Returns:
            L2-normalized embeddings of shape (N, 512)
        """
        import pycuda.driver as cuda

        self._load_engine()

        # Preprocess
        input_tensor = self.preprocess(images)
        batch_size = input_tensor.shape[0]

        # Set dynamic batch size
        self._context.set_binding_shape(
            self._input_binding_idx,
            (batch_size,) + tuple(self._input_shape[1:]),
        )

        # Allocate buffers if needed
        if self._d_input is None:
            self._allocate_buffers(batch_size)

        # Copy input to device
        cuda.memcpy_htod_async(
            self._d_input,
            input_tensor.ravel(),
            self._stream,
        )

        # Run inference
        self._context.execute_async_v2(
            bindings=[int(self._d_input), int(self._d_output)],
            stream_handle=self._stream.handle,
        )

        # Copy output to host
        output_size = batch_size * int(np.prod(self._output_shape[1:]))
        h_output = np.empty(output_size, dtype=np.float32)

        cuda.memcpy_dtoh_async(h_output, self._d_output, self._stream)
        self._stream.synchronize()

        # Reshape and normalize
        embedding_dim = int(np.prod(self._output_shape[1:]))
        embeddings = h_output.reshape(batch_size, embedding_dim)

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        return embeddings

    def __del__(self):
        """Clean up resources."""
        # PyCUDA handles cleanup automatically via context
        pass


def run_tensorrt_embeddings(
    engine: Union[TensorRTArcFace, Path],
    images: Union[np.ndarray, List[np.ndarray]],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run TensorRT embeddings on a batch of images.

    Args:
        engine: TensorRTArcFace instance or path to engine file
        images: Face images (N, 112, 112, 3) BGR
        batch_size: Processing batch size

    Returns:
        Embeddings array (N, 512)
    """
    if isinstance(engine, Path):
        engine = TensorRTArcFace(engine_path=engine)

    if isinstance(images, list):
        images = np.stack(images)

    n_images = len(images)
    all_embeddings = []

    for i in range(0, n_images, batch_size):
        batch = images[i : i + batch_size]
        embeddings = engine.embed(batch)
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


def get_pytorch_arcface_embeddings(images: np.ndarray) -> Optional[np.ndarray]:
    """
    Get embeddings using PyTorch ArcFace (reference implementation).

    Args:
        images: Face images (N, 112, 112, 3) BGR

    Returns:
        Embeddings array (N, 512) or None if not available
    """
    try:
        from insightface.app import FaceAnalysis

        # Initialize InsightFace
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))

        # Get recognition model directly
        rec_model = app.models.get("recognition")
        if rec_model is None:
            logger.warning("InsightFace recognition model not found")
            return None

        # Process images
        embeddings = []
        for img in images:
            # InsightFace expects RGB
            img_rgb = img[..., ::-1].copy()
            emb = rec_model.get(img_rgb, None)
            if emb is not None:
                embeddings.append(emb)
            else:
                # Fallback: use zeros
                embeddings.append(np.zeros(512, dtype=np.float32))

        return np.stack(embeddings)

    except ImportError:
        logger.warning("InsightFace not available for reference embeddings")
        return None
    except Exception as e:
        logger.warning(f"Failed to get PyTorch embeddings: {e}")
        return None


def warmup_engine(engine: TensorRTArcFace, n_warmup: int = 3) -> float:
    """
    Warmup TensorRT engine and measure latency.

    Args:
        engine: TensorRTArcFace instance
        n_warmup: Number of warmup iterations

    Returns:
        Mean inference time in milliseconds
    """
    import time

    # Create dummy input
    dummy_input = np.random.randint(0, 255, (1, 112, 112, 3), dtype=np.uint8)

    times = []
    for _ in range(n_warmup + 5):
        start = time.perf_counter()
        engine.embed(dummy_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    # Skip warmup iterations
    return float(np.mean(times[n_warmup:]))
