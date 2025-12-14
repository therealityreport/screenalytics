"""
Central CPU thread limit configuration for SCREENALYTICS.

This module provides a consistent way to cap CPU usage across all ML workloads
by limiting thread counts in BLAS libraries, ONNX Runtime, PyTorch, and thread pools.

Default behavior targets ~300% CPU usage (roughly 3 logical cores) to prevent
overheating on laptops while maintaining good performance.

Environment Variables:
    SCREENALYTICS_MAX_CPU_THREADS: Integer core/thread cap (default: 3)
    SCREENALYTICS_MAX_CPU_PERCENT: Alternative percentage-based limit (e.g., 300 for 3 cores)

    Deprecated aliases (backward compatible for now):
        SCREANALYTICS_MAX_CPU_THREADS
        SCREANALYTICS_MAX_CPU_PERCENT

Usage:
    Call apply_global_cpu_limits() once at the very top of each process entrypoint,
    before importing or initializing any ML libraries.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        import onnxruntime
    except ImportError:
        pass

LOGGER = logging.getLogger(__name__)

# Conservative default: 3 logical cores = ~300% CPU usage on Activity Monitor
DEFAULT_MAX_THREADS = 3

_limits_applied = False

_warned_deprecated_env_vars: set[str] = set()


def _warn_deprecated_env_var(deprecated: str, canonical: str) -> None:
    if deprecated in _warned_deprecated_env_vars:
        return
    _warned_deprecated_env_vars.add(deprecated)
    LOGGER.warning("Deprecated env var %s is set; use %s instead", deprecated, canonical)


def get_max_threads_from_env(default_cores: int = DEFAULT_MAX_THREADS) -> int:
    """
    Determine the maximum number of threads to use based on environment configuration.

    Checks SCREENALYTICS_MAX_CPU_THREADS first, then SCREENALYTICS_MAX_CPU_PERCENT.
    Falls back to min(default_cores, actual_cpu_count) to prevent over-subscribing
    on low-core machines.

    Args:
        default_cores: Default core limit if no environment variable is set

    Returns:
        Maximum number of threads (integer >= 1)
    """
    threads_key = "SCREENALYTICS_MAX_CPU_THREADS"
    percent_key = "SCREENALYTICS_MAX_CPU_PERCENT"
    deprecated_threads_key = "SCREANALYTICS_MAX_CPU_THREADS"
    deprecated_percent_key = "SCREANALYTICS_MAX_CPU_PERCENT"

    # Direct thread count override
    if threads_key in os.environ:
        try:
            max_threads = int(os.environ[threads_key])
            if max_threads >= 1:
                return max_threads
            LOGGER.warning("%s=%d is invalid; using default", threads_key, max_threads)
        except (ValueError, TypeError):
            LOGGER.warning("Invalid %s value; using default", threads_key)

    if threads_key not in os.environ and deprecated_threads_key in os.environ:
        _warn_deprecated_env_var(deprecated_threads_key, threads_key)
        try:
            max_threads = int(os.environ[deprecated_threads_key])
            if max_threads >= 1:
                return max_threads
            LOGGER.warning("%s=%d is invalid; using default", deprecated_threads_key, max_threads)
        except (ValueError, TypeError):
            LOGGER.warning("Invalid %s value; using default", deprecated_threads_key)

    # Percentage-based limit (e.g., 300 = 3 cores at 100% each)
    if percent_key in os.environ:
        try:
            percent = int(os.environ[percent_key])
            max_threads = max(1, percent // 100)
            LOGGER.info("Computed max_threads=%d from %s=%d", max_threads, percent_key, percent)
            return max_threads
        except (ValueError, TypeError):
            LOGGER.warning("Invalid %s value; using default", percent_key)

    if percent_key not in os.environ and deprecated_percent_key in os.environ:
        _warn_deprecated_env_var(deprecated_percent_key, percent_key)
        try:
            percent = int(os.environ[deprecated_percent_key])
            max_threads = max(1, percent // 100)
            LOGGER.info("Computed max_threads=%d from %s=%d", max_threads, deprecated_percent_key, percent)
            return max_threads
        except (ValueError, TypeError):
            LOGGER.warning("Invalid %s value; using default", deprecated_percent_key)

    # Fallback: don't exceed physical core count
    try:
        cpu_count = os.cpu_count() or default_cores
        return min(default_cores, cpu_count)
    except Exception:
        return default_cores


def apply_global_cpu_limits(max_threads: int | None = None) -> None:
    """
    Apply CPU thread limits globally across all libraries and frameworks.

    This function MUST be called once at process startup, before any ML libraries
    are imported or initialized. It sets environment variables that control BLAS
    thread pools and attempts to configure PyTorch if available.

    Args:
        max_threads: Override for maximum threads (uses env config if None)

    Note:
        This is a soft limit intended to keep CPU usage under ~300% on Activity Monitor.
        Actual CPU usage may spike briefly during I/O or startup but should stabilize.
    """
    global _limits_applied

    if _limits_applied:
        LOGGER.debug("CPU limits already applied; skipping duplicate call")
        return

    if max_threads is None:
        max_threads = get_max_threads_from_env()

    LOGGER.info("Applying global CPU limits: max_threads=%d", max_threads)

    # Set BLAS library thread limits (must be done before libraries load)
    # Use 1 thread for most libraries to prevent thread explosions
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # OpenCV thread control
    os.environ.setdefault("OPENCV_NUM_THREADS", "1")

    # ONNX Runtime thread control (allow 2 intra-op threads for better CoreML/CPU performance)
    os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", "2")
    os.environ.setdefault("ORT_INTER_OP_NUM_THREADS", "1")

    # Try to configure PyTorch if it's already imported
    try:
        import torch

        torch.set_num_threads(max_threads)
        # Use half the threads for inter-op parallelism (conservative)
        torch.set_num_interop_threads(max(1, max_threads // 2))
        LOGGER.info("Configured PyTorch: num_threads=%d, num_interop_threads=%d", max_threads, max(1, max_threads // 2))
    except ImportError:
        LOGGER.debug("PyTorch not installed; skipping torch thread configuration")
    except Exception as exc:
        LOGGER.warning("Failed to configure PyTorch threads: %s", exc)

    _limits_applied = True
    LOGGER.info("✅ Global CPU limits applied successfully (target: ~%d00%% CPU usage)", max_threads)


def configure_onnx_session_options(session_options: onnxruntime.SessionOptions, max_threads: int | None = None) -> None:
    """
    Configure ONNX Runtime SessionOptions to respect CPU thread limits.

    Apply this to ALL ONNX sessions (CPU, CoreML, GPU) to ensure any CPU fallback
    paths don't spawn unbounded threads.

    Args:
        session_options: ONNX SessionOptions object to configure
        max_threads: Override for maximum threads (uses env config if None)

    Example:
        >>> import onnxruntime
        >>> from apps.common.cpu_limits import configure_onnx_session_options
        >>> so = onnxruntime.SessionOptions()
        >>> configure_onnx_session_options(so)
        >>> sess = onnxruntime.InferenceSession("model.onnx", sess_options=so)
    """
    if max_threads is None:
        max_threads = get_max_threads_from_env()

    # Intra-op: threads used within a single operator (e.g., matrix multiplication)
    session_options.intra_op_num_threads = max_threads

    # Inter-op: threads used to run operators in parallel (keep conservative)
    session_options.inter_op_num_threads = 1

    # Disable spinning (reduces CPU usage when waiting)
    session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    LOGGER.debug("Configured ONNX SessionOptions: intra_op=%d, inter_op=1", max_threads)


def get_pool_max_workers(io_bound: bool = False) -> int:
    """
    Get the recommended max_workers for ThreadPoolExecutor/ProcessPoolExecutor.

    Args:
        io_bound: If True, allows slightly higher concurrency for I/O operations
                 (e.g., S3 uploads). If False, uses CPU limits directly.

    Returns:
        Recommended max_workers count
    """
    max_threads = get_max_threads_from_env()

    if io_bound:
        # For I/O-bound tasks (S3, network), allow 2x the CPU limit
        return min(max_threads * 2, 8)

    # For CPU-bound tasks, use the CPU limit directly
    return max_threads
