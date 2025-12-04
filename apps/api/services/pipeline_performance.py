"""Performance optimizations for the pipeline: async decode, batch uploads, caching, parallelism.

This module provides:
- C26: Async frame decoding + detection pipeline
- C27: Batch S3 uploads
- C29: Cache embeddings between stages
- C30: Parallel track processing in faces embed
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar

import numpy as np

LOGGER = logging.getLogger(__name__)


# =============================================================================
# C26: Async Frame Decoding + Detection Pipeline
# =============================================================================


@dataclass
class DecodedFrame:
    """A decoded video frame with metadata."""

    frame_idx: int
    timestamp: float  # seconds
    image: np.ndarray  # BGR numpy array
    width: int
    height: int


class FrameDecodeProducer:
    """Producer that decodes frames in a background thread.

    This implements requirement C26: Synchronous frame decoding + detection.

    Uses a producer/consumer pattern where:
    - Producer thread decodes frames and puts them in a queue
    - Consumer (detection pipeline) takes frames from queue
    - Backpressure prevents memory explosion
    """

    def __init__(
        self,
        video_path: Path | str,
        frame_stride: int = 1,
        max_queue_size: int = 32,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> None:
        """Initialize frame decoder.

        Args:
            video_path: Path to video file
            frame_stride: Sample every N frames
            max_queue_size: Maximum frames to buffer (backpressure)
            start_frame: Starting frame index (for resume)
            end_frame: Ending frame index (None = to end)
        """
        self._video_path = Path(video_path)
        self._stride = max(1, frame_stride)
        self._max_queue_size = max_queue_size
        self._start_frame = start_frame
        self._end_frame = end_frame

        self._queue: queue.Queue[Optional[DecodedFrame]] = queue.Queue(maxsize=max_queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._error: Optional[Exception] = None

        # Video metadata
        self._fps: float = 0
        self._total_frames: int = 0
        self._width: int = 0
        self._height: int = 0

    def start(self) -> None:
        """Start the background decode thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._decode_loop,
            daemon=True,
            name="frame-decoder",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the decode thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def get_frame(self, timeout: float = 30.0) -> Optional[DecodedFrame]:
        """Get the next decoded frame.

        Returns None when no more frames are available.
        Raises exception if decode thread encountered an error.
        """
        if self._error:
            raise self._error

        try:
            frame = self._queue.get(timeout=timeout)
            if frame is None and self._error:
                raise self._error
            return frame
        except queue.Empty:
            if self._error:
                raise self._error
            raise TimeoutError("Frame decode timeout")

    def __iter__(self) -> Iterator[DecodedFrame]:
        """Iterate over decoded frames."""
        self.start()
        try:
            while True:
                frame = self.get_frame()
                if frame is None:
                    break
                yield frame
        finally:
            self.stop()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._width, self._height

    def _decode_loop(self) -> None:
        """Background thread that decodes frames."""
        try:
            import cv2
        except ImportError:
            self._error = ImportError("OpenCV (cv2) not installed")
            self._queue.put(None)
            return

        cap = None
        try:
            cap = cv2.VideoCapture(str(self._video_path))
            if not cap.isOpened():
                self._error = RuntimeError(f"Cannot open video: {self._video_path}")
                self._queue.put(None)
                return

            # Get video metadata
            self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Seek to start frame if needed
            if self._start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)

            frame_idx = self._start_frame
            end_frame = self._end_frame or self._total_frames

            while frame_idx < end_frame and not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                # Only emit frames at stride interval
                if (frame_idx - self._start_frame) % self._stride == 0:
                    timestamp = frame_idx / self._fps

                    decoded = DecodedFrame(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        image=frame,
                        width=self._width,
                        height=self._height,
                    )

                    # Put with timeout to handle backpressure
                    while not self._stop_event.is_set():
                        try:
                            self._queue.put(decoded, timeout=1.0)
                            break
                        except queue.Full:
                            continue

                frame_idx += 1

            # Signal end of stream
            self._queue.put(None)

        except Exception as exc:
            self._error = exc
            LOGGER.exception("[frame-decoder] Error in decode loop")
            self._queue.put(None)

        finally:
            if cap is not None:
                cap.release()


class DetectionConsumer:
    """Consumer that runs detection on frames from the decode queue.

    Part of the async decode/detect pipeline (C26).
    """

    def __init__(
        self,
        detector: Any,  # Detection model
        batch_size: int = 1,
        device: str = "cpu",
    ) -> None:
        self._detector = detector
        self._batch_size = batch_size
        self._device = device
        self._stats = {
            "frames_processed": 0,
            "faces_detected": 0,
            "processing_time": 0.0,
        }

    def process_frames(
        self,
        frame_producer: FrameDecodeProducer,
        on_detection: Callable[[int, List[Dict]], None],
    ) -> Dict[str, Any]:
        """Process frames from producer, calling callback for each detection.

        Args:
            frame_producer: Producer providing decoded frames
            on_detection: Callback(frame_idx, detections) for each frame

        Returns:
            Processing statistics
        """
        batch: List[DecodedFrame] = []
        start_time = time.time()

        for frame in frame_producer:
            batch.append(frame)

            if len(batch) >= self._batch_size:
                self._process_batch(batch, on_detection)
                batch.clear()

        # Process remaining frames
        if batch:
            self._process_batch(batch, on_detection)

        self._stats["processing_time"] = time.time() - start_time
        return self._stats

    def _process_batch(
        self,
        batch: List[DecodedFrame],
        on_detection: Callable[[int, List[Dict]], None],
    ) -> None:
        """Process a batch of frames."""
        for frame in batch:
            try:
                # Run detection
                # Note: Actual implementation depends on detector interface
                detections = []
                if hasattr(self._detector, "detect"):
                    detections = self._detector.detect(frame.image)
                elif hasattr(self._detector, "__call__"):
                    detections = self._detector(frame.image)

                self._stats["frames_processed"] += 1
                self._stats["faces_detected"] += len(detections)

                # Callback with results
                on_detection(frame.frame_idx, detections)

            except Exception as exc:
                LOGGER.warning(
                    "[detection-consumer] Error processing frame %d: %s",
                    frame.frame_idx,
                    exc,
                )


# =============================================================================
# C27: Batch S3 Uploads
# =============================================================================


@dataclass
class UploadTask:
    """A single upload task."""

    key: str
    data: bytes
    content_type: str = "application/octet-stream"
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class BatchUploadResult:
    """Result of a batch upload operation."""

    total: int
    succeeded: int
    failed: int
    bytes_uploaded: int
    duration_sec: float
    errors: List[Tuple[str, str]] = field(default_factory=list)  # (key, error)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "bytes_uploaded": self.bytes_uploaded,
            "duration_sec": round(self.duration_sec, 2),
            "throughput_mbps": round(
                (self.bytes_uploaded / (1024 * 1024)) / max(self.duration_sec, 0.001),
                2,
            ),
            "error_count": len(self.errors),
        }


class BatchS3Uploader:
    """Batch uploader for S3 with concurrent uploads.

    This implements requirement C27: No batch S3 uploads.

    Features:
    - Uploads multiple files concurrently
    - Configurable concurrency limit
    - Automatic retry for transient failures
    - Progress tracking
    """

    def __init__(
        self,
        bucket: str,
        max_concurrency: int = 8,
        max_retries: int = 3,
        s3_client: Any = None,
    ) -> None:
        """Initialize batch uploader.

        Args:
            bucket: S3 bucket name
            max_concurrency: Maximum concurrent uploads
            max_retries: Max retry attempts for failed uploads
            s3_client: Optional boto3 S3 client (created if not provided)
        """
        self._bucket = bucket
        self._max_concurrency = max_concurrency
        self._max_retries = max_retries
        self._client = s3_client

    def _get_client(self):
        """Get or create S3 client."""
        if self._client is None:
            try:
                import boto3

                self._client = boto3.client("s3")
            except ImportError:
                raise ImportError("boto3 not installed; cannot use S3 batch uploader")
        return self._client

    def upload_batch(
        self,
        tasks: List[UploadTask],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchUploadResult:
        """Upload a batch of files concurrently.

        Args:
            tasks: List of upload tasks
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            BatchUploadResult with statistics
        """
        if not tasks:
            return BatchUploadResult(
                total=0,
                succeeded=0,
                failed=0,
                bytes_uploaded=0,
                duration_sec=0,
            )

        client = self._get_client()
        start_time = time.time()

        succeeded = 0
        failed = 0
        bytes_uploaded = 0
        errors: List[Tuple[str, str]] = []
        completed = 0

        def upload_one(task: UploadTask) -> Tuple[bool, int, Optional[str]]:
            """Upload a single file with retries."""
            for attempt in range(self._max_retries):
                try:
                    client.put_object(
                        Bucket=self._bucket,
                        Key=task.key,
                        Body=task.data,
                        ContentType=task.content_type,
                        Metadata=task.metadata,
                    )
                    return True, len(task.data), None
                except Exception as exc:
                    if attempt == self._max_retries - 1:
                        return False, 0, str(exc)
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
            return False, 0, "Max retries exceeded"

        with ThreadPoolExecutor(max_workers=self._max_concurrency) as executor:
            futures = {executor.submit(upload_one, task): task for task in tasks}

            for future in as_completed(futures):
                task = futures[future]
                try:
                    success, size, error = future.result()
                    if success:
                        succeeded += 1
                        bytes_uploaded += size
                    else:
                        failed += 1
                        errors.append((task.key, error or "Unknown error"))
                except Exception as exc:
                    failed += 1
                    errors.append((task.key, str(exc)))

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(tasks))

        duration = time.time() - start_time

        LOGGER.info(
            "[batch-upload] Completed: %d/%d succeeded, %d bytes in %.2fs",
            succeeded,
            len(tasks),
            bytes_uploaded,
            duration,
        )

        return BatchUploadResult(
            total=len(tasks),
            succeeded=succeeded,
            failed=failed,
            bytes_uploaded=bytes_uploaded,
            duration_sec=duration,
            errors=errors,
        )

    def upload_directory(
        self,
        local_dir: Path,
        s3_prefix: str,
        pattern: str = "*",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchUploadResult:
        """Upload all files in a directory.

        Args:
            local_dir: Local directory path
            s3_prefix: S3 key prefix
            pattern: Glob pattern to match files
            progress_callback: Optional progress callback

        Returns:
            BatchUploadResult
        """
        tasks = []
        for path in local_dir.rglob(pattern):
            if not path.is_file():
                continue

            relative = path.relative_to(local_dir)
            key = f"{s3_prefix.rstrip('/')}/{relative.as_posix()}"

            # Determine content type
            content_type = "application/octet-stream"
            suffix = path.suffix.lower()
            if suffix in (".jpg", ".jpeg"):
                content_type = "image/jpeg"
            elif suffix == ".png":
                content_type = "image/png"
            elif suffix in (".json", ".jsonl"):
                content_type = "application/json"

            tasks.append(UploadTask(
                key=key,
                data=path.read_bytes(),
                content_type=content_type,
            ))

        return self.upload_batch(tasks, progress_callback)


# =============================================================================
# C29: Cache Embeddings Between Stages
# =============================================================================


@dataclass
class CachedEmbedding:
    """A cached embedding with metadata."""

    track_id: int
    frame_idx: int
    embedding: np.ndarray
    model_name: str
    computed_at: datetime
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
            "embedding": self.embedding.tolist(),
            "model_name": self.model_name,
            "computed_at": self.computed_at.isoformat(),
            "confidence": self.confidence,
        }


class EmbeddingCache:
    """Cache for face embeddings between pipeline stages.

    This implements requirement C29: Embeddings recomputed per stage.

    Features:
    - Persists embeddings to disk for reuse
    - Validates cache based on model and config
    - Memory-efficient loading
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        model_name: str = "arcface",
    ) -> None:
        if cache_dir is None:
            data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
            cache_dir = data_root / "embedding_cache"
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._model_name = model_name
        self._memory_cache: Dict[str, Dict[int, np.ndarray]] = {}  # ep_id -> {track_id -> embedding}
        self._lock = threading.Lock()

    def _cache_path(self, ep_id: str) -> Path:
        return self._cache_dir / ep_id / f"embeddings_{self._model_name}.npz"

    def _meta_path(self, ep_id: str) -> Path:
        return self._cache_dir / ep_id / f"embeddings_{self._model_name}.json"

    def _config_hash(self, config: Dict[str, Any]) -> str:
        """Hash config to detect changes that invalidate cache."""
        relevant = {
            k: v for k, v in config.items()
            if k in ("model_name", "embedding_size", "preprocess", "normalize")
        }
        return hashlib.md5(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]

    def has_cache(self, ep_id: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Check if valid cache exists for episode."""
        cache_path = self._cache_path(ep_id)
        meta_path = self._meta_path(ep_id)

        if not cache_path.exists() or not meta_path.exists():
            return False

        if config:
            try:
                meta = json.loads(meta_path.read_text())
                cached_hash = meta.get("config_hash", "")
                current_hash = self._config_hash(config)
                if cached_hash != current_hash:
                    LOGGER.debug(
                        "[embedding-cache] Cache invalid for %s: config changed",
                        ep_id,
                    )
                    return False
            except Exception:
                return False

        return True

    def save_embeddings(
        self,
        ep_id: str,
        embeddings: Dict[int, np.ndarray],  # track_id -> embedding
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save embeddings to cache.

        Args:
            ep_id: Episode ID
            embeddings: Dict mapping track_id to embedding array
            config: Optional config for cache validation

        Returns:
            True if saved successfully
        """
        cache_path = self._cache_path(ep_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Save embeddings as compressed numpy archive
            np.savez_compressed(
                cache_path,
                track_ids=np.array(list(embeddings.keys())),
                embeddings=np.stack(list(embeddings.values())),
            )

            # Save metadata
            meta = {
                "model_name": self._model_name,
                "track_count": len(embeddings),
                "embedding_dim": list(embeddings.values())[0].shape[0] if embeddings else 0,
                "created_at": datetime.utcnow().isoformat(),
                "config_hash": self._config_hash(config or {}),
            }
            self._meta_path(ep_id).write_text(json.dumps(meta, indent=2))

            # Update memory cache
            with self._lock:
                self._memory_cache[ep_id] = embeddings.copy()

            LOGGER.info(
                "[embedding-cache] Saved %d embeddings for %s",
                len(embeddings),
                ep_id,
            )
            return True

        except Exception as exc:
            LOGGER.exception("[embedding-cache] Failed to save: %s", exc)
            return False

    def load_embeddings(
        self,
        ep_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[int, np.ndarray]]:
        """Load embeddings from cache.

        Args:
            ep_id: Episode ID
            config: Optional config for validation

        Returns:
            Dict mapping track_id to embedding, or None if cache invalid/missing
        """
        # Check memory cache first
        with self._lock:
            if ep_id in self._memory_cache:
                return self._memory_cache[ep_id].copy()

        # Check disk cache
        if not self.has_cache(ep_id, config):
            return None

        try:
            cache_path = self._cache_path(ep_id)
            data = np.load(cache_path)
            track_ids = data["track_ids"]
            embeddings_arr = data["embeddings"]

            result = {
                int(tid): emb
                for tid, emb in zip(track_ids, embeddings_arr)
            }

            # Populate memory cache
            with self._lock:
                self._memory_cache[ep_id] = result.copy()

            LOGGER.info(
                "[embedding-cache] Loaded %d embeddings for %s",
                len(result),
                ep_id,
            )
            return result

        except Exception as exc:
            LOGGER.warning("[embedding-cache] Failed to load: %s", exc)
            return None

    def get_embedding(self, ep_id: str, track_id: int) -> Optional[np.ndarray]:
        """Get a single track's embedding."""
        # Try memory cache
        with self._lock:
            if ep_id in self._memory_cache:
                return self._memory_cache[ep_id].get(track_id)

        # Load from disk
        embeddings = self.load_embeddings(ep_id)
        if embeddings:
            return embeddings.get(track_id)
        return None

    def invalidate(self, ep_id: str) -> bool:
        """Invalidate cache for an episode."""
        with self._lock:
            self._memory_cache.pop(ep_id, None)

        cache_path = self._cache_path(ep_id)
        meta_path = self._meta_path(ep_id)

        try:
            if cache_path.exists():
                cache_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
            LOGGER.info("[embedding-cache] Invalidated cache for %s", ep_id)
            return True
        except Exception as exc:
            LOGGER.warning("[embedding-cache] Failed to invalidate: %s", exc)
            return False


# =============================================================================
# C30: Parallel Track Processing in Faces Embed
# =============================================================================


T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ParallelProcessingResult(Generic[T]):
    """Result of parallel processing."""

    results: List[T]
    total: int
    succeeded: int
    failed: int
    duration_sec: float
    errors: List[Tuple[int, str]] = field(default_factory=list)  # (track_id, error)


class ParallelTrackProcessor:
    """Process tracks in parallel for faces_embed stage.

    This implements requirement C30: No parallel track processing in faces embed.

    Features:
    - Processes multiple tracks concurrently
    - Bounded parallelism to avoid resource exhaustion
    - Progress reporting compatible with parallel workers
    - Graceful error handling per track
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Initialize processor.

        Args:
            max_workers: Maximum parallel workers (defaults to CPU count)
            progress_callback: Optional callback(completed, total) for progress
        """
        import multiprocessing
        self._max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self._progress_callback = progress_callback

    def process_tracks(
        self,
        track_ids: List[int],
        process_fn: Callable[[int], R],
        batch_size: int = 1,
    ) -> ParallelProcessingResult[R]:
        """Process tracks in parallel.

        Args:
            track_ids: List of track IDs to process
            process_fn: Function to process a single track, returns result
            batch_size: Optional batch size for grouped processing

        Returns:
            ParallelProcessingResult with all results
        """
        start_time = time.time()
        results: List[R] = []
        errors: List[Tuple[int, str]] = []
        succeeded = 0
        completed = 0
        total = len(track_ids)

        lock = threading.Lock()

        def process_one(track_id: int) -> Tuple[int, Optional[R], Optional[str]]:
            """Process one track and return result."""
            try:
                result = process_fn(track_id)
                return track_id, result, None
            except Exception as exc:
                LOGGER.warning(
                    "[parallel-tracks] Error processing track %d: %s",
                    track_id,
                    exc,
                )
                return track_id, None, str(exc)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(process_one, tid): tid
                for tid in track_ids
            }

            for future in as_completed(futures):
                track_id = futures[future]
                try:
                    tid, result, error = future.result()
                    with lock:
                        completed += 1
                        if error:
                            errors.append((tid, error))
                        else:
                            results.append(result)
                            succeeded += 1

                    if self._progress_callback:
                        self._progress_callback(completed, total)

                except Exception as exc:
                    with lock:
                        completed += 1
                        errors.append((track_id, str(exc)))

        duration = time.time() - start_time

        LOGGER.info(
            "[parallel-tracks] Processed %d tracks: %d succeeded, %d failed (%.2fs)",
            total,
            succeeded,
            len(errors),
            duration,
        )

        return ParallelProcessingResult(
            results=results,
            total=total,
            succeeded=succeeded,
            failed=len(errors),
            duration_sec=duration,
            errors=errors,
        )


def process_tracks_parallel(
    track_ids: List[int],
    embed_fn: Callable[[int], np.ndarray],
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[int, np.ndarray]:
    """Convenience function to process tracks and get embeddings.

    Args:
        track_ids: Track IDs to embed
        embed_fn: Function to compute embedding for a track
        max_workers: Max parallel workers
        progress_callback: Optional progress callback

    Returns:
        Dict mapping track_id to embedding
    """
    processor = ParallelTrackProcessor(
        max_workers=max_workers,
        progress_callback=progress_callback,
    )

    def process_one(track_id: int) -> Tuple[int, np.ndarray]:
        embedding = embed_fn(track_id)
        return track_id, embedding

    result = processor.process_tracks(track_ids, process_one)

    return {tid: emb for tid, emb in result.results}


# =============================================================================
# Module exports
# =============================================================================

# Global embedding cache instance
_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache(model_name: str = "arcface") -> EmbeddingCache:
    """Get global embedding cache."""
    global _embedding_cache
    if _embedding_cache is None or _embedding_cache._model_name != model_name:
        _embedding_cache = EmbeddingCache(model_name=model_name)
    return _embedding_cache


__all__ = [
    # Async decode/detect
    "DecodedFrame",
    "FrameDecodeProducer",
    "DetectionConsumer",
    # Batch uploads
    "UploadTask",
    "BatchUploadResult",
    "BatchS3Uploader",
    # Embedding cache
    "CachedEmbedding",
    "EmbeddingCache",
    "get_embedding_cache",
    # Parallel processing
    "ParallelProcessingResult",
    "ParallelTrackProcessor",
    "process_tracks_parallel",
]
