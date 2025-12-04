"""Enhanced manifest handling with buffering, versioning, validation, and checksums.

This module provides:
- C28: Buffered manifest writes
- D31: Checksum for uploaded artifacts
- D32: Manifest versioning
- D33: Atomic manifest writes (partial write corruption prevention)
- D34: Manifest structure validation
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Set, Type, TypeVar, Union

LOGGER = logging.getLogger(__name__)

# Configuration
MANIFEST_BUFFER_SIZE = 100  # Flush after this many entries
MANIFEST_BUFFER_TIMEOUT = 5.0  # Flush at least every N seconds
MANIFEST_VERSION_HISTORY = 5  # Keep this many historical versions


# =============================================================================
# Manifest Schema Definitions (D34)
# =============================================================================


class ManifestType(Enum):
    """Types of manifests with their expected schemas."""

    FACES = auto()
    TRACKS = auto()
    IDENTITIES = auto()
    DETECTIONS = auto()
    EMBEDDINGS = auto()
    PROGRESS = auto()
    AUDIO_SEGMENTS = auto()
    UNKNOWN = auto()


# Required fields for each manifest type
MANIFEST_SCHEMAS: Dict[ManifestType, Dict[str, type]] = {
    ManifestType.FACES: {
        "track_id": int,
        "frame_idx": int,
        # Optional but common: "confidence", "bbox", "embedding"
    },
    ManifestType.TRACKS: {
        "track_id": int,
        "frames": list,
        # Optional: "start_frame", "end_frame", "face_count"
    },
    ManifestType.IDENTITIES: {
        "identity_id": str,
        "track_ids": list,
    },
    ManifestType.DETECTIONS: {
        "frame_idx": int,
        "detections": list,
    },
    ManifestType.EMBEDDINGS: {
        "track_id": int,
        "embedding": list,
    },
    ManifestType.PROGRESS: {
        "phase": str,
        # Optional: "frames_done", "frames_total", "percent"
    },
    ManifestType.AUDIO_SEGMENTS: {
        "start": (int, float),
        "end": (int, float),
    },
}


def infer_manifest_type(filename: str) -> ManifestType:
    """Infer manifest type from filename."""
    name_lower = filename.lower()
    if "face" in name_lower:
        return ManifestType.FACES
    if "track" in name_lower:
        return ManifestType.TRACKS
    if "identit" in name_lower:
        return ManifestType.IDENTITIES
    if "detect" in name_lower:
        return ManifestType.DETECTIONS
    if "embed" in name_lower:
        return ManifestType.EMBEDDINGS
    if "progress" in name_lower:
        return ManifestType.PROGRESS
    if "audio" in name_lower or "segment" in name_lower:
        return ManifestType.AUDIO_SEGMENTS
    return ManifestType.UNKNOWN


@dataclass
class ValidationError:
    """Details about a manifest validation error."""

    line_number: Optional[int] = None
    field: Optional[str] = None
    message: str = ""
    raw_data: Optional[str] = None


@dataclass
class ManifestValidationResult:
    """Result of manifest validation."""

    is_valid: bool
    manifest_type: ManifestType
    total_entries: int = 0
    valid_entries: int = 0
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "manifest_type": self.manifest_type.name,
            "total_entries": self.total_entries,
            "valid_entries": self.valid_entries,
            "error_count": len(self.errors),
            "errors": [
                {
                    "line": e.line_number,
                    "field": e.field,
                    "message": e.message,
                }
                for e in self.errors[:10]  # Limit to first 10
            ],
            "warnings": self.warnings[:10],
        }


def validate_manifest_entry(
    entry: Dict[str, Any],
    manifest_type: ManifestType,
) -> List[ValidationError]:
    """Validate a single manifest entry against its schema."""
    errors = []
    schema = MANIFEST_SCHEMAS.get(manifest_type, {})

    for field_name, expected_type in schema.items():
        if field_name not in entry:
            errors.append(ValidationError(
                field=field_name,
                message=f"Missing required field: {field_name}",
            ))
            continue

        value = entry[field_name]

        # Handle tuple of types (multiple allowed types)
        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                errors.append(ValidationError(
                    field=field_name,
                    message=f"Field '{field_name}' has wrong type: expected {expected_type}, got {type(value).__name__}",
                ))
        else:
            if not isinstance(value, expected_type):
                errors.append(ValidationError(
                    field=field_name,
                    message=f"Field '{field_name}' has wrong type: expected {expected_type.__name__}, got {type(value).__name__}",
                ))

    return errors


def validate_manifest_file(
    path: Path,
    manifest_type: Optional[ManifestType] = None,
    sample_size: int = 100,
    strict: bool = False,
) -> ManifestValidationResult:
    """Validate a manifest file's structure.

    This implements requirement D34: No validation of manifest structure.

    Args:
        path: Path to manifest file
        manifest_type: Expected type (inferred from filename if not provided)
        sample_size: Number of entries to validate (0 for all)
        strict: If True, any error makes manifest invalid. If False, only fail on majority errors.

    Returns:
        ManifestValidationResult with details
    """
    if manifest_type is None:
        manifest_type = infer_manifest_type(path.name)

    result = ManifestValidationResult(
        is_valid=False,
        manifest_type=manifest_type,
    )

    if not path.exists():
        result.errors.append(ValidationError(message=f"File not found: {path}"))
        return result

    # Determine if JSONL or JSON
    is_jsonl = path.suffix.lower() in (".jsonl", ".ndjson")

    try:
        if is_jsonl:
            result = _validate_jsonl_file(path, manifest_type, sample_size, strict)
        else:
            result = _validate_json_file(path, manifest_type)
    except Exception as exc:
        result.errors.append(ValidationError(
            message=f"Error reading manifest: {exc}"
        ))

    return result


def _validate_jsonl_file(
    path: Path,
    manifest_type: ManifestType,
    sample_size: int,
    strict: bool,
) -> ManifestValidationResult:
    """Validate a JSONL manifest file."""
    result = ManifestValidationResult(
        is_valid=False,
        manifest_type=manifest_type,
    )

    with path.open("r", encoding="utf-8") as f:
        line_number = 0
        validated = 0

        for line in f:
            line_number += 1
            line = line.strip()

            if not line:
                continue

            result.total_entries += 1

            # Only validate sample
            if sample_size > 0 and validated >= sample_size:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                result.errors.append(ValidationError(
                    line_number=line_number,
                    message=f"Invalid JSON: {exc}",
                    raw_data=line[:100],
                ))
                continue

            if not isinstance(entry, dict):
                result.errors.append(ValidationError(
                    line_number=line_number,
                    message="Entry is not a JSON object",
                ))
                continue

            entry_errors = validate_manifest_entry(entry, manifest_type)
            if entry_errors:
                for err in entry_errors:
                    err.line_number = line_number
                    result.errors.append(err)
            else:
                result.valid_entries += 1

            validated += 1

    # Determine validity
    if result.total_entries == 0:
        result.warnings.append("Manifest is empty")
        result.is_valid = True
    elif strict:
        result.is_valid = len(result.errors) == 0
    else:
        # Non-strict: valid if majority of sampled entries are valid
        error_rate = len(result.errors) / max(validated, 1)
        result.is_valid = error_rate < 0.5

    return result


def _validate_json_file(
    path: Path,
    manifest_type: ManifestType,
) -> ManifestValidationResult:
    """Validate a JSON manifest file."""
    result = ManifestValidationResult(
        is_valid=False,
        manifest_type=manifest_type,
    )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result.errors.append(ValidationError(
            message=f"Invalid JSON file: {exc}"
        ))
        return result

    # Handle array of entries
    if isinstance(data, list):
        result.total_entries = len(data)
        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                result.errors.append(ValidationError(
                    line_number=idx,
                    message="Entry is not a JSON object",
                ))
                continue

            entry_errors = validate_manifest_entry(entry, manifest_type)
            if entry_errors:
                for err in entry_errors:
                    err.line_number = idx
                    result.errors.append(err)
            else:
                result.valid_entries += 1

    # Handle single object (like identities.json)
    elif isinstance(data, dict):
        result.total_entries = 1
        entry_errors = validate_manifest_entry(data, manifest_type)
        if entry_errors:
            result.errors.extend(entry_errors)
        else:
            result.valid_entries = 1

    result.is_valid = len(result.errors) == 0
    return result


# =============================================================================
# Checksum Support (D31)
# =============================================================================


def compute_checksum(data: bytes, algorithm: str = "sha256") -> str:
    """Compute checksum of data.

    Args:
        data: Bytes to checksum
        algorithm: Hash algorithm (md5, sha256, sha512)

    Returns:
        Hex-encoded checksum string
    """
    if algorithm == "md5":
        return hashlib.md5(data).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(data).hexdigest()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def compute_file_checksum(path: Path, algorithm: str = "sha256") -> str:
    """Compute checksum of a file."""
    if algorithm == "md5":
        hasher = hashlib.md5()
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "sha512":
        hasher = hashlib.sha512()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


@dataclass
class ArtifactChecksum:
    """Checksum information for an artifact."""

    path: str
    checksum: str
    algorithm: str
    size_bytes: int
    computed_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "checksum": self.checksum,
            "algorithm": self.algorithm,
            "size_bytes": self.size_bytes,
            "computed_at": self.computed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactChecksum":
        return cls(
            path=data["path"],
            checksum=data["checksum"],
            algorithm=data.get("algorithm", "sha256"),
            size_bytes=data.get("size_bytes", 0),
            computed_at=datetime.fromisoformat(data["computed_at"]),
        )


class ChecksumRegistry:
    """Registry of artifact checksums for verification.

    This implements requirement D31: No checksum for uploaded artifacts.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        if base_dir is None:
            data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
            base_dir = data_root / "checksums"
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _registry_path(self, ep_id: str) -> Path:
        return self._base_dir / f"{ep_id}_checksums.json"

    def record_checksum(
        self,
        ep_id: str,
        artifact_path: str,
        data: Optional[bytes] = None,
        algorithm: str = "sha256",
    ) -> ArtifactChecksum:
        """Record checksum for an artifact.

        Args:
            ep_id: Episode ID
            artifact_path: Relative path of artifact
            data: Optional bytes to checksum. If not provided, reads from artifact_path.
            algorithm: Hash algorithm to use

        Returns:
            ArtifactChecksum record
        """
        if data is not None:
            checksum = compute_checksum(data, algorithm)
            size = len(data)
        else:
            path = Path(artifact_path)
            checksum = compute_file_checksum(path, algorithm)
            size = path.stat().st_size

        record = ArtifactChecksum(
            path=artifact_path,
            checksum=checksum,
            algorithm=algorithm,
            size_bytes=size,
            computed_at=datetime.utcnow(),
        )

        # Save to registry
        with self._lock:
            registry_path = self._registry_path(ep_id)
            registry: Dict[str, Dict] = {}
            if registry_path.exists():
                try:
                    registry = json.loads(registry_path.read_text())
                except json.JSONDecodeError:
                    pass

            registry[artifact_path] = record.to_dict()
            registry_path.write_text(json.dumps(registry, indent=2))

        return record

    def verify_checksum(
        self,
        ep_id: str,
        artifact_path: str,
        data: Optional[bytes] = None,
    ) -> Optional[bool]:
        """Verify artifact checksum.

        Returns:
            True if matches, False if mismatch, None if no record exists
        """
        with self._lock:
            registry_path = self._registry_path(ep_id)
            if not registry_path.exists():
                return None

            try:
                registry = json.loads(registry_path.read_text())
            except json.JSONDecodeError:
                return None

            record_data = registry.get(artifact_path)
            if not record_data:
                return None

            record = ArtifactChecksum.from_dict(record_data)

        # Compute current checksum
        if data is not None:
            current_checksum = compute_checksum(data, record.algorithm)
        else:
            path = Path(artifact_path)
            if not path.exists():
                return False
            current_checksum = compute_file_checksum(path, record.algorithm)

        return current_checksum == record.checksum

    def get_checksums(self, ep_id: str) -> Dict[str, ArtifactChecksum]:
        """Get all checksums for an episode."""
        registry_path = self._registry_path(ep_id)
        if not registry_path.exists():
            return {}

        try:
            registry = json.loads(registry_path.read_text())
            return {
                path: ArtifactChecksum.from_dict(data)
                for path, data in registry.items()
            }
        except Exception:
            return {}


# =============================================================================
# Manifest Versioning (D32)
# =============================================================================


@dataclass
class ManifestVersion:
    """Information about a manifest version."""

    version: int
    path: str
    created_at: datetime
    size_bytes: int
    checksum: Optional[str] = None
    is_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "path": self.path,
            "created_at": self.created_at.isoformat(),
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "is_active": self.is_active,
        }


class ManifestVersionManager:
    """Manage versioned manifests with history.

    This implements requirement D32: No manifest versioning.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        max_versions: int = MANIFEST_VERSION_HISTORY,
    ) -> None:
        if base_dir is None:
            data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
            base_dir = data_root / "manifests"
        self._base_dir = base_dir
        self._max_versions = max_versions
        self._lock = threading.Lock()

    def _ep_manifest_dir(self, ep_id: str) -> Path:
        return self._base_dir / ep_id

    def _version_index_path(self, ep_id: str) -> Path:
        return self._ep_manifest_dir(ep_id) / ".manifest_versions.json"

    def _load_version_index(self, ep_id: str) -> Dict[str, List[Dict]]:
        """Load the version index for an episode."""
        index_path = self._version_index_path(ep_id)
        if not index_path.exists():
            return {}
        try:
            return json.loads(index_path.read_text())
        except json.JSONDecodeError:
            return {}

    def _save_version_index(self, ep_id: str, index: Dict[str, List[Dict]]) -> None:
        """Save the version index."""
        index_path = self._version_index_path(ep_id)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index, indent=2))

    def save_versioned_manifest(
        self,
        ep_id: str,
        manifest_name: str,
        content: Union[str, bytes],
        checksum: Optional[str] = None,
    ) -> ManifestVersion:
        """Save a new version of a manifest.

        Args:
            ep_id: Episode ID
            manifest_name: Manifest filename (e.g., "faces.jsonl")
            content: Manifest content
            checksum: Optional pre-computed checksum

        Returns:
            ManifestVersion for the new version
        """
        with self._lock:
            manifest_dir = self._ep_manifest_dir(ep_id)
            manifest_dir.mkdir(parents=True, exist_ok=True)

            # Load existing versions
            index = self._load_version_index(ep_id)
            versions = index.get(manifest_name, [])

            # Determine new version number
            current_max = max((v.get("version", 0) for v in versions), default=0)
            new_version = current_max + 1

            # Create versioned filename
            base_name, ext = os.path.splitext(manifest_name)
            versioned_name = f"{base_name}_v{new_version:04d}{ext}"
            versioned_path = manifest_dir / versioned_name

            # Write versioned file
            if isinstance(content, str):
                versioned_path.write_text(content, encoding="utf-8")
                size_bytes = len(content.encode("utf-8"))
            else:
                versioned_path.write_bytes(content)
                size_bytes = len(content)

            # Compute checksum if not provided
            if checksum is None:
                if isinstance(content, str):
                    checksum = compute_checksum(content.encode("utf-8"))
                else:
                    checksum = compute_checksum(content)

            # Create version record
            version_record = ManifestVersion(
                version=new_version,
                path=str(versioned_path),
                created_at=datetime.utcnow(),
                size_bytes=size_bytes,
                checksum=checksum,
                is_active=True,
            )

            # Mark all previous versions as inactive
            for v in versions:
                v["is_active"] = False

            versions.append(version_record.to_dict())

            # Prune old versions if needed
            if len(versions) > self._max_versions:
                # Keep the most recent versions
                versions = sorted(versions, key=lambda v: v["version"], reverse=True)
                to_delete = versions[self._max_versions:]
                versions = versions[:self._max_versions]

                for old_v in to_delete:
                    old_path = Path(old_v["path"])
                    if old_path.exists():
                        try:
                            old_path.unlink()
                        except Exception as exc:
                            LOGGER.warning(
                                "[manifest-version] Failed to delete old version %s: %s",
                                old_path,
                                exc,
                            )

            index[manifest_name] = versions
            self._save_version_index(ep_id, index)

            # Also write/update the "current" manifest file
            current_path = manifest_dir / manifest_name
            if isinstance(content, str):
                current_path.write_text(content, encoding="utf-8")
            else:
                current_path.write_bytes(content)

            LOGGER.debug(
                "[manifest-version] Saved %s v%d for %s",
                manifest_name,
                new_version,
                ep_id,
            )

            return version_record

    def list_versions(self, ep_id: str, manifest_name: str) -> List[ManifestVersion]:
        """List all versions of a manifest."""
        index = self._load_version_index(ep_id)
        versions_data = index.get(manifest_name, [])
        return [
            ManifestVersion(
                version=v["version"],
                path=v["path"],
                created_at=datetime.fromisoformat(v["created_at"]),
                size_bytes=v.get("size_bytes", 0),
                checksum=v.get("checksum"),
                is_active=v.get("is_active", False),
            )
            for v in sorted(versions_data, key=lambda x: x["version"], reverse=True)
        ]

    def get_active_version(self, ep_id: str, manifest_name: str) -> Optional[ManifestVersion]:
        """Get the currently active version of a manifest."""
        versions = self.list_versions(ep_id, manifest_name)
        for v in versions:
            if v.is_active:
                return v
        return versions[0] if versions else None

    def rollback_to_version(
        self,
        ep_id: str,
        manifest_name: str,
        version: int,
    ) -> Optional[ManifestVersion]:
        """Rollback a manifest to a specific version.

        Args:
            ep_id: Episode ID
            manifest_name: Manifest filename
            version: Version number to rollback to

        Returns:
            The activated version, or None if not found
        """
        with self._lock:
            index = self._load_version_index(ep_id)
            versions = index.get(manifest_name, [])

            target_version = None
            for v in versions:
                if v["version"] == version:
                    target_version = v
                    break

            if target_version is None:
                LOGGER.warning(
                    "[manifest-version] Version %d not found for %s/%s",
                    version,
                    ep_id,
                    manifest_name,
                )
                return None

            # Mark all as inactive, target as active
            for v in versions:
                v["is_active"] = v["version"] == version

            index[manifest_name] = versions
            self._save_version_index(ep_id, index)

            # Copy versioned file to current
            versioned_path = Path(target_version["path"])
            current_path = self._ep_manifest_dir(ep_id) / manifest_name

            if versioned_path.exists():
                shutil.copy2(versioned_path, current_path)
            else:
                LOGGER.warning(
                    "[manifest-version] Version file missing: %s",
                    versioned_path,
                )
                return None

            LOGGER.info(
                "[manifest-version] Rolled back %s/%s to v%d",
                ep_id,
                manifest_name,
                version,
            )

            return ManifestVersion(
                version=target_version["version"],
                path=target_version["path"],
                created_at=datetime.fromisoformat(target_version["created_at"]),
                size_bytes=target_version.get("size_bytes", 0),
                checksum=target_version.get("checksum"),
                is_active=True,
            )


# =============================================================================
# Buffered Manifest Writer (C28, D33)
# =============================================================================


class BufferedManifestWriter:
    """Buffered writer for JSONL manifests with atomic writes.

    This implements requirements:
    - C28: Manifest files appended line-by-line (buffered writes)
    - D33: Partial writes can corrupt manifests (atomic writes)

    Features:
    - Buffers entries in memory and flushes in batches
    - Uses atomic write pattern (temp file + rename)
    - Validates entries before writing
    - Tracks line count for corruption detection
    """

    def __init__(
        self,
        path: Path,
        manifest_type: Optional[ManifestType] = None,
        buffer_size: int = MANIFEST_BUFFER_SIZE,
        auto_flush_interval: float = MANIFEST_BUFFER_TIMEOUT,
        validate_entries: bool = True,
    ) -> None:
        """Initialize buffered writer.

        Args:
            path: Path to manifest file
            manifest_type: Type for validation (inferred if not provided)
            buffer_size: Number of entries to buffer before flushing
            auto_flush_interval: Maximum time between flushes
            validate_entries: Whether to validate entries before writing
        """
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_type = manifest_type or infer_manifest_type(path.name)
        self._buffer_size = buffer_size
        self._auto_flush_interval = auto_flush_interval
        self._validate_entries = validate_entries

        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self._total_written = 0
        self._closed = False

        # Initialize footer tracking for corruption detection
        self._footer_path = path.with_suffix(path.suffix + ".meta")

    def write(self, entry: Dict[str, Any]) -> bool:
        """Write an entry to the buffer.

        Returns True if entry was accepted, False if validation failed.
        """
        if self._closed:
            raise RuntimeError("Cannot write to closed manifest writer")

        # Validate if enabled
        if self._validate_entries:
            errors = validate_manifest_entry(entry, self._manifest_type)
            if errors:
                LOGGER.warning(
                    "[manifest-buffer] Entry validation failed: %s",
                    errors[0].message,
                )
                return False

        with self._lock:
            self._buffer.append(entry)

            # Check if we should flush
            should_flush = (
                len(self._buffer) >= self._buffer_size
                or time.time() - self._last_flush >= self._auto_flush_interval
            )

            if should_flush:
                self._flush_locked()

        return True

    def write_many(self, entries: List[Dict[str, Any]]) -> int:
        """Write multiple entries. Returns count of accepted entries."""
        accepted = 0
        for entry in entries:
            if self.write(entry):
                accepted += 1
        return accepted

    def flush(self) -> None:
        """Force flush the buffer."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Internal flush (must hold lock)."""
        if not self._buffer:
            return

        try:
            # Read existing content if file exists
            existing_lines = []
            if self._path.exists():
                existing_lines = self._path.read_text(encoding="utf-8").splitlines()

            # Add buffered entries
            new_lines = [json.dumps(entry, separators=(",", ":")) for entry in self._buffer]
            all_lines = existing_lines + new_lines

            # Write atomically via temp file
            content = "\n".join(all_lines)
            if content and not content.endswith("\n"):
                content += "\n"

            tmp_path = self._path.with_suffix(".tmp")
            tmp_path.write_text(content, encoding="utf-8")
            tmp_path.replace(self._path)

            # Update footer with line count (for corruption detection)
            footer = {
                "line_count": len(all_lines),
                "last_flush": datetime.utcnow().isoformat(),
                "checksum": compute_checksum(content.encode("utf-8"))[:16],
            }
            self._footer_path.write_text(json.dumps(footer))

            self._total_written += len(self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()

            LOGGER.debug(
                "[manifest-buffer] Flushed %d entries to %s (total: %d)",
                len(new_lines),
                self._path.name,
                len(all_lines),
            )

        except Exception as exc:
            LOGGER.exception("[manifest-buffer] Failed to flush: %s", exc)
            raise

    def close(self) -> None:
        """Close the writer, flushing any remaining entries."""
        if self._closed:
            return

        self.flush()
        self._closed = True

    @property
    def pending_count(self) -> int:
        """Number of entries in buffer."""
        return len(self._buffer)

    @property
    def total_written(self) -> int:
        """Total entries written to disk."""
        return self._total_written

    def verify_integrity(self) -> bool:
        """Verify manifest integrity using footer metadata.

        Returns True if manifest appears intact.
        """
        if not self._path.exists():
            return True  # Empty is valid

        if not self._footer_path.exists():
            # No footer - can't verify, assume ok
            return True

        try:
            footer = json.loads(self._footer_path.read_text())
            expected_lines = footer.get("line_count", 0)
            expected_checksum = footer.get("checksum", "")

            # Count actual lines
            content = self._path.read_text(encoding="utf-8")
            actual_lines = len([l for l in content.splitlines() if l.strip()])

            if actual_lines != expected_lines:
                LOGGER.warning(
                    "[manifest-buffer] Line count mismatch for %s: expected %d, got %d",
                    self._path,
                    expected_lines,
                    actual_lines,
                )
                return False

            # Verify checksum
            actual_checksum = compute_checksum(content.encode("utf-8"))[:16]
            if actual_checksum != expected_checksum:
                LOGGER.warning(
                    "[manifest-buffer] Checksum mismatch for %s",
                    self._path,
                )
                return False

            return True

        except Exception as exc:
            LOGGER.warning("[manifest-buffer] Integrity check failed: %s", exc)
            return False

    def __enter__(self) -> "BufferedManifestWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# =============================================================================
# Convenience Functions
# =============================================================================


def atomic_write_manifest(
    path: Path,
    content: Union[str, bytes],
    backup: bool = True,
) -> bool:
    """Write a manifest file atomically.

    This implements requirement D33: Partial writes can corrupt manifests.

    Args:
        path: Target path
        content: Content to write
        backup: If True, keep backup of previous version

    Returns:
        True on success
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Backup existing
        if backup and path.exists():
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)

        # Write to temp and rename
        tmp_path = path.with_suffix(".tmp")
        if isinstance(content, str):
            tmp_path.write_text(content, encoding="utf-8")
        else:
            tmp_path.write_bytes(content)
        tmp_path.replace(path)

        return True

    except Exception as exc:
        LOGGER.exception("[manifest] Atomic write failed for %s: %s", path, exc)
        return False


def read_manifest_safe(
    path: Path,
    validate: bool = True,
) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """Read a manifest file safely with optional validation.

    Returns:
        Tuple of (entries, error_message). Entries is None on error.
    """
    if not path.exists():
        return None, f"File not found: {path}"

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        return None, f"Read error: {exc}"

    entries = []
    errors = []

    is_jsonl = path.suffix.lower() in (".jsonl", ".ndjson")

    if is_jsonl:
        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_num}: {exc}")
    else:
        try:
            data = json.loads(content)
            if isinstance(data, list):
                entries = data
            else:
                entries = [data]
        except json.JSONDecodeError as exc:
            return None, f"JSON parse error: {exc}"

    if errors:
        return entries, f"Partial parse errors: {len(errors)} lines failed"

    if validate:
        manifest_type = infer_manifest_type(path.name)
        result = validate_manifest_file(path, manifest_type)
        if not result.is_valid:
            return entries, f"Validation errors: {len(result.errors)}"

    return entries, None


# =============================================================================
# Module exports
# =============================================================================

# Global instances
_checksum_registry: Optional[ChecksumRegistry] = None
_version_manager: Optional[ManifestVersionManager] = None


def get_checksum_registry() -> ChecksumRegistry:
    """Get global checksum registry."""
    global _checksum_registry
    if _checksum_registry is None:
        _checksum_registry = ChecksumRegistry()
    return _checksum_registry


def get_version_manager() -> ManifestVersionManager:
    """Get global manifest version manager."""
    global _version_manager
    if _version_manager is None:
        _version_manager = ManifestVersionManager()
    return _version_manager


__all__ = [
    # Validation
    "ManifestType",
    "ManifestValidationResult",
    "ValidationError",
    "validate_manifest_entry",
    "validate_manifest_file",
    "infer_manifest_type",
    # Checksums
    "ArtifactChecksum",
    "ChecksumRegistry",
    "compute_checksum",
    "compute_file_checksum",
    "get_checksum_registry",
    # Versioning
    "ManifestVersion",
    "ManifestVersionManager",
    "get_version_manager",
    # Buffered writing
    "BufferedManifestWriter",
    "MANIFEST_BUFFER_SIZE",
    "MANIFEST_BUFFER_TIMEOUT",
    # Convenience
    "atomic_write_manifest",
    "read_manifest_safe",
]
