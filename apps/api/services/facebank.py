"""Facebank service for managing seed images and embeddings."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_DATA_ROOT = Path("data").expanduser()
FACEBANK_BACKEND = os.getenv("FACEBANK_BACKEND", "faiss")  # faiss or pgvector
SEED_DET_BOOST_SIM = float(os.getenv("SEED_DET_BOOST_SIM", "0.42"))
SEED_ATTACH_SIM = float(os.getenv("SEED_ATTACH_SIM", "0.45"))
SEED_CLUSTER_DELTA = float(os.getenv("SEED_CLUSTER_DELTA", "0.05"))


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _compute_similarity_stats(seeds: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = {
        "sampled": 0,
        "summary": None,
        "per_seed": {},
    }
    embeddings: List[np.ndarray] = []
    seed_ids: List[str] = []
    for seed in seeds:
        embedding = seed.get("embedding")
        seed_id = seed.get("fb_id")
        if embedding is None or seed_id is None:
            continue
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        embeddings.append(vec / (norm + 1e-12))
        seed_ids.append(seed_id)
    stats["sampled"] = len(seed_ids)
    if not seed_ids:
        return stats

    matrix = np.dot(np.stack(embeddings), np.stack(embeddings).T)
    matrix = np.clip(matrix, -1.0, 1.0)
    np.fill_diagonal(matrix, 1.0)

    per_seed: Dict[str, Dict[str, float]] = {}
    for idx, seed_id in enumerate(seed_ids):
        others = np.delete(matrix[idx], idx)
        if others.size:
            per_seed[seed_id] = {
                "mean": round(float(np.mean(others)), 3),
                "min": round(float(np.min(others)), 3),
                "max": round(float(np.max(others)), 3),
            }
        else:
            per_seed[seed_id] = {"mean": 1.0, "min": 1.0, "max": 1.0}
    stats["per_seed"] = per_seed

    pairwise = matrix[np.triu_indices(len(seed_ids), k=1)]
    if pairwise.size:
        stats["summary"] = {
            "mean": round(float(np.mean(pairwise)), 3),
            "median": round(float(np.median(pairwise)), 3),
            "min": round(float(np.min(pairwise)), 3),
            "max": round(float(np.max(pairwise)), 3),
        }
    else:
        stats["summary"] = {"mean": 1.0, "median": 1.0, "min": 1.0, "max": 1.0}

    return stats


class FacebankService:
    """Manage facebank seeds and exemplars with embeddings."""

    def __init__(self, data_root: Path | str | None = None):
        self.data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
        self.facebank_dir = self.data_root / "facebank"
        self.facebank_dir.mkdir(parents=True, exist_ok=True)

    def _facebank_path(self, show_id: str, cast_id: str) -> Path:
        """Get path to facebank.json for a cast member."""
        cast_dir = self.facebank_dir / show_id / cast_id
        cast_dir.mkdir(parents=True, exist_ok=True)
        return cast_dir / "facebank.json"

    def _seeds_dir(self, show_id: str, cast_id: str) -> Path:
        """Get directory for seed images."""
        seeds_dir = self.facebank_dir / show_id / cast_id / "seeds"
        seeds_dir.mkdir(parents=True, exist_ok=True)
        return seeds_dir

    def _load_facebank(self, show_id: str, cast_id: str) -> Dict[str, Any]:
        """Load facebank.json or create empty structure."""
        path = self._facebank_path(show_id, cast_id)
        default_payload = {
            "show_id": show_id,
            "cast_id": cast_id,
            "seeds": [],
            "exemplars": [],
            "updated_at": _now_iso(),
            "featured_seed_id": None,
        }
        if not path.exists():
            return default_payload
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default_payload
        if not isinstance(data, dict):
            return default_payload
        data.setdefault("featured_seed_id", None)
        seeds = data.get("seeds")
        if isinstance(seeds, list):
            for seed in seeds:
                if isinstance(seed, dict) and "image_s3_key" not in seed:
                    seed["image_s3_key"] = None
                if isinstance(seed, dict) and "display_s3_key" not in seed:
                    seed["display_s3_key"] = seed.get("image_s3_key")
                if isinstance(seed, dict) and "embed_s3_key" not in seed:
                    seed["embed_s3_key"] = None
                if isinstance(seed, dict) and "embed_uri" not in seed:
                    seed["embed_uri"] = None
                if isinstance(seed, dict) and "orig_uri" not in seed:
                    seed["orig_uri"] = None
                if isinstance(seed, dict) and "orig_s3_key" not in seed:
                    seed["orig_s3_key"] = None
                if isinstance(seed, dict) and "storage_seed_id" not in seed:
                    seed["storage_seed_id"] = None
                if isinstance(seed, dict) and "display_dims" not in seed:
                    seed["display_dims"] = None
                if isinstance(seed, dict) and "embed_dims" not in seed:
                    seed["embed_dims"] = None
                if isinstance(seed, dict) and "display_low_res" not in seed:
                    seed["display_low_res"] = False
                if isinstance(seed, dict) and "display_key" not in seed:
                    seed["display_key"] = seed.get("display_s3_key") or seed.get(
                        "image_s3_key"
                    )
        return data

    def _save_facebank(self, show_id: str, cast_id: str, data: Dict[str, Any]) -> None:
        """Save facebank.json."""
        path = self._facebank_path(show_id, cast_id)
        data["updated_at"] = _now_iso()
        payload = json.dumps(data, indent=2)
        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, path)
        except OSError as exc:
            try:
                tmp_path.unlink(missing_ok=True)
            except TypeError:  # Python < 3.8 compatibility
                if tmp_path.exists():
                    tmp_path.unlink()
            raise RuntimeError(
                f"Failed to persist facebank data for {show_id}/{cast_id}: {exc}"
            ) from exc

    def get_facebank(self, show_id: str, cast_id: str) -> Dict[str, Any]:
        """Get facebank data for a cast member."""
        data = self._load_facebank(show_id, cast_id)

        # Add stats
        seeds = data.get("seeds", [])
        exemplars = data.get("exemplars", [])

        stats = {
            "total_seeds": len(seeds),
            "total_exemplars": len(exemplars),
            "updated_at": data.get("updated_at"),
        }
        similarity = _compute_similarity_stats(seeds)

        featured_id = data.get("featured_seed_id")
        seeds_with_flag = []
        for seed in seeds:
            entry = seed.copy()
            entry["featured"] = entry.get("fb_id") == featured_id
            seeds_with_flag.append(entry)

        return {
            "show_id": show_id,
            "cast_id": cast_id,
            "seeds": seeds_with_flag,
            "exemplars": exemplars,
            "stats": stats,
            "featured_seed_id": featured_id,
            "similarity": similarity,
        }

    def add_seed(
        self,
        show_id: str,
        cast_id: str,
        image_path: str,
        embedding: np.ndarray,
        quality: Optional[Dict[str, Any]] = None,
        image_s3_key: Optional[str] = None,
        embed_image_path: Optional[str] = None,
        embed_s3_key: Optional[str] = None,
        *,
        seed_id: str | None = None,
        seed_storage_id: str | None = None,
        display_uri: str | None = None,
        embed_uri: str | None = None,
        orig_image_path: str | None = None,
        orig_s3_key: str | None = None,
        display_dims: Optional[List[int]] = None,
        embed_dims: Optional[List[int]] = None,
        display_low_res: bool = False,
        detector_mode: str | None = None,
    ) -> Dict[str, Any]:
        """Add a seed image with its embedding."""
        data = self._load_facebank(show_id, cast_id)
        seeds = data.get("seeds", [])

        seed_id = seed_id or str(uuid.uuid4())
        storage_seed_id = seed_storage_id or seed_id

        seed_entry = {
            "fb_id": seed_id,
            "cast_id": cast_id,
            "type": "seed",
            "embedding": (
                embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            ),
            "embedding_dim": len(embedding),
            "quality": quality or {},
            "source": "upload",
            "image_uri": display_uri or image_path,
            "created_at": _now_iso(),
            "image_s3_key": image_s3_key,
            "display_s3_key": image_s3_key,
            "display_key": image_s3_key,
            "display_uri": display_uri or image_path,
            "embed_uri": embed_uri or embed_image_path,
            "embed_s3_key": embed_s3_key,
            "orig_uri": orig_image_path,
            "orig_s3_key": orig_s3_key,
            "storage_seed_id": storage_seed_id,
            "display_dims": list(display_dims) if display_dims else None,
            "embed_dims": list(embed_dims) if embed_dims else None,
            "display_low_res": bool(display_low_res),
            "detector_mode": detector_mode,
        }

        seeds.append(seed_entry)
        data["seeds"] = seeds
        if not data.get("featured_seed_id"):
            data["featured_seed_id"] = seed_id
        self._save_facebank(show_id, cast_id, data)

        return seed_entry

    def delete_seeds(self, show_id: str, cast_id: str, seed_ids: List[str]) -> int:
        """Delete seed entries by ID."""
        data = self._load_facebank(show_id, cast_id)
        seeds = data.get("seeds", [])

        original_count = len(seeds)
        seeds = [s for s in seeds if s["fb_id"] not in seed_ids]
        deleted_count = original_count - len(seeds)

        if deleted_count > 0:
            data["seeds"] = seeds
            if data.get("featured_seed_id") in seed_ids:
                data["featured_seed_id"] = seeds[0]["fb_id"] if seeds else None
            self._save_facebank(show_id, cast_id, data)

        return deleted_count

    def set_featured_seed(
        self, show_id: str, cast_id: str, seed_id: str
    ) -> Dict[str, Any]:
        """Mark a specific seed as featured."""
        data = self._load_facebank(show_id, cast_id)
        seeds = data.get("seeds", [])
        for seed in seeds:
            if seed.get("fb_id") == seed_id:
                data["featured_seed_id"] = seed_id
                self._save_facebank(show_id, cast_id, data)
                return seed
        raise ValueError(f"Seed {seed_id} not found for {show_id}/{cast_id}")

    def get_all_seeds_for_show(self, show_id: str) -> List[Dict[str, Any]]:
        """Get all seed embeddings for a show (for detection boosting)."""
        show_dir = self.facebank_dir / show_id
        if not show_dir.exists():
            return []

        all_seeds = []
        for cast_dir in show_dir.iterdir():
            if cast_dir.is_dir():
                cast_id = cast_dir.name
                data = self._load_facebank(show_id, cast_id)
                seeds = data.get("seeds", [])
                for seed in seeds:
                    seed["cast_id"] = cast_id
                    all_seeds.append(seed)

        return all_seeds

    def find_matching_seed(
        self,
        show_id: str,
        embedding: np.ndarray,
        min_similarity: float = SEED_ATTACH_SIM,
    ) -> Optional[Tuple[str, str, float]]:
        """Find the best matching seed for an embedding.

        Returns (cast_id, seed_id, similarity) if match found, else None.
        """
        all_seeds = self.get_all_seeds_for_show(show_id)

        best_cast_id = None
        best_seed_id = None
        best_sim = -1.0

        for seed in all_seeds:
            seed_embedding = np.array(seed["embedding"], dtype=np.float32)
            sim = cosine_similarity(embedding, seed_embedding)

            if sim > best_sim:
                best_sim = sim
                best_cast_id = seed["cast_id"]
                best_seed_id = seed["fb_id"]

        if best_sim >= min_similarity:
            return (best_cast_id, best_seed_id, best_sim)

        return None


__all__ = [
    "FacebankService",
    "FACEBANK_BACKEND",
    "SEED_DET_BOOST_SIM",
    "SEED_ATTACH_SIM",
    "SEED_CLUSTER_DELTA",
]
