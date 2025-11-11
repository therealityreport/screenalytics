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
        if not path.exists():
            return {
                "show_id": show_id,
                "cast_id": cast_id,
                "seeds": [],
                "exemplars": [],
                "updated_at": _now_iso(),
            }
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {
                "show_id": show_id,
                "cast_id": cast_id,
                "seeds": [],
                "exemplars": [],
                "updated_at": _now_iso(),
            }

    def _save_facebank(self, show_id: str, cast_id: str, data: Dict[str, Any]) -> None:
        """Save facebank.json."""
        path = self._facebank_path(show_id, cast_id)
        data["updated_at"] = _now_iso()
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

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

        return {
            "show_id": show_id,
            "cast_id": cast_id,
            "seeds": seeds,
            "exemplars": exemplars,
            "stats": stats,
        }

    def add_seed(
        self,
        show_id: str,
        cast_id: str,
        image_path: str,
        embedding: np.ndarray,
        quality: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a seed image with its embedding."""
        data = self._load_facebank(show_id, cast_id)
        seeds = data.get("seeds", [])

        seed_id = str(uuid.uuid4())

        seed_entry = {
            "fb_id": seed_id,
            "cast_id": cast_id,
            "type": "seed",
            "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            "embedding_dim": len(embedding),
            "quality": quality or {},
            "source": "upload",
            "image_uri": image_path,
            "created_at": _now_iso(),
        }

        seeds.append(seed_entry)
        data["seeds"] = seeds
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
            self._save_facebank(show_id, cast_id, data)

        return deleted_count

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
