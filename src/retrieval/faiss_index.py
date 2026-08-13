import os
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import faiss


class FAISSIndex:
    """FAISS index manager using inner product (cosine similarity after L2 normalization)."""

    def __init__(self, embedding_dim: int):
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be a positive integer, got {embedding_dim}")

        self.embedding_dim: int = embedding_dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(embedding_dim)
        self.num_items: int = 0

    def build(self, embeddings: np.ndarray) -> None:
        """
        Build index from item embeddings.

        Args:
            embeddings: NumPy array of shape [num_items, embedding_dim]
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2-dimensional, got shape with ndim={embeddings.ndim}")

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}"
            )

        # Make a contiguous float32 copy for FAISS
        embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_f32)

        # Re-initialize index to clear any previous items
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings_f32)
        self.num_items = self.index.ntotal

    def search(
        self, query_embeddings: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search top-k nearest item embeddings for query embeddings.

        Args:
            query_embeddings: NumPy array of shape [num_queries, embedding_dim]
            k: Number of nearest neighbors to retrieve

        Returns:
            Tuple of (scores [num_queries, k], item_indices [num_queries, k])
        """
        if self.num_items == 0 or self.index.ntotal == 0:
            raise ValueError("Index is empty. Call build() before searching.")

        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        if not isinstance(query_embeddings, np.ndarray):
            query_embeddings = np.array(query_embeddings)

        if query_embeddings.ndim != 2:
            raise ValueError(
                f"Query embeddings must be 2-dimensional, got shape with ndim={query_embeddings.ndim}"
            )

        if query_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dim}, got {query_embeddings.shape[1]}"
            )

        # Make a contiguous float32 copy for FAISS
        queries_f32 = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        faiss.normalize_L2(queries_f32)

        scores, indices = self.index.search(queries_f32, k)
        return scores, indices

    def save(self, path: Union[str, Path]) -> None:
        """Save FAISS index to disk."""
        if self.index is None:
            raise ValueError("Index is not initialized.")
        path_str = str(path)
        os.makedirs(os.path.dirname(os.path.abspath(path_str)), exist_ok=True)
        faiss.write_index(self.index, path_str)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FAISSIndex":
        """Load FAISS index from disk and return a new FAISSIndex instance."""
        path_str = str(path)
        if not os.path.exists(path_str):
            raise ValueError(f"Index file not found at path: {path_str}")

        loaded_index = faiss.read_index(path_str)
        obj = cls(embedding_dim=loaded_index.d)
        obj.index = loaded_index
        obj.num_items = loaded_index.ntotal
        return obj


ProductIndex = FAISSIndex
BusinessIndex = FAISSIndex
