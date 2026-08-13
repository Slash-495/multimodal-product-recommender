import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pytest
from src.retrieval.faiss_index import ProductIndex, BusinessIndex, FAISSIndex


def test_faiss_index_initialization_and_build():
    """Verify FAISS index builds correctly and tracks item count."""
    dim = 128
    num_items = 50
    embeddings = np.random.randn(num_items, dim).astype(np.float32)

    index = ProductIndex(embedding_dim=dim)
    index.build(embeddings)

    assert index.num_items == num_items
    assert index.index.ntotal == num_items


def test_faiss_search_exact_vector():
    """Verify searching an exact vector returns the correct item index with cosine similarity ~ 1.0."""
    dim = 128
    num_items = 20
    np.random.seed(42)
    embeddings = np.random.randn(num_items, dim).astype(np.float32)

    index = ProductIndex(embedding_dim=dim)
    index.build(embeddings)

    # Query with item at index 7
    target_idx = 7
    query = embeddings[target_idx : target_idx + 1]

    scores, indices = index.search(query, k=5)

    assert indices.shape == (1, 5)
    assert scores.shape == (1, 5)
    assert indices[0, 0] == target_idx
    assert np.isclose(scores[0, 0], 1.0, atol=1e-5)


def test_faiss_search_output_shapes():
    """Verify search returns correct score and index output matrix shapes."""
    dim = 64
    num_items = 30
    num_queries = 4
    k = 8

    embeddings = np.random.randn(num_items, dim).astype(np.float32)
    queries = np.random.randn(num_queries, dim).astype(np.float32)

    index = BusinessIndex(embedding_dim=dim)
    index.build(embeddings)

    scores, indices = index.search(queries, k=k)

    assert scores.shape == (num_queries, k)
    assert indices.shape == (num_queries, k)


def test_faiss_save_and_load(tmp_path):
    """Verify saving and loading the index preserves search results."""
    dim = 128
    num_items = 40
    embeddings = np.random.randn(num_items, dim).astype(np.float32)
    queries = np.random.randn(3, dim).astype(np.float32)

    index = ProductIndex(embedding_dim=dim)
    index.build(embeddings)

    orig_scores, orig_indices = index.search(queries, k=5)

    save_path = tmp_path / "faiss_product.index"
    index.save(save_path)

    loaded_index = ProductIndex.load(save_path)

    assert loaded_index.embedding_dim == dim
    assert loaded_index.num_items == num_items

    loaded_scores, loaded_indices = loaded_index.search(queries, k=5)

    np.testing.assert_array_equal(orig_indices, loaded_indices)
    np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-5)


def test_faiss_validation_errors(tmp_path):
    """Verify proper ValueError exceptions are raised for invalid inputs."""
    dim = 128

    # Invalid initialization dimension
    with pytest.raises(ValueError, match="embedding_dim must be a positive integer"):
        ProductIndex(embedding_dim=0)

    with pytest.raises(ValueError, match="embedding_dim must be a positive integer"):
        ProductIndex(embedding_dim=-5)

    index = ProductIndex(embedding_dim=dim)

    # Search before build
    queries = np.random.randn(1, dim).astype(np.float32)
    with pytest.raises(ValueError, match="Index is empty"):
        index.search(queries, k=5)

    # Invalid embedding dimension in build
    wrong_dim_embeddings = np.random.randn(10, 64).astype(np.float32)
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        index.build(wrong_dim_embeddings)

    # 1D array in build
    with pytest.raises(ValueError, match="Embeddings must be 2-dimensional"):
        index.build(np.random.randn(128).astype(np.float32))

    # Build valid index
    embeddings = np.random.randn(20, dim).astype(np.float32)
    index.build(embeddings)

    # Invalid k in search
    with pytest.raises(ValueError, match="k must be a positive integer"):
        index.search(queries, k=0)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        index.search(queries, k=-3)

    # Query dimension mismatch
    wrong_query = np.random.randn(1, 32).astype(np.float32)
    with pytest.raises(ValueError, match="Query embedding dimension mismatch"):
        index.search(wrong_query, k=5)

    # Query 1D array
    with pytest.raises(ValueError, match="Query embeddings must be 2-dimensional"):
        index.search(np.random.randn(128).astype(np.float32), k=5)

    # Non-existent file in load
    fake_path = tmp_path / "non_existent.index"
    with pytest.raises(ValueError, match="Index file not found"):
        ProductIndex.load(fake_path)
