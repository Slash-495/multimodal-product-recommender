import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from src.models.two_tower import TwoTowerModel
from src.retrieval.retriever import TwoTowerRetriever
from src.utils.config import DEFAULT_CONFIG


def create_synthetic_item_loader(num_items: int = 50, batch_size: int = 16) -> DataLoader:
    """Create a synthetic item DataLoader yielding item feature dictionaries."""
    config = DEFAULT_CONFIG["model"]
    
    # Original business IDs range from 1000 to 1000 + num_items - 1
    business_ids = torch.arange(1000, 1000 + num_items, dtype=torch.float32).unsqueeze(1)
    business_stats = torch.randn(num_items, 2)  # 2 statistical features
    business_features = torch.cat([business_ids, business_stats], dim=1)

    category_features = torch.zeros(num_items, 10)
    category_indices = torch.randint(0, 10, (num_items,))
    category_features[torch.arange(num_items), category_indices] = 1.0

    dataset = TensorDataset(business_features, category_features)

    def collate_fn(batch):
        bf, cf = zip(*batch)
        return {
            "business_features": torch.stack(bf, dim=0),
            "category_features": torch.stack(cf, dim=0),
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


def create_synthetic_user_batch(num_users: int = 4) -> dict:
    """Create a synthetic user batch dictionary."""
    config = DEFAULT_CONFIG["model"]
    user_ids = torch.randint(0, config["num_users"], (num_users,), dtype=torch.float32).unsqueeze(1)
    user_stats = torch.randn(num_users, 3)
    user_features = torch.cat([user_ids, user_stats], dim=1)

    return {"user_features": user_features}


def test_build_item_index():
    """Verify retriever builds item index and preserves original item ID mapping."""
    model = TwoTowerModel(DEFAULT_CONFIG["model"])
    retriever = TwoTowerRetriever(model=model, device="cpu")

    item_loader = create_synthetic_item_loader(num_items=50, batch_size=16)
    retriever.build_item_index(item_loader)

    assert retriever.faiss_index.num_items == 50
    assert retriever.item_ids is not None
    assert len(retriever.item_ids) == 50

    # Verify original business IDs 1000..1049 are preserved in order
    expected_ids = np.arange(1000, 1050)
    np.testing.assert_array_equal(retriever.item_ids, expected_ids)


def test_user_retrieval_end_to_end():
    """Verify user encoding, FAISS search, and result shape & item ID validity."""
    model = TwoTowerModel(DEFAULT_CONFIG["model"])
    retriever = TwoTowerRetriever(model=model, device="cpu")

    item_loader = create_synthetic_item_loader(num_items=50, batch_size=16)
    retriever.build_item_index(item_loader)

    num_users = 4
    k = 5
    user_batch = create_synthetic_user_batch(num_users=num_users)

    item_ids, scores = retriever.retrieve(user_batch, k=k)

    # Check shapes
    assert item_ids.shape == (num_users, k)
    assert scores.shape == (num_users, k)

    # Verify all returned item IDs are valid original business IDs (1000..1049)
    valid_id_set = set(range(1000, 1050))
    for user_retrieved in item_ids:
        for item_id in user_retrieved:
            assert int(item_id) in valid_id_set


def test_retrieval_scores_descending_order():
    """Verify returned similarity scores are sorted in descending order per user."""
    model = TwoTowerModel(DEFAULT_CONFIG["model"])
    retriever = TwoTowerRetriever(model=model, device="cpu")

    item_loader = create_synthetic_item_loader(num_items=50, batch_size=16)
    retriever.build_item_index(item_loader)

    user_batch = create_synthetic_user_batch(num_users=3)
    _, scores = retriever.retrieve(user_batch, k=10)

    for user_scores in scores:
        # Each score should be >= next score
        diffs = np.diff(user_scores)
        assert np.all(diffs <= 1e-6), f"Scores not sorted descending: {user_scores}"


def test_retriever_save_and_load(tmp_path):
    """Verify saving and loading retriever state preserves search results."""
    model = TwoTowerModel(DEFAULT_CONFIG["model"])
    retriever = TwoTowerRetriever(model=model, device="cpu")

    item_loader = create_synthetic_item_loader(num_items=50, batch_size=16)
    retriever.build_item_index(item_loader)

    user_batch = create_synthetic_user_batch(num_users=4)
    orig_item_ids, orig_scores = retriever.retrieve(user_batch, k=5)

    save_path = tmp_path / "retriever_test"
    retriever.save(save_path)

    # Load retriever back
    loaded_retriever = TwoTowerRetriever.load(save_path, model=model, device="cpu")

    assert loaded_retriever.embedding_dim == retriever.embedding_dim
    assert loaded_retriever.faiss_index.num_items == 50

    loaded_item_ids, loaded_scores = loaded_retriever.retrieve(user_batch, k=5)

    np.testing.assert_array_equal(orig_item_ids, loaded_item_ids)
    np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-5)


def test_retriever_validation_errors():
    """Verify proper ValueError exceptions are raised for invalid retriever usage."""
    model = TwoTowerModel(DEFAULT_CONFIG["model"])
    retriever = TwoTowerRetriever(model=model, device="cpu")

    user_batch = create_synthetic_user_batch(num_users=2)

    # Search before build_item_index
    with pytest.raises(ValueError, match="Item index is empty"):
        retriever.retrieve(user_batch, k=5)

    # Save before build_item_index
    with pytest.raises(ValueError, match="Cannot save empty retriever index"):
        retriever.save("dummy_path")

    # Build index
    item_loader = create_synthetic_item_loader(num_items=20, batch_size=10)
    retriever.build_item_index(item_loader)

    # Invalid k <= 0
    with pytest.raises(ValueError, match="k must be a positive integer"):
        retriever.retrieve(user_batch, k=0)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        retriever.retrieve(user_batch, k=-2)
