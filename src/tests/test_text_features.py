import copy
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.features.text_features import MovieTitleTextEmbedder
from src.models.towers.item_tower import ItemTower
from src.models.two_tower import TwoTowerModel
from src.utils.config import DEFAULT_CONFIG


def test_tfidf_fit_and_transform():
    """Verify TF-IDF fitting and transformation produces valid embeddings."""
    titles = ["Toy Story (1995)", "Jumanji (1995)", "Grumpier Old Men (1995)"]
    embedder = MovieTitleTextEmbedder(tfidf_max_features=100, svd_components=2, random_state=42)
    embedder.fit(titles)

    assert embedder.is_fitted
    embs = embedder.transform(titles)
    assert embs.shape == (3, 2)
    assert embs.dtype == np.float32


def test_svd_output_dimensionality_and_no_nans():
    """Verify SVD output shape matches svd_components and contains zero NaNs."""
    titles = [f"Movie Title {i}" for i in range(50)]
    embedder = MovieTitleTextEmbedder(tfidf_max_features=500, svd_components=16, random_state=42)
    embs = embedder.fit_transform(titles)

    assert embs.shape == (50, 16)
    assert not np.isnan(embs).any()


def test_deterministic_transformation():
    """Verify text transformation is 100% deterministic with fixed random seed."""
    titles = [f"Movie Special Action Drama Title {i}" for i in range(20)]

    embedder1 = MovieTitleTextEmbedder(tfidf_max_features=200, svd_components=8, random_state=42)
    embs1 = embedder1.fit_transform(titles)

    embedder2 = MovieTitleTextEmbedder(tfidf_max_features=200, svd_components=8, random_state=42)
    embs2 = embedder2.fit_transform(titles)

    assert np.allclose(embs1, embs2, atol=1e-6)


def test_save_and_load_consistency(tmp_path):
    """Verify saved MovieTitleTextEmbedder loads with identical transformation output."""
    titles = ["Toy Story (1995)", "GoldenEye (1995)", "Heat (1995)"]
    save_path = tmp_path / "embedder.joblib"

    embedder = MovieTitleTextEmbedder(tfidf_max_features=300, svd_components=4, random_state=42)
    embs_orig = embedder.fit_transform(titles)
    embedder.save(save_path)

    loaded_embedder = MovieTitleTextEmbedder.load(save_path)
    embs_loaded = loaded_embedder.transform(titles)

    assert np.allclose(embs_orig, embs_loaded, atol=1e-6)


def test_item_tower_backward_compatible_when_text_disabled():
    """Verify ItemTower behavior is 100% backward compatible when use_text_features=False."""
    item_tower = ItemTower(
        num_items=100,
        embedding_dim=128,
        hidden_dims=[256, 128],
        dropout=0.1,
        use_text_features=False,
    )

    item_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    b_features = torch.zeros((3, 2), dtype=torch.float32)
    c_features = torch.zeros((3, 10), dtype=torch.float32)

    output = item_tower(item_ids, b_features, c_features)
    assert output.shape == (3, 128)


def test_item_tower_forward_pass_with_text_features():
    """Verify ItemTower forward pass succeeds when use_text_features=True."""
    item_tower = ItemTower(
        num_items=100,
        embedding_dim=128,
        hidden_dims=[256, 128],
        dropout=0.1,
        use_text_features=True,
        text_embedding_dim=32,
    )

    item_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    b_features = torch.zeros((3, 2), dtype=torch.float32)
    c_features = torch.zeros((3, 10), dtype=torch.float32)
    t_features = torch.randn((3, 32), dtype=torch.float32)

    output = item_tower(item_ids, b_features, c_features, text_features=t_features)
    assert output.shape == (3, 128)

    # Verify missing text_features raises ValueError when use_text_features=True
    with pytest.raises(ValueError, match="text_features was not provided"):
        item_tower(item_ids, b_features, c_features)


def test_two_tower_synthetic_end_to_end_pipeline_with_text():
    """Verify full synthetic forward & backward pass with content text features enabled."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["model"]["num_users"] = 50
    config["model"]["num_items"] = 50
    config["model"]["item_tower"]["use_text_features"] = True
    config["model"]["item_tower"]["text_embedding_dim"] = 32

    model = TwoTowerModel(config["model"])

    batch = {
        "user_features": torch.randn((8, 4), dtype=torch.float32),
        "business_features": torch.randn((8, 3), dtype=torch.float32),
        "category_features": torch.randn((8, 10), dtype=torch.float32),
        "text_features": torch.randn((8, 32), dtype=torch.float32),
    }

    # Ensure user_idx and business_idx are valid longs
    batch["user_features"][:, 0] = torch.randint(0, 50, (8,)).float()
    batch["business_features"][:, 0] = torch.randint(0, 50, (8,)).float()

    u_emb, i_emb = model(batch)
    assert u_emb.shape == (8, 128)
    assert i_emb.shape == (8, 128)

    # Verify backward gradient flow
    loss = (u_emb * i_emb).sum()
    loss.backward()

    assert model.item_tower.item_embedding.weight.grad is not None
