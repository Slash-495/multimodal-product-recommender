import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import torch
import pytest

from src.data.preprocess import process_features, split_data
from src.data.data_loader import get_dataloader
from src.models.two_tower import TwoTowerModel
from src.utils.losses import InfoNCELoss
from src.utils.config import DEFAULT_CONFIG


def test_synthetic_end_to_end_data_to_training_pipeline(tmp_path):
    """
    Synthetic end-to-end test verifying:
    raw dataframe -> feature processing & splitting -> YelpDataset & DataLoader
    -> TwoTowerModel -> InfoNCELoss -> backward pass.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Create synthetic raw data
    num_samples = 60
    reviews = pd.DataFrame(
        {
            "user_id": [f"user_{i % 10}" for i in range(num_samples)],
            "business_id": [f"biz_{i % 15}" for i in range(num_samples)],
            "stars": np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=num_samples),
            "date": pd.date_range("2023-01-01", periods=num_samples, freq="D").astype(str),
        }
    )

    users = pd.DataFrame(
        {
            "user_id": [f"user_{i}" for i in range(10)],
            "review_count": np.random.randint(1, 100, size=10),
            "yelping_since": ["2020-01-15"] * 10,
            "average_stars": np.random.uniform(2.5, 5.0, size=10),
        }
    )

    businesses = pd.DataFrame(
        {
            "business_id": [f"biz_{i}" for i in range(15)],
            "stars": np.random.uniform(2.0, 5.0, size=15),
            "review_count": np.random.randint(5, 500, size=15),
            "categories": [
                "Restaurants, Food",
                "Bars, Nightlife",
                "Shopping, Fashion",
                "Restaurants, Italian",
                "Coffee & Tea, Food",
            ]
            * 3,
        }
    )

    # 2. Preprocess features and split
    p_reviews, p_users, p_biz = process_features(
        reviews, users, businesses, top_k_categories=10, reference_date="2024-01-01"
    )
    train_df, valid_df, test_df = split_data(p_reviews, train_ratio=0.8, valid_ratio=0.1, time_based=True)

    # 3. Save CSVs to temporary data directory
    train_df.to_csv(tmp_path / "train_interactions.csv", index=False)
    valid_df.to_csv(tmp_path / "valid_interactions.csv", index=False)
    test_df.to_csv(tmp_path / "test_interactions.csv", index=False)
    p_users.to_csv(tmp_path / "user_features.csv", index=False)
    p_biz.to_csv(tmp_path / "business_features.csv", index=False)

    # 4. Create DataLoaders
    train_loader = get_dataloader(data_path=tmp_path, batch_size=16, mode="train", num_workers=0)
    valid_loader = get_dataloader(data_path=tmp_path, batch_size=16, mode="valid", num_workers=0)

    assert len(train_loader) > 0
    assert len(valid_loader) > 0

    # 5. Instantiate Model and Loss
    model = TwoTowerModel(DEFAULT_CONFIG["model"])
    criterion = InfoNCELoss(temperature=DEFAULT_CONFIG["training"]["temperature"])

    # 6. Pass batch through model -> InfoNCE -> backward pass
    batch = next(iter(train_loader))
    user_emb, item_emb = model(batch)

    assert user_emb.shape == (16, 128)
    assert item_emb.shape == (16, 128)

    loss = criterion(user_emb, item_emb)
    assert torch.isfinite(loss), "InfoNCE loss must be finite"

    loss.backward()

    # Verify gradients computed
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    assert has_grad, "Gradients should be computed for model parameters after backward pass"
