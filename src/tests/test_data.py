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

from src.data.preprocess import process_features, get_top_categories, split_data
from src.data.data_loader import YelpDataset


def test_category_selection_deterministic():
    """Verify category selection is deterministic and frequency-sorted."""
    biz_df = pd.DataFrame(
        {
            "business_id": ["b1", "b2", "b3", "b4"],
            "stars": [4.0, 3.5, 5.0, 2.0],
            "review_count": [10, 20, 30, 40],
            "categories": [
                "Restaurants, Food, Italian",
                "Restaurants, Bars",
                "Food, Coffee & Tea",
                "Restaurants, Food",
            ],
        }
    )

    top_cats_1 = get_top_categories(biz_df, top_k=10)
    top_cats_2 = get_top_categories(biz_df, top_k=10)

    assert top_cats_1 == top_cats_2
    assert top_cats_1[0] == "Food"         # Count 3 ('Food' < 'Restaurants' alphabetically)
    assert top_cats_1[1] == "Restaurants"  # Count 3


def test_process_features_shapes_and_no_nans():
    """Verify process_features generates 10 category columns without NaNs."""
    reviews = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "business_id": ["b1", "b2"],
            "stars": [5.0, 3.0],
            "date": ["2023-01-01", "2023-01-02"],
        }
    )
    users = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "review_count": [15, None],
            "yelping_since": ["2020-01-01", None],
            "average_stars": [4.2, 3.8],
        }
    )
    businesses = pd.DataFrame(
        {
            "business_id": ["b1", "b2"],
            "stars": [4.0, None],
            "review_count": [100, 50],
            "categories": ["Restaurants, Bars", None],
        }
    )

    p_reviews, p_users, p_biz = process_features(reviews, users, businesses, top_k_categories=10)

    # Check 10 category columns
    cat_cols = [c for c in p_biz.columns if c.startswith("cat_")]
    assert len(cat_cols) == 10

    # Verify no NaNs in any dataframe
    assert not p_reviews.isna().any().any()
    assert not p_users.isna().any().any()
    assert not p_biz.isna().any().any()


def test_consistent_id_mapping_and_unknown_handling(tmp_path):
    """Verify YelpDataset uses consistent ID mappings and handles unknown entities safely."""
    # Write processed CSV files
    train_inter = pd.DataFrame(
        {"user_id": ["u1", "u2"], "business_id": ["b1", "b2"], "stars": [5, 4]}
    )
    valid_inter = pd.DataFrame(
        {"user_id": ["u1", "u_unknown"], "business_id": ["b_unknown", "b2"], "stars": [3, 2]}
    )

    user_df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u_unknown"],
            "review_count": [10, 20, 5],
            "average_stars": [4.0, 3.5, 2.0],
            "yelping_days": [1000, 500, 100],
        }
    )
    biz_df = pd.DataFrame(
        {
            "business_id": ["b1", "b2", "b_unknown"],
            "review_count": [50, 100, 10],
            "stars": [4.5, 4.0, 3.0],
        }
    )
    for i in range(10):
        biz_df[f"cat_{i}"] = 0.0

    train_inter.to_csv(tmp_path / "train_interactions.csv", index=False)
    valid_inter.to_csv(tmp_path / "valid_interactions.csv", index=False)
    user_df.to_csv(tmp_path / "user_features.csv", index=False)
    biz_df.to_csv(tmp_path / "business_features.csv", index=False)

    train_ds = YelpDataset(data_path=tmp_path, mode="train")
    valid_ds = YelpDataset(
        data_path=tmp_path,
        mode="valid",
        user2idx=train_ds.user2idx,
        business2idx=train_ds.business2idx,
    )

    # u1 should have exact same index in both train and valid
    train_sample_0 = train_ds[0]
    valid_sample_0 = valid_ds[0]

    assert train_sample_0["user_features"][0].item() == valid_sample_0["user_features"][0].item()

    # Unknown user and unknown business in valid set should be assigned fallback index 0 without raising error
    valid_sample_1 = valid_ds[1]
    assert valid_sample_1["user_features"][0].item() == 0.0
    assert valid_sample_0["business_features"][0].item() == 0.0


def test_dataset_tensor_dimensions_and_no_nans(tmp_path):
    """Verify returned tensors match expected shapes [4], [3], [10] with zero NaNs."""
    train_inter = pd.DataFrame({"user_id": ["u1"], "business_id": ["b1"], "stars": [5]})
    user_df = pd.DataFrame(
        {"user_id": ["u1"], "review_count": [10], "average_stars": [4.0], "yelping_days": [100]}
    )
    biz_df = pd.DataFrame({"business_id": ["b1"], "review_count": [50], "stars": [4.5]})
    for i in range(10):
        biz_df[f"cat_{i}"] = 1.0 if i == 2 else 0.0

    train_inter.to_csv(tmp_path / "train_interactions.csv", index=False)
    user_df.to_csv(tmp_path / "user_features.csv", index=False)
    biz_df.to_csv(tmp_path / "business_features.csv", index=False)

    ds = YelpDataset(data_path=tmp_path, mode="train")
    item = ds[0]

    uf = item["user_features"]
    bf = item["business_features"]
    cf = item["category_features"]

    assert uf.shape == (4,)
    assert bf.shape == (3,)
    assert cf.shape == (10,)

    assert not torch.isnan(uf).any()
    assert not torch.isnan(bf).any()
    assert not torch.isnan(cf).any()
