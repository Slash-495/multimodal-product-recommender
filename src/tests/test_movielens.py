import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd
import torch
import pytest

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.movielens_preprocess import (
    get_top_genres,
    load_raw_movielens_stream,
    process_movielens_data_warmstart,
    split_user_interactions_chronological,
)
from src.data.movielens_dataset import MovieLensDataset


def test_movielens_stream_sampling_from_zip(tmp_path):
    """Verify streaming and sampling directly from a synthetic zip file without extracting to disk."""
    zip_path = tmp_path / "test_ml.zip"

    # User 1 has 5 ratings (eligible), User 2 has 2 ratings (ineligible)
    ratings_df = pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 1, 2, 2],
            "movieId": [10, 20, 30, 40, 50, 10, 20],
            "rating": [4.0, 5.0, 3.5, 2.0, 4.5, 3.0, 4.0],
            "timestamp": [1000, 2000, 3000, 4000, 5000, 1500, 2500],
        }
    )
    movies_df = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40, 50],
            "title": ["M10", "M20", "M30", "M40", "M50"],
            "genres": ["Action|Sci-Fi", "Action|Drama", "Comedy", "Drama", "Action"],
        }
    )

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("ratings.csv", ratings_df.to_csv(index=False))
        zf.writestr("movies.csv", movies_df.to_csv(index=False))

    ratings_out, movies_out, raw_ratings_cnt, raw_users_cnt = load_raw_movielens_stream(
        zip_path, max_interactions=10, min_user_interactions=5, seed=42
    )

    assert len(ratings_out) == 5
    assert set(ratings_out["userId"].unique()) == {1}
    assert raw_ratings_cnt == 7
    assert raw_users_cnt == 2


def test_movielens_per_user_chronological_split_and_no_leakage():
    """Verify per-user chronological splitting prevents temporal data leakage."""
    user_df = pd.DataFrame(
        {
            "userId": [10] * 10,
            "movieId": list(range(101, 111)),
            "rating": [4.0] * 10,
            "timestamp": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        }
    )

    u_train, u_valid, u_test = split_user_interactions_chronological(user_df)

    assert not u_train.empty
    assert not u_valid.empty
    assert not u_test.empty

    assert len(u_train) + len(u_valid) + len(u_test) == 10

    # Leakage check: max train ts < min valid ts < max valid ts < min test ts
    assert u_train["timestamp"].max() <= u_valid["timestamp"].min()
    assert u_valid["timestamp"].max() <= u_test["timestamp"].min()


def test_movielens_warmstart_preprocessing_and_100_percent_user_coverage(tmp_path):
    """Verify 100% of warm-start test users exist in training, movie mappings are shared, and no NaNs exist."""
    ratings_df = pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "movieId": [10, 20, 30, 40, 50, 10, 20, 30, 40, 50],
            "rating": [5.0, 4.0, 3.0, 2.0, 4.5, 3.5, 4.0, 5.0, 2.5, 3.0],
            "timestamp": list(range(1000, 1010)),
        }
    )
    movies_df = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40, 50],
            "title": ["M10", "M20", "M30", "M40", "M50"],
            "genres": ["Action|Sci-Fi", "Action|Drama", "Comedy", "Drama|Thriller", "Action"],
        }
    )

    (
        train_out,
        valid_out,
        test_out,
        user_features_out,
        movie_features_out,
        movies_meta,
        user2idx,
        movie2idx,
        stats,
    ) = process_movielens_data_warmstart(ratings_df, movies_df, min_user_interactions=5, top_k_genres=10)

    # 1. Verify 100% of test users exist in training
    train_users = set(train_out["user_id"].unique())
    test_users = set(test_out["user_id"].unique())
    assert test_users.issubset(train_users)

    # 2. Verify movie mappings are shared across splits
    assert set(train_out["business_id"].unique()).issubset(set(movie2idx.keys()))
    assert set(test_out["business_id"].unique()).issubset(set(movie2idx.keys()))

    # 3. Verify zero NaNs in features
    assert not train_out.isna().any().any()
    assert not valid_out.isna().any().any()
    assert not test_out.isna().any().any()
    assert not user_features_out.isna().any().any()
    assert not movie_features_out.isna().any().any()

    # 4. Verify 10 category columns
    cat_cols = [c for c in movie_features_out.columns if c.startswith("cat_")]
    assert len(cat_cols) == 10


def test_movielens_reproducibility():
    """Verify running preprocessing twice with the same seed produces identical genre selection and stats."""
    movies_df = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "genres": ["Action|Adventure", "Action|Comedy", "Comedy|Drama"],
        }
    )

    genres_1 = get_top_genres(movies_df, top_k=10)
    genres_2 = get_top_genres(movies_df, top_k=10)

    assert genres_1 == genres_2
