import io
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
    process_movielens_data,
)
from src.data.movielens_dataset import MovieLensDataset, get_movielens_dataloader


def test_movielens_stream_sampling_from_zip(tmp_path):
    """Verify streaming and sampling directly from a synthetic zip file without extracting to disk."""
    zip_path = tmp_path / "test_ml.zip"

    ratings_df = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3],
            "movieId": [10, 20, 10, 30, 20],
            "rating": [4.0, 5.0, 3.5, 2.0, 4.5],
            "timestamp": [1000, 2000, 3000, 4000, 5000],
        }
    )
    movies_df = pd.DataFrame(
        {
            "movieId": [10, 20, 30],
            "title": ["Movie A", "Movie B", "Movie C"],
            "genres": ["Action|Sci-Fi", "Action|Drama", "Comedy"],
        }
    )

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("ratings.csv", ratings_df.to_csv(index=False))
        zf.writestr("movies.csv", movies_df.to_csv(index=False))

    ratings_out, movies_out = load_raw_movielens_stream(zip_path, max_interactions=3, seed=42)

    assert len(ratings_out) == 3
    assert len(movies_out) <= 3
    assert "userId" in ratings_out.columns
    assert "movieId" in movies_out.columns


def test_movielens_deterministic_genre_selection():
    """Verify top 10 genres are selected deterministically by frequency and alphabetical tie-breaking."""
    movies_df = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "genres": ["Action|Adventure", "Action|Comedy", "Comedy|Drama"],
        }
    )

    genres_1 = get_top_genres(movies_df, top_k=10)
    genres_2 = get_top_genres(movies_df, top_k=10)

    assert genres_1 == genres_2
    # Action (2), Comedy (2), Adventure (1), Drama (1) -> Action & Comedy tied at 2 ('Action' < 'Comedy')
    assert genres_1[0] == "Action"
    assert genres_1[1] == "Comedy"


def test_movielens_feature_extraction_and_leakage_prevention(tmp_path):
    """
    Verify user/movie features and ID mappings are derived from training split only,
    and exactly 10 genre features are generated without NaNs.
    """
    ratings_df = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "movieId": [10, 20, 10, 30, 20, 30, 10, 20, 30, 40],
            "rating": [5.0, 4.0, 3.0, 2.0, 4.5, 3.5, 4.0, 5.0, 2.5, 3.0],
            "timestamp": list(range(1000, 1010)),
        }
    )
    movies_df = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "title": ["M1", "M2", "M3", "M4"],
            "genres": ["Action|Sci-Fi", "Action|Drama", "Comedy", "Drama|Thriller"],
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
    ) = process_movielens_data(ratings_df, movies_df, train_ratio=0.8, valid_ratio=0.1, top_k_genres=10)

    # 1. Check exactly 10 category columns
    cat_cols = [c for c in movie_features_out.columns if c.startswith("cat_")]
    assert len(cat_cols) == 10

    # 2. Check zero NaNs
    assert not train_out.isna().any().any()
    assert not valid_out.isna().any().any()
    assert not test_out.isna().any().any()
    assert not user_features_out.isna().any().any()
    assert not movie_features_out.isna().any().any()

    # 3. Save CSVs to tmp_path and verify MovieLensDataset
    train_out.to_csv(tmp_path / "train_interactions.csv", index=False)
    valid_out.to_csv(tmp_path / "valid_interactions.csv", index=False)
    test_out.to_csv(tmp_path / "test_interactions.csv", index=False)
    user_features_out.to_csv(tmp_path / "user_features.csv", index=False)
    movie_features_out.to_csv(tmp_path / "business_features.csv", index=False)

    with open(tmp_path / "user2idx.json", "w") as f:
        json.dump(user2idx, f)
    with open(tmp_path / "business2idx.json", "w") as f:
        json.dump(movie2idx, f)

    ds = MovieLensDataset(data_path=tmp_path, mode="train")
    sample = ds[0]

    uf = sample["user_features"]
    bf = sample["business_features"]
    cf = sample["category_features"]

    # 4. Check tensor dimensions
    assert uf.shape == (4,)
    assert bf.shape == (3,)
    assert cf.shape == (10,)

    assert not torch.isnan(uf).any()
    assert not torch.isnan(bf).any()
    assert not torch.isnan(cf).any()


def test_movielens_unknown_entity_handling(tmp_path):
    """Verify unknown entities appearing only in validation/test dataset map safely to fallback index 0."""
    train_inter = pd.DataFrame({"user_id": ["1", "2"], "business_id": ["10", "20"], "stars": [5.0, 4.0]})
    valid_inter = pd.DataFrame({"user_id": ["1", "999_unknown"], "business_id": ["888_unknown", "20"], "stars": [3.0, 2.0]})

    user_df = pd.DataFrame(
        {
            "user_id": ["1", "2", "999_unknown"],
            "review_count": [10, 20, 5],
            "average_stars": [4.0, 3.5, 2.0],
            "yelping_days": [100, 50, 10],
        }
    )
    biz_df = pd.DataFrame(
        {
            "business_id": ["10", "20", "888_unknown"],
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

    train_ds = MovieLensDataset(data_path=tmp_path, mode="train")
    valid_ds = MovieLensDataset(
        data_path=tmp_path,
        mode="valid",
        user2idx=train_ds.user2idx,
        business2idx=train_ds.business2idx,
    )

    valid_sample_0 = valid_ds[0]  # user '1', biz '888_unknown'
    valid_sample_1 = valid_ds[1]  # user '999_unknown', biz '20'

    # Known user '1' index matches train_ds
    assert valid_sample_0["user_features"][0].item() == train_ds[0]["user_features"][0].item()

    # Unknown biz '888_unknown' and unknown user '999_unknown' map to fallback 0.0
    assert valid_sample_0["business_features"][0].item() == 0.0
    assert valid_sample_1["user_features"][0].item() == 0.0
