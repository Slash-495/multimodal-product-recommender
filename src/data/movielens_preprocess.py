import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union
import zipfile
import numpy as np
import pandas as pd


def find_zip_member(zf: zipfile.ZipFile, filename: str) -> str:
    """Find a target file inside a zip archive matching filename suffix."""
    for name in zf.namelist():
        if name.endswith(filename):
            return name
    raise FileNotFoundError(f"File '{filename}' not found inside zip archive.")


def load_raw_movielens_stream(
    zip_path: Union[str, Path], max_interactions: int = 500000, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stream and sample ratings and movie metadata directly from MovieLens zip archive.
    Does not extract the full dataset to disk.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"MovieLens zip file not found at: '{zip_path}'")

    with zipfile.ZipFile(zip_path, "r") as zf:
        ratings_member = find_zip_member(zf, "ratings.csv")
        movies_member = find_zip_member(zf, "movies.csv")

        # Pass 1: Count total rows in ratings.csv
        total_rows = 0
        with zf.open(ratings_member, "r") as f:
            for chunk in pd.read_csv(f, usecols=["userId"], chunksize=200000):
                total_rows += len(chunk)

        print(f"Total ratings in raw dataset: {total_rows:,}")

        # Pass 2: Select row indices to keep
        rng = np.random.default_rng(seed)
        if total_rows <= max_interactions:
            selected_indices = set(range(total_rows))
            print(f"Selecting all {total_rows:,} ratings.")
        else:
            selected_indices = set(rng.choice(total_rows, size=max_interactions, replace=False))
            print(f"Sampling {max_interactions:,} ratings uniformly from {total_rows:,} ratings.")

        # Pass 3: Stream and filter selected rows
        sampled_chunks = []
        current_idx = 0
        with zf.open(ratings_member, "r") as f:
            for chunk in pd.read_csv(f, chunksize=200000):
                chunk_len = len(chunk)
                # Find overlap with selected indices
                chunk_indices = set(range(current_idx, current_idx + chunk_len)).intersection(selected_indices)
                if chunk_indices:
                    rel_indices = [idx - current_idx for idx in chunk_indices]
                    sampled_chunks.append(chunk.iloc[rel_indices])
                current_idx += chunk_len

        ratings = pd.concat(sampled_chunks, ignore_index=True)

        # Load movie metadata
        with zf.open(movies_member, "r") as f:
            movies_all = pd.read_csv(f)

        unique_movies = set(ratings["movieId"].unique())
        movies = movies_all[movies_all["movieId"].isin(unique_movies)].copy()

    return ratings, movies


def get_top_genres(movies: pd.DataFrame, top_k: int = 10) -> List[str]:
    """Deterministically select top-k most frequent genres across movies."""
    genre_counts = Counter()

    for genres in movies["genres"].fillna(""):
        if isinstance(genres, str) and genres.strip() and genres != "(no genres listed)":
            g_list = [g.strip() for g in genres.split("|") if g.strip()]
            genre_counts.update(g_list)

    # Sort by frequency descending, then alphabetically for tie-breaking
    sorted_genres = sorted(genre_counts.items(), key=lambda x: (-x[1], x[0]))
    top_genres = [g for g, _ in sorted_genres[:top_k]]
    return top_genres


def process_movielens_data(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    top_k_genres: int = 10,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, int],
    Dict[str, int],
]:
    """
    Process MovieLens ratings and movie metadata using strict leakage prevention.
    Training user and movie features are derived strictly from train interactions.
    """
    # Sort interactions by timestamp for time-based split
    ratings_sorted = ratings.sort_values("timestamp").reset_index(drop=True)

    n = len(ratings_sorted)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train_df = ratings_sorted.iloc[:train_end].copy()
    valid_df = ratings_sorted.iloc[train_end:valid_end].copy()
    test_df = ratings_sorted.iloc[valid_end:].copy()

    # Build ID mappings from TRAIN dataset ONLY
    unique_users = sorted(train_df["userId"].unique())
    unique_movies = sorted(train_df["movieId"].unique())

    user2idx = {str(uid): i for i, uid in enumerate(unique_users)}
    movie2idx = {str(mid): i for i, mid in enumerate(unique_movies)}

    # Calculate user features from TRAIN set ONLY
    user_stats = (
        train_df.groupby("userId")
        .agg(
            review_count=("rating", "count"),
            average_stars=("rating", "mean"),
            min_ts=("timestamp", "min"),
            max_ts=("timestamp", "max"),
        )
        .reset_index()
    )
    user_stats["yelping_days"] = (user_stats["max_ts"] - user_stats["min_ts"]) / 86400.0
    user_stats["yelping_days"] = user_stats["yelping_days"].fillna(0.0)

    # Collect all unique users across all splits
    all_users = pd.DataFrame({"userId": list(set(ratings_sorted["userId"].unique()))})
    user_features = all_users.merge(user_stats[["userId", "review_count", "average_stars", "yelping_days"]], on="userId", how="left")
    user_features["user_id"] = user_features["userId"].astype(str)
    user_features["review_count"] = user_features["review_count"].fillna(0.0)
    user_features["average_stars"] = user_features["average_stars"].fillna(0.0)
    user_features["yelping_days"] = user_features["yelping_days"].fillna(0.0)

    # Calculate movie features from TRAIN set ONLY
    movie_stats = (
        train_df.groupby("movieId")
        .agg(
            review_count=("rating", "count"),
            stars=("rating", "mean"),
        )
        .reset_index()
    )

    # Collect all unique movies across splits
    all_movies = pd.DataFrame({"movieId": list(set(ratings_sorted["movieId"].unique()))})
    movie_features = all_movies.merge(movie_stats[["movieId", "review_count", "stars"]], on="movieId", how="left")
    movie_features["business_id"] = movie_features["movieId"].astype(str)
    movie_features["review_count"] = movie_features["review_count"].fillna(0.0)
    movie_features["stars"] = movie_features["stars"].fillna(0.0)

    # Add genre features to movies
    top_g = get_top_genres(movies, top_k=top_k_genres)
    movies_genre_map = movies.set_index("movieId")["genres"].to_dict()

    for i in range(top_k_genres):
        col_name = f"cat_{i}"
        if i < len(top_g):
            g_target = top_g[i]
            movie_features[col_name] = movie_features["movieId"].apply(
                lambda mid: 1.0
                if isinstance(movies_genre_map.get(mid, ""), str)
                and g_target in [x.strip() for x in movies_genre_map.get(mid, "").split("|")]
                else 0.0
            )
        else:
            movie_features[col_name] = 0.0

    # Format interaction dataframes
    for df in [train_df, valid_df, test_df]:
        df["user_id"] = df["userId"].astype(str)
        df["business_id"] = df["movieId"].astype(str)
        df["stars"] = df["rating"].astype(float)

    cols_to_keep = ["user_id", "business_id", "stars", "timestamp"]
    train_out = train_df[cols_to_keep]
    valid_out = valid_df[cols_to_keep]
    test_out = test_df[cols_to_keep]

    user_features_out = user_features[["user_id", "review_count", "average_stars", "yelping_days"]]
    movie_features_out = movie_features[["business_id", "review_count", "stars"] + [f"cat_{i}" for i in range(top_k_genres)]]

    return train_out, valid_out, test_out, user_features_out, movie_features_out, movies, user2idx, movie2idx


def main():
    parser = argparse.ArgumentParser(description="Preprocess MovieLens dataset into Two-Tower contract.")
    parser.add_argument("--input-zip", type=str, default="data/raw/ml-latest.zip", help="Path to raw MovieLens ml-latest.zip")
    parser.add_argument("--output-dir", type=str, default="data/processed/movielens", help="Directory to save processed datasets")
    parser.add_argument("--max-interactions", type=int, default=500000, help="Maximum number of interaction ratings to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming and sampling up to {args.max_interactions:,} interactions from '{args.input_zip}'...")
    ratings, movies = load_raw_movielens_stream(args.input_zip, max_interactions=args.max_interactions, seed=args.seed)

    print("Processing features and building time-based split with leakage prevention...")
    (
        train_out,
        valid_out,
        test_out,
        user_features_out,
        movie_features_out,
        movies_meta,
        user2idx,
        movie2idx,
    ) = process_movielens_data(ratings, movies)

    # Save output CSV files and ID mappings
    train_out.to_csv(out_dir / "train_interactions.csv", index=False)
    valid_out.to_csv(out_dir / "valid_interactions.csv", index=False)
    test_out.to_csv(out_dir / "test_interactions.csv", index=False)
    user_features_out.to_csv(out_dir / "user_features.csv", index=False)
    movie_features_out.to_csv(out_dir / "business_features.csv", index=False)
    movies_meta.to_csv(out_dir / "movie_metadata.csv", index=False)

    with open(out_dir / "user2idx.json", "w") as f:
        json.dump(user2idx, f)

    with open(out_dir / "business2idx.json", "w") as f:
        json.dump(movie2idx, f)

    print(f"MovieLens preprocessing completed! Output saved to '{out_dir}'.")
    print(f"Train samples: {len(train_out):,}, Valid samples: {len(valid_out):,}, Test samples: {len(test_out):,}")


if __name__ == "__main__":
    main()
