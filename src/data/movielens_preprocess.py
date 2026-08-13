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
    zip_path: Union[str, Path],
    max_interactions: int = 500000,
    min_user_interactions: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Stream ratings directly from MovieLens zip archive.
    Filters users with >= min_user_interactions and samples up to max_interactions reproducibly.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"MovieLens zip file not found at: '{zip_path}'")

    with zipfile.ZipFile(zip_path, "r") as zf:
        ratings_member = find_zip_member(zf, "ratings.csv")
        movies_member = find_zip_member(zf, "movies.csv")

        # Pass 1: Count total ratings and per-user interaction frequency
        total_raw_ratings = 0
        user_counts = Counter()
        with zf.open(ratings_member, "r") as f:
            for chunk in pd.read_csv(f, usecols=["userId"], chunksize=200000):
                total_raw_ratings += len(chunk)
                user_counts.update(chunk["userId"].tolist())

        total_users = len(user_counts)
        eligible_users = {u for u, c in user_counts.items() if c >= min_user_interactions}

        # Pass 2: Stream and extract ratings belonging to eligible users
        eligible_chunks = []
        with zf.open(ratings_member, "r") as f:
            for chunk in pd.read_csv(f, chunksize=200000):
                rel = chunk[chunk["userId"].isin(eligible_users)]
                if not rel.empty:
                    eligible_chunks.append(rel)

        eligible_ratings = pd.concat(eligible_chunks, ignore_index=True)
        total_eligible_ratings = len(eligible_ratings)

        # Pass 3: Draw a reproducible random sample if total eligible ratings > max_interactions
        rng = np.random.default_rng(seed)
        if total_eligible_ratings <= max_interactions:
            sampled_ratings = eligible_ratings
        else:
            sample_idx = rng.choice(total_eligible_ratings, size=max_interactions, replace=False)
            sampled_ratings = eligible_ratings.iloc[sample_idx].copy()

        # Load movie metadata
        with zf.open(movies_member, "r") as f:
            movies_all = pd.read_csv(f)

        unique_movies = set(sampled_ratings["movieId"].unique())
        movies = movies_all[movies_all["movieId"].isin(unique_movies)].copy()

    return sampled_ratings, movies, total_raw_ratings, total_users


def get_top_genres(movies: pd.DataFrame, top_k: int = 10) -> List[str]:
    """Deterministically select top-k most frequent genres across movies."""
    genre_counts = Counter()

    for genres in movies["genres"].fillna(""):
        if isinstance(genres, str) and genres.strip() and genres != "(no genres listed)":
            g_list = [g.strip() for g in genres.split("|") if g.strip()]
            genre_counts.update(g_list)

    sorted_genres = sorted(genre_counts.items(), key=lambda x: (-x[1], x[0]))
    top_genres = [g for g, _ in sorted_genres[:top_k]]
    return top_genres


def split_user_interactions_chronological(
    user_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a single user's interaction sequence chronologically into train (80%), valid (10%), test (10%).
    Ensures train >= 1, valid >= 1, test >= 1.
    """
    u_sorted = user_df.sort_values("timestamp").reset_index(drop=True)
    n = len(u_sorted)

    if n < 3:
        # Cannot split into 1 train, 1 valid, 1 test
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    test_cnt = max(1, int(round(n * 0.10)))
    valid_cnt = max(1, int(round(n * 0.10)))
    train_cnt = n - valid_cnt - test_cnt

    if train_cnt < 1:
        train_cnt = 1
        valid_cnt = 1
        test_cnt = n - 2

    u_train = u_sorted.iloc[:train_cnt]
    u_valid = u_sorted.iloc[train_cnt : train_cnt + valid_cnt]
    u_test = u_sorted.iloc[train_cnt + valid_cnt :]

    return u_train, u_valid, u_test


def process_movielens_data_warmstart(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    min_user_interactions: int = 5,
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
    Dict[str, int],
]:
    """
    Process MovieLens ratings using per-user chronological splitting for warm-start benchmark.
    Training user and movie features are derived strictly from train interactions.
    """
    train_chunks = []
    valid_chunks = []
    test_chunks = []

    warm_start_user_ids = []
    excluded_user_ids = []

    grouped = ratings.groupby("userId")
    for uid, u_df in grouped:
        if len(u_df) < 3:
            excluded_user_ids.append(uid)
            continue

        u_train, u_valid, u_test = split_user_interactions_chronological(u_df)
        if u_train.empty or u_valid.empty or u_test.empty:
            excluded_user_ids.append(uid)
        else:
            train_chunks.append(u_train)
            valid_chunks.append(u_valid)
            test_chunks.append(u_test)
            warm_start_user_ids.append(uid)

    if not train_chunks:
        raise ValueError("No eligible users with >= 3 ratings in sampled dataset.")

    train_df = pd.concat(train_chunks, ignore_index=True)
    valid_df = pd.concat(valid_chunks, ignore_index=True)
    test_df = pd.concat(test_chunks, ignore_index=True)

    # Verify per-user temporal non-leakage
    for uid in warm_start_user_ids[:100]:
        ut = train_df[train_df["userId"] == uid]["timestamp"]
        uv = valid_df[valid_df["userId"] == uid]["timestamp"]
        ue = test_df[test_df["userId"] == uid]["timestamp"]
        assert ut.max() <= uv.min(), f"Leakage detected: train_max {ut.max()} > valid_min {uv.min()} for user {uid}"
        assert uv.max() <= ue.min(), f"Leakage detected: valid_max {uv.max()} > test_min {ue.min()} for user {uid}"

    # Build ID mappings from TRAIN users and ALL candidate movies
    unique_users = sorted(train_df["userId"].unique())
    unique_movies = sorted(ratings["movieId"].unique())

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

    # Collect all warm-start users
    all_users = pd.DataFrame({"userId": warm_start_user_ids})
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

    all_movies = pd.DataFrame({"movieId": list(set(ratings["movieId"].unique()))})
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

    # Format interaction outputs
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

    stats_summary = {
        "warm_start_users": len(warm_start_user_ids),
        "excluded_users": len(excluded_user_ids),
        "total_movies": len(all_movies),
        "train_interactions": len(train_out),
        "valid_interactions": len(valid_out),
        "test_interactions": len(test_out),
    }

    return (
        train_out,
        valid_out,
        test_out,
        user_features_out,
        movie_features_out,
        movies,
        user2idx,
        movie2idx,
        stats_summary,
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess MovieLens dataset into Warm-Start Two-Tower benchmark.")
    parser.add_argument("--input-zip", type=str, default="data/raw/ml-latest.zip", help="Path to raw MovieLens ml-latest.zip")
    parser.add_argument("--output-dir", type=str, default="data/processed/movielens", help="Directory to save processed dataset")
    parser.add_argument("--max-interactions", type=int, default=500000, help="Maximum interaction ratings to sample")
    parser.add_argument("--min-user-interactions", type=int, default=5, help="Minimum ratings per user threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming ratings from '{args.input_zip}' (filtering users with >={args.min_user_interactions} ratings)...")
    ratings, movies, raw_ratings_cnt, raw_users_cnt = load_raw_movielens_stream(
        args.input_zip,
        max_interactions=args.max_interactions,
        min_user_interactions=args.min_user_interactions,
        seed=args.seed,
    )

    print("Building per-user chronological split with strict temporal non-leakage...")
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
    ) = process_movielens_data_warmstart(
        ratings=ratings,
        movies=movies,
        min_user_interactions=args.min_user_interactions,
        top_k_genres=10,
    )

    # Save processed files
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

    test_users_count = len(test_out["user_id"].unique())
    test_users_in_train = len(set(test_out["user_id"].unique()).intersection(set(train_out["user_id"].unique())))

    warm_pct = (test_users_in_train / test_users_count * 100.0) if test_users_count > 0 else 0.0
    cold_pct = 100.0 - warm_pct

    print("\n================ DATASET QUALITY REPORT ================")
    print(f"Raw ratings considered: {raw_ratings_cnt:,}")
    print(f"Total raw users: {raw_users_cnt:,}")
    print(f"Sampled/retained interactions: {len(ratings):,}")
    print(f"Warm-start users: {stats['warm_start_users']:,}")
    print(f"Excluded users: {stats['excluded_users']:,}")
    print(f"Total unique movies: {stats['total_movies']:,}")
    print(f"Train interactions: {stats['train_interactions']:,}")
    print(f"Validation interactions: {stats['valid_interactions']:,}")
    print(f"Test interactions: {stats['test_interactions']:,}")
    print(f"Warm-start test users: {test_users_in_train:,} ({warm_pct:.2f}%)")
    print(f"Cold-start test users: {test_users_count - test_users_in_train:,} ({cold_pct:.2f}%)")
    print("========================================================\n")


if __name__ == "__main__":
    main()
