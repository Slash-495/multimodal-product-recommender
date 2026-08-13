import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(
    data_dir: Union[str, Path], sample_size: int = 100000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a subset of raw Yelp JSON data."""
    data_dir = Path(data_dir)

    review_file = data_dir / "yelp_academic_dataset_review.json"
    user_file = data_dir / "yelp_academic_dataset_user.json"
    business_file = data_dir / "yelp_academic_dataset_business.json"

    if not review_file.exists() or not user_file.exists() or not business_file.exists():
        raise FileNotFoundError(
            f"Required Yelp JSON files not found in '{data_dir}'. "
            "Expected: yelp_academic_dataset_review.json, "
            "yelp_academic_dataset_user.json, yelp_academic_dataset_business.json"
        )

    chunks = pd.read_json(review_file, lines=True, chunksize=10000)
    sampled_reviews = []
    total_rows = 0

    for chunk in chunks:
        if total_rows >= sample_size:
            break
        sample_size_chunk = min(len(chunk), sample_size - total_rows)
        sampled_chunk = chunk.sample(n=sample_size_chunk, random_state=42)
        sampled_reviews.append(sampled_chunk)
        total_rows += sample_size_chunk

    reviews = pd.concat(sampled_reviews)[["user_id", "business_id", "stars", "date"]]

    unique_users = set(reviews["user_id"].unique())
    unique_businesses = set(reviews["business_id"].unique())

    users_list = []
    for chunk in pd.read_json(user_file, lines=True, chunksize=10000):
        rel = chunk[chunk["user_id"].isin(unique_users)]
        if not rel.empty:
            users_list.append(rel)
    users = (
        pd.concat(users_list)[["user_id", "review_count", "yelping_since", "average_stars"]]
        if users_list
        else pd.DataFrame(columns=["user_id", "review_count", "yelping_since", "average_stars"])
    )

    businesses_list = []
    for chunk in pd.read_json(business_file, lines=True, chunksize=10000):
        rel = chunk[chunk["business_id"].isin(unique_businesses)]
        if not rel.empty:
            businesses_list.append(rel)
    businesses = (
        pd.concat(businesses_list)[["business_id", "stars", "review_count", "categories"]]
        if businesses_list
        else pd.DataFrame(columns=["business_id", "stars", "review_count", "categories"])
    )

    return reviews, users, businesses


def get_top_categories(businesses: pd.DataFrame, top_k: int = 10) -> List[str]:
    """Deterministically select top-k most frequent business categories."""
    category_counts = Counter()

    if "categories" in businesses.columns:
        for cats in businesses["categories"].fillna(""):
            if isinstance(cats, str) and cats.strip():
                cat_list = [c.strip() for c in cats.split(",") if c.strip()]
                category_counts.update(cat_list)

    # Sort by frequency descending, then alphabetically ascending for tie-breaking
    sorted_cats = sorted(category_counts.items(), key=lambda x: (-x[1], x[0]))
    top_categories = [cat for cat, _ in sorted_cats[:top_k]]
    return top_categories


def process_features(
    reviews: pd.DataFrame,
    users: pd.DataFrame,
    businesses: pd.DataFrame,
    top_k_categories: int = 10,
    reference_date: str = "2024-01-01",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process numerical and categorical features safely without NaNs."""
    reviews = reviews.copy()
    users = users.copy()
    businesses = businesses.copy()

    # Process user features
    if "yelping_since" in users.columns:
        yelp_dates = pd.to_datetime(users["yelping_since"], errors="coerce")
        ref_dt = pd.to_datetime(reference_date)
        users["yelping_days"] = (ref_dt - yelp_dates).dt.days.fillna(0).astype(float)
        users = users.drop(columns=["yelping_since"])
    else:
        users["yelping_days"] = 0.0

    users["review_count"] = users["review_count"].fillna(0.0).astype(float)
    users["average_stars"] = users["average_stars"].fillna(0.0).astype(float)

    # Process business features
    top_cats = get_top_categories(businesses, top_k=top_k_categories)

    businesses["categories"] = businesses["categories"].fillna("")
    businesses["review_count"] = businesses["review_count"].fillna(0.0).astype(float)
    businesses["stars"] = businesses["stars"].fillna(0.0).astype(float)

    # Generate top category one-hot columns (padded to top_k_categories)
    for i in range(top_k_categories):
        col_name = f"cat_{i}"
        if i < len(top_cats):
            cat_name = top_cats[i]
            businesses[col_name] = (
                businesses["categories"]
                .apply(lambda c: 1.0 if isinstance(c, str) and cat_name in [x.strip() for x in c.split(",")] else 0.0)
            )
        else:
            businesses[col_name] = 0.0

    if "categories" in businesses.columns:
        businesses = businesses.drop(columns=["categories"])

    # Ensure no NaNs remain
    reviews = reviews.fillna(0.0)
    users = users.fillna(0.0)
    businesses = businesses.fillna(0.0)

    return reviews, users, businesses


def split_data(
    reviews: pd.DataFrame,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    time_based: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions dataset into train, validation, and test sets."""
    reviews = reviews.copy()

    if time_based and "date" in reviews.columns:
        reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
        reviews = reviews.sort_values("date").reset_index(drop=True)

        n = len(reviews)
        train_idx = int(n * train_ratio)
        valid_idx = int(n * (train_ratio + valid_ratio))

        train = reviews.iloc[:train_idx]
        valid = reviews.iloc[train_idx:valid_idx]
        test = reviews.iloc[valid_idx:]
    else:
        train, temp = train_test_split(reviews, train_size=train_ratio, random_state=42)
        valid, test = train_test_split(
            temp, train_size=valid_ratio / (1 - train_ratio), random_state=42
        )

    return train, valid, test


def main(data_dir: str = "data/raw", output_dir: str = "data/processed", sample_size: int = 100000):
    """Main preprocessing pipeline execution."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw data from '{data_path}'...")
    reviews, users, businesses = load_raw_data(data_path, sample_size=sample_size)

    print("Processing features...")
    reviews, users, businesses = process_features(reviews, users, businesses)

    print("Splitting datasets...")
    train, valid, test = split_data(reviews, time_based=True)

    # Save processed CSV files
    train.to_csv(out_path / "train_interactions.csv", index=False)
    valid.to_csv(out_path / "valid_interactions.csv", index=False)
    test.to_csv(out_path / "test_interactions.csv", index=False)
    users.to_csv(out_path / "user_features.csv", index=False)
    businesses.to_csv(out_path / "business_features.csv", index=False)

    print(f"Preprocessing completed! Saved to '{out_path}'.")
    print(f"Train samples: {len(train)}, Valid samples: {len(valid)}, Test samples: {len(test)}")


if __name__ == "__main__":
    main()