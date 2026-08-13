import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class MovieLensDataset(Dataset):
    """
    MovieLens dataset adapter adhering to the domain-agnostic TwoTowerModel interface.
    
    Returns:
        "user_features": [user_idx, rating_count, average_rating, activity_days] (shape [4])
        "business_features": [movie_idx, rating_count, average_rating] (shape [3])
        "category_features": [cat_0 .. cat_9] (shape [10])
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        mode: str = "train",
        user2idx: Optional[Dict[str, int]] = None,
        business2idx: Optional[Dict[str, int]] = None,
    ):
        self.data_path = Path(data_path)
        self.mode = mode
        self.user2idx = user2idx
        self.business2idx = business2idx

        self.data = self._load_data()
        self._setup_id_mappings()
        self._process_data()

    def _load_data(self) -> pd.DataFrame:
        """Load interaction dataframe and merge user and movie features."""
        inter_file = self.data_path / f"{self.mode}_interactions.csv"
        user_file = self.data_path / "user_features.csv"
        biz_file = self.data_path / "business_features.csv"

        if not inter_file.exists():
            raise FileNotFoundError(f"Interactions file not found: '{inter_file}'")
        if not user_file.exists():
            raise FileNotFoundError(f"User features file not found: '{user_file}'")
        if not biz_file.exists():
            raise FileNotFoundError(f"Movie/Business features file not found: '{biz_file}'")

        df = pd.read_csv(inter_file)
        users_df = pd.read_csv(user_file)
        biz_df = pd.read_csv(biz_file)

        df["user_id"] = df["user_id"].astype(str)
        df["business_id"] = df["business_id"].astype(str)
        users_df["user_id"] = users_df["user_id"].astype(str)
        biz_df["business_id"] = biz_df["business_id"].astype(str)

        df = df.merge(users_df, on="user_id", how="left")
        df = df.merge(biz_df, on="business_id", how="left")

        df = df.fillna(0.0)
        return df

    def _setup_id_mappings(self) -> None:
        """Load or build ID mappings for user and movie entities."""
        user_map_path = self.data_path / "user2idx.json"
        biz_map_path = self.data_path / "business2idx.json"

        if self.user2idx is None or self.business2idx is None:
            if user_map_path.exists() and biz_map_path.exists():
                with open(user_map_path, "r") as f:
                    self.user2idx = json.load(f)
                with open(biz_map_path, "r") as f:
                    self.business2idx = json.load(f)
            else:
                train_inter_file = self.data_path / "train_interactions.csv"
                if train_inter_file.exists():
                    train_df = pd.read_csv(train_inter_file)
                    unique_users = sorted(train_df["user_id"].astype(str).unique())
                    unique_biz = sorted(train_df["business_id"].astype(str).unique())
                else:
                    unique_users = sorted(self.data["user_id"].astype(str).unique())
                    unique_biz = sorted(self.data["business_id"].astype(str).unique())

                self.user2idx = {uid: i for i, uid in enumerate(unique_users)}
                self.business2idx = {bid: i for i, bid in enumerate(unique_biz)}

                if self.data_path.exists():
                    with open(user_map_path, "w") as f:
                        json.dump(self.user2idx, f)
                    with open(biz_map_path, "w") as f:
                        json.dump(self.business2idx, f)

    def _process_data(self) -> None:
        """Map user and movie string IDs to consistent integer indices."""
        self.data["user_idx"] = (
            self.data["user_id"].astype(str).map(self.user2idx).fillna(0).astype(int)
        )
        self.data["business_idx"] = (
            self.data["business_id"].astype(str).map(self.business2idx).fillna(0).astype(int)
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]

        cat_cols = [col for col in self.data.columns if col.startswith("cat_")]
        cat_cols = sorted(cat_cols)

        cat_vals = [float(row[col]) for col in cat_cols[:10]]
        while len(cat_vals) < 10:
            cat_vals.append(0.0)

        user_features = torch.tensor(
            [
                float(row["user_idx"]),
                float(row.get("review_count_x", row.get("review_count", 0.0))),
                float(row.get("average_stars", 0.0)),
                float(row.get("yelping_days", 0.0)),
            ],
            dtype=torch.float32,
        )

        business_features = torch.tensor(
            [
                float(row["business_idx"]),
                float(row.get("review_count_y", row.get("review_count", 0.0))),
                float(row.get("stars_y", row.get("stars", 0.0))),
            ],
            dtype=torch.float32,
        )

        category_features = torch.tensor(cat_vals, dtype=torch.float32)

        return {
            "user_features": user_features,
            "business_features": business_features,
            "category_features": category_features,
        }


def get_movielens_dataloader(
    data_path: Union[str, Path],
    batch_size: int = 128,
    mode: str = "train",
    num_workers: int = 0,
    user2idx: Optional[Dict[str, int]] = None,
    business2idx: Optional[Dict[str, int]] = None,
) -> DataLoader:
    """Create DataLoader for MovieLens dataset."""
    dataset = MovieLensDataset(
        data_path=data_path,
        mode=mode,
        user2idx=user2idx,
        business2idx=business2idx,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
