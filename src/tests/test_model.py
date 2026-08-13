import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import pytest

from src.models.two_tower import TwoTowerModel
from src.utils.config import DEFAULT_CONFIG


class TestTwoTowerModel:
    """Tests for the Two-Tower recommendation model."""

    @pytest.fixture(scope="class")
    def model(self):
        """Create model instance for testing."""
        return TwoTowerModel(DEFAULT_CONFIG["model"])

    @pytest.fixture(scope="class")
    def batch_inputs(self):
        """Create dummy batch matching the model's expected input schema."""

        batch_size = 32
        config = DEFAULT_CONFIG["model"]

        user_ids = torch.randint(
            0,
            config["num_users"],
            (batch_size,),
            dtype=torch.long,
        )

        item_ids = torch.randint(
            0,
            config["num_items"],
            (batch_size,),
            dtype=torch.long,
        )

        # UserTower expects 3 statistical features:
        # review_count, average_stars, yelping_days
        user_features = torch.randn(batch_size, 3)

        # ItemTower expects 2 business features:
        # review_count, stars
        business_features = torch.randn(batch_size, 2)

        # ItemTower expects 10-dimensional one-hot category features
        category_features = torch.zeros(batch_size, 10)

        # Give each sample one category
        category_indices = torch.randint(0, 10, (batch_size,))
        category_features[
            torch.arange(batch_size),
            category_indices
        ] = 1.0

        return {
            "user_features": torch.cat(
                [
                    user_ids.float().unsqueeze(1),
                    user_features,
                ],
                dim=1,
            ),
            "business_features": torch.cat(
                [
                    item_ids.float().unsqueeze(1),
                    business_features,
                ],
                dim=1,
            ),
            "category_features": category_features,
        }

    def test_user_tower_output_shape(self, model, batch_inputs):
        """User tower should produce [batch_size, 128]."""

        user_emb, _ = model(batch_inputs)

        expected_shape = (
            32,
            DEFAULT_CONFIG["model"]["user_tower"]["hidden_dims"][-1],
        )

        assert user_emb.shape == expected_shape

    def test_item_tower_output_shape(self, model, batch_inputs):
        """Item tower should produce [batch_size, 128]."""

        _, item_emb = model(batch_inputs)

        expected_shape = (
            32,
            DEFAULT_CONFIG["model"]["item_tower"]["hidden_dims"][-1],
        )

        assert item_emb.shape == expected_shape

    def test_embedding_dimension_match(self, model, batch_inputs):
        """User and item embeddings should have the same dimension."""

        user_emb, item_emb = model(batch_inputs)

        assert user_emb.shape[1] == item_emb.shape[1]