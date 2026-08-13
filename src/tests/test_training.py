import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import pytest

from src.models.two_tower import TwoTowerModel
from src.utils.losses import InfoNCELoss
from src.utils.config import DEFAULT_CONFIG


def test_training_pipeline_smoke():
    """Smoke test for model forward, InfoNCE loss, and backward pass."""
    model = TwoTowerModel(DEFAULT_CONFIG["model"])
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

    user_features = torch.randn(batch_size, 3)
    business_features = torch.randn(batch_size, 2)
    category_features = torch.zeros(batch_size, 10)
    category_indices = torch.randint(0, 10, (batch_size,))
    category_features[torch.arange(batch_size), category_indices] = 1.0

    batch = {
        "user_features": torch.cat(
            [user_ids.float().unsqueeze(1), user_features], dim=1
        ),
        "business_features": torch.cat(
            [item_ids.float().unsqueeze(1), business_features], dim=1
        ),
        "category_features": category_features,
    }

    # Forward pass
    user_emb, item_emb = model(batch)

    # Compute loss
    criterion = InfoNCELoss(temperature=DEFAULT_CONFIG["training"]["temperature"])
    loss = criterion(user_emb, item_emb)

    # Verify loss is finite
    assert torch.isfinite(loss), "InfoNCE loss returned infinite or NaN value"

    # Backward pass
    loss.backward()

    # Verify gradients were computed for model parameters
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    assert has_grad, "No gradients computed after backward pass"
