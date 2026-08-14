import copy
import sys
from pathlib import Path
import pytest
import torch

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.towers.item_tower import ItemTower
from src.models.two_tower import TwoTowerModel
from src.utils.config import DEFAULT_CONFIG


def test_gated_item_tower_output_shape_and_gate_range():
    """Verify ItemTower with gated fusion outputs [B, 128] and gate is bounded in [0, 1]."""
    tower = ItemTower(
        num_items=100,
        embedding_dim=64,
        hidden_dims=[256, 128],
        dropout=0.1,
        use_text_features=True,
        text_embedding_dim=64,
        fusion_type="gated",
    )

    batch_size = 16
    item_ids = torch.randint(0, 100, (batch_size,), dtype=torch.long)
    b_feats = torch.randn((batch_size, 2), dtype=torch.float32)
    c_feats = torch.randn((batch_size, 10), dtype=torch.float32)
    t_feats = torch.randn((batch_size, 64), dtype=torch.float32)

    out = tower(item_ids, b_feats, c_feats, text_features=t_feats)
    assert out.shape == (batch_size, 128)

    gate = tower.get_last_gate()
    assert gate is not None
    assert gate.shape == (batch_size, 128)
    assert (gate >= 0.0).all() and (gate <= 1.0).all()


def test_gated_fusion_gradient_propagation():
    """Verify gradients propagate end-to-end through base, text, and gate layers."""
    tower = ItemTower(
        num_items=50,
        embedding_dim=64,
        hidden_dims=[256, 128],
        dropout=0.1,
        use_text_features=True,
        text_embedding_dim=64,
        fusion_type="gated",
    )

    item_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    b_feats = torch.randn((3, 2), dtype=torch.float32)
    c_feats = torch.randn((3, 10), dtype=torch.float32)
    t_feats = torch.randn((3, 64), dtype=torch.float32)

    out = tower(item_ids, b_feats, c_feats, text_features=t_feats)
    loss = out.sum()
    loss.backward()

    assert tower.item_embedding.weight.grad is not None
    assert tower.text_layer.weight.grad is not None
    assert tower.gate_layer[0].weight.grad is not None


def test_gated_two_tower_synthetic_end_to_end():
    """Verify synthetic forward and backward pass for TwoTowerModel with gated fusion."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["model"]["num_users"] = 50
    config["model"]["num_items"] = 50
    config["model"]["item_tower"]["use_text_features"] = True
    config["model"]["item_tower"]["text_embedding_dim"] = 64
    config["model"]["item_tower"]["fusion_type"] = "gated"

    model = TwoTowerModel(config["model"])

    batch = {
        "user_features": torch.randn((8, 4), dtype=torch.float32),
        "business_features": torch.randn((8, 3), dtype=torch.float32),
        "category_features": torch.randn((8, 10), dtype=torch.float32),
        "text_features": torch.randn((8, 64), dtype=torch.float32),
    }

    batch["user_features"][:, 0] = torch.randint(0, 50, (8,)).float()
    batch["business_features"][:, 0] = torch.randint(0, 50, (8,)).float()

    u_emb, i_emb = model(batch)
    assert u_emb.shape == (8, 128)
    assert i_emb.shape == (8, 128)

    gate = model.get_last_gate()
    assert gate is not None
    assert gate.shape == (8, 128)
    assert not torch.isnan(gate).any()


def test_gated_fusion_backward_compatibility():
    """Verify ItemTower maintains exact behavior for text_disabled and concat modes."""
    # 1. Disabled
    tower_off = ItemTower(num_items=50, embedding_dim=64, hidden_dims=[256, 128], dropout=0.1, use_text_features=False)
    out_off = tower_off(torch.tensor([0, 1]), torch.zeros((2, 2)), torch.zeros((2, 10)))
    assert out_off.shape == (2, 128)
    assert tower_off.get_last_gate() is None

    # 2. Concat mode
    tower_concat = ItemTower(num_items=50, embedding_dim=64, hidden_dims=[256, 128], dropout=0.1, use_text_features=True, fusion_type="concat")
    out_concat = tower_concat(torch.tensor([0, 1]), torch.zeros((2, 2)), torch.zeros((2, 10)), text_features=torch.randn((2, 64)))
    assert out_concat.shape == (2, 128)
    assert tower_concat.get_last_gate() is None
