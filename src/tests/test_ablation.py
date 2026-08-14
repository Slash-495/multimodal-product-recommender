import sys
from pathlib import Path
import pytest
import torch

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.two_tower import TwoTowerModel
from src.utils.config import DEFAULT_CONFIG


def test_ablation_model_d_no_text_forward_pass():
    """Verify Model D (Hard-Negative Two-Tower Without Text) input shapes and forward pass."""
    config = DEFAULT_CONFIG.copy()
    config["model"]["item_tower"]["use_text_features"] = False
    config["model"]["item_tower"]["fusion_type"] = "concat"

    model = TwoTowerModel(config["model"])
    model.eval()

    # User inputs
    user_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    user_stats = torch.randn(3, 3)
    user_emb = model.encode_user(user_ids, user_stats)
    assert user_emb.shape == (3, 128)

    # Item inputs (no text features)
    item_ids = torch.tensor([10, 20, 30], dtype=torch.long)
    item_stats = torch.randn(3, 2)
    category_features = torch.randn(3, 10)
    item_emb = model.encode_item(item_ids, item_stats, category_features, text_features=None)
    assert item_emb.shape == (3, 128)

    assert not torch.isnan(user_emb).any()
    assert not torch.isnan(item_emb).any()


def test_ablation_checkpoint_loading_compatibility(tmp_path):
    """Verify Model D checkpoint save/load compatibility."""
    chk_path = tmp_path / "hard_negative_no_text.pt"

    config = DEFAULT_CONFIG.copy()
    config["model"]["item_tower"]["use_text_features"] = False
    model = TwoTowerModel(config["model"])
    model.eval()

    torch.save({"model_state_dict": model.state_dict()}, chk_path)

    loaded_model = TwoTowerModel(config["model"])
    chk = torch.load(chk_path)
    loaded_model.load_state_dict(chk["model_state_dict"])
    loaded_model.eval()

    user_ids = torch.tensor([1], dtype=torch.long)
    user_stats = torch.randn(1, 3)

    with torch.no_grad():
        emb_orig = model.encode_user(user_ids, user_stats)
        emb_loaded = loaded_model.encode_user(user_ids, user_stats)

    assert torch.allclose(emb_orig, emb_loaded, atol=1e-6)
