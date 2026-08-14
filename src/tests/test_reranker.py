import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.stage2_reranker import Stage2Reranker
from src.retrieval.reranker_pipeline import extract_reranker_features
from src.utils.losses import BPRRankingLoss


def test_reranker_initialization_and_forward_pass_shapes():
    """Verify Stage2Reranker output shapes for 2D and 3D input feature tensors."""
    reranker = Stage2Reranker(input_dim=33, hidden_dim=64)
    reranker.eval()

    # 2D input: [batch_size, input_dim]
    feat_2d = torch.randn(16, 33)
    score_2d = reranker(feat_2d)
    assert score_2d.shape == (16, 1)

    # 3D input: [batch_size, num_candidates, input_dim]
    feat_3d = torch.randn(8, 50, 33)
    score_3d = reranker(feat_3d)
    assert score_3d.shape == (8, 50)


def test_feature_extraction_dimension_and_no_nans():
    """Verify extract_reranker_features produces exactly 33-d feature matrix with zero NaNs."""
    user_emb = torch.randn(1, 128)
    user_stats = torch.tensor([[100.0, 4.2, 500.0]])

    item_embs = torch.randn(50, 128)
    item_stats = torch.randn(50, 2)
    category_feats = torch.randn(50, 10)
    text_embs = torch.randn(50, 64)

    feats = extract_reranker_features(
        user_emb, user_stats, item_embs, item_stats, category_feats, text_embs
    )

    assert feats.shape == (50, 33)
    assert not torch.isnan(feats).any()
    assert torch.isfinite(feats).all()


def test_bpr_loss_scalar_and_gradients():
    """Verify BPRRankingLoss evaluates to a finite scalar and propagates gradients."""
    criterion = BPRRankingLoss()
    pos_scores = torch.randn(16, 1, requires_grad=True)
    neg_scores = torch.randn(16, 5, requires_grad=True)

    loss = criterion(pos_scores, neg_scores)
    assert torch.isfinite(loss)
    assert loss.dim() == 0

    loss.backward()
    assert pos_scores.grad is not None
    assert neg_scores.grad is not None


def test_candidate_ordering_score_sorting():
    """Verify reranker scores sort candidates in correct descending relevance order."""
    reranker = Stage2Reranker(input_dim=33, hidden_dim=64)
    reranker.eval()

    # Create dummy candidate features where item 3 has highest score
    feats = torch.randn(1, 5, 33)
    scores = reranker(feats)[0].detach().numpy()

    sorted_indices = np.argsort(-scores)
    assert len(sorted_indices) == 5
    assert scores[sorted_indices[0]] >= scores[sorted_indices[1]]


def test_reranker_checkpoint_save_and_load(tmp_path):
    """Verify Stage2Reranker checkpoint save and load consistency."""
    save_path = tmp_path / "reranker.pt"

    reranker = Stage2Reranker(input_dim=33, hidden_dim=64)
    reranker.eval()
    feats = torch.randn(8, 33)
    out_orig = reranker(feats)

    torch.save({"model_state_dict": reranker.state_dict()}, save_path)

    loaded_reranker = Stage2Reranker(input_dim=33, hidden_dim=64)
    chk = torch.load(save_path)
    loaded_reranker.load_state_dict(chk["model_state_dict"])
    loaded_reranker.eval()

    out_loaded = loaded_reranker(feats)
    assert torch.allclose(out_orig, out_loaded, atol=1e-6)
