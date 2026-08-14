import sys
from pathlib import Path
import pytest
import torch

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.losses import HardNegativeInfoNCELoss, InfoNCELoss


def test_hard_negative_loss_finite_scalar_and_gradients():
    """Verify HardNegativeInfoNCELoss returns finite loss scalar and gradient propagation."""
    criterion = HardNegativeInfoNCELoss(temperature=0.07)

    batch_size = 8
    dim = 64
    num_hard_negs = 5

    u_emb = torch.randn(batch_size, dim, requires_grad=True)
    i_emb = torch.randn(batch_size, dim, requires_grad=True)
    h_emb = torch.randn(batch_size, num_hard_negs, dim, requires_grad=True)

    loss = criterion(u_emb, i_emb, h_emb)

    assert torch.isfinite(loss)
    assert loss.dim() == 0  # Scalar

    loss.backward()
    assert u_emb.grad is not None
    assert i_emb.grad is not None
    assert h_emb.grad is not None


def test_hard_negative_loss_encourages_positive_score_over_negatives():
    """Verify loss decreases when positive items match user embeddings better than hard negatives."""
    criterion = HardNegativeInfoNCELoss(temperature=0.07)
    dim = 32

    # High match: positive items equal user embeddings, hard negatives orthogonal/opposite
    u_emb = torch.randn(4, dim)
    pos_match = u_emb.clone()
    hard_mismatch = -u_emb.unsqueeze(1).repeat(1, 3, 1)

    loss_match = criterion(u_emb, pos_match, hard_mismatch).item()

    # Low match: positive items orthogonal/opposite, hard negatives match user embeddings
    pos_mismatch = -u_emb
    hard_match = u_emb.unsqueeze(1).repeat(1, 3, 1)

    loss_mismatch = criterion(u_emb, pos_mismatch, hard_match).item()

    assert loss_match < loss_mismatch, "Matching positive pairs should result in significantly lower loss"


def test_existing_infonce_loss_unbroken():
    """Verify original InfoNCELoss behaves correctly."""
    criterion = InfoNCELoss(temperature=0.07)
    u_emb = torch.randn(4, 16, requires_grad=True)
    i_emb = torch.randn(4, 16, requires_grad=True)

    loss = criterion(u_emb, i_emb)
    assert torch.isfinite(loss)
    loss.backward()
    assert u_emb.grad is not None
