import sys
from pathlib import Path
import pytest
import torch

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.negative_sampling import RandomNegativeSampler, HardNegativeSampler


def test_random_negative_sampler_valid_and_non_seen():
    """Verify RandomNegativeSampler returns valid IDs excluding known positive items."""
    all_items = [f"movie_{i}" for i in range(100)]
    sampler = RandomNegativeSampler(all_items, random_state=42)

    seen = {f"movie_{i}" for i in range(10)}
    samples = sampler.sample("user_1", seen, num_samples=5)

    assert len(samples) == 5
    for s in samples:
        assert s in all_items
        assert s not in seen


def test_hard_negative_sampler_indices_and_no_false_negatives():
    """Verify HardNegativeSampler returns valid indices and strictly excludes seen items."""
    all_items = [f"movie_{i}" for i in range(50)]
    sampler = HardNegativeSampler(all_items, random_state=42)

    batch_size = 4
    dim = 16
    user_embs = torch.randn(batch_size, dim)
    cand_embs = torch.randn(50, dim)

    user_ids = [f"user_{i}" for i in range(batch_size)]
    train_seen = {
        "user_0": {"movie_0", "movie_1", "movie_2"},
        "user_1": {"movie_5", "movie_6"},
        "user_2": set(),
        "user_3": {f"movie_{i}" for i in range(48)},  # Almost all seen
    }

    hard_negs = sampler.sample_hard_negatives(
        user_embeddings=user_embs,
        candidate_item_embeddings=cand_embs,
        user_ids=user_ids,
        candidate_item_ids=all_items,
        train_seen_dict=train_seen,
        num_negatives=5,
        candidate_pool_size=20,
    )

    assert len(hard_negs) == batch_size
    for i in range(batch_size):
        uid = user_ids[i]
        u_seen = train_seen[uid]
        u_negs = hard_negs[i]
        assert len(u_negs) == 5

        for idx in u_negs:
            assert 0 <= idx < 50
            neg_item_id = all_items[idx]
            assert neg_item_id not in u_seen, f"False negative detected! {neg_item_id} is in train_seen for {uid}"


def test_sampler_determinism_under_fixed_seed():
    """Verify HardNegativeSampler outputs identical samples under a fixed random seed."""
    all_items = [f"movie_{i}" for i in range(30)]
    user_embs = torch.randn(2, 16)
    cand_embs = torch.randn(30, 16)
    user_ids = ["u1", "u2"]
    train_seen = {"u1": {"movie_0"}, "u2": {"movie_1"}}

    sampler1 = HardNegativeSampler(all_items, random_state=42)
    negs1 = sampler1.sample_hard_negatives(user_embs, cand_embs, user_ids, all_items, train_seen, num_negatives=4)

    sampler2 = HardNegativeSampler(all_items, random_state=42)
    negs2 = sampler2.sample_hard_negatives(user_embs, cand_embs, user_ids, all_items, train_seen, num_negatives=4)

    assert negs1 == negs2
