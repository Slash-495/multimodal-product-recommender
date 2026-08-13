import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.utils.metrics import (
    compute_user_mrr_at_k,
    compute_user_ndcg_at_k,
    compute_user_recall_at_k,
    filter_seen_candidates,
)


def test_recall_at_k_multi_label():
    """Verify Recall@K metric with multiple relevant items in ground truth."""
    # 2 hits out of 4 ground truth items in top 5 recommendations -> Recall@5 = 2 / 4 = 0.5
    recommended_items = ["m10", "m20", "m30", "m40", "m50"]
    ground_truth_set = {"m20", "m40", "m60", "m70"}

    recall_5 = compute_user_recall_at_k(recommended_items, ground_truth_set, k=5)
    assert recall_5 == 0.5

    # If k=2, only "m20" is hit -> Recall@2 = 1 / 4 = 0.25
    recall_2 = compute_user_recall_at_k(recommended_items, ground_truth_set, k=2)
    assert recall_2 == 0.25


def test_ndcg_at_k_multi_label():
    """Verify NDCG@K calculation with multiple relevant test items."""
    # Hits at rank 1 ("m10") and rank 3 ("m30") out of 2 ground truth items
    recommended_items = ["m10", "m20", "m30", "m40", "m50"]
    ground_truth_set = {"m10", "m30"}

    # DCG@5 = 1 / log2(1+1) + 1 / log2(3+1) = 1.0 + 1 / 2.0 = 1.5
    # IDCG@5 = 1 / log2(1+1) + 1 / log2(2+1) = 1.0 + 1 / 1.58496 = 1.63092975
    # NDCG@5 = 1.5 / 1.63092975 ≈ 0.9197
    ndcg_5 = compute_user_ndcg_at_k(recommended_items, ground_truth_set, k=5)

    expected_dcg = 1.0 / np.log2(2.0) + 1.0 / np.log2(4.0)
    expected_idcg = 1.0 / np.log2(2.0) + 1.0 / np.log2(3.0)
    expected_ndcg = expected_dcg / expected_idcg

    assert np.isclose(ndcg_5, expected_ndcg, atol=1e-4)


def test_mrr_at_k_multi_label():
    """Verify MRR@K returns 1 / rank of first ground truth hit."""
    recommended_items = ["m10", "m20", "m30", "m40"]
    ground_truth_set = {"m30", "m40"}

    # First hit is at rank 3 ("m30") -> MRR@5 = 1 / 3 ≈ 0.3333
    mrr_5 = compute_user_mrr_at_k(recommended_items, ground_truth_set, k=5)
    assert np.isclose(mrr_5, 1.0 / 3.0, atol=1e-4)

    # If no hit in top 2 -> MRR@2 = 0.0
    mrr_2 = compute_user_mrr_at_k(recommended_items, ground_truth_set, k=2)
    assert mrr_2 == 0.0


def test_seen_item_candidate_filtering():
    """Verify filter_seen_candidates removes training-seen items from recommendations."""
    retrieved_items = ["m10", "m20", "m30", "m40", "m50"]
    retrieved_scores = [0.9, 0.85, 0.8, 0.75, 0.7]

    train_seen_set = {"m10", "m30"}

    filt_items, filt_scores = filter_seen_candidates(
        retrieved_items, retrieved_scores, train_seen_set, top_k=3
    )

    assert filt_items == ["m20", "m40", "m50"]
    assert filt_scores == [0.85, 0.75, 0.7]


def test_users_with_no_test_interactions():
    """Verify empty ground truth returns 0.0 for all ranking metrics."""
    recommended_items = ["m10", "m20"]
    empty_gt = set()

    assert compute_user_recall_at_k(recommended_items, empty_gt, k=5) == 0.0
    assert compute_user_ndcg_at_k(recommended_items, empty_gt, k=5) == 0.0
    assert compute_user_mrr_at_k(recommended_items, empty_gt, k=5) == 0.0


def test_unknown_cold_start_users_handling():
    """Verify unknown cold-start user partitioning logic."""
    train_users = {"u1", "u2"}
    test_users = {"u1", "u2", "u_cold1", "u_cold2"}

    cold_start = [u for u in test_users if u not in train_users]
    evaluated = [u for u in test_users if u in train_users]

    assert len(cold_start) == 2
    assert len(evaluated) == 2
    assert "u_cold1" in cold_start
    assert "u1" in evaluated
