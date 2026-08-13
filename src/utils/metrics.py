import numpy as np
from typing import List
import torch

def compute_metrics(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    true_labels: torch.Tensor,
    k_list: List[int] = [5, 10, 20]
) -> dict:
    """
    Compute recommendation metrics
    
    Args:
        user_embeddings: User embeddings (batch_size, embedding_dim)
        item_embeddings: Item embeddings (batch_size, embedding_dim)
        true_labels: Ground truth labels (batch_size,)
        k_list: List of k values for metrics@k
    
    Returns:
        Dictionary containing metrics
    """
    # Compute similarity scores
    sim_matrix = torch.matmul(
        user_embeddings,
        item_embeddings.transpose(0, 1)
    )
    
    metrics = {}
    for k in k_list:
        # Top-k predictions
        _, topk_indices = torch.topk(sim_matrix, k, dim=1)
        
        # Compute Recall@k
        recall = compute_recall_at_k(topk_indices, true_labels, k)
        metrics[f'recall@{k}'] = recall.item()
        
        # Compute NDCG@k
        ndcg = compute_ndcg_at_k(topk_indices, true_labels, k)
        metrics[f'ndcg@{k}'] = ndcg.item()
        
        # Compute MRR@k
        mrr = compute_mrr_at_k(topk_indices, true_labels, k)
        metrics[f'mrr@{k}'] = mrr.item()
    
    return metrics

def compute_recall_at_k(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Compute Recall@K
    
    Args:
        predictions: Predicted item indices (batch_size, k)
        labels: Ground truth labels (batch_size,)
        k: Number of top items to consider
    """
    # Convert labels to set of relevant items
    relevant_items = labels.unsqueeze(1).expand_as(predictions)
    
    # Check if relevant items are in top-k predictions
    hits = (predictions == relevant_items).any(dim=1).float()
    
    return hits.mean()

def compute_ndcg_at_k(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Compute NDCG@K
    
    Args:
        predictions: Predicted item indices (batch_size, k)
        labels: Ground truth labels (batch_size,)
        k: Number of top items to consider
    """
    # Create position weights
    position_weights = 1.0 / torch.log2(torch.arange(k, device=predictions.device) + 2.0)
    
    # Check if predictions match labels
    hits = (predictions == labels.unsqueeze(1)).float()
    
    # Compute DCG
    dcg = (hits * position_weights.unsqueeze(0)).sum(dim=1)
    
    # Compute ideal DCG (always 1.0 for binary relevance)
    idcg = position_weights[0]
    
    return (dcg / idcg).mean()

def compute_mrr_at_k(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Compute MRR@K (Mean Reciprocal Rank)
    
    Args:
        predictions: Predicted item indices (batch_size, k)
        labels: Ground truth labels (batch_size,)
        k: Number of top items to consider
    """
    # Find position of relevant items
    hits = (predictions == labels.unsqueeze(1))
    
    # Get reciprocal rank (1/position)
    ranks = torch.arange(1, k + 1, device=predictions.device).float()
    rr = (hits / ranks.unsqueeze(0)).sum(dim=1)
    
    return rr.mean()


# =====================================================================
# Multi-Label Offline Evaluation Functions
# =====================================================================

def filter_seen_candidates(
    retrieved_items: list,
    retrieved_scores: list,
    train_seen_set: set,
    top_k: int
) -> tuple:
    """
    Exclude items present in train_seen_set from retrieved recommendations, keeping top_k un-seen items.
    
    Args:
        retrieved_items: Ordered list of retrieved item IDs
        retrieved_scores: Ordered list of similarity scores
        train_seen_set: Set of item IDs the user interacted with in training
        top_k: Maximum number of un-seen items to return
    
    Returns:
        Tuple of (filtered_items, filtered_scores)
    """
    filtered_items = []
    filtered_scores = []

    for item, score in zip(retrieved_items, retrieved_scores):
        if item not in train_seen_set:
            filtered_items.append(item)
            filtered_scores.append(score)
            if len(filtered_items) == top_k:
                break

    return filtered_items, filtered_scores


def compute_user_recall_at_k(
    recommended_items: list,
    ground_truth_set: set,
    k: int
) -> float:
    """
    Compute Recall@K for a single user with multiple ground-truth test items.
    
    Recall@K = |Recommended@K ∩ GroundTruth| / |GroundTruth|
    """
    if not ground_truth_set:
        return 0.0

    recs_at_k = recommended_items[:k]
    hits = sum(1 for item in recs_at_k if item in ground_truth_set)
    return float(hits) / float(len(ground_truth_set))


def compute_user_ndcg_at_k(
    recommended_items: list,
    ground_truth_set: set,
    k: int
) -> float:
    """
    Compute NDCG@K for a single user with multiple ground-truth test items.
    
    DCG@K = ∑_{i=1}^K I(r_i ∈ GroundTruth) / log2(i + 1)
    IDCG@K = ∑_{i=1}^{min(|GroundTruth|, K)} 1 / log2(i + 1)
    NDCG@K = DCG@K / IDCG@K
    """
    if not ground_truth_set:
        return 0.0

    recs_at_k = recommended_items[:k]
    dcg = 0.0
    for i, item in enumerate(recs_at_k):
        if item in ground_truth_set:
            dcg += 1.0 / np.log2(i + 2.0)

    # Compute Ideal DCG (IDCG)
    idcg_items = min(len(ground_truth_set), k)
    idcg = sum(1.0 / np.log2(i + 2.0) for i in range(idcg_items))

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def compute_user_mrr_at_k(
    recommended_items: list,
    ground_truth_set: set,
    k: int
) -> float:
    """
    Compute MRR@K for a single user with multiple ground-truth test items.
    
    MRR@K = 1 / (rank of first ground truth hit in top K), or 0.0 if no hit.
    """
    if not ground_truth_set:
        return 0.0

    recs_at_k = recommended_items[:k]
    for i, item in enumerate(recs_at_k):
        if item in ground_truth_set:
            return 1.0 / float(i + 1)

    return 0.0