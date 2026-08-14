import random
from typing import Dict, List, Optional, Set, Union
import numpy as np
import torch
import torch.nn.functional as F


class RandomNegativeSampler:
    """Deterministic random negative item sampler excluding known positive training items."""

    def __init__(self, all_item_ids: List[str], random_state: int = 42):
        self.all_item_ids = list(all_item_ids)
        self.random_state = random_state
        self.rng = random.Random(random_state)

    def sample(self, user_id: str, train_seen: Set[str], num_samples: int = 5) -> List[str]:
        """Sample random negative item IDs for a user excluding seen items."""
        candidate_pool = [item for item in self.all_item_ids if item not in train_seen]
        if not candidate_pool:
            # Fallback if user has seen all items (extreme edge case)
            candidate_pool = self.all_item_ids

        if len(candidate_pool) >= num_samples:
            return self.rng.sample(candidate_pool, num_samples)
        else:
            return self.rng.choices(candidate_pool, k=num_samples)


class HardNegativeSampler:
    """
    Memory-efficient hard-negative sampler using model embedding cosine similarity.
    Retrieves top competitor candidate items per user and filters out known training positive items.
    """

    def __init__(self, all_item_ids: List[str], random_state: int = 42):
        self.all_item_ids = list(all_item_ids)
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.all_item_ids)}
        self.random_sampler = RandomNegativeSampler(all_item_ids, random_state=random_state)
        self.random_state = random_state
        self.rng = random.Random(random_state)

    def sample_hard_negatives(
        self,
        user_embeddings: torch.Tensor,
        candidate_item_embeddings: torch.Tensor,
        user_ids: List[str],
        candidate_item_ids: List[str],
        train_seen_dict: Dict[str, Set[str]],
        num_negatives: int = 5,
        candidate_pool_size: int = 50,
    ) -> List[List[int]]:
        """
        Sample hard negative item indices for a batch of users.

        Args:
            user_embeddings: [batch_size, embedding_dim]
            candidate_item_embeddings: [num_candidate_items, embedding_dim]
            user_ids: List of user string IDs of length batch_size
            candidate_item_ids: List of candidate item string IDs of length num_candidate_items
            train_seen_dict: Dict mapping user_id -> set of seen item_ids in training set
            num_negatives: Number of hard negatives per user (default 5)
            candidate_pool_size: Number of top candidate items retrieved per user (default 50)

        Returns:
            List of lists containing integer indices into candidate_item_ids of shape [batch_size, num_negatives]
        """
        batch_size = user_embeddings.size(0)
        num_candidates = candidate_item_embeddings.size(0)

        # Normalize embeddings for cosine similarity
        u_norm = F.normalize(user_embeddings, p=2, dim=1)
        i_norm = F.normalize(candidate_item_embeddings, p=2, dim=1)

        # Cosine similarity matrix [batch_size, num_candidates]
        sim_matrix = torch.matmul(u_norm, i_norm.T)

        pool_k = min(candidate_pool_size, num_candidates)
        _, top_k_indices = torch.topk(sim_matrix, k=pool_k, dim=1)  # [batch_size, pool_k]
        top_k_indices_np = top_k_indices.cpu().numpy()

        batch_hard_neg_indices = []

        for i in range(batch_size):
            uid = str(user_ids[i])
            seen_set = train_seen_dict.get(uid, set())

            user_hard_neg_indices = []
            for item_idx in top_k_indices_np[i]:
                item_id = candidate_item_ids[item_idx]
                if item_id not in seen_set:
                    user_hard_neg_indices.append(int(item_idx))
                    if len(user_hard_neg_indices) == num_negatives:
                        break

            # Fallback to random negatives if insufficient hard negatives were found
            if len(user_hard_neg_indices) < num_negatives:
                needed = num_negatives - len(user_hard_neg_indices)
                rand_item_ids = self.random_sampler.sample(uid, seen_set, num_samples=needed)
                for r_id in rand_item_ids:
                    r_idx = self.item_id_to_idx.get(r_id, 0)
                    user_hard_neg_indices.append(r_idx)

            batch_hard_neg_indices.append(user_hard_neg_indices[:num_negatives])

        return batch_hard_neg_indices
