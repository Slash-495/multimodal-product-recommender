import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.stage2_reranker import Stage2Reranker
from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_index import FAISSIndex


def extract_reranker_features(
    user_emb: torch.Tensor,
    user_stats: torch.Tensor,
    item_embs: torch.Tensor,
    item_stats: torch.Tensor,
    category_feats: torch.Tensor,
    text_embs: torch.Tensor,
) -> torch.Tensor:
    """
    Extract 33-dimensional Stage-2 ranking feature matrix for a candidate pool.

    Args:
        user_emb: [1, 128] or [num_candidates, 128]
        user_stats: [1, 3] or [num_candidates, 3] (review_count, average_stars, yelping_days)
        item_embs: [num_candidates, 128]
        item_stats: [num_candidates, 2] (review_count, stars)
        category_feats: [num_candidates, 10]
        text_embs: [num_candidates, 64]

    Returns:
        Feature tensor of shape [num_candidates, 33]
    """
    num_candidates = item_embs.size(0)
    device = item_embs.device

    if user_emb.size(0) == 1 and num_candidates > 1:
        user_emb = user_emb.repeat(num_candidates, 1)
    if user_stats.size(0) == 1 and num_candidates > 1:
        user_stats = user_stats.repeat(num_candidates, 1)

    # 1. Cosine similarity between Two-Tower user & item embeddings (1-d)
    u_norm = F.normalize(user_emb, p=2, dim=1)
    i_norm = F.normalize(item_embs, p=2, dim=1)
    two_tower_sim = torch.sum(u_norm * i_norm, dim=1, keepdim=True)  # [num_candidates, 1]

    # 2. Text SVD similarity with user embedding (1-d)
    t_norm = F.normalize(text_embs, p=2, dim=1)
    # Cosine similarity between user_emb[:64] and text_embs[:64]
    text_sim = torch.sum(u_norm[:, :64] * t_norm, dim=1, keepdim=True)  # [num_candidates, 1]

    # 3. Elementwise interaction summary (16-d)
    elem_product = (u_norm[:, :16] * i_norm[:, :16])  # [num_candidates, 16]

    # Concatenate features into 33-d vector:
    # 1 (two_tower_sim) + 3 (user_stats) + 2 (item_stats) + 10 (category_feats) + 1 (text_sim) + 16 (elem_prod) = 33
    feat_matrix = torch.cat(
        [
            two_tower_sim,
            user_stats,
            item_stats,
            category_feats,
            text_sim,
            elem_product,
        ],
        dim=1,
    )

    return feat_matrix


class Stage2RerankingPipeline:
    """End-to-End Two-Tower FAISS candidate retrieval + Stage-2 MLP reranking pipeline."""

    def __init__(
        self,
        two_tower_model: TwoTowerModel,
        reranker_model: Stage2Reranker,
        faiss_index: FAISSIndex,
        all_candidate_ids: List[str],
        candidate_tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        user_stats_map: Dict[str, Tuple[float, float, float]],
        user2idx: Dict[str, int],
        business2idx: Dict[str, int],
        device: Union[str, torch.device] = "cpu",
    ):
        self.two_tower_model = two_tower_model
        self.reranker_model = reranker_model
        self.faiss_index = faiss_index
        self.all_candidate_ids = all_candidate_ids
        self.candidate_tensors = candidate_tensors
        self.user_stats_map = user_stats_map
        self.user2idx = user2idx
        self.business2idx = business2idx
        self.device = torch.device(device)
        self.candidate_id_arr = np.array(all_candidate_ids)

        self.two_tower_model.to(self.device).eval()
        self.reranker_model.to(self.device).eval()

    def rerank_user(
        self,
        user_id: str,
        train_seen_set: Set[str],
        candidate_k: int = 50,
        final_k: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieve candidate_k items via FAISS, extract features, and rerank to return final_k items.
        """
        u_idx = float(self.user2idx.get(user_id, 0))
        u_stats = self.user_stats_map.get(user_id, (0.0, 0.0, 0.0))

        u_tensor = torch.tensor([[u_idx, u_stats[0], u_stats[1], u_stats[2]]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            u_emb = self.two_tower_model.encode_user(u_tensor[:, :1].long(), u_tensor[:, 1:])

        # FAISS search
        fetch_k = max(candidate_k + len(train_seen_set), 100)
        fetch_k = min(fetch_k, len(self.all_candidate_ids))

        u_emb_np = u_emb.cpu().numpy().astype(np.float32)
        scores, faiss_indices = self.faiss_index.search(u_emb_np, k=fetch_k)

        retrieved_ids = self.candidate_id_arr[faiss_indices[0]].tolist()

        # Filter seen items
        cand_ids = [mid for mid in retrieved_ids if mid not in train_seen_set][:candidate_k]

        if not cand_ids:
            return [], []

        # Extract features for candidate pool
        cand_b_stats = []
        cand_c_feats = []
        cand_t_feats = []
        cand_m_ids = []

        for mid in cand_ids:
            bf, cf, tf = self.candidate_tensors[mid]
            cand_m_ids.append(int(bf[0]))
            cand_b_stats.append(bf[1:])
            cand_c_feats.append(cf)
            cand_t_feats.append(tf)

        b_stats_t = torch.tensor(cand_b_stats, dtype=torch.float32, device=self.device)
        c_feats_t = torch.tensor(cand_c_feats, dtype=torch.float32, device=self.device)
        t_feats_t = torch.tensor(np.array(cand_t_feats), dtype=torch.float32, device=self.device)
        m_ids_t = torch.tensor(cand_m_ids, dtype=torch.long, device=self.device)

        with torch.no_grad():
            cand_item_embs = self.two_tower_model.encode_item(m_ids_t, b_stats_t, c_feats_t, text_features=t_feats_t)

            feat_matrix = extract_reranker_features(
                user_emb=u_emb,
                user_stats=u_tensor[:, 1:],
                item_embs=cand_item_embs,
                item_stats=b_stats_t,
                category_feats=c_feats_t,
                text_embs=t_feats_t,
            )

            rerank_scores = self.reranker_model(feat_matrix).flatten().cpu().numpy()

        # Sort candidate items by reranker score descending
        sorted_indices = np.argsort(-rerank_scores)
        top_reranked_ids = [cand_ids[idx] for idx in sorted_indices[:final_k]]
        top_reranked_scores = [float(rerank_scores[idx]) for idx in sorted_indices[:final_k]]

        return top_reranked_ids, top_reranked_scores
