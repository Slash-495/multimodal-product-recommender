import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE loss for contrastive learning with in-batch negatives."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            user_embeddings: (batch_size, embedding_dim)
            item_embeddings: (batch_size, embedding_dim)
        Returns:
            Symmetric InfoNCE loss scalar
        """
        # 1. Normalize user embeddings using L2 normalization
        user_norm = F.normalize(user_embeddings, p=2, dim=1)
        
        # 2. Normalize item embeddings using L2 normalization
        item_norm = F.normalize(item_embeddings, p=2, dim=1)
        
        # 3 & 4. Compute similarity matrix divided by temperature
        logits = torch.matmul(user_norm, item_norm.T) / self.temperature
        
        # 5, 6, 7. Targets on diagonal (in-batch positive pairs)
        batch_size = user_embeddings.size(0)
        targets = torch.arange(batch_size, device=logits.device)
        
        # 8. User-to-item cross entropy loss
        loss_u2i = F.cross_entropy(logits, targets)
        
        # 9. Item-to-user cross entropy loss
        loss_i2u = F.cross_entropy(logits.T, targets)
        
        # 10. Return average of the two losses
        return (loss_u2i + loss_i2u) / 2.0


class HardNegativeInfoNCELoss(nn.Module):
    """
    Hard-Negative-Aware InfoNCE loss for contrastive retrieval learning.

    Mathematical Formulation:
    -------------------------
    Given user L2-normalized embedding u_i, positive item L2-normalized embedding v_i,
    in-batch item embeddings v_j (j = 1..B), and explicit hard negative item embeddings v_{i, k}^{hard} (k = 1..K):

        s_{i, pos} = <u_i, v_i> / tau
        s_{i, j}   = <u_i, v_j> / tau               (j = 1..B)
        s_{i, k}^{hard} = <u_i, v_{i, k}^{hard}> / tau  (k = 1..K)

    The per-sample loss L_i is defined as:

        L_i = -s_{i, pos} + log( sum_{j=1}^B exp(s_{i, j}) + sum_{k=1}^K exp(s_{i, k}^{hard}) )

    For numerical stability, we avoid explicit exp() followed by log() and evaluate using torch.logsumexp:

        L_i = -s_{i, pos} + logsumexp( [s_{i, 1}, ..., s_{i, B}, s_{i, 1}^{hard}, ..., s_{i, K}^{hard}] )

    The final scalar loss is the mean across batch samples: L = (1 / B) * sum_{i=1}^B L_i.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        hard_neg_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            user_embeddings: [batch_size, embedding_dim]
            item_embeddings: [batch_size, embedding_dim] (positive items)
            hard_neg_embeddings: [batch_size, num_hard_negatives, embedding_dim]

        Returns:
            Scalar loss tensor
        """
        batch_size = user_embeddings.size(0)
        num_hard_negs = hard_neg_embeddings.size(1)

        # L2-normalize all embeddings
        u_norm = F.normalize(user_embeddings, p=2, dim=1)           # [B, D]
        i_norm = F.normalize(item_embeddings, p=2, dim=1)           # [B, D]
        h_norm = F.normalize(hard_neg_embeddings, p=2, dim=2)       # [B, K, D]

        # 1. Positive item similarity scores (diagonal of in-batch similarity matrix)
        pos_scores = torch.sum(u_norm * i_norm, dim=1) / self.temperature  # [B]

        # 2. In-batch similarity scores (user i to all in-batch items j)
        in_batch_scores = torch.matmul(u_norm, i_norm.T) / self.temperature  # [B, B]

        # 3. Hard negative similarity scores (user i to its K hard negatives)
        # u_norm.unsqueeze(1): [B, 1, D], h_norm: [B, K, D] -> sum over dim=2
        hard_scores = torch.sum(u_norm.unsqueeze(1) * h_norm, dim=2) / self.temperature  # [B, K]

        # 4. Concatenate all candidate scores for logsumexp denominator
        all_candidate_scores = torch.cat([in_batch_scores, hard_scores], dim=1)  # [B, B + K]

        # 5. Numerically stable Loss = -pos_score + logsumexp(all_candidate_scores)
        log_denominator = torch.logsumexp(all_candidate_scores, dim=1)  # [B]
        loss_u2i = -pos_scores + log_denominator

        return loss_u2i.mean()


class BPRRankingLoss(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) Pairwise Loss for Stage-2 Reranking.

    Mathematical Formulation:
    -------------------------
    Given reranker score R(u, i^+) for positive item i^+ and reranker score R(u, j^-) for negative candidate j^-:

        L_BPR = -log sigmoid( R(u, i^+) - R(u, j^-) ) = softplus( -(R(u, i^+) - R(u, j^-)) )

    This encourages the reranker to output higher scalar scores for positive candidates relative to negative candidates.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_scores: Tensor of shape [batch_size] or [batch_size, 1]
            neg_scores: Tensor of shape [batch_size] or [batch_size, num_negatives]

        Returns:
            Scalar loss tensor
        """
        if pos_scores.dim() == 1:
            pos_scores = pos_scores.unsqueeze(1)  # [batch_size, 1]
        if neg_scores.dim() == 1:
            neg_scores = neg_scores.unsqueeze(1)  # [batch_size, 1]

        # Difference: [batch_size, num_negatives]
        score_diff = pos_scores - neg_scores
        loss = F.softplus(-score_diff)
        return loss.mean() 