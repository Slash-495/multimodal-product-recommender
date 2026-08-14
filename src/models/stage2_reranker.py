import torch
import torch.nn as nn


class Stage2Reranker(nn.Module):
    """
    Lightweight Stage-2 Reranking MLP for refining Two-Tower FAISS candidate item rankings.
    Converts candidate feature vectors into scalar relevance ranking scores.
    """

    def __init__(self, input_dim: int = 33, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape [batch_size, input_dim] or [batch_size, num_candidates, input_dim]

        Returns:
            scores: Tensor of shape [batch_size, 1] or [batch_size, num_candidates]
        """
        if features.dim() == 3:
            batch_size, num_candidates, feat_dim = features.shape
            flat_features = features.view(batch_size * num_candidates, feat_dim)
            flat_scores = self.mlp(flat_features)
            return flat_scores.view(batch_size, num_candidates)

        return self.mlp(features)
