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