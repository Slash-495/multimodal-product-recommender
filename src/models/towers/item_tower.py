import torch
import torch.nn as nn

class ItemTower(nn.Module):
    """Item tower of the two-tower model with optional content text feature fusion."""
    
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        hidden_dims: list,
        dropout: float,
        use_text_features: bool = False,
        text_embedding_dim: int = 64,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 商家统计特征维度：review_count + stars = 2
        self.business_features_dim = 2
        # 类别特征维度：10个类别的one-hot编码
        self.category_features_dim = 10
        self.use_text_features = use_text_features
        self.text_embedding_dim = text_embedding_dim if use_text_features else 0
        
        # 修改MLP的输入维度
        layers = []
        input_dim = embedding_dim + self.business_features_dim + self.category_features_dim + self.text_embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        
    def forward(
        self, 
        item_ids: torch.Tensor, 
        business_features: torch.Tensor,
        category_features: torch.Tensor,
        text_features: torch.Tensor = None,
    ) -> torch.Tensor:
        # item_ids: [batch_size]
        # business_features: [batch_size, business_features_dim]
        # category_features: [batch_size, category_features_dim]
        # text_features: [batch_size, text_embedding_dim] (optional)
        
        id_embedding = self.item_embedding(item_ids)  # [batch_size, embedding_dim]
        inputs = [id_embedding, business_features, category_features]
        
        if self.use_text_features:
            if text_features is None:
                raise ValueError("use_text_features is True, but text_features was not provided to ItemTower forward pass.")
            inputs.append(text_features)
            
        x = torch.cat(inputs, dim=1)
        return self.mlp(x) 