import torch
import torch.nn as nn

class ItemTower(nn.Module):
    """Item tower of the two-tower model with optional content text feature fusion (concat or gated)."""
    
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        hidden_dims: list,
        dropout: float,
        use_text_features: bool = False,
        text_embedding_dim: int = 64,
        fusion_type: str = "concat",
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 商家统计特征维度：review_count + stars = 2
        self.business_features_dim = 2
        # 类别特征维度：10个类别的one-hot编码
        self.category_features_dim = 10
        self.use_text_features = use_text_features
        self.text_embedding_dim = text_embedding_dim if use_text_features else 0
        self.fusion_type = fusion_type.lower()
        self.last_gate = None
        
        output_dim = hidden_dims[-1]
        
        if self.use_text_features and self.fusion_type == "gated":
            # 1. Base Item Branch MLP
            base_in_dim = embedding_dim + self.business_features_dim + self.category_features_dim
            base_layers = []
            cur_dim = base_in_dim
            for h_dim in hidden_dims:
                base_layers.extend([
                    nn.Linear(cur_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                cur_dim = h_dim
            self.base_mlp = nn.Sequential(*base_layers)
            
            # 2. Text Branch Linear Layer
            self.text_layer = nn.Linear(text_embedding_dim, output_dim)
            
            # 3. Gate Layer (256 -> 128 -> Sigmoid)
            self.gate_layer = nn.Sequential(
                nn.Linear(output_dim + output_dim, output_dim),
                nn.Sigmoid()
            )
            self.mlp = None
        else:
            # Standard Concat or Text-Disabled MLP
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
            self.base_mlp = None
            self.text_layer = None
            self.gate_layer = None

    def get_last_gate(self) -> torch.Tensor:
        """Returns the most recently computed gate tensor."""
        return self.last_gate

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
        
        if self.use_text_features and self.fusion_type == "gated":
            if text_features is None:
                raise ValueError("use_text_features is True and fusion_type is 'gated', but text_features was not provided.")
            
            base_in = torch.cat([id_embedding, business_features, category_features], dim=1)
            base_emb = self.base_mlp(base_in)           # [batch_size, 128]
            text_emb = self.text_layer(text_features)   # [batch_size, 128]
            
            gate_in = torch.cat([base_emb, text_emb], dim=1)
            gate = self.gate_layer(gate_in)             # [batch_size, 128]
            self.last_gate = gate
            
            fused_emb = base_emb + gate * text_emb      # [batch_size, 128]
            return fused_emb
        else:
            inputs = [id_embedding, business_features, category_features]
            if self.use_text_features:
                if text_features is None:
                    raise ValueError("use_text_features is True, but text_features was not provided to ItemTower forward pass.")
                inputs.append(text_features)
                
            x = torch.cat(inputs, dim=1)
            return self.mlp(x) 