import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .base_trainer import BaseTrainer
from src.utils.losses import InfoNCELoss
from src.utils.metrics import compute_metrics

class TwoTowerTrainer(BaseTrainer):
    """Trainer class for two-tower model"""
    
    def __init__(
        self,
        model,
        optimizer,
        device,
        temperature: float = 0.07
    ):
        super().__init__(model, optimizer, device)
        self.criterion = InfoNCELoss(temperature)
        
    def _move_batch_to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        
        with tqdm(train_loader, desc='Training') as pbar:
            for batch in pbar:
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                user_embeddings, item_embeddings = self.model(batch)
                loss = self.criterion(user_embeddings, item_embeddings)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def validate(self, valid_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in valid_loader:
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                user_embeddings, item_embeddings = self.model(batch)
                loss = self.criterion(user_embeddings, item_embeddings)
                total_loss += loss.item()
            
        return total_loss / len(valid_loader) 