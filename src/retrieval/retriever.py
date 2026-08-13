import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.retrieval.faiss_index import FAISSIndex


class TwoTowerRetriever:
    """Retriever class combining TwoTowerModel inference encoding with FAISS index search."""

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "cpu",
        embedding_dim: Optional[int] = None,
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        if embedding_dim is None:
            embedding_dim = self._infer_embedding_dim(model)

        self.embedding_dim: int = embedding_dim
        self.faiss_index: FAISSIndex = FAISSIndex(embedding_dim=self.embedding_dim)
        self.item_ids: Optional[np.ndarray] = None

    def _infer_embedding_dim(self, model: nn.Module) -> int:
        """Infer embedding dimension from model structure if possible."""
        try:
            # Check last layer of UserTower MLP
            if hasattr(model, "user_tower") and hasattr(model.user_tower, "mlp"):
                for layer in reversed(model.user_tower.mlp):
                    if isinstance(layer, nn.Linear):
                        return layer.out_features
        except Exception:
            pass
        return 128

    def build_item_index(self, item_loader: DataLoader) -> None:
        """
        Encode all items from item_loader and build the FAISS index.

        Args:
            item_loader: DataLoader returning item batches
        """
        self.model.eval()
        all_embeddings = []
        all_item_ids = []

        with torch.no_grad():
            for batch in item_loader:
                # Extract item features and IDs
                if isinstance(batch, dict):
                    if "business_features" in batch:
                        bf = batch["business_features"]
                        if bf.ndim == 2 and bf.shape[1] >= 3:
                            item_ids = bf[:, 0].long()
                            business_features = bf[:, 1:]
                        else:
                            item_ids = batch["item_ids"].long()
                            business_features = bf
                    else:
                        item_ids = batch["item_ids"].long()
                        business_features = batch["business_stats"]

                    category_features = batch["category_features"]
                else:
                    raise ValueError("Item batch must be a dictionary.")

                item_ids_dev = item_ids.to(self.device)
                business_features_dev = business_features.to(self.device)
                category_features_dev = category_features.to(self.device)

                # Encode items
                embeddings = self.model.encode_item(
                    item_ids_dev, business_features_dev, category_features_dev
                )

                all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
                all_item_ids.append(item_ids.cpu().numpy().astype(np.int64))

        if len(all_embeddings) == 0:
            raise ValueError("Item loader provided no batches.")

        concat_embeddings = np.vstack(all_embeddings)
        self.item_ids = np.concatenate(all_item_ids, axis=0)

        # Build FAISS index
        self.faiss_index.build(concat_embeddings)

    def retrieve(
        self,
        user_batch: Union[Dict[str, torch.Tensor], torch.Tensor],
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k items for user queries.

        Args:
            user_batch: Dict containing 'user_features' or tensor of user features
            k: Number of nearest items to retrieve per user

        Returns:
            Tuple of:
                retrieved_item_ids: [num_users, k] original business/item IDs
                scores: [num_users, k] similarity scores sorted descending
        """
        if self.item_ids is None or self.faiss_index.num_items == 0:
            raise ValueError("Item index is empty. Call build_item_index() first.")

        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        self.model.eval()

        if isinstance(user_batch, dict):
            uf = user_batch["user_features"]
            if uf.ndim == 2 and uf.shape[1] >= 4:
                user_ids = uf[:, 0].long()
                user_features = uf[:, 1:]
            else:
                user_ids = user_batch["user_ids"].long()
                user_features = user_batch["user_stats"]
        elif isinstance(user_batch, torch.Tensor):
            if user_batch.ndim == 2 and user_batch.shape[1] >= 4:
                user_ids = user_batch[:, 0].long()
                user_features = user_batch[:, 1:]
            else:
                raise ValueError("User tensor must have shape [batch_size, 4].")
        else:
            raise ValueError("Unsupported user_batch type.")

        user_ids_dev = user_ids.to(self.device)
        user_features_dev = user_features.to(self.device)

        with torch.no_grad():
            user_embeddings = self.model.encode_user(user_ids_dev, user_features_dev)

        user_embeddings_np = user_embeddings.cpu().numpy().astype(np.float32)

        # Search FAISS index
        scores, faiss_indices = self.faiss_index.search(user_embeddings_np, k=k)

        # Map FAISS vector positions back to original item/business IDs
        retrieved_item_ids = self.item_ids[faiss_indices]

        return retrieved_item_ids, scores

    def save(self, path: Union[str, Path]) -> None:
        """
        Save retriever state (FAISS index, item ID mapping, metadata) to disk.

        Args:
            path: Directory or base file path to save retriever assets
        """
        if self.item_ids is None or self.faiss_index.num_items == 0:
            raise ValueError("Cannot save empty retriever index. Call build_item_index() first.")

        path_obj = Path(path)
        if path_obj.suffix:
            save_dir = path_obj.parent
            base_name = path_obj.stem
        else:
            save_dir = path_obj
            base_name = "retriever"

        save_dir.mkdir(parents=True, exist_ok=True)

        index_path = save_dir / f"{base_name}.index"
        items_path = save_dir / f"{base_name}_items.npy"
        meta_path = save_dir / f"{base_name}_meta.json"

        self.faiss_index.save(index_path)
        np.save(items_path, self.item_ids)

        metadata = {
            "embedding_dim": self.embedding_dim,
            "num_items": int(self.faiss_index.num_items),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        model: nn.Module,
        device: Union[str, torch.device] = "cpu",
    ) -> "TwoTowerRetriever":
        """
        Load retriever state from disk and associate with model.

        Args:
            path: Saved directory or base file path
            model: PyTorch TwoTowerModel instance
            device: Target device for model inference

        Returns:
            Reconstructed TwoTowerRetriever instance
        """
        path_obj = Path(path)
        if path_obj.suffix:
            save_dir = path_obj.parent
            base_name = path_obj.stem
        else:
            save_dir = path_obj
            base_name = "retriever"

        index_path = save_dir / f"{base_name}.index"
        items_path = save_dir / f"{base_name}_items.npy"
        meta_path = save_dir / f"{base_name}_meta.json"

        if not index_path.exists():
            raise ValueError(f"FAISS index file not found at: {index_path}")
        if not items_path.exists():
            raise ValueError(f"Item IDs file not found at: {items_path}")
        if not meta_path.exists():
            raise ValueError(f"Metadata file not found at: {meta_path}")

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        embedding_dim = metadata["embedding_dim"]

        retriever = cls(model=model, device=device, embedding_dim=embedding_dim)
        retriever.faiss_index = FAISSIndex.load(index_path)
        retriever.item_ids = np.load(items_path)

        return retriever
