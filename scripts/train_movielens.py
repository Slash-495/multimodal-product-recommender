import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data.movielens_dataset import MovieLensDataset
from src.models.two_tower import TwoTowerModel
from src.trainers.two_tower_trainer import TwoTowerTrainer
from src.utils.config import DEFAULT_CONFIG
from src.features.text_features import MovieTitleTextEmbedder
from src.utils.model_utils import save_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Two-Tower Recommendation Model on MovieLens Data")
    parser.add_argument("--data-path", type=str, default="data/processed/movielens", help="Path to processed MovieLens directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training and validation")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for AdamW optimizer")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for InfoNCE loss")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/movielens", help="Directory to save model checkpoints")
    parser.add_argument("--output-dir", type=str, default="outputs/movielens", help="Directory to save training logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    parser.add_argument("--max-samples", type=int, default=None, help="Small-scale smoke training max samples limit")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loader worker threads")
    parser.add_argument("--model-name", type=str, default="two_tower_movielens", help="Name prefix for saved checkpoints")
    parser.add_argument("--use-text-features", action="store_true", help="Enable TF-IDF + TruncatedSVD movie title content features")
    parser.add_argument("--text-embedding-dim", type=int, default=64, help="Dimension of SVD text embedding")
    parser.add_argument("--tfidf-max-features", type=int, default=5000, help="Max vocabulary features for TF-IDF vectorizer")
    parser.add_argument("--svd-components", type=int, default=64, help="Number of components for TruncatedSVD")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: MovieLens data directory '{data_path}' does not exist. Run preprocessing first.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_map = None
    if args.use_text_features:
        meta_file = data_path / "movie_metadata.csv"
        if not meta_file.exists():
            print(f"Error: Movie metadata file '{meta_file}' required for text features does not exist.")
            sys.exit(1)
        
        print("Fitting TF-IDF + TruncatedSVD text embedder on movie catalog titles...")
        meta_df = pd.read_csv(meta_file)
        meta_df["movieId"] = meta_df["movieId"].astype(str)

        embedder = MovieTitleTextEmbedder(
            tfidf_max_features=args.tfidf_max_features,
            svd_components=args.svd_components,
            random_state=42,
        )
        text_embs = embedder.fit_transform(meta_df["title"].tolist())
        text_map = {mid: emb for mid, emb in zip(meta_df["movieId"], text_embs)}

        embedder_save_path = output_dir / "title_text_embedder.joblib"
        embedder.save(embedder_save_path)
        print(f"Saved fitted text embedder artifact to '{embedder_save_path}'.")

    print(f"Initializing MovieLens dataset from '{data_path}' (text_features={args.use_text_features})...")
    train_dataset = MovieLensDataset(data_path=data_path, mode="train", text_embeddings_map=text_map)
    valid_dataset = MovieLensDataset(
        data_path=data_path,
        mode="valid",
        user2idx=train_dataset.user2idx,
        business2idx=train_dataset.business2idx,
        text_embeddings_map=text_map,
    )

    if args.max_samples is not None and args.max_samples > 0:
        train_samples = min(len(train_dataset), args.max_samples)
        valid_samples = min(len(valid_dataset), max(1, args.max_samples // 5))
        print(f"Small-scale smoke training active: capping train to {train_samples} and valid to {valid_samples} samples.")

        train_dataset = Subset(train_dataset, list(range(train_samples)))
        valid_dataset = Subset(valid_dataset, list(range(valid_samples)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"MovieLens Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    config = DEFAULT_CONFIG.copy()
    config["training"]["batch_size"] = args.batch_size
    config["training"]["learning_rate"] = args.learning_rate
    config["training"]["temperature"] = args.temperature
    config["training"]["num_epochs"] = args.epochs
    
    if args.use_text_features:
        config["model"]["item_tower"]["use_text_features"] = True
        config["model"]["item_tower"]["text_embedding_dim"] = args.text_embedding_dim

    device = torch.device(args.device)
    print(f"Using device: {device}")

    model = TwoTowerModel(config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    trainer = TwoTowerTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        temperature=args.temperature,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    start_time = time.time()

    print(f"Starting MovieLens training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()

        train_loss = trainer.train_epoch(train_loader)
        valid_loss = trainer.validate(valid_loader)

        ep_duration = time.time() - ep_start

        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | "
            f"Time: {ep_duration:.2f}s"
        )

        metrics = {"train_loss": train_loss, "valid_loss": valid_loss}

        save_model(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            save_dir=str(checkpoint_dir),
            model_name=args.model_name,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "valid_loss": float(valid_loss),
                "elapsed_seconds": round(ep_duration, 3),
            }
        )

    total_time = time.time() - start_time
    history_file = output_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(
            {
                "total_elapsed_seconds": round(total_time, 3),
                "config": {
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                    "temperature": args.temperature,
                    "device": str(device),
                    "max_samples": args.max_samples,
                },
                "epochs": history,
            },
            f,
            indent=2,
        )

    print(f"MovieLens training complete! Total time: {total_time:.2f}s.")
    print(f"Saved history log to '{history_file}'.")


if __name__ == "__main__":
    main()
