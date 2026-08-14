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

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data.movielens_dataset import MovieLensDataset
from src.features.text_features import MovieTitleTextEmbedder
from src.models.two_tower import TwoTowerModel
from src.trainers.two_tower_trainer import TwoTowerTrainer
from src.training.negative_sampling import HardNegativeSampler
from src.utils.config import DEFAULT_CONFIG
from src.utils.losses import HardNegativeInfoNCELoss
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
    parser.add_argument("--fusion-type", type=str, default="concat", choices=["concat", "gated"], help="Text feature fusion type ('concat' or 'gated')")
    parser.add_argument("--use-hard-negatives", action="store_true", help="Enable hard negative mining training strategy")
    parser.add_argument("--hard-negatives-per-positive", type=int, default=5, help="Number of hard negatives sampled per positive pair")
    parser.add_argument("--candidate-pool-size", type=int, default=50, help="Candidate pool size for retrieving competitor items")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint file to resume training from")
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
        config["model"]["item_tower"]["fusion_type"] = args.fusion_type

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

    train_seen_dict = {}
    if args.use_hard_negatives:
        print("Setting up Hard Negative Sampler and train_seen interaction map...")
        train_inter_file = data_path / "train_interactions.csv"
        train_df = pd.read_csv(train_inter_file)
        train_df["user_id"] = train_df["user_id"].astype(str)
        train_df["business_id"] = train_df["business_id"].astype(str)
        train_seen_dict = train_df.groupby("user_id")["business_id"].apply(set).to_dict()

        biz_df = pd.read_csv(data_path / "business_features.csv")
        biz_df["business_id"] = biz_df["business_id"].astype(str)
        all_candidate_ids = biz_df["business_id"].tolist()

        with open(data_path / "user2idx.json", "r") as f:
            u2i = json.load(f)
        with open(data_path / "business2idx.json", "r") as f:
            b2i = json.load(f)

        idx2user = {v: k for k, v in u2i.items()}

        hard_sampler = HardNegativeSampler(all_candidate_ids, random_state=42)
        hard_criterion = HardNegativeInfoNCELoss(temperature=args.temperature)

        # Pre-build tensor slices for candidate item encoding
        cat_cols = sorted([c for c in biz_df.columns if c.startswith("cat_")])[:10]
        candidate_tensor_list = []
        for _, row in biz_df.iterrows():
            mid_str = str(row["business_id"])
            m_idx = float(b2i.get(mid_str, 0))
            rc = float(row.get("review_count", 0.0))
            st = float(row.get("stars", 0.0))
            bf_vec = [m_idx, rc, st]
            cf_vec = [float(row[col]) for col in cat_cols]
            while len(cf_vec) < 10:
                cf_vec.append(0.0)
            tf_vec = text_map.get(mid_str, np.zeros(64, dtype=np.float32)) if text_map else np.zeros(64, dtype=np.float32)
            candidate_tensor_list.append((bf_vec, cf_vec, tf_vec))

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    history = []
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            print(f"Error: Resume checkpoint path '{resume_path}' does not exist.")
            sys.exit(1)
        print(f"Loading model checkpoint from '{resume_path}'...")
        chk = torch.load(resume_path, map_location=device)
        if "model_state_dict" in chk:
            model.load_state_dict(chk["model_state_dict"])
        else:
            model.load_state_dict(chk)

        if "optimizer_state_dict" in chk and optimizer is not None:
            optimizer.load_state_dict(chk["optimizer_state_dict"])

        start_epoch = chk.get("epoch", 0) + 1
        print(f"Successfully loaded checkpoint '{resume_path}' (completed epoch {chk.get('epoch', 0)}). Resuming from epoch {start_epoch}...")

    start_time = time.time()

    print(f"Starting MovieLens training (hard_negatives={args.use_hard_negatives}) from epoch {start_epoch} to {args.epochs}...")
    for epoch in range(start_epoch, args.epochs + 1):
        ep_start = time.time()

        if not args.use_hard_negatives:
            train_loss = trainer.train_epoch(train_loader)
        else:
            # Custom hard negative train epoch
            model.train()
            total_train_loss = 0.0

            # 1. Encode all candidate items once per epoch or batch for sampling
            model.eval()
            with torch.no_grad():
                cand_embs_list = []
                batch_sz = 512
                for i in range(0, len(candidate_tensor_list), batch_sz):
                    slice_b = candidate_tensor_list[i : i + batch_sz]
                    bf_b = torch.tensor([b[0] for b in slice_b], dtype=torch.float32, device=device)
                    cf_b = torch.tensor([b[1] for b in slice_b], dtype=torch.float32, device=device)
                    tf_b = torch.tensor(np.array([b[2] for b in slice_b]), dtype=torch.float32, device=device) if args.use_text_features else None
                    m_ids = bf_b[:, 0].long()
                    b_stats = bf_b[:, 1:]
                    c_embs = model.encode_item(m_ids, b_stats, cf_b, text_features=tf_b)
                    cand_embs_list.append(c_embs)
                all_candidate_embs = torch.cat(cand_embs_list, dim=0)  # [num_candidates, 128]

            model.train()
            from tqdm import tqdm
            with tqdm(train_loader, desc=f"Hard-Neg Training Ep {epoch}") as pbar:
                for batch in pbar:
                    batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    u_embs, i_embs = model(batch_dev)

                    # Map batch user_idx -> string user_ids
                    user_idx_arr = batch_dev["user_features"][:, 0].long().cpu().numpy()
                    user_id_batch = [idx2user.get(int(uidx), str(uidx)) for uidx in user_idx_arr]

                    # Sample hard negative item indices
                    hard_neg_idx_lists = hard_sampler.sample_hard_negatives(
                        user_embeddings=u_embs.detach(),
                        candidate_item_embeddings=all_candidate_embs,
                        user_ids=user_id_batch,
                        candidate_item_ids=all_candidate_ids,
                        train_seen_dict=train_seen_dict,
                        num_negatives=args.hard_negatives_per_positive,
                        candidate_pool_size=args.candidate_pool_size,
                    )

                    # Encode hard negative item features
                    # Flatten batch x K hard neg indices
                    flat_hard_indices = [idx for u_list in hard_neg_idx_lists for idx in u_list]
                    flat_slice = [candidate_tensor_list[idx] for idx in flat_hard_indices]

                    bf_h = torch.tensor([b[0] for b in flat_slice], dtype=torch.float32, device=device)
                    cf_h = torch.tensor([b[1] for b in flat_slice], dtype=torch.float32, device=device)
                    tf_h = torch.tensor(np.array([b[2] for b in flat_slice]), dtype=torch.float32, device=device) if args.use_text_features else None

                    m_ids_h = bf_h[:, 0].long()
                    b_stats_h = bf_h[:, 1:]

                    flat_hard_embs = model.encode_item(m_ids_h, b_stats_h, cf_h, text_features=tf_h)
                    # Reshape to [batch_size, num_hard_negatives, 128]
                    hard_neg_embs = flat_hard_embs.view(len(user_id_batch), args.hard_negatives_per_positive, -1)

                    loss = hard_criterion(u_embs, i_embs, hard_neg_embs)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})

            train_loss = total_train_loss / len(train_loader)

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
