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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.features.text_features import MovieTitleTextEmbedder
from src.models.stage2_reranker import Stage2Reranker
from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.reranker_pipeline import extract_reranker_features
from src.utils.config import DEFAULT_CONFIG
from src.utils.losses import BPRRankingLoss
from src.utils.model_utils import save_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage-2 Reranker Model on Top-N FAISS Candidates")
    parser.add_argument("--data-path", type=str, default="data/processed/movielens", help="Path to processed MovieLens directory")
    parser.add_argument("--retriever-checkpoint", type=str, default="checkpoints/movielens/hard_negative_two_tower_epoch_1.pt", help="Path to authoritative Two-Tower retriever checkpoint")
    parser.add_argument("--candidate-k", type=int, default=50, help="Number of FAISS candidate items retrieved for Stage-2 reranking")
    parser.add_argument("--final-k", type=int, default=10, help="Final top K recommendation cutoff")
    parser.add_argument("--reranker-hidden-dim", type=int, default=64, help="Hidden dimension of Stage2Reranker MLP")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for reranker training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for AdamW optimizer")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/movielens", help="Directory to save reranker checkpoints")
    parser.add_argument("--output-dir", type=str, default="outputs/movielens", help="Directory to save training logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    parser.add_argument("--max-samples", type=int, default=None, help="Small-scale smoke training max samples limit")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data directory '{data_path}' does not exist.")
        sys.exit(1)

    retriever_chk_path = Path(args.retriever_checkpoint)
    if not retriever_chk_path.exists():
        print(f"Error: Authoritative Two-Tower checkpoint '{retriever_chk_path}' does not exist.")
        sys.exit(1)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Load Authoritative Two-Tower Retriever Model
    print(f"Loading authoritative Two-Tower retriever from '{retriever_chk_path}'...")
    config = DEFAULT_CONFIG.copy()
    config["model"]["item_tower"]["use_text_features"] = True
    config["model"]["item_tower"]["text_embedding_dim"] = 64
    config["model"]["item_tower"]["fusion_type"] = "concat"

    two_tower = TwoTowerModel(config["model"]).to(device)
    chk = torch.load(retriever_chk_path, map_location=device)
    if "model_state_dict" in chk:
        two_tower.load_state_dict(chk["model_state_dict"])
    else:
        two_tower.load_state_dict(chk)
    two_tower.eval()

    # 2. Build Text Embedder & Candidate Feature Tensors
    meta_file = data_path / "movie_metadata.csv"
    meta_df = pd.read_csv(meta_file)
    meta_df["movieId"] = meta_df["movieId"].astype(str)
    embedder = MovieTitleTextEmbedder(tfidf_max_features=5000, svd_components=64, random_state=42)
    text_embs = embedder.fit_transform(meta_df["title"].tolist())
    text_map = {mid: emb for mid, emb in zip(meta_df["movieId"], text_embs)}

    biz_df = pd.read_csv(data_path / "business_features.csv")
    biz_df["business_id"] = biz_df["business_id"].astype(str)
    user_df = pd.read_csv(data_path / "user_features.csv")
    user_df["user_id"] = user_df["user_id"].astype(str)

    with open(data_path / "user2idx.json", "r") as f:
        user2idx = json.load(f)
    with open(data_path / "business2idx.json", "r") as f:
        business2idx = json.load(f)

    all_candidate_ids = biz_df["business_id"].tolist()
    cat_cols = sorted([c for c in biz_df.columns if c.startswith("cat_")])[:10]

    candidate_tensor_dict = {}
    candidate_tensor_list = []
    for _, row in biz_df.iterrows():
        mid_str = str(row["business_id"])
        m_idx = float(business2idx.get(mid_str, 0))
        rc = float(row.get("review_count", 0.0))
        st = float(row.get("stars", 0.0))
        bf_vec = [m_idx, rc, st]
        cf_vec = [float(row[col]) for col in cat_cols]
        while len(cf_vec) < 10:
            cf_vec.append(0.0)
        tf_vec = text_map.get(mid_str, np.zeros(64, dtype=np.float32))
        candidate_tensor_dict[mid_str] = (bf_vec, cf_vec, tf_vec)
        candidate_tensor_list.append((bf_vec, cf_vec, tf_vec))

    # Pre-encode all candidate movie items with Two-Tower ItemTower
    print("Encoding all candidate movie items with Two-Tower ItemTower...")
    with torch.no_grad():
        cand_embs_list = []
        batch_sz = 512
        for i in range(0, len(candidate_tensor_list), batch_sz):
            slice_b = candidate_tensor_list[i : i + batch_sz]
            bf_b = torch.tensor([b[0] for b in slice_b], dtype=torch.float32, device=device)
            cf_b = torch.tensor([b[1] for b in slice_b], dtype=torch.float32, device=device)
            tf_b = torch.tensor(np.array([b[2] for b in slice_b]), dtype=torch.float32, device=device)
            m_ids = bf_b[:, 0].long()
            b_stats = bf_b[:, 1:]
            c_embs = two_tower.encode_item(m_ids, b_stats, cf_b, text_features=tf_b)
            cand_embs_list.append(c_embs)
        all_candidate_embs = torch.cat(cand_embs_list, dim=0)

    # 3. Build FAISS index for fast candidate retrieval
    faiss_index = FAISSIndex(embedding_dim=128)
    faiss_index.build(all_candidate_embs.cpu().numpy().astype(np.float32))
    candidate_id_arr = np.array(all_candidate_ids)

    # Load train interactions
    train_df = pd.read_csv(data_path / "train_interactions.csv")
    train_df["user_id"] = train_df["user_id"].astype(str)
    train_df["business_id"] = train_df["business_id"].astype(str)

    if args.max_samples is not None and args.max_samples > 0:
        print(f"Smoke training active: capping train interactions to {args.max_samples}.")
        train_df = train_df.iloc[: args.max_samples]

    user_df_map = user_df.set_index("user_id").to_dict(orient="index")

    # 4. Generate Stage-2 Training Feature Batches
    print(f"Generating Stage-2 Reranker training pairs for {len(train_df):,} interactions...")
    sample_features_pos = []
    sample_features_neg = []

    with torch.no_grad():
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting Reranker Features"):
            uid = str(row["user_id"])
            pos_mid = str(row["business_id"])

            if uid not in user2idx or pos_mid not in candidate_tensor_dict:
                continue

            u_idx = float(user2idx[uid])
            u_data = user_df_map.get(uid, {})
            u_stats = [
                float(u_data.get("review_count", 0.0)),
                float(u_data.get("average_stars", 0.0)),
                float(u_data.get("yelping_days", 0.0)),
            ]

            u_tensor = torch.tensor([[u_idx] + u_stats], dtype=torch.float32, device=device)
            u_emb = two_tower.encode_user(u_tensor[:, 0].long(), u_tensor[:, 1:])

            # FAISS candidate retrieval
            scores, indices = faiss_index.search(u_emb.cpu().numpy().astype(np.float32), k=args.candidate_k + 1)
            retrieved_mids = candidate_id_arr[indices[0]].tolist()

            # Select negative candidate item (different from positive item)
            neg_mids = [m for m in retrieved_mids if m != pos_mid]
            if not neg_mids:
                continue
            neg_mid = neg_mids[0]

            # Extract positive candidate features
            pos_bf, pos_cf, pos_tf = candidate_tensor_dict[pos_mid]
            pos_m_id = torch.tensor([int(pos_bf[0])], dtype=torch.long, device=device)
            pos_b_stat = torch.tensor([pos_bf[1:]], dtype=torch.float32, device=device)
            pos_c_feat = torch.tensor([pos_cf], dtype=torch.float32, device=device)
            pos_t_feat = torch.tensor(np.array([pos_tf]), dtype=torch.float32, device=device)

            pos_item_emb = two_tower.encode_item(pos_m_id, pos_b_stat, pos_c_feat, text_features=pos_t_feat)
            pos_feat_vec = extract_reranker_features(
                user_emb=u_emb,
                user_stats=u_tensor[:, 1:],
                item_embs=pos_item_emb,
                item_stats=pos_b_stat,
                category_feats=pos_c_feat,
                text_embs=pos_t_feat,
            )

            # Extract negative candidate features
            neg_bf, neg_cf, neg_tf = candidate_tensor_dict[neg_mid]
            neg_m_id = torch.tensor([int(neg_bf[0])], dtype=torch.long, device=device)
            neg_b_stat = torch.tensor([neg_bf[1:]], dtype=torch.float32, device=device)
            neg_c_feat = torch.tensor([neg_cf], dtype=torch.float32, device=device)
            neg_t_feat = torch.tensor(np.array([neg_tf]), dtype=torch.float32, device=device)

            neg_item_emb = two_tower.encode_item(neg_m_id, neg_b_stat, neg_c_feat, text_features=neg_t_feat)
            neg_feat_vec = extract_reranker_features(
                user_emb=u_emb,
                user_stats=u_tensor[:, 1:],
                item_embs=neg_item_emb,
                item_stats=neg_b_stat,
                category_feats=neg_c_feat,
                text_embs=neg_t_feat,
            )

            sample_features_pos.append(pos_feat_vec.cpu().numpy()[0])
            sample_features_neg.append(neg_feat_vec.cpu().numpy()[0])

    pos_feats_arr = np.array(sample_features_pos, dtype=np.float32)
    neg_feats_arr = np.array(sample_features_neg, dtype=np.float32)
    print(f"Constructed Stage-2 Reranker training dataset of shape {pos_feats_arr.shape} (pos/neg pairs).")

    # 5. Initialize Stage-2 Reranker Model & Optimizer
    reranker = Stage2Reranker(input_dim=33, hidden_dim=args.reranker_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=args.learning_rate)
    criterion = BPRRankingLoss()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Train Reranker Model
    print(f"Training Stage-2 Reranker for {args.epochs} epochs...")
    num_samples = len(pos_feats_arr)
    batch_size = args.batch_size
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        reranker.train()
        total_loss = 0.0

        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for i in range(0, num_samples, batch_size):
            b_idx = indices[i : i + batch_size]
            b_pos = torch.tensor(pos_feats_arr[b_idx], dtype=torch.float32, device=device)
            b_neg = torch.tensor(neg_feats_arr[b_idx], dtype=torch.float32, device=device)

            pos_scores = reranker(b_pos)
            neg_scores = reranker(b_neg)

            loss = criterion(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(b_idx)

        avg_loss = total_loss / num_samples
        ep_duration = time.time() - ep_start

        print(f"Epoch {epoch}/{args.epochs} - Reranker Train Loss: {avg_loss:.6f} | Time: {ep_duration:.2f}s")

        save_model(
            model=reranker,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"train_loss": avg_loss},
            save_dir=str(checkpoint_dir),
            model_name="stage2_reranker",
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(avg_loss, 6),
                "elapsed_seconds": round(ep_duration, 3),
            }
        )

    total_time = time.time() - start_time
    history_file = output_dir / "reranker_training_history.json"
    with open(history_file, "w") as f:
        json.dump(
            {
                "total_elapsed_seconds": round(total_time, 3),
                "config": {
                    "candidate_k": args.candidate_k,
                    "final_k": args.final_k,
                    "reranker_hidden_dim": args.reranker_hidden_dim,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                },
                "epochs": history,
            },
            f,
            indent=2,
        )

    print(f"Stage-2 Reranker training complete! Total time: {total_time:.2f}s.")
    print(f"Saved history log to '{history_file}'.")


if __name__ == "__main__":
    main()
