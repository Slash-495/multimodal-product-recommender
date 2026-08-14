import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch

from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_index import FAISSIndex
from src.utils.config import DEFAULT_CONFIG
from src.utils.metrics import (
    compute_user_mrr_at_k,
    compute_user_ndcg_at_k,
    compute_user_recall_at_k,
    filter_seen_candidates,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Two-Tower Recommendation Model on Held-out MovieLens Test Data")
    parser.add_argument("--data-path", type=str, default="data/processed/movielens", help="Path to processed MovieLens data directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained PyTorch model checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    parser.add_argument("--top-k", type=int, default=50, help="Maximum top K items for evaluation metrics")
    parser.add_argument("--output-dir", type=str, default="outputs/movielens", help="Directory to save evaluation results")
    parser.add_argument("--num-samples-to-log", type=int, default=10, help="Number of sample user recommendation cards to log")
    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the most recent checkpoint in checkpoint_dir if not explicitly specified."""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory '{checkpoint_dir}' does not exist.")
    
    pt_files = sorted(list(checkpoint_dir.glob("*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No checkpoint (.pt) files found in '{checkpoint_dir}'. Train model first.")
    
    # Pick last sorted file (highest epoch)
    return pt_files[-1]


def evaluate_retrieval(
    model: TwoTowerModel,
    data_path: Path,
    device: torch.device,
    top_k: int = 50,
    num_samples_to_log: int = 10,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Perform FAISS offline retrieval evaluation with candidate filtering and cold-start handling.
    """
    model.eval()

    # Load data
    train_inter = pd.read_csv(data_path / "train_interactions.csv")
    test_inter = pd.read_csv(data_path / "test_interactions.csv")
    user_df = pd.read_csv(data_path / "user_features.csv")
    biz_df = pd.read_csv(data_path / "business_features.csv")

    meta_file = data_path / "movie_metadata.csv"
    title_map = {}
    if meta_file.exists():
        meta_df = pd.read_csv(meta_file)
        meta_df["movieId"] = meta_df["movieId"].astype(str)
        title_map = meta_df.set_index("movieId")["title"].to_dict()

    with open(data_path / "user2idx.json", "r") as f:
        user2idx = json.load(f)
    with open(data_path / "business2idx.json", "r") as f:
        business2idx = json.load(f)

    user_df["user_id"] = user_df["user_id"].astype(str)
    biz_df["business_id"] = biz_df["business_id"].astype(str)
    train_inter["user_id"] = train_inter["user_id"].astype(str)
    train_inter["business_id"] = train_inter["business_id"].astype(str)
    test_inter["user_id"] = test_inter["user_id"].astype(str)
    test_inter["business_id"] = test_inter["business_id"].astype(str)

    from src.features.text_features import MovieTitleTextEmbedder

    use_text = getattr(model.item_tower, "use_text_features", False)
    text_map = {}
    if use_text:
        meta_file = data_path / "movie_metadata.csv"
        if not meta_file.exists():
            raise FileNotFoundError(f"Movie metadata file required for text features not found at '{meta_file}'")
        meta_df = pd.read_csv(meta_file)
        meta_df["movieId"] = meta_df["movieId"].astype(str)

        text_dim = getattr(model.item_tower, "text_embedding_dim", 64)
        embedder = MovieTitleTextEmbedder(tfidf_max_features=5000, svd_components=text_dim, random_state=42)
        text_embs = embedder.fit_transform(meta_df["title"].tolist())
        text_map = {mid: emb for mid, emb in zip(meta_df["movieId"], text_embs)}

    # 1. Encode all eligible movies once
    cat_cols = sorted([col for col in biz_df.columns if col.startswith("cat_")])
    cat_cols = cat_cols[:10]

    all_movie_ids = biz_df["business_id"].tolist()
    movie_tensor_list = []

    for _, row in biz_df.iterrows():
        mid_str = str(row["business_id"])
        m_idx = float(business2idx.get(mid_str, 0))
        rc = float(row.get("review_count", 0.0))
        st = float(row.get("stars", 0.0))

        bf_vec = [m_idx, rc, st]
        cf_vec = [float(row[col]) for col in cat_cols]
        while len(cf_vec) < 10:
            cf_vec.append(0.0)

        tf_vec = None
        if use_text:
            if mid_str in text_map:
                tf_vec = text_map[mid_str]
            else:
                tf_vec = np.zeros(text_dim, dtype=np.float32)

        movie_tensor_list.append((bf_vec, cf_vec, tf_vec))

    # Batch encode movies
    batch_size = 512
    all_movie_embeddings = []
    with torch.no_grad():
        for i in range(0, len(movie_tensor_list), batch_size):
            batch_slice = movie_tensor_list[i : i + batch_size]
            bf_batch = torch.tensor([b[0] for b in batch_slice], dtype=torch.float32, device=device)
            cf_batch = torch.tensor([b[1] for b in batch_slice], dtype=torch.float32, device=device)

            m_ids = bf_batch[:, 0].long()
            b_stats = bf_batch[:, 1:]

            tf_batch = None
            if use_text:
                tf_batch = torch.tensor([b[2] for b in batch_slice], dtype=torch.float32, device=device)

            embs = model.encode_item(m_ids, b_stats, cf_batch, text_features=tf_batch)
            all_movie_embeddings.append(embs.cpu().numpy().astype(np.float32))

    concat_movie_embeddings = np.vstack(all_movie_embeddings)
    embedding_dim = concat_movie_embeddings.shape[1]

    # 2. Build FAISS index
    faiss_index = FAISSIndex(embedding_dim=embedding_dim)
    faiss_index.build(concat_movie_embeddings)
    movie_id_arr = np.array(all_movie_ids)

    # 3. Build user sets: train seen and test positives
    train_seen: Dict[str, Set[str]] = train_inter.groupby("user_id")["business_id"].apply(set).to_dict()
    test_positives: Dict[str, Set[str]] = test_inter.groupby("user_id")["business_id"].apply(set).to_dict()

    total_test_users = len(test_positives)
    cold_start_users = [uid for uid in test_positives if uid not in user2idx]
    evaluated_user_ids = [uid for uid in test_positives if uid in user2idx]

    user_df_map = user_df.set_index("user_id").to_dict(orient="index")

    # 4. Perform evaluation loop over evaluated users
    recall_10_list = []
    recall_k_list = []
    ndcg_10_list = []
    mrr_10_list = []

    sample_cards = []

    for uid in evaluated_user_ids:
        u_idx = float(user2idx[uid])
        u_data = user_df_map.get(uid, {})
        rc = float(u_data.get("review_count", 0.0))
        avg_s = float(u_data.get("average_stars", 0.0))
        y_days = float(u_data.get("yelping_days", 0.0))

        u_tensor = torch.tensor([[u_idx, rc, avg_s, y_days]], dtype=torch.float32, device=device)
        u_ids_dev = u_tensor[:, 0].long()
        u_stats_dev = u_tensor[:, 1:]

        with torch.no_grad():
            u_emb = model.encode_user(u_ids_dev, u_stats_dev).cpu().numpy().astype(np.float32)

        seen_set = train_seen.get(uid, set())
        gt_set = test_positives[uid]

        # Request extra candidates to account for seen-item filtering
        fetch_k = max(top_k + len(seen_set), 100)
        fetch_k = min(fetch_k, len(all_movie_ids))

        scores, indices = faiss_index.search(u_emb, k=fetch_k)
        retrieved_ids = movie_id_arr[indices[0]].tolist()
        retrieved_scores = scores[0].tolist()

        # Candidate filtering (remove seen training items)
        filt_ids, filt_scores = filter_seen_candidates(retrieved_ids, retrieved_scores, seen_set, top_k=top_k)

        # Compute metrics
        r10 = compute_user_recall_at_k(filt_ids, gt_set, k=10)
        rk = compute_user_recall_at_k(filt_ids, gt_set, k=top_k)
        n10 = compute_user_ndcg_at_k(filt_ids, gt_set, k=10)
        m10 = compute_user_mrr_at_k(filt_ids, gt_set, k=10)

        recall_10_list.append(r10)
        recall_k_list.append(rk)
        ndcg_10_list.append(n10)
        mrr_10_list.append(m10)

        # Log sample recommendation card
        if len(sample_cards) < num_samples_to_log:
            rec_titles = [title_map.get(mid, f"Movie {mid}") for mid in filt_ids[:10]]
            sample_cards.append(
                {
                    "userId": uid,
                    "recommended_movie_ids": filt_ids[:10],
                    "recommended_titles": rec_titles,
                    "similarity_scores": [round(s, 4) for s in filt_scores[:10]],
                }
            )

    avg_r10 = float(np.mean(recall_10_list)) if recall_10_list else 0.0
    avg_rk = float(np.mean(recall_k_list)) if recall_k_list else 0.0
    avg_n10 = float(np.mean(ndcg_10_list)) if ndcg_10_list else 0.0
    avg_m10 = float(np.mean(mrr_10_list)) if mrr_10_list else 0.0

    metrics_summary = {
        "recall@10": round(avg_r10, 4),
        f"recall@{top_k}": round(avg_rk, 4),
        "ndcg@10": round(avg_n10, 4),
        "mrr@10": round(avg_m10, 4),
        "num_total_test_users": total_test_users,
        "num_evaluated_users": len(evaluated_user_ids),
        "num_cold_start_users": len(cold_start_users),
        "num_test_interactions": len(test_inter),
    }

    return metrics_summary, sample_cards


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Processed MovieLens directory '{data_path}' does not exist.")
        sys.exit(1)

    if args.checkpoint is None:
        chk_dir = Path("checkpoints/movielens")
        if not chk_dir.exists():
            chk_dir = Path("checkpoints")
        checkpoint_path = find_latest_checkpoint(chk_dir)
    else:
        checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint '{checkpoint_path}' does not exist.")
        sys.exit(1)

    device = torch.device(args.device)
    print(f"Loading checkpoint from '{checkpoint_path}' onto device '{device}'...")

    config = DEFAULT_CONFIG.copy()
    model = TwoTowerModel(config["model"]).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print(f"Running FAISS offline retrieval evaluation on '{data_path}' (Top-K={args.top_k})...")
    metrics_summary, sample_cards = evaluate_retrieval(
        model=model,
        data_path=data_path,
        device=device,
        top_k=args.top_k,
        num_samples_to_log=args.num_samples_to_log,
    )

    metrics_summary["checkpoint"] = str(checkpoint_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_file = out_dir / "evaluation.json"
    cards_file = out_dir / "sample_recommendations.json"

    with open(eval_file, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    with open(cards_file, "w") as f:
        json.dump(sample_cards, f, indent=2)

    print("\n================ EVALUATION SUMMARY ================")
    for k, v in metrics_summary.items():
        print(f"  {k}: {v}")
    print("====================================================")
    print(f"Saved evaluation metrics to '{eval_file}'.")
    print(f"Saved sample recommendations to '{cards_file}'.")


if __name__ == "__main__":
    main()
