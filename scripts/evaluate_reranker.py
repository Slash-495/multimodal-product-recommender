import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch

from src.features.text_features import MovieTitleTextEmbedder
from src.models.stage2_reranker import Stage2Reranker
from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.reranker_pipeline import extract_reranker_features
from src.utils.config import DEFAULT_CONFIG
from src.utils.metrics import (
    compute_user_mrr_at_k,
    compute_user_ndcg_at_k,
    compute_user_recall_at_k,
    filter_seen_candidates,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Stage-2 Reranker on Held-out MovieLens Test Data")
    parser.add_argument("--data-path", type=str, default="data/processed/movielens", help="Path to processed MovieLens data directory")
    parser.add_argument("--retriever-checkpoint", type=str, default="checkpoints/movielens/hard_negative_two_tower_epoch_1.pt", help="Path to authoritative Two-Tower retriever checkpoint")
    parser.add_argument("--reranker-checkpoint", type=str, default="checkpoints/movielens/stage2_reranker_epoch_1.pt", help="Path to trained Stage-2 Reranker checkpoint")
    parser.add_argument("--candidate-k", type=int, default=50, help="Candidate pool K retrieved by FAISS for Stage-2 reranking")
    parser.add_argument("--final-k", type=int, default=10, help="Final top K cutoff for evaluation metrics")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    parser.add_argument("--output-dir", type=str, default="outputs/movielens", help="Directory to save evaluation results")
    return parser.parse_args()


def evaluate_stage2_reranker(
    two_tower: TwoTowerModel,
    reranker: Stage2Reranker,
    data_path: Path,
    device: torch.device,
    candidate_k: int = 50,
    final_k: int = 10,
) -> Tuple[Dict[str, float], float]:
    """
    Offline evaluation of Two-Tower FAISS retrieval + Stage-2 Reranker over held-out test data.
    """
    two_tower.eval()
    reranker.eval()

    # Load data
    train_inter = pd.read_csv(data_path / "train_interactions.csv")
    test_inter = pd.read_csv(data_path / "test_interactions.csv")
    user_df = pd.read_csv(data_path / "user_features.csv")
    biz_df = pd.read_csv(data_path / "business_features.csv")

    meta_file = data_path / "movie_metadata.csv"
    meta_df = pd.read_csv(meta_file)
    meta_df["movieId"] = meta_df["movieId"].astype(str)

    embedder = MovieTitleTextEmbedder(tfidf_max_features=5000, svd_components=64, random_state=42)
    text_embs = embedder.fit_transform(meta_df["title"].tolist())
    text_map = {mid: emb for mid, emb in zip(meta_df["movieId"], text_embs)}

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

    all_candidate_ids = biz_df["business_id"].tolist()
    cat_cols = sorted([col for col in biz_df.columns if col.startswith("cat_")])[:10]

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

    # 1. Encode all candidate items once
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

    # 2. Build FAISS index
    faiss_index = FAISSIndex(embedding_dim=128)
    faiss_index.build(all_candidate_embs.cpu().numpy().astype(np.float32))
    candidate_id_arr = np.array(all_candidate_ids)

    # 3. User sets
    train_seen: Dict[str, Set[str]] = train_inter.groupby("user_id")["business_id"].apply(set).to_dict()
    test_positives: Dict[str, Set[str]] = test_inter.groupby("user_id")["business_id"].apply(set).to_dict()

    evaluated_user_ids = [uid for uid in test_positives if uid in user2idx]
    user_df_map = user_df.set_index("user_id").to_dict(orient="index")

    recall_10_list = []
    recall_50_list = []
    ndcg_10_list = []
    mrr_10_list = []

    eval_start = time.time()

    # 4. Evaluation Loop
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
            u_emb = two_tower.encode_user(u_ids_dev, u_stats_dev)

        seen_set = train_seen.get(uid, set())
        gt_set = test_positives[uid]

        fetch_k = max(candidate_k + len(seen_set), 100)
        fetch_k = min(fetch_k, len(all_candidate_ids))

        scores, indices = faiss_index.search(u_emb.cpu().numpy().astype(np.float32), k=fetch_k)
        retrieved_mids = candidate_id_arr[indices[0]].tolist()
        retrieved_scores = scores[0].tolist()

        # Candidate filtering (remove seen items)
        filt_mids, _ = filter_seen_candidates(retrieved_mids, retrieved_scores, seen_set, top_k=candidate_k)

        if not filt_mids:
            continue

        # Extract features for candidate pool
        cand_b_stats = []
        cand_c_feats = []
        cand_t_feats = []
        cand_m_ids = []

        for mid in filt_mids:
            bf, cf, tf = candidate_tensor_dict[mid]
            cand_m_ids.append(int(bf[0]))
            cand_b_stats.append(bf[1:])
            cand_c_feats.append(cf)
            cand_t_feats.append(tf)

        b_stats_t = torch.tensor(cand_b_stats, dtype=torch.float32, device=device)
        c_feats_t = torch.tensor(cand_c_feats, dtype=torch.float32, device=device)
        t_feats_t = torch.tensor(np.array(cand_t_feats), dtype=torch.float32, device=device)
        m_ids_t = torch.tensor(cand_m_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            cand_item_embs = two_tower.encode_item(m_ids_t, b_stats_t, c_feats_t, text_features=t_feats_t)

            feat_matrix = extract_reranker_features(
                user_emb=u_emb,
                user_stats=u_stats_dev,
                item_embs=cand_item_embs,
                item_stats=b_stats_t,
                category_feats=c_feats_t,
                text_embs=t_feats_t,
            )

            rerank_scores = reranker(feat_matrix).flatten().cpu().numpy()

        sorted_indices = np.argsort(-rerank_scores)
        reranked_mids = [filt_mids[idx] for idx in sorted_indices]

        # Compute metrics on reranked list
        r10 = compute_user_recall_at_k(reranked_mids, gt_set, k=10)
        r50 = compute_user_recall_at_k(reranked_mids, gt_set, k=50)
        n10 = compute_user_ndcg_at_k(reranked_mids, gt_set, k=10)
        m10 = compute_user_mrr_at_k(reranked_mids, gt_set, k=10)

        recall_10_list.append(r10)
        recall_50_list.append(r50)
        ndcg_10_list.append(n10)
        mrr_10_list.append(m10)

    eval_time = time.time() - eval_start

    metrics_summary = {
        "recall@10": round(float(np.mean(recall_10_list)), 6),
        "recall@50": round(float(np.mean(recall_50_list)), 6),
        "ndcg@10": round(float(np.mean(ndcg_10_list)), 6),
        "mrr@10": round(float(np.mean(mrr_10_list)), 6),
        "num_evaluated_users": len(evaluated_user_ids),
        "evaluation_time_seconds": round(eval_time, 2),
        "ms_per_user": round((eval_time / len(evaluated_user_ids)) * 1000.0, 3) if evaluated_user_ids else 0.0,
    }

    return metrics_summary, eval_time


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    retriever_chk_path = Path(args.retriever_checkpoint)
    reranker_chk_path = Path(args.reranker_checkpoint)

    device = torch.device(args.device)
    print(f"Loading retriever checkpoint from '{retriever_chk_path}'...")
    config = DEFAULT_CONFIG.copy()
    config["model"]["item_tower"]["use_text_features"] = True
    config["model"]["item_tower"]["text_embedding_dim"] = 64
    config["model"]["item_tower"]["fusion_type"] = "concat"

    two_tower = TwoTowerModel(config["model"]).to(device)
    chk_t = torch.load(retriever_chk_path, map_location=device)
    two_tower.load_state_dict(chk_t["model_state_dict"] if "model_state_dict" in chk_t else chk_t)

    print(f"Loading reranker checkpoint from '{reranker_chk_path}'...")
    reranker = Stage2Reranker(input_dim=33, hidden_dim=64).to(device)
    chk_r = torch.load(reranker_chk_path, map_location=device)
    reranker.load_state_dict(chk_r["model_state_dict"] if "model_state_dict" in chk_r else chk_r)

    print(f"Evaluating Stage-2 Reranker over {55404} warm-start test users...")
    metrics, eval_time = evaluate_stage2_reranker(
        two_tower=two_tower,
        reranker=reranker,
        data_path=data_path,
        device=device,
        candidate_k=args.candidate_k,
        final_k=args.final_k,
    )

    print("\n=== STAGE-2 RERANKER EVALUATION METRICS ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 5. Build 3-Way Comparison JSON Artifact
    base_metrics = {"model": "Original Baseline", "epoch": 1, "recall@10": 0.0115, "recall@50": 0.0515, "ndcg@10": 0.0050, "mrr@10": 0.0031}
    hard_metrics = {"model": "Hard-Negative Content-Concat", "epoch": 1, "recall@10": 0.0119, "recall@50": 0.0533, "ndcg@10": 0.0052, "mrr@10": 0.0033}
    rerank_metrics = {"model": "Hard-Negative Concat + Stage-2 Reranker", "epoch": 1, "recall@10": metrics["recall@10"], "recall@50": metrics["recall@50"], "ndcg@10": metrics["ndcg@10"], "mrr@10": metrics["mrr@10"]}

    comparisons = {}
    for m in ["recall@10", "recall@50", "ndcg@10", "mrr@10"]:
        h_val = hard_metrics[m]
        r_val = rerank_metrics[m]
        abs_d = r_val - h_val
        rel_p = (abs_d / h_val) * 100.0 if h_val > 0 else 0.0
        comparisons[m] = {
            "hard_negative": h_val,
            "reranked": r_val,
            "abs_improvement": round(abs_d, 6),
            "rel_improvement_pct": round(rel_p, 2),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comp_file = output_dir / "reranker_comparison.json"

    with open(comp_file, "w") as f:
        json.dump(
            {
                "experiment": "stage2_reranker_comparison",
                "candidate_k": args.candidate_k,
                "final_k": args.final_k,
                "models": {
                    "baseline_epoch_1": base_metrics,
                    "hard_negative_epoch_1": hard_metrics,
                    "reranked_epoch_1": rerank_metrics,
                },
                "improvements_vs_hard_negative": comparisons,
                "evaluation_metadata": metrics,
            },
            f,
            indent=2,
        )

    print(f"\nSaved reranker comparison artifact to '{comp_file}'.")


if __name__ == "__main__":
    main()
