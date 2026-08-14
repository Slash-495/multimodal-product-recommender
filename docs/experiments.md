# Experimental Progression & Research Log

This document records the sequential hypothesis-driven experiments conducted throughout the development of the multimodal product recommender system.

---

## Experiment 1: Baseline Two-Tower Model Setup

- **Hypothesis**: A dual-encoder (User Tower + Item Tower) trained with symmetric InfoNCE loss on collaborative interaction data can learn dense 128-dimensional representations suitable for fast inner-product retrieval.
- **Implementation**: `src/models/two_tower.py`, `src/trainers/trainer.py`. Trained for 5 epochs on MovieLens warm-start benchmark (250,230 interactions).
- **Result**: Validation loss reached its minimum at Epoch 1 (`5.4789`). Evaluation across 55,404 warm-start test users yielded Recall@10 = 0.0115, Recall@50 = 0.0515, NDCG@10 = 0.0050, MRR@10 = 0.0031.
- **Decision**: Adopt Epoch 1 as the authoritative baseline (`baseline_two_tower_epoch_1.pt`).

---

## Experiment 2: FAISS Sub-Millisecond Candidate Retrieval Integration

- **Hypothesis**: Pre-indexing L2-normalized 128-d item embeddings in an exact FAISS `IndexFlatIP` vector index enables fast sub-millisecond full-catalog ($20,822$ candidate items) retrieval without metric degradation.
- **Implementation**: `src/retrieval/faiss_index.py`, `src/retrieval/retriever.py`.
- **Result**: Search latency per user was measured at < 0.1 ms on CPU, returning identical ranking lists to exact PyTorch matrix multiplication.
- **Decision**: Adopt FAISS `IndexFlatIP` as the standard retrieval engine across all evaluation pipelines.

---

## Experiment 3: Movie Title Text Features (TF-IDF + TruncatedSVD)

- **Hypothesis**: Incorporating dense semantic representations of movie catalog titles via TF-IDF vectorization (5,000 max features) and TruncatedSVD dimensionality reduction (64 components) will enrich Item Tower representations.
- **Implementation**: `src/features/text_features.py`, `src/models/towers/item_tower.py` (`fusion_type="concat"`).
- **Result**: Trained for 5 epochs. Epoch 1 reached Recall@10 = 0.0110, Recall@50 = 0.0527, NDCG@10 = 0.0050, MRR@10 = 0.0033. Recall@50 improved (+2.33%), but Recall@10 slightly dropped (-4.35%).
- **Decision**: Retain content concatenation as an experimental baseline (`content_two_tower_epoch_1.pt`).

---

## Experiment 4: Learned Gated Content Fusion

- **Hypothesis**: Replacing simple feature concatenation with an elementwise adaptive gating network $\mathbf{g} = \sigma(W [\mathbf{e}_{\text{base}}, \mathbf{e}_{\text{text}}] + b)$ will allow the item tower to dynamically filter text noise for warm-start items.
- **Implementation**: `src/models/towers/item_tower.py` (`fusion_type="gated"`).
- **Result**: Trained for 5 epochs (`gated_two_tower_epoch_1.pt`). Retrieval performance degraded significantly: Recall@10 dropped to 0.0100 (-13.04% vs baseline), NDCG@10 dropped to 0.0045 (-10.00%).
- **Decision**: **Negative Result**. Reject gated fusion for future models.

---

## Experiment 5: Hard-Negative Mining with InfoNCE Loss

- **Hypothesis**: In-batch negative sampling provides negatives that are too easy for mature embeddings. Dynamically mining $K_{\text{hard}}=5$ hard negative items per positive pair from top-$M$ ($M=50$) candidate competitors will force sharper decision boundaries.
- **Implementation**: `src/training/negative_sampling.py`, `src/utils/losses.py` (`HardNegativeInfoNCELoss`).
- **Result**: Trained Content-Concat model with hard negatives (`hard_negative_two_tower_epoch_1.pt`). Recall@10 reached 0.0119 (+8.18% over in-batch content concat, +3.48% over baseline), Recall@50 reached 0.0533 (+3.50% over baseline).
- **Decision**: Adopt Hard-Negative Mining as the primary training objective.

---

## Experiment 6: Stage-2 Reranking System

- **Hypothesis**: A Stage-2 MLP reranker trained on 33-d engineered candidate features (similarities, interaction summaries, statistics) using pairwise Bayesian Personalized Ranking (BPR) loss will refine FAISS top-50 candidates.
- **Implementation**: `src/models/stage2_reranker.py`, `src/retrieval/reranker_pipeline.py`, `scripts/train_reranker.py`.
- **Result**: Evaluated over 55,404 test users (`reranker_comparison.json`). Recall@10 dropped from 0.0119 to 0.0103 (-13.17%), NDCG@10 dropped to 0.0048 (-8.46%).
- **Decision**: **Negative Result**. Reject Stage-2 MLP reranking. Retain single-stage FAISS retrieval.

---

## Experiment 7: Controlled Retrieval Ablation Study & Final Model Selection

- **Hypothesis**: Disentangling text features and hard-negative mining in a 6-model controlled ablation will isolate the true contribution of each component.
- **Implementation**: Trained Model D (Hard-Negative Without Text) for 5 epochs on CPU (`hard_negative_no_text_epoch_1.pt`). Evaluated all 6 models over 55,404 test users.
- **Result**:
  - Model D (Hard-Negative No Text): Recall@10 = **0.011987**, Recall@50 = **0.053355**, NDCG@10 = **0.005227**, MRR@10 = 0.003248.
  - Model E (Hard-Negative Concat): Recall@10 = 0.011900, Recall@50 = 0.053300, NDCG@10 = 0.005200, MRR@10 = 0.003300.
- **Decision**: **Select Model D (Hard-Negative Two-Tower Without Text) as the Final Authoritative Recommender System**.
