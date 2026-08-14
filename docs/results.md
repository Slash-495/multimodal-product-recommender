# Benchmark Results & Model Selection

This document provides the complete empirical performance comparison across all model architectures and training strategies evaluated on the **MovieLens Warm-Start Retrieval Benchmark**.

---

## 1. Complete Ablation Comparison Table

Evaluated over all **55,404 warm-start test users** across **20,822 candidate catalog movies**:

| Rank | Model ID | Architecture / Training Strategy | Recall@10 | Recall@50 | NDCG@10 | MRR@10 | Training Time | Selected Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Model D** | **Hard-Negative Two-Tower (No Text)** | **0.011987** | **0.053355** | **0.005227** | 0.003248 | ~35 min | **FINAL SELECTED MODEL** |
| **2** | **Model E** | **Hard-Negative Content-Concat** | 0.011900 | 0.053300 | 0.005200 | **0.003300** | ~36 min | Authoritative Baseline |
| **3** | **Model A** | **Original Two-Tower Baseline (In-Batch)** | 0.011500 | 0.051500 | 0.005000 | 0.003100 | ~29 min | Initial Baseline |
| **4** | **Model B** | **Content-Concat (In-Batch InfoNCE)** | 0.011000 | 0.052700 | 0.005000 | **0.003300** | ~30 min | Experimental |
| **5** | **Model F** | **Hard-Neg Concat + Stage-2 Reranker** | 0.010333 | 0.053252 | 0.004760 | 0.003114 | ~36 min | Experimental (Negative Result) |
| **6** | **Model C** | **Content-Gated Two-Tower (In-Batch)** | 0.010000 | 0.047400 | 0.004500 | 0.002900 | ~30 min | Experimental (Negative Result) |

---

## 2. Relative Improvements vs. Original Baseline (Model A)

| Model ID | Model Strategy | Recall@10 (Abs / Rel %) | Recall@50 (Abs / Rel %) | NDCG@10 (Abs / Rel %) | MRR@10 (Abs / Rel %) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Model D** | **Hard-Negative No Text** | **+0.000487 (+4.23%)** | **+0.001855 (+3.60%)** | **+0.000227 (+4.54%)** | **+0.000148 (+4.77%)** |
| **Model E** | **Hard-Negative Concat** | +0.000400 (+3.48%) | +0.001800 (+3.50%) | +0.000200 (+4.00%) | +0.000200 (+6.45%) |
| **Model B** | **Content-Concat** | -0.000500 (-4.35%) | +0.001200 (+2.33%) | 0.000000 (0.00%) | +0.000200 (+6.45%) |
| **Model F** | **Stage-2 Reranker** | -0.001167 (-10.15%) | +0.001752 (+3.40%) | -0.000240 (-4.80%) | +0.000014 (+0.45%) |
| **Model C** | **Content-Gated** | -0.001500 (-13.04%) | -0.004100 (-7.96%) | -0.000500 (-10.00%) | -0.000200 (-6.45%) |

---

## 3. Key Research Insights & Negative Findings

1. **Hard-Negative Mining is the Dominant Optimization Signal**:
   - Explicit hard negative sampling ($K_{\text{hard}}=5, M=50$) provided the single largest performance jump across all metrics (+4.23% in Recall@10, +3.60% in Recall@50).
2. **Text Features do not Add Value over Hard-Negative ID Embeddings on Warm-Start Users**:
   - Model D (without text) achieved a higher Recall@10 (0.011987) and Recall@50 (0.053355) than Model E with text (0.011900 and 0.053300). On warm-start interactions, ID embeddings trained with hard negatives absorb almost all collaborative retrieval signal.
3. **Learned Gating Mechanism is Harmful**:
   - Learned elementwise gating added optimization complexity and underperformed simple feature concatenation across all metrics (Recall@10 dropped to 0.010000).
4. **Stage-2 Reranking Underperformed Single-Stage Vector Retrieval**:
   - Reranking FAISS top-50 candidates using a 33-d MLP with BPR loss distorted global vector space geometry, reducing Recall@10 from 0.011900 to 0.010333 (-13.17%). Single-stage FAISS retrieval remains the superior production architecture.
