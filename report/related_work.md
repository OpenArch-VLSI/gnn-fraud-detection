# Related Work: Camouflage-Resistant GNNs

This section summarizes key literature in the field of camouflage-resistant Graph Neural Networks (GNNs), particularly focusing on fraud detection, and outlines how our proposed approach differs from existing state-of-the-art methods.

### 1. CARE-GNN (CIKM 2020)
**Mechanism Summary:**
CARE-GNN (Dou et al.) addresses the camouflage problem in review-fraud graphs by filtering neighbors per relation type. It computes a label-aware similarity measure to select which neighbors should be aggregated. To adapt to varying levels of camouflage across different relations and training epochs, CARE-GNN employs a Reinforcement Learning (RL) module that dynamically adjusts the similarity threshold during training.

### 2. PC-GNN (WWW 2021)
**Mechanism Summary:**
PC-GNN (Liu et al.) focuses on both the camouflage and class-imbalance problems simultaneously. It introduces a node-level resampler ("pick and choose") to balance the graph by selecting a subset of nodes and edges. It then uses a label-aware neighbor selector to filter out noisy, camouflaged edges before aggregation, creating a cleaner neighborhood for GNN message passing.

### 3. RL-GNN (Scientific Reports 2025)
**Mechanism Summary:**
RL-GNN applies a Graph Attention Network (GAT) directly to the IEEE-CIS transaction graph. To improve fraud detection performance, it couples the GAT with a Reinforcement Learning controller that optimizes the model's predictive performance (reporting 0.872 AUROC). However, RL-GNN does not explicitly target heterophily or camouflage resistance; rather, it uses the RL agent to tune the learning process directly.

### 4. PROD (Knowledge-Based Systems 2026)
**Mechanism Summary:**
PROD addresses both feature-camouflage and relation-camouflage. It measures the inconsistency between a node's features and its neighborhood features, dropping edges that show high discrepancy. This creates a structurally cleaner graph that makes it difficult for a fraudster to blend into a benign neighborhood by merely establishing connections to normal users.

---

### Our Proposed Twist: Similarity-Gated Attention (CamouflageGAT)
Unlike CARE-GNN and RL-GNN, which rely on complex reinforcement learning controllers to adapt thresholds during training, our approach introduces a straightforward **similarity-gated attention** mechanism directly within the GAT layer. We compute a feature-level cosine similarity score between connected nodes and add this (weighted by a learnable parameter) directly to the unnormalized attention coefficients *before* applying the softmax. 

This explicit cosine-similarity bias forces the attention weights to structurally down-weight edges that connect highly dissimilar nodes—a strong signal of camouflage in transaction graphs where fraudsters link to normal users. By integrating this directly into the message-passing layer and performing ablations against from-scratch GraphSAGE and GAT baselines, we can isolate the performance gains attributable purely to this camouflage-resistant filtering without the overhead of an RL agent.
