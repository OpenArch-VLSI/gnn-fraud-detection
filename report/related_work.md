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

### Our Proposed Twist: Label-Aware Score-Agreement Gating (CamouflageGAT)
Like CARE-GNN, our trust signal is label-aware, but the mechanism is our own and not a port of CARE-GNN's code, and it does not use a reinforcement-learning controller. Each CamouflageGAT layer contains a small score head that predicts a fraud-likelihood score for each node from its own projected features. The score head is supervised by an auxiliary loss computed only on training nodes. For each edge, trust is exp(−|s_src − s_dst|), so neighbors whose predicted scores agree receive more attention. This trust term is added to the GAT attention logits before the softmax, scaled by a learnable, positive weight. We deliberately do not use raw feature similarity: a camouflaged fraud node is built to look similar to normal nodes in raw features, so rewarding that similarity would reward the camouflage itself. 

The contribution to test is whether this gating gives a measurable gain over a from-scratch GAT baseline trained on the same splits and seeds, and whether any gain holds up as synthetic camouflage is injected at inference time (Phase 8). No results for this mechanism exist yet. RL-GNN's reported 0.872 AUROC on IEEE-CIS is an external reference point, not a like-for-like comparison, since splits and preprocessing differ.
