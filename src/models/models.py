"""
Graph models built on PyTorch Geometric's message-passing layers.

GraphSAGEModel     2-layer GraphSAGE (PyG SAGEConv, mean aggregation).
FraudGAT           2-layer GAT (PyG GATConv, multi-head attention).
FraudCamouflageGNN GAT whose attention is biased by a label-supervised
                   neighbor-trust signal (see the class docstring).

The hand-written layers in src/models/layers/ are no longer used by these
models; they are kept only as reference implementations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv


class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.3):
        super().__init__()
        # SAGEConv with root weight (PyG default) computes
        # W_self * h_v + W_neigh * mean(h_u), equivalent to the CONCAT
        # formulation W * [h_v || mean(h_u)] up to how the weights are split.
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        return self.classifier(x)


class FraudGAT(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        return self.classifier(x)


class FraudCamouflageGNN(nn.Module):
    """
    GAT with a label-supervised neighbor-trust signal.

    - score_head: a small MLP on each node's own input features (no graph)
      that predicts a fraud-likelihood score s_v. It is trained by an
      auxiliary BCE loss on training seed nodes (see train.py).
    - trust(u, v) = exp(-|s_u - s_v|), in (0, 1]. Neighbors whose predicted
      scores agree get trust near 1.
    - trust is passed to each GATConv as a 1-dim edge feature. GATConv adds a
      learned per-head term that is linear in the trust value to the attention
      logits, before the LeakyReLU and softmax. Its sign is learned, not
      constrained; use trust_gains() after training to check the sign.
    - detach_trust (default True): trust is computed from detached scores, so
      the main loss does not train score_head. score_head then learns only
      from the auxiliary loss and its scores keep their fraud-likelihood
      meaning. detach_trust=False lets the main loss shape the scores too.
      Choose between the two on validation PR-AUC only.
    """

    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 4,
                 dropout: float = 0.3, detach_trust: bool = True):
        super().__init__()
        self.score_head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True,
                             dropout=dropout, edge_dim=1)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False,
                             dropout=dropout, edge_dim=1)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.detach_trust = detach_trust
        self._node_scores = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        node_scores = self.score_head(x).squeeze(-1)  # [num_nodes]
        self._node_scores = node_scores
        trust_scores = node_scores.detach() if self.detach_trust else node_scores
        src, dst = edge_index[0], edge_index[1]
        trust = torch.exp(-(trust_scores[src] - trust_scores[dst]).abs()).unsqueeze(-1)  # [E, 1]

        x = self.conv1(x, edge_index, edge_attr=trust)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr=trust)
        x = F.elu(x)
        x = self.dropout(x)
        return self.classifier(x)

    def auxiliary_node_scores(self) -> torch.Tensor:
        """
        Per-node fraud-likelihood scores from the most recent forward pass.
        Call after forward(); train.py supervises these on seed nodes only.
        """
        if self._node_scores is None:
            raise RuntimeError("auxiliary_node_scores() called before forward().")
        return self._node_scores

    @torch.no_grad()
    def trust_gains(self) -> dict:
        """
        Learned per-head gain k_h on the trust feature for each layer.
        Attention logit contribution for an edge is k_h * trust(u, v); k_h > 0
        means agreeing neighbors receive more attention than disagreeing ones.
        """
        out = {}
        for name, conv in (('conv1', self.conv1), ('conv2', self.conv2)):
            heads = conv.heads
            channels = conv.out_channels
            w = conv.lin_edge.weight.view(heads, channels)   # [heads, C]
            a = conv.att_edge.view(heads, channels)          # [heads, C]
            out[name] = (w * a).sum(dim=-1)                  # [heads]
        return out