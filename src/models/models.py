import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.gat import GATConv
from .layers.camouflage_gat import CamouflageGATConv
from .layers.sage import SAGEConv

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
        out = self.classifier(x)
        return out

class FraudCamouflageGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.conv1 = CamouflageGATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.conv2 = CamouflageGATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        out = self.classifier(x)
        return out

    def auxiliary_node_scores(self) -> torch.Tensor:
        """
        Returns the fraud-likelihood scores predicted by each
        CamouflageGATConv layer's internal `score_head`, averaged across
        heads and layers into a single per-node score.

        Must be called AFTER `forward()` in the same iteration, since it
        reads `last_node_scores`, which each layer caches during its own
        forward pass.

        This is what train.py uses to build the auxiliary supervised loss
        that actually teaches the camouflage-resistance mechanism what
        "fraud-likely" means -- without this supervision, the label-aware
        trust signal in CamouflageGATConv would be untrained noise rather
        than a genuine defense.
        """
        # Each layer's last_node_scores has shape [num_nodes, heads, 1].
        # Average over heads to get one score per node per layer, then
        # average over layers to get a single node-level auxiliary score.
        scores1 = self.conv1.last_node_scores.mean(dim=1).squeeze(-1)  # [num_nodes]
        scores2 = self.conv2.last_node_scores.mean(dim=1).squeeze(-1)  # [num_nodes]
        return (scores1 + scores2) / 2.0