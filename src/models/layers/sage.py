import torch
import torch.nn as nn

class SAGEConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # GraphSAGE concatenates the node's own features with the aggregated neighbor features,
        # so the linear transformation takes in 2 * in_channels.
        self.lin = nn.Linear(in_channels * 2, out_channels)
        self.act = nn.ReLU()
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        """
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        
        # 1. Gather neighbor features
        # src_features: [num_edges, in_channels]
        src_features = x[src]
        
        # 2. Aggregate neighbor features (Mean aggregation)
        # Initialize an empty tensor for aggregated features: [num_nodes, in_channels]
        aggr_out = torch.zeros((num_nodes, x.size(1)), device=x.device, dtype=x.dtype)
        
        # We scatter-reduce (mean) the src_features into the dst nodes
        # dst.unsqueeze(1).expand(-1, in_channels) creates the index tensor matching src_features shape
        aggr_out.scatter_reduce_(dim=0, 
                                 index=dst.unsqueeze(1).expand(-1, x.size(1)), 
                                 src=src_features, 
                                 reduce="mean", 
                                 include_self=False)
        
        # 2b. Zero-degree fix: self-loop
        # scatter_reduce_ with include_self=False silently leaves zero-degree
        # nodes as all-zero rows (PyTorch's identity value for "mean" reduction).
        # We explicitly override this: a node with no incoming neighbors uses its
        # own features as its "neighbor average" (self-loop), instead of a zero
        # vector. This is a deliberate design choice for isolated nodes, not
        # an accident of the reduction's default behavior.
        #
        # IMPORTANT: we use torch.where here (out-of-place) instead of masked
        # in-place assignment (aggr_out[mask] = ...), because in-place edits to
        # a tensor that autograd is already tracking as the output of
        # scatter_reduce_ break gradient computation (raises a "version counter"
        # RuntimeError during loss.backward()). torch.where builds a brand-new
        # tensor instead of mutating aggr_out's memory, which keeps autograd's
        # recorded history intact.
        degree = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        zero_degree_mask = (degree == 0).unsqueeze(1)  # [num_nodes, 1] to broadcast against [num_nodes, in_channels]
        aggr_out = torch.where(zero_degree_mask, x, aggr_out)
        
        # 3. Concatenate self features with aggregated neighbor features
        # concat_out: [num_nodes, in_channels * 2]
        concat_out = torch.cat([x, aggr_out], dim=-1)
        
        # 4. Apply linear transformation and non-linearity
        # out: [num_nodes, out_channels]
        out = self.act(self.lin(concat_out))
        
        # Optional: GraphSAGE often L2-normalizes the output embeddings
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        
        return out

class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        # Binary classification output (logits)
        out = self.classifier(x)
        return out