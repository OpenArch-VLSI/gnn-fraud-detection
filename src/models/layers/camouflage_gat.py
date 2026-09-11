import torch
import torch.nn as nn
import torch.nn.functional as F

class CamouflageGATConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, 
                 concat: bool = True, dropout: float = 0.0, add_self_loops: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        
        # --- Camouflage resistance (label-aware neighbor trust) ---
        # Raw feature similarity is NOT used here. Fraud rings often
        # deliberately shape their raw features to look similar to
        # legitimate ones ("camouflage") -- so trusting a neighbor because
        # it looks similar can reward exactly the camouflage pattern this
        # module is supposed to defend against. Instead, following the
        # CARE-GNN approach (Dou et al., 2020, "Enhancing Graph Neural
        # Network-based Fraud Detectors against Camouflaged Fraudsters"),
        # trust between two nodes is based on whether the model's OWN
        # current belief about their fraud-likelihood agrees -- a
        # label-supervised signal, not an incidental feature similarity.
        #
        # `score_head` is a small per-node MLP that predicts a
        # fraud-likelihood score from a node's own projected features
        # (no graph structure), trained jointly via the auxiliary loss
        # this layer exposes through `self.last_node_scores`. Two nodes
        # whose predicted scores are close are treated as more mutually
        # trustworthy neighbors; nodes whose predicted fraud-likelihoods
        # sharply disagree are downweighted, since that disagreement is
        # exactly the signature of a camouflaged edge (e.g. a fraud node
        # attached to a normal-looking one, or vice versa).
        self.score_head = nn.Linear(out_channels, 1)
        # Learnable temperature controlling how sharply score disagreement
        # is penalized (kept positive via softplus, same pattern as before).
        self.trust_weight = nn.Parameter(torch.tensor(1.0))
        
        if concat:
            self.bias = nn.Parameter(torch.empty(heads * out_channels))
        else:
            self.bias = nn.Parameter(torch.empty(out_channels))
            
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        
        if self.add_self_loops:
            self_loop_edges = torch.arange(num_nodes, device=x.device, dtype=torch.long)
            self_loop_edges = self_loop_edges.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loop_edges], dim=1)

        x_proj = self.lin(x).view(-1, self.heads, self.out_channels)

        alpha_src = (x_proj * self.att_src).sum(dim=-1, keepdim=True)
        alpha_dst = (x_proj * self.att_dst).sum(dim=-1, keepdim=True)

        src, dst = edge_index[0], edge_index[1]

        # Base GAT attention
        alpha = alpha_src[src] + alpha_dst[dst]
        alpha = self.leaky_relu(alpha)
        
        # --- Camouflage Resistance Module (label-aware trust) ---
        # Predict a per-node, per-head fraud-likelihood score directly from
        # this layer's own projected features. This is intentionally cheap
        # (a single linear layer) since its only job is to expose a
        # label-correlated signal for neighbor trust, not to be the final
        # classifier -- the model's real classifier head still sits on top
        # of the full network in models.py.
        node_scores = self.score_head(x_proj)  # [num_nodes, heads, 1]
        # Expose raw scores for an optional auxiliary loss term added in
        # the training loop (see train.py): supervising these scores
        # directly with the true fraud labels is what makes "agreement
        # between scores" a genuinely label-aware trust signal rather than
        # an untrained, meaningless one early in training.
        self.last_node_scores = node_scores

        # Two nodes are "mutually trustworthy" if the model's current
        # belief about their fraud-likelihood agrees -- i.e. small
        # |score_src - score_dst|. This is the opposite failure mode of
        # raw feature similarity: a fraud node camouflaged to look
        # feature-similar to a normal node will still, once the score head
        # is even partially trained, tend to disagree in PREDICTED fraud
        # likelihood with genuinely normal neighbors, so this signal
        # degrades much more gracefully under camouflage.
        score_diff = torch.abs(node_scores[src] - node_scores[dst])  # [num_edges, heads, 1]
        trust = torch.exp(-score_diff)  # close scores -> trust near 1; disagreement -> trust -> 0

        trust_weight_positive = F.softplus(self.trust_weight)
        alpha = alpha + trust_weight_positive * trust
        # -------------------------------------

        # Softmax over neighborhood
        alpha_max = torch.zeros((num_nodes, self.heads, 1), device=x.device, dtype=x.dtype)
        alpha_max.scatter_reduce_(0, dst.view(-1, 1, 1).expand(-1, self.heads, 1), alpha, reduce="amax", include_self=False)
        alpha_exp = torch.exp(alpha - alpha_max[dst])
        
        alpha_sum = torch.zeros((num_nodes, self.heads, 1), device=x.device, dtype=x.dtype)
        alpha_sum.scatter_add_(0, dst.view(-1, 1, 1).expand(-1, self.heads, 1), alpha_exp)
        
        alpha_softmax = alpha_exp / (alpha_sum[dst] + 1e-16)
        
        alpha_softmax = F.dropout(alpha_softmax, p=self.dropout, training=self.training)
        
        self._alpha = alpha_softmax

        messages = x_proj[src] * alpha_softmax
        
        out = torch.zeros((num_nodes, self.heads, self.out_channels), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand(-1, self.heads, self.out_channels), messages)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        out = out + self.bias
        
        return out