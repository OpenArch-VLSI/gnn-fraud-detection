import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.layers.camouflage_gat import CamouflageGATConv

def test_camouflage_gat_conv_shape():
    in_channels = 16
    out_channels = 8
    heads = 4
    num_nodes = 10
    num_edges = 20
    
    x = torch.randn((num_nodes, in_channels))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    conv = CamouflageGATConv(in_channels, out_channels, heads=heads, concat=True)
    out = conv(x, edge_index)
    
    assert out.size() == (num_nodes, heads * out_channels)

def test_camouflage_attention_sums_to_1():
    in_channels = 16
    out_channels = 8
    heads = 4
    num_nodes = 10
    num_edges = 30
    
    x = torch.randn((num_nodes, in_channels))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    conv = CamouflageGATConv(in_channels, out_channels, heads=heads, concat=True, dropout=0.0)
    out = conv(x, edge_index)
    
    alpha = conv._alpha
    dst = torch.cat([edge_index[1], torch.arange(num_nodes, dtype=torch.long)])
    
    alpha_sums = torch.zeros((num_nodes, heads, 1))
    alpha_sums.scatter_add_(0, dst.view(-1, 1, 1).expand(-1, heads, 1), alpha)
    
    assert torch.allclose(alpha_sums, torch.ones_like(alpha_sums), atol=1e-5), "Attention weights do not sum to 1"

def test_camouflage_gradient_flow():
    in_channels = 16
    out_channels = 8
    heads = 4
    num_nodes = 10
    num_edges = 20
    
    x = torch.randn((num_nodes, in_channels), requires_grad=True)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    conv = CamouflageGATConv(in_channels, out_channels, heads=heads, concat=True)
    out = conv(x, edge_index)
    
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert torch.sum(torch.abs(x.grad)) > 0
    assert conv.lin.weight.grad is not None
    assert conv.att_src.grad is not None
    # Trust-term parameters: gradients must reach both the learned trust
    # temperature and the score head that produces the per-node scores.
    assert conv.trust_weight.grad is not None
    assert conv.score_head.weight.grad is not None

def test_camouflage_behavior():
    """
    Trust is based on agreement of the per-node predicted fraud scores,
    not on raw feature similarity. Here the raw features are deliberately
    arranged so that node 2 looks *far* from node 0 in feature space while
    node 1 is close; the score head is set so that node 1's predicted score
    agrees with node 0's and node 2's does not. The score-agreement signal
    must then give node 1's edge more attention.
    """
    in_channels = 4
    out_channels = 4
    heads = 1
    num_nodes = 3

    # Node 0 is target, Node 1 is a score-agreeing neighbor, Node 2 a
    # score-disagreeing neighbor.
    x = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],    # Target
        [1.1, 0.9, 1.0, 1.1],    # Agrees with target in predicted score
        [-1.0, -1.0, -1.0, -1.0] # Disagrees in predicted score
    ], dtype=torch.float32)

    # Edges from 1 to 0 and 2 to 0 (add_self_loops=False keeps exactly 2 edges)
    edge_index = torch.tensor([
        [1, 2],
        [0, 0]
    ], dtype=torch.long)

    conv = CamouflageGATConv(in_channels, out_channels, heads=heads, concat=True,
                             dropout=0.0, add_self_loops=False)

    # Remove everything except the trust term from the attention logits:
    # identity projection, zeroed GAT attention vectors.
    conv.lin.weight.data = torch.eye(4)
    conv.att_src.data.zero_()
    conv.att_dst.data.zero_()
    # Score head predicts fraud-likelihood = first feature (projection is the
    # identity, so the score is just x[:, 0]): node 0 -> 1.0, node 1 -> 1.1
    # (agrees), node 2 -> -1.0 (disagrees).
    conv.score_head.weight.data = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    conv.score_head.bias.data.zero_()

    conv(x, edge_index)
    alpha = conv._alpha  # [num_edges, heads, 1], softmax over node 0's neighborhood

    att_agree = alpha[0].item()     # Edge 1->0: score diff 0.1 -> trust ~0.90
    att_disagree = alpha[1].item()  # Edge 2->0: score diff 2.0 -> trust ~0.14

    assert att_agree > att_disagree, (
        f"Score-agreeing neighbor attention {att_agree} is not greater than "
        f"score-disagreeing neighbor attention {att_disagree}"
    )
