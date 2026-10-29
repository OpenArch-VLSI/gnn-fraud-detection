import os
import sys

import torch
from torch_geometric.nn import GATConv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.models import FraudCamouflageGNN, FraudGAT, GraphSAGEModel


def _toy_graph(num_nodes=12, in_channels=8, num_edges=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, in_channels, generator=g)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), generator=g)
    return x, edge_index


def test_models_forward_shape_and_full_gradient_flow():
    x, edge_index = _toy_graph()
    models = [
        GraphSAGEModel(8, 16),
        FraudGAT(8, 16, heads=2),
        # Undetached trust lets the main loss reach score_head, so full gradient flow holds.
        FraudCamouflageGNN(8, 16, heads=2, detach_trust=False),
    ]
    for model in models:
        out = model(x, edge_index)
        assert out.shape == (x.size(0), 1), type(model).__name__
        out.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{type(model).__name__}.{name} got no gradient"


def test_camouflage_detached_score_head_trained_only_by_auxiliary_loss():
    x, edge_index = _toy_graph()
    model = FraudCamouflageGNN(8, 16, heads=2, detach_trust=True)

    # With trust detached, the main output must not reach score_head.
    out = model(x, edge_index)
    out.sum().backward()
    for name, p in model.score_head.named_parameters():
        assert p.grad is None, f"score_head.{name} received main-loss gradient"

    # The auxiliary loss must reach score_head.
    model.zero_grad()
    model(x, edge_index)
    model.auxiliary_node_scores().sum().backward()
    for name, p in model.score_head.named_parameters():
        assert p.grad is not None, f"score_head.{name} got no auxiliary gradient"


def test_camouflage_auxiliary_scores_are_per_node_and_track_forward():
    x, edge_index = _toy_graph()
    model = FraudCamouflageGNN(8, 16, heads=2)
    model(x, edge_index)
    scores = model.auxiliary_node_scores()
    assert scores.shape == (x.size(0),)
    assert scores.requires_grad  # must be trainable by the auxiliary loss


def test_camouflage_auxiliary_scores_require_forward_first():
    model = FraudCamouflageGNN(8, 16, heads=2)
    try:
        model.auxiliary_node_scores()
    except RuntimeError:
        return
    raise AssertionError("auxiliary_node_scores() should refuse to run before forward()")


def test_trust_term_favors_score_agreement_at_layer_level():
    """
    With a positive learned gain on the trust feature, an edge whose endpoints
    agree (trust near 1) must get more attention than one whose endpoints
    disagree (trust near 0). Attention vectors are zeroed so the trust term is
    the only thing that differs between the two edges.
    """
    conv = GATConv(4, 4, heads=1, concat=False, edge_dim=1, add_self_loops=False)
    conv.lin.weight.data = torch.eye(4)
    conv.att_src.data.zero_()
    conv.att_dst.data.zero_()
    conv.lin_edge.weight.data = torch.ones(4, 1)   # trust -> 4 identical channels
    conv.att_edge.data = torch.ones(1, 1, 4)       # gain k = 4 > 0

    x = torch.zeros(3, 4)
    edge_index = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)  # 1->0, 2->0
    trust = torch.tensor([[0.9], [0.1]])           # agree, disagree

    _, (_, alpha) = conv(x, edge_index, edge_attr=trust, return_attention_weights=True)
    alpha = alpha.view(-1)
    assert alpha[0].item() > alpha[1].item(), (alpha[0].item(), alpha[1].item())


def test_trust_gains_shape_matches_heads():
    model = FraudCamouflageGNN(8, 16, heads=4)
    gains = model.trust_gains()
    assert gains['conv1'].shape == (4,)
    assert gains['conv2'].shape == (1,)


def test_model_passes_score_agreement_trust_to_attention():
    """The edge feature the model hands to conv1 must be exp(-|s_u - s_v|)."""
    x, edge_index = _toy_graph()
    model = FraudCamouflageGNN(8, 16, heads=2, detach_trust=False)
    captured = {}

    def hook(module, args, kwargs):
        captured['edge_attr'] = kwargs['edge_attr'].detach().clone()

    model.conv1.register_forward_pre_hook(hook, with_kwargs=True)
    model(x, edge_index)

    s = model.auxiliary_node_scores().detach()
    src, dst = edge_index[0], edge_index[1]
    expected = torch.exp(-(s[src] - s[dst]).abs()).unsqueeze(-1)
    assert captured['edge_attr'].shape == expected.shape
    assert torch.allclose(captured['edge_attr'], expected)