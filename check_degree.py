import torch

data = torch.load("data/processed/graph.pt", weights_only=False)
num_nodes = data.x.size(0)

src = data.edge_index[0]
dst = data.edge_index[1]

degree = torch.zeros(num_nodes)
degree.index_add_(0, src, torch.ones(src.size(0)))
degree.index_add_(0, dst, torch.ones(dst.size(0)))

zero_degree_count = (degree == 0).sum().item()
print(f"Nodes with zero degree: {zero_degree_count} out of {num_nodes}")