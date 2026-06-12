"""Graph Convolutional Network for graph-level molecular property regression."""

from pathlib import Path

import torch
from torch.nn import Linear, ReLU
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNModel(torch.nn.Module):
    """GCN that maps a molecular graph to a single regression value."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 1) -> None:
        super().__init__()

        # GCNConv: graph convolution layer. Each node aggregates features from its neighbors
        # via message passing, then applies a learned linear transform.
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.relu = ReLU()

        # Final prediction layer: maps the pooled graph embedding to one scalar output.
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Message passing round 1: atoms exchange information along bonds (edge_index).
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        # Message passing round 2: deeper neighborhood aggregation.
        x = self.conv2(x, edge_index)
        x = self.relu(x)

        # Global mean pooling: average node embeddings into one graph-level vector.
        x = global_mean_pool(x, batch)

        # Linear head: produce a single predicted property value for the whole molecule.
        return self.lin(x)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data"

    dataset = MoleculeNet(root=str(data_root), name="ESOL")
    first_molecule = dataset[0]

    model = GCNModel(
        in_channels=dataset.num_node_features,
        hidden_channels=64,
        out_channels=1,
    )
    model.eval()

    # Batch vector tells pooling which nodes belong to the same graph (all zeros for one molecule).
    batch = torch.zeros(first_molecule.num_nodes, dtype=torch.long)

    with torch.no_grad():
        output = model(first_molecule.x.float(), first_molecule.edge_index, batch)

    print("Model output (predicted solubility):")
    print(output)


if __name__ == "__main__":
    main()
