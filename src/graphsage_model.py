"""GraphSAGE model for graph-level molecular property regression.

GraphSAGE (Graph SAmple and aggreGatE) differs from GCN in how it aggregates
neighbor information:

- GCN (Kipf & Welling 2017) uses a symmetric normalized sum of neighbor
  features, which is a spectral convolution.  Every node's update depends on
  the global degree normalization of the graph.

- GraphSAGE (Hamilton et al. 2017) concatenates a node's *own* embedding with
  the *mean* of its neighbors' embeddings and applies a learned linear
  transform.  This makes the update inductive: the model generalizes to
  unseen nodes and graphs without needing to see the full graph at training
  time — a useful property for molecular property prediction on new compounds.

Both models share the same constructor signature so they are drop-in
interchangeable in training and evaluation scripts.
"""

import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGEModel(torch.nn.Module):
    """Two-layer GraphSAGE network that maps a molecular graph to one scalar.

    Architecture
    ------------
    1. SAGEConv(in_channels  → hidden_channels)  + ReLU
    2. SAGEConv(hidden_channels → hidden_channels) + ReLU
    3. global_mean_pool  →  graph-level embedding  (hidden_channels,)
    4. Linear(hidden_channels → out_channels)      →  predicted property

    The depth (2 layers) and width (hidden_channels=64 by default) match the
    GCN baseline so that differences in test metrics reflect the aggregation
    strategy rather than model capacity.

    Parameters
    ----------
    in_channels:
        Number of input node features.  For the ESOL MoleculeNet dataset this
        is 9 (the PyG SMILES featurizer produces nine atom descriptors).
    hidden_channels:
        Width of the hidden GraphSAGE layers.  Default 64 matches the GCN.
    out_channels:
        Number of output values per graph.  1 for solubility regression.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        # SAGEConv layer 1: maps raw atom features to the hidden space.
        # aggr="mean" is the standard GraphSAGE mean aggregation.
        # The layer learns W_self (for the node itself) and W_neigh (for the
        # aggregated neighbour representation) separately, then sums them.
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")

        # SAGEConv layer 2: refines the hidden representation with a second
        # round of neighbourhood aggregation.
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")

        self.relu = ReLU()

        # Prediction head: maps the pooled graph embedding to one scalar.
        # This is identical to the GCN head so the only difference between
        # architectures is in conv1 / conv2.
        self.lin = Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: message passing → pooling → regression head.

        Parameters
        ----------
        x:
            Node feature matrix of shape (num_nodes, in_channels).
        edge_index:
            Graph connectivity in COO format, shape (2, num_edges).
        batch:
            Batch assignment vector mapping each node to its graph index,
            shape (num_nodes,). Required by global_mean_pool.

        Returns
        -------
        torch.Tensor
            Predicted property values, shape (batch_size, out_channels).
        """

        # Round 1: each atom aggregates information from its bonded neighbours.
        # SAGEConv concatenates the node's own features with the mean of its
        # neighbours, applies a linear transform, and adds a bias.
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        # Round 2: atoms now aggregate from neighbours that have themselves
        # already incorporated one hop of neighbourhood context, giving each
        # node a 2-hop receptive field — the same depth as the GCN baseline.
        x = self.conv2(x, edge_index)
        x = self.relu(x)

        # Global mean pooling: collapse all node embeddings for each graph
        # in the batch into a single fixed-size graph-level vector.
        x = global_mean_pool(x, batch)

        # Regression head: produce one predicted solubility value per graph.
        return self.lin(x)
