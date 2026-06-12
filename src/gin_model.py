"""Graph Isomorphism Network (GIN) for molecular property regression.

GIN (Xu et al., 2019 — "How Powerful are Graph Neural Networks?") is
theoretically the most expressive of the three architectures in this
benchmark.  Xu et al. proved that GIN is as powerful as the
Weisfeiler-Leman (WL) graph isomorphism test: it can distinguish any pair
of non-isomorphic graphs that WL can distinguish, while GCN and GraphSAGE
with mean aggregation cannot.

The key theoretical insight is that *sum* aggregation over neighbours is
strictly more powerful than mean aggregation when distinguishing
multisets of neighbour features.  GIN formalises this with the update rule:

    h_v^{(k)} = MLP^{(k)}((1 + ε) · h_v^{(k-1)} + Σ_{u ∈ N(v)} h_u^{(k-1)})

where ε (epsilon) is either fixed at 0 or learned (train_eps=True), and the
MLP replaces the single linear layer used by GCN and GraphSAGE.

Comparison with the other architectures in this benchmark
---------------------------------------------------------
- GCN        — symmetric-normalised mean sum; spectral / transductive
- GraphSAGE  — concatenate(self, mean(neighbours)); inductive
- GIN        — sum aggregation + MLP per layer; maximally expressive under WL
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, global_add_pool


class GINModel(torch.nn.Module):
    """Two-layer GIN that maps a molecular graph to a single regression value.

    Architecture
    ------------
    Layer 1: GINConv( MLP: Linear → BN → ReLU → Linear ) + ReLU
    Layer 2: GINConv( MLP: Linear → BN → ReLU → Linear ) + ReLU
    Pooling: global_add_pool  →  graph embedding  (hidden_channels,)
    Head:    Linear(hidden_channels → out_channels)

    Design choices
    --------------
    * **MLP inside each GINConv**: A two-layer MLP with BatchNorm gives the
      network enough capacity per message-passing step to satisfy the
      theoretical expressiveness guarantee.  A single linear layer (as in GCN
      or GraphSAGE) is strictly weaker.

    * **BatchNorm1d**: Stabilises training by normalising node-feature
      distributions between the first and second linear layer of each MLP.
      This is standard practice in GIN implementations.

    * **train_eps=True**: The ε parameter in ``(1 + ε) · h_v`` is treated as
      a learnable scalar rather than fixed at 0, giving the model one extra
      degree of freedom per layer to weight self vs. neighbour contributions.

    * **global_add_pool**: Sum pooling preserves the multiset cardinality
      information that mean pooling discards.  It is the theoretically
      motivated choice for GIN; using mean pooling would undermine the
      expressiveness advantage.

    * **Depth and width** match GCN and GraphSAGE (2 layers, hidden_channels=64)
      so that differences in test metrics reflect the aggregation strategy, not
      model capacity.

    Parameters
    ----------
    in_channels:
        Number of input node features per atom.  9 for ESOL (PyG SMILES
        featurizer produces 9 categorical atom descriptors).
    hidden_channels:
        Width of all hidden layers.  Default 64 matches the other models.
    out_channels:
        Number of scalar outputs per graph.  1 for solubility regression.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # GIN layer 1                                                         #
        # ------------------------------------------------------------------ #
        # The MLP has two linear transforms with BatchNorm + ReLU in between.
        # Input  →  hidden  (with BN and ReLU)  →  hidden
        mlp1 = Sequential(
            Linear(in_channels, hidden_channels),
            BatchNorm1d(hidden_channels),   # normalise across nodes in the batch
            ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        # GINConv wraps the MLP and applies the GIN update rule.
        # train_eps=True makes ε a learnable parameter (one per layer).
        self.conv1 = GINConv(mlp1, train_eps=True)

        # ------------------------------------------------------------------ #
        # GIN layer 2                                                         #
        # ------------------------------------------------------------------ #
        mlp2 = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        self.conv2 = GINConv(mlp2, train_eps=True)

        # ------------------------------------------------------------------ #
        # Regression head                                                      #
        # ------------------------------------------------------------------ #
        # Maps the pooled graph-level embedding to the target scalar.
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
            Node feature matrix, shape (num_nodes, in_channels).
            Cast to float32 before calling if stored as int (ESOL uses int
            categorical features that must be floated for linear layers).
        edge_index:
            Graph connectivity in COO format, shape (2, num_edges).
        batch:
            Node-to-graph assignment vector, shape (num_nodes,).
            Required by global_add_pool to aggregate per graph.

        Returns
        -------
        torch.Tensor
            Predicted property values, shape (batch_size, out_channels).
        """

        # --- Layer 1: first round of neighbourhood aggregation ------------- #
        # GINConv computes: h_v ← MLP1((1+ε1)·h_v + Σ_{u∈N(v)} h_u)
        # The result is then passed through ReLU before the second layer.
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # --- Layer 2: second round gives each node a 2-hop receptive field - #
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # --- Graph-level pooling ------------------------------------------- #
        # global_add_pool sums all node embeddings belonging to the same graph.
        # Sum (not mean) is essential for GIN's expressiveness: it preserves
        # information about the number of nodes with each feature pattern.
        x = global_add_pool(x, batch)

        # --- Regression head ----------------------------------------------- #
        return self.lin(x)
