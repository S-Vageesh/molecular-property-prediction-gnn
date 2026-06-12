"""Educational inspection of a single molecular graph from the ESOL dataset."""

from pathlib import Path

from torch_geometric.datasets import MoleculeNet


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data"

    dataset = MoleculeNet(root=str(data_root), name="ESOL")
    first_molecule = dataset[0]

    # x: node feature matrix — one row per atom (node), columns are atom descriptors
    #     (e.g. atomic number, degree, hybridization) used as GNN input features.
    x = first_molecule.x

    # edge_index: COO-format connectivity — row 0 is source atom indices, row 1 is target
    #     atom indices; each column is a directed bond between two atoms.
    edge_index = first_molecule.edge_index

    # y: graph-level regression target — ESOL solubility (log solubility in mol/L) for this molecule.
    y = first_molecule.y

    print("Tensor shapes:")
    print(f"  x shape: {tuple(x.shape)}")
    print(f"  edge_index shape: {tuple(edge_index.shape)}")
    print(f"  y shape: {tuple(y.shape)}")

    print()
    print("First 5 node feature vectors (rows of x):")
    print(x[:5])

    print()
    print("First 10 edges (source -> target atom indices from edge_index):")
    for edge_idx in range(min(10, edge_index.size(1))):
        source = edge_index[0, edge_idx].item()
        target = edge_index[1, edge_idx].item()
        print(f"  Edge {edge_idx}: {source} -> {target}")


if __name__ == "__main__":
    main()
