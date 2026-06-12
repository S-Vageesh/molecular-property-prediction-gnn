"""Load and inspect the ESOL dataset from MoleculeNet."""

from pathlib import Path

from torch_geometric.datasets import MoleculeNet


def main() -> None:
    # Step 1: Resolve paths relative to this file so `python src/dataset.py` works from the repo root.
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data"

    # Step 2: Download (if needed) and load the ESOL solubility dataset from MoleculeNet.
    # PyTorch Geometric fetches the raw CSV, converts SMILES strings into graph objects, and caches them under data/.
    dataset = MoleculeNet(root=str(data_root), name="ESOL")

    # Step 3: Collect dataset-level metadata for inspection.
    dataset_name = dataset.names[dataset.name][0]
    num_molecules = len(dataset)
    num_node_features = dataset.num_node_features
    num_targets = dataset[0].y.shape[-1]

    # Step 4: Print a summary of the full dataset.
    print(f"Dataset name: {dataset_name}")
    print(f"Number of molecules: {num_molecules}")
    print(f"Number of node features: {num_node_features}")
    print(f"Number of target properties: {num_targets}")

    # Step 5: Inspect the first molecule graph (atoms, bonds, and solubility target).
    first_molecule = dataset[0]
    num_atoms = first_molecule.num_nodes
    num_edges = first_molecule.edge_index.size(1)
    target_value = first_molecule.y.squeeze().item()

    print()
    print("First molecule:")
    print(f"  Number of atoms (nodes): {num_atoms}")
    print(f"  Number of bonds (edges): {num_edges}")
    print(f"  Target value: {target_value}")


if __name__ == "__main__":
    main()
