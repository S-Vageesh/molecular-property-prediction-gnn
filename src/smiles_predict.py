"""Predict molecular solubility from an arbitrary SMILES string.

This module is designed for reuse by scripts, FastAPI routes, frontend backend
adapters, and future explainability tools. It performs inference only:
- no optimizer is created,
- no backward pass is run,
- no model weights are updated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.smiles import from_smiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow this file to be executed directly with:
#   python src/smiles_predict.py
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import load_model  # noqa: E402


# MoleculeNet ESOL uses the PyTorch Geometric SMILES featurizer, which produces
# nine categorical atom features per node. Keeping this value explicit lets the
# saved GCN model be instantiated without downloading ESOL during API inference.
MOLECULENET_ESOL_NODE_FEATURES = 9


def smiles_to_graph(smiles: str) -> Data:
    """Convert a SMILES string into a PyTorch Geometric graph.

    RDKit first parses the user-provided SMILES string. If parsing fails, the
    input is not a valid molecule and a clear ValueError is raised instead of
    allowing a lower-level model or graph-construction error to leak out.
    """

    cleaned_smiles = smiles.strip()
    if not cleaned_smiles:
        raise ValueError("Invalid SMILES: input is empty.")

    # SMILES parsing: RDKit validates the molecular syntax and chemistry. A
    # return value of None means the string could not be interpreted as a valid
    # molecule, for example "XYZ123".
    molecule = Chem.MolFromSmiles(cleaned_smiles)
    if molecule is None:
        raise ValueError(
            f"Invalid SMILES: {smiles!r} could not be parsed as a molecule."
        )

    # Canonicalizing through RDKit normalizes equivalent SMILES strings before
    # graph construction while preserving the same molecular structure.
    canonical_smiles = Chem.MolToSmiles(molecule)

    # Graph construction: PyG's from_smiles utility is the same featurization
    # path used by MoleculeNet-style molecular datasets. It creates a Data
    # object with x, edge_index, and edge_attr fields.
    graph = from_smiles(canonical_smiles)

    # Node features: graph.x has one row per atom. The columns are categorical
    # atom descriptors used by MoleculeNet, including atomic number, chirality,
    # degree, formal charge, hydrogen count, radical electrons, hybridization,
    # aromaticity, and ring membership.
    if graph.x is None or graph.x.numel() == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r} produced an empty graph.")

    if graph.x.size(-1) != MOLECULENET_ESOL_NODE_FEATURES:
        raise ValueError(
            "SMILES graph feature size does not match the ESOL MoleculeNet "
            f"format: expected {MOLECULENET_ESOL_NODE_FEATURES}, "
            f"got {graph.x.size(-1)}."
        )

    # Edge construction: graph.edge_index stores directed bond connections.
    # PyG includes both directions for each molecular bond so message passing
    # can flow from atom A to atom B and from atom B to atom A.
    if graph.edge_index is None:
        raise ValueError(f"Invalid SMILES: {smiles!r} produced no edge index.")

    return graph


def predict_smiles(smiles: str) -> float:
    """Predict solubility for a single SMILES string.

    The trained GCN is loaded from models/gcn_esol.pth, moved to GPU when one is
    available, and executed under torch.no_grad() so inference does not track
    gradients or update model weights.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = smiles_to_graph(smiles)
    model = load_model(device, MOLECULENET_ESOL_NODE_FEATURES)

    # DataLoader creates the batch vector expected by graph-level GCN models.
    # Even for one molecule, batching is useful because the model can use
    # global pooling over graph.batch exactly as it did during training.
    loader = DataLoader([graph], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)

    # Inference: no_grad disables autograd bookkeeping. The model remains in
    # eval mode because load_model() calls model.eval() after loading weights.
    with torch.no_grad():
        prediction = model(
            batch.x.float(), 
            batch.edge_index,
            batch.batch
        ).view(-1)[0].item()

    return float(prediction)


def main() -> None:
    """Interactive command-line entry point."""

    print("Enter SMILES:")
    smiles = input().strip()

    try:
        prediction = predict_smiles(smiles)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise

    print()
    print("Predicted Solubility:")
    print(f"{prediction:.4f}")


if __name__ == "__main__":
    main()
