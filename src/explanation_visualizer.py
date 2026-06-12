"""Visualize GNNExplainer importance scores on molecular structures.

This module maps node importance scores from GNNExplainer back onto a 2D
molecular diagram using RDKit, highlighting the atoms that were most
influential for the GIN model's prediction.
"""

from __future__ import annotations

from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from molecule_visualizer import _smiles_to_filename

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPLANATION_DIR = PROJECT_ROOT / "generated_explanations"


def visualize_explanation(
    smiles: str,
    node_importance: torch.Tensor,
    output_path: Path | str | None = None,
) -> Path:
    """Render a molecule with highlighted atoms based on importance scores.

    Parameters
    ----------
    smiles : str
        The SMILES string of the molecule.
    node_importance : torch.Tensor
        Importance scores for each atom, shape (num_atoms,).
    output_path : Path | str | None
        Where to save the PNG. If None, uses generated_explanations/ directory.

    Returns
    -------
    Path
        Absolute path to the saved visualization.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Canonicalize to ensure atom ordering matches the graph featurization
    # Note: from_smiles in PyG canonicalizes, so we must be consistent.
    # However, from_smiles(smiles) returns a graph where nodes match the
    # order of atoms in Chem.MolFromSmiles(canonical_smiles).
    canonical_smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(canonical_smiles)
    
    # Scale importance scores to [0, 1] for colormapping
    scores = node_importance.detach().cpu().numpy()
    if scores.max() > scores.min():
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        norm_scores = scores * 0.0 + 0.5  # All equal if no variance

    # Create colormap (White to Red)
    cmap = plt.cm.Reds
    
    atom_colors = {}
    highlight_atoms = []
    
    for i in range(mol.GetNumAtoms()):
        color = cmap(norm_scores[i])
        atom_colors[i] = tuple(color[:3])  # (R, G, B)
        highlight_atoms.append(i)

    # Setup drawing
    d2d = rdMolDraw2D.MolDraw2DCairo(600, 600)
    dos = d2d.drawOptions()
    dos.useBWAtomPalette()
    
    # Draw
    rdMolDraw2D.PrepareAndDrawMolecule(
        d2d, 
        mol, 
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors
    )
    d2d.FinishDrawing()
    
    # Save
    if output_path is None:
        EXPLANATION_DIR.mkdir(parents=True, exist_ok=True)
        stem = _smiles_to_filename(canonical_smiles)
        output_path = EXPLANATION_DIR / f"{stem}_explanation.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(d2d.GetDrawingText())

    return output_path


if __name__ == "__main__":
    # Test visualization
    test_smiles = "c1ccccc1"
    # Dummy importance scores
    dummy_scores = torch.tensor([0.1, 0.2, 0.8, 0.4, 0.5, 0.6])
    path = visualize_explanation(test_smiles, dummy_scores)
    print(f"Saved explanation to: {path}")
