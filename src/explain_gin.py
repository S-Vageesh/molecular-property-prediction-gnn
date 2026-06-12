"""Generate explanations for GIN model predictions using GNNExplainer.

This module uses PyTorch Geometric's Explainer framework to identify which
atoms (nodes) and bonds (edges) in a molecule were most influential for
the GIN model's solubility prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smiles_predict import smiles_to_graph, MOLECULENET_ESOL_NODE_FEATURES
from evaluate_gin import load_gin_model
from explanation_visualizer import visualize_explanation


def explain_prediction(smiles: str) -> tuple[float, Data]:
    """Generate an explanation for a GIN prediction on a SMILES string.

    Parameters
    ----------
    smiles : str
        The SMILES string of the molecule to explain.

    Returns
    -------
    tuple[float, Data]
        The predicted solubility value and the explanation object containing
        node_mask and edge_mask.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Preprocess SMILES to graph - smiles_to_graph validates the SMILES
    graph = smiles_to_graph(smiles).to(device)
    
    # 2. Load GIN model
    model = load_gin_model(device, MOLECULENET_ESOL_NODE_FEATURES)
    model.eval()

    # 3. Configure Explainer
    # We use GNNExplainer to identify a subgraph that maximizes the prediction.
    # For regression, it identifies the most influential features.
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )

    # 4. Generate Explanation
    # GIN model expects (x, edge_index, batch)
    # Explainer passes additional arguments as kwargs
    batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
    explanation = explainer(
        graph.x.float(),
        graph.edge_index,
        batch=batch
    )

    # 5. Get prediction
    with torch.no_grad():
        prediction = model(graph.x.float(), graph.edge_index, batch).item()

    return prediction, explanation


def main() -> None:
    """Interactive command-line entry point for GIN explanations."""
    print("Enter SMILES:")
    smiles = input().strip()

    if not smiles:
        print("Error: SMILES string is empty.")
        return

    try:
        print(f"\nGenerating explanation for: {smiles}...")
        
        # 1. Generate explanation data
        prediction, explanation = explain_prediction(smiles)
        
        # 2. Visualize explanation
        # node_mask identifies atom importance
        node_importance = explanation.node_mask.sum(dim=1)
        image_path = visualize_explanation(smiles, node_importance)
        
        # 3. Report results
        print(f"Prediction: {prediction:.4f}")
        print(f"Explanation saved to: {image_path.relative_to(PROJECT_ROOT)}")

    except ValueError as exc:
        print(f"Error: {exc}")
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")


if __name__ == "__main__":
    main()
