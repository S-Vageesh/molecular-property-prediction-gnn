"""Compare GCN, GraphSAGE, and GIN models on a random sample of test molecules.

This script loads the trained checkpoints for all three architectures,
selects 20 molecules from the deterministic ESOL test split (seed=42),
and prints a side-by-side comparison of their predictions and errors.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure project modules are importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import load_model as load_gcn_model
from evaluate import load_esol_test_set
from evaluate_graphsage import load_graphsage_model
from evaluate_gin import load_gin_model


def main() -> None:
    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load the deterministic 80/10/10 test split (seed=42)
    dataset, test_dataset = load_esol_test_set()
    num_test_mols = len(test_dataset)
    num_samples = min(20, num_test_mols)

    # ------------------------------------------------------------------ #
    # Load Models                                                          #
    # ------------------------------------------------------------------ #
    print("Loading models...")
    try:
        gcn_model = load_gcn_model(device, dataset.num_node_features)
        sage_model = load_graphsage_model(device, dataset.num_node_features)
        gin_model = load_gin_model(device, dataset.num_node_features)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all three models are trained before running this comparison.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Sampling                                                             #
    # ------------------------------------------------------------------ #
    # Select 20 random indices from the test set
    # We use a separate random seed for the sampling itself so it's not 
    # always the same 20 molecules, but the test set itself is deterministic.
    sample_indices = random.sample(range(num_test_mols), num_samples)
    
    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #
    gcn_errors: list[float] = []
    sage_errors: list[float] = []
    gin_errors: list[float] = []

    print(f"\n--- Comparing Predictions on {num_samples} Sampled Test Molecules ---\n")

    for i, idx in enumerate(sample_indices):
        molecule = test_dataset[idx]
        actual = molecule.y.item()
        
        # DataLoader handles batching even for a single molecule
        loader = DataLoader([molecule], batch_size=1)
        batch = next(iter(loader)).to(device)

        with torch.no_grad():
            # GCN
            gcn_pred = gcn_model(batch.x.float(), batch.edge_index, batch.batch).item()
            gcn_err = abs(actual - gcn_pred)
            gcn_errors.append(gcn_err)

            # GraphSAGE
            sage_pred = sage_model(batch.x.float(), batch.edge_index, batch.batch).item()
            sage_err = abs(actual - sage_pred)
            sage_errors.append(sage_err)

            # GIN
            gin_pred = gin_model(batch.x.float(), batch.edge_index, batch.batch).item()
            gin_err = abs(actual - gin_pred)
            gin_errors.append(gin_err)

        print(f"Molecule Index: {idx}")
        print(f"Actual:         {actual:.4f}")
        print(f"GCN Prediction: {gcn_pred:.4f} (Error: {gcn_err:.4f})")
        print(f"SAGE Prediction: {sage_pred:.4f} (Error: {sage_err:.4f})")
        print(f"GIN Prediction:  {gin_pred:.4f} (Error: {gin_err:.4f})")
        print("-" * 40)

    # ------------------------------------------------------------------ #
    # Summary Report                                                       #
    # ------------------------------------------------------------------ #
    avg_gcn_err = sum(gcn_errors) / num_samples
    avg_sage_err = sum(sage_errors) / num_samples
    avg_gin_err = sum(gin_errors) / num_samples

    print("\n--- Summary: Average Absolute Error (Sampled) ---")
    print(f"GCN:       {avg_gcn_err:.4f}")
    print(f"GraphSAGE: {avg_sage_err:.4f}")
    print(f"GIN:       {avg_gin_err:.4f}")


if __name__ == "__main__":
    main()
