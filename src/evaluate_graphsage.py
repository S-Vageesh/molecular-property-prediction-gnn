"""Evaluate the saved GraphSAGE ESOL model on the test set.

Inference-only: loads the trained checkpoint from models/graphsage_esol.pth,
evaluates the held-out test split, and reports MSE, MAE, and RMSE.

Reuses load_esol_test_set() from evaluate.py to guarantee the evaluation uses
the exact same 80/10/10 deterministic split (seed=42) as both training scripts.
No model weights are updated.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "graphsage_esol.pth"

# Allow direct execution with:  python src/evaluate_graphsage.py
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Reuse the shared dataset-loading helper from evaluate.py so both evaluation
# scripts use the identical split without duplicating split logic.
from evaluate import load_esol_test_set        # noqa: E402
from graphsage_model import GraphSAGEModel     # noqa: E402


BATCH_SIZE = 32


def load_graphsage_model(device: torch.device, num_node_features: int) -> GraphSAGEModel:
    """Load the trained GraphSAGE checkpoint and prepare it for inference.

    Parameters
    ----------
    device:
        Target device (cpu or cuda).
    num_node_features:
        Number of input node features used during training (9 for ESOL).

    Returns
    -------
    GraphSAGEModel
        Model in eval() mode on the selected device.

    Raises
    ------
    FileNotFoundError
        If models/graphsage_esol.pth has not been created yet.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}\n"
            "Run  python src/train_graphsage.py  first."
        )

    # map_location ensures a GPU-trained checkpoint can be loaded on CPU.
    state_dict = torch.load(MODEL_PATH, map_location=device)

    # Infer hidden_channels from the weight tensor of the first SAGE layer.
    # SAGEConv stores its weight as lin_l.weight with shape
    # (hidden_channels, in_channels + hidden_channels) for the concatenated
    # [self || mean_neighbours] input, or as a bias of shape (hidden_channels,).
    hidden_channels = 64  # default that matches train_graphsage.py
    for key, tensor in state_dict.items():
        if "conv1" in key and tensor.ndim == 1:
            # bias vector length == hidden_channels
            hidden_channels = tensor.shape[0]
            break

    model = GraphSAGEModel(
        in_channels=num_node_features,
        hidden_channels=hidden_channels,
        out_channels=1,
    )
    model.load_state_dict(state_dict)
    model.to(device)

    # eval() disables any training-time behaviour (e.g. dropout if added later).
    model.eval()
    return model


def compute_metrics(
    actual: list[float],
    predicted: list[float],
) -> tuple[float, float, float]:
    """Compute MSE, MAE, and RMSE from flat lists of values.

    Parameters
    ----------
    actual:
        Ground-truth solubility values.
    predicted:
        Model-predicted solubility values.

    Returns
    -------
    tuple[float, float, float]
        (mse, mae, rmse)
    """
    squared_errors  = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    absolute_errors = [abs(a - p)   for a, p in zip(actual, predicted)]

    mse  = sum(squared_errors)  / len(squared_errors)
    mae  = sum(absolute_errors) / len(absolute_errors)
    rmse = math.sqrt(mse)

    return mse, mae, rmse


def evaluate_graphsage() -> tuple[float, float, float]:
    """Run test-set inference and print MSE, MAE, RMSE, and sample outputs.

    Returns
    -------
    tuple[float, float, float]
        (mse, mae, rmse) — also printed to stdout.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load_esol_test_set() is imported from evaluate.py and creates the same
    # deterministic 80/10/10 split used by both training scripts.
    dataset, test_dataset = load_esol_test_set()
    model = load_graphsage_model(device, dataset.num_node_features)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    predictions:   list[float] = []
    actual_values: list[float] = []

    # no_grad() prevents gradient tracking; model weights are never updated.
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            output = model(
                batch.x.float(),
                batch.edge_index,
                batch.batch,
            ).view(-1)
            target = batch.y.view(-1)

            predictions.extend(output.cpu().tolist())
            actual_values.extend(target.cpu().tolist())

    if not predictions:
        raise RuntimeError("No predictions were produced for the test set.")

    mse, mae, rmse = compute_metrics(actual_values, predictions)

    print("## GraphSAGE Test Results")
    print()
    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print()
    print("## Sample Predictions (first 10)")
    print()
    for actual, predicted in list(zip(actual_values, predictions))[:10]:
        print(f"Actual: {actual:.4f} | Predicted: {predicted:.4f}")

    return mse, mae, rmse


if __name__ == "__main__":
    evaluate_graphsage()
