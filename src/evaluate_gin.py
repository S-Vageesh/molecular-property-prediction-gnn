"""Evaluate the saved GIN ESOL model on the test set.

Inference-only: loads the trained checkpoint from models/gin_esol.pth,
evaluates the held-out test split, and reports MSE, MAE, and RMSE.

Reuses:
- load_esol_test_set()  from evaluate.py        — identical 80/10/10 split
- compute_metrics()     from evaluate_graphsage.py — shared metric formula
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "gin_esol.pth"

# Allow direct execution with:  python src/evaluate_gin.py
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate import load_esol_test_set                # noqa: E402
from evaluate_graphsage import compute_metrics         # noqa: E402
from gin_model import GINModel                         # noqa: E402


BATCH_SIZE = 32


def load_gin_model(device: torch.device, num_node_features: int) -> GINModel:
    """Load the trained GIN checkpoint and prepare it for inference.

    The hidden_channels are inferred from the BatchNorm weight vector stored
    inside conv1's MLP, so the evaluator does not need to hard-code the
    architecture width.

    Parameters
    ----------
    device:
        Target device (cpu or cuda).
    num_node_features:
        Number of input node features used during training (9 for ESOL).

    Returns
    -------
    GINModel
        Model in eval() mode, moved to ``device``.

    Raises
    ------
    FileNotFoundError
        If models/gin_esol.pth has not been created yet.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}\n"
            "Run  python src/train_gin.py  first."
        )

    # map_location lets a GPU-trained checkpoint load on a CPU-only machine.
    state_dict = torch.load(MODEL_PATH, map_location=device)

    # Infer hidden_channels from the BatchNorm1d weight inside conv1's MLP.
    # GINConv stores the MLP as conv1.nn, and the BN is at index 1, so the
    # key is "conv1.nn.1.weight" with shape (hidden_channels,).
    hidden_channels = 64  # safe default matching train_gin.py
    for key, tensor in state_dict.items():
        if key == "conv1.nn.1.weight" and tensor.ndim == 1:
            hidden_channels = tensor.shape[0]
            break

    model = GINModel(
        in_channels=num_node_features,
        hidden_channels=hidden_channels,
        out_channels=1,
    )
    model.load_state_dict(state_dict)
    model.to(device)

    # eval() disables BatchNorm's running-stat updates and any future dropout.
    model.eval()
    return model


def evaluate_gin() -> tuple[float, float, float]:
    """Run test-set inference and print MSE, MAE, RMSE, and sample outputs.

    Returns
    -------
    tuple[float, float, float]
        (mse, mae, rmse) — values are also printed to stdout.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load_esol_test_set() from evaluate.py creates the deterministic
    # 80/10/10 split with seed=42, identical to all three training scripts.
    dataset, test_dataset = load_esol_test_set()
    model = load_gin_model(device, dataset.num_node_features)
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

    # compute_metrics() is imported from evaluate_graphsage.py — one shared
    # implementation of MSE / MAE / RMSE for all three evaluators.
    mse, mae, rmse = compute_metrics(actual_values, predictions)

    print("## GIN Test Results")
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
    evaluate_gin()
