"""Evaluate the saved GCN ESOL model on the test set.

This script is inference-only. It loads the trained model from disk, evaluates
the held-out ESOL test split, and reports regression metrics without retraining
or updating model weights.
"""

from __future__ import annotations

import math

import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

from predict import DATA_ROOT, load_model


BATCH_SIZE = 32


def load_esol_test_set():
    """Load ESOL and create the same deterministic 80/10/10 split."""

    dataset = MoleculeNet(root=str(DATA_ROOT), name="ESOL")

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Keep the split deterministic so evaluation uses the same test molecules
    # each time and matches the common train/validation/test setup.
    _, _, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    if len(test_dataset) == 0:
        raise RuntimeError("The ESOL test split is empty.")

    return dataset, test_dataset


def evaluate() -> None:
    """Run test-set inference and print MSE, MAE, RMSE, and sample outputs."""

    # Automatically use GPU when available, otherwise use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, test_dataset = load_esol_test_set()
    model = load_model(device, dataset.num_node_features)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    predictions: list[float] = []
    actual_values: list[float] = []

    # no_grad() prevents gradient tracking, so this cannot update model weights.
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Forward pass only: use the trained model to predict solubility.
            output = model(
                batch.x.float(),
                batch.edge_index,
                batch.batch
            ).view(-1)
            target = batch.y.view(-1)

            predictions.extend(output.cpu().tolist())
            actual_values.extend(target.cpu().tolist())

    if not predictions:
        raise RuntimeError("No predictions were produced for the test set.")

    squared_errors = [
        (actual - predicted) ** 2
        for actual, predicted in zip(actual_values, predictions)
    ]
    absolute_errors = [
        abs(actual - predicted)
        for actual, predicted in zip(actual_values, predictions)
    ]

    mse = sum(squared_errors) / len(squared_errors)
    mae = sum(absolute_errors) / len(absolute_errors)
    rmse = math.sqrt(mse)

    print("## Test Results")
    print()
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print()
    print("## Sample Predictions")
    print()

    for actual, predicted in list(zip(actual_values, predictions))[:10]:
        print(f"Actual: {actual:.4f} | Predicted: {predicted:.4f}")


if __name__ == "__main__":
    evaluate()
