from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "gcn_esol.pth"
DATA_ROOT = PROJECT_ROOT / "data" / "ESOL"

# Make project modules importable when this file is run as:
#   python src/predict.py
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


class FallbackGCNModel(torch.nn.Module):
    """Small GCN architecture used only if no project GCNModel can be imported."""
    
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 64,
        num_outputs: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_outputs)

    def forward(self, data: Any) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


def get_project_gcn_model_class() -> type[torch.nn.Module]:
    """Find the repository's GCNModel class without importing training scripts."""

    for module_name in ("model", "models", "gcn", "src.model", "src.models", "src.gcn"):
        try:
            module = __import__(module_name, fromlist=["GCNModel"])
        except Exception:
            continue

        model_cls = getattr(module, "GCNModel", None)
        if isinstance(model_cls, type) and issubclass(model_cls, torch.nn.Module):
            return model_cls



    return FallbackGCNModel


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Return a plain state_dict from common checkpoint formats."""

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint

    raise TypeError("Checkpoint does not contain a model state_dict.")


def infer_hidden_channels(state_dict: dict[str, torch.Tensor], default: int = 64) -> int:
    """Infer the hidden size from common GCN checkpoint tensor shapes."""

    for key, value in state_dict.items():
        if value.ndim == 2 and (
            key.endswith("conv1.lin.weight") or key.endswith("conv1.weight")
        ):
            return int(value.shape[0])

    for key, value in state_dict.items():
        if value.ndim == 1 and "conv1" in key and "bias" in key:
            return int(value.shape[0])


    return default


def build_model(
    model_cls: type[torch.nn.Module],
    num_node_features: int,
    hidden_channels: int,
) -> torch.nn.Module:
    """Instantiate GCNModel while accommodating common constructor names."""

    signature = inspect.signature(model_cls)
    kwargs: dict[str, Any] = {}

    for name in signature.parameters:
        if name in {"num_node_features", "in_channels", "input_dim", "input_channels"}:
            kwargs[name] = num_node_features
        elif name in {"hidden_channels", "hidden_dim", "hidden_size"}:
            kwargs[name] = hidden_channels
        elif name in {"out_channels", "output_dim", "num_outputs"}:
            kwargs[name] = 1

    if kwargs:
        return model_cls(**kwargs)

    try:
        return model_cls(num_node_features, hidden_channels, 1)
    except TypeError:
        return model_cls(num_node_features, hidden_channels)


def load_model(device: torch.device, num_node_features: int) -> torch.nn.Module:
    """Load the saved model file and move it to the selected device."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

    # map_location lets a GPU-trained model load on CPU-only machines too.
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        state_dict = extract_state_dict(checkpoint)
        model_cls = get_project_gcn_model_class()
        hidden_channels = infer_hidden_channels(state_dict)
        model = build_model(model_cls, num_node_features, hidden_channels)
        model.load_state_dict(state_dict)

    # eval() disables training-time behavior such as dropout and batch norm updates.
    model.to(device)
    model.eval()
    return model


def load_esol_test_molecule():
    """Load ESOL and return one graph from a deterministic test split."""

    dataset = MoleculeNet(root=str(DATA_ROOT), name="ESOL")

    # Use the same deterministic 80/10/10 split shape commonly used for ESOL demos.
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    _, _, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    if len(test_dataset) == 0:
        raise RuntimeError("The ESOL test split is empty.")

    return dataset, test_dataset[0]


def main() -> None:
    # Automatically use GPU when available, otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, molecule = load_esol_test_molecule()
    model = load_model(device, dataset.num_node_features)

    # DataLoader creates the batch vector expected by graph-level GCN models.
    loader = DataLoader([molecule], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)

    # no_grad() guarantees inference does not build gradients or update weights.
    with torch.no_grad():
        prediction = model(
            batch.x.float(),
            batch.edge_index,
            batch.batch
        )

    actual_solubility = batch.y.view(-1)[0].item()
    predicted_solubility = prediction.view(-1)[0].item()
    absolute_error = abs(actual_solubility - predicted_solubility)

    print(f"Actual solubility: {actual_solubility:.4f}")
    print(f"Predicted solubility: {predicted_solubility:.4f}")
    print(f"Absolute error: {absolute_error:.4f}")


if __name__ == "__main__":
    main()