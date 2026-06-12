"""Train GINModel on the ESOL solubility dataset.

Reuses train_one_epoch() and evaluate() from train.py — those functions are
architecture-agnostic and work with any model that accepts
(x, edge_index, batch). Only the model class and checkpoint path differ from
the GCN and GraphSAGE training scripts.

The dataset split (80/10/10, seed=42), batch size (32), optimiser (Adam,
lr=0.001), loss (MSE), and number of epochs (20) are kept identical to
train.py and train_graphsage.py so that any difference in final metrics
reflects the architecture, not the training setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow direct execution with:  python src/train_gin.py
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Reuse the architecture-agnostic training loop from train.py.
# train_one_epoch and evaluate call model(batch.x.float(), batch.edge_index,
# batch.batch) — they work with any GNN sharing that forward signature.
from train import train_one_epoch, evaluate  # noqa: E402
from gin_model import GINModel               # noqa: E402


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    # Device                                                               #
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    # Dataset — identical to train.py and train_graphsage.py              #
    # ------------------------------------------------------------------ #
    dataset = MoleculeNet(root=str(data_root), name="ESOL")

    # Fixed 80/10/10 split with seed=42 — all three models see exactly the
    # same training, validation, and test molecules.
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size   = int(0.1 * dataset_size)
    test_size  = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # ------------------------------------------------------------------ #
    # Data loaders                                                          #
    # ------------------------------------------------------------------ #
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------ #
    # Model                                                                 #
    # ------------------------------------------------------------------ #
    # hidden_channels=64 matches GCN and GraphSAGE to control for capacity.
    # GIN has more parameters per layer due to the two-layer MLP + BatchNorm
    # inside each GINConv, but the hidden width is the same.
    model = GINModel(
        in_channels=dataset.num_node_features,
        hidden_channels=64,
        out_channels=1,
    ).to(device)

    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # ------------------------------------------------------------------ #
    # Training loop                                                         #
    # ------------------------------------------------------------------ #
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    model_path = models_dir / "gin_esol.pth"

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print()

        # Save after every epoch so a valid checkpoint exists even if the
        # process is interrupted before the final epoch completes.
        # Each save overwrites the previous one; only the latest is kept.
        torch.save(model.state_dict(), model_path)

    # ------------------------------------------------------------------ #
    # Final test loss                                                       #
    # ------------------------------------------------------------------ #
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Loss (MSE): {test_loss:.4f}")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
