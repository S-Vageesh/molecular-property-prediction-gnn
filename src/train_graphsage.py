"""Train GraphSAGEModel on the ESOL solubility dataset.

Reuses train_one_epoch() and evaluate() from train.py — those functions are
architecture-agnostic and work with any model that accepts
(x, edge_index, batch).  Only the model class and checkpoint path differ from
the GCN training script.

The dataset split (80/10/10, seed=42), batch size (32), optimiser (Adam,
lr=0.001), loss (MSE), and number of epochs (20) are kept identical to
train.py so that any difference in final test loss reflects the architecture,
not the training setup.
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

# Allow direct execution with:  python src/train_graphsage.py
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Reuse the architecture-agnostic training loop from train.py.
# train_one_epoch and evaluate only call model(batch.x, batch.edge_index,
# batch.batch) so they work with any GNN that shares that forward signature.
from train import train_one_epoch, evaluate  # noqa: E402
from graphsage_model import GraphSAGEModel   # noqa: E402


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
    # Dataset — identical to train.py                                      #
    # ------------------------------------------------------------------ #
    # MoleculeNet downloads and caches ESOL the first time; subsequent runs
    # use the local copy under data/ESOL/.
    dataset = MoleculeNet(root=str(data_root), name="ESOL")

    # 80 / 10 / 10 split with fixed seed — same as the GCN baseline so that
    # both models see exactly the same training, validation, and test graphs.
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

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
    # hidden_channels=64 matches the GCN so capacity is controlled.
    model = GraphSAGEModel(
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

    # 20 epochs — identical to GCN baseline for a fair wall-clock comparison.
    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        # train_one_epoch and evaluate are imported from train.py and are
        # architecture-agnostic: they call model(batch.x.float(), ...)
        # without knowing whether the model is GCN or GraphSAGE.
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print()

    # ------------------------------------------------------------------ #
    # Final test loss and checkpoint                                        #
    # ------------------------------------------------------------------ #
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Loss (MSE): {test_loss:.4f}")

    model_path = models_dir / "graphsage_esol.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
