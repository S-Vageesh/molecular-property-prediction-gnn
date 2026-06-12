"""Train GCNModel on the ESOL solubility dataset."""

from pathlib import Path

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

from model import GCNModel


def train_one_epoch(
    model: GCNModel,
    loader: DataLoader,
    optimizer: Adam,
    criterion: MSELoss,
    device: torch.device,
) -> float:
    """Run one training epoch and return the average loss."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        # Move the batched graphs to CPU or GPU.
        batch = batch.to(device)

        # Reset gradients from the previous step before the next update.
        optimizer.zero_grad()

        # Forward pass: run the GCN on node features, bonds, and batch assignment.
        predictions = model(batch.x.float(), batch.edge_index, batch.batch)

        # Loss computation: compare predicted solubility to ground-truth targets.
        loss = criterion(predictions, batch.y)

        # Backpropagation: compute gradients of the loss with respect to model weights.
        loss.backward()

        # Optimizer step: update model parameters using the computed gradients.
        optimizer.step()

        # Accumulate loss weighted by the number of graphs in this batch.
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: GCNModel,
    loader: DataLoader,
    criterion: MSELoss,
    device: torch.device,
) -> float:
    """Evaluate the model and return the average loss."""
    model.eval()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        predictions = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = criterion(predictions, batch.y)
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # Use GPU when available; otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the full ESOL dataset (molecular graphs with solubility labels).
    dataset = MoleculeNet(root=str(data_root), name="ESOL")

    # Dataset splitting: hold out 10% for validation and 10% for testing.
    # The remaining 80% is used for training. A fixed seed keeps splits reproducible.
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Batching: DataLoader groups multiple molecules into one mini-batch.
    # PyG merges graphs and provides a `batch` vector for global pooling.
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GCNModel(
        in_channels=dataset.num_node_features,
        hidden_channels=64,
        out_channels=1,
    ).to(device)

    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print()

    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}")

    model_path = models_dir / "gcn_esol.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
