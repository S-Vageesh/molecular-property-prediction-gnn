# GNN Architecture Comparison — ESOL Solubility Prediction

## Dataset

**ESOL (Delaney)** — aqueous solubility benchmark  
- Task: regression, predict log(mol/L) solubility from molecular graph  
- Split: 80% train / 10% validation / 10% test (seed=42, deterministic)  
- Node features: 9 atom descriptors (PyTorch Geometric SMILES featurizer)  

## Training Configuration

All models trained under identical conditions for a fair comparison:

| Setting          | Value                     |
|------------------|---------------------------|
| Optimiser        | Adam                      |
| Learning rate    | 0.001                     |
| Loss             | Mean Squared Error (MSE)  |
| Epochs           | 20                        |
| Batch size       | 32                        |
| Hidden channels  | 64                        |
| Output channels  | 1 (solubility scalar)     |

## Results

| Model     |  MSE   |  MAE   | RMSE   |
|-----------|--------|--------|--------|
| GCN       | 3.3883 | 1.4526 | 1.8407 |
| GraphSAGE | 3.1080 | 1.3789 | 1.7630 |

*Lower is better for all metrics.*

## Architecture Differences

### GCN (Kipf & Welling, 2017)
- **Aggregation:** symmetric normalised sum — `D^{-1/2} A D^{-1/2} X W`
- **Nature:** spectral / transductive; normalisation depends on the global degree matrix
- **Checkpoint:** `models/gcn_esol.pth`
- **Parameters:** ~8,900

### GraphSAGE (Hamilton et al., 2017)
- **Aggregation:** concatenate self-embedding with mean of neighbours — `W [h_v || mean(h_u)]`
- **Nature:** inductive; the update rule does not depend on the full graph degree, making it generalise to unseen molecules without re-training
- **Checkpoint:** `models/graphsage_esol.pth`
- **Parameters:** ~9,537

## Observations

- GraphSAGE outperforms GCN on all three metrics under the same training budget
- The inductive mean-aggregation of GraphSAGE produces slightly lower error,
  suggesting it captures local neighbourhood structure more effectively than
  spectral normalisation on this dataset
- Both models converged within 20 epochs; neither shows signs of overfitting
  at this depth and width

## Reproducing Results

```bash
# Train
python src/train.py            # GCN       → models/gcn_esol.pth
python src/train_graphsage.py  # GraphSAGE → models/graphsage_esol.pth

# Evaluate
python src/evaluate.py            # GCN metrics
python src/evaluate_graphsage.py  # GraphSAGE metrics
```
