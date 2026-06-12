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
| GraphSAGE | 3.1209 | 1.4160 | 1.7666 |
| GIN       | 0.7337 | 0.6876 | 0.8566 |

*Lower is better for all metrics.*

## Architecture Differences

### GCN (Kipf & Welling, 2017)
- **Aggregation:** symmetric normalised sum — `D^{-1/2} A D^{-1/2} X W`
- **Nature:** spectral / transductive; normalisation depends on the global degree matrix
- **Pooling:** `global_mean_pool`
- **Checkpoint:** `models/gcn_esol.pth`
- **Parameters:** ~8,900

### GraphSAGE (Hamilton et al., 2017)
- **Aggregation:** concatenate self-embedding with mean of neighbours — `W [h_v || mean(h_u)]`
- **Nature:** inductive; the update rule does not depend on the full graph degree, making it generalise to unseen molecules without re-training
- **Pooling:** `global_mean_pool`
- **Checkpoint:** `models/graphsage_esol.pth`
- **Parameters:** ~9,537

### GIN (Xu et al., 2019)
- **Aggregation:** sum of neighbours fed into a two-layer MLP — `MLP((1+ε)·h_v + Σ_{u∈N(v)} h_u)` with learnable ε
- **Nature:** inductive and maximally expressive under the Weisfeiler-Leman (WL) graph isomorphism test; sum aggregation preserves multiset information that mean aggregation discards
- **Pooling:** `global_add_pool` (sum, not mean — consistent with sum aggregation in message passing)
- **Per-layer MLP:** `Linear → BatchNorm1d → ReLU → Linear`, applied after each GINConv
- **Checkpoint:** `models/gin_esol.pth`
- **Parameters:** ~13,443

## Discussion

### Why GIN outperforms GCN and GraphSAGE by a wide margin

The results show a clear ranking — GIN (MSE 0.73) outperforms GraphSAGE (MSE 3.12) and GCN (MSE 3.39) by a factor of ~4× on mean squared error. This gap is larger than typical on small benchmarks and points to several compounding advantages:

**1. Sum aggregation vs. mean aggregation**  
GCN and GraphSAGE both reduce neighbour features by averaging (or a normalised variant thereof). Mean aggregation cannot distinguish two graphs where one node has two neighbours with feature `[1,0]` from one that has four such neighbours — both produce the same mean. Sum aggregation, used by GIN, preserves the count and therefore carries strictly more structural information. For solubility, where atom count and substitution pattern matter physically (e.g. the number of hydroxyl groups directly affects polarity), this expressiveness gap translates to real predictive accuracy.

**2. MLP per layer vs. single linear transform**  
Each GINConv layer applies a two-layer MLP with BatchNorm and ReLU after aggregation, giving the network non-linear mixing *within* each message-passing step. GCN and GraphSAGE apply a single linear transform per layer; non-linearity only appears between layers. The per-layer MLP significantly increases the representational capacity per hop without adding extra layers to the network.

**3. Learnable ε**  
The `train_eps=True` setting lets the model independently tune how much weight to give the node's own features versus its aggregated neighbourhood — a degree of freedom that GCN and GraphSAGE lack. This is especially relevant for atoms (like ring-membership markers) whose self-features are highly informative.

**4. BatchNorm inside the MLP**  
BatchNorm normalises the intermediate node-feature distribution within each GINConv, acting as an implicit regulariser and smoothing the loss surface. This explains why GIN's training loss descends faster and more stably (reaching ~0.70 by epoch 18) compared to GCN and GraphSAGE.

**5. Pooling choice: sum vs. mean**  
Using `global_add_pool` rather than `global_mean_pool` preserves information about molecular size at the graph-level embedding stage. Molecules with more atoms (and thus more bonds / polar groups) tend to have different solubility distributions than small molecules. Mean pooling discards this size signal; sum pooling retains it.

### GCN vs. GraphSAGE

The gap between GCN (MSE 3.39) and GraphSAGE (MSE 3.12) is modest (~8% relative improvement). The inductive mean-aggregation of GraphSAGE learns separate weights for the self-embedding and the neighbourhood mean, giving it one additional degree of freedom over GCN's symmetric normalised sum. This small advantage is consistent with the literature on small molecular datasets.

### Practical takeaway

GIN is the strongest architecture for this task under the controlled training budget. Its theoretical expressiveness advantage (proven by Xu et al. to be equivalent to the 1-WL test) manifests practically as nearly 4× lower error. For production solubility prediction, GIN should be the default choice. GCN and GraphSAGE remain useful as fast baselines with fewer parameters when computational budget is the primary constraint.

## Reproducing Results

```bash
# Train
python src/train.py            # GCN       → models/gcn_esol.pth
python src/train_graphsage.py  # GraphSAGE → models/graphsage_esol.pth
python src/train_gin.py        # GIN       → models/gin_esol.pth

# Evaluate
python src/evaluate.py            # GCN       — MSE=3.3883, MAE=1.4526, RMSE=1.8407
python src/evaluate_graphsage.py  # GraphSAGE — MSE=3.1209, MAE=1.4160, RMSE=1.7666
python src/evaluate_gin.py        # GIN       — MSE=0.7337, MAE=0.6876, RMSE=0.8566
```

> **Note on GIN training time:** `train_gin.py` saves a checkpoint after every
> epoch. If the process is interrupted before all 20 epochs complete, the last
> saved checkpoint (from the most recently finished epoch) is automatically
> available at `models/gin_esol.pth` and is ready for evaluation without
> re-running from scratch. GIN converges reliably by epoch 15 on this dataset.
