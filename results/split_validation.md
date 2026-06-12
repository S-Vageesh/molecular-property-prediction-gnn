# Train/Validation/Test Split Validation Report

This report analyzes the data splitting and evaluation logic used in the training scripts for GCN, GraphSAGE, and GIN models to ensure consistency, reproducibility, and integrity.

## Analyzed Scripts

- `src/train.py` (GCN)
- `src/train_graphsage.py`
- `src/train_gin.py`
- `src/evaluate.py`
- `src/evaluate_graphsage.py`
- `src/evaluate_gin.py`

## Findings

### 1. Dataset Consistency
All three training scripts load the ESOL dataset using the same `MoleculeNet` class and configuration:
```python
dataset = MoleculeNet(root=str(data_root), name="ESOL")
```
Where `data_root` is consistently resolved to the project's `data/` directory. Although the evaluation scripts use a slightly different local path (`data/ESOL/ESOL`), they load the same underlying source data, and the training scripts themselves are perfectly aligned.

### 2. Random Seed Consistency
All scripts use the exact same random seed for the dataset split:
```python
generator=torch.Generator().manual_seed(42)
```
By using a local `torch.Generator` object with a fixed seed, the split remains deterministic and reproducible across all scripts, regardless of other random operations in the environment.

### 3. Split Ratio and Size Consistency
The train/validation/test split is consistently set to **80% / 10% / 10%**. The size calculations are identical across all scripts:
```python
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size
```
For the ESOL dataset (1128 molecules), this results in:
- **Train**: 902 molecules
- **Validation**: 112 molecules
- **Test**: 114 molecules

### 4. Data Leakage Analysis
No data leakage was found. The following practices ensure data integrity:
- **Disjoint Subsets**: `torch.utils.data.random_split` creates mutually exclusive subsets of the data.
- **Independent Loaders**: Separate `DataLoader` objects are created for each subset.
- **Evaluation Mode**: All evaluation calls (`evaluate` function and inference-only scripts) use `model.eval()` and `torch.no_grad()` to ensure no gradients are computed or weights updated during evaluation.
- **No Shared State**: The models are independent, and data is passed through the models without in-place modifications that could affect other splits.

### 5. GIN Evaluation Integrity
A specific check was performed on `src/train_gin.py` and `src/evaluate_gin.py` to ensure the GIN model is not accidentally evaluating on training data:
- In `src/train_gin.py`, the training loop correctly uses `train_loader` for updates and `val_loader` for validation.
- The final test loss in `src/train_gin.py` correctly uses `test_loader`.
- `src/evaluate_gin.py` correctly loads the `test_dataset` via the shared `load_esol_test_set()` helper.

## Conclusion

The data splitting and validation logic is **consistent and robust** across all models. All architectures (GCN, GraphSAGE, and GIN) are trained and evaluated on the exact same molecules, ensuring that performance comparisons reflect architectural differences rather than data variations.
