# Molecular Property Prediction using Graph Neural Networks

Predict molecular properties from chemical graph structures using Graph Neural Networks (GNNs). This project explores how graph-based deep learning can capture structural information in molecules to support property prediction tasks.

## Tech Stack

- **PyTorch** — deep learning framework
- **PyTorch Geometric** — graph neural network library
- **RDKit** — cheminformatics and molecular graph construction
- **pandas** — data manipulation and analysis
- **NumPy** — numerical computing
- **scikit-learn** — machine learning utilities and evaluation
- **matplotlib** — visualization
- **NetworkX** — graph analysis and utilities

## Project Structure

```
molecular-property-prediction-gnn/
├── data/           # Raw and processed datasets
├── notebooks/      # Exploratory analysis and experiments
├── src/            # Source code (data loading, training, utilities)
├── models/         # Saved model checkpoints and artifacts
├── tests/          # Unit and integration tests
├── requirements.txt
└── README.md
```

## Status

Project setup in progress.

## Model Explainability

The project includes an explainability pipeline for the **GIN (Graph Isomorphism Network)** model using **PyTorch Geometric GNNExplainer**.

This allows users to identify which atoms in a molecule most strongly influenced the model's solubility prediction.

### Features
- **GNNExplainer Integration**: Identifies influential nodes (atoms) and edges (bonds) via subgraph optimization.
- **Heatmap Visualization**: Highlights important atoms in red on a 2D molecular diagram.
- **API Support**: A dedicated `/explain` endpoint for real-time explanations.

### Usage

**Python API:**
```python
from src.explain_gin import explain_prediction
from src.explanation_visualizer import visualize_explanation

prediction, explanation = explain_prediction("c1ccccc1")
node_importance = explanation.node_mask.sum(dim=1)
image_path = visualize_explanation("c1ccccc1", node_importance)
print(f"Explanation saved to: {image_path}")
```

**REST API:**
```bash
curl -X POST "http://localhost:5000/explain" \
     -H "Content-Type: application/json" \
     -d '{"smiles": "c1ccccc1"}'
```

The response includes the prediction and a path to the generated PNG:
```json
{
  "smiles": "c1ccccc1",
  "prediction": -2.84,
  "explanation_image": "generated_explanations/c1ccccc1_explanation.png"
}
```
