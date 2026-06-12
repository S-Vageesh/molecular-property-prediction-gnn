# Molecular Property Prediction Platform Roadmap

This roadmap outlines the path from a working Graph Neural Network prototype to a production-quality Molecular Property Prediction Platform. The platform centers on molecular graph learning, reproducible model evaluation, reliable prediction services, and a polished user-facing experience.

## Phase 1 - Core GCN System

### Goals

- Establish the foundational machine learning workflow for molecular property prediction.
- Build a clear, reproducible GCN pipeline using the ESOL dataset.
- Verify that training, evaluation, and inference work end to end.

### Deliverables

- Dataset loading pipeline for ESOL.
- Graph inspection utilities for molecular graph structure, node features, edge indices, and labels.
- GCN model implementation.
- Training script with loss tracking and checkpoint saving.
- Evaluation script reporting MSE, MAE, and RMSE.
- Inference script that loads a trained model and predicts solubility for a test molecule.

### Success Criteria

- ESOL loads successfully without manual preprocessing.
- Molecular graphs can be inspected and validated.
- GCN model trains without runtime errors.
- Trained model checkpoint is saved to `models/gcn_esol.pth`.
- Evaluation produces clear regression metrics on the test set.
- Inference can load the saved model without retraining.

### Estimated Complexity

Medium

## Phase 2 - Molecular Prediction Service

### Goals

- Extend prediction beyond the static ESOL test set.
- Allow users to submit arbitrary SMILES strings for molecular property prediction.
- Convert user-provided molecules into graph inputs compatible with the trained model.

### Deliverables

- SMILES-based prediction module.
- RDKit graph construction pipeline.
- Molecule rendering utility for visual inspection.
- Validation and error handling for invalid molecules.
- Clear user-facing error messages for malformed SMILES or unsupported structures.

### Success Criteria

- Valid SMILES strings are converted into PyTorch Geometric graph objects.
- Predictions can be generated for molecules outside the ESOL dataset.
- Invalid SMILES strings fail gracefully without crashing the application.
- Rendered molecule images match the submitted molecular structures.

### Estimated Complexity

High

## Phase 3 - Model Benchmarking

### Goals

- Compare multiple graph neural network architectures under consistent conditions.
- Identify the strongest baseline model for ESOL-style molecular property prediction.
- Build confidence in model selection through reproducible benchmarking.

### Deliverables

- GCN benchmark implementation.
- GraphSAGE benchmark implementation.
- GIN benchmark implementation.
- Shared training and evaluation loop for fair comparison.
- Comparison table covering MSE, MAE, RMSE, training time, and model size.
- Performance analysis summarizing tradeoffs between architectures.

### Success Criteria

- All benchmarked models use the same dataset split.
- Metrics are computed consistently across architectures.
- Results are captured in a readable comparison table.
- Performance analysis explains which model should be used and why.

### Estimated Complexity

High

## Phase 4 - Explainability

### Goals

- Make model predictions more interpretable.
- Help users understand which atoms and bonds influenced a prediction.
- Provide explainability outputs suitable for both debugging and presentation.

### Deliverables

- GNNExplainer integration.
- Node importance visualization for atoms.
- Edge importance visualization for bonds.
- Explainability report for selected molecules.
- Utilities for mapping graph explanations back to molecular structures.

### Success Criteria

- Explanations can be generated for individual predictions.
- Important nodes and edges are visually distinguishable.
- Explainability outputs are stable enough for demos and documentation.
- Visualizations help users reason about model behavior.

### Estimated Complexity

High

## Phase 5 - Backend API

### Goals

- Expose molecular prediction functionality through a reliable backend service.
- Provide clean REST endpoints for prediction, health checks, and model metadata.
- Prepare the platform for frontend and deployment integration.

### Deliverables

- FastAPI application.
- REST endpoint for SMILES prediction.
- REST endpoint for model metadata.
- Health check endpoint.
- Model loading and serving layer.
- Request and response schemas.
- API error handling and validation.

### Success Criteria

- API starts reliably in a local development environment.
- Prediction endpoint returns structured JSON responses.
- Invalid input returns appropriate HTTP errors.
- Model is loaded once at service startup rather than per request.
- Health check endpoint confirms service readiness.

### Estimated Complexity

Medium

## Phase 6 - React Frontend

### Goals

- Build a modern interface for interacting with the molecular prediction platform.
- Let users submit molecules, view predictions, inspect model metrics, and explore molecule visuals.
- Present the system as a polished, portfolio-quality application.

### Deliverables

- Modern dashboard layout.
- Molecule viewer.
- Prediction interface for SMILES input.
- Metrics visualization for model performance.
- Prediction history or recent results panel.
- Loading, empty, success, and error states.

### Success Criteria

- Users can submit SMILES strings and receive predictions from the backend.
- Molecule viewer displays submitted structures clearly.
- Metrics are easy to scan and understand.
- UI handles invalid input and backend errors gracefully.
- Frontend feels complete enough for demos.

### Estimated Complexity

High

## Phase 7 - Deployment

### Goals

- Package the platform for reproducible local and cloud deployment.
- Make setup simple for reviewers, collaborators, and portfolio visitors.
- Add automation to reduce deployment risk.

### Deliverables

- Dockerfile for backend service.
- Dockerfile or build configuration for frontend.
- Docker Compose setup for local full-stack execution.
- Cloud deployment configuration.
- CI/CD workflow for testing and deployment.
- Environment variable documentation.

### Success Criteria

- Full platform can run locally through Docker Compose.
- Cloud deployment can serve both frontend and backend.
- CI checks run automatically on changes.
- Deployment steps are documented and repeatable.
- Secrets and environment-specific configuration are not hardcoded.

### Estimated Complexity

High

## Phase 8 - Portfolio Polish

### Goals

- Present the project as a professional, production-oriented machine learning platform.
- Make the system easy to understand through visuals, documentation, and demos.
- Highlight engineering quality, ML workflow design, and product thinking.

### Deliverables

- Screenshots of the dashboard, molecule viewer, prediction results, and metrics views.
- Architecture diagrams covering data flow, model serving, frontend, and deployment.
- Demo GIFs showing prediction workflow and explainability views.
- Technical documentation for setup, training, evaluation, API usage, and deployment.
- README refresh with project overview, features, stack, and roadmap status.

### Success Criteria

- A new visitor can understand the project purpose within one minute.
- Setup instructions allow the project to run from a fresh clone.
- Documentation clearly separates training, evaluation, inference, API, and frontend workflows.
- Visual assets demonstrate the platform without requiring live execution.
- Portfolio materials communicate both ML depth and software engineering maturity.

### Estimated Complexity

Medium
