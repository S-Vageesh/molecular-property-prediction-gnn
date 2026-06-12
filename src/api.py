"""FastAPI service exposing the trained GCN molecular property predictor.

Endpoints
---------
GET  /           — service banner
GET  /health      — liveness check
POST /predict     — GCN solubility prediction from a SMILES string
POST /visualize   — 2D molecule image generation from a SMILES string
POST /analyze     — combined prediction + visualization in one call

Design notes
------------
- The GCN model is loaded **once** at application startup via FastAPI's
  lifespan context manager and stored in a module-level cache. This avoids
  the overhead of reading the checkpoint file on every request.
- SMILES validation is delegated to smiles_to_graph() (via RDKit) so that
  both the CLI and the API share the same validation path.
- Invalid SMILES always produce HTTP 400 with a descriptive detail message.
- The architecture (GCNModel) is never modified here. Inference is read-only:
  torch.no_grad() is used throughout and model.eval() is set at load time.
- All shared logic lives in two private helpers (_infer_solubility and
  _generate_image_path) so /predict, /visualize, and /analyze never
  duplicate code.

Running
-------
From the project root:

    uvicorn src.api:app --reload --host 0.0.0.0 --port 5000

Interactive API docs are available at:
    http://localhost:5000/docs   (Swagger UI)
    http://localhost:5000/redoc  (ReDoc)
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from torch_geometric.loader import DataLoader


# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
# All src/ scripts resolve imports by adding the project root and src/ to
# sys.path. We follow the same convention so that api.py can import from
# sibling modules (smiles_predict, molecule_visualizer, predict) regardless
# of the working directory from which uvicorn is invoked.

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Local module imports — placed after sys.path setup.
from smiles_predict import smiles_to_graph, MOLECULENET_ESOL_NODE_FEATURES  # noqa: E402
from predict import load_model  # noqa: E402
from evaluate_gin import load_gin_model  # noqa: E402
from molecule_visualizer import visualize_molecule  # noqa: E402
from explain_gin import explain_prediction  # noqa: E402
from explanation_visualizer import visualize_explanation  # noqa: E402


# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
# Populated during lifespan startup; referenced by every inference endpoint.
# Using a plain dict avoids global-statement linting noise.

_cache: dict = {
    "device": None,  # torch.device — cpu or cuda
    "gcn_model": None,   # torch.nn.Module — the loaded GCN
    "gin_model": None,   # torch.nn.Module — the loaded GIN
}


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the models once when the server starts, release on shutdown.

    Using the lifespan pattern (rather than @app.on_event) is the recommended
    approach in FastAPI ≥ 0.93. The models are loaded before the server begins
    serving requests so the first inference call has no extra latency.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load_model reads models/gcn_esol.pth
    gcn_model = load_model(device, MOLECULENET_ESOL_NODE_FEATURES)
    
    # load_gin_model reads models/gin_esol.pth
    try:
        gin_model = load_gin_model(device, MOLECULENET_ESOL_NODE_FEATURES)
    except FileNotFoundError:
        print("[startup] WARNING: GIN model not found. /explain will be unavailable.")
        gin_model = None

    _cache["device"] = device
    _cache["gcn_model"] = gcn_model
    _cache["gin_model"] = gin_model

    print(f"[startup] Models loaded on {device}")
    print(f"[startup] GCN path: {PROJECT_ROOT / 'models' / 'gcn_esol.pth'}")
    if gin_model:
        print(f"[startup] GIN path: {PROJECT_ROOT / 'models' / 'gin_esol.pth'}")

    yield  # Server handles requests between here and the code below.

    # Shutdown: clear cache for clean test isolation and process shutdown.
    _cache["device"] = None
    _cache["gcn_model"] = None
    _cache["gin_model"] = None
    print("[shutdown] Model cache cleared.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Molecular Property Prediction API",
    description=(
        "A REST API that exposes GNN models trained on the ESOL dataset to "
        "predict aqueous solubility (log mol/L) from SMILES. Includes GCN "
        "and GIN architectures, with GNNExplainer support for GIN.\n\n"
        "**Quickstart:** Use `POST /analyze` for GCN results, or `POST /explain` "
        "to see which atoms influence the GIN prediction."
    ),
    version="1.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class SMILESRequest(BaseModel):
    """Shared request body for all endpoints that accept a SMILES string."""

    smiles: str = Field(
        ...,
        description="SMILES string representing the molecule (e.g. 'CCO' for ethanol).",
        examples=["CCO"],
        min_length=1,
    )


class PredictResponse(BaseModel):
    """Response body returned by POST /predict."""

    smiles: str = Field(
        description="The original SMILES string supplied by the caller."
    )
    predicted_solubility: float = Field(
        description=(
            "GCN-predicted aqueous solubility in log(mol/L). "
            "More negative values mean lower solubility."
        )
    )


class VisualizeResponse(BaseModel):
    """Response body returned by POST /visualize."""

    image_path: str = Field(
        description=(
            "Relative path (from the project root) to the saved PNG file, "
            "e.g. 'generated_molecules/CCO.png'."
        )
    )


class AnalyzeResponse(BaseModel):
    """Response body returned by POST /analyze."""

    smiles: str = Field(
        description="The original SMILES string supplied by the caller."
    )
    predicted_solubility: float = Field(
        description=(
            "GCN-predicted aqueous solubility in log(mol/L). "
            "More negative values mean lower solubility."
        )
    )
    image_path: str = Field(
        description=(
            "Relative path (from the project root) to the saved PNG file, "
            "e.g. 'generated_molecules/CCO.png'."
        )
    )


class ExplainResponse(BaseModel):
    """Response body returned by POST /explain."""

    smiles: str = Field(
        description="The original SMILES string supplied by the caller."
    )
    prediction: float = Field(
        description="GIN-predicted aqueous solubility in log(mol/L)."
    )
    explanation_image: str = Field(
        description=(
            "Relative path to the GNNExplainer visualization, "
            "e.g. 'generated_explanations/benzene_explanation.png'."
        )
    )


class HealthResponse(BaseModel):
    """Response body returned by GET /health."""

    status: str = Field(description="Always 'healthy' when the server is up.")


class RootResponse(BaseModel):
    """Response body returned by GET /."""

    message: str


# ---------------------------------------------------------------------------
# Private helpers — shared inference and visualization logic
# ---------------------------------------------------------------------------

def _require_gcn() -> None:
    """Raise HTTP 503 if the GCN model was not loaded at startup."""
    if _cache["gcn_model"] is None:
        raise HTTPException(
            status_code=503,
            detail="GCN model is not available.",
        )


def _require_gin() -> None:
    """Raise HTTP 503 if the GIN model was not loaded at startup."""
    if _cache["gin_model"] is None:
        raise HTTPException(
            status_code=503,
            detail="GIN model is not available.",
        )


def _infer_solubility(smiles: str) -> float:
    """Validate a SMILES string and run GCN inference; return log solubility.

    This is the single source of truth for inference logic. Both /predict and
    /analyze call this function so the forward-pass code is never duplicated.

    Parameters
    ----------
    smiles:
        Raw SMILES string from the request body.

    Returns
    -------
    float
        Predicted aqueous solubility in log(mol/L), rounded to 4 decimal
        places.

    Raises
    ------
    HTTPException (400)
        If ``smiles`` is empty or cannot be parsed as a valid molecule.
    """

    # smiles_to_graph() validates with RDKit, canonicalizes, and builds a PyG
    # Data object. It raises ValueError for any invalid molecule string.
    try:
        graph = smiles_to_graph(smiles)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    device: torch.device = _cache["device"]
    model: torch.nn.Module = _cache["gcn_model"]

    # DataLoader wraps the single graph in a mini-batch and adds the batch
    # assignment vector that global_mean_pool expects.
    loader = DataLoader([graph], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)

    # no_grad() ensures inference never updates model weights or stores
    # gradient state. model.eval() was already called by load_model().
    with torch.no_grad():
        raw = model(
            batch.x.float(),
            batch.edge_index,
            batch.batch,
        ).view(-1)[0].item()

    return round(raw, 4)


def _generate_image_path(smiles: str) -> str:
    """Validate a SMILES string, render a 2D image, and return its relative path.

    This is the single source of truth for visualization logic. Both
    /visualize and /analyze call this function so the image-generation code
    is never duplicated.

    Parameters
    ----------
    smiles:
        Raw SMILES string from the request body.

    Returns
    -------
    str
        Relative path from the project root to the saved PNG, e.g.
        ``'generated_molecules/CCO.png'``.

    Raises
    ------
    HTTPException (400)
        If ``smiles`` is empty or cannot be parsed as a valid molecule.
    """

    # visualize_molecule validates with RDKit, computes 2D coords, renders the
    # structure, saves the PNG, and returns the absolute Path.
    try:
        absolute_path = visualize_molecule(smiles)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Return a project-root-relative string so callers are not tied to the
    # server's absolute filesystem layout.
    return str(absolute_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    response_model=RootResponse,
    summary="Service banner",
    tags=["meta"],
)
async def root() -> RootResponse:
    """Return a simple banner confirming the API is reachable.

    This endpoint requires no authentication and performs no inference.
    It is useful as a quick smoke-test that the service is running.
    """
    return RootResponse(message="Molecular Property Prediction API")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    tags=["meta"],
)
async def health() -> HealthResponse:
    """Confirm that the server process is alive and the models are loaded.

    Returns HTTP 200 with ``{"status": "healthy"}`` when the GCN is ready.
    Returns HTTP 503 if startup failed or the model has not loaded yet.
    """
    _require_gcn()
    return HealthResponse(status="healthy")


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict aqueous solubility from SMILES (GCN)",
    tags=["inference"],
)
async def predict(request: SMILESRequest) -> PredictResponse:
    """Run GCN inference and return the predicted log solubility."""
    _require_gcn()
    solubility = _infer_solubility(request.smiles)
    return PredictResponse(smiles=request.smiles, predicted_solubility=solubility)


@app.post(
    "/visualize",
    response_model=VisualizeResponse,
    summary="Generate a 2D molecule image from SMILES",
    tags=["visualization"],
)
async def visualize(request: SMILESRequest) -> VisualizeResponse:
    """Render a 2D molecular structure image and save it as a PNG."""
    image_path = _generate_image_path(request.smiles)
    return VisualizeResponse(image_path=image_path)


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Predict solubility (GCN) and generate molecule image in one call",
    tags=["inference", "visualization"],
)
async def analyze(request: SMILESRequest) -> AnalyzeResponse:
    """Run GCN prediction and 2D visualization together."""
    _require_gcn()
    solubility = _infer_solubility(request.smiles)
    image_path = _generate_image_path(request.smiles)

    return AnalyzeResponse(
        smiles=request.smiles,
        predicted_solubility=solubility,
        image_path=image_path,
    )


@app.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Generate GNNExplainer visualization for GIN prediction",
    tags=["explainability"],
)
async def explain(request: SMILESRequest) -> ExplainResponse:
    """Explain a GIN solubility prediction using GNNExplainer.

    This endpoint:
    1. Runs the GIN model to predict solubility.
    2. Uses GNNExplainer to find the most influential atoms and features.
    3. Produces a 2D visualization where influential atoms are highlighted in red.
    4. Returns the predicted value and the path to the explanation image.

    **Example request:**
    ```json
    { "smiles": "c1ccccc1" }
    ```

    - **Valid input** → HTTP 200 with prediction + explanation image path
    - **Invalid SMILES** → HTTP 400 with a descriptive error message
    - **GIN missing** → HTTP 503 if `models/gin_esol.pth` is not available
    """
    _require_gin()

    try:
        # explain_prediction performs inference and runs GNNExplainer
        prediction, explanation = explain_prediction(request.smiles)
        
        # node_mask identifies atom importance
        node_importance = explanation.node_mask.sum(dim=1)
        
        # visualize_explanation renders the highlighted PNG
        image_path = visualize_explanation(request.smiles, node_importance)
        
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(exc)}")

    return ExplainResponse(
        smiles=request.smiles,
        prediction=round(prediction, 4),
        explanation_image=str(image_path.relative_to(PROJECT_ROOT)),
    )
