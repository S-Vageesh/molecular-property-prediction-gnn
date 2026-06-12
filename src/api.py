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
from molecule_visualizer import visualize_molecule  # noqa: E402


# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
# Populated during lifespan startup; referenced by every inference endpoint.
# Using a plain dict avoids global-statement linting noise.

_cache: dict = {
    "device": None,  # torch.device — cpu or cuda
    "model": None,   # torch.nn.Module — the loaded GCN
}


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the GCN model once when the server starts, release on shutdown.

    Using the lifespan pattern (rather than @app.on_event) is the recommended
    approach in FastAPI ≥ 0.93. The model is loaded before the server begins
    serving requests so the first inference call has no extra latency.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load_model reads models/gcn_esol.pth, infers the checkpoint format,
    # instantiates GCNModel with matching dimensions, calls model.eval(), and
    # moves the model to the selected device — all in one call.
    model = load_model(device, MOLECULENET_ESOL_NODE_FEATURES)

    _cache["device"] = device
    _cache["model"] = model

    print(f"[startup] GCN model loaded on {device}")
    print(f"[startup] Model path: {PROJECT_ROOT / 'models' / 'gcn_esol.pth'}")

    yield  # Server handles requests between here and the code below.

    # Shutdown: clear cache for clean test isolation and process shutdown.
    _cache["device"] = None
    _cache["model"] = None
    print("[shutdown] Model cache cleared.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Molecular Property Prediction API",
    description=(
        "A REST API that exposes a Graph Convolutional Network (GCN) trained "
        "on the ESOL dataset to predict aqueous solubility (log mol/L) from "
        "SMILES molecular representations. Also generates 2D structure images.\n\n"
        "**Quickstart:** Use `POST /analyze` to get both prediction and image in "
        "one request — ideal for frontend consumption."
    ),
    version="1.1.0",
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
    """Response body returned by POST /analyze.

    Combines the outputs of /predict and /visualize so a frontend can
    retrieve everything it needs — solubility value and molecule image path
    — with a single HTTP request instead of two sequential calls.
    """

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


class HealthResponse(BaseModel):
    """Response body returned by GET /health."""

    status: str = Field(description="Always 'healthy' when the server is up.")


class RootResponse(BaseModel):
    """Response body returned by GET /."""

    message: str


# ---------------------------------------------------------------------------
# Private helpers — shared inference and visualization logic
# ---------------------------------------------------------------------------

def _require_model() -> None:
    """Raise HTTP 503 if the model cache was not populated at startup."""
    if _cache["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. The server may still be starting up.",
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
    model: torch.nn.Module = _cache["model"]

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
    """Confirm that the server process is alive and the GCN model is loaded.

    Returns HTTP 200 with ``{"status": "healthy"}`` when the model is ready.
    Returns HTTP 503 if startup failed or the model has not loaded yet.
    Suitable for use as a Kubernetes/Docker liveness or readiness probe.
    """
    _require_model()
    return HealthResponse(status="healthy")


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict aqueous solubility from SMILES",
    tags=["inference"],
)
async def predict(request: SMILESRequest) -> PredictResponse:
    """Run GCN inference and return the predicted log solubility.

    The SMILES string is validated with RDKit before the model is touched.
    The cached GCN model is used — no checkpoint file is read per request.

    - **Valid input** → HTTP 200 with predicted solubility
    - **Invalid SMILES** → HTTP 400 with a descriptive error message
    - **Empty string** → HTTP 422 (caught by Pydantic `min_length=1`)

    > Tip: Use `POST /analyze` to get both the solubility and molecule image
    > in a single request.
    """
    _require_model()
    solubility = _infer_solubility(request.smiles)
    return PredictResponse(smiles=request.smiles, predicted_solubility=solubility)


@app.post(
    "/visualize",
    response_model=VisualizeResponse,
    summary="Generate a 2D molecule image from SMILES",
    tags=["visualization"],
)
async def visualize(request: SMILESRequest) -> VisualizeResponse:
    """Render a 2D molecular structure image and save it as a PNG.

    Delegates to `molecule_visualizer.visualize_molecule` which handles
    RDKit parsing, 2D coordinate generation, and PNG export. The image is
    saved under `generated_molecules/` in the project root.

    - **Valid input** → HTTP 200 with relative image path
    - **Invalid SMILES** → HTTP 400 with a descriptive error message
    - **Empty string** → HTTP 422 (caught by Pydantic `min_length=1`)

    > Tip: Use `POST /analyze` to get both the solubility and molecule image
    > in a single request.
    """
    image_path = _generate_image_path(request.smiles)
    return VisualizeResponse(image_path=image_path)


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Predict solubility and generate molecule image in one call",
    tags=["inference", "visualization"],
)
async def analyze(request: SMILESRequest) -> AnalyzeResponse:
    """Run GCN prediction and 2D visualization together; return both results.

    This is the recommended endpoint for frontend clients. Instead of making
    two sequential requests (`/predict` then `/visualize`), a single call to
    `/analyze` returns everything needed to render a result card:

    - `smiles` — the input molecule
    - `predicted_solubility` — GCN log solubility in mol/L
    - `image_path` — relative path to the saved PNG structure image

    Both operations reuse the same shared helpers used by `/predict` and
    `/visualize`, so there is no duplicated logic.

    **Example request:**
    ```json
    { "smiles": "CCO" }
    ```

    **Example response:**
    ```json
    {
      "smiles": "CCO",
      "predicted_solubility": -2.2591,
      "image_path": "generated_molecules/CCO.png"
    }
    ```

    - **Valid input** → HTTP 200 with solubility + image path
    - **Invalid SMILES** → HTTP 400 with a descriptive error message
    - **Empty string** → HTTP 422 (caught by Pydantic `min_length=1`)
    """
    _require_model()

    # Both helpers share the same RDKit validation path. If the SMILES is
    # invalid, _infer_solubility raises HTTP 400 before _generate_image_path
    # is called, so we never waste time rendering an image for bad input.
    solubility = _infer_solubility(request.smiles)
    image_path = _generate_image_path(request.smiles)

    return AnalyzeResponse(
        smiles=request.smiles,
        predicted_solubility=solubility,
        image_path=image_path,
    )
