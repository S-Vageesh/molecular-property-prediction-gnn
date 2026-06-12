"""FastAPI service exposing the trained GCN molecular property predictor.

Endpoints
---------
GET  /           — service banner
GET  /health      — liveness check
POST /predict     — GCN solubility prediction from a SMILES string
POST /visualize   — 2D molecule image generation from a SMILES string

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

Running
-------
From the project root:

    uvicorn src.api:app --reload --host 0.0.0.0 --port 5000

Or with explicit host/port for production:

    uvicorn src.api:app --host 0.0.0.0 --port 5000

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
# These are populated during the lifespan startup event and referenced by the
# /predict endpoint. Using a plain dict avoids global-statement linting noise.

_cache: dict = {
    "device": None,   # torch.device — cpu or cuda
    "model": None,    # torch.nn.Module — the loaded GCN
}


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the GCN model once when the server starts, release on shutdown.

    Using the lifespan pattern (rather than @app.on_event) is the recommended
    approach in FastAPI ≥ 0.93. The model is loaded before the server begins
    serving requests so the first /predict call has no extra latency.
    """

    # Choose GPU if one is available; otherwise fall back to CPU transparently.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load_model reads models/gcn_esol.pth, infers the checkpoint format,
    # instantiates GCNModel with matching dimensions, calls model.eval(), and
    # moves the model to the selected device — all in one call.
    model = load_model(device, MOLECULENET_ESOL_NODE_FEATURES)

    _cache["device"] = device
    _cache["model"] = model

    print(f"[startup] GCN model loaded on {device}")
    print(f"[startup] Model path: {PROJECT_ROOT / 'models' / 'gcn_esol.pth'}")

    # Yield control to FastAPI — the server serves requests between here and
    # the code after yield (which runs on shutdown).
    yield

    # Shutdown: nothing explicit to release for a CPU/GPU PyTorch model, but
    # clearing the cache is good practice for test isolation.
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
        "SMILES molecular representations. Also generates 2D structure images."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class SMILESRequest(BaseModel):
    """Shared request body for endpoints that accept a SMILES string."""

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


class HealthResponse(BaseModel):
    """Response body returned by GET /health."""

    status: str = Field(description="Always 'healthy' when the server is up.")


class RootResponse(BaseModel):
    """Response body returned by GET /."""

    message: str


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
    """Confirm that the server process is alive and the model is loaded.

    Returns HTTP 200 with ``{"status": "healthy"}`` when the GCN model has
    been loaded successfully. This endpoint is suitable for use as a
    Kubernetes/Docker liveness or readiness probe.
    """
    # If lifespan startup failed, _cache["model"] would be None. We surface
    # that as a 503 rather than silently returning a misleading "healthy".
    if _cache["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. The server may still be starting up.",
        )
    return HealthResponse(status="healthy")


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict aqueous solubility from SMILES",
    tags=["inference"],
)
async def predict(request: SMILESRequest) -> PredictResponse:
    """Run GCN inference and return the predicted log solubility.

    The SMILES string is first validated with RDKit; any syntactically or
    chemically invalid input returns HTTP 400 before the model is touched.
    The cached GCN model is used — no checkpoint file is read per request.

    Parameters
    ----------
    request:
        JSON body containing a ``smiles`` field.

    Returns
    -------
    PredictResponse
        The original SMILES and the predicted solubility as a float.

    Raises
    ------
    HTTPException (400)
        If the SMILES string is empty or cannot be parsed by RDKit.
    HTTPException (503)
        If the model was not loaded at startup (should not happen in normal
        operation).
    """

    # Guard: model must be available (populated by lifespan startup).
    if _cache["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not available.")

    # --- SMILES validation and graph construction ----------------------------
    # smiles_to_graph() runs RDKit's Chem.MolFromSmiles, canonicalizes the
    # SMILES, and builds a PyTorch Geometric Data object. It raises ValueError
    # for any input that is not a valid molecule.
    try:
        graph = smiles_to_graph(request.smiles)
    except ValueError as exc:
        # Translate the domain-level ValueError into an HTTP 400 response so
        # clients receive a standard error format with a descriptive message.
        raise HTTPException(status_code=400, detail=str(exc))

    device: torch.device = _cache["device"]
    model: torch.nn.Module = _cache["model"]

    # --- Batching and inference ----------------------------------------------
    # DataLoader wraps the single-molecule graph in a mini-batch and adds the
    # batch assignment vector required by global_mean_pool in the GCN.
    loader = DataLoader([graph], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)

    # no_grad() guarantees no gradients are computed or stored — inference
    # only. The model is already in eval() mode from startup.
    with torch.no_grad():
        raw_prediction = model(
            batch.x.float(),
            batch.edge_index,
            batch.batch,
        ).view(-1)[0].item()

    return PredictResponse(
        smiles=request.smiles,
        predicted_solubility=round(raw_prediction, 4),
    )


@app.post(
    "/visualize",
    response_model=VisualizeResponse,
    summary="Generate a 2D molecule image from SMILES",
    tags=["visualization"],
)
async def visualize(request: SMILESRequest) -> VisualizeResponse:
    """Render a 2D molecular structure image and save it as a PNG.

    Delegates to ``molecule_visualizer.visualize_molecule`` which handles
    RDKit parsing, 2D coordinate generation, and PNG export. The image is
    saved under ``generated_molecules/`` in the project root and the
    relative path is returned to the caller.

    Parameters
    ----------
    request:
        JSON body containing a ``smiles`` field.

    Returns
    -------
    VisualizeResponse
        Relative path to the saved PNG file.

    Raises
    ------
    HTTPException (400)
        If the SMILES string is empty or cannot be parsed by RDKit.
    """

    # visualize_molecule validates with RDKit, computes 2D coords, renders the
    # structure, saves the PNG, and returns the absolute Path. ValueError is
    # raised for any invalid input.
    try:
        absolute_path = visualize_molecule(request.smiles)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Return a path relative to the project root so the response is portable
    # and not tied to the absolute filesystem layout of the server.
    relative_path = absolute_path.relative_to(PROJECT_ROOT)

    return VisualizeResponse(image_path=str(relative_path))
