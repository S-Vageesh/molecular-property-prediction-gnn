"""Tests for the FastAPI molecular property prediction service.

Covers all four endpoints with happy-path, validation, and error cases.
Uses FastAPI's TestClient (synchronous wrapper around httpx) so no async
test runner is needed — plain pytest works out of the box.

The TestClient triggers the full lifespan (model load on enter, cache clear
on exit), so these are integration tests that exercise the real GCN model
and RDKit validation, not mocks.

Running
-------
From the project root:

    pytest tests/test_api.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# sys.path — mirror the same setup used in src/api.py so the test process
# can resolve all sibling modules (smiles_predict, predict, molecule_visualizer)
# regardless of how pytest is invoked.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.api import app  # noqa: E402


# ---------------------------------------------------------------------------
# Shared client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Yield a TestClient that loads the GCN model once for the entire module.

    ``scope="module"`` means the lifespan (and therefore model load) runs
    only once per test module, keeping the test suite fast.
    """
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestRoot:
    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200

    def test_response_shape(self, client: TestClient) -> None:
        data = client.get("/").json()
        assert set(data.keys()) == {"message"}

    def test_message_content(self, client: TestClient) -> None:
        data = client.get("/").json()
        assert data["message"] == "Molecular Property Prediction API"


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_shape(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert set(data.keys()) == {"status"}

    def test_status_value(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "healthy"


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_valid_ethanol(self, client: TestClient) -> None:
        """CCO (ethanol) should produce a valid solubility float."""
        response = client.post("/predict", json={"smiles": "CCO"})
        assert response.status_code == 200

    def test_response_shape(self, client: TestClient) -> None:
        data = client.post("/predict", json={"smiles": "CCO"}).json()
        assert set(data.keys()) == {"smiles", "predicted_solubility"}

    def test_smiles_echoed(self, client: TestClient) -> None:
        data = client.post("/predict", json={"smiles": "CCO"}).json()
        assert data["smiles"] == "CCO"

    def test_solubility_is_float(self, client: TestClient) -> None:
        data = client.post("/predict", json={"smiles": "CCO"}).json()
        assert isinstance(data["predicted_solubility"], float)

    def test_valid_benzene(self, client: TestClient) -> None:
        response = client.post("/predict", json={"smiles": "c1ccccc1"})
        assert response.status_code == 200
        assert isinstance(response.json()["predicted_solubility"], float)

    def test_valid_aspirin(self, client: TestClient) -> None:
        response = client.post("/predict", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
        assert response.status_code == 200

    def test_invalid_smiles_returns_400(self, client: TestClient) -> None:
        response = client.post("/predict", json={"smiles": "INVALID_XYZ"})
        assert response.status_code == 400

    def test_invalid_smiles_has_detail(self, client: TestClient) -> None:
        data = client.post("/predict", json={"smiles": "INVALID_XYZ"}).json()
        assert "detail" in data

    def test_empty_smiles_returns_422(self, client: TestClient) -> None:
        """Empty string is rejected by Pydantic (min_length=1) before RDKit."""
        response = client.post("/predict", json={"smiles": ""})
        assert response.status_code == 422

    def test_missing_smiles_field_returns_422(self, client: TestClient) -> None:
        response = client.post("/predict", json={})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /visualize
# ---------------------------------------------------------------------------

class TestVisualize:
    def test_valid_ethanol(self, client: TestClient) -> None:
        response = client.post("/visualize", json={"smiles": "CCO"})
        assert response.status_code == 200

    def test_response_shape(self, client: TestClient) -> None:
        data = client.post("/visualize", json={"smiles": "CCO"}).json()
        assert set(data.keys()) == {"image_path"}

    def test_image_path_is_relative(self, client: TestClient) -> None:
        data = client.post("/visualize", json={"smiles": "CCO"}).json()
        assert not Path(data["image_path"]).is_absolute()

    def test_image_path_starts_with_directory(self, client: TestClient) -> None:
        data = client.post("/visualize", json={"smiles": "CCO"}).json()
        assert data["image_path"].startswith("generated_molecules/")

    def test_image_path_ends_with_png(self, client: TestClient) -> None:
        data = client.post("/visualize", json={"smiles": "CCO"}).json()
        assert data["image_path"].endswith(".png")

    def test_file_actually_created(self, client: TestClient) -> None:
        data = client.post("/visualize", json={"smiles": "CCO"}).json()
        assert (PROJECT_ROOT / data["image_path"]).exists()

    def test_invalid_smiles_returns_400(self, client: TestClient) -> None:
        response = client.post("/visualize", json={"smiles": "INVALID_XYZ"})
        assert response.status_code == 400

    def test_empty_smiles_returns_422(self, client: TestClient) -> None:
        response = client.post("/visualize", json={"smiles": ""})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_valid_ethanol_returns_200(self, client: TestClient) -> None:
        response = client.post("/analyze", json={"smiles": "CCO"})
        assert response.status_code == 200

    def test_response_has_all_three_fields(self, client: TestClient) -> None:
        """Response must contain smiles, predicted_solubility, and image_path."""
        data = client.post("/analyze", json={"smiles": "CCO"}).json()
        assert set(data.keys()) == {"smiles", "predicted_solubility", "image_path"}

    def test_smiles_field_echoed(self, client: TestClient) -> None:
        data = client.post("/analyze", json={"smiles": "CCO"}).json()
        assert data["smiles"] == "CCO"

    def test_predicted_solubility_is_float(self, client: TestClient) -> None:
        data = client.post("/analyze", json={"smiles": "CCO"}).json()
        assert isinstance(data["predicted_solubility"], float)

    def test_image_path_is_string(self, client: TestClient) -> None:
        data = client.post("/analyze", json={"smiles": "CCO"}).json()
        assert isinstance(data["image_path"], str)

    def test_image_path_relative(self, client: TestClient) -> None:
        data = client.post("/analyze", json={"smiles": "CCO"}).json()
        assert not Path(data["image_path"]).is_absolute()

    def test_image_path_starts_with_directory(self, client: TestClient) -> None:
        data = client.post("/analyze", json={"smiles": "CCO"}).json()
        assert data["image_path"].startswith("generated_molecules/")

    def test_image_path_ends_with_png(self, client: TestClient) -> None:
        data = client.post("/analyze", json={"smiles": "CCO"}).json()
        assert data["image_path"].endswith(".png")

    def test_png_file_exists_on_disk(self, client: TestClient) -> None:
        data = client.post("/analyze", json={"smiles": "CCO"}).json()
        assert (PROJECT_ROOT / data["image_path"]).exists()

    def test_valid_benzene(self, client: TestClient) -> None:
        response = client.post("/analyze", json={"smiles": "c1ccccc1"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["predicted_solubility"], float)

    def test_valid_aspirin(self, client: TestClient) -> None:
        """Multi-functional molecule with rings, double bonds, and heteroatoms."""
        response = client.post("/analyze", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["predicted_solubility"], float)
        assert data["image_path"].endswith(".png")

    def test_valid_caffeine(self, client: TestClient) -> None:
        response = client.post("/analyze", json={"smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C"})
        assert response.status_code == 200

    def test_invalid_smiles_returns_400(self, client: TestClient) -> None:
        """Chemically invalid strings must yield HTTP 400, not 500."""
        response = client.post("/analyze", json={"smiles": "INVALID_XYZ"})
        assert response.status_code == 400

    def test_invalid_smiles_detail_present(self, client: TestClient) -> None:
        data = client.post("/analyze", json={"smiles": "INVALID_XYZ"}).json()
        assert "detail" in data
        assert len(data["detail"]) > 0

    def test_empty_smiles_returns_422(self, client: TestClient) -> None:
        """Empty string is caught by Pydantic before any inference runs."""
        response = client.post("/analyze", json={"smiles": ""})
        assert response.status_code == 422

    def test_missing_smiles_field_returns_422(self, client: TestClient) -> None:
        response = client.post("/analyze", json={})
        assert response.status_code == 422

    def test_solubility_matches_predict_endpoint(self, client: TestClient) -> None:
        """Solubility from /analyze must equal the value from /predict exactly."""
        smiles = "CCO"
        analyze_data = client.post("/analyze", json={"smiles": smiles}).json()
        predict_data = client.post("/predict", json={"smiles": smiles}).json()
        assert analyze_data["predicted_solubility"] == predict_data["predicted_solubility"]

    def test_image_path_matches_visualize_endpoint(self, client: TestClient) -> None:
        """Image path from /analyze must equal the path from /visualize exactly."""
        smiles = "CCO"
        analyze_data = client.post("/analyze", json={"smiles": smiles}).json()
        visualize_data = client.post("/visualize", json={"smiles": smiles}).json()
        assert analyze_data["image_path"] == visualize_data["image_path"]
