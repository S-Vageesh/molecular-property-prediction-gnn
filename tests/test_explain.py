"""Integration tests for the /explain endpoint."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

# Add project root to sys.path is handled in api.py, but we might need it for imports here.
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import app

client = TestClient(app)


def test_explain_benzene():
    """Verify that /explain returns a valid prediction and image path for benzene."""
    response = client.post("/explain", json={"smiles": "c1ccccc1"})
    
    # If GIN model is missing, it might return 503.
    # In a proper CI environment, we'd ensure models are present.
    if response.status_code == 503:
        pytest.skip("GIN model not available for testing.")
        
    assert response.status_code == 200
    data = response.json()
    assert data["smiles"] == "c1ccccc1"
    assert isinstance(data["prediction"], float)
    assert "generated_explanations" in data["explanation_image"]
    assert data["explanation_image"].endswith(".png")
    
    # Check if file actually exists
    image_path = PROJECT_ROOT / data["explanation_image"]
    assert image_path.exists()


def test_explain_invalid_smiles():
    """Verify that /explain returns 400 for an invalid SMILES string."""
    response = client.post("/explain", json={"smiles": "NOT_A_SMILES"})
    assert response.status_code == 400
    assert "Invalid SMILES" in response.json()["detail"]


def test_explain_empty_smiles():
    """Verify that /explain returns 422 for an empty SMILES string."""
    response = client.post("/explain", json={"smiles": ""})
    assert response.status_code == 422
