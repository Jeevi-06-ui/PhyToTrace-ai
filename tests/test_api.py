"""Unit tests for FastAPI endpoints and input validation."""

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from phytotrace.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "mps_available" in data


def test_version_endpoint(client):
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert data["api_version"] == "0.1.0"
    assert "torch_version" in data


def test_models_status_endpoint(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 4


def test_predict_endpoint_valid_image(client):
    # Create test JPEG in memory
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[50:250, 50:250] = (40, 160, 50)
    img[::8, :] = (50, 150, 50)
    _, buf = cv2.imencode(".jpg", img)

    files = {"file": ("test_leaf.jpg", buf.tobytes(), "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["image_id"] == "test_leaf"
    assert "decision" in data
    assert "safety_notice" in data


def test_predict_endpoint_invalid_extension(client):
    files = {"file": ("malicious.exe", b"executable_data", "application/octet-stream")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Unsupported image format" in response.json()["detail"]

