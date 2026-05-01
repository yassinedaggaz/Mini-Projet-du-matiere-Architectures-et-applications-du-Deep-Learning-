"""
Unit tests for the FastAPI prediction API.

Run with:
    pytest app/tests/test_api.py -v
"""

import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when running from any directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Mock the model_loader BEFORE importing the FastAPI app so that no real
# model weights are needed to run the tests.
# ---------------------------------------------------------------------------
MOCK_PREDICTION = {
    "prediction": "keyboard",
    "confidence": 0.9345,
    "top_predictions": [
        {"class_name": "keyboard", "confidence": 0.9345},
        {"class_name": "mouse",    "confidence": 0.0412},
        {"class_name": "laptop",   "confidence": 0.0201},
    ],
    "inference_time_ms": 18.5,
}

MOCK_CLASSES = ["keyboard", "laptop", "monitor", "mouse"]
MOCK_VERSION = "1.0.0-test"


def _make_fake_image_bytes(mode: str = "JPEG") -> bytes:
    """Return in-memory bytes of a tiny 64×64 test image."""
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(120, 80, 40))
    img.save(buf, format=mode)
    buf.seek(0)
    return buf.read()


# Patch model_loader functions before importing main
with (
    patch("app.model_loader.load_model"),
    patch("app.model_loader.is_loaded", return_value=True),
    patch("app.model_loader.get_classes", return_value=MOCK_CLASSES),
    patch("app.model_loader.get_version", return_value=MOCK_VERSION),
    patch("app.model_loader.predict_image", return_value=MOCK_PREDICTION),
):
    from app.main import app  # noqa: E402  (import after patching)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        with patch("app.main.is_loaded", return_value=True), \
             patch("app.main.get_version", return_value=MOCK_VERSION), \
             patch("app.main.get_classes", return_value=MOCK_CLASSES):
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client):
        with patch("app.main.is_loaded", return_value=True), \
             patch("app.main.get_version", return_value=MOCK_VERSION), \
             patch("app.main.get_classes", return_value=MOCK_CLASSES):
            data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "classes" in data
        assert len(data["classes"]) == 4


class TestPredict:
    def test_predict_jpeg(self, client):
        img_bytes = _make_fake_image_bytes("JPEG")
        with patch("app.main.predict_image", return_value=MOCK_PREDICTION):
            resp = client.post(
                "/predict",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            )
        assert resp.status_code == 200

    def test_predict_response_schema(self, client):
        img_bytes = _make_fake_image_bytes("JPEG")
        with patch("app.main.predict_image", return_value=MOCK_PREDICTION):
            data = client.post(
                "/predict",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            ).json()
        assert data["prediction"] == "keyboard"
        assert data["confidence"] == pytest.approx(0.9345, rel=1e-3)
        assert "%" in data["confidence_pct"]
        assert len(data["top_predictions"]) == 3
        assert "model_version" in data
        assert "inference_time_ms" in data

    def test_predict_png(self, client):
        img_bytes = _make_fake_image_bytes("PNG")
        with patch("app.main.predict_image", return_value=MOCK_PREDICTION):
            resp = client.post(
                "/predict",
                files={"file": ("test.png", img_bytes, "image/png")},
            )
        assert resp.status_code == 200

    def test_predict_unsupported_type(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 415

    def test_predict_no_file(self, client):
        resp = client.post("/predict")
        assert resp.status_code == 422  # FastAPI validation error


class TestMetrics:
    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        # Should contain at least one of our custom counters
        assert b"predict_requests_total" in resp.content
