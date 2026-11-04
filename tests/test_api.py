"""
Unit and integration tests for FastAPI endpoints, request validation, CORS, and batch processing.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Provides test client with lifespan startup executed."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Checks root endpoint metadata."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "YouTube Sentiment Analysis API"
    assert data["status"] == "operational"


def test_health_endpoint(client):
    """Checks /health endpoint monitoring."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data
    assert len(data["supported_labels"]) == 3


def test_single_prediction_positive(client):
    """Tests single prediction for a clearly positive comment."""
    payload = {"text": "This tutorial is absolutely amazing, thank you so much! ❤️"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["sentiment"] == "positive"
    assert data["label"] == 1
    assert data["confidence"] > 0.5
    assert "probabilities" in data
    assert data["probabilities"]["positive"] > data["probabilities"]["negative"]


def test_single_prediction_negative(client):
    """Tests single prediction for a clearly negative comment."""
    payload = {"text": "Horrible audio, completely unwatchable waste of time 👎"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["sentiment"] == "negative"
    assert data["label"] == -1
    assert data["confidence"] > 0.5


def test_batch_prediction_and_statistics(client):
    """Tests /predict_batch endpoint and aggregated sentiment statistics."""
    payload = {
        "comments": [
            "Brilliant video, loved every minute! 🔥",
            "This was awful and totally inaccurate 😡",
            "What is the github repo link?",
            "Good explanation overall."
        ]
    }
    resp = client.post("/predict_batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "processing_time_ms" in data
    assert len(data["predictions"]) == 4

    stats = data["statistics"]
    assert stats["total_comments"] == 4
    assert stats["positive_count"] + stats["neutral_count"] + stats["negative_count"] == 4
    assert 0.0 <= stats["average_confidence"] <= 1.0
    assert -1.0 <= stats["sentiment_score"] <= 1.0
    assert abs((stats["positive_pct"] + stats["neutral_pct"] + stats["negative_pct"]) - 100.0) < 0.1


def test_batch_prediction_50_items(client):
    """Tests /predict_batch with full batch of 50 comments."""
    comments = ["Insightful and clear, thanks!" if i % 2 == 0 else "Needs better audio" for i in range(50)]
    resp = client.post("/predict_batch", json={"comments": comments})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["predictions"]) == 50
    assert data["processing_time_ms"] < 250.0  # Entire HTTP roundtrip well under 250ms


def test_empty_batch_validation_error(client):
    """Tests that empty batch raises validation error (422)."""
    resp = client.post("/predict_batch", json={"comments": []})
    assert resp.status_code in [400, 422]


def test_cors_headers(client):
    """Verifies CORS headers for Chrome extension integration."""
    resp = client.options(
        "/predict_batch",
        headers={
            "Origin": "chrome-extension://abcdefghijklmnop",
            "Access-Control-Request-Method": "POST"
        }
    )
    assert resp.headers.get("access-control-allow-origin") in ["*", "chrome-extension://abcdefghijklmnop"]
