"""
Unit tests for the inference API.

Required by the spec: at least 3 cases covering success, invalid input, and
missing payload. We add health/version/metrics for good measure.

Run from the api/ directory:  MODEL_PATH=model_v1.0.0.pkl pytest -q
"""

import os

os.environ.setdefault("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "model_v1.0.0.pkl"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Context-manager form triggers FastAPI startup (lifespan) -> model loads.
    with TestClient(app) as c:
        yield c

# A realistic "good wine" feature row, using the spaced alias names.
VALID_PAYLOAD = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.35,
    "citric_acid": 0.4,
    "residual_sugar": 2.0,
    "chlorides": 0.07,
    "free_sulfur_dioxide": 15.0,
    "total_sulfur_dioxide": 40.0,
    "density": 0.9968,
    "pH": 3.3,
    "sulphates": 0.7,
    "alcohol": 11.5,
}


def test_predict_success(client):
    """Case 1: a well-formed payload returns a valid prediction + confidence."""
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("good", "not_good")
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"] == "1.0.0"


def test_predict_invalid_input(client):
    """Case 2: a wrong type / out-of-range value is rejected with 422."""
    bad = dict(VALID_PAYLOAD)
    bad["alcohol"] = "not-a-number"
    bad["pH"] = 99  # also out of the 0..14 bound
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_missing_payload(client):
    """Case 3: missing required fields -> 422 validation error."""
    r = client.post("/predict", json={"alcohol": 11.5})
    assert r.status_code == 422


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "OK"
    assert r.json()["model_loaded"] is True


def test_version(client):
    r = client.get("/version")
    body = r.json()
    assert body["api_version"] == "1.0.0"
    assert body["model_version"] == "1.0.0"


def test_metrics_increment(client):
    before = client.get("/metrics").json()["calls"]
    client.post("/predict", json=VALID_PAYLOAD)
    after = client.get("/metrics").json()["calls"]
    assert after == before + 1
