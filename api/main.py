"""
Containerised ML inference API (FastAPI).

Endpoints (all required by the spec):
  GET  /health   -> liveness + model load time          (used by Container Apps probe)
  GET  /version  -> model + API versions
  POST /predict  -> prediction with confidence score
  GET  /metrics  -> in-process counters (calls, errors, avg latency)

Config is environment-only (12-factor): nothing about the deployment is baked in.
"""

import os
import time
from contextlib import asynccontextmanager
from threading import Lock

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

API_VERSION = os.getenv("API_VERSION", "1.0.0")
MODEL_PATH = os.getenv("MODEL_PATH", "model_v1.0.0.pkl")

# --- Runtime state ------------------------------------------------------------
STATE: dict = {"bundle": None, "load_time_ms": None}

METRICS = {"calls": 0, "errors": 0, "total_latency_ms": 0.0}
_metrics_lock = Lock()


def _record(latency_ms: float, error: bool) -> None:
    with _metrics_lock:
        METRICS["calls"] += 1
        METRICS["total_latency_ms"] += latency_ms
        if error:
            METRICS["errors"] += 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    bundle = joblib.load(MODEL_PATH)
    # Guard: the API's snake_case order must match the trained column order.
    expected = [c.replace(" ", "_") for c in bundle["feature_names"]]
    if expected != FEATURE_ORDER:
        raise RuntimeError(
            f"feature order mismatch: model={expected} api={FEATURE_ORDER}"
        )
    STATE["bundle"] = bundle
    STATE["load_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    yield
    STATE["bundle"] = None


app = FastAPI(title="Wine Quality Inference API", version=API_VERSION, lifespan=lifespan)


# --- Schemas ------------------------------------------------------------------
# The model was trained on CSV columns whose names contain spaces. We expose a
# clean snake_case API and map to the trained column order via FEATURE_ORDER.
# FEATURE_ORDER[i] is the snake_case name of the i-th trained feature.
FEATURE_ORDER = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol",
]


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(ge=0)
    volatile_acidity: float = Field(ge=0)
    citric_acid: float = Field(ge=0)
    residual_sugar: float = Field(ge=0)
    chlorides: float = Field(ge=0)
    free_sulfur_dioxide: float = Field(ge=0)
    total_sulfur_dioxide: float = Field(ge=0)
    density: float = Field(ge=0)
    pH: float = Field(ge=0, le=14)
    sulphates: float = Field(ge=0)
    alcohol: float = Field(ge=0)


class PredictResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    model_version: str
    duration_ms: float


# --- Endpoints ----------------------------------------------------------------
@app.get("/health")
def health():
    loaded = STATE["bundle"] is not None
    return {
        "status": "OK" if loaded else "KO",
        "model_loaded": loaded,
        "model_load_time_ms": STATE["load_time_ms"],
    }


@app.get("/version")
def version():
    bundle = STATE["bundle"] or {}
    return {
        "api_version": API_VERSION,
        "model_version": bundle.get("model_version"),
        "sklearn_version": bundle.get("sklearn_version"),
        "trained_at": bundle.get("trained_at"),
    }


@app.get("/metrics")
def metrics():
    with _metrics_lock:
        calls = METRICS["calls"]
        avg = round(METRICS["total_latency_ms"] / calls, 2) if calls else 0.0
        error_rate = round(METRICS["errors"] / calls, 4) if calls else 0.0
        return {
            "calls": calls,
            "errors": METRICS["errors"],
            "error_rate": error_rate,
            "avg_latency_ms": avg,
        }


@app.post("/predict", response_model=PredictResponse)
def predict(features: WineFeatures):
    t0 = time.perf_counter()
    bundle = STATE["bundle"]
    if bundle is None:
        _record((time.perf_counter() - t0) * 1000, error=True)
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        # Build the feature vector in the exact trained column order.
        row = np.array([[getattr(features, name) for name in FEATURE_ORDER]])

        model = bundle["model"]
        pred = int(model.predict(row)[0])
        proba = model.predict_proba(row)[0]
        confidence = round(float(proba[pred]), 4)
        label = bundle["classes"][str(pred)]
    except Exception as exc:  # noqa: BLE001
        _record((time.perf_counter() - t0) * 1000, error=True)
        raise HTTPException(status_code=400, detail=f"prediction failed: {exc}") from exc

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    _record(duration_ms, error=False)
    return PredictResponse(
        prediction=pred,
        label=label,
        confidence=confidence,
        model_version=bundle["model_version"],
        duration_ms=duration_ms,
    )
