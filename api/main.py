"""
main.py — FastAPI application for the Predictive Maintenance System.

Endpoints:
  GET  /health           — liveness check, confirms models are loaded
  POST /predict          — failure prediction (XGBoost)
  POST /anomaly-score    — anomaly detection (Isolation Forest)
  GET  /docs             — auto-generated Swagger UI (free from FastAPI)

Run locally:
  uvicorn api.main:app --reload --port 8000

Via Docker:
  docker-compose up
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    AnomalyScoreRequest, AnomalyScoreResponse,
    FailurePredictionRequest, FailurePredictionResponse,
    HealthResponse,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

# ── Global model store ────────────────────────────────────────────────────────
# Models are loaded once at startup and reused for every request
# This is the standard pattern for ML APIs — loading on every request is too slow
models: Dict = {}


def load_models():
    """Load all models and artefacts into the global models dict."""
    logger.info("Loading models from %s ...", MODELS_DIR)

    # XGBoost failure prediction model
    models["xgb"] = joblib.load(MODELS_DIR / "xgboost_failure.joblib")
    logger.info("XGBoost loaded")

    # Scaler for failure prediction features
    models["scaler_failure"] = joblib.load(MODELS_DIR / "scaler_failure.joblib")
    logger.info("Scaler (failure) loaded")

    # Isolation Forest anomaly detection model
    models["iso_forest"] = joblib.load(MODELS_DIR / "iso_forest.joblib")
    logger.info("Isolation Forest loaded")

    # Scaler for anomaly detection features
    models["scaler_anomaly"] = joblib.load(MODELS_DIR / "scaler.joblib")
    logger.info("Scaler (anomaly) loaded")

    # Model metadata: thresholds, feature lists, metrics
    with open(MODELS_DIR / "model_meta.json") as f:
        models["meta"] = json.load(f)
    logger.info("Model metadata loaded")

    # Anomaly detection feature list
    with open(MODELS_DIR / "feature_cols.json") as f:
        models["anomaly_features"] = json.load(f)
    logger.info("Feature columns loaded")

    logger.info("All models loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    load_models()
    yield
    models.clear()
    logger.info("Models unloaded")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "ML-powered predictive maintenance system for industrial equipment. "
        "Predicts failures 24 hours in advance using sensor telemetry data. "
        "Built with XGBoost, Isolation Forest, and LSTM Autoencoder."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for development — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper: build feature vector ──────────────────────────────────────────────
def reading_to_dict(reading) -> dict:
    """Convert a SensorReading Pydantic model to a plain dict."""
    return reading.model_dump()


def build_failure_feature_vector(reading) -> np.ndarray:
    """
    Build the feature vector expected by the XGBoost failure prediction model.
    Falls back to raw sensor values where rolling features are missing.
    """
    meta     = models["meta"]
    feat_cols = meta["feature_cols"]
    data     = reading_to_dict(reading)

    vector = []
    for col in feat_cols:
        if col in data and data[col] is not None:
            vector.append(float(data[col]))
        else:
            # Fallback: use raw sensor value for missing rolling features
            base = col.split("_")[0]
            vector.append(float(data.get(base, 0.0)))

    return np.array(vector).reshape(1, -1)


def build_anomaly_feature_vector(reading) -> np.ndarray:
    """
    Build the feature vector expected by the Isolation Forest anomaly model.
    """
    feat_cols = models["anomaly_features"]
    data      = reading_to_dict(reading)

    vector = []
    for col in feat_cols:
        if col in data and data[col] is not None:
            vector.append(float(data[col]))
        else:
            base = col.split("_")[0]
            vector.append(float(data.get(base, 0.0)))

    return np.array(vector).reshape(1, -1)


def probability_to_risk(prob: float) -> str:
    """Map failure probability to a human-readable risk level."""
    if prob < 0.25:
        return "LOW"
    elif prob < 0.50:
        return "MEDIUM"
    elif prob < 0.75:
        return "HIGH"
    else:
        return "CRITICAL"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Liveness check. Returns 200 if the API is running and models are loaded.
    Use this endpoint in your Docker health check and Kubernetes readiness probe.
    """
    return HealthResponse(
        status="healthy",
        models_loaded=list(models.keys()),
    )


@app.post("/predict", response_model=FailurePredictionResponse, tags=["Prediction"])
async def predict_failure(request: FailurePredictionRequest):
    """
    Predict whether this machine will fail in the next 24 hours.

    - **failure_probability**: value between 0 and 1
    - **failure_predicted**: True if probability exceeds the optimised threshold
    - **risk_level**: LOW / MEDIUM / HIGH / CRITICAL
    - **model_used**: XGBoost (trained on Azure PdM dataset)

    The threshold is optimised for maximum F1 score on the held-out test set.
    """
    try:
        reading   = request.reading
        threshold = models["meta"]["xgb_threshold"]

        # Build and scale feature vector
        X_raw    = build_failure_feature_vector(reading)
        X_scaled = models["scaler_failure"].transform(X_raw)

        # Predict
        prob      = float(models["xgb"].predict_proba(X_scaled)[0, 1])
        predicted = prob >= threshold

        return FailurePredictionResponse(
            machineID=reading.machineID,
            failure_probability=round(prob, 4),
            failure_predicted=predicted,
            threshold=round(threshold, 4),
            risk_level=probability_to_risk(prob),
            model_used="XGBoost (F1=0.995, ROC-AUC=1.000)",
        )

    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/anomaly-score", response_model=AnomalyScoreResponse, tags=["Anomaly Detection"])
async def anomaly_score(request: AnomalyScoreRequest):
    """
    Compute an anomaly score for this machine's current sensor readings.

    Uses an Isolation Forest trained on normal operating data.
    High anomaly score = sensor behaviour deviates from normal patterns.

    - **anomaly_score**: higher values indicate more anomalous behaviour
    - **is_anomaly**: True if score exceeds the 98th percentile threshold
    """
    try:
        reading   = request.reading
        threshold = models["meta"].get("iso_threshold",
                    float(models["meta"].get("xgb_threshold", 0.5)))

        # Build and scale feature vector
        X_raw    = build_anomaly_feature_vector(reading)
        X_scaled = models["scaler_anomaly"].transform(X_raw)

        # Isolation Forest returns negative scores — negate so higher = more anomalous
        score    = float(-models["iso_forest"].score_samples(X_scaled)[0])
        is_anom  = score >= threshold

        return AnomalyScoreResponse(
            machineID=reading.machineID,
            anomaly_score=round(score, 4),
            is_anomaly=is_anom,
            threshold=round(threshold, 4),
            model_used="IsolationForest (ROC-AUC=0.974)",
        )

    except Exception as e:
        logger.error("Anomaly scoring error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Anomaly scoring failed: {str(e)}")
