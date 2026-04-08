"""
schemas.py — Pydantic request/response models for the Predictive Maintenance API.

Pydantic validates all incoming data automatically.
If a request is missing a field or has the wrong type, FastAPI returns a 422 error
with a clear message — no manual validation needed.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class SensorReading(BaseModel):
    """
    A single row of sensor data for one machine at one timestamp.
    All fields match the raw telemetry columns from the Azure dataset.
    """
    machineID:         int   = Field(..., description="Machine identifier (1-100)")
    volt:              float = Field(..., description="Voltage reading")
    rotate:            float = Field(..., description="Rotation speed")
    pressure:          float = Field(..., description="Pressure reading")
    vibration:         float = Field(..., description="Vibration reading")
    age:               float = Field(..., description="Machine age in years")
    hours_since_maint: float = Field(..., description="Hours since last maintenance (9999 if never)")
    hour_of_day:       int   = Field(..., ge=0, le=23,  description="Hour of day (0-23)")
    day_of_week:       int   = Field(..., ge=1, le=7,   description="Day of week (1=Sun, 7=Sat)")
    day_of_month:      int   = Field(..., ge=1, le=31,  description="Day of month")
    month:             int   = Field(..., ge=1, le=12,  description="Month (1-12)")
    is_weekend:        int   = Field(..., ge=0, le=1,   description="1 if weekend, 0 otherwise")

    # Rolling features — pre-computed by the feature pipeline
    # The API accepts pre-engineered features for low-latency inference
    volt_mean_24h:       Optional[float] = None
    rotate_mean_24h:     Optional[float] = None
    pressure_mean_24h:   Optional[float] = None
    vibration_mean_24h:  Optional[float] = None
    volt_std_24h:        Optional[float] = None
    rotate_std_24h:      Optional[float] = None
    pressure_std_24h:    Optional[float] = None
    vibration_std_24h:   Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "machineID": 1,
                "volt": 170.0,
                "rotate": 450.0,
                "pressure": 100.0,
                "vibration": 40.0,
                "age": 5.0,
                "hours_since_maint": 120.0,
                "hour_of_day": 14,
                "day_of_week": 2,
                "day_of_month": 15,
                "month": 6,
                "is_weekend": 0,
                "volt_mean_24h": 168.5,
                "rotate_mean_24h": 448.2,
                "pressure_mean_24h": 99.8,
                "vibration_mean_24h": 40.1,
                "volt_std_24h": 3.2,
                "rotate_std_24h": 5.1,
                "pressure_std_24h": 1.2,
                "vibration_std_24h": 0.8
            }
        }


class FailurePredictionRequest(BaseModel):
    """Request body for the /predict endpoint."""
    reading: SensorReading


class FailurePredictionResponse(BaseModel):
    """Response from the /predict endpoint."""
    machineID:           int
    failure_probability: float = Field(..., description="Probability of failure in next 24h (0-1)")
    failure_predicted:   bool  = Field(..., description="True if failure predicted (above threshold)")
    threshold:           float = Field(..., description="Decision threshold used")
    risk_level:          str   = Field(..., description="LOW / MEDIUM / HIGH / CRITICAL")
    model_used:          str   = Field(..., description="Model that produced this prediction")


class AnomalyScoreRequest(BaseModel):
    """Request body for the /anomaly-score endpoint."""
    reading: SensorReading


class AnomalyScoreResponse(BaseModel):
    """Response from the /anomaly-score endpoint."""
    machineID:     int
    anomaly_score: float = Field(..., description="Anomaly score (higher = more anomalous)")
    is_anomaly:    bool  = Field(..., description="True if anomaly detected")
    threshold:     float
    model_used:    str


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status:         str
    models_loaded:  List[str]
    version:        str = "1.0.0"
