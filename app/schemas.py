"""
Pydantic schemas for the prediction API.
"""
from typing import List
from pydantic import BaseModel


class TopPrediction(BaseModel):
    """A single class prediction with its confidence score."""
    class_name: str
    confidence: float  # 0.0 – 1.0


class PredictionResponse(BaseModel):
    """Response returned by POST /predict."""
    prediction: str
    confidence: float         # confidence of the top-1 prediction (0.0 – 1.0)
    confidence_pct: str       # human-readable, e.g. "93.45%"
    top_predictions: List[TopPrediction]
    model_version: str
    inference_time_ms: float  # wall-clock time for the forward pass


class HealthResponse(BaseModel):
    """Response returned by GET /health."""
    status: str               # "ok"
    model_loaded: bool
    model_version: str
    num_classes: int
    classes: List[str]
