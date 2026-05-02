"""
FastAPI application — Image Classification Service.

Endpoints:
    GET  /           → root (redirect to docs)
    GET  /health     → liveness / readiness probe
    GET  /metrics    → Prometheus metrics
    POST /predict    → upload image → classification result
"""

import io
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import Image
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

from app.model_loader import get_classes, get_version, is_loaded, load_model, predict_image
from app.schemas import HealthResponse, PredictionResponse, TopPrediction

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
PREDICT_REQUESTS = Counter(
    "predict_requests_total",
    "Total number of prediction requests",
    ["status"],  # "success" | "error"
)
PREDICT_LATENCY = Histogram(
    "predict_latency_seconds",
    "End-to-end prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_ms",
    "Model-only inference latency in milliseconds",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500],
)
CLASS_PREDICTIONS = Counter(
    "predicted_class_total",
    "Count of predictions per class",
    ["class_name"],
)
MODEL_LOADED = Gauge("model_loaded", "1 if the model is loaded, 0 otherwise")

# ---------------------------------------------------------------------------
# Application lifespan (load model once at startup)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights at startup; clean up on shutdown."""
    logger.info("🚀 Starting up — loading model …")
    load_model()
    MODEL_LOADED.set(1 if is_loaded() else 0)
    logger.info("✅ Model loaded. API ready.")
    yield
    logger.info("🛑 Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Deep Learning Image Classifier",
    description=(
        "ResNet18 Transfer Learning API — classifies computer hardware images "
        "into 4 categories: keyboard, laptop, monitor, mouse." \
        "\n\n Project by: Youssef Sdiri and Yassine Daggaz"
    ),
    version=get_version(),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the interactive Swagger docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health():
    """
    Liveness and readiness probe.
    Returns 200 when the model is loaded and ready to serve predictions.
    Returns 503 if the model is not yet loaded.
    """
    loaded = is_loaded()
    if not loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_version=get_version(),
        num_classes=len(get_classes()),
        classes=get_classes(),
    )


@app.get("/metrics", tags=["Ops"], include_in_schema=False)
async def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)")):
    """
    Classify an uploaded image.

    - **file**: image file (JPEG / PNG / WebP / AVIF).

    Returns the predicted class, confidence score, and top-3 predictions.

    ### Example with curl
    ```bash
    curl -X POST http://localhost:8000/predict \\
         -F "file=@my_keyboard.jpg"
    ```
    """
    # --- Validate content type ---
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/avif"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Use JPEG, PNG, or WebP.",
        )

    t_start = time.perf_counter()

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        PREDICT_REQUESTS.labels(status="error").inc()
        logger.error("Failed to open image: %s", exc)
        raise HTTPException(status_code=400, detail=f"Cannot open image: {exc}")

    try:
        result = predict_image(image, top_k=3)
    except Exception as exc:
        PREDICT_REQUESTS.labels(status="error").inc()
        logger.error("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    elapsed_s = time.perf_counter() - t_start

    # Update metrics
    PREDICT_REQUESTS.labels(status="success").inc()
    PREDICT_LATENCY.observe(elapsed_s)
    INFERENCE_LATENCY.observe(result["inference_time_ms"])
    CLASS_PREDICTIONS.labels(class_name=result["prediction"]).inc()

    logger.info(
        "prediction=%s confidence=%.4f inference_ms=%.1f",
        result["prediction"],
        result["confidence"],
        result["inference_time_ms"],
    )

    return PredictionResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        confidence_pct=f"{result['confidence'] * 100:.2f}%",
        top_predictions=[TopPrediction(**p) for p in result["top_predictions"]],
        model_version=get_version(),
        inference_time_ms=result["inference_time_ms"],
    )
