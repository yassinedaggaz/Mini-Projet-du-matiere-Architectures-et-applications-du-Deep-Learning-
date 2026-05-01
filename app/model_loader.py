"""
Singleton model loader.

The model is loaded ONCE at application startup and kept in memory for all
subsequent prediction requests. This avoids the latency of loading ~45 MB
weights on every request.

Environment variables (with defaults):
    MODEL_PATH      – path to .pth weights file      (default: models/best_model.pth)
    CLASS_MAP_PATH  – path to class_to_idx.json      (default: models/class_to_idx.json)
    MODEL_VERSION   – arbitrary version string        (default: 1.0.0)
"""

import json
import logging
import os
import time

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read from environment variables with sensible defaults)
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pth")
CLASS_MAP_PATH = os.environ.get("CLASS_MAP_PATH", "models/class_to_idx.json")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")

# ---------------------------------------------------------------------------
# Image preprocessing — identical to the transforms used during evaluation
# ---------------------------------------------------------------------------
PREPROCESS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# ---------------------------------------------------------------------------
# Global singleton state
# ---------------------------------------------------------------------------
_model: nn.Module | None = None
_idx_to_class: dict[int, str] = {}
_device: torch.device = torch.device("cpu")


def _build_resnet18(num_classes: int) -> nn.Module:
    """Rebuild the same architecture used during training."""
    model = models.resnet18(weights=None)  # weights=None → we load ours
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model() -> None:
    """
    Load the model into memory (called once at startup via lifespan).
    Idempotent: calling it multiple times is safe.
    """
    global _model, _idx_to_class, _device

    if _model is not None:
        logger.info("Model already loaded — skipping.")
        return

    # Read class mapping
    logger.info("Loading class map from %s", CLASS_MAP_PATH)
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        class_to_idx: dict[str, int] = json.load(f)
    _idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Build model architecture
    num_classes = len(class_to_idx)
    logger.info("Building ResNet18 with %d classes", num_classes)
    model = _build_resnet18(num_classes)

    # Load weights
    logger.info("Loading weights from %s", MODEL_PATH)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = model.to(_device)

    logger.info(
        "Model ready | version=%s | device=%s | classes=%s",
        MODEL_VERSION,
        _device,
        list(class_to_idx.keys()),
    )


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict_image(image: Image.Image, top_k: int = 3) -> dict:
    """
    Run inference on a PIL image and return a structured result dict.

    Args:
        image:  PIL.Image (any mode — will be converted to RGB internally).
        top_k:  number of top predictions to return.

    Returns:
        {
            "prediction": str,
            "confidence": float,
            "top_predictions": [{"class_name": str, "confidence": float}, ...],
            "inference_time_ms": float,
        }
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    image = image.convert("RGB")
    tensor = PREPROCESS(image).unsqueeze(0).to(_device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)
    inference_ms = (time.perf_counter() - t0) * 1000

    top_k = min(top_k, probs.shape[1])
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

    top_predictions = [
        {"class_name": _idx_to_class[idx.item()], "confidence": prob.item()}
        for prob, idx in zip(top_probs[0], top_indices[0])
    ]

    return {
        "prediction": top_predictions[0]["class_name"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
        "inference_time_ms": round(inference_ms, 2),
    }


# ---------------------------------------------------------------------------
# Accessors (used by /health)
# ---------------------------------------------------------------------------

def is_loaded() -> bool:
    return _model is not None


def get_classes() -> list[str]:
    return [_idx_to_class[i] for i in sorted(_idx_to_class)]


def get_version() -> str:
    return MODEL_VERSION
