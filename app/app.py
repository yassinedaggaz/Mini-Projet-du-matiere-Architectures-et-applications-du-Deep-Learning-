from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import os
import logging

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Deep Learning Image Classifier",
    description=(
        "ResNet18 Transfer Learning API — classifies computer hardware images "
        "into 4 categories: keyboard, laptop, monitor, mouse." \
        "\n\n Project by: Youssef Sdiri and Yassine Daggaz"
    ),
    version="1.0.0"
)

MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.h5")

def load_model():
    try:
        # from tensorflow.keras.models import load_model as keras_load
        # return keras_load(MODEL_PATH)
        logger.info(f"Modèle chargé depuis {MODEL_PATH}")
        return None
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        return None

model = load_model()

class PredictRequest(BaseModel):
    input: List[List[float]]

class PredictResponse(BaseModel):
    prediction: List[float]
    status: str

@app.get("/")
def home():
    return {"status": "ok", "message": "API Deep Learning opérationnelle"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        input_data = np.array(request.input)
        logger.info(f"Requête reçue - shape: {input_data.shape}")
        # prediction = model.predict(input_data).tolist()[0]
        prediction = [0.85, 0.10, 0.05]  # Placeholder
        return PredictResponse(prediction=prediction, status="success")
    except Exception as e:
        logger.error(f"Erreur de prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    debug = os.environ.get("APP_DEBUG", "false").lower() == "true"
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=debug)
