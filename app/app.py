from flask import Flask, request, jsonify
import numpy as np
import os
import logging

app = Flask(__name__)

# Configuration des logs selon l'environnement
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# Chargement du modèle
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.h5")

def load_model():
    try:
        # Exemple avec Keras/TensorFlow
        # from tensorflow.keras.models import load_model as keras_load
        # return keras_load(MODEL_PATH)
        logger.info(f"Modèle chargé depuis {MODEL_PATH}")
        return None  # Remplacez par votre modèle réel
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        return None

model = load_model()

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "API Deep Learning opérationnelle"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Données JSON manquantes"}), 400

        # Prétraitement (à adapter selon votre projet)
        input_data = np.array(data.get("input", []))
        logger.info(f"Requête reçue - shape: {input_data.shape}")

        # Inférence (à adapter)
        # prediction = model.predict(input_data)
        prediction = [0.85, 0.10, 0.05]  # Placeholder

        return jsonify({
            "prediction": prediction,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Erreur de prédiction : {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug)
