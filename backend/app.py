import os
import joblib
import pandas as pd
import tempfile
import numpy as np
import torch

from flask import Flask, request, jsonify
from ultralytics import YOLO

# -----------------------------------
# Prevent Torch / OpenMP thread issues
# -----------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

app = Flask(__name__)

# -----------------------------------
# Safe Base Directory
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Tea Price Model
try:
    tea_model = joblib.load(os.path.join(MODEL_DIR, "tea_price_model.pkl"))
except Exception as e:
    tea_model = None

# YOLO Classification Model
try:
    cls_model = YOLO(os.path.join(MODEL_DIR, "best.pt"))
except Exception as e:
    cls_model = None

# Soil Model
try:
    soil_model = joblib.load(os.path.join(MODEL_DIR, "soil_analyzer.pkl"))
except Exception as e:
    soil_model = None


# -----------------------------------
# Health Check
# -----------------------------------
@app.route("/")
def home():
    return "Tea API Running"


# -----------------------------------
# Tea Price Prediction
# -----------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if tea_model is None:
        return jsonify({
            "success": False,
            "error": "Tea price model not available"
        }), 500

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON body provided"
            }), 400

        missing = [f for f in tea_model.feature_names_in_ if f not in data]
        if missing:
            return jsonify({
                "success": False,
                "error": f"Missing fields: {missing}"
            }), 400

        df = pd.DataFrame([data])
        df = df[tea_model.feature_names_in_]

        prediction = tea_model.predict(df)[0]

        return jsonify({
            "success": True,
            "prediction": float(prediction)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

# -----------------------------------
# Image Classification
# -----------------------------------
@app.route("/classify", methods=["POST"])
def classify_image():
    if cls_model is None:
        return jsonify({
            "success": False,
            "error": "YOLO model not available"
        }), 500

    temp_path = None

    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"success": False, "error": "Empty filename"}), 400

        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            file.save(temp.name)
            temp_path = temp.name

        results = cls_model.predict(temp_path, verbose=False)

        probs_tensor = results[0].probs.data
        probs = probs_tensor.cpu().numpy()
        class_names = cls_model.names

        probabilities = {
            class_names[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(probs))
        }

        max_index = np.argmax(probs)
        max_prob = float(probs[max_index]) * 100
        predicted_class = class_names[max_index]

        CONF_THRESHOLD = 70
        is_uncertain = max_prob < CONF_THRESHOLD

        if is_uncertain:
            predicted_class = "unknown"

        return jsonify({
            "success": True,
            "predicted_class": predicted_class,
            "confidence_percent": round(max_prob, 2),
            "all_probabilities": probabilities,
            "is_uncertain": is_uncertain
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# -----------------------------------
# Soil Analysis
# -----------------------------------
SOIL_FEATURES = [
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "pH",
    "Moisture"
]

@app.route("/analyze-soil", methods=["POST"])
def analyze_soil():
    if soil_model is None:
        return jsonify({
            "success": False,
            "error": "Soil model not available"
        }), 500

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON body provided"
            }), 400

        missing = [f for f in SOIL_FEATURES if f not in data]
        if missing:
            return jsonify({
                "success": False,
                "error": f"Missing fields: {missing}"
            }), 400

        input_data = np.array([[data[f] for f in SOIL_FEATURES]])

        prediction = soil_model.predict(input_data)

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        return jsonify({
            "success": True,
            "predicted_class": predicted_class,
            "confidence_percent": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# -----------------------------------
# Run Server
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)