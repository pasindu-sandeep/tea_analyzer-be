import os
import joblib
import pandas as pd
import tempfile
import numpy as np

from flask import Flask, request, jsonify
from ultralytics import YOLO

# -----------------------------------
# Prevent Torch / OpenMP thread issues (Mac fix)
# -----------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

app = Flask(__name__)

# -----------------------------------
# Safe Base Directory
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------
# Load Models Once at Startup
# -----------------------------------
model = joblib.load(os.path.join(BASE_DIR, "models", "tea_price_model.pkl"))
cls_model = YOLO(os.path.join(BASE_DIR, "models", "best.pt"))
soil_model = joblib.load(os.path.join(BASE_DIR, "models", "soil_analyzer.pkl"))

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
    try:
        data = request.get_json()

        if not data:
            return jsonify({"success": False, "error": "No JSON body provided"}), 400

        missing = [f for f in model.feature_names_in_ if f not in data]
        if missing:
            return jsonify({
                "success": False,
                "error": f"Missing fields: {missing}"
            }), 400

        df = pd.DataFrame([data])
        df = df[model.feature_names_in_]

        prediction = model.predict(df)[0]

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
    temp_path = None

    try:
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided"
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "Empty filename"
            }), 400

        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            file.save(temp.name)
            temp_path = temp.name

        # Run prediction
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

SOIL_FEATURES = [
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "pH",
    "Moisture"
]

# -----------------------------------
# Soil Analysis
# -----------------------------------
@app.route("/analyze-soil", methods=["POST"])
def analyze_soil():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON body provided"
            }), 400

        # Validate fields
        missing = [f for f in SOIL_FEATURES if f not in data]
        if missing:
            return jsonify({
                "success": False,
                "error": f"Missing fields: {missing}"
            }), 400

        # Convert to numpy array in correct order
        input_data = np.array([[data[f] for f in SOIL_FEATURES]])

        prediction = soil_model.predict(input_data)

        # If classification
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)

        return jsonify({
            "success": True,
            "predicted_class": int(predicted_class),
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