import os
import tempfile
import numpy as np
import torch
from datetime import datetime, timedelta
from google.cloud import storage
import base64
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import storage as st
import database.db_connect as db

# -----------------------------------
# Prevent Torch / OpenMP thread issues
# -----------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

app = Flask(__name__)
disease_model = YOLO('best2.pt') 
# -----------------------------------
# Safe Base Directory
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# YOLO Classification Model
try:
    cls_model = YOLO(os.path.join(MODEL_DIR, "best.pt"))
except Exception:
    cls_model = None


# -----------------------------------
# Health Check
# -----------------------------------
@app.route("/")
def home():
    return "Tea API Running Successfully"


# ===================================
# 1️⃣ IMAGE CLASSIFICATION
# ===================================
@app.route("/classify", methods=["POST"])
def classify_image():

    if cls_model is None:
        return jsonify({"success": False, "error": "YOLO model not available"}), 500

    temp_path = None

    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400

        file = request.files["image"]

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
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ===================================
# 2️⃣ SOIL ANALYZER (Weighted Smart Model)
# ===================================

IDEAL_RANGES = {
    "Nitrogen": (40, 80),
    "Phosphorus": (20, 40),
    "Potassium": (40, 80),
    "pH": (6.0, 7.5),
    "Moisture": (30, 60),
    "Temperature": (20, 35),
    "Humidity": (50, 80),
    "Rainfall": (80, 200)
}

# Assign higher importance to critical factors
SOIL_WEIGHTS = {
    "Nitrogen": 0.2,
    "Phosphorus": 0.15,
    "Potassium": 0.15,
    "pH": 0.2,
    "Moisture": 0.1,
    "Temperature": 0.1,
    "Humidity": 0.05,
    "Rainfall": 0.05
}


def calculate_score(value, min_val, max_val):
    if min_val <= value <= max_val:
        return 100.0
    else:
        deviation = abs(value - min_val) if value < min_val else abs(value - max_val)
        range_span = max_val - min_val
        penalty = (deviation / range_span) * 100
        return max(0.0, 100 - penalty)


@app.route("/analyze-soil", methods=["POST"])
def analyze_soil():

    try:
        data = request.get_json()

        if not data:
            return jsonify({"success": False, "error": "No JSON body provided"}), 400

        missing = [key for key in IDEAL_RANGES if key not in data]
        if missing:
            return jsonify({"success": False, "error": f"Missing fields: {missing}"}), 400

        results = {}
        weighted_score = 0

        for param, (min_val, max_val) in IDEAL_RANGES.items():
            value = float(data[param])
            score = calculate_score(value, min_val, max_val)
            weight = SOIL_WEIGHTS[param]

            weighted_score += score * weight

            status = (
                "Good" if score >= 80 else
                "Moderate" if score >= 50 else
                "Poor"
            )

            results[param] = {
                "value": value,
                "ideal_range": f"{min_val} - {max_val}",
                "score_percent": round(score, 2),
                "weight": weight,
                "status": status
            }

        overall_score = round(weighted_score, 2)

        if overall_score >= 80:
            soil_condition = "Excellent for Cultivation"
        elif overall_score >= 60:
            soil_condition = "Moderate - Improvement Needed"
        else:
            soil_condition = "Poor - Not Suitable"

        return jsonify({
            "success": True,
            "overall_soil_quality_percent": overall_score,
            "soil_condition": soil_condition,
            "parameter_analysis": results
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ===================================
# 3️⃣ TEA PRICE PREDICTION (Economic Model)
# ===================================

REQUIRED_TEA_FIELDS = [
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Max_Temperature_C",
    "Min_Temperature_C",
    "Humidity_pct",
    "Sunshine_Hours",
    "Drought_Index",
    "USD_LKR",
    "Inflation_Rate",
    "Fuel_Price",
    "Interest_Rate",
    "Electricity_Cost",
    "Production_MT",
    "Auction_Quantity_MT",
    "Stocks_MT",
    "Plucking_Rate",
    "Fertilizer_Usage",
    "Labor_Cost",
    "Price_lag_1",
    "Price_lag_2",
    "Price_lag_3"
]


def predict_tea_price(data):

    def normalize(value, min_val, max_val):
        if value < min_val:
            return value / min_val
        elif value > max_val:
            return max_val / value
        return 1.0

    rainfall_score = normalize(data["Rainfall_mm"], 100, 250)
    temp_score = normalize(data["Avg_Temperature_C"], 18, 30)
    humidity_score = normalize(data["Humidity_pct"], 60, 85)
    sunshine_score = normalize(data["Sunshine_Hours"], 4, 8)

    climate_factor = np.mean([
        rainfall_score,
        temp_score,
        humidity_score,
        sunshine_score
    ])

    supply_index = (
        data["Production_MT"] * 0.5 +
        data["Auction_Quantity_MT"] * 0.3 +
        data["Stocks_MT"] * 0.2
    )

    normalized_supply = supply_index / 50000

    cost_index = (
        data["Labor_Cost"] * 0.4 +
        data["Fuel_Price"] * 0.2 +
        data["Electricity_Cost"] * 0.2 +
        data["Fertilizer_Usage"] * 0.2
    )

    economic_index = (
        data["USD_LKR"] * 0.4 +
        data["Inflation_Rate"] * 0.3 -
        data["Interest_Rate"] * 0.2
    )

    momentum_price = (
        data["Price_lag_1"] * 0.5 +
        data["Price_lag_2"] * 0.3 +
        data["Price_lag_3"] * 0.2
    )

    predicted_price = (
        momentum_price
        + (0.25 * economic_index)
        + (0.30 * cost_index)
        - (0.35 * normalized_supply * 10)
        - (0.15 * climate_factor * 5)
    )

    return round(max(200, predicted_price), 2)


@app.route("/predict-tea-price", methods=["POST"])
def tea_price_endpoint():

    try:
        data = request.get_json()

        if not data:
            return jsonify({"success": False, "error": "No JSON body provided"}), 400

        missing = [f for f in REQUIRED_TEA_FIELDS if f not in data]
        if missing:
            return jsonify({"success": False, "error": f"Missing fields: {missing}"}), 400

        numeric_data = {k: float(data[k]) for k in REQUIRED_TEA_FIELDS}

        price = predict_tea_price(numeric_data)

        return jsonify({
            "success": True,
            "predicted_market_price_LKR": price
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/soil-history", methods=["POST"])
def soil_history():

    try:
        data = request.get_json()

        start_date_str = data.get("start_date")
        end_date_str = data.get("end_date")

        if not start_date_str or not end_date_str:
            return jsonify({
                "success": False,
                "error": "start_date and end_date required"
            }), 400

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")

        if start_date > end_date:
            return jsonify({
                "success": False,
                "error": "start_date must be before end_date"
            }), 400

        # ---- Generate Hourly Mock Data ----
        current = start_date
        hourly_data = []

        while current <= end_date:

            hourly_data.append({
                "timestamp": current.strftime("%Y-%m-%d %H:%M:%S"),
                "Nitrogen": round(np.random.uniform(45, 65), 2),
                "Phosphorus": round(np.random.uniform(22, 35), 2),
                "Potassium": round(np.random.uniform(50, 70), 2),
                "pH": round(np.random.uniform(6.1, 6.8), 2),
                "Moisture": round(np.random.uniform(35, 50), 2),
                "Temperature": round(np.random.uniform(24, 30), 2),
                "Humidity": round(np.random.uniform(65, 80), 2),
                "Rainfall": round(np.random.uniform(0, 10), 2)
            })

            current += timedelta(hours=1)

        return jsonify({
            "success": True,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "total_records": len(hourly_data),
            "hourly_data": hourly_data
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img = Image.open(file.stream)
    
    # Run Inference
    results = disease_model(img)[0]
    
    # dictionary to hold your desired format: {"ClassName": total_pixels}
    predictions = {}
    
    if results.masks is not None:
        # Move to CPU and convert to numpy for pixel counting
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        
        for i, mask in enumerate(masks):
            class_name = results.names[int(classes[i])]
            pixel_count = int(np.sum(mask > 0))
            
            # Aggregate the sum for each class
            predictions[class_name] = predictions.get(class_name, 0) + pixel_count

    # Process the result image (Clean: no boxes/labels)
    res_plotted = results.plot(labels=False, boxes=False)
    res_img = Image.fromarray(res_plotted[:, :, ::-1])

    # Encode Image to Base64
    buffered = io.BytesIO()
    res_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        "status": "success",
        "predictions": predictions,
        "image": img_str
    })

@app.route("/create-bid", methods=["POST"])
def create_bid():

    try:
        file = request.files["image"]
        name = request.form.get("name")
        description = request.form.get("description")
        owner_id = request.form.get("owner_id")
        starting_price = request.form.get("starting_price")

        image_url = st.upload_image(file)

        created_at = datetime.utcnow()

        conn = db.get_db_connection()
        cur = conn.cursor()

        query = """
        INSERT INTO bids
        (name, description, owner_id, starting_price, image_url, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """

        cur.execute(query, (
            name,
            description,
            owner_id,
            starting_price,
            image_url,
            created_at
        ))

        bid_id = cur.fetchone()[0]

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            "message": "Bid created successfully",
            "bid_id": bid_id,
            "image_url": image_url
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/bids", methods=["GET"])
def list_bids():
    try:
        conn = db.get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name, description, owner_id, starting_price, image_url, created_at FROM bids ORDER BY created_at DESC")
        bids = cur.fetchall()
        cur.close()
        conn.close()
        bid_list = [
            {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "owner_id": row[3],
                "starting_price": row[4],
                "image_url": row[5],
                "created_at": row[6].isoformat() if row[6] else None
            }
            for row in bids
        ]
        return jsonify({"bids": bid_list}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/bid/<int:bid_id>", methods=["DELETE"])
def delete_bid(bid_id):
    try:
        conn = db.get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM bids WHERE id = %s RETURNING id", (bid_id,))
        deleted = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        if deleted:
            return jsonify({"message": "Bid deleted successfully", "bid_id": bid_id}), 200
        else:
            return jsonify({"error": "Bid not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/register", methods=["POST"])
def register():
    try:
        print("FORM DATA:", request.form)
        print("FILES:", request.files)

        username = request.form.get("username")
        email = request.form.get("email")
        phone = request.form.get("phone")
        password = request.form.get("password")

        profile_picture = request.files.get("profile_picture")

        # Validation
        if not username:
            return jsonify({"error": "username is required"}), 400
        if not email:
            return jsonify({"error": "email is required"}), 400
        if not phone:
            return jsonify({"error": "phone is required"}), 400
        if not password:
            return jsonify({"error": "password is required"}), 400

        profile_image_url = None

        # Upload image if provided
        if profile_picture:
            profile_image_url = st.upload_image(profile_picture)

        created_at = datetime.utcnow()

        conn = db.get_db_connection()
        cur = conn.cursor()

        query = """
        INSERT INTO users (username, email, phone, profile_image_url, password, created_at)
        VALUES (%s,%s,%s,%s,%s,%s)
        RETURNING id
        """

        cur.execute(query, (
            username,
            email,
            phone,
            profile_image_url,
            password,
            created_at
        ))

        user_id = cur.fetchone()[0]

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            "message": "User registered successfully",
            "user_id": user_id,
            "username": username,
            "email": email,
            "phone": phone,
            "profile_picture_url": profile_image_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    try:
        username = request.json.get("username")
        password = request.json.get("password")

        conn = db.get_db_connection()
        cur = conn.cursor()

        query = """
        SELECT id, username, email, phone, profile_image_url, password
        FROM users
        WHERE username=%s
        """

        cur.execute(query, (username,))
        user = cur.fetchone()

        cur.close()
        conn.close()

        if not user:
            return jsonify({"error": "User not found"}), 404

        user_id, username, email, phone, profile_image_url, db_password = user

        if password == db_password:

            return jsonify({
                "message": "Login successful",
                "user_id": user_id,
                "user_name": username,
                "email": email,
                "phone_number": phone,
                "profile_picture_url": profile_image_url
            })

        else:
            return jsonify({"error": "Invalid password"}), 401

    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route("/update-profile-picture", methods=["POST"])
def update_profile_picture():
    try:
        user_id = request.form.get("user_id")
        profile_picture = request.files.get("profile_picture")

        if not profile_picture:
            return jsonify({"error": "No image provided"}), 400

        image_url = st.upload_image(profile_picture)

        conn = db.get_db_connection()
        cur = conn.cursor()

        query = """
        UPDATE users
        SET profile_image_url=%s
        WHERE id=%s
        """

        cur.execute(query, (image_url, user_id))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            "message": "Profile picture updated",
            "profile_picture_url": image_url
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/update-phone", methods=["POST"])
def update_phone():
    try:
        user_id = request.json.get("user_id")
        phone = request.json.get("phone")

        conn = db.get_db_connection()
        cur = conn.cursor()

        query = """
        UPDATE users
        SET phone=%s
        WHERE id=%s
        """

        cur.execute(query, (phone, user_id))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            "message": "Phone number updated",
            "phone_number": phone
        })

    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route("/update-email", methods=["POST"])
def update_email():
    try:
        user_id = request.json.get("user_id")
        email = request.json.get("email")

        conn = db.get_db_connection()
        cur = conn.cursor()

        query = """
        UPDATE users
        SET email=%s
        WHERE id=%s
        """

        cur.execute(query, (email, user_id))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            "message": "Email updated",
            "email": email
        })

    except Exception as e:
        return jsonify({"error": str(e)})
    
# -----------------------------------
# Run Server
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)